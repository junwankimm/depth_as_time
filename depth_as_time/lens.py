"""Layer-wise readout ("logit lens") utilities for FLM / FMLM backbones.

Mirrors `models.dit.DIT.forward` but exposes every residual-stream state
h_0..h_L and decodes each one with the model's own output head, following
Eq. (2) of "Depth as Time in One-Step Generative Models" (arXiv:2610.03626):

    v_i := W_out(h_i; tau_task)

Here W_out is `DDiTFinalLayer` (LayerNorm + AdaLN modulation + linear),
followed by the same tanh soft-cap + log-softmax that FLMBase applies, so
the readout at i = L is exactly the model output.

Time convention (FLM paper, arXiv:2602.16813): t = 0 is noise, t = 1 is data,
x_t = (1 - t) z + t onehot(y). The networks are conditioned on the
reparameterized time tau(t) (decoding-error-rate schedule, Eq. 25).
"""

import os
import sys

import torch
import torch.nn.functional as F

FLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FLM_ROOT not in sys.path:
    sys.path.insert(0, FLM_ROOT)

import models.dit  # noqa: E402
import utils  # noqa: E402

CKPT_DIR = os.environ.get('FLM_CKPT_DIR', '/ddn/scratch/nyu/awg4562/models/flm')
CAP_VALUE = 30.0  # FLMBase._process_model_output soft-cap


def load_backbone(name, device='cuda'):
    """Load a DIT backbone from a HF `PyTorchModelHubMixin` snapshot dir."""
    path = name if os.path.isdir(name) else os.path.join(CKPT_DIR, name)
    model = models.dit.DIT.from_pretrained(path)
    return model.to(device).eval()


class TimeLUT:
    """tau <-> t maps (FLMBase._tau_to_t / _t_to_tau)."""

    def __init__(self, vocab_size):
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=vocab_size)

    def tau_to_t(self, tau):
        return utils.alpha_to_gamma(tau, self.lut_a2g)

    def t_to_tau(self, t):
        return utils.gamma_to_alpha(t, self.lut_g2a)


def _as_batch(tau, batch_size, device):
    if not torch.is_tensor(tau):
        tau = torch.tensor(float(tau), device=device)
    tau = tau.to(device=device, dtype=torch.float32)
    return tau.expand(batch_size) if tau.ndim == 0 else tau


def _cond(backbone, tau, tau_prime):
    t_emb = backbone.sigma_map(tau)
    if tau_prime is not None:
        if backbone.sigma_map_prime is not None:
            t_emb = t_emb + backbone.sigma_map_prime(tau_prime)
        else:
            t_emb = t_emb + backbone.sigma_map(tau_prime)
    return F.silu(t_emb)


def _logprobs(logits):
    logits = logits.float()
    logits = CAP_VALUE * torch.tanh(logits / CAP_VALUE)
    return logits.log_softmax(dim=-1)


@torch.no_grad()
def hidden_states(backbone, x, tau, tau_prime=None):
    """Return ([h_0, ..., h_L], c) for one network evaluation.

    h_0 is the input embedding E(x); h_i is the residual stream after block i.
    """
    B = x.shape[0]
    tau = _as_batch(tau, B, x.device)
    if tau_prime is not None:
        tau_prime = _as_batch(tau_prime, B, x.device)
    with torch.amp.autocast(device_type=x.device.type, dtype=torch.float32):
        h = backbone.vocab_embed(x)
        c = _cond(backbone, tau, tau_prime)
        rotary_cos_sin = backbone.rotary_emb(h)
        hs = [h]
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block in backbone.blocks:
                h = block(h, rotary_cos_sin, c=c, seqlens=None)
                hs.append(h)
    return hs, c


@torch.no_grad()
def readout_logits(backbone, h, c):
    """Raw W_out(h; c) logits (before soft-cap), computed as in DIT.forward."""
    with torch.amp.autocast(device_type=h.device.type, dtype=torch.bfloat16):
        return backbone.output_layer(h, c=c).float()


@torch.no_grad()
def readout(backbone, h, c):
    """Layer-wise log-probabilities log softmax(cap(W_out(h; c)))."""
    return _logprobs(readout_logits(backbone, h, c))


@torch.no_grad()
def layer_logprobs(backbone, x, tau, tau_prime=None):
    """Log-probability readouts for every depth i = 0..L (list of L+1)."""
    hs, c = hidden_states(backbone, x, tau, tau_prime)
    return [readout(backbone, h, c) for h in hs], hs, c


@torch.no_grad()
def model_logprobs(backbone, x, tau, tau_prime=None):
    """Ordinary model output (matches FLMBase.forward)."""
    B = x.shape[0]
    tau = _as_batch(tau, B, x.device)
    if tau_prime is not None:
        tau_prime = _as_batch(tau_prime, B, x.device)
    with torch.amp.autocast(device_type=x.device.type, dtype=torch.float32):
        out = backbone(x, tau, tau_prime)
    return _logprobs(out)


@torch.no_grad()
def flm_trajectory(backbone, lut, z, num_steps, callback=None):
    """Euler sampling of the FLM teacher (algo.FLM.generate_samples).

    callback(k, tau_k, t_k, x_k, logp_k) is called at every step with the
    current state x_{t_k} and the denoiser output log D_{t_k}(x_{t_k}).
    Returns the final argmax tokens.
    """
    B = z.shape[0]
    device = z.device
    tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    x = z
    for i in range(num_steps):
        tau_in = tau_vals[i].expand(B)
        t_in = lut.tau_to_t(tau_in)
        dt = lut.tau_to_t(tau_vals[i + 1].expand(B)) - t_in
        logp = model_logprobs(backbone, x, tau_in)
        if callback is not None:
            callback(i, tau_vals[i].item(), t_in[0].item(), x, logp)
        probs = logp.exp()
        if i == num_steps - 1:
            x = probs
            break
        v = (probs - x) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
        x = x + dt.view(-1, 1, 1) * v
    return x.argmax(dim=-1)


def fmlm_step_times(lut, num_steps, gamma=0.0, device='cuda'):
    """(tau_s, tau_target, t_s, t_target) per step, as in FMLM.generate_samples."""
    tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    out = []
    for i in range(num_steps):
        t_curr = lut.tau_to_t(tau_vals[i].view(1))
        t_next = lut.tau_to_t(tau_vals[i + 1].view(1))
        sigma_tilde = (1.0 - t_next) * (1.0 - gamma ** 2) ** 0.5
        t_tilde = 1.0 - sigma_tilde
        tau_tilde = lut.t_to_tau(t_tilde)
        out.append((tau_vals[i].item(), tau_tilde.item(),
                    t_curr.item(), t_tilde.item(), t_next.item()))
    return out


def fmlm_update(x, logp, t_s, t_tilde, t_next, gamma=0.0):
    """One FMLM flow-map step X_{s,t}(x) given log delta_{s,t}(x) (Eq. 19)."""
    D = logp.exp()
    z_tilde = ((1.0 - t_tilde) * x + (t_tilde - t_s) * D) / (1.0 - t_s)
    if gamma > 0:
        sigma_target = 1.0 - t_next
        sigma_tilde = 1.0 - t_tilde
        return (z_tilde + (sigma_tilde - sigma_target) * D
                + gamma * sigma_target * torch.randn_like(x))
    return z_tilde


def layer_logprobs_grad(backbone, x, tau, tau_prime=None):
    """Differentiable layer_logprobs, for training with losses on intermediate readouts."""
    hs, c = hidden_states.__wrapped__(backbone, x, tau, tau_prime)
    return [_logprobs(readout_logits.__wrapped__(backbone, h, c)) for h in hs], hs, c
