"""Shared helpers for the depth-as-time probes on ELF (Embedded Language Flows).

ELF (arXiv 2605.10938) runs flow matching in a frozen T5-small embedding space
(512-d, normalized by latent_std) and discretizes only at t=1. The same network
then runs in "decode" mode: mode tokens on, t=1, logits from a factored
unembedding. The one-step model is the released progressive-distillation student
(ELF-B-PD-owt-torch). One step from t=0 gives x_hat = net(z0, t=0), and z1 = x_hat
because v = (x_hat - z) / (1 - t). A second, decode-mode pass then maps x_hat to tokens.

This file mirrors ELF.forward (vendored in elf_src/, unmodified) but also
returns the hidden state after every block, so the probes can read out each
block through the model's own heads.
"""
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elf_src'))
from modules.model import ELF_models  # noqa: E402

ELF_DIR = '/ddn/scratch/nyu/awg4562/models/elf'
STUDENT = f'{ELF_DIR}/ELF-B-PD-owt-torch'
TEACHER = f'{ELF_DIR}/ELF-B-owt-torch'
T5_TOK = f'{ELF_DIR}/t5-small'


def load_elf(path, device='cuda', use_ema=True):
    """Build ELF from <path>/config.yml and load the checkpoint_* inside (EMA by default)."""
    cfg = yaml.safe_load(open(f'{path}/config.yml'))
    ckpt_file = sorted(f for f in os.listdir(path) if f.startswith('checkpoint_'))[-1]
    ckpt = torch.load(f'{path}/{ckpt_file}', map_location='cpu', weights_only=False)
    sd = ckpt['ema_params1'] if use_ema and ckpt.get('ema_params1') else ckpt['params']
    model = ELF_models[cfg['model']](
        text_encoder_dim=sd['final_layer.linear.weight'].shape[0],
        max_length=cfg['max_length'],
        bottleneck_dim=cfg['bottleneck_dim'],
        num_time_tokens=cfg['num_time_tokens'],
        num_self_cond_cfg_tokens=cfg['num_self_cond_cfg_tokens'],
        num_model_mode_tokens=cfg['num_model_mode_tokens'],
        vocab_size=sd['unembed_kernel'].shape[1],
    )
    model.load_state_dict(sd, strict=True)
    model.requires_grad_(False)
    model.cfg = cfg
    return model.to(device).eval()


def _prefix(model, x, t, sc_scale, decode):
    """Input projection + prefix tokens, as in ELF.forward. Returns (x_with_prefix, n_prefix)."""
    B = x.shape[0]
    with torch.amp.autocast('cuda', enabled=False):
        if x.shape[-1] == 2 * model.text_encoder_dim:
            x = model.self_cond_proj(x.float())
        x = model.text_proj(x.float())
        ctx = model.build_context(t, sc_scale)
    mode = model.mode_tokens.expand(B, -1, -1) * float(decode)
    x = torch.cat([mode, x], dim=1)
    pre = torch.cat(ctx, dim=1)
    return torch.cat([pre, x], dim=1), pre.shape[1] + model.num_model_mode_tokens


@torch.no_grad()
def run_blocks(model, x_in, t, sc_scale, decode=False, return_hidden=True, bf16=True):
    """ELF forward that also returns hidden states.

    x_in: (B, L, 2C) self-cond input [z, x_prev]. t, sc_scale: (B,).
    Returns (out, logits, hiddens): out = final_layer(h_K) (x-pred), logits only
    when decode=True, hiddens = [h_0 (input proj), h_1, ..., h_K] over the main
    tokens (B, L, D) in float32.
    """
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        x, off = _prefix(model, x_in, t, sc_scale, decode)
        hs = [x[:, off:].float()] if return_hidden else None
        for blk in model.blocks:
            x = blk(x, rope_fn=model.feat_rope, attention_mask=None, deterministic=True)
            if return_hidden:
                hs.append(x[:, off:].float())
        h = x[:, off:]
        with torch.amp.autocast('cuda', enabled=False):
            out = model.final_layer(h.float())
            logits = decode_head(model, h) if decode else None
    return out, logits, hs


def decode_head(model, h):
    """Factored unembedding of ELF.forward (decode mode)."""
    h = h.float()
    hid = torch.nn.functional.gelu(h @ model.proj_kernel + model.proj_bias, approximate='tanh')
    return hid @ model.unembed_kernel + model.unembed_bias


def readout_x(model, h):
    """Block readout in latent space: the model's own x-pred head on an intermediate hidden."""
    with torch.amp.autocast('cuda', enabled=False):
        return model.final_layer(h.float())


@torch.no_grad()
def decode_latent(model, x, sc_scale, bf16=True):
    """Map a clean latent to token logits with the decode-mode pass (t=1, no self-cond)."""
    B = x.shape[0]
    t = torch.ones(B, device=x.device)
    sc = torch.full((B,), float(sc_scale), device=x.device)
    _, logits, _ = run_blocks(model, torch.cat([x, torch.zeros_like(x)], -1), t, sc,
                              decode=True, return_hidden=False, bf16=bf16)
    return logits


def noise(model, n, generator=None, device='cuda'):
    c = model.cfg
    return torch.randn((n, c['max_length'], model.text_encoder_dim), generator=generator,
                       device=device) * c['denoiser_noise_scale']


@torch.no_grad()
def one_step(model, z, sc_scale=2.5, return_hidden=False):
    """Released ELF+PD 1-step sampler: one ODE step from t=0 (x_prev = 0)."""
    B = z.shape[0]
    t = torch.zeros(B, device=z.device)
    sc = torch.full((B,), float(sc_scale), device=z.device)
    x_hat, _, hs = run_blocks(model, torch.cat([z, torch.zeros_like(z)], -1), t, sc,
                              return_hidden=return_hidden)
    return x_hat, hs


@torch.no_grad()
def sample_multistep(model, z, t_steps, sc_scale, gamma=0.0, generator=None, record=False):
    """ELF sampler (ODE if gamma=0, else the hybrid SDE of sampling_utils._sde_step).

    Mirrors generation_utils._generate_samples_single_batch: self-cond x_prev is
    carried between steps and the last step is always ODE. With record=True it also
    returns the list of (t, z_t, x_pred) visited by the sampler.
    """
    eps_t = model.cfg['t_eps']
    B = z.shape[0]
    sc = torch.full((B,), float(sc_scale), device=z.device)
    x_prev = torch.zeros_like(z)
    traj = []
    n = len(t_steps)
    for i in range(n - 1):
        t, t_next = float(t_steps[i]), float(t_steps[i + 1])
        if gamma > 0 and i < n - 2:
            alpha = max(0.0, min(1.0, 1.0 - gamma * (t_next - t)))
            eps = torch.randn(z.shape, generator=generator, device=z.device) * model.cfg['denoiser_noise_scale']
            z = alpha * z + (1 - alpha) * eps
            t = alpha * t
        tb = torch.full((B,), t, device=z.device)
        x_hat, _, _ = run_blocks(model, torch.cat([z, x_prev], -1), tb, sc, return_hidden=False)
        if record:
            traj.append((t, z.clone(), x_hat.clone()))
        v = (x_hat - z) / max(1.0 - t, eps_t)
        z = z + (t_next - t) * v
        x_prev = x_hat
    return z, traj


def logit_normal_steps(n, cfg, generator=None):
    """get_sampling_steps(..., 'logit_normal'): sorted sigmoid(N(P_mean, P_std)) with 0/1 ends."""
    s = torch.sigmoid(torch.randn(n - 1, generator=generator) * cfg['denoiser_p_std'] + cfg['denoiser_p_mean'])
    return torch.cat([torch.zeros(1), torch.sort(s).values, torch.ones(1)]).tolist()


def mask_after_eos(ids, eos_id, pad_id):
    keep = (ids == eos_id).cumsum(1) == 0
    return torch.where(keep, ids, torch.full_like(ids, pad_id))


def to_text(tok, ids):
    ids = mask_after_eos(ids, tok.eos_token_id, tok.pad_token_id)
    return [tok.decode(r.tolist(), skip_special_tokens=True) for r in ids.cpu()]


# ---- PD-faithful reference (plan v2, stage A) --------------------------------------------

def pd_grid(cfg, n_steps=64, generator=None, quantile=False):
    """Sub-step times of distill_train_step for (t_s, t_t) = (0, 1): 0, sorted
    sigmoid(N(P_mean, P_std)) interior points, 1. quantile=True gives the deterministic
    quantile grid of the same distribution (midpoint quantiles)."""
    m, s = cfg['denoiser_p_mean'], cfg['denoiser_p_std']
    if quantile:
        q = (torch.arange(1, n_steps, dtype=torch.float64) - 0.5) / (n_steps - 1)
        u = torch.sigmoid(m + s * torch.special.ndtri(q))
    else:
        u = torch.sort(torch.sigmoid(torch.randn(n_steps - 1, generator=generator) * s + m)).values
    return torch.cat([torch.zeros(1), u.float(), torch.ones(1)]).tolist()


@torch.no_grad()
def pd_ruler(model, z0, times, w, x_prev0=None, record=True):
    """distill_utils.teacher_pd_target as a trajectory: Euler along v = (x_hat - z)/max(1 - t, t_eps),
    self-cond x_prev0 (zeros by default = use_sc 0, the 1-step inference case) on the first sub-step,
    then chained. Returns (z_1, list of (t_k, z_k, x_hat_k)) where x_hat_k = net(z_k, t_k)."""
    eps_t = model.cfg['t_eps']
    B = z0.shape[0]
    sc = torch.full((B,), float(w), device=z0.device)
    z = z0
    x_prev = torch.zeros_like(z0) if x_prev0 is None else x_prev0
    traj = []
    for k in range(len(times) - 1):
        t, t_next = float(times[k]), float(times[k + 1])
        x_hat, _, _ = run_blocks(model, torch.cat([z, x_prev], -1), torch.full((B,), t, device=z.device),
                                 sc, return_hidden=False)
        x_hat = x_hat.float()
        if record:
            traj.append((t, z, x_hat))
        z = z + (t_next - t) * (x_hat - z) / max(1.0 - t, eps_t)
        x_prev = x_hat
    return z, traj


def viewer_scale(t):
    """Map z_t = t x + (1-t) 2 eps onto the decoder's training inputs lambda x + (1-lambda) 5 eps
    (same per-dim SNR): lambda/(1-lambda) = 2.5 t/(1-t), input = (lambda/t) z_t = 2.5/(1 + 1.5 t) z_t.
    Valid while lambda >~ 0.3 (logit(lambda) ~ N(0.8, 0.8) in training), i.e. t >~ 0.15."""
    return 2.5 / (1.0 + 1.5 * t)


def viewer_lambda(t):
    return 2.5 * t / (1.0 + 1.5 * t)
