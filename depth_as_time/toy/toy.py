"""Toy FLM / FMLM with an exact posterior, for a depth-as-time Fig. 1 analogue.

Datasets (finite support, uniform weights):
  cities: L = 2, V = 4 = {new, san, york, diego}; data {new york, san diego}
          (the two-mode example of FLM Fig. 3).
  ring8:  L = 2, V = 8; data {(k, k)}: 8 modes, the language counterpart of
          the 8-Gaussians toy of depth-as-time Fig. 1.

With finite support {X_j} and x_t = (1 - t) z + t X, the denoiser is exact:
    D*_t(x) = sum_j softmax_j( t <x, X_j> / (1 - t)^2 ) X_j
(||X_j||^2 = L for every j), so the exact flow, flow map and posterior at
every noise level are available as ground truth.

Pipeline (mirrors algo.FLM / algo.FMLM with algo/fmlm.yaml defaults):
  1. train FLM: tokenwise softmax denoiser, cross-entropy, tau ~ U[0, 1].
  2. distill FMLM (PSD / semigroup KL, diagonal fraction 0.5, boundary pair
     (0, 1) with prob 1/32 of the off-diagonal part, uniform_diff sampling,
     midpoint u), initialized from the FLM. Teacher: the trained FLM
     ("flm") or the exact denoiser ("exact").
  3. probe: own-head readout of every block against the exact flow.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # depth_as_time/
import lens  # noqa: E402
import models.dit  # noqa: E402

DATASETS = {
    'cities': dict(V=4, tokens=['new', 'san', 'york', 'diego'], seqs=[[0, 2], [1, 3]]),
    'ring8': dict(V=8, tokens=[str(k) for k in range(8)], seqs=[[k, k] for k in range(8)]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cities', choices=list(DATASETS))
    p.add_argument('--n_blocks', type=int, default=8)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--flm_steps', type=int, default=20000)
    p.add_argument('--fmlm_steps', type=int, default=30000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--n_probe', type=int, default=2048)
    p.add_argument('--ode_steps', type=int, default=1000)
    p.add_argument('--teachers', default='exact,flm',
                   help='which FMLM variants to distill (teacher = exact denoiser or trained FLM)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/toy')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Exact quantities
# ----------------------------------------------------------------------------

class Exact:
    def __init__(self, seqs, V, device):
        self.X = F.one_hot(torch.tensor(seqs, device=device), V).float()  # (M, L, V)
        self.L = self.X.shape[1]

    def posterior(self, x, t):
        """Mode posterior w_j(x, t), shape (B, M). t: (B,) in [0, 1)."""
        t = t.view(-1, 1)
        score = torch.einsum('blv,mlv->bm', x, self.X)
        return torch.softmax(t * score / (1 - t).clamp_min(1e-6) ** 2, -1)

    def denoiser(self, x, t):
        return torch.einsum('bm,mlv->blv', self.posterior(x, t), self.X)

    def flow(self, z, t_grid):
        """Euler ODE dx/dt = (D*_t(x) - x)/(1 - t) on t_grid (ending at 1).
        Returns states x_{t_k} and posteriors D*_{t_k}(x_{t_k}) for every k."""
        xs, Ds = [], []
        x = z
        for k in range(len(t_grid) - 1):
            t = t_grid[k].expand(z.shape[0])
            D = self.denoiser(x, t)
            xs.append(x)
            Ds.append(D)
            if k == len(t_grid) - 2:
                x = D
                break
            x = x + (t_grid[k + 1] - t_grid[k]) * (D - x) / (1 - t_grid[k])
        xs.append(x)
        return xs, Ds

    def flow_map_delta(self, z, t_end, n_steps):
        """Exact two-time denoiser delta*_{0,t}(z) = z + (X_{0,t}(z) - z) / t."""
        if t_end >= 1.0:
            return self.flow(z, torch.linspace(0, 1, n_steps + 1, device=z.device))[1][-1]
        grid = torch.linspace(0, t_end, n_steps + 1, device=z.device)
        x = z
        for k in range(n_steps):
            t = grid[k].expand(z.shape[0])
            x = x + (grid[k + 1] - grid[k]) * (self.denoiser(x, t) - x) / (1 - grid[k])
        return z + (x - z) / t_end


class GPUTimeLUT:
    """tau <-> t maps on the GPU, tabulated from utils.build_luts."""

    def __init__(self, V, device, n=20001):
        lut = lens.TimeLUT(V)
        self.tau_grid = torch.linspace(0, 1, n, device=device)
        self.t_of_tau = torch.from_numpy(
            np.clip(lut.lut_a2g(self.tau_grid.cpu().numpy()), 0, 1)).float().to(device)
        self.t_of_tau[0], self.t_of_tau[-1] = 0.0, 1.0
        self.t_of_tau = torch.cummax(self.t_of_tau, 0).values

    def tau_to_t(self, tau):
        idx = tau.clamp(0, 1) * (len(self.tau_grid) - 1)
        lo = idx.floor().long().clamp(max=len(self.tau_grid) - 2)
        w = idx - lo
        return self.t_of_tau[lo] * (1 - w) + self.t_of_tau[lo + 1] * w

    def t_to_tau(self, t):
        i = torch.searchsorted(self.t_of_tau, t.clamp(0, 1).contiguous()).clamp(1, len(self.tau_grid) - 1)
        t0, t1 = self.t_of_tau[i - 1], self.t_of_tau[i]
        w = ((t - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0, 1)
        return self.tau_grid[i - 1] * (1 - w) + self.tau_grid[i] * w


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def make_dit(args, V, name):
    cfg = omegaconf.OmegaConf.create({
        'algo': {'name': name, 'causal_attention': False, 'double_temb': name == 'fmlm',
                 'learnable_loss_weighting': False},
        'model': {'hidden_size': args.hidden, 'cond_dim': 128, 'n_heads': 4,
                  'n_blocks': args.n_blocks, 'dropout': 0.0, 'scale_by_sigma': True,
                  'length': getattr(args, 'length', 2)},
    })
    return models.dit.DIT(cfg, vocab_size=V).cuda()


def logprobs(model, x, tau, tau_prime=None):
    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        out = model(x, tau, tau_prime)
    return lens._logprobs(out)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def sample_x1(exact, B):
    idx = torch.randint(0, exact.X.shape[0], (B,), device='cuda')
    return exact.X[idx]


def train_flm(args, exact, lut, V):
    model = make_dit(args, V, 'flm')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    ema = EMA(model, args.ema)
    t0 = time.time()
    for step in range(args.flm_steps):
        x1 = sample_x1(exact, args.batch_size)
        tau = torch.rand(args.batch_size, device='cuda')
        t = lut.tau_to_t(tau).view(-1, 1, 1)
        x_t = (1 - t) * torch.randn_like(x1) + t * x1
        loss = -(x1 * logprobs(model, x_t, tau)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        if step % 2000 == 0 or step == args.flm_steps - 1:
            print(f'[flm] step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)', flush=True)
    ema.copy_to(model)
    return model.eval()


def train_fmlm(args, exact, lut, V, flm, teacher):
    """PSD distillation as in algo.FMLM.loss (fmlm.yaml defaults)."""
    model = make_dit(args, V, 'fmlm')
    sd = model.state_dict()
    for k, v in flm.state_dict().items():
        if k in sd:
            sd[k].copy_(v)
    for n, p in model.sigma_map_prime.named_parameters():
        if 'mlp.2' in n:
            p.data.zero_()
    model.load_state_dict(sd)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    ema = EMA(model, args.ema)
    B = args.batch_size
    n_diag = B // 2
    n_bnd = max(1, int(B * 0.5 / 32))
    t0 = time.time()
    for step in range(args.fmlm_steps):
        x1 = sample_x1(exact, B)
        tau_diag = torch.rand(B, device='cuda')
        tau_d = torch.rand(B, device='cuda')
        tau_s_od = torch.rand(B, device='cuda') * (1 - tau_d)
        tau_t_od = tau_s_od + tau_d
        perm = torch.randperm(B, device='cuda')
        i_diag, i_bnd, i_od = perm[:n_diag], perm[n_diag:n_diag + n_bnd], perm[n_diag + n_bnd:]
        tau_s, tau_t = tau_s_od.clone(), tau_t_od.clone()
        tau_s[i_diag], tau_t[i_diag] = tau_diag[i_diag], tau_diag[i_diag]
        tau_s[i_bnd], tau_t[i_bnd] = 0.0, 1.0
        tau_u = 0.5 * (tau_s + tau_t)
        s, u, t = lut.tau_to_t(tau_s), lut.tau_to_t(tau_u), lut.tau_to_t(tau_t)
        x_s = (1 - s.view(-1, 1, 1)) * torch.randn_like(x1) + s.view(-1, 1, 1) * x1

        log_D_st = logprobs(model, x_s, tau_s, tau_t)
        with torch.no_grad():
            xd = x_s[i_diag]
            if teacher == 'exact':
                diag_target = exact.denoiser(xd, s[i_diag])
            else:
                diag_target = logprobs(flm, xd, tau_s[i_diag]).exp()
            i_off = torch.cat([i_od, i_bnd])
            xo = x_s[i_off]
            so, uo, to = (v[i_off].view(-1, 1, 1) for v in (s, u, t))
            D_su = logprobs(model, xo, tau_s[i_off], tau_u[i_off]).exp()
            X_su = (1 - uo) / (1 - so + 1e-8) * xo + (uo - so) / (1 - so + 1e-8) * D_su
            D_ut = logprobs(model, X_su, tau_u[i_off], tau_t[i_off]).exp()
            lam = (1 - to) * (uo - so) / ((1 - uo) * (to - so) + 1e-8)
            off_target = lam * D_su + (1 - lam) * D_ut
        loss = torch.cat([-(diag_target * log_D_st[i_diag]).sum(-1),
                          -(off_target * log_D_st[i_off]).sum(-1)]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        if step % 2000 == 0 or step == args.fmlm_steps - 1:
            print(f'[fmlm/{teacher}] step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)',
                  flush=True)
    ema.copy_to(model)
    return model.eval()


# ----------------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------------

@torch.no_grad()
def layer_probs(model, x, tau, tau_prime):
    logps, _, _ = lens.layer_logprobs(model, x, torch.as_tensor(tau, device='cuda'),
                                      None if tau_prime is None else torch.as_tensor(tau_prime, device='cuda'))
    return torch.stack([lp.exp() for lp in logps], 1)  # (B, nL, L, V)


def kl_tok(p, q):
    """Mean over positions of KL(p || q); p, q (..., L, V)."""
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).mean(-1)


@torch.no_grad()
def probe(args, exact, lut, V, nets):
    torch.manual_seed(args.seed + 1)
    N = args.n_probe
    z = torch.randn(N, exact.L, V, device='cuda')
    tau_grid = torch.linspace(0, 1, args.ode_steps + 1, device='cuda')
    t_grid = lut.tau_to_t(tau_grid)
    xs, Ds = exact.flow(z, t_grid)
    D_stack = torch.stack(Ds, 1)  # (N, K, L, V) exact posterior along the exact flow
    x_stack = torch.stack(xs[:-1], 1)
    y_exact = xs[-1].argmax(-1)  # exact flow endpoint
    out = {'tau_grid': tau_grid[:-1].cpu().numpy(), 't_grid': t_grid[:-1].cpu().numpy(),
           'z': z.cpu().numpy(), 'exact_D': D_stack[:, ::10].cpu().numpy(),
           'exact_x': x_stack[:, ::10].cpu().numpy(), 'exact_y': y_exact.cpu().numpy(),
           'exact_tau_sub': tau_grid[:-1][::10].cpu().numpy()}
    valid = {tuple(s) for s in exact.X.argmax(-1).tolist()}

    def is_valid(tok):
        return torch.tensor([tuple(r) in valid for r in tok.tolist()], device='cuda')

    sub = D_stack[:, ::5]
    for name, (model, kind) in nets.items():
        res = {}
        if kind == 'fmlm':
            P = layer_probs(model, z, 0.0, 1.0)
        else:  # FLM: one NFE at tau = 0 is the marginal; probe its first Euler step instead
            P = layer_probs(model, z, 0.0, None)
        res['probs'] = P.cpu().numpy()
        nL = P.shape[1]
        y_final = P[:, -1].argmax(-1)
        res['agree_exact_endpoint'] = (y_final == y_exact).all(-1).float().mean().item()
        res['valid_by_layer'] = [is_valid(P[:, i].argmax(-1)).float().mean().item() for i in range(nL)]
        res['agree_final_by_layer'] = [(P[:, i].argmax(-1) == y_final).float().mean().item() for i in range(nL)]
        res['ent_by_layer'] = [(-(P[:, i] * P[:, i].clamp_min(1e-12).log()).sum(-1)).mean().item()
                               for i in range(nL)]
        # effective exact time of each layer: argmin_k KL(D*_{t_k} || delta^i)
        cross = torch.stack([kl_tok(sub, P[:, i:i + 1]).mean(0) for i in range(nL)])  # (nL, K/5)
        res['cross_kl'] = cross.cpu().numpy()
        res['eff_tau'] = tau_grid[::5][:cross.shape[1]][cross.argmin(1)].cpu().numpy()
        res['l2_to_final'] = [(P[:, i] - P[:, -1]).pow(2).sum((-1, -2)).sqrt().cpu().numpy()
                              for i in range(nL)]
        if kind == 'fmlm':
            # noisy endpoints: delta^i_{0,t} vs the exact delta*_{0,t}
            sweep = {}
            for tau_t in [0.25, 0.5, 0.75, 1.0]:
                t_end = lut.tau_to_t(torch.tensor([tau_t], device='cuda')).item()
                Pt = layer_probs(model, z, 0.0, tau_t)
                target = exact.flow_map_delta(z, t_end, 2000)
                clean = xs[-1]
                sweep[f'{tau_t:.2f}'] = {
                    'probs': Pt.cpu().numpy(), 'exact_delta': target.cpu().numpy(),
                    'kl_to_exact_delta': [kl_tok(target, Pt[:, i]).mean().item() for i in range(nL)],
                    'kl_clean': [kl_tok(clean, Pt[:, i]).mean().item() for i in range(nL)],
                    'ent': [(-(Pt[:, i] * Pt[:, i].clamp_min(1e-12).log()).sum(-1)).mean().item()
                            for i in range(nL)],
                    'exact_ent': (-(target * target.clamp_min(1e-12).log()).sum(-1)).mean().item(),
                }
            res['sweep'] = sweep
        out[name] = res
    return out


def main():
    args = parse_args()
    out_dir = os.path.join(HERE, '..', args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    ds = DATASETS[args.dataset]
    V = ds['V']
    exact = Exact(ds['seqs'], V, 'cuda')
    lut = GPUTimeLUT(V, 'cuda')

    flm_path = os.path.join(out_dir, 'flm.pt')
    if os.path.exists(flm_path):  # shared across the teacher variants
        flm = make_dit(args, V, 'flm')
        flm.load_state_dict(torch.load(flm_path))
        flm.eval()
        print('loaded', flm_path)
    else:
        flm = train_flm(args, exact, lut, V)
        torch.save(flm.state_dict(), flm_path)
    nets = {}
    for teacher in args.teachers.split(','):
        m = train_fmlm(args, exact, lut, V, flm, teacher=teacher)
        torch.save(m.state_dict(), os.path.join(out_dir, f'fmlm_{teacher}.pt'))
        nets[f'fmlm_{teacher}'] = (m, 'fmlm')

    res = probe(args, exact, lut, V, nets)
    # pack: arrays to npz, scalars/lists to json
    arrays, summary = {}, {'args': vars(args), 'dataset': ds}
    for k, v in res.items():
        if isinstance(v, dict):
            summary[k] = {}
            for kk, vv in v.items():
                if kk == 'sweep':
                    summary[k]['sweep'] = {}
                    for tt, d in vv.items():
                        arrays[f'{k}/sweep/{tt}/probs'] = d.pop('probs')
                        arrays[f'{k}/sweep/{tt}/exact_delta'] = d.pop('exact_delta')
                        summary[k]['sweep'][tt] = d
                elif isinstance(vv, np.ndarray) or (isinstance(vv, list) and vv and isinstance(vv[0], np.ndarray)):
                    arrays[f'{k}/{kk}'] = np.asarray(vv)
                else:
                    summary[k][kk] = vv
        else:
            arrays[k] = v
    tag = args.teachers.replace(',', '_')
    np.savez_compressed(os.path.join(out_dir, f'probe_{tag}.npz'), **arrays)
    with open(os.path.join(out_dir, f'summary_{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=lambda o: np.asarray(o).tolist())
    for k in nets:
        s = summary[k]
        print(k, 'endpoint agreement with exact flow', round(s['agree_exact_endpoint'], 3))
        print('  valid by layer', np.round(s['valid_by_layer'], 3).tolist())
        print('  agree final   ', np.round(s['agree_final_by_layer'], 3).tolist())
        print('  entropy       ', np.round(s['ent_by_layer'], 3).tolist())
        for tt, d in s['sweep'].items():
            print(f'  sweep tau_t={tt} ent', np.round(d['ent'], 3).tolist(),
                  'exact', round(d['exact_ent'], 3))
            print('              kl_to_exact_delta', np.round(d['kl_to_exact_delta'], 3).tolist())
    print('done ->', out_dir)


if __name__ == '__main__':
    main()
