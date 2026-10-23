"""Trajectory supervision on the toy (direction C): can a one-step FMLM be made to follow the
flow uniformly across depth, and what does it cost?

Fine-tune a trained toy FMLM (toy_sweep.py) on the one-step query (0, 1) with
    L = sum_{i=1}^{n} w_i KL(delta*_{0, tau_i}(z) || delta^i(z)),   tau_i = i / n,
where delta^i is the readout after block i (own head), delta*_{0,t} the exact two-time
denoiser (from the exact flow, delta* = z + (x_t - z) / t), w_n = 1 (the endpoint) and
w_i = lam for i < n. lam = 0 is the control: the same fine-tuning with the endpoint only.

Prediction from the toy sweep: with independent tokens (c = 1) the uniform schedule costs
little; with fully joint tokens (c = 8) the mid-depth stall is needed, so forcing it away
should hurt the endpoint.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
import lens  # noqa: E402
import toy  # noqa: E402
import toy_sweep  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, choices=list(toy_sweep.CONFIGS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_blocks', type=int, default=8)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--ode_steps', type=int, default=200)
    p.add_argument('--n_probe', type=int, default=2048)
    p.add_argument('--out', default='results/toy_trajsup')
    return p.parse_args()


def block_logprobs(model, z):
    """Log-readouts after blocks 1..n for the one-step query, list of (B, L, V)."""
    tau = torch.zeros(z.shape[0], device=z.device)
    logps, _, _ = (lens.layer_logprobs_grad if torch.is_grad_enabled() else lens.layer_logprobs)(model, z, tau, tau + 1)
    return logps[1:]


def targets(exact, lut, z, n, ode_steps):
    """Exact delta*_{0, tau_i}(z) for tau_i = i / n, i = 1..n (the last is the clean endpoint)."""
    tau_grid = torch.linspace(0, 1, ode_steps + 1, device=z.device)
    t_grid = lut.tau_to_t(tau_grid)
    xs, _ = exact.flow(z, t_grid)
    out = []
    for i in range(1, n + 1):
        k = round(i / n * ode_steps)
        if k >= ode_steps:
            out.append(xs[-1])
        else:
            out.append((z + (xs[k] - z) / t_grid[k]).clamp_min(0))
    out = [d / d.sum(-1, keepdim=True).clamp_min(1e-12) for d in out]
    return out


def main():
    args = parse_args()
    L, V, c, m = toy_sweep.CONFIGS[args.config]
    args.length = L
    torch.manual_seed(args.seed + 100)
    seqs = toy_sweep.make_sequences(L, V, c, m, args.seed)
    exact = toy.Exact(seqs, V, 'cuda')
    lut = toy.GPUTimeLUT(V, 'cuda')
    tag = args.config + (f'_s{args.seed}' if args.seed else '') + (f'_b{args.n_blocks}' if args.n_blocks != 8 else '')
    base = toy.make_dit(args, V, 'fmlm')
    base.load_state_dict(torch.load(os.path.join(HERE, '..', 'results', 'toy_sweep', f'{tag}_fmlm.pt')))
    base.eval()
    model = copy.deepcopy(base).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = toy.EMA(model, args.ema)
    n = args.n_blocks
    w = torch.tensor([args.lam] * (n - 1) + [1.0], device='cuda')
    t0 = time.time()
    for step in range(args.steps):
        z = torch.randn(args.batch_size, L, V, device='cuda')
        with torch.no_grad():
            tg = targets(exact, lut, z, n, args.ode_steps)
        lps = block_logprobs(model, z)
        kls = torch.stack([(d * (d.clamp_min(1e-12).log() - lp)).sum(-1).mean() for d, lp in zip(tg, lps)])
        loss = (w * kls).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if step % 1000 == 0 or step == args.steps - 1:
            print(f'[{tag} lam={args.lam}] step {step} loss {loss.item():.4f} end-KL {kls[-1].item():.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    ema.copy_to(model)
    model.eval()

    # evaluation on fresh noise: before (base) and after fine-tuning
    torch.manual_seed(args.seed + 1)
    z = torch.randn(args.n_probe, L, V, device='cuda')
    with torch.no_grad():
        tg = targets(exact, lut, z, n, args.ode_steps)
        y_exact = tg[-1].argmax(-1)
        valid = {tuple(s) for s in exact.X.argmax(-1).tolist()}
        res = {'config': args.config, 'tag': tag, 'lam': args.lam, 'args': vars(args)}
        for name, net in [('before', base), ('after', model)]:
            lps = block_logprobs(net, z)
            toks = [lp.argmax(-1) for lp in lps]
            res[name] = {
                'endpoint_agree_exact': (toks[-1] == y_exact).all(-1).float().mean().item(),
                'endpoint_token_agree': (toks[-1] == y_exact).float().mean().item(),
                'endpoint_kl': (tg[-1] * (tg[-1].clamp_min(1e-12).log() - lps[-1])).sum(-1).mean().item(),
                'kl_to_schedule': [(d * (d.clamp_min(1e-12).log() - lp)).sum(-1).mean().item() for d, lp in zip(tg, lps)],
                # early exit: argmax of block i as the sample
                'valid_by_block': [float(np.mean([tuple(r) in valid for r in t.tolist()])) for t in toks],
                'agree_exact_by_block': [(t == y_exact).float().mean().item() for t in toks],
                'ent_by_block': [(-(lp.exp() * lp).sum(-1)).mean().item() for lp in lps],
            }
    out = os.path.join(HERE, '..', args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{tag}_lam{args.lam:g}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    for name in ['before', 'after']:
        r = res[name]
        print(f'{name:6s} endpoint {r["endpoint_agree_exact"]:.3f} tokens {r["endpoint_token_agree"]:.3f} '
              f'KL {r["endpoint_kl"]:.3f} valid_by_block {np.round(r["valid_by_block"], 2).tolist()}')
        print('       kl_to_schedule', np.round(r['kl_to_schedule'], 3).tolist())


if __name__ == '__main__':
    main()
