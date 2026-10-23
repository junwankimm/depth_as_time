"""Toy sweep: when does the one-step depth schedule jump instead of sweeping gradually?

Hypothesis: language posteriors are mixtures over discrete sentences, so a one-step map must
at some depth pick a mode jointly across positions (symmetry breaking). The schedule should then
become more jump-like as the dependency between tokens grows.

Datasets with an exact posterior: sentences of length L over a vocabulary V, built from L/c
independent chunks; each chunk takes one of m random c-token strings. c = 1 gives independent
tokens, c = L a fully joint mixture of M = m sentences; in general M = m^(L/c).

For each config: train FLM, distill an FMLM from the exact denoiser (toy.py recipe), then
measure on 2048 noise samples:
  * progress across depth (agreement with own output, valid fraction) and its jumpiness:
    J = largest one-block gain / total gain; d50 = first block with half of the gain;
  * per-token commit depth and its spread within a sample (a global commit has small spread);
  * state-matching time tau*(i) against the exact flow (delta-space L2), and its largest jump.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
import toy  # noqa: E402

CONFIGS = {  # name: (L, V, c, m)
    'dep_c1': (8, 16, 1, 4), 'dep_c2': (8, 16, 2, 4), 'dep_c4': (8, 16, 4, 4), 'dep_c8': (8, 16, 8, 4),
    'modes_m2': (8, 16, 8, 2), 'modes_m16': (8, 16, 8, 16), 'modes_m64': (8, 16, 8, 64),
    'modes_m256': (8, 16, 8, 256),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, choices=list(CONFIGS))
    p.add_argument('--n_blocks', type=int, default=8)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--flm_steps', type=int, default=6000)
    p.add_argument('--fmlm_steps', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--n_probe', type=int, default=2048)
    p.add_argument('--ode_steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/toy_sweep')
    return p.parse_args()


def make_sequences(L, V, c, m, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(L // c):
        opts = set()
        while len(opts) < m:
            opts.add(tuple(rng.integers(0, V, c).tolist()))
        chunks.append(sorted(opts))
    return [sum(combo, ()) for combo in itertools.product(*chunks)]


def lock_index(tok, final):
    """tok (n, B, L), final (B, L) -> first index from which tok == final to the end."""
    same = tok == final.unsqueeze(0)
    suffix = torch.flip(torch.cumprod(torch.flip(same, [0]).int(), 0), [0]).bool()
    n = tok.shape[0]
    return torch.where(suffix, torch.arange(n, device=tok.device).view(-1, 1, 1), n).min(0).values


def jumpiness(curve):
    inc = np.diff(np.asarray(curve))
    tot = inc.sum()
    if tot <= 1e-9:
        return 0.0, 0
    J = float(inc.max() / tot)
    cum = np.cumsum(inc) / tot
    return J, int(np.argmax(cum >= 0.5) + 1)


@torch.no_grad()
def measure(args, exact, lut, model, V):
    torch.manual_seed(args.seed + 1)
    N = args.n_probe
    z = torch.randn(N, exact.L, V, device='cuda')
    P = toy.layer_probs(model, z, 0.0, 1.0)  # (N, nL, L, V)
    nL = P.shape[1]
    tok = P.argmax(-1).transpose(0, 1)  # (nL, N, L)
    y = tok[-1]
    valid = {tuple(s) for s in exact.X.argmax(-1).tolist()}
    is_valid = lambda t: np.mean([tuple(r) in valid for r in t.tolist()])  # noqa: E731
    agree = [(tok[i] == y).float().mean().item() for i in range(nL)]
    valid_by = [float(is_valid(tok[i])) for i in range(nL)]
    tau_grid = torch.linspace(0, 1, args.ode_steps + 1, device='cuda')
    t_grid = lut.tau_to_t(tau_grid)
    xs, Ds = exact.flow(z, t_grid)
    y_exact = xs[-1].argmax(-1)
    commit = lock_index(tok, y)
    # state matching against the exact flow: delta*_{0,t_k} = z + (x_k - z) / t_k, k >= 1
    ks = list(range(1, len(xs) - 1))
    dist = torch.zeros(nL, len(ks), device='cuda')
    for j, k in enumerate(ks):
        d = z + (xs[k] - z) / t_grid[k]
        dist[:, j] = (P - d.unsqueeze(1)).pow(2).sum((-1, -2)).sqrt().mean(0)
    best = dist.argmin(1)
    tau_star = tau_grid[torch.tensor(ks, device='cuda')[best]].cpu().numpy()
    J, d50 = jumpiness(agree)
    Jv, d50v = jumpiness(valid_by)
    return {
        'endpoint_agree_exact': (y == y_exact).all(-1).float().mean().item(),
        'final_valid': valid_by[-1], 'agree_by_layer': agree, 'valid_by_layer': valid_by,
        'ent_by_layer': [(-(P[:, i] * P[:, i].clamp_min(1e-12).log()).sum(-1)).mean().item() for i in range(nL)],
        'jump_agree': J, 'd50_agree': d50, 'jump_valid': Jv, 'd50_valid': d50v,
        'commit_hist': (torch.bincount(commit.flatten(), minlength=nL).float() / commit.numel()).cpu().numpy().tolist(),
        'commit_mean': commit.float().mean().item(), 'commit_std_all': commit.float().std().item(),
        'commit_std_within': commit.float().std(1).mean().item(),
        'tau_star': tau_star.tolist(), 'tau_star_max_jump': float(np.diff(tau_star).max()),
        'match_min_l2': dist.min(1).values.cpu().numpy().tolist(),
    }


def main():
    args = parse_args()
    L, V, c, m = CONFIGS[args.config]
    args.length = L
    torch.manual_seed(args.seed)
    seqs = make_sequences(L, V, c, m, args.seed)
    exact = toy.Exact(seqs, V, 'cuda')
    lut = toy.GPUTimeLUT(V, 'cuda')
    t0 = time.time()
    flm = toy.train_flm(args, exact, lut, V)
    fmlm = toy.train_fmlm(args, exact, lut, V, flm, teacher='exact')
    res = measure(args, exact, lut, fmlm, V)
    res.update({'config': args.config, 'L': L, 'V': V, 'c': c, 'm': m, 'M': len(seqs),
                'minutes': (time.time() - t0) / 60, 'args': vars(args)})
    out = os.path.join(HERE, '..', args.out)
    os.makedirs(out, exist_ok=True)
    # seed 0, 8 blocks, width 128 keep the original names
    tag = args.config + (f'_s{args.seed}' if args.seed else '') + \
        (f'_b{args.n_blocks}' if args.n_blocks != 8 else '') + (f'_h{args.hidden}' if args.hidden != 128 else '')
    res['tag'] = tag
    torch.save(fmlm.state_dict(), os.path.join(out, f'{tag}_fmlm.pt'))
    with open(os.path.join(out, f'{tag}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'args'}))


if __name__ == '__main__':
    main()
