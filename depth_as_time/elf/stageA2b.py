"""Plan v2, stage A2b: does the PD student track the grid-averaged teacher?

PD round 5 regresses (MSE) the student onto the teacher endpoint of a *random* logit-normal sub-step
grid, and stage A showed the teacher endpoint itself moves a lot with the grid (token agreement
~0.44 between grids). The MSE minimizer is then E_grid[z_1 | z0]. Compare the student output with
the mean over R random grids, against each single-grid endpoint vs the mean of the other R-1.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elf_common as E  # noqa: E402

cos = torch.nn.functional.cosine_similarity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--w', type=float, default=2.5)
    p.add_argument('--R', type=int, default=8)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/elf/v2/stageA2b.json')
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev = 'cuda'
    stu = E.load_elf(E.STUDENT, dev)
    tea = E.load_elf(E.TEACHER, dev)
    gcpu = torch.Generator().manual_seed(a.seed + 100)
    grids = [E.pd_grid(tea.cfg, 64, generator=gcpu) for _ in range(a.R)]
    g = torch.Generator(device=dev).manual_seed(a.seed)
    f = lambda x: x.flatten(1)  # noqa: E731
    dec = lambda m, x: E.decode_latent(m, x, a.w).argmax(-1)  # noqa: E731
    rows = {k: [] for k in ['cos_s_mean', 'cos_r_looMean', 'cos_s_single', 'cos_r_r', 'tok_s_mean', 'tok_r_looMean',
                            'norm_s', 'norm_mean', 'norm_single']}
    for b0 in range(0, a.n, a.bs):
        z0 = E.noise(stu, min(a.bs, a.n - b0), g)
        xs = E.one_step(stu, z0, a.w)[0].float()
        Z = torch.stack([E.pd_ruler(tea, z0, gr, a.w, record=False)[0] for gr in grids])   # (R, B, L, C)
        zm = Z.mean(0)
        rows['cos_s_mean'].append(cos(f(xs), f(zm)).mean().item())
        rows['cos_s_single'].append(np.mean([cos(f(xs), f(Z[r])).mean().item() for r in range(a.R)]))
        rows['cos_r_r'].append(np.mean([cos(f(Z[r]), f(Z[q])).mean().item() for r in range(a.R) for q in range(r)]))
        loo = [(Z.sum(0) - Z[r]) / (a.R - 1) for r in range(a.R)]
        rows['cos_r_looMean'].append(np.mean([cos(f(Z[r]), f(loo[r])).mean().item() for r in range(a.R)]))
        tm = dec(tea, zm)
        rows['tok_s_mean'].append((dec(stu, xs) == tm).float().mean().item())
        rows['tok_r_looMean'].append(np.mean([(dec(tea, Z[r]) == dec(tea, loo[r])).float().mean().item()
                                              for r in range(a.R)]))
        rows['norm_s'].append(f(xs).norm(dim=1).mean().item())
        rows['norm_mean'].append(f(zm).norm(dim=1).mean().item())
        rows['norm_single'].append(f(Z[0]).norm(dim=1).mean().item())
        print('batch', flush=True)
    res = {k: float(np.mean(v)) for k, v in rows.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
