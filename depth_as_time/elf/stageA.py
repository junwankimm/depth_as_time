"""Plan v2, stage A: get the reference right for the ELF-B-PD one-step student.

A1  PD-faithful ruler: teacher Euler on the PD logit-normal sub-step grid (64 sub-steps, deterministic
    quantile grid; plus random grids), SC-CFG w, x_prev = 0 on the first sub-step then chained.
A2  Does the student track it? Student output vs ruler endpoint (latent cosine / rel. error, tokens),
    against the teacher's own grid-to-grid spread and the v1 uniform-grid ruler. Also at w = 1, 4.
A3  Progress clock along the ruler: viewer tokens of z_t (calibrated, elf_common.viewer_scale) and of
    x_hat_t vs the final tokens; signal / noise coefficients of z_t.
A4  Self-cond sensitivity: restart the teacher from its own z_t with x_prev = chained (sanity, = 1),
    zero, or a fresh uncond seed (PD's seeding rule), and measure final-token agreement.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elf_common as E  # noqa: E402
from probe_onestep import GenPPL  # noqa: E402
from smoke import retok_entropy  # noqa: E402

cos = torch.nn.functional.cosine_similarity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--w', type=float, default=2.5)
    p.add_argument('--w_extra', default='1.0,4.0')
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--n_rand', type=int, default=3)
    p.add_argument('--restarts', default='4,8,16,24,32,40,48,56')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/elf/v2/stageA.json')
    return p.parse_args()


def agree(a, b):
    return (a == b).float().mean().item()


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    dev = 'cuda'
    tok = transformers.AutoTokenizer.from_pretrained(E.T5_TOK)
    stu = E.load_elf(E.STUDENT, dev)
    tea = E.load_elf(E.TEACHER, dev)
    cfg = tea.cfg
    grid_q = E.pd_grid(cfg, a.steps, quantile=True)
    gcpu = torch.Generator().manual_seed(a.seed)
    grids_r = [E.pd_grid(cfg, a.steps, generator=gcpu) for _ in range(a.n_rand)]
    grid_u = np.linspace(0, 1, a.steps + 1).tolist()
    restarts = [int(k) for k in a.restarts.split(',')]
    T = len(grid_q)
    dec = lambda m, x, w=a.w: E.decode_latent(m, x, w).argmax(-1)  # noqa: E731

    A2 = {k: [] for k in ['cos', 'rel_err', 'tok_s_q', 'tok_sT_q', 'tok_s_r', 'tok_q_r', 'tok_r_r',
                          'tok_s_u', 'tok_q_u', 'cos_q_r']}
    A2w = {w: {'cos': [], 'tok_s_q': []} for w in map(float, a.w_extra.split(','))}
    A3 = {k: np.zeros(T) for k in ['view_z', 'raw_z', 'dec_x', 'coef_x1', 'coef_z0', 'cos_x_x1']}
    A4 = {c: np.zeros(len(restarts)) for c in ['chained', 'zero', 'seed']}
    texts = {'student': [], 'ruler_q': [], 'ruler_u': []}
    nb = 0
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for b0 in range(0, a.n, a.bs):
        B = min(a.bs, a.n - b0)
        z0 = E.noise(stu, B, g)
        xs, _ = E.one_step(stu, z0, a.w)
        xs = xs.float()
        tok_s, tok_sT = dec(stu, xs), dec(tea, xs)
        z1q, traj = E.pd_ruler(tea, z0, grid_q, a.w)
        tok_q = dec(tea, z1q)
        f = lambda x: x.flatten(1)  # noqa: E731
        A2['cos'].append(cos(f(xs), f(z1q)).mean().item())
        A2['rel_err'].append(((f(xs) - f(z1q)).norm(dim=1) / f(z1q).norm(dim=1)).mean().item())
        A2['tok_s_q'].append(agree(tok_s, tok_q))
        A2['tok_sT_q'].append(agree(tok_sT, tok_q))
        tok_r = []
        for gr in grids_r:
            z1r, _ = E.pd_ruler(tea, z0, gr, a.w, record=False)
            tok_r.append(dec(tea, z1r))
            A2['cos_q_r'].append(cos(f(z1r), f(z1q)).mean().item())
        A2['tok_s_r'].append(np.mean([agree(tok_s, t) for t in tok_r]))
        A2['tok_q_r'].append(np.mean([agree(tok_q, t) for t in tok_r]))
        A2['tok_r_r'].append(np.mean([agree(tok_r[i], tok_r[j]) for i in range(len(tok_r))
                                      for j in range(i + 1, len(tok_r))]))
        z1u, _ = E.sample_multistep(tea, z0, grid_u, sc_scale=a.w)
        tok_u = dec(tea, z1u)
        A2['tok_s_u'].append(agree(tok_s, tok_u))
        A2['tok_q_u'].append(agree(tok_q, tok_u))
        texts['student'] += E.to_text(tok, tok_s)
        texts['ruler_q'] += E.to_text(tok, tok_q)
        texts['ruler_u'] += E.to_text(tok, tok_u)

        # A3: progress clock along the quantile-grid ruler (states z_k at t_k, k = 0..T-1; z_T = z1q)
        states = [(t, z, x) for t, z, x in traj] + [(1.0, z1q, z1q)]
        for k, (t, z, x) in enumerate(states):
            A3['view_z'][k] += agree(dec(tea, z * E.viewer_scale(t)), tok_q)
            A3['raw_z'][k] += agree(dec(tea, z), tok_q)
            A3['dec_x'][k] += agree(dec(tea, x), tok_q)
            A3['coef_x1'][k] += ((f(z) * f(z1q)).sum(1) / f(z1q).pow(2).sum(1)).mean().item()
            A3['coef_z0'][k] += ((f(z) * f(z0)).sum(1) / f(z0).pow(2).sum(1)).mean().item()
            A3['cos_x_x1'][k] += cos(f(x), f(z1q)).mean().item()

        # A4: restart the teacher from its own z_k with three self-cond conventions
        for j, k in enumerate(restarts):
            t, z, _ = traj[k]
            sc = torch.full((B,), a.w, device=dev)
            seed, _, _ = E.run_blocks(tea, torch.cat([z, torch.zeros_like(z)], -1),
                                      torch.full((B,), t, device=dev), sc, return_hidden=False)
            for c, xp in [('chained', traj[k - 1][2]), ('zero', None), ('seed', seed.float())]:
                zf, _ = E.pd_ruler(tea, z, grid_q[k:], a.w, x_prev0=xp, record=False)
                A4[c][j] += agree(dec(tea, zf), tok_q)
        del traj, states

        for w in A2w:
            xw, _ = E.one_step(stu, z0, w)
            zw, _ = E.pd_ruler(tea, z0, grid_q, w, record=False)
            A2w[w]['cos'].append(cos(f(xw.float()), f(zw)).mean().item())
            A2w[w]['tok_s_q'].append(agree(dec(stu, xw.float(), w), dec(tea, zw, w)))
        nb += 1
        print(f'batch {nb} done', flush=True)

    ppl = GenPPL('openai-community/gpt2-large', dev)
    res = {'grid_q': grid_q, 'restarts': restarts, 'restart_t': [grid_q[k] for k in restarts], 'n': a.n,
           'A2': {k: float(np.mean(v)) for k, v in A2.items()},
           'A2_w': {str(w): {k: float(np.mean(v)) for k, v in d.items()} for w, d in A2w.items()},
           'A3': {k: (v / nb).tolist() for k, v in A3.items()},
           'A4': {k: (v / nb).tolist() for k, v in A4.items()},
           'quality': {k: {'gen_ppl': ppl(v, max_length=1024), 'entropy': retok_entropy(ppl, v)}
                       for k, v in texts.items()},
           'samples': {k: v[0][:300] for k, v in texts.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('A2', json.dumps(res['A2'], indent=1))
    print('A2_w', json.dumps(res['A2_w']))
    print('quality', json.dumps(res['quality']))
    print('| k | t | lambda | view z_t | raw z_t | dec x_hat_t | coef x1 | coef z0 | cos x_hat x1 |')
    for k in list(range(0, T, 4)) + [T - 1]:
        t = grid_q[k]
        print(f"| {k} | {t:.3f} | {E.viewer_lambda(t):.2f} | {res['A3']['view_z'][k]:.3f} | "
              f"{res['A3']['raw_z'][k]:.3f} | {res['A3']['dec_x'][k]:.3f} | {res['A3']['coef_x1'][k]:.3f} | "
              f"{res['A3']['coef_z0'][k]:.3f} | {res['A3']['cos_x_x1'][k]:.3f} |")
    print('| restart k | t | chained | zero | seed |')
    for j, k in enumerate(restarts):
        print(f"| {k} | {grid_q[k]:.3f} | {res['A4']['chained'][j]:.3f} | {res['A4']['zero'][j]:.3f} | "
              f"{res['A4']['seed'][j]:.3f} |")


if __name__ == '__main__':
    main()
