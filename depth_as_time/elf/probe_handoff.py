"""Is a block readout of the ELF one-step student an intermediate *state* of the teacher flow?

(a) Interior-peak test. Over the teacher ODE path z_t from the same z0, compare the
    best cosine cos(x_i, z_t) with what the two endpoints alone explain:
      path  = max_t cos(x_i, z_t)
      line  = max_a cos(x_i, a*z0 + (1-a)*z1)       (straight line between the endpoints)
      span  = ||proj_{span(z0, z1)} x_i|| / ||x_i||  (best point in the endpoint plane, >= line)
    If path <= span, "matching a time" is just mixing noise and answer.
(b) Handoff test. Hand the block state to the teacher at tau_i (cosine match, block mean)
    and finish the teacher ODE from there. Compare with the teacher continued from its own
    z_{tau_i}, and with z0 handed at tau_i (no information beyond the noise).
    Block states are handed raw and norm-matched to ||z_{tau_i}|| (readouts are off-scale).
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
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--sc', type=float, default=2.5)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/elf/handoff.json')
    return p.parse_args()


def span_cos(x, a, b):
    """||proj of x onto span(a, b)|| / ||x||, per sample (all flattened to (B, D))."""
    A = torch.stack([a, b], -1)                                   # (B, D, 2)
    G = A.transpose(1, 2) @ A                                     # (B, 2, 2)
    c = torch.linalg.solve(G, (A.transpose(1, 2) @ x[..., None]))  # (B, 2, 1)
    p = (A @ c).squeeze(-1)
    return p.norm(dim=1) / x.norm(dim=1)


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    dev = 'cuda'
    tok = transformers.AutoTokenizer.from_pretrained(E.T5_TOK)
    stu = E.load_elf(E.STUDENT, dev)
    tea = E.load_elf(E.TEACHER, dev)
    K = len(stu.blocks)
    ts = np.linspace(0, 1, a.steps + 1)
    alphas = torch.linspace(0, 1, 201, device=dev)
    g = torch.Generator(device=dev).manual_seed(a.seed)

    # pass 1: interior-peak test, block-mean tau, and the states needed for the handoff
    peak = {k: [] for k in ['path', 'line', 'span', 'end0', 'end1', 'tau']}
    for b0 in range(0, a.n, a.bs):
        B = min(a.bs, a.n - b0)
        z0 = E.noise(stu, B, g)
        xK, hs = E.one_step(stu, z0, a.sc, return_hidden=True)
        xs = [E.readout_x(stu, h) for h in hs]
        z1, traj = E.sample_multistep(tea, z0, ts, sc_scale=a.sc, record=True)
        zt = torch.stack([z for _, z, _ in traj] + [z1]).flatten(2)     # (T, B, D)
        f0, f1 = z0.flatten(1), z1.flatten(1)
        line = torch.stack([al * f0 + (1 - al) * f1 for al in alphas])  # (A, B, D)
        rows = {k: [] for k in peak}
        for x in xs:
            f = x.flatten(1)
            c = cos(zt, f[None], dim=2)                                  # (T, B)
            rows['path'].append(c.max(0).values)
            rows['tau'].append(torch.tensor(ts, device=dev)[c.argmax(0)])
            rows['line'].append(cos(line, f[None], dim=2).max(0).values)
            rows['span'].append(span_cos(f, f0, f1))
            rows['end0'].append(cos(f, f0))
            rows['end1'].append(cos(f, f1))
        for k in peak:
            peak[k].append(torch.stack(rows[k]).cpu())
        print(f'pass 1 batch {b0 // a.bs}', flush=True)
    res = {'K': K, 'n': a.n}
    for k in peak:
        res[k] = torch.cat(peak[k], 1).mean(1).tolist()
    tau = [float(ts[np.abs(ts - t).argmin()]) for t in res['tau']]  # block-mean tau snapped to grid
    res['tau_grid'] = tau

    # pass 2: handoff
    variants = ['own_z', 'block_raw', 'block_norm', 'noise_z0']
    texts = {v: [[] for _ in range(K + 1)] for v in variants}
    agree = {v: {w: [[] for _ in range(K + 1)] for w in ['student', 'teacher']} for v in variants}
    g = torch.Generator(device=dev).manual_seed(a.seed)   # replay the same noises
    for b0 in range(0, a.n, a.bs):
        z0 = E.noise(stu, min(a.bs, a.n - b0), g)
        xK, hs = E.one_step(stu, z0, a.sc, return_hidden=True)
        xs = [E.readout_x(stu, h) for h in hs]
        z1, traj = E.sample_multistep(tea, z0, ts, sc_scale=a.sc, record=True)
        zt = torch.stack([z for _, z, _ in traj] + [z1])
        tok_s = E.decode_latent(stu, xK, a.sc).argmax(-1).cpu()
        tok_t = E.decode_latent(tea, z1, a.sc).argmax(-1).cpu()
        del traj
        for i in range(K + 1):
            ti = int(np.abs(ts - tau[i]).argmin())
            grid = ts[ti:]
            if len(grid) < 2:
                grid = np.array([ts[-2], 1.0]); ti = len(ts) - 2
            ref = zt[ti]
            x = xs[i]
            scale = ref.flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1)
            n0 = ref.flatten(1).norm(dim=1) / z0.flatten(1).norm(dim=1)
            starts = {'own_z': ref, 'block_raw': x, 'block_norm': x * scale[:, None, None],
                      'noise_z0': z0 * n0[:, None, None]}
            for v, s in starts.items():
                zf, _ = E.sample_multistep(tea, s, grid, sc_scale=a.sc)
                tk = E.decode_latent(tea, zf, a.sc).argmax(-1).cpu()
                texts[v][i] += E.to_text(tok, tk)
                agree[v]['student'][i].append((tk == tok_s).float().mean().item())
                agree[v]['teacher'][i].append((tk == tok_t).float().mean().item())
        print('pass 2 batch done', flush=True)

    ppl = GenPPL('openai-community/gpt2-large', dev)
    res['handoff'] = {}
    for v in variants:
        res['handoff'][v] = {
            'gen_ppl': [ppl(texts[v][i], max_length=1024) for i in range(K + 1)],
            'entropy': [retok_entropy(ppl, texts[v][i]) for i in range(K + 1)],
            'agree_student': [float(np.mean(agree[v]['student'][i])) for i in range(K + 1)],
            'agree_teacher': [float(np.mean(agree[v]['teacher'][i])) for i in range(K + 1)],
            'sample': texts[v][6][0][:300],
        }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)

    print('| block | tau | cos z0 | cos z1 | line max | span | path max |')
    for i in range(K + 1):
        print(f"| {i} | {tau[i]:.3f} | {res['end0'][i]:.3f} | {res['end1'][i]:.3f} | {res['line'][i]:.3f} | "
              f"{res['span'][i]:.3f} | {res['path'][i]:.3f} |")
    for v in variants:
        h = res['handoff'][v]
        print(f'\n[{v}] | block | PPL | ent | agree student | agree teacher |')
        for i in range(K + 1):
            print(f"| {i} | {h['gen_ppl'][i]:.1f} | {h['entropy'][i]:.3f} | {h['agree_student'][i]:.3f} | "
                  f"{h['agree_teacher'][i]:.3f} |")


if __name__ == '__main__':
    main()
