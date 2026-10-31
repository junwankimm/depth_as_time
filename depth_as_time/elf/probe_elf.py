"""Depth-as-time probes on the ELF-B-PD one-step student (OWT).

For each block i of the one-step query (t=0), with h_i the hidden after block i:
  latent readout  x_i = final_layer(h_i)                (model's own x-pred head)
  token readout   tok_i = argmax decode(x_i)            (model's own decode pass)
  direct lens     argmax decode_head(h_i)               (decoder unembedding on the
                                                          denoise-mode hidden; off-distribution)
Metrics per block:
  - latent: relative error to the final x_K, cosine to x_K, distance to the input noise z0
  - tokens: agreement with final tokens, Gen. PPL and entropy of the decoded text
  - state matching: tau_i = argmin_t ||x_i - z_t|| along an ODE path z_t from the
    same z0 (self ruler = the student run with many steps; teacher ruler = ELF-B).
    Also the argmin against the ruler's clean estimate x_hat_t.
  - per-token commit block: first block from which tok_i stays equal to the final token
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--sc', type=float, default=2.5)
    p.add_argument('--ruler_steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/elf/probe.json')
    return p.parse_args()


def commit_block(tok_blocks):
    """tok_blocks: (K+1, B, L). First block index from which the token equals the final one."""
    same = tok_blocks == tok_blocks[-1:]
    # suffix-and over blocks: stays equal from block k on
    stay = torch.flip(torch.cumprod(torch.flip(same, [0]).int(), 0), [0]).bool()
    return stay.int().argmax(0)  # first True


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    dev = 'cuda'
    tok = transformers.AutoTokenizer.from_pretrained(E.T5_TOK)
    stu = E.load_elf(E.STUDENT, dev)
    tea = E.load_elf(E.TEACHER, dev)
    K = len(stu.blocks)
    ts = np.linspace(0, 1, a.ruler_steps + 1)
    g = torch.Generator(device=dev).manual_seed(a.seed)

    acc = {k: [] for k in ['rel_err', 'cos', 'dist_z0', 'cos_z0', 'agree', 'agree_lens', 'agree_chance', 'commit']}
    match = {r: {'tau_z': [], 'tau_x': [], 'curve_z': [], 'tau_zcos': [], 'tau_xcos': []} for r in ['self', 'teacher']}
    texts = [[] for _ in range(K + 1)]
    for b0 in range(0, a.n, a.bs):
        B = min(a.bs, a.n - b0)
        z0 = E.noise(stu, B, g)
        xK, hs = E.one_step(stu, z0, a.sc, return_hidden=True)
        xs = [E.readout_x(stu, h) for h in hs]
        toks = torch.stack([E.decode_latent(stu, x, a.sc).argmax(-1) for x in xs])
        lens = torch.stack([E.decode_head(stu, h).argmax(-1) for h in hs])
        nK = xK.flatten(1).norm(dim=1)
        acc['rel_err'].append(torch.stack([(x - xK).flatten(1).norm(dim=1) / nK for x in xs]).cpu())
        acc['cos'].append(torch.stack([torch.nn.functional.cosine_similarity(x.flatten(1), xK.flatten(1))
                                       for x in xs]).cpu())
        acc['dist_z0'].append(torch.stack([(x - z0).flatten(1).norm(dim=1) / z0.flatten(1).norm(dim=1)
                                           for x in xs]).cpu())
        acc['cos_z0'].append(torch.stack([torch.nn.functional.cosine_similarity(x.flatten(1), z0.flatten(1))
                                          for x in xs]).cpu())
        acc['agree'].append((toks == toks[-1:]).float().mean(-1).cpu())
        # chance level: each block's tokens vs the final tokens of a different noise
        acc['agree_chance'].append((toks == toks[-1:].roll(1, 1)).float().mean(-1).cpu())
        acc['agree_lens'].append((lens == toks[-1:]).float().mean(-1).cpu())
        acc['commit'].append(commit_block(toks).flatten().cpu())
        for i in range(K + 1):
            texts[i] += E.to_text(tok, toks[i])

        for name, ruler in [('self', stu), ('teacher', tea)]:
            z1, traj = E.sample_multistep(ruler, z0, ts, sc_scale=a.sc, record=True)
            zt = torch.stack([z for _, z, _ in traj] + [z1])               # z_t at every ts
            xt = torch.stack([x for _, _, x in traj] + [z1])
            dz = torch.stack([(zt - x[None]).flatten(2).norm(dim=2) for x in xs])  # (K+1, T, B)
            dx = torch.stack([(xt - x[None]).flatten(2).norm(dim=2) for x in xs])
            match[name]['tau_z'].append(torch.tensor(ts)[dz.argmin(1).cpu()])     # (K+1, B)
            match[name]['tau_x'].append(torch.tensor(ts)[dx.argmin(1).cpu()])
            match[name]['curve_z'].append((dz / dz[:, :1]).cpu())
            # scale-free matching: argmax_t cosine (the block readouts are off-scale, see rel_err)
            cz = torch.stack([torch.nn.functional.cosine_similarity(zt.flatten(2), x.flatten(1)[None], dim=2)
                              for x in xs])
            cx = torch.stack([torch.nn.functional.cosine_similarity(xt.flatten(2), x.flatten(1)[None], dim=2)
                              for x in xs])
            match[name]['tau_zcos'].append(torch.tensor(ts)[cz.argmax(1).cpu()])
            match[name]['tau_xcos'].append(torch.tensor(ts)[cx.argmax(1).cpu()])
        print(f'batch {b0 // a.bs} done', flush=True)

    ppl = GenPPL('openai-community/gpt2-large', dev)
    res = {'K': K, 'n': a.n, 'sc': a.sc, 'ruler_ts': ts.tolist()}
    for k in ['rel_err', 'cos', 'dist_z0', 'cos_z0', 'agree', 'agree_lens', 'agree_chance']:
        v = torch.cat(acc[k], 1)
        res[k] = v.mean(1).tolist()
    cb = torch.cat(acc['commit'])
    res['commit_hist'] = torch.bincount(cb, minlength=K + 1).tolist()
    for name in match:
        tz = torch.cat(match[name]['tau_z'], 1).float()
        tx = torch.cat(match[name]['tau_x'], 1).float()
        res[f'tau_z_{name}'] = tz.mean(1).tolist()
        res[f'tau_z_{name}_std'] = tz.std(1).tolist()
        res[f'tau_x_{name}'] = tx.mean(1).tolist()
        for k in ['tau_zcos', 'tau_xcos']:
            res[f'{k}_{name}'] = torch.cat(match[name][k], 1).float().mean(1).tolist()
        res[f'curve_z_{name}'] = torch.cat(match[name]['curve_z'], 2).mean(2).tolist()
    res['gen_ppl'] = [ppl(t, max_length=1024) for t in texts]
    res['entropy'] = [retok_entropy(ppl, t) for t in texts]
    res['samples'] = {i: texts[i][0][:400] for i in range(K + 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('| block | rel_err | cos | agree | lens agree | tau_z self | tau_z teacher | tau_x teacher | PPL | ent |')
    for i in range(K + 1):
        print(f"| {i} | {res['rel_err'][i]:.3f} | {res['cos'][i]:.3f} | {res['agree'][i]:.3f} | "
              f"{res['agree_lens'][i]:.3f} | {res['tau_z_self'][i]:.3f} | {res['tau_z_teacher'][i]:.3f} | "
              f"{res['tau_x_teacher'][i]:.3f} | {res['gen_ppl'][i]:.1f} | {res['entropy'][i]:.3f} |")
    print('commit hist', res['commit_hist'])
    print('| block | cos z0 | chance agree | tau_zcos self | tau_zcos teacher | tau_xcos self | tau_xcos teacher |')
    for i in range(K + 1):
        print(f"| {i} | {res['cos_z0'][i]:.3f} | {res['agree_chance'][i]:.3f} | {res['tau_zcos_self'][i]:.3f} | "
              f"{res['tau_zcos_teacher'][i]:.3f} | {res['tau_xcos_self'][i]:.3f} | {res['tau_xcos_teacher'][i]:.3f} |")


if __name__ == '__main__':
    main()
