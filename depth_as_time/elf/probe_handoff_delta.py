"""Handoff test for the ELF one-step student, with the same construction as FMLM.

The flow map from z0 is X_{0,t}(z0) = (1 - t) z0 + t delta_{0,t}(z0), so the teacher ODE
path z_t defines delta_{0,t} = (z_t - (1 - t) z0) / t. Block i's latent readout x_i plays the
role of delta^i:
  tau_i  = argmin_t mean ||x_i - delta_{0,t}||   (teacher ruler, mean over samples, as in
           probe_statematch.py T_l2_tau_star)
  state  = (1 - tau_i) z0 + tau_i x_i, handed to the teacher ODE at tau_i.
Starting states (same as probe_handoff_fmlm.py):
  own     teacher z_{tau_i};  layer  (1 - tau_i) z0 + tau_i x_i
  answer  (1 - tau_i) z0 + tau_i x_K;  noise  z0
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

VARIANTS = ['own', 'layer', 'answer', 'noise']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--sc', type=float, default=2.5)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/elf/handoff_delta.json')
    return p.parse_args()


def batches(a, stu, tea, ts, dev):
    """Replayable stream of (z0, block readouts, teacher path z_t, student / teacher tokens)."""
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for b0 in range(0, a.n, a.bs):
        z0 = E.noise(stu, min(a.bs, a.n - b0), g)
        xK, hs = E.one_step(stu, z0, a.sc, return_hidden=True)
        xs = [E.readout_x(stu, h) for h in hs]
        z1, traj = E.sample_multistep(tea, z0, ts, sc_scale=a.sc, record=True)
        zt = torch.stack([z for _, z, _ in traj] + [z1])
        del traj
        yield z0, xs, zt, E.decode_latent(stu, xK, a.sc).argmax(-1), E.decode_latent(tea, z1, a.sc).argmax(-1)


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    dev = 'cuda'
    tok = transformers.AutoTokenizer.from_pretrained(E.T5_TOK)
    stu = E.load_elf(E.STUDENT, dev)
    tea = E.load_elf(E.TEACHER, dev)
    K = len(stu.blocks)
    ts = np.linspace(0, 1, a.steps + 1)
    tt = torch.tensor(ts[1:], device=dev, dtype=torch.float32).view(-1, 1, 1, 1)

    # pass 1: delta-space distance curves (K+1, T-1), summed over samples
    curve = torch.zeros(K + 1, len(ts) - 1, device=dev)
    for z0, xs, zt, _, _ in batches(a, stu, tea, ts, dev):
        delta = (zt[1:] - (1 - tt) * z0[None]) / tt                       # (T-1, B, L, C)
        for i, x in enumerate(xs):
            curve[i] += (delta - x[None]).flatten(2).norm(dim=2).sum(1)
        print('pass 1 batch', flush=True)
    curve /= a.n
    ki = (curve.argmin(1) + 1).tolist()                                    # grid index of tau_i
    tau = [float(ts[k]) for k in ki]

    toks = {v: [[] for _ in range(K + 1)] for v in VARIANTS}
    agree = {v: {w: [[] for _ in range(K + 1)] for w in ['student', 'teacher']} for v in VARIANTS}
    for z0, xs, zt, tok_s, tok_t in batches(a, stu, tea, ts, dev):
        for i in range(K + 1):
            k = min(ki[i], len(ts) - 2)
            t = float(ts[k])
            starts = {'own': zt[k], 'layer': (1 - t) * z0 + t * xs[i],
                      'answer': (1 - t) * z0 + t * xs[-1], 'noise': z0}
            for v, s in starts.items():
                zf, _ = E.sample_multistep(tea, s, ts[k:], sc_scale=a.sc)
                tk = E.decode_latent(tea, zf, a.sc).argmax(-1)
                toks[v][i] += E.to_text(tok, tk)
                agree[v]['student'][i].append((tk == tok_s).float().mean().item())
                agree[v]['teacher'][i].append((tk == tok_t).float().mean().item())
        print('pass 2 batch', flush=True)

    ppl = GenPPL('openai-community/gpt2-large', dev)
    res = {'tau': tau, 'curve': curve.tolist(), 'n': a.n, 'handoff': {}}
    for v in VARIANTS:
        res['handoff'][v] = {
            'gen_ppl': [ppl(toks[v][i], max_length=1024) for i in range(K + 1)],
            'entropy': [retok_entropy(ppl, toks[v][i]) for i in range(K + 1)],
            'agree_student': [float(np.mean(agree[v]['student'][i])) for i in range(K + 1)],
            'agree_teacher': [float(np.mean(agree[v]['teacher'][i])) for i in range(K + 1)],
            'sample': toks[v][6][0][:300],
        }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('tau*', [round(t, 3) for t in tau])
    for v in VARIANTS:
        h = res['handoff'][v]
        print(f'\n[{v}] | block | tau | PPL | ent | agree student | agree teacher |')
        for i in range(K + 1):
            print(f"| {i} | {tau[i]:.3f} | {h['gen_ppl'][i]:.1f} | {h['entropy'][i]:.3f} | "
                  f"{h['agree_student'][i]:.3f} | {h['agree_teacher'][i]:.3f} |")


if __name__ == '__main__':
    main()
