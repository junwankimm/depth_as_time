"""Handoff test for a one-step FMLM: does layer i act as the flow state at its matched time?

Layer i's readout delta^i (W_out lens of the (0 -> 1) query) is placed at its matched time
t_i as x = (1 - t_i) z + t_i delta^i (FLM Eq. 19; t_i from the teacher-ruler state matching,
probe_statematch.py T_l2_tau_star), and the FLM teacher finishes the Euler trajectory from
there. Starting states compared at the same t_i:
  own     the teacher's own Euler state x_{t_i} from the same z (upper reference)
  layer   (1 - t_i) z + t_i delta^i
  answer  (1 - t_i) z + t_i delta^L (the one-step output placed at t_i; is layer i special?)
  noise   z (no information beyond the noise)
The same construction is used for ELF in elf/probe_handoff_delta.py.
"""
import argparse
import json
import os

import numpy as np
import torch
import transformers

import lens
from probe_onestep import GenPPL, unigram_entropy

VARIANTS = ['own', 'layer', 'answer', 'noise']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-OWT')
    p.add_argument('--teacher', default='FLM-B-OWT')
    p.add_argument('--tokenizer', default='openai-community/gpt2')
    p.add_argument('--statematch', default='results/owt/statematch/summary.json')
    p.add_argument('--n_samples', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--teacher_steps', type=int, default=128)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/owt/handoff.json')
    return p.parse_args()


@torch.no_grad()
def flm_continue(backbone, lut, x, k0, num_steps):
    """lens.flm_trajectory started at grid step k0 (tau = k0 / num_steps) from state x."""
    B = x.shape[0]
    tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=x.device)
    for i in range(k0, num_steps):
        tau_in = tau_vals[i].expand(B)
        t_in = lut.tau_to_t(tau_in)
        probs = lens.model_logprobs(backbone, x, tau_in).exp()
        if i == num_steps - 1:
            return probs.argmax(-1)
        dt = lut.tau_to_t(tau_vals[i + 1].expand(B)) - t_in
        x = x + dt.view(-1, 1, 1) * (probs - x) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    student = lens.load_backbone(a.student, dev)
    teacher = lens.load_backbone(a.teacher, dev)
    tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)
    V, L = student.vocab_size, student.config.model.length
    lut = lens.TimeLUT(V)
    K = a.teacher_steps
    tau = json.load(open(a.statematch))['T_l2_tau_star']
    nL = len(tau)
    ks = [min(max(int(round(t * K)), 1), K - 1) for t in tau]       # grid step of each layer's time
    ts = [lut.tau_to_t(torch.tensor([k / K], device=dev)).item() for k in ks]

    toks = {v: [[] for _ in range(nL)] for v in VARIANTS}
    agree = {v: {w: [[] for _ in range(nL)] for w in ['student', 'teacher']} for v in VARIANTS}
    for b in range(a.n_samples // a.batch_size):
        z = torch.randn((a.batch_size, L, V), device=dev)
        lps, _, _ = lens.layer_logprobs(student, z, 0.0, 1.0)
        tok_s = lps[-1].argmax(-1)
        saved = {}

        def keep(k, tau_k, t_k, x_k, logp_k):
            if k in ks:
                saved[k] = x_k.clone()
        tok_t = lens.flm_trajectory(teacher, lut, z, K, keep)
        for i in range(nL):
            t = ts[i]
            starts = {'own': saved[ks[i]], 'layer': (1 - t) * z + t * lps[i].exp(),
                      'answer': (1 - t) * z + t * lps[-1].exp(), 'noise': z}
            for v, x in starts.items():
                out = flm_continue(teacher, lut, x, ks[i], K)
                toks[v][i].append(out.cpu())
                agree[v]['student'][i].append((out == tok_s).float().mean().item())
                agree[v]['teacher'][i].append((out == tok_t).float().mean().item())
        del lps, saved
        torch.cuda.empty_cache()
        print(f'batch {b} done', flush=True)

    del student, teacher
    gen = GenPPL('openai-community/gpt2-large', dev)
    res = {'tau': tau, 't': ts, 'k': ks, 'n': a.n_samples, 'handoff': {}}
    for v in VARIANTS:
        rows = {'gen_ppl': [], 'entropy': [], 'agree_student': [], 'agree_teacher': []}
        for i in range(nL):
            tk = torch.cat(toks[v][i])
            rows['gen_ppl'].append(gen(tok.batch_decode(tk), L))
            rows['entropy'].append(unigram_entropy(tk))
            rows['agree_student'].append(float(np.mean(agree[v]['student'][i])))
            rows['agree_teacher'].append(float(np.mean(agree[v]['teacher'][i])))
        rows['sample'] = tok.decode(toks[v][6][0][0][:80])
        res['handoff'][v] = rows
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    for v in VARIANTS:
        h = res['handoff'][v]
        print(f'\n[{v}] | layer | tau | PPL | ent | agree student | agree teacher |')
        for i in range(nL):
            print(f"| {i} | {tau[i]:.3f} | {h['gen_ppl'][i]:.1f} | {h['entropy'][i]:.3f} | "
                  f"{h['agree_student'][i]:.3f} | {h['agree_teacher'][i]:.3f} |")


if __name__ == '__main__':
    main()
