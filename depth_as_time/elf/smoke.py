"""Sanity check for the ELF wrappers: reproduce the released numbers.

1. run_blocks matches ELF.forward (denoise and decode mode).
2. ELF-B-PD 1-step (SC-CFG 2.5): README reports Gen. PPL 136.1 / entropy 5.26.
3. ELF-B teacher 32-step SDE (gamma 1.5, SC-CFG 3): README reports 24.1 / 5.15.
Entropy follows ELF's metrics (unigram entropy of the GPT-2 retokenized sample).
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


def retok_entropy(ppl, texts):
    enc = ppl.tok(texts, truncation=True, max_length=1024)['input_ids']
    out = []
    for ids in enc:
        _, c = np.unique(ids, return_counts=True)
        p = c / c.sum()
        out.append(float(-(p * np.log(p + 1e-10)).sum()))
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_student', type=int, default=1024)
    p.add_argument('--n_teacher', type=int, default=256)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/elf/smoke.json')
    a = p.parse_args()
    dev = 'cuda'
    tok = transformers.AutoTokenizer.from_pretrained(E.T5_TOK)
    stu = E.load_elf(E.STUDENT, dev)
    print('student params', sum(x.numel() for x in stu.parameters()) / 1e6, 'M')

    # 1. wrapper == reference forward
    g = torch.Generator(device=dev).manual_seed(0)
    z = E.noise(stu, 2, g)
    t0, sc = torch.zeros(2, device=dev), torch.full((2,), 2.5, device=dev)
    xin = torch.cat([z, torch.zeros_like(z)], -1)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        ref, _ = stu(xin, t0, self_cond_cfg_scale=sc)
    mine, _, hs = E.run_blocks(stu, xin, t0, sc)
    print('denoise max|diff|', (ref - mine).abs().max().item(), 'n_hidden', len(hs),
          'readout(h_K)==out', (E.readout_x(stu, hs[-1]) - mine).abs().max().item())
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, ref_l = stu(torch.cat([mine, torch.zeros_like(mine)], -1), torch.ones(2, device=dev),
                       self_cond_cfg_scale=sc, decoder_step_active=True)
    print('decode max|diff|', (ref_l - E.decode_latent(stu, mine, 2.5)).abs().max().item())

    ppl = GenPPL('openai-community/gpt2-large', dev)
    res = {}
    g = torch.Generator(device=dev).manual_seed(a.seed)
    texts = []
    for i in range(0, a.n_student, a.bs):
        z = E.noise(stu, min(a.bs, a.n_student - i), g)
        x, _ = E.one_step(stu, z, 2.5)
        texts += E.to_text(tok, E.decode_latent(stu, x, 2.5).argmax(-1))
    res['student_1step'] = {'gen_ppl': ppl(texts, max_length=1024), 'entropy': retok_entropy(ppl, texts),
                            'n': len(texts), 'samples': texts[:3]}
    print('student 1-step', {k: v for k, v in res['student_1step'].items() if k != 'samples'})
    print(texts[0][:600])
    del stu

    tea = E.load_elf(E.TEACHER, dev)
    gcpu = torch.Generator().manual_seed(a.seed)
    texts = []
    for i in range(0, a.n_teacher, a.bs):
        z = E.noise(tea, min(a.bs, a.n_teacher - i), g)
        steps = E.logit_normal_steps(32, tea.cfg, gcpu)
        zf, _ = E.sample_multistep(tea, z, steps, sc_scale=3.0, gamma=1.5, generator=g)
        texts += E.to_text(tok, E.decode_latent(tea, zf, 3.0).argmax(-1))
    res['teacher_32sde'] = {'gen_ppl': ppl(texts, max_length=1024), 'entropy': retok_entropy(ppl, texts),
                            'n': len(texts), 'samples': texts[:3]}
    print('teacher 32-step SDE', {k: v for k, v in res['teacher_32sde'].items() if k != 'samples'})
    print(texts[0][:600])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
