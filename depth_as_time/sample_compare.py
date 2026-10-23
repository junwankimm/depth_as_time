"""Decode one-step samples of the teacher and the compressed students from the same noise,
and quantify repetition: unigram entropy (paper metric), distinct-1/2, repeated-bigram rate,
share of the 10 most frequent tokens.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lens  # noqa: E402
from compress import load_teacher  # noqa: E402
from eval_compress import load_student  # noqa: E402
from probe_onestep import unigram_entropy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='FMLM-B-LM1B')
    p.add_argument('--runs', default='lm1b_tied_struct_K12,lm1b_untied_K6,lm1b_tied2_alt_K12,lm1b_tied1_K12,lm1b_tied1_noclock_K12')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--n_stats', type=int, default=1024)
    p.add_argument('--n_show', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--out', default='results/compress_eval/samples.json')
    return p.parse_args()


def rep_stats(toks):
    d1, d2, rep2 = [], [], []
    counts = collections.Counter()
    for s in toks.tolist():
        bi = list(zip(s, s[1:]))
        d1.append(len(set(s)) / len(s))
        d2.append(len(set(bi)) / len(bi))
        rep2.append(1 - len(set(bi)) / len(bi))
        counts.update(s)
    top10 = sum(c for _, c in counts.most_common(10)) / sum(counts.values())
    return {'unigram_entropy': unigram_entropy(toks), 'distinct1': float(np.mean(d1)),
            'distinct2': float(np.mean(d2)), 'repeated_bigram_rate': float(np.mean(rep2)), 'top10_token_share': top10}


@torch.no_grad()
def main():
    args = parse_args()
    device = 'cuda'
    teacher, info = load_teacher(args.teacher, device)
    V, Lseq = info['vocab'], info['length']
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    g = torch.Generator(device=device).manual_seed(args.seed)
    zs = [torch.randn((64, Lseq, V), device=device, generator=g) for _ in range(args.n_stats // 64)]
    models = {'teacher': lambda z: lens.model_logprobs(teacher, z, 0.0, 1.0)}
    for name in args.runs.split(','):
        s, _, _, _ = load_student(os.path.join(HERE, 'results', 'compress', name), teacher, device)
        models[name] = (lambda m: (lambda z: m(z, z.new_zeros(z.shape[0]), z.new_ones(z.shape[0]))))(s)
    out = {}
    for name, fn in models.items():
        toks = torch.cat([fn(z).argmax(-1).cpu() for z in zs])
        st = rep_stats(toks)
        st['samples'] = [tok.decode(t) for t in toks[:args.n_show]]
        top = collections.Counter(toks.flatten().tolist()).most_common(10)
        st['top10_tokens'] = [(tok.convert_ids_to_tokens(i), round(c / toks.numel(), 4)) for i, c in top]
        out[name] = st
        print(f"== {name}: " + ', '.join(f'{k} {v:.3f}' for k, v in st.items() if isinstance(v, float)))
        print('   top10', st['top10_tokens'])
        for smp in st['samples']:
            print('   >', smp[:400])
    os.makedirs(os.path.dirname(os.path.join(HERE, args.out)), exist_ok=True)
    with open(os.path.join(HERE, args.out), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
