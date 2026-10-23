"""Full evaluation of the weight-tied FMLM students (compress.py) under the FLM paper's
protocol: 1,024 one-step samples per seed, GPT-2 Large Gen. PPL and per-sample unigram
entropy (paper Sec. 6, "Evaluation"), plus token agreement and KL with the teacher from
the same noise. Several seeds give a spread; the teacher is evaluated on the same noise.

Each student is rebuilt from its eval.json (variant, K, groups, warm-start blocks) and
last.pt, with the EMA weights swapped in (as in compress.py's evaluation).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lens  # noqa: E402
from compress import TiedFMLM, load_teacher  # noqa: E402
from probe_onestep import GenPPL, unigram_entropy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='FMLM-B-LM1B')
    p.add_argument('--runs', default='results/compress/lm1b_*', help='glob of compress.py run dirs')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--ppl_model', default='openai-community/gpt2-large')
    p.add_argument('--n_samples', type=int, default=1024, help='per seed (paper: 1,024)')
    p.add_argument('--seeds', default='1,2,3')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--out', default='results/compress_eval/lm1b.json')
    return p.parse_args()


def load_student(run_dir, teacher, device):
    e = json.load(open(os.path.join(run_dir, 'eval.json')))
    a = e['args']
    K = a['K'] or len(teacher.blocks)
    s = TiedFMLM(teacher, a['variant'], K, a['split'], a.get('groups', '0|1|2-5|6|7-10|11'), e['init']).to(device)
    ck = torch.load(os.path.join(run_dir, 'last.pt'), map_location=device)
    s.load_state_dict(ck['student'])
    params = [p for p in s.parameters() if p.requires_grad]
    assert len(params) == len(ck['ema'])
    with torch.no_grad():
        for p, v in zip(params, ck['ema']):
            p.copy_(v)
    return s.eval(), e['name'], ck['step'], sum(p.numel() for p in params)


@torch.no_grad()
def sample_tokens(model_fn, z_batches):
    return [model_fn(z).argmax(-1).cpu() for z in z_batches]


def main():
    args = parse_args()
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher, info = load_teacher(args.teacher, device)
    V, Lseq = info['vocab'], info['length']
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    gen = GenPPL(args.ppl_model, device)
    seeds = [int(s) for s in args.seeds.split(',')]
    runs = sorted(d for d in glob.glob(os.path.join(HERE, args.runs)) if os.path.exists(os.path.join(d, 'last.pt')))
    print('runs:', [os.path.basename(r) for r in runs], flush=True)

    def z_batches(seed):
        g = torch.Generator(device=device).manual_seed(seed)
        n_b = args.n_samples // args.batch_size
        for _ in range(n_b):
            yield torch.randn((args.batch_size, Lseq, V), device=device, generator=g)

    def teacher_fn(z):
        return lens.model_logprobs(teacher, z, 0.0, 1.0)

    # teacher tokens per seed (reference for agreement), and the teacher's own quality
    ref = {}
    res = {'protocol': {'n_samples_per_seed': args.n_samples, 'seeds': seeds, 'ppl_model': args.ppl_model,
                        'teacher': args.teacher}, 'models': {}}
    rows = {'ppl': [], 'ent': []}
    for sd in seeds:
        toks = torch.cat(sample_tokens(teacher_fn, z_batches(sd)))
        ref[sd] = toks
        rows['ppl'].append(gen(tok.batch_decode(toks), Lseq))
        rows['ent'].append(unigram_entropy(toks))
    res['models']['teacher'] = {'gen_ppl': rows['ppl'], 'entropy': rows['ent'], 'agree': [1.0] * len(seeds),
                                'kl': [0.0] * len(seeds), 'n_trainable': None}
    print('teacher', np.mean(rows['ppl']), np.mean(rows['ent']), flush=True)

    for rd in runs:
        student, name, step, n_tr = load_student(rd, teacher, device)
        z0, o0 = torch.zeros(args.batch_size, device=device), torch.ones(args.batch_size, device=device)
        r = {'gen_ppl': [], 'entropy': [], 'agree': [], 'kl': [], 'step': step, 'n_trainable': n_tr}
        for sd in seeds:
            toks, kls = [], []
            for z in z_batches(sd):
                ls = student(z, z0, o0)
                lt = teacher_fn(z)
                toks.append(ls.argmax(-1).cpu())
                kls.append((lt.exp() * (lt - ls)).sum(-1).mean().item())
            toks = torch.cat(toks)
            r['agree'].append((toks == ref[sd]).float().mean().item())
            r['kl'].append(float(np.mean(kls)))
            r['gen_ppl'].append(gen(tok.batch_decode(toks), Lseq))
            r['entropy'].append(unigram_entropy(toks))
        res['models'][name] = r
        print(name, {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) for k, v in r.items()
                     if isinstance(v, list)}, flush=True)
        del student
        torch.cuda.empty_cache()

    out = os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('\n| model | Gen. PPL | entropy | agree | KL |')
    for k, v in res['models'].items():
        f = lambda x: f"{np.mean(x):.3f} ± {np.std(x):.3f}"  # noqa: E731
        print(f"| {k} | {f(v['gen_ppl'])} | {f(v['entropy'])} | {f(v['agree'])} | {f(v['kl'])} |")


if __name__ == '__main__':
    main()
