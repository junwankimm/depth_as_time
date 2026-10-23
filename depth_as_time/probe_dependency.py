"""Do dependent tokens commit together? Token-level analysis on the one-step FMLM.

1. Dependency: for a noise sample z, resample the noise at one position j and measure how
   much the one-step output distribution moves at every other position i,
   D_ij = KL(p_i(z) || p_i(z with z_j resampled)). Averaged over two resamples.
   We test whether pairs with larger dependency commit at more similar depths, controlling
   for their distance, and whether context-dependent tokens (large sum_j D_ij) commit later.
2. Token classes: commit depth by kind of token (punctuation, function word, word piece,
   number, content word).
3. Multi-step sampling: within every step of 4-step FMLM sampling, the per-token commit
   depth (relative to that step's own output) and its rank correlation with confidence.
"""

import argparse
import json
import os
import string
import time

import numpy as np
import torch
import transformers

import lens
from probe_commit import lock_index, spearman

FUNCTION_WORDS = set('''a an the and or but if of to in on at by for with from as is are was were be been being
have has had do does did not no it its this that these those he she they we you i me him her them us my your
his our their there here which who whom whose what when where why how all any each some such than then so too
very can could will would shall should may might must just also only about into over after before up down out
off again further once more most other s t 's'''.split())


def token_class(tok_str, gpt=False):
    s = tok_str.replace('Ġ', '').replace('▁', '').strip().lower()
    if tok_str.startswith('##') or (gpt and not tok_str.startswith('Ġ') and s.isalpha()):
        return 'word piece'
    if s == '' or all(ch in string.punctuation for ch in s) or s in ('[cls]', '[sep]', '<|endoftext|>'):
        return 'punctuation / special'
    if s.replace(',', '').replace('.', '').isdigit():
        return 'number'
    if s in FUNCTION_WORDS:
        return 'function word'
    return 'content word'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-LM1B')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--n_dep', type=int, default=32, help='noise samples for the dependency matrices')
    p.add_argument('--n_resample', type=int, default=2)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--n_cls', type=int, default=256, help='samples for the token-class analysis')
    p.add_argument('--n_roll', type=int, default=128, help='samples for the multi-step analysis')
    p.add_argument('--roll_steps', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/dependency/lm1b')
    return p.parse_args()


@torch.no_grad()
def commit_and_conf(model, x, tau_s, tau_t):
    lps, _, _ = lens.layer_logprobs(model, x, tau_s, tau_t)
    tok = torch.stack([lp.argmax(-1) for lp in lps])
    y = tok[-1]
    conf = lps[-1].gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
    return lock_index(tok, y), y, conf, lps[-1]


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    model = lens.load_backbone(args.student, device)
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    V, L = model.vocab_size, model.config.model.length
    nL = len(model.blocks) + 1
    res = {}
    t0 = time.time()

    # ---------------- 1. dependency matrices ----------------
    commits, deps, confs = [], [], []
    for n in range(args.n_dep):
        z = torch.randn((1, L, V), device=device)
        c, y, conf, lp = commit_and_conf(model, z, 0.0, 1.0)
        D = torch.zeros(L, L, device=device)  # D[j, i]: effect of resampling position j on position i
        for _ in range(args.n_resample):
            for j0 in range(0, L, args.chunk):
                js = torch.arange(j0, min(L, j0 + args.chunk), device=device)
                zz = z.expand(len(js), L, V).clone()
                zz[torch.arange(len(js)), js] = torch.randn((len(js), V), device=device)
                lp2 = lens.model_logprobs(model, zz, 0.0, 1.0)
                D[js] += (lp.exp() * (lp - lp2)).sum(-1) / args.n_resample
        commits.append(c[0].cpu().numpy()); deps.append(D.cpu().numpy()); confs.append(conf[0].cpu().numpy())
        if n % 8 == 0:
            print(f'dep sample {n} ({time.time() - t0:.0f}s)', flush=True)
    C, Dm, Cf = np.stack(commits), np.stack(deps), np.stack(confs)  # (N, L), (N, L, L), (N, L)
    np.savez_compressed(os.path.join(args.out, 'dependency.npz'), commit=C, dep=Dm, conf=Cf)
    iu = np.triu_indices(L, 1)
    dist = (iu[1] - iu[0])
    sym = [(Dm[n] + Dm[n].T)[iu] / 2 for n in range(len(C))]
    gap = [np.abs(C[n][iu[0]] - C[n][iu[1]]) for n in range(len(C))]
    # within distance bins: Spearman(dependency, -|commit gap|), pooled over samples
    bins = [(1, 1), (2, 3), (4, 8), (9, 16), (17, 32), (33, 127)]
    by_bin = {}
    for lo, hi in bins:
        m = (dist >= lo) & (dist <= hi)
        a = np.concatenate([s[m] for s in sym]); b = np.concatenate([g[m] for g in gap])
        by_bin[f'{lo}-{hi}'] = {'spearman_dep_vs_neg_gap': spearman(a, -b), 'mean_dep': float(a.mean()),
                                'mean_gap': float(b.mean())}
    # high- vs low-dependency pairs at the same distance: commit gap
    q = []
    for lo, hi in bins:
        m = (dist >= lo) & (dist <= hi)
        a = np.concatenate([s[m] for s in sym]); b = np.concatenate([g[m] for g in gap])
        hi_m = a >= np.quantile(a, 0.9); lo_m = a <= np.quantile(a, 0.5)
        q.append({'bin': f'{lo}-{hi}', 'gap_top10pct_dep': float(b[hi_m].mean()), 'gap_bottom50pct_dep': float(b[lo_m].mean())})
    # context dependence of a position: how much other positions' noise moves it
    dep_in = np.stack([Dm[n].sum(0) - np.diag(Dm[n]) for n in range(len(C))])
    self_dep = np.stack([np.diag(Dm[n]) for n in range(len(C))])
    res['dependency'] = {
        'n_samples': len(C), 'by_distance': by_bin, 'gap_by_dependency': q,
        'spearman_commit_vs_context_dependence': spearman(C.ravel(), dep_in.ravel()),
        'spearman_commit_vs_own_noise_dependence': spearman(C.ravel(), -self_dep.ravel()),
        'spearman_conf_vs_context_dependence': spearman(Cf.ravel(), -dep_in.ravel()),
        'mean_dep_by_distance': [float(np.mean([s[dist == d].mean() for s in sym])) for d in range(1, 33)],
        'commit_corr_by_distance': [float(np.corrcoef(np.concatenate([C[n][:-d] for n in range(len(C))]),
                                                      np.concatenate([C[n][d:] for n in range(len(C))]))[0, 1]) for d in range(1, 33)],
    }
    print('dependency', json.dumps(res['dependency']['by_distance']), flush=True)

    # ---------------- 2. token classes ----------------
    cls_commit = {}
    for b in range(args.n_cls // args.batch_size):
        z = torch.randn((args.batch_size, L, V), device=device)
        c, y, _, _ = commit_and_conf(model, z, 0.0, 1.0)
        for ci, yi in zip(c.flatten().tolist(), y.flatten().tolist()):
            cls_commit.setdefault(token_class(tok.convert_ids_to_tokens(yi), 'gpt2' in args.tokenizer), []).append(ci)
    res['token_class'] = {k: {'n': len(v), 'share': len(v) / sum(len(x) for x in cls_commit.values()),
                              'mean_commit': float(np.mean(v)),
                              'hist': (np.bincount(v, minlength=nL)[:nL] / len(v)).round(4).tolist()}
                          for k, v in sorted(cls_commit.items())}
    print('token classes', {k: (round(v['share'], 3), round(v['mean_commit'], 2)) for k, v in res['token_class'].items()}, flush=True)

    # ---------------- 3. multi-step sampling ----------------
    lut = lens.TimeLUT(V)
    sched = lens.fmlm_step_times(lut, args.roll_steps, 0.0, device)
    per_step = [{'commit': [], 'conf': []} for _ in sched]
    for b in range(args.n_roll // args.batch_size):
        x = torch.randn((args.batch_size, L, V), device=device)
        for j, (tau_s, tau_tilde, t_s, t_tilde, t_next) in enumerate(sched):
            c, y, conf, lp = commit_and_conf(model, x, tau_s, tau_tilde)
            per_step[j]['commit'].append(c.cpu().numpy()); per_step[j]['conf'].append(conf.cpu().numpy())
            x = lp.exp() if j == len(sched) - 1 else lens.fmlm_update(x, lp, t_s, t_tilde, t_next, 0.0)
    res['multistep'] = []
    for j, d in enumerate(per_step):
        c = np.concatenate(d['commit']).ravel(); cf = np.concatenate(d['conf']).ravel()
        res['multistep'].append({'step': j, 'tau': [round(float(sched[j][0]), 3), round(float(sched[j][1]), 3)],
                                 'spearman_commit_vs_conf': spearman(c, -cf), 'mean_commit': float(c.mean()),
                                 'hist': (np.bincount(c, minlength=nL)[:nL] / c.size).round(4).tolist()})
    print('multistep', [(r['step'], round(r['spearman_commit_vs_conf'], 3), round(r['mean_commit'], 2)) for r in res['multistep']], flush=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
