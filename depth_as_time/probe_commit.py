"""Per-token commit analysis: does the one-step FMLM unmask tokens across depth?

For every position of the one-step sample y (block-12 argmax) we record
  * commit depth c: the first block i such that argmax(delta^j) = y for all j >= i;
  * the number of argmax switches across depth, and what the position showed just
    before committing (the full-noise marginal's argmax, i.e. a mask-like filler;
    the noise-seeded token argmax(log m + 2 z); or another token);
  * covariates: final confidence p_12(y), final entropy, marginal probability m(y),
    whether y is noise-seeded, position index;
  * the teacher's lock step: along the FLM teacher's 128-step trajectory from the
    same z, the first step after which its clean prediction's argmax stays at its own
    final token.

Outputs per-position arrays (stats.npz) and a summary (summary.json).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import lens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-LM1B')
    p.add_argument('--teacher', default='FLM-B-LM1B')
    p.add_argument('--n_samples', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--teacher_steps', type=int, default=128)
    p.add_argument('--beta', type=float, default=2.0, help='noise-seeding weight')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/commit/lm1b')
    return p.parse_args()


def lock_index(tok, final):
    """tok: (n, B, L) argmax sequence; returns the first index from which tok == final
    holds to the end, shape (B, L)."""
    same = (tok == final.unsqueeze(0))
    # suffix-all: flip, cumulative product, flip back
    suffix = torch.flip(torch.cumprod(torch.flip(same, [0]).int(), 0), [0]).bool()
    n = tok.shape[0]
    first = torch.where(suffix, torch.arange(n, device=tok.device).view(-1, 1, 1), n).min(0).values
    return first


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    student = lens.load_backbone(args.student, device)
    teacher = lens.load_backbone(args.teacher, device)
    V, Lseq = student.vocab_size, student.config.model.length
    nL = len(student.blocks) + 1
    lut = lens.TimeLUT(V)
    K = args.teacher_steps
    keys = ['commit', 'switches', 'pre', 'conf', 'ent', 'marg_p', 'seeded', 'pos', 't_lock', 'y', 'y_t']
    acc = {k: [] for k in keys}
    t0 = time.time()
    for b in range(args.n_samples // args.batch_size):
        B = args.batch_size
        z = torch.randn((B, Lseq, V), device=device)
        s_logp, _, _ = lens.layer_logprobs(student, z, 0.0, 1.0)
        tok = torch.stack([lp.argmax(-1) for lp in s_logp])  # (nL, B, L)
        y = tok[-1]
        fin = s_logp[-1]
        conf = fin.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
        ent = -(fin.exp() * fin).sum(-1)
        del s_logp
        log_m = lens.model_logprobs(teacher, z, 0.0)  # D_0(z): per-position full-noise marginal
        marg_tok = log_m.argmax(-1)
        seed_tok = (log_m + args.beta * z).argmax(-1)
        marg_p = log_m.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
        commit = lock_index(tok, y)
        switches = (tok[1:] != tok[:-1]).sum(0)
        prev = tok.gather(0, (commit - 1).clamp_min(0).unsqueeze(0)).squeeze(0)
        # what the position showed right before committing: 0 committed at block 0,
        # 1 marginal filler, 2 noise-seeded token, 3 other
        pre = torch.full_like(commit, 3)
        pre = torch.where(prev == seed_tok, 2, pre)
        pre = torch.where(prev == marg_tok, 1, pre)
        pre = torch.where(commit == 0, 0, pre)

        t_tok = []
        lens.flm_trajectory(teacher, lut, z, K, lambda k, tau, t, x, lp: t_tok.append(lp.argmax(-1)))
        t_tok = torch.stack(t_tok)  # (K, B, L)
        y_t = t_tok[-1]
        t_lock = lock_index(t_tok, y_t)

        for k, v in [('commit', commit), ('switches', switches), ('pre', pre), ('conf', conf), ('ent', ent),
                     ('marg_p', marg_p), ('seeded', (seed_tok == y)),
                     ('pos', torch.arange(Lseq, device=device).expand(B, Lseq)), ('t_lock', t_lock),
                     ('y', y), ('y_t', y_t)]:
            acc[k].append(v.cpu().numpy())
        del z, log_m, tok, t_tok
        torch.cuda.empty_cache()
        print(f'batch {b + 1} ({time.time() - t0:.0f}s)', flush=True)

    A = {k: np.concatenate(v, 0) for k, v in acc.items()}  # (N, L)
    np.savez_compressed(os.path.join(args.out, 'stats.npz'), **A)
    c, f = A['commit'], lambda x: x.reshape(-1)
    hist = np.bincount(f(c), minlength=nL + 1)[:nL] / c.size
    agree = A['y'] == A['y_t']
    # within-sample spread of commit depth vs. overall spread (global commit => small within-sample std)
    within = float(np.mean(c.std(1)))
    summary = {
        'n_positions': int(c.size), 'commit_hist': hist.round(4).tolist(),
        'commit_mean': float(c.mean()), 'commit_std_all': float(c.std()), 'commit_std_within_sample': within,
        'switches_hist': (np.bincount(f(A['switches']), minlength=6)[:6] / c.size).round(4).tolist(),
        'pre_commit_kind': {name: float((A['pre'] == i).mean()) for i, name in
                            enumerate(['at_block0', 'marginal_filler', 'noise_seeded', 'other'])},
        'spearman_commit_vs': {
            'final_conf': spearman(f(c), -f(A['conf'])),  # > 0: confident tokens commit earlier
            'final_entropy': spearman(f(c), f(A['ent'])),
            'marginal_prob': spearman(f(c), -f(A['marg_p'])),  # > 0: frequent tokens commit earlier
            'position': spearman(f(c), f(A['pos'])),
            'teacher_lock_step (positions where student = teacher)': spearman(c[agree], A['t_lock'][agree]),
        },
        'commit_mean_by_seeded': {'seeded': float(c[A['seeded']].mean()), 'not_seeded': float(c[~A['seeded']].mean())},
        'frac_seeded': float(A['seeded'].mean()), 'frac_student_eq_teacher': float(agree.mean()),
    }
    # mean commit depth by final-confidence decile
    q = np.quantile(f(A['conf']), np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(f(A['conf']), q[1:-1]), 0, 9)
    summary['commit_by_conf_decile'] = [float(f(c)[bins == i].mean()) for i in range(10)]
    # per-sample Spearman between commit depth and teacher lock step (agreeing positions)
    per = [spearman(c[n][agree[n]], A['t_lock'][n][agree[n]]) for n in range(c.shape[0]) if agree[n].sum() > 5]
    summary['spearman_teacher_lock_per_sample_mean'] = float(np.nanmean(per))
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
