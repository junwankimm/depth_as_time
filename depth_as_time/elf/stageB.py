"""Plan v2, stage B: read the ELF-B-PD blocks with closed-form ridge probes (frozen models).

Features: block hiddens h_i (main tokens, 768-d + bias) of the one-step query (t=0, x_prev=0, w).
Targets along the PD-faithful teacher ruler (stage A1, quantile grid), every 2nd sub-step:
  xhat   teacher clean estimate x_hat_t = net(z_t, t)            (primary)
  delta  flow-map endpoint delta_{0,t} = (z_t - (1 - t) z0) / t   (delta at t=0 := x_hat_0)
  z      the state z_t itself                                     (secondary; noise part is
                                                                   rank-128 visible, see PLAN.md)
  answer the student's own output (one target)
Feature sources: the student (main), a randomly initialized ELF-B fed the same input (control ii),
and the student with targets taken from another sequence (control iii). h_0 is control i.

With --center_pos (default), features and targets have their per-position train means removed, so
the probes see only sequence-specific variation. Without it, the x_hat_t targets at small t are
dominated by a sequence-independent per-position mean that any position code predicts (v2 run 1:
shuffled-target R^2 0.76).

Ridge is solved from sufficient statistics accumulated per split (by sequence: train / val / test),
so no tokens are stored. lambda is chosen on val per (source, target, block); R^2 is reported on test
(variance-weighted over the 512 latent dims, test-mean centred). The native lens final_layer(h_i)
is scored on the same test tokens without fitting (B2). B3 decodes probe-predicted x_hat_{t*} for a
few test sequences.
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
from modules.model import ELF_models  # noqa: E402

SPLITS = ['train', 'val', 'test']
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # times mean eigenvalue of X^T X


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--w', type=float, default=2.5)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--n_text', type=int, default=2)
    p.add_argument('--center_pos', type=int, default=1,
                   help='subtract per-position means (train split) from features and targets')
    p.add_argument('--out', default='results/elf/v2/stageB.json')
    return p.parse_args()


def split_of(n):
    r = n % 16
    return 'train' if r < 10 else ('val' if r < 13 else 'test')


class Stats:
    """Per split: X^T X (K+1 blocks), X^T Y and sum ||Y||^2 per target group."""

    def __init__(self, K, D, G, C, dev):
        self.A = {s: torch.zeros(K, D + 1, D + 1, dtype=torch.float64, device=dev) for s in SPLITS}
        self.B = {s: torch.zeros(K, D + 1, G * C, dtype=torch.float64, device=dev) for s in SPLITS}
        self.yy = {s: torch.zeros(G, dtype=torch.float64, device=dev) for s in SPLITS}
        self.n = {s: 0 for s in SPLITS}
        self.G, self.C = G, C

    def add(self, split, feats, Y):
        """feats: list over blocks of (N, D); Y: (N, G*C)."""
        Y = Y.double()
        for i, f in enumerate(feats):
            X = torch.cat([f.double(), torch.ones(f.shape[0], 1, dtype=torch.float64, device=f.device)], 1)
            self.A[split][i] += X.T @ X
            self.B[split][i] += X.T @ Y
        self.yy[split] += Y.pow(2).view(-1, self.G, self.C).sum((0, 2))
        self.n[split] += Y.shape[0]

    def r2(self, W, split, i):
        """Per-group R^2 of predictor X W on split (test-mean centred)."""
        A, Bm, n = self.A[split][i], self.B[split][i], self.n[split]
        G, C = self.G, self.C
        sse = self.yy[split] - 2 * (W * Bm).view(-1, G, C).sum((0, 2)) + (W * (A @ W)).view(-1, G, C).sum((0, 2))
        ysum = Bm[-1].view(G, C)                         # bias row = sum of Y
        sst = self.yy[split] - ysum.pow(2).sum(1) / n
        return 1 - sse / sst

    def solve(self, i):
        """lambda chosen on val (mean R^2 over groups); returns (W, lambda, test R^2 per group)."""
        A = self.A['train'][i]
        D1 = A.shape[0]
        scale = torch.diagonal(A)[:-1].mean()
        reg = torch.eye(D1, dtype=A.dtype, device=A.device)
        reg[-1, -1] = 0
        best = None
        for lam in LAMBDAS:
            W = torch.linalg.solve(A + lam * scale * reg, self.B['train'][i])
            v = self.r2(W, 'val', i).mean().item()
            if best is None or v > best[0]:
                best = (v, lam, W)
        return best[2], best[1], self.r2(best[2], 'test', i)


def random_elf(cfg, ref, dev, seed=0):
    torch.manual_seed(seed)
    m = ELF_models[cfg['model']](
        text_encoder_dim=ref.text_encoder_dim, max_length=cfg['max_length'], bottleneck_dim=cfg['bottleneck_dim'],
        num_time_tokens=cfg['num_time_tokens'], num_self_cond_cfg_tokens=cfg['num_self_cond_cfg_tokens'],
        num_model_mode_tokens=cfg['num_model_mode_tokens'], vocab_size=ref.vocab_size)
    m.cfg = cfg
    return m.to(dev).eval().requires_grad_(False)


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    dev = 'cuda'
    tok = transformers.AutoTokenizer.from_pretrained(E.T5_TOK)
    stu = E.load_elf(E.STUDENT, dev)
    tea = E.load_elf(E.TEACHER, dev)
    rnd = random_elf(stu.cfg, stu, dev)
    grid = E.pd_grid(tea.cfg, a.steps, quantile=True)
    ks = list(range(0, len(grid), a.stride))
    ts = [grid[k] for k in ks]
    K, D, C, Tn = len(stu.blocks) + 1, stu.hidden_size, stu.text_encoder_dim, len(ks)
    tg = {'xhat': Tn, 'delta': Tn, 'z': Tn, 'answer': 1}
    sources = {'student': ['xhat', 'delta', 'z', 'answer'], 'random': ['xhat', 'delta', 'answer'],
               'shuffled': ['xhat', 'delta', 'answer']}
    stats = {(s, t): Stats(K, D, tg[t], C, dev) for s, ts_ in sources.items() for t in ts_}
    lens_sse = torch.zeros(K, Tn, dtype=torch.float64)
    lens_n = 0
    lens_yy = torch.zeros(Tn, dtype=torch.float64)
    lens_ysum = torch.zeros(Tn, C, dtype=torch.float64)

    def batch_stream():
        g = torch.Generator(device=dev).manual_seed(a.seed)
        for b0 in range(0, a.n, a.bs):
            B = min(a.bs, a.n - b0)
            z0 = E.noise(stu, B, g)
            xs, hs = E.one_step(stu, z0, a.w, return_hidden=True)
            _, hr = E.one_step(rnd, z0, a.w, return_hidden=True)
            z1, traj = E.pd_ruler(tea, z0, grid, a.w)
            states = [(t, z, x) for t, z, x in traj] + [(1.0, z1, z1)]
            tgt = {'xhat': torch.stack([states[k][2] for k in ks], 2),            # (B, L, Tn, C)
                   'delta': torch.stack([states[k][2] if k == 0 else
                                         (states[k][1] - (1 - states[k][0]) * z0) / states[k][0] for k in ks], 2),
                   'z': torch.stack([states[k][1] for k in ks], 2),
                   'answer': xs.float()[:, :, None]}
            del traj, states
            yield b0, B, z0, hs, hr, tgt

    # pass 0: per-position means over train sequences (features per block and source, targets per type)
    mu_f = {s: torch.zeros(K, stu.cfg['max_length'], D, device=dev) for s in ['student', 'random']}
    mu_t = {t: torch.zeros(stu.cfg['max_length'], g * C, device=dev) for t, g in tg.items()}
    n_tr = 0
    if a.center_pos:
        for b0, B, z0, hs, hr, tgt in batch_stream():
            for n in range(B):
                if split_of(b0 + n) != 'train':
                    continue
                for i in range(K):
                    mu_f['student'][i] += hs[i][n]
                    mu_f['random'][i] += hr[i][n]
                for t in tg:
                    mu_t[t] += tgt[t][n].reshape(stu.cfg['max_length'], -1)
                n_tr += 1
        for d in (mu_f, mu_t):
            for k in d:
                d[k] /= n_tr
        print(f'per-position means from {n_tr} train sequences', flush=True)

    for b0, B, z0, hs_raw, hr, tgt_raw in batch_stream():
        hs = [h - mu_f['student'][i] for i, h in enumerate(hs_raw)]
        hr = [h - mu_f['random'][i] for i, h in enumerate(hr)]
        tgt = {t: v - mu_t[t].view(1, v.shape[1], v.shape[2], C) for t, v in tgt_raw.items()}
        for n in range(B):
            sp = split_of(b0 + n)
            m = (n + 1) % B  # another sequence in the batch, for the shuffled control
            for s, tnames in sources.items():
                feats = [h[n] for h in (hr if s == 'random' else hs)]
                for t in tnames:
                    Y = tgt[t][m if s == 'shuffled' else n].reshape(feats[0].shape[0], -1)
                    stats[(s, t)].add(sp, feats, Y)
            if sp == 'test':  # B2: native lens on the raw (uncentred) states, no fitting
                Y = tgt_raw['xhat'][n]                                        # (L, Tn, C)
                for i, h in enumerate(hs_raw):
                    P = E.readout_x(stu, h[n])[:, None]
                    lens_sse[i] += (Y - P).pow(2).sum((0, 2)).double().cpu()
                lens_yy += Y.pow(2).sum((0, 2)).double().cpu()
                lens_ysum += Y.sum(0).double().cpu()
                lens_n += Y.shape[0]
        print(f'batch {b0 // a.bs} accumulated', flush=True)

    res = {'t': ts, 'ks': ks, 'n': a.n, 'w': a.w, 'r2': {}, 'lambda': {}}
    W_keep = {}
    for (s, t), st in stats.items():
        rows, lams = [], []
        for i in range(K):
            W, lam, r2 = st.solve(i)
            rows.append(r2.cpu().tolist())
            lams.append(lam)
            if s == 'student' and t == 'xhat':
                W_keep[i] = W.float()
        res['r2'][f'{s}/{t}'] = rows
        res['lambda'][f'{s}/{t}'] = lams
    lens_sst = lens_yy - lens_ysum.pow(2).sum(1) / lens_n
    res['r2']['lens/xhat'] = (1 - lens_sse / lens_sst[None]).tolist()

    # summaries: t* per block, interior-peak margins vs t -> 0 and vs the answer
    summ = {}
    for key in ['student/xhat', 'student/delta', 'student/z', 'random/xhat', 'random/delta',
                'shuffled/xhat', 'lens/xhat']:
        r = np.array(res['r2'][key])
        best = r.argmax(1)
        ans_key = key.split('/')[0] + '/answer'
        ans = np.array(res['r2'][ans_key])[:, 0] if ans_key in res['r2'] else np.full(K, np.nan)
        summ[key] = {'t_star': [ts[j] for j in best], 'r2_star': r.max(1).tolist(), 'r2_t0': r[:, 0].tolist(),
                     'r2_t1': r[:, -1].tolist(), 'r2_answer': ans.tolist()}
    res['summary'] = summ

    # B3: decode probe-predicted x_hat_{t*} for the first test sequences, next to the ruler's view
    t_star_idx = np.array(res['r2']['student/xhat']).argmax(1)
    texts = []
    for b0, B, z0, hs, hr, tgt in batch_stream():
        test_ids = [n for n in range(B) if split_of(b0 + n) == 'test'][:a.n_text]
        for n in test_ids:
            rows = []
            for i in range(K):
                j = int(t_star_idx[i])
                X = torch.cat([hs[i][n] - mu_f['student'][i], torch.ones(hs[i].shape[1], 1, device=dev)], 1)
                pred = ((X @ W_keep[i]).view(-1, Tn, C) + mu_t['xhat'].view(-1, Tn, C))[None, :, j]
                rows.append({'block': i, 't_star': ts[j],
                             'probe_view': E.to_text(tok, E.decode_latent(tea, pred, a.w).argmax(-1))[0][:200],
                             'ruler_view': E.to_text(tok, E.decode_latent(tea, tgt['xhat'][n:n + 1, :, j], a.w)
                                                     .argmax(-1))[0][:200]})
            texts.append(rows)
        break
    res['views'] = texts
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)

    print('t grid', [round(t, 3) for t in ts])
    for key, s in summ.items():
        print(f'\n[{key}] | block | t* | R2* | R2(t=0) | R2(t=1) | R2(answer) |')
        for i in range(K):
            print(f"| {i} | {s['t_star'][i]:.3f} | {s['r2_star'][i]:.3f} | {s['r2_t0'][i]:.3f} | "
                  f"{s['r2_t1'][i]:.3f} | {s['r2_answer'][i]:.3f} |")


if __name__ == '__main__':
    main()
