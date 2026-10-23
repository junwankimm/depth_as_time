"""Training-free diagnostics for tying FMLM blocks (before depth-as-time Sec. 5).

For the one-step query (tau_s, tau_t) = (0, 1):
  * residual-stream norm ||h_i|| per token across depth (the paper normalizes the
    hidden-state loss because norms grow with depth);
  * norm and cosine similarity of consecutive residual updates
    Delta_i = h_i - h_{i-1};
  * block substitution: run the network with block i replaced by block j
    (all other blocks unchanged) and measure the output's token agreement with,
    and KL from, the original output. Interchangeable blocks are what a single
    shared block must emulate.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lens  # noqa: E402
from compress import load_teacher  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='FMLM-B-LM1B')
    p.add_argument('--n_samples', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--out', default='results/blocks')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(0)
    model, info = load_teacher(args.teacher, 'cuda')
    V, Lseq, L = info['vocab'], info['length'], len(model.blocks)
    out_dir = os.path.join(HERE, args.out, info['dataset'])
    os.makedirs(out_dir, exist_ok=True)
    norms, upd_norm, upd_cos = [], [], []
    swap_agree = np.zeros((L, L))
    swap_kl = np.zeros((L, L))
    n_b = max(1, args.n_samples // args.batch_size)
    original = list(model.blocks)
    for _ in range(n_b):
        z = torch.randn((args.batch_size, Lseq, V), device='cuda')
        hs, c = lens.hidden_states(model, z, 0.0, 1.0)
        hs = [h.float() for h in hs]
        norms.append([h.norm(dim=-1).mean().item() for h in hs])
        d = [hs[i] - hs[i - 1] for i in range(1, L + 1)]
        upd_norm.append([x.norm(dim=-1).mean().item() for x in d])
        upd_cos.append([[torch.nn.functional.cosine_similarity(d[i], d[j], dim=-1).mean().item()
                         for j in range(L)] for i in range(L)])
        ref = lens.readout(model, hs[-1], c)
        y = ref.argmax(-1)
        for i in range(L):
            for j in range(L):
                model.blocks[i] = original[j]
                lp = lens.model_logprobs(model, z, 0.0, 1.0)
                model.blocks[i] = original[i]
                swap_agree[i, j] += (lp.argmax(-1) == y).float().mean().item() / n_b
                swap_kl[i, j] += (ref.exp() * (ref - lp)).sum(-1).mean().item() / n_b
    res = {'norm': np.mean(norms, 0).round(3).tolist(),
           'update_norm': np.mean(upd_norm, 0).round(3).tolist(),
           'update_cos': np.mean(upd_cos, 0).round(3).tolist(),
           'swap_agree': swap_agree.round(3).tolist(), 'swap_kl': swap_kl.round(3).tolist()}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(info['dataset'])
    print('norm by depth  ', np.array(res['norm']))
    print('update norm    ', np.array(res['update_norm']))
    print('update cos (i, i+1)', np.array([res['update_cos'][i][i + 1] for i in range(L - 1)]))
    print('swap agreement (row: position i, col: block j)')
    print(swap_agree)
    print('swap KL')
    print(swap_kl)


if __name__ == '__main__':
    main()
