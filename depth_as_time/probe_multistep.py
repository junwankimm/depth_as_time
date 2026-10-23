"""Depth-as-time probe for FMLM queries with a noisy endpoint (t < 1).

Depth-as-time (Sec. 4.2) reports that interval-conditioned flow maps
(MeanFlow, Shortcut) "denoise, then renoise" across depth when the requested
endpoint is noisy. FMLM is interval-conditioned on (tau_s, tau_t) and outputs
the two-time denoiser delta_{s,t}, from which

    X_{s,t}(x) = (1-t)/(1-s) x + (t-s)/(1-s) delta_{s,t}(x)      (FLM Eq. 19).

delta_{s,t} is a convex average of single-time denoisers D_u(x_u) along the
path u in [s, t] (FLM Prop. 3.3), so for t < 1 the *target* is blurrier
(higher entropy, less mass on the eventual tokens) than the clean endpoint
delta_{s,1}. Because the noise part of X_{s,t} is fixed by the
parameterization, "renoise" in this model can only show up in delta: the
language analogue of denoise-then-renoise is sharpen-then-blur across depth,
i.e. intermediate layers are closer to the clean endpoint than the last layer.

Two experiments:
  sweep: fixed start (tau_s in {0, 0.5}), vary tau_t; decode every layer.
  rollout: N-step FMLM sampling (as in FMLM.generate_samples); decode every
           layer of every NFE (the analogue of Figs. 6/7).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import lens
from probe_onestep import agree, entropy, kl, max_prob, nll_of


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-LM1B')
    p.add_argument('--n_samples', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--sweep_starts', default='0.0,0.5')
    p.add_argument('--sweep_targets', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    p.add_argument('--rollout_steps', default='2,4,8')
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/multistep')
    return p.parse_args()


def layer_stats(logps, ref_tok, ref_logp=None):
    """Per-layer statistics of one NFE, each (B, L+1).

    ref_tok: tokens of the clean reference; ref_logp: optional clean reference
    distribution (KL(ref || delta^i) is recorded as kl_clean).
    """
    fin = logps[-1]
    fin_tok = fin.argmax(-1)
    keys = ['ent', 'maxp', 'kl_final', 'agree_final', 'agree_clean', 'nll_clean']
    out = {k: [] for k in keys + (['kl_clean'] if ref_logp is not None else [])}
    for lp in logps:
        tok = lp.argmax(-1)
        out['ent'].append(entropy(lp))
        out['maxp'].append(max_prob(lp))
        out['kl_final'].append(kl(fin, lp))
        out['agree_final'].append(agree(tok, fin_tok))
        out['agree_clean'].append(agree(tok, ref_tok))
        out['nll_clean'].append(nll_of(lp, ref_tok))
        if ref_logp is not None:
            out['kl_clean'].append(kl(ref_logp, lp))
    return {k: torch.stack(v, -1).cpu().numpy() for k, v in out.items()}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    model = lens.load_backbone(args.student, device)
    V, Lseq = model.vocab_size, model.config.model.length
    lut = lens.TimeLUT(V)
    starts = [float(s) for s in args.sweep_starts.split(',')]
    targets = [float(t) for t in args.sweep_targets.split(',')]
    rollouts = [int(n) for n in args.rollout_steps.split(',')]

    acc = {}

    def push(key, stats):
        for k, v in stats.items():
            acc.setdefault(f'{key}/{k}', []).append(v)

    t0 = time.time()
    for b in range(args.n_samples // args.batch_size):
        B = args.batch_size
        z = torch.randn((B, Lseq, V), device=device)

        # ---------------- sweep over the requested endpoint ----------------
        for tau_s in starts:
            if tau_s == 0.0:
                x_s = z
            else:  # reach x_s with the model's own one-jump flow map
                t_s = lut.tau_to_t(torch.tensor([tau_s], device=device)).item()
                lp = lens.model_logprobs(model, z, 0.0, tau_s)
                x_s = lens.fmlm_update(z, lp, 0.0, t_s, t_s)
            t_s = lut.tau_to_t(torch.tensor([tau_s], device=device)).item()
            # clean reference from the same state: delta_{s,1}(x_s)
            ref = lens.model_logprobs(model, x_s, tau_s, 1.0)
            ref_tok = ref.argmax(-1)
            for tau_t in targets:
                if tau_t <= tau_s:
                    continue
                logps, _, _ = lens.layer_logprobs(model, x_s, tau_s, tau_t)
                push(f'sweep/s{tau_s:.2f}/t{tau_t:.2f}', layer_stats(logps, ref_tok, ref))
                del logps

        # ---------------- multi-step rollouts ------------------------------
        for n in rollouts:
            sched = lens.fmlm_step_times(lut, n, args.gamma, device)
            x, per_step = z, []
            for j, (tau_s, tau_tilde, t_s, t_tilde, t_next) in enumerate(sched):
                logps, _, _ = lens.layer_logprobs(model, x, tau_s, tau_tilde)
                per_step.append((x, tau_s, logps))
                x = (logps[-1].exp() if j == n - 1 else
                     lens.fmlm_update(x, logps[-1], t_s, t_tilde, t_next, args.gamma))
            y = x.argmax(-1)  # final sample of the rollout
            for j, (x_j, tau_s, logps) in enumerate(per_step):
                # reference = final sample y; also the clean jump delta_{s,1}(x_j)
                st = layer_stats(logps, y)
                ref = lens.model_logprobs(model, x_j, tau_s, 1.0)
                st['kl_jump'] = torch.stack([kl(ref, lp) for lp in logps], -1).cpu().numpy()
                push(f'rollout/n{n}/step{j}', st)
            acc.setdefault(f'rollout/n{n}/sched', [np.array(sched)])
            del per_step
        torch.cuda.empty_cache()
        print(f'batch {b + 1} ({time.time() - t0:.0f}s)', flush=True)

    out = {k: np.concatenate(v, 0) if not k.endswith('sched') else v[0]
           for k, v in acc.items()}
    np.savez_compressed(os.path.join(args.out, 'stats.npz'), **out)
    summary = {k: v.mean(0).round(4).tolist() for k, v in out.items()
               if not k.endswith('sched')}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=0)
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    for k in sorted(summary):
        if k.endswith('/ent') or k.endswith('/nll_clean'):
            print(k, ' '.join(f'{v:.2f}' for v in summary[k]))
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
