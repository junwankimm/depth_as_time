"""State-space depth-to-time matching for a one-step FMLM (LM1B or OWT).

Along a flow started at z, the state is x_t = X_{0,t}(z) = (1 - t) z + t delta_{0,t}(z)
(FLM Eq. 19), so placing the one-step layer readout delta^i at time t,
(1 - t) z + t delta^i, reproduces x_t iff delta^i = delta_{0,t}(z). We therefore
compare every layer readout with delta_{0,t} for a grid of t (distances in delta
space: the state distance carries a factor t and would always favour t = 0).

Two rulers for delta_{0,t}(z):
  teacher: the FLM teacher's 128-step Euler trajectory from the same z,
           delta^T_{0,t} = z + (x_t - z) / t.
  self:    FMLM's own flow-map output for the query (0 -> tau_t); identical to
           the last layer at tau_t = 1, so it has no endpoint mismatch.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import transformers

import lens
from probe_onestep import kl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-LM1B')
    p.add_argument('--teacher', default='FLM-B-LM1B')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--n_samples', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--teacher_steps', type=int, default=128)
    p.add_argument('--self_taus', type=int, default=33, help='grid size for the self ruler')
    p.add_argument('--n_text', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/statematch')
    return p.parse_args()


def l2(p, q):
    return (p - q).pow(2).sum((-1, -2)).sqrt()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    student = lens.load_backbone(args.student, device)
    teacher = lens.load_backbone(args.teacher, device)
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    V, Lseq = student.vocab_size, student.config.model.length
    nL = len(student.blocks) + 1
    lut = lens.TimeLUT(V)
    K = args.teacher_steps
    self_tau = torch.linspace(0, 1, args.self_taus, device=device)[1:]  # skip tau = 0 (t = 0)
    self_t = lut.tau_to_t(self_tau)
    self_t[-1] = 1.0
    nS = len(self_tau)

    n_b = args.n_samples // args.batch_size
    N = n_b * args.batch_size
    R = {k: np.zeros((N, nL, K - 1)) for k in ['T_l2', 'T_kl']}
    R.update({k: np.zeros((N, nL, nS)) for k in ['S_l2', 'S_kl']})
    teacher_tau, teacher_t = np.zeros(K - 1), np.zeros(K - 1)
    text = []
    t0 = time.time()
    for b in range(n_b):
        sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
        z = torch.randn((args.batch_size, Lseq, V), device=device)
        s_logp, _, _ = lens.layer_logprobs(student, z, 0.0, 1.0)
        s_p = [lp.exp() for lp in s_logp]

        # teacher ruler: delta^T_{0,t_k} from the Euler states x_{t_k}, k >= 1
        keep = {}

        def callback(k, tau_k, t_k, x_k, logp_k):
            if k == 0:
                return
            teacher_tau[k - 1], teacher_t[k - 1] = tau_k, t_k
            d = (z + (x_k - z) / t_k).clamp_min(0)
            d = d / d.sum(-1, keepdim=True)  # project the Euler estimate onto the simplex
            for i in range(nL):
                R['T_l2'][sl, i, k - 1] = l2(s_p[i], d).cpu().numpy()
                R['T_kl'][sl, i, k - 1] = kl(d.clamp_min(1e-12).log(), s_logp[i]).cpu().numpy()
            if b == 0:
                keep[k] = x_k[:args.n_text].argmax(-1).cpu()
        lens.flm_trajectory(teacher, lut, z, K, callback)

        # self ruler: FMLM's own delta_{0,t}(z)
        self_states = []
        for j in range(nS):
            lp = lens.model_logprobs(student, z, 0.0, self_tau[j])
            d = lp.exp()
            for i in range(nL):
                R['S_l2'][sl, i, j] = l2(s_p[i], d).cpu().numpy()
                R['S_kl'][sl, i, j] = kl(lp, s_logp[i]).cpu().numpy()
            if b == 0:
                self_states.append(((1 - self_t[j]) * z[:args.n_text] + self_t[j] * d[:args.n_text]).argmax(-1).cpu())

        if b == 0:  # qualitative: argmax of the layer state at its matched time vs the ruler state
            best_S = R['S_l2'][sl][:args.n_text].mean(0).argmin(1)
            for n in range(args.n_text):
                rows = []
                for i in range(nL):
                    j = int(best_S[i])
                    st = ((1 - self_t[j]) * z[n] + self_t[j] * s_p[i][n]).argmax(-1).cpu()
                    rows.append({'layer': i, 'tau': round(self_tau[j].item(), 3), 't': round(self_t[j].item(), 3),
                                 'layer_state': tok.decode(st[:48]),
                                 'self_state': tok.decode(self_states[j][n][:48])})
                text.append(rows)
            text_teacher = {int(k): [tok.decode(v[n][:48]) for n in range(args.n_text)]
                            for k, v in keep.items() if k in (1, 8, 16, 32, 64, 96, K - 1)}
        del s_logp, s_p
        torch.cuda.empty_cache()
        print(f'batch {b + 1}/{n_b} ({time.time() - t0:.0f}s)', flush=True)

    summary = {'teacher_tau': teacher_tau.round(4).tolist(), 'teacher_t': teacher_t.round(4).tolist(),
               'self_tau': self_tau.cpu().numpy().round(4).tolist(), 'self_t': self_t.cpu().numpy().round(4).tolist()}
    for ruler, taus in [('T', teacher_tau), ('S', self_tau.cpu().numpy())]:
        for metric in ['l2', 'kl']:
            m = R[f'{ruler}_{metric}'].mean(0)  # (nL, K)
            best = m.argmin(1)
            summary[f'{ruler}_{metric}_tau_star'] = taus[best].round(4).tolist()
            summary[f'{ruler}_{metric}_min'] = m.min(1).round(4).tolist()
            summary[f'{ruler}_{metric}_curve'] = m.round(4).tolist()
    summary['text'] = text
    summary['text_teacher_states'] = text_teacher
    np.savez_compressed(os.path.join(args.out, 'stats.npz'), **R)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for k in ['T_l2', 'T_kl', 'S_l2', 'S_kl']:
        print(k, 'tau*', summary[f'{k}_tau_star'])
        print(k, 'min ', summary[f'{k}_min'])
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
