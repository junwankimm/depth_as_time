"""Export toy probe results (results/toy/*) into artifact/toy.json for the page."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'toy'))
from toy_plots import embed, mode_of, rho, state_match  # noqa: E402

ROOT = os.path.join(HERE, '..', 'results', 'toy')
N_SHOW, N_PATHS = 300, 60


def r(a, n=3):
    return np.round(np.asarray(a, dtype=float), n).tolist()


out = {}
for ds in ['cities', 'ring8']:
    out[ds] = {}
    for tag in ['exact', 'flm']:
        f = os.path.join(ROOT, ds, f'probe_{tag}.npz')
        if not os.path.exists(f):
            continue
        npz = np.load(f)
        S = json.load(open(os.path.join(ROOT, ds, f'summary_{tag}.json')))
        net = f'fmlm_{tag}'
        P = npz[f'{net}/probs'][:N_SHOW]
        nL = P.shape[1]
        tau = npz['exact_tau_sub']
        D = npz['exact_D'][:N_SHOW]
        y = P[:, -1].argmax(-1)
        k_match = [int(round(i / (nL - 1) * (len(tau) - 1))) for i in range(nL)]
        Dfull = npz['exact_D']
        d_ex = np.sqrt(((Dfull - Dfull[:, -1:]) ** 2).sum((-1, -2)))
        rex = rho(d_ex)
        sweep = {}
        z = npz['z'][:N_SHOW]
        for tt, d in S[net]['sweep'].items():
            t_q = 1.0 if float(tt) >= 1 else float(np.interp(float(tt), npz['tau_grid'], npz['t_grid']))
            Pq = npz[f'{net}/sweep/{tt}/probs'][:N_SHOW]
            Tq = npz[f'{net}/sweep/{tt}/exact_delta'][:N_SHOW]
            state_q = embed((1 - t_q) * z[:, None] + t_q * Pq, ds).transpose(1, 0, 2)
            state_t = embed((1 - t_q) * z + t_q * Tq, ds)
            Pt = npz[f'{net}/sweep/{tt}/probs'][:N_PATHS]
            T = npz[f'{net}/sweep/{tt}/exact_delta'][:N_PATHS]
            sweep[tt] = {'paths': r(embed(Pt, ds), 3), 'target': r(embed(T, ds), 3),
                         'ent': r(d['ent']), 'exact_ent': round(d['exact_ent'], 3),
                         'kl_clean': r(d['kl_clean']), 'kl_target': r(d['kl_to_exact_delta']),
                         't': round(t_q, 3), 'state': r(state_q, 2), 'state_target': r(state_t, 2)}
        best, sm_t, sm_tau, sm_err, sm_X = state_match(npz, net)
        zz = npz['z'][:N_SHOW]
        match = {'tau': r(sm_tau[best]), 't': r(sm_t[best]), 'err': r(sm_err / sm_err.max()),
                 'layer_state': [r(embed((1 - sm_t[k]) * zz + sm_t[k] * P[:, i], ds), 2) for i, k in enumerate(best)],
                 'exact_state': [r(embed(sm_X[:N_SHOW, k], ds), 2) for k in best]}
        out[ds][tag] = {
            'match': match,
            'mode': mode_of(y, ds).tolist(),
            'student': r(embed(P, ds).transpose(1, 0, 2), 3),   # (nL, N, 2)
            'exact': r(embed(D[:, k_match], ds).transpose(1, 0, 2), 3),
            'state': r(embed(npz['exact_x'][:N_SHOW][:, k_match], ds).transpose(1, 0, 2), 3),
            'exact_tau': r(tau[k_match]),
            'valid': r(S[net]['valid_by_layer']), 'agree': r(S[net]['agree_final_by_layer']),
            'ent': r(S[net]['ent_by_layer']),
            'rho_depth': r(rho(npz[f'{net}/l2_to_final'].T)),
            'rho_exact': r(rex[::max(1, len(rex) // 100)]),
            'endpoint_agree': round(S[net]['agree_exact_endpoint'], 3),
            'sweep': sweep,
        }
with open(os.path.join(HERE, 'toy.json'), 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(os.path.getsize(os.path.join(HERE, 'toy.json')), {k: list(v) for k, v in out.items()})
