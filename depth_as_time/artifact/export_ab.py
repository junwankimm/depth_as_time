"""Add direction A (per-token commit, D.commit), direction B (toy sweep, D.sweepB) and the
full-protocol reuse curve (D.reuseCurve) to data.json."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results')
D = json.load(open(os.path.join(HERE, 'data.json')))


def r3(x):
    return np.round(np.asarray(x, dtype=float), 3).tolist()


# ---- A: per-token commit depth ----
commit = {}
for ds in ['lm1b', 'owt']:
    s = json.load(open(os.path.join(R, 'commit', ds, 'summary.json')))
    A = np.load(os.path.join(R, 'commit', ds, 'stats.npz'))
    c, seeded = A['commit'], A['seeded']
    nL = len(s['commit_hist'])
    commit[ds] = {
        'hist': r3(s['commit_hist']),
        'hist_seeded': r3(np.bincount(c[seeded], minlength=nL)[:nL] / max(1, seeded.sum())),
        'hist_rest': r3(np.bincount(c[~seeded], minlength=nL)[:nL] / max(1, (~seeded).sum())),
        'by_conf_decile': r3(s['commit_by_conf_decile']),
        'spearman': {k.split(' ')[0]: round(v, 3) for k, v in s['spearman_commit_vs'].items()},
        'pre': {k: round(v, 3) for k, v in s['pre_commit_kind'].items()},
        'switches': r3(s['switches_hist']),
        'std_within': round(s['commit_std_within_sample'], 2), 'std_all': round(s['commit_std_all'], 2),
        'frac_seeded': round(s['frac_seeded'], 3),
        'mean_seeded': round(s['commit_mean_by_seeded']['seeded'], 2),
        'mean_rest': round(s['commit_mean_by_seeded']['not_seeded'], 2),
    }
D['commit'] = commit

# ---- B: toy sweep ----
runs = {}
for f in glob.glob(os.path.join(R, 'toy_sweep', '*.json')):
    r = json.load(open(f))
    a = r['args']
    runs.setdefault((r['config'], a['n_blocks'], a['hidden']), []).append(r)
order = ['dep_c1', 'dep_c2', 'dep_c4', 'dep_c8', 'modes_m2', 'modes_m16', 'modes_m64', 'modes_m256']
sweep = {'b8': {}, 'b16': {}}
for cfg in order:
    for nb, key in [(8, 'b8'), (16, 'b16')]:
        rs = runs.get((cfg, nb, 128))
        if not rs:
            continue
        g = lambda k: np.array([x[k] for x in rs])  # noqa: E731
        sweep[key][cfg] = {
            'n': len(rs), 'c': rs[0]['c'], 'm': rs[0]['m'], 'M': rs[0]['M'],
            'tau_star': r3(np.mean([x['tau_star'] for x in rs], 0)),
            'tau_star_std': r3(np.std([x['tau_star'] for x in rs], 0)),
            'tau0': [round(float(g('tau_star')[:, 0].mean()), 3), round(float(g('tau_star')[:, 0].std()), 3)],
            'jump': [round(float(g('tau_star_max_jump').mean()), 3), round(float(g('tau_star_max_jump').std()), 3)],
            'within': [round(float(g('commit_std_within').mean()), 3), round(float(g('commit_std_within').std()), 3)],
            'endpoint': round(float(g('endpoint_agree_exact').mean()), 3),
            'agree': r3(np.mean([x['agree_by_layer'] for x in rs], 0)),
        }
D['sweepB'] = sweep

# ---- reuse curve from the full-protocol evaluation ----
ev = None
for f in ['lm1b_c.json', 'lm1b.json']:
    p = os.path.join(R, 'compress_eval', f)
    if os.path.exists(p):
        ev = json.load(open(p))
        break
curve = []
if ev:
    for name, v in ev['models'].items():
        if name == 'teacher':
            continue
        e = json.load(open(os.path.join(R, 'compress', name, 'eval.json')))
        groups = e['args'].get('groups') if e['args']['variant'] == 'tied_struct' else None
        curve.append({'name': name, 'variant': e['args']['variant'], 'groups': groups,
                      'n_blocks': len(e['init']), 'K': e['args']['K'] or 12,
                      'params_m': round(e['n_trainable'] / 1e6, 1),
                      'agree': [round(float(np.mean(v['agree'])), 3), round(float(np.std(v['agree'])), 3)],
                      'kl': [round(float(np.mean(v['kl'])), 3), round(float(np.std(v['kl'])), 3)],
                      'gen_ppl': [round(float(np.mean(v['gen_ppl'])), 1), round(float(np.std(v['gen_ppl'])), 1)],
                      'entropy': [round(float(np.mean(v['entropy'])), 3), round(float(np.std(v['entropy'])), 3)]})
    t = ev['models']['teacher']
    D['reuseTeacher'] = {'gen_ppl': [round(float(np.mean(t['gen_ppl'])), 1), round(float(np.std(t['gen_ppl'])), 1)],
                         'entropy': [round(float(np.mean(t['entropy'])), 3), round(float(np.std(t['entropy'])), 3)]}
D['reuseCurve'] = curve
po = os.path.join(R, 'compress_eval', 'owt.json')
if os.path.exists(po):
    eo = json.load(open(po))['models']
    ms = lambda v: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]  # noqa: E731
    D['reuseOwt'] = {k: {kk: ms(vv) for kk, vv in v.items() if isinstance(vv, list)} for k, v in eo.items()}
blob = json.dumps(D, separators=(',', ':'))
assert '</' not in blob
open(os.path.join(HERE, 'data.json'), 'w').write(blob)
print('commit', list(commit), 'sweep b8', list(sweep['b8']), 'b16', list(sweep['b16']), 'curve', len(curve))
