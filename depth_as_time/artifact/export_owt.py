"""Add the OWT replication (D.owt) and layer-reuse results (D.reuse) to data.json."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results')
D = json.load(open(os.path.join(HERE, 'data.json')))


def r3(x):
    return np.round(np.asarray(x, dtype=float), 3).tolist()


def model_stats(root):
    d = np.load(os.path.join(root, 'onestep', 'stats.npz'))
    eff = json.load(open(os.path.join(root, 'figs', 'summary.json')))['onestep']['effective_tau']
    sm = json.load(open(os.path.join(root, 'statematch', 'summary.json')))
    ms = json.load(open(os.path.join(root, 'multistep', 'summary.json')))
    q = json.load(open(os.path.join(root, 'onestep', 'quality.json')))
    return {
        'agree': r3(d['s_agree'].mean(0)), 'ent': r3(d['s_ent'].mean(0)),
        'rand_agree': r3(d['c_rand_agree'].mean(0)),
        'eff': r3(eff['progress match (agreement w/ own final)']),
        'self_star': r3(sm['S_l2_tau_star']), 'teacher_star': r3(sm['T_l2_tau_star']),
        'nll_t01': r3(ms['sweep/s0.00/t0.10/nll_clean']), 'ent_t01': r3(ms['sweep/s0.00/t0.10/ent']),
        'st_agree': round(float(d['st_final_agree'].mean()), 3),
        'gen_ppl': round(q['student_layer'][-1]['gen_ppl'], 1),
        'gen_ppl_teacher': round(q['teacher_final']['gen_ppl'], 1),
    }


D['owt'] = {'lm1b': model_stats(R), 'owt': model_stats(os.path.join(R, 'owt'))}
for k, root in [('lm1b', R), ('owt', os.path.join(R, 'owt'))]:
    D['owt'][k]['seed2'] = json.load(open(os.path.join(root, 'controls', 'summary.json')))['seed_student'][8]

swap = {ds: json.load(open(os.path.join(R, 'blocks', ds, 'summary.json')))['swap_agree']
        for ds in ['lm1b', 'owt', 'cities']}
runs = []
for f in sorted(glob.glob(os.path.join(R, 'compress', '*', 'eval.json'))):
    e = json.load(open(f))
    log = {r['step']: r for r in e['log']}
    step = max(log)
    r = log[step]
    ds, rest = e['name'].split('_', 1)
    runs.append({'ds': ds, 'name': rest.rsplit('_K', 1)[0], 'K': int(rest.rsplit('_K', 1)[1]), 'step': step,
                 'blocks_m': round(e['n_trainable'] / 1e6, 1), 'init': e['init'],
                 'agree': round(r['agree_teacher'], 3), 'kl': round(r['kl_teacher'], 2),
                 'gen_ppl': round(r['gen_ppl'], 1), 'entropy': round(r['entropy'], 3),
                 'curve': [[s, round(log[s]['agree_teacher'], 3)] for s in sorted(log)],
                 'teacher_ppl': round(r['gen_ppl_teacher'], 1), 'teacher_entropy': round(r['entropy_teacher'], 3)})
toy = []
for f in sorted(glob.glob(os.path.join(R, 'compress_toy', '*', 'eval.json'))):
    e = json.load(open(f))
    r = e['log'][-1]
    toy.append({'name': e['name'].split('_', 1)[1], 'params_m': round(e['n_student'] / 1e6, 2),
                'agree': round(r['agree_teacher'], 3), 'kl': r['kl_teacher'], 'valid': round(r['valid'], 3)})
D['reuse'] = {'swap': {k: r3(v) for k, v in swap.items()}, 'lm1b': runs, 'toy': toy}
blob = json.dumps(D, separators=(',', ':'))
assert '</' not in blob
open(os.path.join(HERE, 'data.json'), 'w').write(blob)
print(json.dumps(D['owt'])[:800])
print(json.dumps(runs)[:1200])
print(toy)
