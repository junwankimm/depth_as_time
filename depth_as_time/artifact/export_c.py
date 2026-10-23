"""Add direction C (trajectory supervision: toy and LM1B) to data.json as D.trajsup."""
import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results')
D = json.load(open(os.path.join(HERE, 'data.json')))


def r3(x):
    return np.round(np.asarray(x, dtype=float), 3).tolist()


toy = collections.defaultdict(list)
for f in glob.glob(os.path.join(R, 'toy_trajsup', '*.json')):
    r = json.load(open(f))
    if r['args']['n_blocks'] == 8:
        toy[(r['config'], r['lam'])].append(r)
T = {}
for (cfg, lam), rs in toy.items():
    g = lambda k: np.array([x['after'][k] for x in rs])  # noqa: E731
    T.setdefault(cfg, {})[f'{lam:g}'] = {
        'n': len(rs), 'endpoint': [round(float(g('endpoint_agree_exact').mean()), 3), round(float(g('endpoint_agree_exact').std()), 3)],
        'valid': r3(g('valid_by_block').mean(0)), 'kl_sched': r3(g('kl_to_schedule').mean(0))}
    T[cfg]['base_valid'] = r3(np.mean([x['before']['valid_by_block'] for x in rs], 0))
    T[cfg]['base_kl_sched'] = r3(np.mean([x['before']['kl_to_schedule'] for x in rs], 0))
L = {}
for d in glob.glob(os.path.join(R, 'trajsup', 'fmlm-b-lm1b_*')):
    p = os.path.join(d, 'eval.json')
    if not os.path.exists(p) or d.endswith('_ctrl'):
        continue
    e = json.load(open(p))
    L[os.path.basename(d).replace('fmlm-b-lm1b_', '')] = {
        'gen_ppl': [round(float(np.mean(e['endpoint']['gen_ppl'])), 1), round(float(np.std(e['endpoint']['gen_ppl'])), 1)],
        'entropy': [round(float(np.mean(e['endpoint']['entropy'])), 3), round(float(np.std(e['endpoint']['entropy'])), 3)],
        'agree_original': round(float(np.mean(e['endpoint']['agree_original'])), 3),
        'kl_sched': r3(e['kl_to_schedule']), 'taus': r3(e['taus']),
        'ee_ppl': r3(e['early_exit']['gen_ppl']), 'ee_ent': r3(e['early_exit']['entropy']),
        'ee_agree': r3(e['early_exit']['agree_own_final'])}
D['trajsup'] = {'toy': T, 'lm1b': L}
blob = json.dumps(D, separators=(',', ':'))
assert '</' not in blob
open(os.path.join(HERE, 'data.json'), 'w').write(blob)
print('toy', sorted(T), 'lm1b', sorted(L))
