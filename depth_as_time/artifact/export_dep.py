"""Add the token-dependency, token-class and multi-step commit results to data.json (D.dep)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results', 'dependency')
D = json.load(open(os.path.join(HERE, 'data.json')))
out = {}
for ds in ['lm1b', 'owt']:
    s = json.load(open(os.path.join(R, ds, 'summary.json')))
    d = s['dependency']
    out[ds] = {
        'ctx_spearman': round(d['spearman_commit_vs_context_dependence'], 3),
        'self_spearman': round(d['spearman_commit_vs_own_noise_dependence'], 3),
        'pair_spearman': {k: round(v['spearman_dep_vs_neg_gap'], 3) for k, v in d['by_distance'].items()},
        'direction': [[b, e, l, r] for b, e, l, r in d['direction']['by_distance']],
        'classes': {k: {'share': round(v['share'], 3), 'mean': round(v['mean_commit'], 2), 'hist': v['hist']}
                    for k, v in s['token_class'].items()},
        'multistep': [{'step': r['step'], 'spearman': round(r['spearman_commit_vs_conf'], 3),
                       'block0': r['hist'][0], 'hist': r['hist']} for r in s['multistep']],
    }
D['dep'] = out
blob = json.dumps(D, separators=(',', ':'))
assert '</' not in blob
open(os.path.join(HERE, 'data.json'), 'w').write(blob)
print({k: list(v['classes']) for k, v in out.items()})
