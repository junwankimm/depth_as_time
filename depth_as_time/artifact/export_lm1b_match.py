"""Add the LM1B state-matching results (probe_statematch.py) to data.json as D.match."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
s = json.load(open(os.path.join(HERE, '..', 'results', 'statematch', 'summary.json')))
D = json.load(open(os.path.join(HERE, 'data.json')))

# thin the 127-point teacher grid to ~32 points for the page
tt = np.array(s['teacher_tau'])
idx = sorted(set(np.linspace(0, len(tt) - 1, 32).round().astype(int).tolist()))
T_curve = np.array(s['T_l2_curve'])[:, idx]
D['match'] = {
    'self_tau': [round(v, 4) for v in s['self_tau']],
    'self_curve': s['S_l2_curve'],
    'self_star': [round(v, 3) for v in s['S_l2_tau_star']],
    'self_min': s['S_l2_min'],
    'teacher_tau': tt[idx].round(4).tolist(),
    'teacher_curve': T_curve.round(3).tolist(),
    'teacher_star': [round(v, 3) for v in s['T_l2_tau_star']],
    'teacher_min': s['T_l2_min'],
    'text': [[{'layer': r['layer'], 'tau': r['tau'], 'layer_state': r['layer_state'], 'self_state': r['self_state']}
              for r in rows] for rows in s['text']],
}
blob = json.dumps(D, separators=(',', ':'))
assert '</' not in blob
open(os.path.join(HERE, 'data.json'), 'w').write(blob)
print('match added;', len(blob), 'bytes')
