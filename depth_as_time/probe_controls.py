"""Two controls for the one-step FMLM depth lens.

(a) Tuned lens: a separate affine probe per depth i, fit by ridge regression
    from LayerNorm(h_i) to the model's final logits on held-in noise and
    evaluated on held-out noise. If the W_out readout's mid-depth "marginal
    plateau" is only a basis mismatch between h_i and W_out, the tuned lens
    will not show it. (The depth-as-time paper's trained probe is fit on the
    final layer only; that one is ~identical to W_out here because the head is
    an affine map of LayerNorm(h) for fixed conditioning.)
(b) Noise seeding: for the linear interpolant x_t = (1-t) z + t e_y, the
    per-token posterior at small t is p(v | x_t) ~ m(v) exp(t x_t[v] / (1-t)^2),
    i.e. a noise-perturbed marginal. We test how well a purely tokenwise,
    context-free rule argmax_v [log m_l(v) + beta z_l(v)] predicts the final
    token, compared with the depth-0 readout (which is also context-free).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import lens
from probe_onestep import agree, entropy, kl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-LM1B')
    p.add_argument('--teacher', default='FLM-B-LM1B')
    p.add_argument('--fit_batches', type=int, default=8)
    p.add_argument('--eval_batches', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--betas', default='0,0.02,0.05,0.1,0.2,0.3,0.5,1,2,5')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default='results/controls')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    student = lens.load_backbone(args.student, device)
    teacher = lens.load_backbone(args.teacher, device)
    V, Lseq = student.vocab_size, student.config.model.length
    nL = len(student.blocks) + 1
    betas = [float(b) for b in args.betas.split(',')]
    lut = lens.TimeLUT(V)

    # ---------------- (a) fit per-layer ridge probes ----------------------
    D = student.config.model.hidden_size + 1
    XtX = torch.zeros(nL, D, D, device=device, dtype=torch.float64)
    XtY = torch.zeros(nL, D, V, device=device)
    for _ in range(args.fit_batches):
        z = torch.randn((args.batch_size, Lseq, V), device=device)
        hs, c = lens.hidden_states(student, z, 0.0, 1.0)
        Y = lens.readout_logits(student, hs[-1], c).reshape(-1, V)
        for i, h in enumerate(hs):
            X = F.layer_norm(h.float(), [h.shape[-1]]).reshape(-1, D - 1)
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=device)], -1)
            XtX[i] += (X.T @ X).double()
            XtY[i] += X.T @ Y
    W = []
    for i in range(nL):
        lam = 1e-3 * XtX[i].trace() / D
        W.append(torch.linalg.solve(XtX[i] + lam * torch.eye(D, device=device, dtype=torch.float64),
                                    XtY[i].double()).float())
    del XtX, XtY

    stats = {k: [] for k in ['tl_agree', 'tl_kl', 'tl_ent', 'wo_agree', 'wo_ent',
                             'seed_student', 'seed_teacher', 'l0_student', 'l0_teacher',
                             'marg_student']}
    for _ in range(args.eval_batches):
        z = torch.randn((args.batch_size, Lseq, V), device=device)
        hs, c = lens.hidden_states(student, z, 0.0, 1.0)
        final = lens.readout(student, hs[-1], c)
        y_s = final.argmax(-1)
        row = {k: [] for k in ['tl_agree', 'tl_kl', 'tl_ent', 'wo_agree', 'wo_ent']}
        for i, h in enumerate(hs):
            X = F.layer_norm(h.float(), [h.shape[-1]])
            X = torch.cat([X, torch.ones(*X.shape[:-1], 1, device=device)], -1)
            lp = lens._logprobs(X @ W[i])
            row['tl_agree'].append(agree(lp.argmax(-1), y_s))
            row['tl_kl'].append(kl(final, lp))
            row['tl_ent'].append(entropy(lp))
            wo = lens.readout(student, h, c)
            row['wo_agree'].append(agree(wo.argmax(-1), y_s))
            row['wo_ent'].append(entropy(wo))
        for k, v in row.items():
            stats[k].append(torch.stack(v, -1).cpu().numpy())

        # ---------------- (b) noise seeding ------------------------------
        y_t = lens.flm_trajectory(teacher, lut, z, 128)
        log_m = lens.model_logprobs(teacher, z, 0.0)  # D_0(z): per-position marginal
        l0 = lens.readout(student, hs[0], c).argmax(-1)
        stats['seed_student'].append(np.stack([
            agree((log_m + b * z).argmax(-1), y_s).cpu().numpy() for b in betas], -1))
        stats['seed_teacher'].append(np.stack([
            agree((log_m + b * z).argmax(-1), y_t).cpu().numpy() for b in betas], -1))
        stats['l0_student'].append(agree(l0, y_s).cpu().numpy())
        stats['l0_teacher'].append(agree(l0, y_t).cpu().numpy())
        stats['marg_student'].append(agree(log_m.argmax(-1), y_s).cpu().numpy())
        del hs, z
        torch.cuda.empty_cache()

    out = {k: np.concatenate(v, 0) for k, v in stats.items()}
    np.savez_compressed(os.path.join(args.out, 'stats.npz'), betas=np.array(betas), **out)
    summary = {k: np.round(v.mean(0), 4).tolist() for k, v in out.items()}
    summary['betas'] = betas
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for k, v in summary.items():
        print(k, v)


if __name__ == '__main__':
    main()
