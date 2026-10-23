"""Figures and tables for the FMLM depth-as-time probes.

Reads results/onestep/stats.npz (probe_onestep.py) and
results/multistep/stats.npz (probe_multistep.py); writes PNGs and
summary.json into results/figs.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (dataviz skill, light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
GRAY = '#8a8984'
INK, INK2 = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'figure.dpi': 140, 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
    'grid.linewidth': 0.6, 'lines.linewidth': 2, 'lines.markersize': 4,
    'legend.frameon': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
})


def rho(d):
    """Eq. (4): per-trajectory normalisation by the max distance, then mean."""
    m = d.max(-1, keepdims=True)
    keep = m[..., 0] > 0
    return (d[keep] / m[keep]).mean(0)


def mean_ci(x):
    return x.mean(0), 1.96 * x.std(0) / np.sqrt(x.shape[0])


def fit_mixing(G, r, yy, base, train_frac=0.7):
    """Shared-coefficient least squares (Appendix B.2) from per-sample Gram stats.

    G (N, D, F, F), r (N, D, F), yy/base (N, D). Returns beta (D, F) fitted on
    the first train_frac samples and the held-out R_i (D,).
    """
    n = int(train_frac * G.shape[0])
    beta = np.stack([np.linalg.lstsq(G[:n, i].sum(0), r[:n, i].sum(0), rcond=None)[0]
                     for i in range(G.shape[1])])
    Gh, rh = G[n:], r[n:]
    resid = (yy[n:] - 2 * np.einsum('ndf,df->nd', rh, beta)
             + np.einsum('df,ndfg,dg->nd', beta, Gh, beta))
    R = resid.sum(0) / np.maximum(base[n:].sum(0), 1e-12)
    return beta, R


def match_curve(student_vals, teacher_vals, teacher_x):
    """Map each student value onto the teacher axis by monotone interpolation
    of the (mean) teacher curve; values outside the teacher range are clipped."""
    order = np.argsort(teacher_vals)
    return np.interp(student_vals, teacher_vals[order], teacher_x[order])


def onestep_figures(d, q, out):
    nL = d['s_ent'].shape[1]
    depth = np.arange(nL)
    K = d['t_ent'].shape[1]
    tau = d['t_tau']
    cross = d['cross_steps']
    summary = {}

    # ---- Fig 1: normalised remaining distance rho (paper Figs. 1/2/8) ----
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (name, sk, tk) in zip(axes, [
            ('token disagreement', 1 - d['s_agree'], 1 - d['t_agree']),
            ('KL(final || readout)', d['s_kl'], d['t_kl']),
            ('L2 (probabilities)', d['s_l2'], d['t_l2'])]):
        ax.plot(depth / (nL - 1), rho(sk), '-o', color=BLUE, label='FMLM depth (i/L)')
        ax.plot(np.arange(K) / (K - 1), rho(tk), '-', color=ORANGE,
                label=f'FLM sampling time (k/K, K={K})')
        if name.startswith('token'):
            ax.plot(depth / (nL - 1), rho(1 - d['c_ridge_agree']), '--', color=AQUA,
                    label='trained ridge probe')
            ax.plot(depth / (nL - 1), rho(1 - d['c_rand_agree']), ':', color=GRAY,
                    label='random probe')
        if name.startswith('KL'):
            ax.plot(depth / (nL - 1), rho(d['c_ridge_kl']), '--', color=AQUA,
                    label='trained ridge probe')
            ax.plot(depth / (nL - 1), rho(d['c_rand_kl']), ':', color=GRAY,
                    label='random probe')
        ax.set_title(name)
        ax.set_xlabel('normalized progress')
    axes[0].set_ylabel(r'normalized remaining $\rho$')
    axes[0].legend(loc='lower left', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig1_rho.png'))
    plt.close(fig)

    # ---- Fig 2: what the readout looks like across depth ----------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    m, c = mean_ci(d['s_ent'])
    ax.errorbar(depth, m, c, fmt='-o', color=BLUE, label='FMLM layer readout')
    ax.axhline(d['t_ent'][:, 0].mean(), color=GRAY, ls=':', lw=1.5,
               label='FLM D at t=0 (marginal)')
    ax.axhline(d['t_ent'][:, -1].mean(), color=ORANGE, ls='--', lw=1.5,
               label='FLM D at last step')
    ax.set_title('per-token entropy [nats]')
    ax.set_xlabel('depth i')
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(depth, d['s_agree'].mean(0), '-o', color=BLUE, label='= own final token')
    ax.plot(depth, d['s_agree_teacher'].mean(0), '-o', color=ORANGE,
            label='= FLM 128-step sample (same z)')
    ax.plot(depth, d['s_agree_marg'].mean(0), '-o', color=GRAY, label='= argmax of marginal')
    ax.plot(depth, d['s_agree_noise'].mean(0), '-o', color=MAGENTA, label='= argmax z (raw noise)')
    ax.set_title('argmax agreement')
    ax.set_xlabel('depth i')
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(depth, d['s_kl_marg'].mean(0), '-o', color=BLUE, label='FMLM layer')
    ax.set_title('KL(readout || marginal)')
    ax.set_xlabel('depth i')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig2_readout_phases.png'))
    plt.close(fig)

    # ---- Fig 3: student layer vs teacher time (same z) -------------------
    xkl = d['x_kl'].mean(0)  # (nL, nC) KL(D^k || delta^i)
    xag = d['x_agree'].mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, mat, title, cmap in [(axes[0], xkl, r'KL($D_{t_k}$ || $\delta^i$)', 'Blues_r'),
                                 (axes[1], xag, 'argmax agreement', 'Blues')]:
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap,
                       extent=[tau[cross[0]], tau[cross[-1]], -0.5, nL - 0.5])
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax.set_xlabel(r'teacher sampling time $\tau_k$')
        ax.set_ylabel('FMLM depth i')
        ax.set_title(title)
        ax.grid(False)
    best = tau[cross[xkl.argmin(1)]]
    axes[0].plot(best, depth, 'o', color=ORANGE, ms=5, label='argmin over k')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3_cross_heatmap.png'))
    plt.close(fig)

    # ---- Fig 4: effective (teacher) time per layer -----------------------
    t_ent, t_agree, t_kl = d['t_ent'].mean(0), d['t_agree'].mean(0), d['t_kl'].mean(0)
    eff = {
        'cross-KL argmin': best,
        'entropy match': match_curve(d['s_ent'].mean(0), t_ent, tau),
        'progress match (agreement w/ own final)': match_curve(d['s_agree'].mean(0), t_agree, tau),
        'progress match (KL to own final)': match_curve(d['s_kl'].mean(0), t_kl, tau),
    }
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for (name, v), col in zip(eff.items(), [ORANGE, MAGENTA, BLUE, AQUA]):
        ax.plot(depth, v, '-o', color=col, label=name)
    ax.plot(depth, depth / (nL - 1), ':', color=GRAY, lw=1.5, label='i/L')
    ax.set_xlabel('FMLM depth i')
    ax.set_ylabel(r'effective teacher time $\tau^*(i)$')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig4_effective_time.png'))
    plt.close(fig)
    summary['effective_tau'] = {k: np.round(v, 3).tolist() for k, v in eff.items()}

    # ---- Fig 5: depthwise mixing coefficients (Eq. 5) ---------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    bp, Rp = fit_mixing(d['p_G'], d['p_r'], d['p_yy'], d['p_base'])
    bl, Rl = fit_mixing(d['l_G'], d['l_r'], d['l_yy'], d['l_base'])
    bt, Rt = fit_mixing(d['tp_G'], d['tp_r'], d['tp_yy'], d['tp_base'])
    ax = axes[0]
    for j, (lab, col) in enumerate([(r'$\alpha$ (final $\delta^L$)', BLUE),
                                    (r'$\mu$ (marginal)', GRAY), (r'$\sigma$ (noise z)', MAGENTA)]):
        ax.plot(depth, bp[:, j], '-o', color=col, label=lab)
    ax.set_title(r'prob. space: $\delta^i \approx \alpha\delta^L + \mu m + \sigma z$')
    ax.set_xlabel('depth i')
    ax.legend(fontsize=7)
    ax = axes[1]
    for j, (lab, col) in enumerate([(r'$\alpha$ (final log $\delta^L$)', BLUE),
                                    (r'$\sigma$ (noise z)', MAGENTA)]):
        ax.plot(depth, bl[:, j], '-o', color=col, label=lab)
    ax.set_title(r'log-prob space: $\log\delta^i \approx \alpha\log\delta^L + \sigma z + c$')
    ax.set_xlabel('depth i')
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(depth[:-1], Rp[:-1], '-o', color=BLUE, label='FMLM depth, prob. space')
    ax.plot(depth[:-1], Rl[:-1], '-o', color=AQUA, label='FMLM depth, log-prob space')
    ax.plot((tau[cross][:-1]) * (nL - 1), Rt[:-1], '-', color=ORANGE,
            label=r'FLM time (x = $\tau_k \cdot L$), prob. space')
    ax.set_title('held-out residual $R_i$ (1 = no better than final)')
    ax.set_xlabel('depth i')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig5_mixing.png'))
    plt.close(fig)
    summary['mixing_prob'] = {'alpha_mu_sigma': np.round(bp, 4).tolist(), 'R': np.round(Rp, 4).tolist()}
    summary['mixing_logp'] = {'alpha_sigma_c': np.round(bl, 4).tolist(), 'R': np.round(Rl, 4).tolist()}
    summary['mixing_teacher_prob'] = {'alpha_mu_sigma': np.round(bt, 4).tolist(),
                                      'R': np.round(Rt, 4).tolist()}

    # ---- Fig 6: nested depth lens of the FLM teacher per NFE (Sec. 4.1) ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    cols = [GRAY, BLUE, AQUA, YELLOW, MAGENTA, ORANGE]
    for ni, k in enumerate(d['nested_steps']):
        lab = rf'$\tau$={tau[k]:.2f}'
        axes[0].plot(depth, rho(d['n_kl'][:, ni]), '-o', color=cols[ni % len(cols)], label=lab)
        axes[1].plot(depth, d['n_agree_final'][:, ni].mean(0), '-o', color=cols[ni % len(cols)], label=lab)
    axes[0].set_title(r'FLM single NFE: $\rho$ (KL to that NFE output)')
    axes[1].set_title('FLM single NFE: agreement with final sample')
    for ax in axes:
        ax.set_xlabel('depth i')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig6_teacher_nested.png'))
    plt.close(fig)

    # ---- Fig 7: sample quality across depth / time ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    sl = q['student_layer']
    tl = q['teacher_step']
    axes[0].semilogy([r['layer'] / (nL - 1) for r in sl], [r['gen_ppl'] for r in sl],
                     '-o', color=BLUE, label='FMLM layer argmax')
    axes[0].semilogy([r['tau'] for r in tl], [r['gen_ppl'] for r in tl], '-o',
                     color=ORANGE, label=r'FLM $D_{t_k}$ argmax')
    axes[0].set_title('Gen. PPL (GPT-2 Large)')
    axes[1].plot([r['layer'] / (nL - 1) for r in sl], [r['entropy'] for r in sl], '-o', color=BLUE)
    axes[1].plot([r['tau'] for r in tl], [r['entropy'] for r in tl], '-o', color=ORANGE)
    axes[1].axhline(4.31, color=GRAY, ls=':', lw=1.5, label='LM1B data (4.31)')
    axes[1].set_title('unigram entropy of decoded text')
    for ax in axes:
        ax.set_xlabel(r'normalized progress (i/L or $\tau_k$)')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig7_quality.png'))
    plt.close(fig)

    summary['student'] = {k: np.round(d[k].mean(0), 4).tolist() for k in [
        's_ent', 's_maxp', 's_agree', 's_kl', 's_l2', 's_kl_marg', 's_nll_final',
        's_agree_teacher', 's_agree_noise', 's_agree_marg', 'c_ridge_agree', 'c_rand_agree']}
    summary['student']['rho_disagree'] = np.round(rho(1 - d['s_agree']), 4).tolist()
    summary['student']['rho_kl'] = np.round(rho(d['s_kl']), 4).tolist()
    summary['student']['rho_kl_ridge'] = np.round(rho(d['c_ridge_kl']), 4).tolist()
    summary['student']['rho_kl_rand'] = np.round(rho(d['c_rand_kl']), 4).tolist()
    idx = [0, 1, 2, 4, 8, 16, 32, 64, 96, K - 1]
    summary['teacher_at_steps'] = {
        'step': idx, 'tau': np.round(tau[idx], 3).tolist(), 't': np.round(d['t_t'][idx], 3).tolist(),
        **{k: np.round(d[k].mean(0)[idx], 4).tolist() for k in [
            't_ent', 't_agree', 't_kl', 't_kl_marg', 't_state_agree']}}
    summary['student_teacher_final_agreement'] = float(d['st_final_agree'].mean())
    summary['nested_rho_kl'] = {f'tau={tau[k]:.3f}': np.round(rho(d['n_kl'][:, ni]), 3).tolist()
                                for ni, k in enumerate(d['nested_steps'])}
    return summary


def multistep_figures(m, out):
    summary = {}
    keys = sorted({k.rsplit('/', 1)[0] for k in m.files if k.startswith('sweep/')})
    starts = sorted({k.split('/')[1] for k in keys})
    fig, axes = plt.subplots(len(starts), 3, figsize=(12, 3.1 * len(starts)), squeeze=False)
    cmap = plt.get_cmap('Blues')
    for si, s in enumerate(starts):
        ks = [k for k in keys if k.split('/')[1] == s]
        for j, k in enumerate(ks):
            tt = float(k.split('/t')[-1])
            col = ORANGE if tt == 1.0 else cmap(0.35 + 0.6 * j / max(len(ks) - 1, 1))
            lab = rf'$\tau_t$={tt:.1f}'
            ent = m[f'{k}/ent'].mean(0)
            depth = np.arange(len(ent))
            axes[si, 0].plot(depth, ent, '-o', color=col, label=lab, ms=3)
            axes[si, 1].plot(depth, m[f'{k}/nll_clean'].mean(0), '-o', color=col, ms=3)
            axes[si, 2].plot(depth, m[f'{k}/kl_clean'].mean(0), '-o', color=col, ms=3)
            dip = ent.argmin()
            summary[k] = {'ent': np.round(ent, 3).tolist(),
                          'nll_clean': np.round(m[f'{k}/nll_clean'].mean(0), 3).tolist(),
                          'kl_clean': np.round(m[f'{k}/kl_clean'].mean(0), 3).tolist(),
                          'argmin_entropy_layer': int(dip),
                          'entropy_overshoot': float(ent[-1] - ent.min())}
        axes[si, 0].set_title(rf'start $\tau_s$={s[1:]}: entropy of $\delta^i_{{s,t}}$')
        axes[si, 1].set_title(r'$-\log\delta^i[y^{clean}]$, $y^{clean}$=argmax $\delta_{s,1}$')
        axes[si, 2].set_title(r'KL($\delta_{s,1}$ || $\delta^i_{s,t}$)')
        axes[si, 0].legend(fontsize=6, ncol=2)
        for ax in axes[si]:
            ax.set_xlabel('depth i')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig8_endpoint_sweep.png'))
    plt.close(fig)

    rolls = sorted({k.split('/')[1] for k in m.files if k.startswith('rollout/')},
                   key=lambda s: int(s[1:]))
    fig, axes = plt.subplots(len(rolls), 3, figsize=(12, 3.0 * len(rolls)), squeeze=False)
    for ri, r in enumerate(rolls):
        n = int(r[1:])
        sched = m[f'rollout/{r}/sched']
        for j in range(n):
            k = f'rollout/{r}/step{j}'
            ent = m[f'{k}/ent'].mean(0)
            L1 = len(ent) - 1
            x = j + np.arange(L1 + 1) / L1
            lab = f'NFE {j + 1}' if j < 1 else None
            axes[ri, 0].plot(x, ent, '-', color=BLUE, label=lab)
            axes[ri, 1].plot(x, m[f'{k}/nll_clean'].mean(0), '-', color=BLUE)
            axes[ri, 2].plot(x, rho(m[f'{k}/kl_final']), '-', color=BLUE)
            axes[ri, 2].plot(x, rho(m[f'{k}/kl_jump']), '--', color=ORANGE,
                             label=r'to $\delta_{s,1}(x_s)$' if j == 0 else None)
            summary[k] = {'tau_s': float(sched[j][0]), 'tau_target': float(sched[j][1]),
                          'ent': np.round(ent, 3).tolist(),
                          'nll_clean': np.round(m[f'{k}/nll_clean'].mean(0), 3).tolist(),
                          'agree_clean': np.round(m[f'{k}/agree_clean'].mean(0), 3).tolist(),
                          'rho_kl_final': np.round(rho(m[f'{k}/kl_final']), 3).tolist()}
        for ax in axes[ri]:
            for j in range(1, n):
                ax.axvline(j, color=GRAY, lw=0.8, ls=':')
            ax.set_xlabel('NFE + depth fraction')
        axes[ri, 0].set_title(f'{n}-step FMLM: entropy of layer readout')
        axes[ri, 1].set_title(r'$-\log\delta^i[y]$, y = final sample')
        axes[ri, 2].set_title(r'$\rho$: to this NFE output (blue), to clean jump (orange)')
        axes[ri, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig9_rollouts.png'))
    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.path.join(os.path.dirname(__file__), 'results'))
    args = p.parse_args()
    out = os.path.join(args.root, 'figs')
    os.makedirs(out, exist_ok=True)
    summary = {}
    one = os.path.join(args.root, 'onestep')
    if os.path.exists(os.path.join(one, 'stats.npz')):
        d = np.load(os.path.join(one, 'stats.npz'))
        q = json.load(open(os.path.join(one, 'quality.json')))
        summary['onestep'] = onestep_figures(d, q, out)
        summary['onestep']['quality'] = q
    multi = os.path.join(args.root, 'multistep')
    if os.path.exists(os.path.join(multi, 'stats.npz')):
        summary['multistep'] = multistep_figures(np.load(os.path.join(multi, 'stats.npz')), out)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
