"""Depth-as-time Fig. 1 style plots for the toy FMLM (see toy.py).

2-D embedding of a per-token distribution pair (p_1, p_2):
  cities: (p_1[new] - p_1[san], p_2[york] - p_2[diego]); valid sentences sit at
          (+1, +1) "new york" and (-1, -1) "san diego", factorization errors at
          (+1, -1) "new diego" and (-1, +1) "san york"; the marginal is (0, 0).
  ring8:  barycentric embedding of p_1 on a regular octagon (mode k at angle
          2 pi k / 8); the marginal is the centre.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
GRAY, INK2 = '#8a8984', '#52514e'
MODE_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
plt.rcParams.update({
    'figure.dpi': 140, 'font.size': 8, 'axes.edgecolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'lines.linewidth': 2,
    'legend.frameon': False,
})


def embed(P, dataset):
    """P (..., L=2, V) -> (..., 2)."""
    if dataset == 'cities':
        return np.stack([P[..., 0, 0] - P[..., 0, 1], P[..., 1, 2] - P[..., 1, 3]], -1)
    ang = 2 * np.pi * np.arange(P.shape[-1]) / P.shape[-1]
    U = np.stack([np.cos(ang), np.sin(ang)], -1)
    return P[..., 0, :] @ U


def frame(ax, dataset, lim=1.25):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if dataset == 'cities':
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        for (x, y, lab, ok) in [(1, 1, 'new york', True), (-1, -1, 'san diego', True),
                                (1, -1, 'new diego', False), (-1, 1, 'san york', False)]:
            ax.plot([x], [y], marker='s' if ok else 'x', ms=6, color=INK2 if ok else RED,
                    mew=1.5, fillstyle='none', zorder=1)
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color='#e6e5e0', lw=0.8, zorder=0)
    else:
        ang = 2 * np.pi * np.arange(9) / 8
        ax.plot(np.cos(ang), np.sin(ang), color='#e6e5e0', lw=0.8, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.plot([0], [0], marker='+', color=GRAY, ms=6, zorder=1)


def mode_of(y, dataset):
    """Final-mode id of each sample from its final tokens y (N, 2)."""
    if dataset == 'cities':
        return (y[:, 0] == 1).astype(int)  # 0: new..., 1: san...
    return y[:, 0]


def rho(d):
    m = d.max(-1, keepdims=True)
    keep = m[..., 0] > 0
    return (d[keep] / m[keep]).mean(0)


def fig1(npz, S, dataset, net, out, n_show=400):
    P = npz[f'{net}/probs'][:n_show]  # (N, nL, 2, V)
    nL = P.shape[1]
    D = npz['exact_D'][:n_show]  # (N, K', 2, V)
    tau = npz['exact_tau_sub']
    y = P[:, -1].argmax(-1)
    col = np.array(MODE_COLORS)[mode_of(y, dataset)]
    E = embed(P, dataset)
    ED = embed(D, dataset)
    EX = embed(npz['exact_x'][:n_show], dataset)  # noisy states x_t (FLM Fig. 3 view)
    # state-space readout of the noisy-endpoint query 0 -> tau_t (paper's x~^i_r):
    # X^i_{0,t} = (1 - t) z + t delta^i_{0,t}; reference X*_{0,t} = (1 - t) z + t delta*_{0,t}
    tq = '0.50'
    t_q = float(np.interp(float(tq), npz['tau_grid'], npz['t_grid']))
    z = npz['z'][:n_show]
    XS = embed((1 - t_q) * z[:, None] + t_q * npz[f'{net}/sweep/{tq}/probs'][:n_show], dataset)
    XT = embed((1 - t_q) * z + t_q * npz[f'{net}/sweep/{tq}/exact_delta'][:n_show], dataset)
    fig, axes = plt.subplots(4, nL, figsize=(1.55 * nL, 7.2))
    for i in range(nL):
        ax = axes[0, i]
        frame(ax, dataset)
        ax.scatter(E[:, i, 0], E[:, i, 1], s=2.5, c=col, alpha=0.6, lw=0)
        ax.set_title(f'layer {i}  (valid {S[net]["valid_by_layer"][i]:.2f})', fontsize=7)
        k = int(round(i / (nL - 1) * (len(tau) - 1)))  # matched normalized progress
        ax = axes[1, i]
        frame(ax, dataset)
        ax.scatter(ED[:, k, 0], ED[:, k, 1], s=2.5, c=col, alpha=0.6, lw=0)
        ax.set_title(rf'exact E[x$_1$|x$_t$], $\tau$={tau[k]:.2f}', fontsize=7)
        ax = axes[2, i]
        frame(ax, dataset, lim=2.6)
        ax.scatter(EX[:, k, 0], EX[:, k, 1], s=2.5, c=col, alpha=0.6, lw=0)
        ax.set_title(rf'exact state x$_t$, $\tau$={tau[k]:.2f}', fontsize=7)
        ax = axes[3, i]
        frame(ax, dataset, lim=2.6)
        ax.scatter(XT[:, 0], XT[:, 1], s=5, facecolors='none', edgecolors=GRAY, lw=0.4, alpha=0.6)
        ax.scatter(XS[:, i, 0], XS[:, i, 1], s=2.5, c=col, alpha=0.7, lw=0)
        ax.set_title(rf'X$^{{{i}}}_{{0,t}}$, $\tau_t$={tq}', fontsize=7)
    fig.suptitle(f'Toy "{dataset}" ({net}): row 1 = one-step FMLM decoded at every layer; '
                 f'row 2 = exact posterior mean, row 3 = exact noisy state (FLM Fig. 3 view), '
                 f'both along the exact flow at matched progress;\nrow 4 = state-space readout '
                 f'(1-t) z + t delta^i of the query 0 -> tau_t={tq} (gray rings: exact state X*_{{0,t}}) '
                 f'(endpoint agreement {S[net]["agree_exact_endpoint"]:.2f})', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f'toy_{dataset}_{net}_fig1.png'))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax = axes[0]
    l2 = npz[f'{net}/l2_to_final'].T  # (N, nL)
    ax.plot(np.arange(nL) / (nL - 1), rho(l2), '-o', color=BLUE, ms=3, label='FMLM depth (i/L)')
    Dfull = npz['exact_D']
    d_ex = np.sqrt(((Dfull - Dfull[:, -1:]) ** 2).sum((-1, -2)))
    ax.plot(np.linspace(0, 1, d_ex.shape[1]), rho(d_ex), color=ORANGE, label='exact flow (k/K)')
    ax.set_xlabel('normalized progress')
    ax.set_ylabel(r'normalized remaining $\rho$ (L2)')
    ax.legend(fontsize=7)
    ax.grid(color='#e6e5e0', lw=0.6)
    ax = axes[1]
    ax.plot(np.arange(nL), S[net]['valid_by_layer'], '-s', color=AQUA, ms=3, label='valid sentence (argmax)')
    ax.plot(np.arange(nL), S[net]['agree_final_by_layer'], '-o', color=BLUE, ms=3, label='= final tokens')
    ax.set_xlabel('layer i')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(color='#e6e5e0', lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'toy_{dataset}_{net}_curves.png'))
    plt.close(fig)


def fig_renoise(npz, S, dataset, net, out, n_show=300):
    taus = sorted(S[net]['sweep'])
    nL = npz[f'{net}/probs'].shape[1]
    fig, axes = plt.subplots(1, len(taus) + 1, figsize=(3.0 * (len(taus) + 1), 3.2))
    for j, tt in enumerate(taus):
        P = npz[f'{net}/sweep/{tt}/probs'][:n_show]
        T = npz[f'{net}/sweep/{tt}/exact_delta'][:n_show]
        E, ET = embed(P, dataset), embed(T, dataset)
        y = npz[f'{net}/probs'][:n_show, -1].argmax(-1)
        col = np.array(MODE_COLORS)[mode_of(y, dataset)]
        ax = axes[j]
        frame(ax, dataset)
        for n in range(0, n_show, 6):
            ax.plot(E[n, :, 0], E[n, :, 1], '-', color=col[n], lw=0.6, alpha=0.5)
        ax.scatter(E[:, -1, 0], E[:, -1, 1], s=5, c=col, lw=0, label='layer L output')
        ax.scatter(ET[:, 0], ET[:, 1], s=14, facecolors='none', edgecolors=INK2, lw=0.6,
                   label=r'exact $\delta^*_{0,t}$')
        ax.set_title(rf'query (0 $\to$ $\tau_t$={tt}): paths over layers', fontsize=8)
        if j == 0:
            ax.legend(fontsize=6, loc='lower right')
    ax = axes[-1]
    cmap = plt.get_cmap('Blues')
    for j, tt in enumerate(taus):
        d = S[net]['sweep'][tt]
        c = ORANGE if tt == '1.00' else cmap(0.4 + 0.5 * j / max(1, len(taus) - 1))
        ax.plot(np.arange(nL), d['ent'], '-o', ms=3, color=c, label=rf'$\tau_t$={tt}')
        ax.axhline(d['exact_ent'], color=c, ls=':', lw=1)
    ax.set_xlabel('layer i')
    ax.set_title('entropy of readout (dotted: exact target)', fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(color='#e6e5e0', lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'toy_{dataset}_{net}_renoise.png'))
    plt.close(fig)


def state_match(npz, net):
    """Map every layer onto the exact flow in state space.

    Along the exact flow x_t = X_{0,t}(z) = (1 - t) z + t delta*_{0,t}(z), so the
    layer-i readout placed at time t, (1 - t) z + t delta^i, equals x_t iff
    delta^i = delta*_{0,t}. Distances are taken between delta's (the state
    distance carries a factor t that would always favour t = 0).
    Returns per-layer matched index into the subsampled grid, the grid t/tau,
    and the mean L2 mismatch.
    """
    z, X = npz['z'], npz['exact_x']
    t = npz['t_grid'][::10]
    tau = npz['exact_tau_sub']
    keep = t > 1e-3
    t, tau, X = t[keep], tau[keep], X[:, keep]
    dstar = z[:, None] + (X - z[:, None]) / t[None, :, None, None]
    P = npz[f'{net}/probs']
    l2 = np.sqrt(((P[:, :, None] - dstar[:, None]) ** 2).sum((-1, -2))).mean(0)  # (nL, K)
    best = l2.argmin(1)
    return best, t, tau, l2.min(1), X


def fig_state_match(npz, S, dataset, net, out, n_show=400):
    best, t, tau, err, X = state_match(npz, net)
    P = npz[f'{net}/probs'][:n_show]
    z = npz['z'][:n_show]
    nL = P.shape[1]
    y = P[:, -1].argmax(-1)
    col = np.array(MODE_COLORS)[mode_of(y, dataset)]
    fig, axes = plt.subplots(2, nL + 1, figsize=(1.55 * (nL + 1), 3.9),
                             gridspec_kw={'width_ratios': [1] * nL + [1.6]})
    for i in range(nL):
        k = best[i]
        Si = embed((1 - t[k]) * z + t[k] * P[:, i], dataset)
        Xi = embed(X[:n_show, k], dataset)
        for row, pts in [(0, Si), (1, Xi)]:
            ax = axes[row, i]
            frame(ax, dataset, lim=2.6)
            ax.scatter(pts[:, 0], pts[:, 1], s=2.5, c=col, alpha=0.6, lw=0)
        axes[0, i].set_title(f'layer {i} at t={t[k]:.2f}', fontsize=7)
        axes[1, i].set_title(rf'exact x$_t$, $\tau$={tau[k]:.2f}', fontsize=7)
    gs = axes[0, nL].get_gridspec()
    for ax in axes[:, nL]:
        ax.remove()
    ax = fig.add_subplot(gs[:, nL])
    ax.plot(np.arange(nL), tau[best], '-o', color=BLUE, ms=3, label=r'matched $\tau^*(i)$')
    ax.plot(np.arange(nL), err / err.max(), '--', color=GRAY, lw=1.5, label='mismatch (rel.)')
    ax.set_xlabel('layer i')
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7)
    ax.grid(color='#e6e5e0', lw=0.6)
    fig.suptitle(f'Toy "{dataset}" ({net}): layer readout placed in state space at its best-matching '
                 f'time t*(i) (top) vs the exact flow state at that time (bottom)', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f'toy_{dataset}_{net}_statematch.png'))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.path.join(os.path.dirname(__file__), '..', 'results', 'toy'))
    args = p.parse_args()
    for dataset in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, dataset)
        for tag in ['exact', 'flm']:
            f = os.path.join(d, f'probe_{tag}.npz')
            if not os.path.exists(f):
                continue
            npz = np.load(f)
            S = json.load(open(os.path.join(d, f'summary_{tag}.json')))
            fig1(npz, S, dataset, f'fmlm_{tag}', d)
            fig_renoise(npz, S, dataset, f'fmlm_{tag}', d)
            fig_state_match(npz, S, dataset, f'fmlm_{tag}', d)
            print('wrote', dataset, tag)


if __name__ == '__main__':
    main()
