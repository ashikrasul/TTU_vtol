"""
bo_plot_from_csv.py
===================
Reconstruct BO history from CSV using the EXACT same GP core as
bo_optimizer.py  (gpc_core: build_kernel, laplace, predict_2d).

With OPTIMISE_HP=False (your run's setting), ls and var are fixed
throughout — making this reconstruction fully deterministic.

CSV format:  scale, hsv_v, k, m   (one row per query, in order)

Usage: edit CONFIG below and run:
    python bo_plot_from_csv.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable   # ← added

# ── Exact same core as bo_optimizer.py — NO sklearn ──────────────────────────
from gpc_core import (
    build_kernel,
    laplace,
    predict_2d,
    acq_EI_pi,
    acq_UCB,
)


# ═══════════════════════════════════════════════════════════════════════════
# ██  CONFIG  — edit everything here, no CLI needed
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Input / output ────────────────────────────────────────────────
    'csv':      './outputs/bo_checkpoint_real_UCB_Hyp_False.csv',
    'out_dir':  './bo_plots',

    # ── Must match bo_optimizer.py settings exactly ───────────────────
    'acq':      'UCB',    # 'EI' or 'UCB'
    'n_init':   5,         # space-filling rows before BO started
    'kappa':    1.5,       # UCB kappa
    'ls':       0.20,      # GP length-scale  (fixed, OPTIMISE_HP=False)
    'var':      2.0,       # GP output variance
    'kernel':   'rbf',     # 'rbf' or 'matern52'

    # ── Snapshot steps (0-indexed into ACTIVE steps only, -1 = last)
    # With n_init=5 and budget=31:
    # snaps=[0, 12, -1] → Step1(n=5), Step13(n=17), Step25(n=29)
    # at active step s: n_obs = n_init + s
    #   s=0  → n=5,  s=12 → n=17,  s=24(last) → n=29
    'snaps':    [0, 12, -1],
}

# ═══════════════════════════════════════════════════════════════════════════


# ── Style constants ───────────────────────────────────────────────────────────
BG       = 'white'
PANEL    = 'white'
BORDER   = '#bbbbbb'
GREY     = '#444444'
GOLD     = '#c77c00'
LAVENDER = '#5544aa'
FONT     = 9


# ── Style helpers ─────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GREY, labelsize=FONT - 1)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])






def _label_contours(ax, cs):
    ax.clabel(cs, cs.levels, inline=True, fmt='%.1f',
              fontsize=FONT - 2, colors='white')

def _label_contours(ax, cs):
    # Place labels only on the longest segment of each contour level
    for collection, level in zip(cs.collections, cs.levels):
        paths = collection.get_paths()
        if not paths:
            continue
        longest = max(paths, key=lambda p: len(p.vertices))
        verts = longest.vertices
        mid   = verts[len(verts) // 2]          # midpoint of longest segment
        ax.clabel(cs, [level], inline=True, fmt='%.1f',
                  fontsize=FONT - 2, colors='white',
                  manual=[mid])



# ── Build history using exact gpc_core pipeline ───────────────────────────────

def build_history(df, acq_type='UCB', n_init=5, kappa=1.5,
                  ls=0.20, var=2.0, kernel='rbf'):
    """
    Reconstruct BO history matching bo_optimizer.py exactly.

    bo_optimizer stores history BEFORE querying x_new, so at active
    step 0: X_tr = init rows only (n_init points), n_obs = n_init.

    From the CSV (31 rows, n_init=5):
        Rows 0..4   = init points  (X_tr at step 0)
        Row  5      = first active query  → history[0] has X_tr=rows 0-4, x_new=row5
        Row  6      = second active query → history[1] has X_tr=rows 0-5, x_new=row6
        ...
    So n_obs at step s = n_init + s, matching original Step1(n=5), Step13(n=17).
    """
    g_fine   = np.linspace(0, 1, 60)
    G1f, G2f = np.meshgrid(g_fine, g_fine)
    X_plot   = np.column_stack([G1f.ravel(), G2f.ravel()])

    X_all = df[['scale', 'hsv_v']].values.astype(float)
    k_all = df['k'].values.astype(float)
    m_all = df['m'].values.astype(float)

    history = []

    # active_step s: X_tr = rows 0..n_init+s-1, x_new = row n_init+s
    n_active = len(df) - n_init
    for s in range(n_active):
        row_end = n_init + s          # X_tr uses rows 0..row_end-1
        X_tr = X_all[:row_end]
        k_tr = k_all[:row_end]
        m_tr = m_all[:row_end]
        x_new = X_all[row_end]        # the query being made at this step

        n = len(X_tr)                 # = n_init + s

        K_tr  = build_kernel(X_tr, X_tr, ls, var, kernel) + 1e-6 * np.eye(n)
        f_map, Sigma = laplace(K_tr, k_tr, m_tr)

        mu_plot, sig_plot, pi_plot = predict_2d(
            X_tr, f_map, Sigma, K_tr, X_plot, ls, var, kernel)

        pi_max = pi_plot.max()
        if acq_type == 'EI':
            acq_vals = acq_EI_pi(mu_plot, sig_plot, pi_max)
        else:
            acq_vals = acq_UCB(mu_plot, sig_plot, kappa)

        history.append({
            'g_fine':      g_fine,
            'X_plot':      X_plot,
            'X_tr':        X_tr.copy(),
            'k_tr':        k_tr.copy(),
            'm_tr':        m_tr.copy(),
            'pi_plot':     pi_plot,
            'acq_plot':    acq_vals,
            'pi_max':      float(pi_max),
            'x_new':       x_new.copy(),
            'n_obs':       n,          # = n_init + s, matches original
            'active_step': s,          # 0-indexed
            'phase':       'bo',
        })

        print(f"  Active step {s+1:>3}/{n_active}"
              f"  n={n}  pi_max={pi_max:.3f}")

    return history


# -- Figure 5-style contour snapshots -----------------------------------------

def plot_figure5(history, acq_type, save_path, snap_steps, kappa=1.5):
    snap_steps = [s if s >= 0 else len(history) + s for s in snap_steps]
    snap_steps = [min(s, len(history) - 1) for s in snap_steps]

    g_fine = history[0]['g_fine']
    nf     = len(g_fine)
    X_plot = history[0]['X_plot']

    pi_levels = np.linspace(0, 1, 101)
    pi_ticks  = np.round(np.linspace(0, 0.99, 10), 2)

    save_dir  = os.path.dirname(save_path)
    save_base = os.path.splitext(os.path.basename(save_path))[0]

    for col_idx, s in enumerate(snap_steps):
        is_final = (col_idx == len(snap_steps) - 1)
        h        = history[s]

        Pi_bar   = h['pi_plot'].reshape(nf, nf)
        Acq      = h['acq_plot'].reshape(nf, nf)
        X_tr     = h['X_tr']
        k_tr     = h['k_tr']
        m_tr     = h['m_tr']
        x_new    = h['x_new']
        n_obs    = h['n_obs']
        emp_rate = k_tr / np.maximum(m_tr, 1)
        step_title = f'Step {h["active_step"] + 1}  (n = {n_obs})'

        # ── π̄(x) panel ───────────────────────────────────────────────
        fig, ax_pi = plt.subplots(figsize=(3.25, 3.25))
        fig.patch.set_facecolor(BG)
        _style_ax(ax_pi)

        cf_pi = ax_pi.contourf(g_fine, g_fine, Pi_bar,
                               levels=pi_levels, cmap='viridis')

        contour_levels = [l for l in [0.3, 0.5, 0.7, 0.9]
                          if Pi_bar.min() < l < Pi_bar.max()]
        if contour_levels:
            if is_final:
                cs = ax_pi.contour(g_fine, g_fine, Pi_bar,
                                   levels=contour_levels,
                                   colors='white', linewidths=1.0, alpha=0.9)
                _label_contours(ax_pi, cs)
            else:
                ax_pi.contour(g_fine, g_fine, Pi_bar,
                              levels=contour_levels,
                              colors='black', linewidths=0.8, alpha=0.6)

        ax_pi.scatter(X_tr[:, 0], X_tr[:, 1],
                      c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                      s=15, zorder=7, edgecolors='black', linewidths=0.5)
        ax_pi.scatter([x_new[0]], [x_new[1]], s=20, marker='*',
                      color=LAVENDER, zorder=10,
                      edgecolors='black', linewidths=0.4)
        if h['phase'] == 'bo':
            xb = X_plot[h['pi_plot'].argmax()]
            ax_pi.scatter([xb[0]], [xb[1]], s=20, marker='D',
                          color=GOLD, zorder=9,
                          edgecolors='black', linewidths=0.5)



        ax_pi.set_title(step_title, color='black', fontsize=9, fontweight='bold')
        ax_pi.set_xlabel('scale  (x1)', color=GREY, fontsize=9)
        ax_pi.set_ylabel('hsv_v  (x2)', color=GREY, fontsize=9)

        legend_elements_pi = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor=LAVENDER,
                markeredgecolor='black', markersize=8, label='Next query  $x_{new}$'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=GOLD,
                markeredgecolor='black', markersize=6, label='Current best  $\\hat{x}$'),
        ]
        ax_pi.legend(handles=legend_elements_pi,
                    loc='upper center', bbox_to_anchor=(0.5, -0.18),
                    ncol=2, fontsize=7, facecolor='white', edgecolor=BORDER,
                    framealpha=0.85, labelcolor='black')




        # ── Colorbar: exact same height as axes ───────────────────────
        divider_pi = make_axes_locatable(ax_pi)
        cax_pi     = divider_pi.append_axes("right", size="5%", pad=0.08)
        cb_pi      = plt.colorbar(cf_pi, cax=cax_pi)
        cb_pi.set_ticks(pi_ticks)
        cb_pi.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=9)

        plt.tight_layout()
        out = os.path.join(save_dir,
                           f'{save_base}_step{h["active_step"]+1}_pi.png')
        plt.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Saved -> {out}')
        plt.close()

        # ── Acquisition panel ─────────────────────────────────────────
        fig, ax_acq = plt.subplots(figsize=(3.25, 3.25))
        fig.patch.set_facecolor(BG)
        _style_ax(ax_acq)

        acq_max = max(Acq.max(), 1e-9)
        cf_acq  = ax_acq.contourf(g_fine, g_fine, Acq,
                                  levels=np.linspace(0, acq_max, 101),
                                  cmap='plasma')
        if Acq.max() > 1e-6:
            ax_acq.contour(g_fine, g_fine, Acq,
                           levels=7, colors='black',
                           linewidths=0.4, alpha=0.35)

        ax_acq.scatter([x_new[0]], [x_new[1]], s=80, marker='*',
                       color=LAVENDER, zorder=10,
                       edgecolors='black', linewidths=0.4)
        ax_acq.scatter(X_tr[:, 0], X_tr[:, 1],
                       s=25, color='#888888', alpha=0.6, zorder=6,
                       edgecolors='black', linewidths=0.4)

        lbl = 'EI_pi  acquisition' if acq_type == 'EI' else 'UCB  acquisition'
        ax_acq.set_title(f'{step_title}  —  {lbl}', color='black', fontsize=9, fontweight='bold')
        ax_acq.set_xlabel('scale  (x1)', color=GREY, fontsize=9)
        ax_acq.set_ylabel('hsv_v  (x2)', color=GREY, fontsize=9)

        legend_elements_acq = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor=LAVENDER,
                markeredgecolor='black', markersize=8, label='Next query  $x_{new}$'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=GOLD,
                markeredgecolor='black', markersize=6, label='Current best  $\\hat{x}$'),
        ]
        ax_acq.legend(handles=legend_elements_acq,
                    loc='upper center', bbox_to_anchor=(0.5, -0.18),
                    ncol=2, fontsize=7, facecolor='white', edgecolor=BORDER,
                    framealpha=0.85, labelcolor='black')

        # ── Colorbar: exact same height as axes ───────────────────────
        divider_acq = make_axes_locatable(ax_acq)
        cax_acq     = divider_acq.append_axes("right", size="5%", pad=0.08)
        cb_acq      = plt.colorbar(cf_acq, cax=cax_acq)
        acq_ticks   = np.linspace(0, acq_max, 9)
        cb_acq.set_ticks(acq_ticks)
        cb_acq.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        cb_acq.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=9)

        plt.tight_layout()
        out = os.path.join(save_dir,
                           f'{save_base}_step{h["active_step"]+1}_acq.png')
        plt.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Saved -> {out}')
        plt.close()


# -- Convergence plot ----------------------------------------------------------

def plot_convergence(histories, save_path, kappa=1.5):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=GREY, labelsize=FONT)

    line_colors = {'EI': '#007a6e', 'UCB': '#5544aa'}
    line_styles = {'EI': '-',       'UCB': '--'}
    line_labels = {
        'EI':  'EI_pi  (Laplace)',
        'UCB': f'UCB  (kappa = {kappa})',
    }

    for acq_type, history in histories.items():
        pi_maxes = [h['pi_max'] for h in history]
        steps    = np.arange(1, len(pi_maxes) + 1)
        final    = pi_maxes[-1]
        ax.plot(steps, pi_maxes,
                color=line_colors.get(acq_type, 'grey'),
                lw=2.2,
                ls=line_styles.get(acq_type, '-'),
                label=f"{line_labels.get(acq_type, acq_type)}   "
                      f"(final pi_max = {final:.3f})")

    ax.set_xlabel('Active query step', color=GREY, fontsize=FONT + 1)
    ax.set_ylabel('pi_max  (model recommendation)', color=GREY, fontsize=FONT + 1)
    ax.set_title('Convergence of pi_max  vs  iteration',
                 color='black', fontsize=FONT + 2, fontweight='bold')
    ax.legend(facecolor='white', edgecolor=BORDER, labelcolor='black',
              fontsize=FONT, framealpha=1)
    ax.set_xlim(1, max(len(hist) for hist in histories.values()))
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#dddddd', linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  Saved -> {save_path}')
    plt.close()


# -- Entry point ---------------------------------------------------------------

def main():
    cfg = CONFIG

    if not os.path.isfile(cfg['csv']):
        print(f'ERROR: file not found: {cfg["csv"]}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(cfg['csv'])
    required = {'scale', 'hsv_v', 'k', 'm'}
    missing  = required - set(df.columns)
    if missing:
        print(f'ERROR: CSV missing columns: {missing}', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(df)} rows from {cfg["csv"]}')
    os.makedirs(cfg['out_dir'], exist_ok=True)

    print(f'\nReconstructing with gpc_core  '
          f'[acq={cfg["acq"]}  ls={cfg["ls"]}  var={cfg["var"]}  '
          f'kernel={cfg["kernel"]}  n_init={cfg["n_init"]}]\n')

    history = build_history(
        df,
        acq_type = cfg['acq'],
        n_init   = cfg['n_init'],
        kappa    = cfg['kappa'],
        ls       = cfg['ls'],
        var      = cfg['var'],
        kernel   = cfg['kernel'],
    )

    fig5_path = os.path.join(cfg['out_dir'], f'figure5_{cfg["acq"]}.png')
    plot_figure5(history, acq_type=cfg['acq'], save_path=fig5_path,
                 snap_steps=cfg['snaps'], kappa=cfg['kappa'])

    conv_path = os.path.join(cfg['out_dir'], f'convergence_{cfg["acq"]}.png')
    plot_convergence({cfg['acq']: history}, save_path=conv_path,
                     kappa=cfg['kappa'])

    print('\nDone.')


if __name__ == '__main__':
    main()