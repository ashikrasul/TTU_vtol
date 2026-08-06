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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable   # ← added

# ── Exact same core as bo_optimizer.py — NO sklearn ──────────────────────────
from utils.gpc_core import (
    build_kernel,
    laplace,
    predict_2d,
    acq_EI_pi,
    acq_UCB,
)
from utils.bo_plotting import plot_acquisition_evolution


# ═══════════════════════════════════════════════════════════════════════════
# ██  CONFIG  — edit everything here, no CLI needed
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Input / output ────────────────────────────────────────────────
    'csv':      './outputs/bo_checkpoint_real_UCB_Hyp_False_run36.csv',
    'out_dir':  './bo_plots',

    # ── Must match bo_optimizer.py settings exactly ───────────────────
    'acq':       'UCB',   # 'EI' or 'UCB'
    'n_init':    5,        # space-filling rows before BO started
    'kappa':     1.5,      # UCB kappa
    'ls':        0.20,     # GP length-scale  (fixed, OPTIMISE_HP=False)
    'var':       2.0,      # GP output variance
    'kernel':    'rbf',    # 'rbf' or 'matern52'
    'm_trials':  10,       # trials per query (m)
    'budget':    30,       # total queries including init
    'use_dummy': False,    # True = synthetic oracle

    # ── Snapshot steps (0-indexed into ACTIVE steps only, -1 = last)
    # With n_init=5 and budget=31:
    # snaps=[0, 12, -1] → Step1(n=5), Step13(n=17), Step25(n=29)
    # at active step s: n_obs = n_init + s
    #   s=0  → n=5,  s=12 → n=17,  s=24(last) → n=29
    'snaps':    [0, 5, -1],
}

# ═══════════════════════════════════════════════════════════════════════════


# ── Style constants ───────────────────────────────────────────────────────────
BG       = 'white'
PANEL    = 'white'
BORDER   = '#bbbbbb'
GREY     = '#444444'
GOLD     = '#c77c00'
LAVENDER = '#5544aa'
BEST_SO_FAR = '#cc3366'
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
                  ls=0.20, var=2.0, kernel='rbf', budget=None):
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
    g_fine   = np.linspace(0, 1, 40)
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
        pi_argmax  = pi_plot.argmax()
        mu_at_max  = float(mu_plot[pi_argmax])
        sig_at_max = float(sig_plot[pi_argmax])
        x1_at_max  = float(X_plot[pi_argmax, 0])
        x2_at_max  = float(X_plot[pi_argmax, 1])
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
            'mu_plot':     mu_plot,
            'sig_plot':    sig_plot,
            'pi_plot':     pi_plot,
            'acq_plot':    acq_vals,
            'pi_max':      float(pi_max),
            'x_new':       x_new.copy(),
            'n_obs':       n,          # = n_init + s, matches original
            'active_step': s,          # 0-indexed
            'phase':       'bo',
        })

        print(f"  Active step {s+1:>3}/{n_active}"
              f"  n={n}  pi_max={pi_max:.3f}"
              f"  mu={mu_at_max:.3f}  sigma={sig_at_max:.3f}"
              f"  x1={x1_at_max:.3f}  x2={x2_at_max:.3f}")

    return history


# -- Figure 5-style contour snapshots -----------------------------------------

def plot_figure5(history, acq_type, save_path, snap_steps, kappa=1.5,
                 m_trials=15, budget=30, real_blackbox=True):
    import matplotlib.cm as cm
    snap_steps = [s if s >= 0 else len(history) + s for s in snap_steps]
    snap_steps = [min(s, len(history) - 1) for s in snap_steps]
    n_cols = len(snap_steps)

    g_fine    = history[0]['g_fine']
    nf        = len(g_fine)
    X_plot    = history[0]['X_plot']
    viridis   = cm.get_cmap('viridis')
    lbl_acq   = r'EI$_\pi$  acquisition' if acq_type == 'EI' else 'UCB  acquisition'
    bb_tag    = '[real blackbox]' if real_blackbox else '[synthetic oracle]'
    kappa_str = f'\\kappa={kappa}'

    pi_levels = np.linspace(0, 1, 101)
    pi_ticks  = np.round(np.linspace(0, 0.99, 10), 2)

    # ── legend items matching reference ──────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=GOLD,
               markeredgecolor='black', markersize=6,
               label=r'$\hat{p}_{\max}$  (model recommendation)'),
        # Line2D([0], [0], marker='s', color='w', markerfacecolor=BEST_SO_FAR,
        #        markeredgecolor='black', markersize=5,
        #        label=r'Best observed $(k/m)$'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=LAVENDER,
               markeredgecolor='black', markersize=6,
               label='Next query point'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cm.get_cmap('RdYlBu')(1.0), markeredgecolor='black',
               markersize=5, label=r'High success rate  $(k/m \to 1)$'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cm.get_cmap('RdYlBu')(0.15), markeredgecolor='black',
               markersize=5, label=r'Low success rate  $(k/m \to 0)$'),
    ]

    # ── figure: 2 rows × n_cols, extra left margin for row labels ────────────
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(3.8 * n_cols + 0.4, 8.0),
                             constrained_layout=False)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88,
                        bottom=0.16, wspace=0.45, hspace=0.38)

    # ── figure title matching reference style ─────────────────────────────────
    # title = (f'2D Binomial-Probit BO  |  {acq_type} $({kappa_str})$  |  '
    #          f'$m$ = {m_trials} trials/query  |  budget = {budget}  |  {bb_tag}')
    # fig.suptitle(title, color='black', fontsize=10, fontweight='bold', y=0.97)

    # ── row labels via fig.text (outside subplots, centred vertically) ────────
    row_mid_top = (fig.subplotpars.top + 0.5 * (fig.subplotpars.top - fig.subplotpars.bottom
                                                 + fig.subplotpars.hspace * 0.5)) / 2 + 0.02
    fig.text(0.01, 0.73, r'Approx  $\bar{p}(x)$',
             va='center', ha='center', rotation=90,
             fontsize=10, color='black')
    fig.text(0.01, 0.31, lbl_acq,
             va='center', ha='center', rotation=90,
             fontsize=10, color='black')

    for col_idx, s in enumerate(snap_steps):
        # is_final   = (col_idx == n_cols - 1)
        is_final  = True 
        h          = history[s]
        Pi_bar     = h['pi_plot'].reshape(nf, nf)
        Acq        = h['acq_plot'].reshape(nf, nf)
        X_tr       = h['X_tr']
        k_tr       = h['k_tr']
        m_tr       = h['m_tr']
        x_new      = h['x_new']
        n_obs      = h['n_obs']
        emp_rate   = k_tr / np.maximum(m_tr, 1)
        step_title = f'Step {h["active_step"] + 1}  (n = {n_obs})'

        # ── row 0 : GP posterior ──────────────────────────────────────────────
        ax_pi = axes[0, col_idx]
        _style_ax(ax_pi)

        cf_pi = ax_pi.contourf(g_fine, g_fine, Pi_bar,
                               levels=pi_levels, cmap='viridis')

        # white contour lines on ALL posterior panels; labels only on final
        contour_levels = [l for l in [0.3, 0.5, 0.7, 0.8, 0.9]
                          if Pi_bar.min() < l < Pi_bar.max()]
        if contour_levels:
            cs = ax_pi.contour(g_fine, g_fine, Pi_bar,
                               levels=contour_levels,
                               colors='white', linewidths=0.9, alpha=0.85)
            if is_final:
                                _label_contours(ax_pi, cs)

        ax_pi.scatter(X_tr[:, 0], X_tr[:, 1],
                      c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                      s=28, zorder=7, edgecolors='black', linewidths=0.5)
        ax_pi.scatter([x_new[0]], [x_new[1]], s=80, marker='*',
                      color=LAVENDER, zorder=10,
                      edgecolors='black', linewidths=0.4)
        if h['phase'] == 'bo':
            xb = X_plot[h['pi_plot'].argmax()]
            ax_pi.scatter([xb[0]], [xb[1]], s=45, marker='D',
                          color=GOLD, zorder=9,
                          edgecolors='black', linewidths=0.5)

            # best_idx = np.argmax(k_tr / m_tr)
            # ax_pi.scatter([X_tr[best_idx, 0]], [X_tr[best_idx, 1]], s=45,
            #               marker='s', color=BEST_SO_FAR, zorder=9,
            #               edgecolors='black', linewidths=0.5)

        ax_pi.text(0.5, 1.14, step_title, transform=ax_pi.transAxes,
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='black', clip_on=False)
        ax_pi.text(0.5, 1.02, r'$\bar{p}(x^*) = $' + f'{h["pi_max"]:.2f}',
                   transform=ax_pi.transAxes,
                   ha='center', va='bottom', fontsize=9, color='black', clip_on=False)
        ax_pi.set_xlabel(r'scale  $(x_1)$', color=GREY, fontsize=8)
        ax_pi.set_ylabel(r'brightness  $(x_2)$', color=GREY, fontsize=8)

        divider = make_axes_locatable(ax_pi)
        cax     = divider.append_axes("right", size="5%", pad=0.06)
        cb      = plt.colorbar(cf_pi, cax=cax)
        cb.set_ticks(pi_ticks)
        cb.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)

        # ── row 1 : acquisition ───────────────────────────────────────────────
        ax_acq = axes[1, col_idx]
        _style_ax(ax_acq)

        acq_max = max(Acq.max(), 1e-9)
        cf_acq  = ax_acq.contourf(g_fine, g_fine, Acq,
                                  levels=np.linspace(0, acq_max, 101),
                                  cmap='plasma')
        if Acq.max() > 1e-6:
            ax_acq.contour(g_fine, g_fine, Acq,
                           levels=7, colors='black',
                           linewidths=0.4, alpha=0.35)

        ax_acq.scatter([x_new[0]], [x_new[1]], s=200, marker='*',
                       color=LAVENDER, zorder=10,
                       edgecolors='black', linewidths=0.5)
        ax_acq.scatter(X_tr[:, 0], X_tr[:, 1],
                       s=40, color='#888888', alpha=0.7, zorder=6,
                       edgecolors='black', linewidths=0.4)

        ax_acq.set_title(step_title, color='black', fontsize=9, fontweight='bold')
        ax_acq.set_xlabel(r'scale  $(x_1)$', color=GREY, fontsize=8)
        ax_acq.set_ylabel(r'brightness  $(x_2)$', color=GREY, fontsize=8)

        divider_acq = make_axes_locatable(ax_acq)
        cax_acq     = divider_acq.append_axes("right", size="5%", pad=0.06)
        cb_acq      = plt.colorbar(cf_acq, cax=cax_acq)
        n_ticks     = 9
        acq_ticks   = np.linspace(0, acq_max, n_ticks)
        cb_acq.set_ticks(acq_ticks)
        cb_acq.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        cb_acq.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)

    # ── shared legend at the bottom (4 items, single row) ────────────────────
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=4, fontsize=FONT + 1,
               facecolor='white', edgecolor=BORDER,
               framealpha=0.95, labelcolor='black',
               bbox_to_anchor=(0.5, -0.04))

    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved -> {save_path}')
    plt.close()


# -- Figure 5 + sigma row (pi / sigma / acq) -----------------------------------

def plot_figure5_sigma(history, acq_type, save_path, snap_steps, kappa=1.5,
                       m_trials=15, budget=30, real_blackbox=True):
    import matplotlib.cm as cm
    from scipy.stats import norm
    snap_steps = [s if s >= 0 else len(history) + s for s in snap_steps]
    snap_steps = [min(s, len(history) - 1) for s in snap_steps]
    n_cols = len(snap_steps)

    g_fine   = history[0]['g_fine']
    nf       = len(g_fine)
    X_plot   = history[0]['X_plot']
    lbl_acq  = r'EI$_\pi$  acquisition' if acq_type == 'EI' else 'UCB  acquisition'

    pi_levels = np.linspace(0, 1, 101)
    pi_ticks  = np.round(np.linspace(0, 0.99, 10), 2)

    # fixed sigma scale across all panels so reduction is visually comparable
    sig_global_max = max(
        history[s]['sig_plot'].max() for s in snap_steps
    )
    sig_levels        = np.linspace(0, sig_global_max, 101)
    sig_ticks         = np.linspace(0, sig_global_max, 6)
    sig_contour_lines = [0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

    acq_global_max = max(
        history[s]['acq_plot'].max() for s in snap_steps
    )
    acq_global_max = max(acq_global_max, 1e-9)
    acq_levels     = np.linspace(0, acq_global_max, 101)
    acq_ticks      = np.linspace(0, acq_global_max, 9)

    viridis = cm.get_cmap('viridis')
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=GOLD,
               markeredgecolor='black', markersize=6,
               label=r'$\hat{p}_{\max}$  (model recommendation)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=LAVENDER,
               markeredgecolor='black', markersize=6,
               label='Next query point'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=viridis(1.0), markeredgecolor='black',
               markersize=5, label=r'High trial outcome  $(k/m \to 1)$'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=viridis(0.0), markeredgecolor='black',
               markersize=5, label=r'Low trial outcome  $(k/m \to 0)$'),
    ]

    fig, axes = plt.subplots(3, n_cols,
                             figsize=(3.8 * n_cols + 0.4, 11.5),
                             constrained_layout=False)
    if n_cols == 1:
        axes = axes.reshape(3, 1)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93,
                        bottom=0.10, wspace=0.45, hspace=0.42)

    fig.text(0.01, 0.78, r'Approx  $\bar{p}(x)$',
             va='center', ha='center', rotation=90,
             fontsize=10, color='black')
    fig.text(0.01, 0.50, r'Uncertainty  $\sigma(x)$',
             va='center', ha='center', rotation=90,
             fontsize=10, color='black')
    fig.text(0.01, 0.22, lbl_acq,
             va='center', ha='center', rotation=90,
             fontsize=10, color='black')

    for col_idx, s in enumerate(snap_steps):
        is_final   = True
        h          = history[s]
        Pi_bar     = h['pi_plot'].reshape(nf, nf)
        Sig        = h['sig_plot'].reshape(nf, nf)
        Acq        = h['acq_plot'].reshape(nf, nf)
        X_tr       = h['X_tr']
        k_tr       = h['k_tr']
        m_tr       = h['m_tr']
        x_new      = h['x_new']
        n_obs      = h['n_obs']
        emp_rate   = k_tr / np.maximum(m_tr, 1)
        step_title = f'Step {h["active_step"] + 1}  (n = {n_obs})'

        # precompute pi_max location and sigma there — used in all rows
        xb         = X_plot[h['pi_plot'].argmax()]
        pi_argmax  = h['pi_plot'].argmax()
        sig_at_xb  = float(h['sig_plot'][pi_argmax])
        mu_at_xb   = float(h['mu_plot'][pi_argmax])
        ci_lo      = float(norm.cdf(mu_at_xb - 1.96 * sig_at_xb))
        ci_hi      = float(norm.cdf(mu_at_xb + 1.96 * sig_at_xb))
        # flip annotation offset away from edges to avoid clipping
        ann_dx = -30 if xb[0] > 0.75 else 6
        ann_dy = -12 if xb[1] > 0.75 else 6

        # ── row 0 : π̄(x) posterior (viridis background) ─────────────────────
        ax_pi = axes[0, col_idx]
        _style_ax(ax_pi)
        cf_pi = ax_pi.contourf(g_fine, g_fine, Pi_bar,
                               levels=pi_levels, cmap='viridis')
        contour_levels = [l for l in [0.3, 0.5, 0.7, 0.8, 0.9]
                          if Pi_bar.min() < l < Pi_bar.max()]
        if contour_levels:
            cs = ax_pi.contour(g_fine, g_fine, Pi_bar,
                               levels=contour_levels,
                               colors='white', linewidths=0.5)
            if is_final:
                ax_pi.clabel(cs, cs.levels, inline=True, fmt='%.1f',
                             fontsize=FONT - 2, colors='white')
        ax_pi.scatter(X_tr[:, 0], X_tr[:, 1],
                      c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                      s=28, zorder=7, edgecolors='black', linewidths=0.5)
        ax_pi.scatter([x_new[0]], [x_new[1]], s=80, marker='*',
                      color=LAVENDER, zorder=10, edgecolors='black', linewidths=0.4)
        if h['phase'] == 'bo':
            ax_pi.scatter([xb[0]], [xb[1]], s=45, marker='D',
                          color=GOLD, zorder=9, edgecolors='black', linewidths=0.5)
        ax_pi.text(0.5, 1.14, step_title, transform=ax_pi.transAxes,
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='black', clip_on=False)
        ax_pi.text(0.5, 1.02,
                   r'$\bar{p}(x^*) = $' + f'{h["pi_max"]:.2f}'
                   + f'  [{ci_lo:.2f}, {ci_hi:.2f}]',
                   transform=ax_pi.transAxes,
                   ha='center', va='bottom', fontsize=9, color='black', clip_on=False)
        ax_pi.set_xlabel(r'scale  $(x_1)$', color=GREY, fontsize=8)
        ax_pi.set_ylabel(r'brightness  $(x_2)$', color=GREY, fontsize=8)
        divider_pi = make_axes_locatable(ax_pi)
        cax_pi = divider_pi.append_axes("right", size="5%", pad=0.06)
        cb_pi = plt.colorbar(cf_pi, cax=cax_pi)
        cb_pi.set_ticks(pi_ticks)
        cb_pi.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)

        # ── row 1 : σ(x) uncertainty (magma background) ──────────────────────
        ax_sig = axes[1, col_idx]
        _style_ax(ax_sig)
        cf_sig = ax_sig.contourf(g_fine, g_fine, Sig,
                                 levels=sig_levels, cmap='magma')
        sig_cl = [l for l in sig_contour_lines
                  if Sig.min() < l < Sig.max()]
        cs_sig = ax_sig.contour(g_fine, g_fine, Sig,
                       levels=sig_cl if sig_cl else sig_contour_lines,
                       colors='white', linewidths=0.5)
        if is_final:
            ax_sig.clabel(cs_sig, cs_sig.levels, inline=True, fmt='%.1f',
                          fontsize=FONT - 2, colors='white')
        ax_sig.scatter(X_tr[:, 0], X_tr[:, 1],
                       c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                       s=28, zorder=7, edgecolors='black', linewidths=0.5)
        ax_sig.scatter([x_new[0]], [x_new[1]], s=80, marker='*',
                       color=LAVENDER, zorder=10, edgecolors='black', linewidths=0.4)
        ax_sig.scatter([xb[0]], [xb[1]], s=45, marker='D',
                       color=GOLD, zorder=9, edgecolors='black', linewidths=0.5)
        divider_sig = make_axes_locatable(ax_sig)
        cax_sig = divider_sig.append_axes("right", size="5%", pad=0.06)
        cb_sig = plt.colorbar(cf_sig, cax=cax_sig)
        cb_sig.set_ticks(sig_ticks)
        cb_sig.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        cb_sig.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)

        ax_sig.set_title(r'$\sigma(x^*) = $' + f'{sig_at_xb:.2f}',
                         color='black', fontsize=9)
        ax_sig.set_xlabel(r'scale  $(x_1)$', color=GREY, fontsize=8)
        ax_sig.set_ylabel(r'brightness  $(x_2)$', color=GREY, fontsize=8)

        # ── row 2 : acquisition (fixed global scale for cross-step comparison) ──
        ax_acq = axes[2, col_idx]
        _style_ax(ax_acq)
        cf_acq = ax_acq.contourf(g_fine, g_fine, Acq,
                                  levels=acq_levels, cmap='plasma')
        ax_acq.contour(g_fine, g_fine, Acq,
                       levels=7, colors='black',
                       linewidths=0.4, alpha=0.35)
        ax_acq.scatter([x_new[0]], [x_new[1]], s=200, marker='*',
                       color=LAVENDER, zorder=10, edgecolors='black', linewidths=0.5)
        ax_acq.scatter(X_tr[:, 0], X_tr[:, 1],
                       s=40, color='#888888', alpha=0.7, zorder=6,
                       edgecolors='black', linewidths=0.4)
        acq_max_val = float(Acq.max())
        ax_acq.set_title(r'$\alpha_{\max} = $' + f'{acq_max_val:.2f}',
                         color='black', fontsize=9)
        ax_acq.set_xlabel(r'scale  $(x_1)$', color=GREY, fontsize=8)
        ax_acq.set_ylabel(r'brightness  $(x_2)$', color=GREY, fontsize=8)
        divider_acq = make_axes_locatable(ax_acq)
        cax_acq = divider_acq.append_axes("right", size="5%", pad=0.06)
        cb_acq = plt.colorbar(cf_acq, cax=cax_acq)
        cb_acq.set_ticks(acq_ticks)
        cb_acq.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        cb_acq.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)

    fig.legend(handles=legend_elements,
               loc='lower center', ncol=4, fontsize=FONT + 1,
               facecolor='white', edgecolor=BORDER,
               framealpha=0.95, labelcolor='black',
               bbox_to_anchor=(0.5, -0.04))

    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved -> {save_path}')
    plt.close()


# -- Zoomed-in view of the GP posterior at the final step ----------------------

def plot_zoomed_last_step(history, save_path,
                           zoom_x=(0.3, .7), zoom_y=(0.3, 0.7)):
    import matplotlib.cm as cm

    h        = history[-1]
    g_fine   = h['g_fine']
    nf       = len(g_fine)
    Pi_bar   = h['pi_plot'].reshape(nf, nf)
    X_tr     = h['X_tr']
    k_tr     = h['k_tr']
    m_tr     = h['m_tr']
    x_new    = h['x_new']
    n_obs    = h['n_obs']
    emp_rate = k_tr / np.maximum(m_tr, 1)

    pi_levels      = np.linspace(0, 1, 101)
    pi_ticks       = np.round(np.linspace(0, 0.99, 10), 2)
    contour_levels = [l for l in [0.3, 0.5, 0.7, 0.9]
                      if Pi_bar.min() < l < Pi_bar.max()]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    cf = ax.contourf(g_fine, g_fine, Pi_bar, levels=pi_levels, cmap='viridis')
    if contour_levels:
        cs = ax.contour(g_fine, g_fine, Pi_bar, levels=contour_levels,
                        colors='white', linewidths=0.9, alpha=0.85)
        _label_contours(ax, cs)

    ax.scatter(X_tr[:, 0], X_tr[:, 1],
               c=emp_rate, cmap='viridis', vmin=0, vmax=1,
               s=55, zorder=7, edgecolors='black', linewidths=0.6)
    ax.scatter([x_new[0]], [x_new[1]], s=160, marker='*',
               color=LAVENDER, zorder=10, edgecolors='black', linewidths=0.5)
    if h['phase'] == 'bo':
        X_plot = h['X_plot']
        xb = X_plot[h['pi_plot'].argmax()]
        ax.scatter([xb[0]], [xb[1]], s=90, marker='D',
                   color=GOLD, zorder=9, edgecolors='black', linewidths=0.6)

        # best_idx = np.argmax(k_tr / m_tr)
        # ax.scatter([X_tr[best_idx, 0]], [X_tr[best_idx, 1]], s=90,
        #            marker='s', color=BEST_SO_FAR, zorder=9,
        #            edgecolors='black', linewidths=0.6)

    ax.set_xlim(*zoom_x)
    ax.set_ylim(*zoom_y)
    ax.set_xticks(np.round(np.linspace(zoom_x[0], zoom_x[1], 6), 2))
    ax.set_yticks(np.round(np.linspace(zoom_y[0], zoom_y[1], 6), 2))

    ax.set_title(f'Zoomed view  —  Step {h["active_step"] + 1}  (n = {n_obs})'
                 f'   |  scale $\\in$ [{zoom_x[0]}, {zoom_x[1]}],  '
                 f'brightness $\\in$ [{zoom_y[0]}, {zoom_y[1]}]',
                 color='black', fontsize=9, fontweight='bold')
    ax.set_xlabel(r'scale  $(x_1)$', color=GREY, fontsize=8)
    ax.set_ylabel(r'brightness  $(x_2)$', color=GREY, fontsize=8)

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.06)
    cb      = plt.colorbar(cf, cax=cax)
    cb.set_ticks(pi_ticks)
    cb.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)

    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved -> {save_path}')
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
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'  Saved -> {save_path}')
    plt.close()


# -- pi_max convergence with propagated 95% CI error bars ---------------------

def plot_pi_max_with_ci(history, acq_type, save_path):
    from scipy.stats import norm

    steps, pi_maxes, pi_lo, pi_hi = [], [], [], []

    for h in history:
        idx      = h['pi_plot'].argmax()
        mu_star  = float(h['mu_plot'][idx])
        sig_star = float(h['sig_plot'][idx])
        pi_c     = float(h['pi_max'])          # norm.cdf(mu* / sqrt(1+sig*²))

        # propagate ±1.96σ latent band through probit link
        lo = float(norm.cdf(mu_star - 1.96 * sig_star))
        hi = float(norm.cdf(mu_star + 1.96 * sig_star))

        steps.append(h['active_step'] + 1)
        pi_maxes.append(pi_c)
        pi_lo.append(pi_c - lo)
        pi_hi.append(hi - pi_c)

    steps    = np.array(steps)
    pi_maxes = np.array(pi_maxes)
    yerr     = np.array([pi_lo, pi_hi])

    color = '#5544aa' if acq_type == 'UCB' else '#007a6e'

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=GREY, labelsize=FONT)

    ax.fill_between(steps, pi_maxes - yerr[0], pi_maxes + yerr[1],
                    color=color, alpha=0.18, label='95% CI  (probit-propagated)')
    ax.plot(steps, pi_maxes,
            color=color, lw=2.0, zorder=3)
    ax.errorbar(steps, pi_maxes, yerr=yerr,
                fmt='o', color=color, ecolor=color, elinewidth=1.2,
                capsize=3, capthick=1.2, markersize=4, zorder=4,
                label=f'{acq_type}  pi_max  (final={pi_maxes[-1]:.3f})')

    ax.set_xlabel('Active query step', color=GREY, fontsize=FONT + 1)
    ax.set_ylabel(r'$\pi_{\max}$  (model recommendation)', color=GREY, fontsize=FONT + 1)
    ax.set_title(r'Convergence of $\pi_{\max}$ with propagated uncertainty  (±95% CI)',
                 color='black', fontsize=FONT + 2, fontweight='bold')
    ax.legend(facecolor='white', edgecolor=BORDER, labelcolor='black',
              fontsize=FONT, framealpha=1)
    ax.set_xlim(1, len(steps))
    ax.set_ylim(bottom=.55, top=min(1.05, (pi_maxes + yerr[1]).max() + 0.05))
    ax.grid(True, color='#dddddd', linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'  Saved -> {save_path}')
    plt.close()


# -- 3D posterior surface at the last BO step ----------------------------------

def plot_posterior_3d(history, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    h       = history[-1]
    g_fine  = h['g_fine']
    nf      = len(g_fine)
    G1, G2  = np.meshgrid(g_fine, g_fine)

    Mu  = h['mu_plot'].reshape(nf, nf)
    Pi  = h['pi_plot'].reshape(nf, nf)

    X_tr     = h['X_tr']
    k_tr     = h['k_tr']
    m_tr     = h['m_tr']
    emp_rate = k_tr / np.maximum(m_tr, 1)
    n_obs    = h['n_obs']
    step     = h['active_step'] + 1

    pi_argmax = h['pi_plot'].argmax()
    xb = h['X_plot'][pi_argmax]

    fig = plt.figure(figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    # ---- left: GP posterior mean (mu) ----------------------------------------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor(BG)
    surf1 = ax1.plot_surface(G1, G2, Mu, cmap='coolwarm',
                              alpha=0.88, linewidth=0, antialiased=True)
    ax1.scatter(X_tr[:, 0], X_tr[:, 1],
                np.interp(emp_rate, [0, 1], [Mu.min(), Mu.max()]),
                c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                s=40, zorder=6, depthshade=False)
    ax1.scatter([xb[0]], [xb[1]], [Mu.max()],
                s=120, marker='*', color=GOLD, zorder=9, depthshade=False)
    ax1.set_xlabel('scale  $(x_1)$',    color=GREY, fontsize=8, labelpad=6)
    ax1.set_ylabel('hsv_v  $(x_2)$',    color=GREY, fontsize=8, labelpad=6)
    ax1.set_zlabel(r'$\mu(\mathbf{x})$', color=GREY, fontsize=8, labelpad=6)
    ax1.set_title(f'Posterior mean  —  Step {step}  (n = {n_obs})',
                  color=GREY, fontsize=9, fontweight='bold')
    ax1.tick_params(colors=GREY, labelsize=7)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.08)

    # ---- right: posterior probability of success (pi) ------------------------
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor(BG)
    surf2 = ax2.plot_surface(G1, G2, Pi, cmap='viridis',
                              alpha=0.88, linewidth=0, antialiased=True)
    ax2.scatter(X_tr[:, 0], X_tr[:, 1], emp_rate,
                c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                s=40, zorder=6, depthshade=False)
    ax2.scatter([xb[0]], [xb[1]], [h['pi_max']],
                s=120, marker='*', color=GOLD, zorder=9, depthshade=False,
                label=f'pi_max={h["pi_max"]:.3f}  @  '
                      f'({xb[0]:.3f}, {xb[1]:.3f})')
    ax2.set_xlabel('scale  $(x_1)$',          color=GREY, fontsize=8, labelpad=6)
    ax2.set_ylabel('hsv_v  $(x_2)$',          color=GREY, fontsize=8, labelpad=6)
    ax2.set_zlabel(r'$\pi(\mathbf{x})$',       color=GREY, fontsize=8, labelpad=6)
    ax2.set_title(f'Posterior P(success)  —  Step {step}  (n = {n_obs})',
                  color=GREY, fontsize=9, fontweight='bold')
    ax2.tick_params(colors=GREY, labelsize=7)
    ax2.legend(fontsize=7, loc='upper left')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved -> {save_path}')
    plt.close()


# -- σ (latent) and CI width — monotone shrinkage -----------------------------

def plot_sigma_ci_shrinkage(history, save_path):
    from scipy.stats import norm

    steps, sigmas, ci_widths = [], [], []
    for h in history:
        idx     = h['pi_plot'].argmax()
        mu_s    = float(h['mu_plot'][idx])
        sig_s   = float(h['sig_plot'][idx])
        lo      = float(norm.cdf(mu_s - 1.96 * sig_s))
        hi      = float(norm.cdf(mu_s + 1.96 * sig_s))
        steps.append(h['active_step'] + 1)
        sigmas.append(sig_s)
        ci_widths.append(hi - lo)

    steps     = np.array(steps)
    sigmas    = np.array(sigmas)
    ci_widths = np.array(ci_widths)

    GREEN = '#1a9e72'
    GOLD_LINE = '#d4900a'

    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    for sp in ax1.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax1.tick_params(colors=GREY, labelsize=FONT)

    ax1.fill_between(steps, 0, ci_widths, color=GREEN, alpha=0.18)
    ax1.plot(steps, ci_widths, color=GREEN, lw=2.0, marker='o',
             markersize=4, zorder=3, label='CI width  (hi − lo)')
    ax1.set_xlabel('Active query step', color=GREY, fontsize=FONT + 1)
    ax1.set_ylabel('CI width', color=GREEN, fontsize=FONT + 1)
    ax1.yaxis.label.set_color(GREEN)
    ax1.tick_params(axis='y', colors=GREEN, labelsize=FONT)
    ax1.set_xlim(1, len(steps))
    ax1.set_ylim(bottom=0)
    ax1.set_xticks(steps[::2])
    ax1.grid(True, color='#e8e8e8', linewidth=0.6, linestyle='-')
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(steps, sigmas, color=GOLD_LINE, lw=1.8, marker='o',
             markersize=4, linestyle='--', zorder=4, label=r'$\sigma$ (latent)')
    ax2.set_ylabel(r'$\sigma$', color=GOLD_LINE, fontsize=FONT + 1)
    ax2.yaxis.label.set_color(GOLD_LINE)
    ax2.tick_params(axis='y', colors=GOLD_LINE, labelsize=FONT)
    for sp in ax2.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax2.set_ylim(bottom=0)

    ax1.set_title(r'$\sigma$ (latent) and CI width — monotone shrinkage',
                  color='black', fontsize=FONT + 2, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor='white', edgecolor=BORDER, labelcolor='black',
               fontsize=FONT, framealpha=1, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
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

    fig5_path = os.path.join(cfg['out_dir'], f'figure5_{cfg["acq"]}_combined.png')
    plot_figure5(history, acq_type=cfg['acq'], save_path=fig5_path,
                 snap_steps=cfg['snaps'], kappa=cfg['kappa'],
                 m_trials=cfg.get('m_trials', 15),
                 budget=cfg.get('budget', len(df)),
                 real_blackbox=not cfg.get('use_dummy', False))

    fig5s_path = os.path.join(cfg['out_dir'], f'figure5_{cfg["acq"]}_with_sigma.png')
    plot_figure5_sigma(history, acq_type=cfg['acq'], save_path=fig5s_path,
                       snap_steps=cfg['snaps'], kappa=cfg['kappa'],
                       m_trials=cfg.get('m_trials', 15),
                       budget=cfg.get('budget', len(df)),
                       real_blackbox=not cfg.get('use_dummy', False))

    zoom_path = os.path.join(cfg['out_dir'], f'figure5_{cfg["acq"]}_zoomed_last_step.png')
    plot_zoomed_last_step(history, save_path=zoom_path,
                          zoom_x=(0.8, 1.0), zoom_y=(0.35, 0.7))

    conv_path = os.path.join(cfg['out_dir'], f'convergence_{cfg["acq"]}.png')
    plot_convergence({cfg['acq']: history}, save_path=conv_path,
                     kappa=cfg['kappa'])

    acq_evol_path = os.path.join(cfg['out_dir'], f'acq_evolution_{cfg["acq"]}.png')
    plot_acquisition_evolution(history, acq_type=cfg['acq'],
                               save_path=acq_evol_path, kappa=cfg['kappa'])

    surf3d_path = os.path.join(cfg['out_dir'], f'posterior_3d_{cfg["acq"]}.png')
    plot_posterior_3d(history, save_path=surf3d_path)

    ci_path = os.path.join(cfg['out_dir'], f'pi_max_convergence_ci_{cfg["acq"]}.png')
    plot_pi_max_with_ci(history, acq_type=cfg['acq'], save_path=ci_path)

    shrink_path = os.path.join(cfg['out_dir'], f'sigma_ci_shrinkage_{cfg["acq"]}.png')
    plot_sigma_ci_shrinkage(history, save_path=shrink_path)

    print('\nDone.')


if __name__ == '__main__':
    main()