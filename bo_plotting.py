"""
bo_plotting.py
==============
All visualisation for the 2D Binomial-Probit BO pipeline.

No pipeline imports, no optimisation logic — pure matplotlib.

Functions
---------
  plot_figure5(history, acq_type, save_path, snap_steps, true_fn)
      Figure 5-style contour snapshots.
      true_fn=None  →  real-blackbox mode  (no reference column, no cross marker)
      true_fn=func  →  synthetic/debug mode (reference column shown)

  plot_convergence(histories, save_path, kappa)
      π̂_max vs iteration for EI and UCB side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── Publication palette (white background) ───────────────────────────────────
BG      = 'white'
PANEL   = 'white'
BORDER  = '#bbbbbb'
GREY    = '#444444'
GOLD    = '#c77c00'
LAVENDER= '#5544aa'


def _style_ax(ax):
    """Apply consistent white-background style to a single axis."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GREY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def _cbar(ax, mappable, label=''):
    """Attach a compact colourbar to ax."""
    cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=GREY, labelcolor=GREY, labelsize=7)
    if label:
        cb.set_label(label, color=GREY, fontsize=7)
    return cb


def _label_contours(ax, cs, color='black', fontsize=7):
    """Inline-label a contour set."""
    ax.clabel(cs, cs.levels, inline=True, fmt='%.1f',
              fontsize=fontsize, colors=color)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5-style contour snapshots
# ═══════════════════════════════════════════════════════════════════════════

def plot_figure5(history, acq_type, save_path,
                 snap_steps=(0, 7, -1),
                 true_fn=None):
    """
    Replicate Figure 5 from Tesch et al. (ICML 2013) for 2D BO results.

    Layout: 2 rows × (1 + n_snaps) columns
      Row 0  — Approximated π̄(x)     [viridis colourmap]
      Row 1  — Acquisition function   [plasma  colourmap]

    Column 0 behaviour depends on true_fn flag:
      true_fn is not None  →  SYNTHETIC / DEBUG MODE
                               Show true π(x) reference in col 0.
                               Useful for sanity-checking the GP surrogate
                               against the known ground truth.
      true_fn is None      →  REAL BLACKBOX MODE
                               No reference column. Col 0 is the earliest
                               snapshot. No true-optimum cross marker shown.

    Parameters
    ----------
    history    : list of dicts from run_bo_2d
    acq_type   : 'EI' or 'UCB'  (used for title and y-axis label)
    save_path  : output file path
    snap_steps : tuple of history indices to show.
                 Use -1 for the last step.
                 Default (0, 7, -1) → step 1, step 8, final step.
    true_fn    : callable X→π  or  None
                 Pass true_pi_2d for synthetic runs.
                 Pass None when running against the real YOLO pipeline.
    """
    # Resolve -1 index
    snap_steps = [s if s >= 0 else len(history) + s for s in snap_steps]

    g_fine = history[0]['g_fine']
    X_plot = history[0]['X_plot']
    nf     = len(g_fine)

    synthetic_mode = (true_fn is not None)

    # True π reference — only in synthetic mode
    if synthetic_mode:
        pi_true_flat = true_fn(X_plot)
        Pi_true      = pi_true_flat.reshape(nf, nf)
        idx_max      = pi_true_flat.argmax()
        x1_max       = X_plot[idx_max, 0]
        x2_max       = X_plot[idx_max, 1]
    else:
        pi_true_flat = None
        x1_max = x2_max = None

    n_snaps = len(snap_steps)
    n_cols  = (n_snaps + 1) if synthetic_mode else n_snaps
    n_rows  = 2

    fig = plt.figure(figsize=(3.8 * n_cols, 3.8 * n_rows))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.30, wspace=0.28)

    pi_levels = np.linspace(0, 1, 101)
    cmap_pi   = 'viridis'
    cmap_acq  = 'plasma'

    # ── Column 0: True π reference (synthetic mode only) ─────────────────
    if synthetic_mode:
        for row in range(n_rows):
            ax = fig.add_subplot(gs[row, 0])
            _style_ax(ax)
            cf = ax.contourf(g_fine, g_fine, Pi_true,
                             levels=pi_levels, cmap=cmap_pi)
            cs = ax.contour(g_fine, g_fine, Pi_true,
                            levels=[0.3, 0.5, 0.7, 0.9],
                            colors='black', linewidths=0.8, alpha=0.7)
            _label_contours(ax, cs)
            ax.scatter([x1_max], [x2_max], s=160, marker='*',
                       color=GOLD, zorder=10, edgecolors='black', linewidths=0.5)
            _cbar(ax, cf, 'π')
            if row == 0:
                ax.set_title(f'True  π(x)\nπ_max = {pi_true_flat.max():.3f}',
                             color='black', fontsize=9, fontweight='bold')
                ax.set_ylabel('Approx  π̄(x)', color=GREY, fontsize=8)
            else:
                ax.set_title('True  π(x)', color='black', fontsize=9,
                             fontweight='bold')
                lbl = 'EI_π  acquisition' if acq_type == 'EI' else 'UCB  acquisition'
                ax.set_ylabel(lbl, color=GREY, fontsize=8)
            ax.set_xlabel('scale  (x₁)', color=GREY, fontsize=8)

    # ── Snapshot columns ──────────────────────────────────────────────────
    for col_idx, s in enumerate(snap_steps):
        col      = col_idx + (1 if synthetic_mode else 0)
        is_final = (col_idx == len(snap_steps) - 1)
        h        = history[s]

        Pi_bar   = h['pi_plot'].reshape(nf, nf)
        Acq      = h['acq_plot'].reshape(nf, nf)
        X_tr     = h['X_tr']
        k_tr     = h['k_tr']
        m_tr     = h['m_tr']
        x_new    = h['x_new']
        n_obs    = h['n_obs']
        emp_rate = k_tr / m_tr

        step_title = f'Step {s + 1}  (n = {n_obs})'

        # ── Row 0: Approximated π̄ ─────────────────────────────────────────
        ax_pi = fig.add_subplot(gs[0, col])
        _style_ax(ax_pi)
        cf_pi = ax_pi.contourf(g_fine, g_fine, Pi_bar,
                               levels=pi_levels, cmap=cmap_pi)

        # Contour iso-lines: labelled on final step (paper style)
        if is_final:
            cs_pi = ax_pi.contour(g_fine, g_fine, Pi_bar,
                                  levels=[0.3, 0.5, 0.7, 0.9],
                                  colors='black', linewidths=1.0, alpha=0.9)
            _label_contours(ax_pi, cs_pi)
        else:
            ax_pi.contour(g_fine, g_fine, Pi_bar,
                          levels=[0.3, 0.5, 0.7, 0.9],
                          colors='black', linewidths=0.5, alpha=0.5)

        # Training points: viridis aligned with contour colourmap
        ax_pi.scatter(X_tr[:, 0], X_tr[:, 1],
                      c=emp_rate, cmap='viridis', vmin=0, vmax=1,
                      s=55, zorder=7, edgecolors='black', linewidths=0.6)

        # Next query star
        ax_pi.scatter([x_new[0]], [x_new[1]], s=180, marker='*',
                      color=LAVENDER, zorder=10,
                      edgecolors='black', linewidths=0.5)

        # π̂_max diamond
        xb = X_plot[h['pi_plot'].argmax()]
        ax_pi.scatter([xb[0]], [xb[1]], s=100, marker='D',
                      color=GOLD, zorder=9, edgecolors='black', linewidths=0.6)

        # True optimum cross — synthetic mode only
        if synthetic_mode:
            ax_pi.scatter([x1_max], [x2_max], s=80, marker='+',
                          color='black', zorder=8, linewidths=1.5)

        ax_pi.set_title(step_title, color='black', fontsize=9, fontweight='bold')
        ax_pi.set_xlabel('scale  (x₁)', color=GREY, fontsize=8)
        if col == (1 if synthetic_mode else 0):
            ax_pi.set_ylabel('Approx  π̄(x)', color=GREY, fontsize=8)
        _cbar(ax_pi, cf_pi)

        # ── Row 1: Acquisition function ───────────────────────────────────
        ax_acq = fig.add_subplot(gs[1, col])
        _style_ax(ax_acq)
        acq_max = max(Acq.max(), 1e-9)
        cf_acq  = ax_acq.contourf(g_fine, g_fine, Acq,
                                  levels=np.linspace(0, acq_max, 101),
                                  cmap=cmap_acq)
        ax_acq.contour(g_fine, g_fine, Acq,
                       levels=5, colors='black', linewidths=0.4, alpha=0.35)

        # Next query star
        ax_acq.scatter([x_new[0]], [x_new[1]], s=180, marker='*',
                       color=LAVENDER, zorder=10,
                       edgecolors='black', linewidths=0.5)

        # Training points (grey, just for spatial reference)
        ax_acq.scatter(X_tr[:, 0], X_tr[:, 1],
                       s=35, color='#888888', alpha=0.5, zorder=6,
                       edgecolors='black', linewidths=0.4)

        ax_acq.set_title(step_title, color='black', fontsize=9, fontweight='bold')
        ax_acq.set_xlabel('scale  (x₁)', color=GREY, fontsize=8)
        if col == (1 if synthetic_mode else 0):
            lbl = 'EI_π  acquisition' if acq_type == 'EI' else 'UCB  acquisition'
            ax_acq.set_ylabel(lbl, color=GREY, fontsize=8)
        _cbar(ax_acq, cf_acq)

        # ── Row 2: Uncertainty — COMMENTED OUT ────────────────────────────
        # uncertainty = 0.5 - np.abs(Pi_bar - 0.5)
        # ...

    # ── Shared legend ─────────────────────────────────────────────────────
    legend_els = [
        Line2D([0],[0], color=GOLD,     lw=0, marker='D', ms=8,
               label='π̂_max  (model recommendation)'),
        Line2D([0],[0], color=LAVENDER, lw=0, marker='*', ms=11,
               label='Next query point'),
        Line2D([0],[0], color='#44aa44', lw=0, marker='o', ms=7,
               label='High success rate  (k/m → 1)'),
        Line2D([0],[0], color='#440044', lw=0, marker='o', ms=7,
               label='Low success rate   (k/m → 0)'),
    ]
    if synthetic_mode:
        legend_els.insert(0, Line2D([0],[0], color=GOLD, lw=0, marker='*',
                                    ms=11, label='True π_max'))
        legend_els.append(Line2D([0],[0], color='black', lw=0, marker='+',
                                 ms=9, label='True optimum (reference)'))

    mode_tag = '[DEBUG: synthetic]' if synthetic_mode else '[real blackbox]'
    acq_lbl  = 'EI_π  (Tesch 2013)' if acq_type == 'EI' else f'UCB  (κ={1.5})'
    m_val    = int(history[0]['m_tr'][0])

    fig.legend(handles=legend_els, loc='lower center',
               ncol=min(len(legend_els), 5),
               facecolor='white', edgecolor=BORDER, labelcolor='black',
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f'2D Binomial-Probit BO  |  {acq_lbl}  |  '
        f'm = {m_val} trials/query  |  budget = {len(history)}  |  {mode_tag}',
        color='black', fontsize=11, fontweight='bold', y=1.02
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved → {save_path}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Convergence plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(histories, save_path, kappa=1.5, true_fn=None):
    """
    Plot π̂_max vs active query step for each acquisition type.

    Parameters
    ----------
    histories : dict  {acq_type: history_list}
                e.g. {'EI': [...], 'UCB': [...]}
    save_path : output file path
    kappa     : UCB exploration weight (used in legend label only)
    true_fn   : callable X→π or None
                If provided, draw a horizontal dashed reference line at
                the true π_max.  Pass None for real blackbox runs.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
        sp.set_linewidth(0.8)
    ax.tick_params(colors=GREY, labelsize=9)

    # True π_max reference line — synthetic mode only
    if true_fn is not None:
        g    = np.linspace(0, 1, 100)
        G1, G2 = np.meshgrid(g, g)
        X_ref  = np.column_stack([G1.ravel(), G2.ravel()])
        true_max = true_fn(X_ref).max()
        ax.axhline(true_max, color='#888888', lw=1.2, ls='--',
                   label=f'True π_max = {true_max:.3f}  [debug ref]')

    line_colors = {'EI': '#007a6e', 'UCB': '#5544aa'}
    line_styles = {'EI': '-',       'UCB': '--'}
    line_labels = {
        'EI':  'EI_π  (Laplace)',
        'UCB': f'UCB  (κ = {kappa})',
    }

    for acq_type, history in histories.items():
        pi_maxes = [h['pi_max'] for h in history]
        steps    = np.arange(1, len(pi_maxes) + 1)
        final    = pi_maxes[-1]
        ax.plot(steps, pi_maxes,
                color=line_colors[acq_type],
                lw=2.2,
                ls=line_styles[acq_type],
                label=f"{line_labels[acq_type]}   (final π̂_max = {final:.3f})")

    ax.set_xlabel('Active query step', color=GREY, fontsize=10)
    ax.set_ylabel('π̂_max  (model recommendation of best point)',
                  color=GREY, fontsize=10)
    ax.set_title('Convergence of π̂_max  vs  iteration',
                 color='black', fontsize=11, fontweight='bold')
    ax.legend(facecolor='white', edgecolor='#bbbbbb', labelcolor='black',
              fontsize=9.5, framealpha=1)
    ax.set_xlim(1, max(len(h) for h in histories.values()))
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#dddddd', linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'  Saved → {save_path}')
    plt.close()
