"""
bo_optimizer.py
===============
Main entry point for 2D Binomial-Probit Bayesian Optimisation of the
YOLO landing pipeline over (scale, hsv_v) ∈ [0,1]².

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE FLAG  (the only line you need to change between runs)

    USE_DUMMY = True
        ↳ Does NOT call the YOLO pipeline.
          Uses true_pi_2d() as a synthetic stand-in.
          Use this to verify the GP surrogate, acquisition logic,
          plotting, and checkpointing before spending real compute.

    USE_DUMMY = False
        ↳ Calls train_and_evaluate() from the real YOLO pipeline.
          success_rate × 10 is recovered as integer k (m=10 fixed).
          Use this for actual hyperparameter optimisation runs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure
---------
  oracle_dummy()        synthetic oracle using true_pi_2d
  oracle_yolo()         real oracle wrapping train_and_evaluate()
  run_bo_2d()           kernel-agnostic BO loop (oracle passed as argument)
  main()                config, run, plot, save

Checkpointing
-------------
  After every observation X_tr, k_tr, m_tr are appended to
  bo_checkpoint.csv so a KeyboardInterrupt never loses data.
  On restart, pass resume=True to load from the checkpoint and
  continue from where the run stopped.
"""

import os
import gc
import csv
import numpy as np
from scipy.stats import norm
from utils.config import load_yaml_file, get_next_optimization_run_number, update_metadata
from utils import constants

# ── Internal modules ──────────────────────────────────────────────────────────
from utils.gpc_core import (
    build_kernel,
    laplace,
    predict_2d,
    acq_EI_pi,
    acq_UCB,
    optimise_hyperparams,
)
from utils.bo_plotting import plot_figure5, plot_convergence

# ── True function (synthetic / debug only) ───────────────────────────────────
def true_pi_2d(X):
    """
    Synthetic 2D success-probability surface on [0,1]².

    Two peaks:
      Primary   at (scale=0.75, hsv_v=0.70) → π ≈ 0.92
      Secondary at (scale=0.25, hsv_v=0.30) → π ≈ 0.70

    Used ONLY when USE_DUMMY = True.
    Has no role in production runs.

    X : (n, 2)  →  pi : (n,)
    """
    X   = np.atleast_2d(X)
    x1  = X[:, 0]   # scale
    x2  = X[:, 1]   # hsv_v

    f1    = 2.5 * np.exp(-0.5 * ((x1-0.75)**2/0.08 + (x2-0.70)**2/0.10))
    f2    = 1.4 * np.exp(-0.5 * ((x1-0.25)**2/0.06 + (x2-0.30)**2/0.07))
    ridge = 0.4 * np.exp(-0.5 * ((x1-0.50)**2/0.20 + (x2-0.50)**2/0.20))
    bg    = -1.2 + 0.5*x1 + 0.3*x2

    return norm.cdf(f1 + f2 + ridge + bg)


# ═══════════════════════════════════════════════════════════════════════════
# Oracles
# ═══════════════════════════════════════════════════════════════════════════

def oracle_dummy(scale, hsv_v, rng, m=15):
    """
    Synthetic oracle — used when USE_DUMMY = True.

    Draws k ~ Binomial(m, π(scale, hsv_v)) from the known true_pi_2d.
    Mimics exactly what oracle_yolo would return but without touching
    the YOLO pipeline.

    Parameters
    ----------
    scale  : float ∈ [0,1]
    hsv_v  : float ∈ [0,1]
    rng    : numpy Generator  (for reproducibility)
    m      : int   number of trials per query  (match your real pipeline)

    Returns
    -------
    k : int    number of successes
    m : int    number of trials (always m)
    """
    X  = np.array([[scale, hsv_v]])
    pi = true_pi_2d(X)[0]
    k  = int(rng.binomial(m, pi))
    return k, m


def oracle_yolo(scale, hsv_v, rng=None, m=15):
    """
    Real oracle — used when USE_DUMMY = False.

    Calls the YOLO training + simulation pipeline once at (scale, hsv_v).
    The pipeline internally runs 10 landing attempts and writes the
    success rate to performance_summary.csv.

    success_rate = k / 10  →  k = round(success_rate × 10)

    rng is accepted for API compatibility but unused (the pipeline
    has its own internal randomness).

    Parameters
    ----------
    scale  : float ∈ [0,1]
    hsv_v  : float ∈ [0,1]
    rng    : ignored (kept for consistent call signature)
    m      : int  must be 10 to match the pipeline's internal trial count

    Returns
    -------
    k : int    number of successful landings out of 10
    m : int    10  (fixed by the pipeline)
    """
    # ── Lazy import: only loaded when USE_DUMMY = False ───────────────────
    # This keeps the dummy mode fully self-contained with no YOLO dependencies.
    import torch
    import subprocess
    import pandas as pd
    from yolo_training.YOLO_training_pipeline import YOLOTrainingPipeline

    from utils import constants

    cfg_file    = './configs/hyp_bayes.yaml'
    results_csv = './utils/performance_summary.csv'
    save_dir    = './perception/yolov5/models'

    try:
        torch.cuda.empty_cache()

        pipeline = YOLOTrainingPipeline(
            cfg_file=cfg_file,
            save_dir=save_dir,
            scale_value=scale,
            hsv_v_value=hsv_v,
        )
        pipeline.run()
        torch.cuda.empty_cache()
        gc.collect()

        subprocess.run(
            ["python3", "rraaa.py", "configs/single-static.yml"],
            check=True
        )

        update_metadata(
            fields={"opt_success": "True"},
            meta_file_path=constants.metadata_file_path,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # ── Read success rate and recover integer k ───────────────────────
        df          = pd.read_csv(results_csv)
        df['Run Number'] = df['Run Number'].str.extract(r'(\d+)').astype(int)
        latest_row  = df[df['Run Number'] == df['Run Number'].max()].iloc[0]
        success_rate = float(latest_row['Success Rate'])

        k = int(round(success_rate * m))   # exact because pipeline counts integers
        print(f"  oracle_yolo: scale={scale:.4f}  hsv_v={hsv_v:.4f}  "
              f"→  k={k}/{m}  (rate={success_rate:.2f})")
        return k, m

    except KeyboardInterrupt:
        print("\n  oracle_yolo: interrupted. Marking opt_success=False.")
        update_metadata(
            fields={"opt_success": "False"},
            meta_file_path=constants.metadata_file_path,
        )
        torch.cuda.empty_cache()
        gc.collect()
        raise   # propagate so the BO loop can checkpoint and exit cleanly


# ═══════════════════════════════════════════════════════════════════════════
# Checkpointing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _checkpoint_append(path, x_new, k_new, m_new):
    """Append one observation row to the checkpoint CSV."""
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['scale', 'hsv_v', 'k', 'm'])
        writer.writerow([f'{x_new[0]:.6f}', f'{x_new[1]:.6f}',
                         int(k_new), int(m_new)])


def _checkpoint_load(path):
    """
    Load a checkpoint CSV and return (X_tr, k_tr, m_tr).
    Returns None if file does not exist or is empty.
    """
    if not os.path.isfile(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    X_tr = df[['scale', 'hsv_v']].values.astype(float)
    k_tr = df['k'].values.astype(float)
    m_tr = df['m'].values.astype(float)
    print(f"  Resumed from checkpoint: {len(df)} observations loaded.")
    return X_tr, k_tr, m_tr


# ═══════════════════════════════════════════════════════════════════════════
# BO loop
# ═══════════════════════════════════════════════════════════════════════════

def run_bo_2d(oracle_fn,
              acq_type          = 'UCB',
              n_init            = 5,
              budget            = 30,
              ls                = 0.20,
              var               = 2.0,
              kappa             = 1.5,
              n_grid            = 22,
              kernel            = 'rbf',
              seed              = 42,
              optimise_hp       = True,
              hp_interval       = 5,
              checkpoint_path   = './bo_checkpoint.csv',
              resume            = False):
    """
    Kernel-agnostic 2D BO loop.

    The oracle is passed as a callable so the same loop works for both
    the dummy and the real pipeline — no if/else inside the loop.

    Parameters
    ----------
    oracle_fn         : callable(scale, hsv_v, rng) → (k, m)
                        Pass oracle_dummy or oracle_yolo from above.
    acq_type          : 'EI' or 'UCB'
    n_init            : number of space-filling initialisation points
    budget            : total queries including init
    ls                : initial GP length-scale
    var               : initial GP output variance
    kappa             : UCB exploration weight
    n_grid            : candidate pool resolution (n_grid × n_grid)
    kernel            : 'rbf' or 'matern52'
    seed              : random seed (for init and dummy oracle)
    optimise_hp       : if True, re-optimise ls and var via marginal
                        likelihood every hp_interval active steps
    hp_interval       : how often (in active steps) to re-optimise
    checkpoint_path   : path to append/read observation CSV
    resume            : if True, load from checkpoint and skip init

    Returns
    -------
    history : list of dicts (one per active step, used by bo_plotting)
    X_tr    : (n, 2)  all queried locations
    k_tr    : (n,)    success counts
    m_tr    : (n,)    trial counts
    """
    rng = np.random.default_rng(seed)

    # ── Candidate pool  (n_grid × n_grid) ────────────────────────────────
    g      = np.linspace(0, 1, n_grid)
    G1, G2 = np.meshgrid(g, g)
    X_pool = np.column_stack([G1.ravel(), G2.ravel()])   # (n_grid², 2)

    # ── Fine plotting grid (60 × 60) ──────────────────────────────────────
    g_fine      = np.linspace(0, 1, 60)
    G1f, G2f    = np.meshgrid(g_fine, g_fine)
    X_plot      = np.column_stack([G1f.ravel(), G2f.ravel()])

    # ── Initialisation ────────────────────────────────────────────────────
    if resume:
        loaded = _checkpoint_load(checkpoint_path)
        if loaded is not None:
            X_tr, k_tr, m_tr = loaded
            print(f"  Skipping init — resuming with {len(X_tr)} existing points.")
        else:
            print("  Checkpoint not found or empty — starting fresh.")
            resume = False

    if not resume:
        # Space-filling grid init (Sobol-like)
        n_side = int(np.ceil(np.sqrt(n_init)))
        g_init = np.linspace(0.1, 0.9, n_side)
        G1i, G2i = np.meshgrid(g_init, g_init)
        X_init = np.column_stack([G1i.ravel(), G2i.ravel()])
        idx    = rng.permutation(len(X_init))[:n_init]
        X_tr   = X_init[idx]
        k_tr   = np.zeros(n_init)
        m_tr   = np.zeros(n_init)

        print(f"\n  Initialising with {n_init} space-filling points...")
        for i, x in enumerate(X_tr):
            k_i, m_i = oracle_fn(x[0], x[1], rng)
            k_tr[i]  = k_i
            m_tr[i]  = m_i
            _checkpoint_append(checkpoint_path, x, k_i, m_i)
            print(f"    Init {i+1}/{n_init}: "
                  f"scale={x[0]:.3f}  hsv_v={x[1]:.3f}  "
                  f"k={k_i}/{m_i}  (rate={k_i/m_i:.2f})")

    # ── Active BO loop ────────────────────────────────────────────────────
    history   = []
    n_active  = budget - n_init
    ls_cur    = ls
    var_cur   = var

    print(f"\n  Starting {n_active} active BO steps "
          f"[acq={acq_type}  kernel={kernel}  κ={kappa}] ...\n")

    for step in range(n_active):
        n = len(X_tr)

        # ── Re-optimise hyperparameters periodically ──────────────────────
        if optimise_hp and step > 0 and step % hp_interval == 0:
            ls_new, var_new, lml = optimise_hyperparams(
                X_tr, k_tr, m_tr, kernel=kernel)
            print(f"  [step {step}] HP update: "
                  f"ls {ls_cur:.3f}→{ls_new:.3f}  "
                  f"var {var_cur:.3f}→{var_new:.3f}  "
                  f"log_mlik={lml:.2f}")
            ls_cur  = ls_new
            var_cur = var_new

        # ── Fit Laplace posterior ─────────────────────────────────────────
        K_tr = build_kernel(X_tr, X_tr, ls_cur, var_cur, kernel) + 1e-6*np.eye(n)
        f_map, Sigma = laplace(K_tr, k_tr, m_tr)

        # ── Predict on candidate pool ─────────────────────────────────────
        mu_pool, sig_pool, pi_pool = predict_2d(
            X_tr, f_map, Sigma, K_tr, X_pool, ls_cur, var_cur, kernel)
        pi_max = pi_pool.max()

        # ── Acquisition ───────────────────────────────────────────────────
        if acq_type == 'EI':
            acq_vals = acq_EI_pi(mu_pool, sig_pool, pi_max)
        else:
            acq_vals = acq_UCB(mu_pool, sig_pool, kappa)

        # Mask already-queried candidates
        queried_set = set(map(tuple, X_tr.round(6)))
        free_mask   = np.array([
            tuple(x) not in queried_set for x in X_pool.round(6)
        ])
        acq_masked = np.where(free_mask, acq_vals, -np.inf)
        x_new      = X_pool[np.argmax(acq_masked)]

        # ── Predict on fine plot grid (stored in history for plotting) ────
        mu_plot, sig_plot, pi_plot = predict_2d(
            X_tr, f_map, Sigma, K_tr, X_plot, ls_cur, var_cur, kernel)

        if acq_type == 'EI':
            acq_plot = acq_EI_pi(mu_plot, sig_plot, pi_plot.max())
        else:
            acq_plot = acq_UCB(mu_plot, sig_plot, kappa)

        history.append({
            'step'     : step,
            'n_obs'    : n,
            'X_tr'     : X_tr.copy(),
            'k_tr'     : k_tr.copy(),
            'm_tr'     : m_tr.copy(),
            'x_new'    : x_new.copy(),
            'X_plot'   : X_plot,
            'pi_plot'  : pi_plot,
            'acq_plot' : acq_plot,
            'pi_max'   : pi_max,
            'g_fine'   : g_fine,
            'ls'       : ls_cur,
            'var'      : var_cur,
        })

        # ── Query oracle ──────────────────────────────────────────────────
        k_new, m_new = oracle_fn(x_new[0], x_new[1], rng)

        # ── Checkpoint before updating arrays (safe on interrupt) ─────────
        _checkpoint_append(checkpoint_path, x_new, k_new, m_new)

        # ── Update training data ──────────────────────────────────────────
        X_tr = np.vstack([X_tr, x_new])
        k_tr = np.append(k_tr, float(k_new))
        m_tr = np.append(m_tr, float(m_new))

        best_idx  = np.argmax(k_tr / m_tr)
        print(f"  Step {step+1:>3}/{n_active}  "
              f"query=({x_new[0]:.3f}, {x_new[1]:.3f})  "
              f"k={k_new}/{m_new}  "
              f"π̂_max={pi_max:.3f}  "
              f"best so far=({X_tr[best_idx,0]:.3f}, {X_tr[best_idx,1]:.3f}) "
              f"k/m={k_tr[best_idx]/m_tr[best_idx]:.2f}")

    return history, X_tr, k_tr, m_tr


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load all settings from config ─────────────────────────────────────
    config    = load_yaml_file(constants.merged_config_path)
    bo        = config['bayesian_optimisation']
    gp        = bo['gp']

    USE_DUMMY   = bo['use_dummy']
    ACQ_TYPE    = bo['acq_type']
    KERNEL      = bo['kernel']
    BUDGET      = bo['budget']
    N_INIT      = bo['n_init']
    M_TRIALS    = bo['m_trials']
    LS          = gp['length_scale']
    VAR         = gp['variance']
    KAPPA       = gp['kappa']
    OPTIMISE_HP = bo['optimise_hp']
    HP_INTERVAL = bo['hp_interval']
    N_GRID      = bo['n_grid']
    SEED        = bo['seed']
    RESUME      = bo['resume']

    # ── Snapshot steps for Figure 5 plot (0-indexed into active steps) ───
    n_active = BUDGET - N_INIT
    SNAPS    = (0, n_active // 2, n_active - 1)   # early / mid / final

    # ── Output paths ──────────────────────────────────────────────────────
    os.makedirs('./outputs', exist_ok=True)
    mode_tag        = 'dummy' if USE_DUMMY else 'real'
    CHECKPOINT_PATH = f'./outputs/bo_checkpoint_{mode_tag}_{ACQ_TYPE}_Hyp_{OPTIMISE_HP}.csv'
    FIG5_PATH       = f'./outputs/bo_{mode_tag}_{ACQ_TYPE}_Hyp_{OPTIMISE_HP}.png'
    CONV_PATH       = f'./outputs/bo_{mode_tag}_{ACQ_TYPE}_Hyp_{OPTIMISE_HP}_convergence.png'

    # ── Select oracle based on flag ───────────────────────────────────────
    from functools import partial
    if USE_DUMMY:
        print("=" * 60)
        print("  MODE: DUMMY (synthetic true_pi_2d)")
        print("  YOLO pipeline will NOT be called.")
        print("=" * 60)
        oracle_fn = partial(oracle_dummy, m=M_TRIALS)
        true_fn   = true_pi_2d    # passed to plotting for reference column
    else:
        print("=" * 60)
        print("  MODE: PRODUCTION (real YOLO pipeline)")
        print("  train_and_evaluate() will be called each iteration.")
        print("=" * 60)
        oracle_fn = partial(oracle_yolo, m=M_TRIALS)
        true_fn   = None           # no ground truth available in production

    history=[]

    # ── Run ───────────────────────────────────────────────────────────────
    try:
        run_number=get_next_optimization_run_number(constants.metadata_file_path)
        history, X_tr, k_tr, m_tr = run_bo_2d(
            oracle_fn       = oracle_fn,
            acq_type        = ACQ_TYPE,
            n_init          = N_INIT,
            budget          = BUDGET,
            ls              = LS,
            var             = VAR,
            kappa           = KAPPA,
            n_grid          = N_GRID,
            kernel          = KERNEL,
            seed            = SEED,
            optimise_hp     = OPTIMISE_HP,
            hp_interval     = HP_INTERVAL,
            checkpoint_path = CHECKPOINT_PATH,
            resume          = RESUME,
        )

    except KeyboardInterrupt:
        print("\n  Interrupted — checkpoint saved, partial history available.")
        # history may be incomplete — plot whatever was collected
        if not history:
            print("  No history to plot. Exiting.")
            raise SystemExit(0)

    # ── Results summary ───────────────────────────────────────────────────
    best_idx  = np.argmax(k_tr / m_tr)
    best_x    = X_tr[best_idx]
    best_rate = k_tr[best_idx] / m_tr[best_idx]

    print("\n" + "=" * 60)
    print("  OPTIMISATION COMPLETE")
    print(f"  Best observed point : scale={best_x[0]:.4f}  hsv_v={best_x[1]:.4f}")
    print(f"  Best observed k/m   : {k_tr[best_idx]:.0f}/{m_tr[best_idx]:.0f}"
          f"  = {best_rate:.3f}")
    print(f"  Final π̂_max (model) : {history[-1]['pi_max']:.4f}")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")

    # Figure 5-style contour snapshots
    # true_fn controls whether reference column is shown
    plot_figure5(
        history    = history,
        acq_type   = ACQ_TYPE,
        save_path  = FIG5_PATH,
        snap_steps = SNAPS,
        true_fn    = true_fn,    # None in production, true_pi_2d in dummy
    )

    # Convergence plot — single run, wrap in dict for the function signature
    plot_convergence(
        histories  = {ACQ_TYPE: history},
        save_path  = CONV_PATH,
        kappa      = KAPPA,
        true_fn    = true_fn,    # None in production
    )

    print("\nDone.")
