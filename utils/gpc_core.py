"""
gpc_core.py
===========
All mathematical components for 2D Binomial-Probit Gaussian Process
Classification with Laplace approximation.

No plotting, no pipeline imports — pure numpy/scipy.

Contents
--------
  Kernels       : rbf_kernel_2d, matern52_kernel_2d, build_kernel
  Likelihood    : log_lik, grad_log_lik, hess_diag           (Binomial-Probit)
  Inference     : laplace                                     (Newton solver)
  Prediction    : predict_2d                                  (diagonal cov, efficient)
  Acquisition   : acq_EI_pi, acq_UCB
  Hyperparams   : log_marginal_lik, optimise_hyperparams      (grid search over ls, var)
"""

import numpy as np
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════════
# Kernels
# ═══════════════════════════════════════════════════════════════════════════

def rbf_kernel_2d(X1, X2, ls=0.20, var=2.0):
    """
    Isotropic RBF (squared-exponential) kernel for 2D inputs.

    k(x, x') = var · exp(−‖x−x'‖² / 2·ls²)

    Assumes infinite smoothness — appropriate when the latent function
    is analytic (e.g. built from Gaussians/polynomials).

    X1 : (n, 2),  X2 : (m, 2)  →  K : (n, m)
    """
    diff = X1[:, None, :] - X2[None, :, :]      # (n, m, 2)
    d2   = np.sum(diff ** 2, axis=-1)            # (n, m)
    return var * np.exp(-0.5 * d2 / ls ** 2)


def matern52_kernel_2d(X1, X2, ls=0.20, var=2.0):
    """
    Isotropic Matérn 5/2 kernel for 2D inputs.

    k(x, x') = var · (1 + √5·r + 5r²/3) · exp(−√5·r)
    where r = ‖x−x'‖ / ls

    Twice differentiable — the standard default for real blackbox functions.
    More robust than RBF when the function has occasional sharp changes.

    X1 : (n, 2),  X2 : (m, 2)  →  K : (n, m)
    """
    diff = X1[:, None, :] - X2[None, :, :]           # (n, m, 2)
    r    = np.sqrt(np.sum(diff ** 2, axis=-1)) / ls   # (n, m)
    r5   = np.sqrt(5.0) * r
    return var * (1.0 + r5 + r5 ** 2 / 3.0) * np.exp(-r5)


def build_kernel(X1, X2, ls=0.20, var=2.0, kernel='rbf'):
    """
    Single dispatch for kernel selection.

    Parameters
    ----------
    kernel : 'rbf' or 'matern52'
        'rbf'      → use when true function is known to be smooth/analytic
                     or for synthetic benchmarks
        'matern52' → default for real blackbox experiments

    Returns K : (n, m) kernel matrix
    """
    if kernel == 'rbf':
        return rbf_kernel_2d(X1, X2, ls, var)
    elif kernel == 'matern52':
        return matern52_kernel_2d(X1, X2, ls, var)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'rbf' or 'matern52'.")


# ═══════════════════════════════════════════════════════════════════════════
# Binomial-Probit likelihood
# ═══════════════════════════════════════════════════════════════════════════
#
# Each observation is (x_i, k_i, m_i):
#   m_i trials at x_i, k_i successes observed.
#   p(k_i | f_i) = C(m,k) · Φ(f_i)^k_i · (1−Φ(f_i))^(m_i−k_i)
#
# m=10, k = int(round(success_rate * 10))  ← your YOLO pipeline
# ═══════════════════════════════════════════════════════════════════════════

def log_lik(f, k, m):
    """
    Sum of Binomial log-likelihoods (C(m,k) constant dropped).
    Σ_i  k_i·log Φ(f_i) + (m_i−k_i)·log(1−Φ(f_i))
    """
    Phi = np.clip(norm.cdf(f), 1e-300, 1 - 1e-300)
    return np.sum(k * np.log(Phi) + (m - k) * np.log(1.0 - Phi))


def grad_log_lik(f, k, m):
    """
    ∂/∂f_i  log p(k_i|f_i) = φ(f_i) · [k_i/Φ(f_i) − (m_i−k_i)/(1−Φ(f_i))]
    """
    Phi = np.clip(norm.cdf(f), 1e-300, 1 - 1e-300)
    phi = norm.pdf(f)
    return phi * (k / Phi - (m - k) / (1.0 - Phi))


def hess_diag(f, k, m):
    """
    W_ii = −∂²/∂f_i²  log p(k_i|f_i)   (positive, used in Newton step)

    = φ²·[k/Φ² + (m−k)/(1−Φ)²]  +  f·φ·[k/Φ − (m−k)/(1−Φ)]

    Second term comes from φ′(f) = −f·φ(f).
    """
    Phi  = np.clip(norm.cdf(f), 1e-300, 1 - 1e-300)
    phi  = norm.pdf(f)
    t1   = phi ** 2 * (k / Phi ** 2 + (m - k) / (1.0 - Phi) ** 2)
    t2   = phi * f   * (k / Phi     - (m - k) / (1.0 - Phi))
    return np.maximum(t1 + t2, 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Laplace approximation
# ═══════════════════════════════════════════════════════════════════════════

def laplace(K, k, m, n_iter=30, tol=1e-9):
    """
    Laplace approximation of the Binomial-Probit posterior.

    Finds f_MAP via Newton iterations on:
        log p(f|k,m) ∝ log p(k|f,m) + log p(f)   where p(f) = N(0, K)

    Then approximates posterior as N(f_MAP, Σ) where:
        Σ = (K⁻¹ + W)⁻¹  computed via Cholesky of  B = I + W^{1/2} K W^{1/2}

    Parameters
    ----------
    K : (n, n)   kernel matrix (with jitter already added on diagonal)
    k : (n,)     success counts at each training point
    m : (n,)     trial counts at each training point  (all 10 for YOLO)

    Returns
    -------
    f_map : (n,)    MAP estimate of latent function
    Sigma : (n, n)  posterior covariance
    """
    n = len(k)
    f = np.zeros(n)

    for _ in range(n_iter):
        W     = hess_diag(f, k, m)
        g     = grad_log_lik(f, k, m)
        sW    = np.sqrt(W)
        B     = np.eye(n) + (sW[:, None] * K) * sW[None, :]
        L     = np.linalg.cholesky(B + 1e-8 * np.eye(n))
        b     = W * f + g
        v     = np.linalg.solve(L, sW * (K @ b))
        f_new = K @ (b - sW * np.linalg.solve(L.T, v))
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    # Posterior covariance  Σ = K − V^T V
    W   = hess_diag(f, k, m)
    sW  = np.sqrt(W)
    B   = np.eye(n) + (sW[:, None] * K) * sW[None, :]
    L   = np.linalg.cholesky(B + 1e-8 * np.eye(n))
    V   = np.linalg.solve(L, sW[:, None] * K)
    return f, K - V.T @ V


# ═══════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════

def predict_2d(X_tr, f_map, Sigma, K_tr, X_te, ls=0.20, var=2.0, kernel='rbf'):
    """
    Predictive distribution at test locations X_te.

    Computes only the diagonal of the predictive covariance — O(m·n²)
    instead of O(m²) for the full matrix. Avoids allocating a 3600×3600
    matrix when predicting on the 60×60 plotting grid.

    Returns
    -------
    mu     : (m,)  posterior predictive mean  of latent f
    sigma  : (m,)  posterior predictive std   of latent f
    pi_bar : (m,)  moment-matched probability  Φ(μ/√(1+σ²))
    """
    K_s      = build_kernel(X_te, X_tr, ls, var, kernel)       # (m, n)
    k_ss_diag = build_kernel(X_te, X_te, ls, var, kernel)      # only need diag below

    K_inv = np.linalg.inv(K_tr + 1e-8 * np.eye(len(f_map)))
    mu    = K_s @ (K_inv @ f_map)

    # Efficient diagonal of predictive covariance
    # Var[f*] = k(x*,x*) − k_s (K⁻¹ − K⁻¹ Σ K⁻¹) k_s^T   → diagonal only
    A      = K_inv @ K_s.T                       # (n, m)
    B      = K_inv @ Sigma @ K_inv @ K_s.T       # (n, m)
    k_diag = np.diag(k_ss_diag)                  # (m,)  diagonal of K(X_te, X_te)
    sig2   = k_diag - np.sum(K_s * (A - B).T, axis=1)
    sigma  = np.sqrt(np.maximum(sig2, 1e-10))

    pi_bar = norm.cdf(mu / np.sqrt(1.0 + sigma ** 2))
    return mu, sigma, pi_bar


# ═══════════════════════════════════════════════════════════════════════════
# Acquisition functions
# ═══════════════════════════════════════════════════════════════════════════

def acq_EI_pi(mu, sigma, pi_max, n_gh=20):
    """
    EI_π : Expected Improvement in probability space (Tesch et al., 2013).

    EI_π(x) = ∫ max(Φ(z) − π̂_max, 0) · N(z; μ, σ²) dz

    Evaluated via Gauss-Hermite quadrature.
    Self-suppresses naturally as π̂_max → 1 near the optimum.

    Best choice when:
      - Noise is low (large m_trials)
      - Budget is sufficient for global exploration first
      - Function is unimodal or you want the best final recommendation
    """
    t, w  = np.polynomial.hermite.hermgauss(n_gh)
    # z = mu + sigma * sqrt(2) * t  →  N(mu, sigma²) quadrature nodes
    z     = mu[:, None] + sigma[:, None] * np.sqrt(2.0) * t[None, :]   # (n_cand, n_gh)
    improv = np.maximum(norm.cdf(z) - pi_max, 0.0)
    return (improv @ w) / np.sqrt(np.pi)


def acq_UCB(mu, sigma, kappa=1.5):
    """
    UCB : Upper Confidence Bound on latent f.

    UCB(x) = μ(x) + κ·σ(x)

    κ controls exploration-exploitation tradeoff:
      κ = 0   → pure exploitation (argmax μ)
      κ → ∞   → pure exploration  (argmax σ)

    Recommended κ:
      κ = 1.5  for budget ≈ 30, smooth function, moderate noise  (your setup)
      κ = 2.0  for more exploratory behaviour or multimodal functions
      κ = 0.5  for final exploitation phase with small remaining budget

    Best choice when:
      - Noise is high (small m_trials)
      - Budget is small relative to dimensionality
      - Function is multimodal (keeps exploring globally)
    """
    return np.maximum(mu + kappa * sigma,0)


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter optimisation  (marginal likelihood)
# ═══════════════════════════════════════════════════════════════════════════

def log_marginal_lik(ls, var, X_tr, k_tr, m_tr, kernel='rbf'):
    """
    Laplace approximation to log p(k | m, ls, var):

    log p ≈ log p(k|f_MAP,m)
           − ½ f_MAP^T K⁻¹ f_MAP
           − ½ log|K|
           − ½ log|I + W^{1/2} K W^{1/2}|

    Higher = better fit of hyperparameters to observed data.
    Used by optimise_hyperparams to tune ls and var automatically.
    """
    n = len(k_tr)
    K = build_kernel(X_tr, X_tr, ls, var, kernel) + 1e-6 * np.eye(n)

    try:
        f_map, _ = laplace(K, k_tr, m_tr)
    except np.linalg.LinAlgError:
        return -np.inf

    # Log likelihood at MAP
    Phi = np.clip(norm.cdf(f_map), 1e-300, 1 - 1e-300)
    ll  = np.sum(k_tr * np.log(Phi) + (m_tr - k_tr) * np.log(1.0 - Phi))

    # GP log prior (up to constant)
    K_inv = np.linalg.inv(K)
    lp    = -0.5 * f_map @ K_inv @ f_map
    _, ld = np.linalg.slogdet(K)
    lp   -= 0.5 * ld

    # Laplace correction  −½ log|I + W^{1/2} K W^{1/2}|
    W  = hess_diag(f_map, k_tr, m_tr)
    sW = np.sqrt(W)
    B  = np.eye(n) + (sW[:, None] * K) * sW[None, :]
    _, ld_B = np.linalg.slogdet(B)

    return ll + lp - 0.5 * ld_B


def optimise_hyperparams(X_tr, k_tr, m_tr, kernel='rbf',
                          ls_grid=None, var_grid=None):
    """
    Grid search for (ls, var) that maximises the Laplace marginal likelihood.

    Called every `hyperparam_interval` steps in the BO loop.
    Cheap relative to the real experiment — 25 evaluations of a
    closed-form formula.

    Parameters
    ----------
    ls_grid  : array of length-scale candidates  (default: 5 log-spaced in [0.05, 0.5])
    var_grid : array of variance candidates       (default: 5 log-spaced in [0.3, 4.0])

    Returns
    -------
    ls_opt  : float  optimal length-scale
    var_opt : float  optimal output variance
    lml_opt : float  log marginal likelihood at optimum (for logging)
    """
    if ls_grid is None:
        ls_grid  = np.exp(np.linspace(np.log(0.05), np.log(0.50), 5))
    if var_grid is None:
        var_grid = np.exp(np.linspace(np.log(0.30), np.log(4.00), 5))

    best_lml = -np.inf
    ls_opt, var_opt = ls_grid[2], var_grid[2]   # centre as fallback

    for ls in ls_grid:
        for var in var_grid:
            lml = log_marginal_lik(ls, var, X_tr, k_tr, m_tr, kernel)
            if lml > best_lml:
                best_lml = lml
                ls_opt   = ls
                var_opt  = var

    return ls_opt, var_opt, best_lml
