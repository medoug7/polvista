"""Least-squares QU fitting for the Faraday-depolarization models in
:mod:`polvista.models`.

Distilled from the `qu_fit()` pipeline in `QU_fitting and model comparison.ipynb`:
just its single `scipy.optimize.least_squares` call (soft-L1 loss, whitened by
each point's q/u sigma) from one initial guess -- the notebook's multi-start
retries and MCMC refinement are not reproduced here.

Each model's EVPA (X-kind) parameters are period-pi angles, but
`scipy.optimize.least_squares` only knows how to do plain bounded-linear
search -- fit directly on X and it's boxed to e.g. (-pi/2, pi/2], with a hard
seam at the boundary that coincides with exactly where posterior/likelihood
mass tends to sit (near +-90 deg). A bounded optimizer can get stuck hard
against that edge instead of continuing "around" it. `qu_fit.py`'s MultiNest
pipeline sidesteps the analogous problem for its nested-sampling prior by
moving the unit-cube's own seam away from +-pi/2 (see its `x_from_cube()`).
`least_squares` has no unit cube to reparametrize, so the fix here is
different in mechanism but the same idea applied one level up: for each
free (p, X) pair -- amplitude and EVPA always appear together in every model
here -- the optimizer is handed the unconstrained Cartesian pair
(a, b) = (p*cos(2X), p*sin(2X)) instead, i.e. it fits the complex amplitude
p*e^(2iX) directly (see `to_cartesian`/`from_cartesian`). Going all the way
around the EVPA circle is then just walking around the origin in (a, b)
space -- there's no seam for the optimizer to get stuck against, and
p = hypot(a, b) >= 0 comes out automatically. The model function itself is
never touched: (a, b) is converted back to physical (p, X) before every
`model(wl, params)` call.
"""
import numpy as np
from scipy.optimize import least_squares

from polvista.models import MODELS, comp2RMdep, comp2intern, comp2RMdep_classic, comp2intern_classic, intensity_shape

EPS = 1e-10


def weighted_residuals(params, wl, y_obs, sigma, model):
    """Real/imag residuals of `model(wl, params)` against complex q+i*u data
    `y_obs`, whitened by the corresponding real/imag entries of `sigma`."""
    y_model = model(wl, params)
    dy = y_model - y_obs
    res_real = dy.real / (sigma.real + EPS)
    res_imag = dy.imag / (sigma.imag + EPS)
    return np.concatenate([res_real, res_imag])


def pX_free_pairs(spec, fixed):
    """(p_idx, X_idx) index pairs from `spec` -- every model pairs each
    amplitude ('p'-kind) with the EVPA ('X'-kind) it multiplies -- that are
    eligible for the unconstrained-Cartesian reparametrization (see module
    docstring), i.e. neither half is held fixed. A pair with only one half
    fixed falls back to fitting the other half directly in physical,
    bounded-linear form: the Cartesian trick only makes sense when both
    halves of a pair are actually free to move together."""
    return [(pi, xi) for pi, xi in zip(spec.indices('p'), spec.indices('X'))
            if pi not in fixed and xi not in fixed]


def to_cartesian(params, pX_pairs):
    """Physical (p, X) -> unconstrained Cartesian (a, b) for each pair in
    `pX_pairs`: a = p*cos(2X), b = p*sin(2X). Every other slot passes
    through unchanged. Returns a new list."""
    params = list(params)
    for pi, xi in pX_pairs:
        p, X = params[pi], params[xi]
        params[pi], params[xi] = p * np.cos(2 * X), p * np.sin(2 * X)
    return params


def from_cartesian(params, pX_pairs):
    """Inverse of `to_cartesian`: (a, b) -> (p, X) = (hypot(a, b),
    0.5*atan2(b, a)), which lands X already wrapped to (-pi/2, pi/2].
    Mutates and returns `params`."""
    for pi, xi in pX_pairs:
        a, b = params[pi], params[xi]
        params[pi], params[xi] = float(np.hypot(a, b)), 0.5 * float(np.arctan2(b, a))
    return params


def qu_fit(wl, q, q_err, u, u_err, model, param_init, bounds, fixed=None):
    """Fit `model`'s parameters to fractional Stokes q=Q/I, u=U/I data via
    weighted nonlinear least squares.

    `fixed` is an optional {param_index: value} map of parameter slots to
    hold constant (e.g. the spectral epsilon/alpha params, which QU-only
    data can't constrain) -- only the remaining slots are actually passed
    to the optimizer.

    Every free (amplitude, EVPA) pair is fit in an unconstrained Cartesian
    reparametrization rather than directly on the physical, bounded-linear
    parameters -- see the module docstring for why. Each Cartesian
    component is boxed to [-p_max, p_max] (p_max = that pair's own upper
    amplitude bound) -- a looser cap than the physical disk |p|<=p_max
    (its corners reach p=p_max*sqrt(2)), but least_squares only needs a
    sane box, not a tight one, and this fully removes the EVPA boundary
    problem instead of just relocating it.

    Returns (best_pars, result): `best_pars` is the full parameter list
    (fixed slots included, at their fixed value; reparametrized pairs
    converted back to physical (p, X)) in the same order as `param_init`;
    `result` is the underlying scipy `OptimizeResult` (its `.x` is in the
    optimizer's own, partly-Cartesian units -- only used elsewhere for its
    length/convergence/nfev, never its values).
    """
    fixed = fixed or {}
    spec = MODELS[model]
    pX_pairs = pX_free_pairs(spec, fixed)

    free_idx = [i for i in range(len(param_init)) if i not in fixed]

    def full_params(free_vals):
        pars = list(param_init)
        for i, v in zip(free_idx, free_vals):
            pars[i] = v
        for i, v in fixed.items():
            pars[i] = v
        return pars

    def residuals(free_vals, wl, y_obs, sigma, model):
        pars = from_cartesian(full_params(free_vals), pX_pairs)
        return weighted_residuals(pars, wl, y_obs, sigma, model)

    param_init_cart = to_cartesian(param_init, pX_pairs)
    p0 = [param_init_cart[i] for i in free_idx]
    lo = [bounds[0][i] for i in free_idx]
    hi = [bounds[1][i] for i in free_idx]
    for pi, xi in pX_pairs:
        p_max = bounds[1][pi]
        for i in (pi, xi):
            j = free_idx.index(i)
            lo[j], hi[j] = -p_max, p_max

    y_obs = np.asarray(q) + 1j * np.asarray(u)
    sigma = np.asarray(q_err) + 1j * np.asarray(u_err)

    result = least_squares(
        residuals, p0, args=(wl, y_obs, sigma, model),
        bounds=(lo, hi), loss='soft_l1', jac='2-point', x_scale='jac',
        ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=10000,
    )

    return from_cartesian(full_params(result.x), pX_pairs), result


def fit_statistics(best_pars, result, wl, q, q_err, u, u_err, model):
    """Goodness-of-fit summary for a qu_fit() result: chi-squared (raw,
    reduced), the Gaussian log-likelihood of the whitened q/u residuals,
    AIC/AICc/BIC, and the optimizer's own convergence report.

    ll uses the standard independent-Gaussian normalization ln(2*pi*sigma_q*
    sigma_u) for the same residuals qu_fit() minimizes (weighted_residuals)
    -- the same normalization (log_norm_const, defined below) the Bayesian
    (MultiNest) path's own bayesian_loglike() uses, so ln L/AIC/BIC values
    are directly comparable between the two fitting schemes for the same
    data and model. Unlike the Bayesian path's loglike(), this does not add
    phi/dphi alias soft priors -- those regularize MultiNest's sampling,
    not this least-squares objective.
    """
    y_obs = np.asarray(q) + 1j * np.asarray(u)
    sigma = np.asarray(q_err) + 1j * np.asarray(u_err)
    wres = weighted_residuals(best_pars, wl, y_obs, sigma, model)
    chi2 = float(np.sum(wres**2))

    n = 2 * len(wl)          # real+imag residuals count as separate data points
    k = len(result.x)        # free (non-fixed) parameters actually optimized
    dof = n - k

    loglike = -0.5 * chi2 - log_norm_const(sigma)

    aic = 2 * k - 2 * loglike
    aicc = aic + (2 * k * (k + 1)) / (dof - 1) if dof > 1 else float('nan')
    bic = k * np.log(n) - 2 * loglike

    return {
        'chi2': chi2, 'dof': dof,
        'chi2_red': chi2 / dof if dof > 0 else float('nan'),
        'loglike': loglike, 'aic': aic, 'aicc': aicc, 'bic': bic,
        'n_free': k, 'n_data': n,
        'converged': bool(result.success), 'nfev': int(result.nfev),
    }


def estimate_alpha(freq, I):
    """Least-squares power-law index of I(nu) ~ (nu/nu_min)**alpha, anchored
    through I(nu_min)=1 exactly (no free intercept) to match the models' own
    normalization convention -- so this only estimates the *shape*, not a
    separate amplitude. `freq` and `I` need not be sorted. Returns 0.0 if
    every point is at the same frequency (slope undefined).

    QU-only fitting can't constrain alpha (it has no effect on Q/I, U/I --
    see stokes_I()'s docstring), so this is a direct closed-form regression
    against the loaded I(nu) data instead, used only to make the displayed
    Stokes I/Q/U curve track the real intensity spectrum; see
    MainWindow.run_fit."""
    freq = np.asarray(freq)
    I = np.asarray(I)
    i0 = np.argmin(freq)
    x = np.log(freq / freq[i0])
    y = np.log(I / I[i0])
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def estimate_ssa_shape(freq, I, nu0_init, alpha_init, nu0_bounds, fit_nu0):
    """Nonlinear regression of loaded I(nu) data against the SSA normalized
    intensity shape I'(nu; nu0, alpha, 'ssa') (models.intensity_shape -- the
    caller must only invoke this for a single-component model whose own
    Spectrum-box shape is 'ssa', see models.set_spectral_shape), anchored
    the same way estimate_alpha
    is: log(I/I[i0]) fit against log(I'/I'[i0]) at the lowest-frequency
    point i0, no free intercept.

    Unlike a plain power law, alpha appears inside an exponential under
    SSA (see intensity_shape), so there's no closed form -- this is a
    genuine nonlinear least-squares solve (scipy `least_squares`, same
    machinery `qu_fit` itself uses).

    If `fit_nu0` is False, nu0 stays pinned at `nu0_init` and only alpha is
    optimized (1-D) -- this is the default (see
    MainWindow.build_nu0_slider): a fit starts out assuming there's no
    turnover in view, since the nu_0 sliders themselves default to fixed,
    deep in the optically-thin regime. Unchecking a nu_0 slider's "fixed"
    box switches this to a joint 2-D solve for (nu0, alpha) together,
    seeded from (nu0_init, alpha_init) with nu0 boxed to `nu0_bounds` --
    both must share freq's own frequency unit (only ratios matter, so any
    consistent unit works; MainWindow uses Hz throughout).

    Returns (alpha_est, nu0_est) -- nu0_est is just nu0_init echoed back
    when `fit_nu0` is False. Falls back to (0.0, nu0_init) if every point
    is at the same frequency (shape undefined), or to (alpha_init,
    nu0_init) if the solve itself doesn't converge."""
    freq = np.asarray(freq)
    I = np.asarray(I)
    i0 = np.argmin(freq)
    if np.all(freq == freq[i0]):
        return 0.0, nu0_init
    target = np.log(I / I[i0])

    def shape_log_ratio(nu0, alpha):
        Iprime = intensity_shape(freq, nu0, alpha, 'ssa')
        return np.log(Iprime / Iprime[i0])

    if not fit_nu0:
        result = least_squares(lambda p: shape_log_ratio(nu0_init, p[0]) - target, x0=[alpha_init])
        alpha_est = float(result.x[0]) if result.success else alpha_init
        return alpha_est, nu0_init

    lo_nu0, hi_nu0 = nu0_bounds
    result = least_squares(
        lambda p: shape_log_ratio(p[0], p[1]) - target,
        x0=[nu0_init, alpha_init], bounds=([lo_nu0, -np.inf], [hi_nu0, np.inf]))
    if not result.success:
        return alpha_init, nu0_init
    nu0_est, alpha_est = float(result.x[0]), float(result.x[1])
    return alpha_est, nu0_est


def estimate_shape_2comp(freq, I, eps, nu0_1_init, nu0_2_init, alpha_init,
                          nu0_bounds, fit_nu0_1, fit_nu0_2, shape1, shape2,
                          T1=None, T2=None, beta1=None, beta2=None):
    """Two-component counterpart of estimate_ssa_shape: nonlinear
    regression of loaded I(nu) data against a shared-alpha blend
    eps*I'(nu; nu0_1, alpha, shape1) + (1-eps)*I'(nu; nu0_2, alpha, shape2)
    (see intensity_shape) -- each component in its own shape, which need
    not match the other's (a 'powerlaw' component and an 'ssa' component
    can be blended together) -- anchored the same log(I/I[i0]) way. alpha
    is always shared between both components -- QU-only fitting has only
    one real I(nu) dataset to anchor it to, same reasoning
    FIT_FIXED_EPSILON already relies on (see MainWindow.fit_spectrum_lsq)
    -- but nu0_1 and nu0_2 are each independently fit or held pinned at
    their own *_init, per fit_nu0_1/fit_nu0_2 (MainWindow.nu0_slider_1/2's
    own fixed checkboxes -- always False for a component whose own shape
    is 'powerlaw', since its reference frequency is always the shared band
    edge, never a free parameter), rather than forced equal: with both
    pinned, only alpha is optimized (1-D); with exactly one free, that one
    nu0 and alpha are jointly solved (2-D); with both free, all three go
    into one joint solve (3-D).

    `T1`/`T2` are each component's own fixed electron temperature [K],
    required (non-None) only when that component's own shape is 'thermal'
    -- unlike nu0_1/nu0_2, T is never fit here (MainWindow.temp_slider_1/2
    are always taken as-is), just passed through to intensity_shape.
    `beta1`/`beta2` are likewise each component's own fixed curvature
    index, required only when that component's own shape is 'logparabola'
    (MainWindow.beta_slider_1/2), also never fit here.

    Returns (alpha_est, nu0_1_est, nu0_2_est) -- a pinned nu0 is just its
    own *_init echoed back. Falls back to all *_init values if every point
    is at the same frequency (shape undefined) or the solve doesn't
    converge."""
    freq = np.asarray(freq)
    I = np.asarray(I)
    i0 = np.argmin(freq)
    if np.all(freq == freq[i0]):
        return 0.0, nu0_1_init, nu0_2_init
    target = np.log(I / I[i0])

    def shape_log_ratio(nu0_1, nu0_2, alpha):
        Iprime = (eps * intensity_shape(freq, nu0_1, alpha, shape1, T=T1, beta=beta1)
                  + (1.0 - eps) * intensity_shape(freq, nu0_2, alpha, shape2, T=T2, beta=beta2))
        return np.log(Iprime / Iprime[i0])

    lo_nu0, hi_nu0 = nu0_bounds

    # Build the free-parameter vector dynamically: whichever of nu0_1/nu0_2
    # is free gets its own slot (boxed to nu0_bounds); alpha is always free.
    free_names, x0, lo, hi = [], [], [], []
    if fit_nu0_1:
        free_names.append('nu0_1'); x0.append(nu0_1_init); lo.append(lo_nu0); hi.append(hi_nu0)
    if fit_nu0_2:
        free_names.append('nu0_2'); x0.append(nu0_2_init); lo.append(lo_nu0); hi.append(hi_nu0)
    free_names.append('alpha')
    x0.append(alpha_init)
    lo.append(-np.inf)
    hi.append(np.inf)

    def residual(p):
        values = dict(zip(free_names, p))
        return shape_log_ratio(values.get('nu0_1', nu0_1_init),
                                values.get('nu0_2', nu0_2_init),
                                values['alpha']) - target

    result = least_squares(residual, x0=x0, bounds=(lo, hi))
    if not result.success:
        return alpha_init, nu0_1_init, nu0_2_init
    values = dict(zip(free_names, result.x))
    alpha_est = float(values['alpha'])
    nu0_1_est = float(values['nu0_1']) if fit_nu0_1 else nu0_1_init
    nu0_2_est = float(values['nu0_2']) if fit_nu0_2 else nu0_2_init
    return alpha_est, nu0_1_est, nu0_2_est


# ── Bayesian (MultiNest) fitting ─────────────────────────────────────────────
# Ported, down-scoped from ~/Downloads/pipe/qu_fit.py's MultiNest pipeline:
# the unit-cube prior reparametrization (X's domain seam moved off +-pi/2,
# phi/dphi optionally log-uniform), the pymultinest.run() call itself, and
# post-sampling mode-family clustering (union-find over raw MultiNest modes,
# picking the family whose own marginal-median point best explains the
# data). Deliberately dropped relative to that pipeline: the phi/dphi
# "alias" soft-priors (barely move the likelihood here) and any
# per-parameter fixed/free bookkeeping -- MultiNest below always samples
# every non-spectral-kind parameter of the model; only the model's
# `alpha`/`eps`-kind spectral params are excluded (see `pol_idx`), fed in
# already-fit from a least-squares pre-fit (see
# MainWindow.run_multinest_fit in app.py). Single-process only: no
# MPI/joblib/ProcessPoolExecutor anywhere -- polvista fits one model at a
# time.
import os
import sys
import contextlib
import threading

from scipy.special import logsumexp

# Models with an exact 1<->2 component exchange symmetry: swapping
# (p1,X1,phi1,dphi1) <-> (p2,X2,phi2,dphi2) leaves the model function
# unchanged, so their raw posterior is doubly degenerate ("label
# switching") -- every genuine solution appears twice, once under each
# labeling. comp2mixdep (and comp2mixdep_classic) is excluded: its two
# components use different functional forms (one internal, one external
# screen), so they're not exchangeable and have no such degeneracy.
# comp2RMdep_classic/comp2intern_classic have the same swap symmetry as
# comp2RMdep/comp2intern themselves
# (classic_sum_combine's plain term1+term2 is as swap-invariant as
# spectral_combine's own blend) -- see models.comp2RMdep_classic.
DEGENERATE_PAIR_MODELS = {comp2RMdep, comp2intern, comp2RMdep_classic, comp2intern_classic}

# Non-winning families below this evidence_share (%) are dropped from
# `other_families` entirely (folded into `dropped` instead) -- see
# best_family().
MIN_FAMILY_EVIDENCE_SHARE = 1.0


def pol_idx(spec):
    """Indices of `spec`'s params that MultiNest actually samples -- every
    kind except the spectral 'alpha'/'eps' ones (QU-only data can't
    constrain those; they're supplied already-fit, see multinest_fit)."""
    return [i for i, p in enumerate(spec.params) if p.kind not in ('alpha', 'eps')]


def log_norm_const(sigma_complex):
    """Sum[log(2*pi * sigma_qi * sigma_ui)] normalization for the
    independent-Gaussian QU likelihood: each (q_j, u_j) pair is two
    independent real Gaussians, so their joint normalization is
    (2*pi*sigma_qj*sigma_uj)^-1, the standard bivariate-uncorrelated-
    Gaussian constant -- constant in the fit parameters, but needed for
    ln L (and the family-lnZ score it feeds) to be an absolute, comparable
    quantity rather than just something to minimize. Shared by both fitting
    paths (fit_statistics()'s least-squares ln L and bayesian_loglike()'s
    MultiNest ln L both call this), so their reported ln L/AIC/BIC are on
    the same absolute scale and can be compared directly."""
    sigma_real = sigma_complex.real + EPS
    sigma_imag = sigma_complex.imag + EPS
    return np.sum(np.log(2 * np.pi * sigma_real * sigma_imag))


def bayesian_loglike(model, pars, wl, y, y_errs):
    """Plain Gaussian QU log-likelihood of `model(wl, pars)` against
    complex data `y` (=q+i*u) with complex errors `y_errs` -- the same
    likelihood multinest_fit()'s own loglike() optimizes, so this is what
    best_family() uses to score each candidate mode-family's winner."""
    wres = weighted_residuals(pars, wl, y, y_errs, model)
    return -0.5 * float(np.sum(wres ** 2)) - float(log_norm_const(y_errs))


# ── Circular (period-pi) EVPA statistics ───────────────────────────────────────
# X (EVPA) is physically periodic with period pi (a polarization line has no
# arrowhead), but a plain bounded-linear prior draws it on (-pi/2, pi/2]. When
# posterior mass sits near +-90 deg, that coincides with the sampler's own hard
# box edge, and MultiNest's ellipsoidal clustering sees the two sides of the
# wrap as maximally far apart -- producing a mode truncated/inflated hard
# against the boundary. x_from_cube() (below) fixes the draw itself; these
# three provide circular-safe distance/median/percentile statistics for
# consuming the resulting samples.
def wrap_pm_halfpi(x):
    """Wrap an angle (radians) into (-pi/2, pi/2], respecting pi-periodicity."""
    return ((x + np.pi / 2) % np.pi) - np.pi / 2


def circular_delta(xi, xj):
    """Shortest signed difference xi-xj for a period-pi angular quantity,
    magnitude <= pi/2. Use for X-kind parameters in distance metrics."""
    return wrap_pm_halfpi(xi - xj)


def circular_median_pctl(X_samples, prob=0.68):
    """Marginal median + asymmetric (lo, hi) prob-mass errors for a
    period-pi circular quantity (EVPA), robust to wraparound at +-pi/2.

    Method: work in theta=2*X space (period 2*pi, ordinary circular stats
    apply), find the circular mean direction, shift all samples so that
    direction sits at 0, take ordinary median/percentile on the shifted
    values, then map the *position* back through the shift+halving, and
    take half-widths directly from the shifted (still-linear, un-wrapped)
    interval."""
    theta = 2.0 * np.asarray(X_samples)
    mean_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    shifted = ((theta - mean_theta + np.pi) % (2 * np.pi)) - np.pi

    med_s = np.median(shifted)
    lo_q, hi_q = (1 - prob) / 2 * 100, (1 + prob) / 2 * 100
    lo_s, hi_s = np.percentile(shifted, [lo_q, hi_q])

    X_median = wrap_pm_halfpi((mean_theta + med_s) / 2.0)
    err_lo = (med_s - lo_s) / 2.0
    err_hi = (hi_s - med_s) / 2.0
    return X_median, err_lo, err_hi


def median_pctl_errs(samples, prob=0.68):
    """Point estimate + asymmetric error bars for a posterior sample array
    -- marginal median per column plus the equal-tailed `prob`-mass
    interval around it. Non-circular; X-kind columns are overridden with
    circular_median_pctl() by callers."""
    pars = np.median(samples, axis=0)
    ndim = samples.shape[1]
    lo_q, hi_q = (1 - prob) / 2 * 100, (1 + prob) / 2 * 100
    errs = np.zeros((2, ndim))
    for i in range(ndim):
        lo, hi = np.percentile(samples[:, i], [lo_q, hi_q])
        errs[0, i] = pars[i] - lo
        errs[1, i] = hi - pars[i]
    return pars, errs


# ── Mode-family clustering (post-sampling, MultiNest-only) ────────────────────
# All of the functions below operate in "pol_idx-local" index space -- i.e.
# `kinds` here is `kinds_pol` (the kind of each MultiNest-sampled parameter,
# in pol_idx order), not the model's full parameter list.
def mode_id_indices(kinds):
    """Indices (local to `kinds`) of the 'identifying' params (X, phi,
    dphi) -- the ones used to decide whether two raw MultiNest modes are
    the same family. Excludes amplitude (p) and shape (scale) params,
    which can differ between otherwise-identical solutions without
    changing the physical identity of the mode."""
    idx = [i for i, k in enumerate(kinds) if k in ('X', 'phi')]
    idx += [i for i, k in enumerate(kinds) if k == 'dphi']
    return sorted(idx)


def swap_perm(kinds):
    """Index permutation (local to `kinds`) mapping each param to its
    label-swapped counterpart, for models with an exact 1<->2
    component-swap symmetry (DEGENERATE_PAIR_MODELS). Identity for any
    kind that doesn't appear exactly twice (shouldn't happen for
    DEGENERATE_PAIR_MODELS members)."""
    ndim = len(kinds)
    perm = np.arange(ndim)
    for kind in ('p', 'X', 'phi', 'dphi'):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        if len(idx) == 2:
            perm[idx[0]], perm[idx[1]] = idx[1], idx[0]
    return perm


def mode_summary(pars_raw, weight, logz, kinds, rng):
    """Resample a raw MultiNest mode's samples (already in pol_idx-local
    order) by its own weight column, and compute a circular-safe
    mean/sigma per parameter."""
    idx = rng.choice(len(weight), size=5000, replace=True, p=weight / weight.sum())
    resampled = pars_raw[idx]
    mean, errs = median_pctl_errs(resampled)
    for j, k in enumerate(kinds):
        if k == 'X':
            med, elo, ehi = circular_median_pctl(resampled[:, j])
            mean[j] = med
            errs[0, j] = elo
            errs[1, j] = ehi
    sigma = (errs[0] + errs[1]) / 2.0
    return dict(weight=weight, pars_raw=pars_raw, logz=logz, mean=mean, sigma=sigma)


def id_param_distance(mode_i, mean_j, sigma_j, id_idx, kinds):
    """Normalized chi-like distance from `mode_i` to a (mean, sigma) pair
    over `id_idx` -- circular delta for X-kind, linear for phi/dphi-kind."""
    diffs, sig = [], []
    for k in id_idx:
        d = (circular_delta(mode_i['mean'][k], mean_j[k]) if kinds[k] == 'X'
             else (mode_i['mean'][k] - mean_j[k]))
        diffs.append(d)
        sig.append(np.sqrt(mode_i['sigma'][k] ** 2 + sigma_j[k] ** 2))
    diffs = np.array(diffs)
    sig = np.where(np.array(sig) < 1e-12, 1e-12, sig)
    return float(np.sqrt(np.sum((diffs / sig) ** 2)))


def id_param_distance_rows(samples, ref, ref_sigma, id_idx, kinds):
    """Vectorized id_param_distance(): per-row normalized chi-like distance
    from every row of `samples` (N, ndim, pol_idx-local order) to a single
    (ref, ref_sigma) point, over `id_idx` -- circular delta for X-kind,
    linear otherwise. Building block for relabel_swap_symmetric()'s
    per-sample direct-vs-swapped assignment."""
    total = np.zeros(len(samples))
    for k in id_idx:
        d = (circular_delta(samples[:, k], ref[k]) if kinds[k] == 'X'
             else (samples[:, k] - ref[k]))
        sig = ref_sigma[k] if ref_sigma[k] > 1e-12 else 1e-12
        total += (d / sig) ** 2
    return np.sqrt(total)


def circular_mean_pm_halfpi(x):
    """Circular mean of period-pi angles (radians), wrapped into
    (-pi/2, pi/2] -- see wrap_pm_halfpi/circular_delta. Unlike a linear
    mean, well-defined even for a sample split across the wrap seam."""
    theta = 2.0 * np.asarray(x)
    mean_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    return wrap_pm_halfpi(mean_theta / 2.0)


def relabel_swap_symmetric(pooled, id_idx, kinds, swap_perm, rng, n_iter=10, n_restarts=20):
    """Relabel `pooled` (N, ndim, pol_idx-local order) onto one consistent
    labeling for a DEGENERATE_PAIR_MODELS family, swapping (via swap_perm)
    whichever rows need it. Returns the relabeled (N, ndim) array.

    Needed because MultiNest's own mode-finding isn't symmetry-aware: it
    can lump both label-swapped branches of an exact component-swap-
    degenerate posterior into one raw mode, so `pooled` may itself be an
    even direct/swapped mix rather than one consistent labeling.

    Uses constrained Lloyd's-algorithm clustering (k=2, one swap-symmetric
    pair of clusters): each pass reassigns every row to whichever of a
    running direct/swapped centroid it's closer to (by normalized
    per-parameter distance over the "identifying" params, mode_id_indices()),
    then refits the centroid from its own newly-assigned rows, denoising
    the same way a whole-mode weighted mean does but from a self-
    consistently-labeled subset. Since this only converges to *a* fixed
    point, not necessarily the best one, `n_restarts` independent random
    starting pivots are tried and the split with the lowest total
    per-sample assignment distance is kept; only which side ends up called
    "swapped" is arbitrary (downstream stats only need one consistent
    labeling, see best_family())."""
    n = len(pooled)
    ndim = pooled.shape[1]

    # Fixed per-parameter scale from the *unswapped* pool as a whole
    # (combines between/within-cluster spread either way -- circular/linear
    # std, unlike mean, doesn't cancel out under an even swap mix, so it's
    # a safe normalization even before any relabeling has happened, and
    # -- being restart-independent -- makes restarts' costs comparable).
    sigma = np.ones(ndim)
    for k in id_idx:
        col = pooled[:, k]
        if kinds[k] == 'X':
            m = circular_mean_pm_halfpi(col)
            sigma[k] = max(float(np.std(wrap_pm_halfpi(col - m))), 1e-6)
        else:
            sigma[k] = max(float(np.std(col)), 1e-6)

    def assign(ref):
        direct = id_param_distance_rows(pooled, ref, sigma, id_idx, kinds)
        swapped = id_param_distance_rows(pooled[:, swap_perm], ref, sigma, id_idx, kinds)
        return swapped < direct, direct, swapped

    best_mask, best_cost = None, np.inf
    for _ in range(n_restarts):
        mask, direct, swapped = assign(pooled[rng.integers(n)])
        for _ in range(n_iter):
            aligned = np.where(mask[:, None], pooled[:, swap_perm], pooled)
            ref = np.zeros(ndim)
            for k in id_idx:
                col = aligned[:, k]
                ref[k] = circular_mean_pm_halfpi(col) if kinds[k] == 'X' else np.mean(col)
            new_mask, direct, swapped = assign(ref)
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        cost = float(np.sum(np.where(mask, swapped, direct)))
        if cost < best_cost:
            best_cost, best_mask = cost, mask

    return np.where(best_mask[:, None], pooled[:, swap_perm], pooled)


def union_find_merge(modes, threshold, id_idx, kinds, swap_perm=None):
    """Group raw mode indices [0, len(modes)) into families. Every raw
    mode is its own family by default (no proximity-based merging -- a
    broad/multi-peaked pooled posterior would poorly represent its family
    even with a fine total evidence_share). The only merge is for
    DEGENERATE_PAIR_MODELS (`swap_perm` given): two raw modes are the same
    physical solution under the component-swap symmetry, merged since
    they're one solution labeled 1<->2 the other way, not two distinct
    ones. This merge is a strict mutual-nearest-neighbor pairing by
    swapped distance (each mode merges only with the one mode that is
    *also* its own nearest match, and only below `threshold`), not
    transitive union-find over a distance graph -- a real exact-symmetry
    solution has exactly one label-swapped twin, so this caps every merged
    group at 2 raw modes and makes chaining (A-B-C lumped together despite
    A/C being unrelated) structurally impossible."""
    n = len(modes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    if swap_perm is not None and n > 1:
        dist = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = id_param_distance(
                        modes[i], modes[j]['mean'][swap_perm], modes[j]['sigma'][swap_perm], id_idx, kinds)
        nearest = dist.argmin(axis=1)
        for i in range(n):
            j = int(nearest[i])
            if nearest[j] == i and dist[i, j] < threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def family_chi2(model, pars, wl, y, y_errs, ndim):
    """Reduced chi2 of `model(wl, pars)` (`pars` full-length) against
    complex data `y` (= q + i u) with complex errors `y_errs`. `ndim` is
    the number of actually-sampled (non-spectral) parameters, not
    `len(pars)` -- the spectral alpha/eps slots in `pars` were fixed by
    the least-squares pre-fit, not fit here, so they don't count against
    degrees of freedom."""
    resid = np.abs(model(wl, pars) - y)
    dof = 2 * len(wl) - ndim
    return float(np.sum((resid / np.abs(y_errs)) ** 2) / dof)


def cube_order(kinds_pol):
    """Cube-order permutation over `kinds_pol` (pol_idx-local order):
    X-kind positions first, then phi-kind, then everything else.
    `cube_order[i]` is the pol_idx-local position placed at cube position
    i. X-kind params are drawn via the seam-at-0 reparametrization
    (x_from_cube) instead of a plain bounded-linear draw, so the sampler's
    own domain seam doesn't coincide with where posterior mass tends to
    sit (near +-90 deg); `n_clustering_params` (the leading X+phi
    dimensions) drives MultiNest's own live-point clustering, so EVPA and
    Faraday depth together decide which raw modes are treated as distinct
    during sampling."""
    X_idx    = [i for i, k in enumerate(kinds_pol) if k == 'X']
    phi_idx  = [i for i, k in enumerate(kinds_pol) if k == 'phi']
    rest_idx = [i for i in range(len(kinds_pol)) if i not in X_idx and i not in phi_idx]
    return X_idx + phi_idx + rest_idx


def x_from_cube(u):
    """Map unit-cube coordinate u in [0,1) to canonical X in (-pi/2, pi/2],
    with the sampler's own domain seam placed at X=0 instead of X=+-pi/2."""
    raw = np.pi * u
    return raw if raw <= np.pi / 2 else raw - np.pi


# Substrings identifying MultiNest console lines to drop, and the header
# line that precedes them with no useful content of its own -- see
# filter_fd_lines().
MULTINEST_SUPPRESSED_SUBSTRINGS = ('converging towards the edge of the prior',)
MULTINEST_SUPPRESSED_HEADERS    = ('MultiNest Warning!',)


@contextlib.contextmanager
def filter_fd_lines(fd_num, block_substrings, hold_substrings=()):
    """Redirect a raw OS file descriptor through a line filter that drops
    any line containing one of `block_substrings`, passing everything else
    through unchanged to the original stream.

    Needed because MultiNest's own diagnostic text (e.g. "MultiNest
    Warning!" followed by "Parameter N of mode M is converging towards the
    edge of the prior") is written directly by the compiled Fortran
    library via the low-level file descriptor, bypassing Python's
    sys.stdout entirely -- so nothing at the Python level can intercept or
    selectively filter it. A plain fd-to-/dev/null redirect would silence
    everything MultiNest prints, not just this one warning, so instead a
    pipe + background thread pumps the fd's output line-by-line, dropping
    only matches (and their surrounding blank-line padding)."""
    saved_fd = os.dup(fd_num)
    passthrough_fd = os.dup(fd_num)
    read_fd, write_fd = os.pipe()

    def _pump():
        pending = []          # buffered blank/header lines, not yet resolved
        after_block = False   # just dropped a blocked line -> swallow trailing blanks

        def is_blank(s):
            return s.strip() == ''

        with os.fdopen(read_fd, 'r', errors='replace') as reader, \
             os.fdopen(passthrough_fd, 'w') as out:
            for line in reader:
                if any(s in line for s in block_substrings):
                    pending = []  # drop whatever was buffered along with this blocked line
                    after_block = True
                    continue
                if after_block and is_blank(line):
                    continue  # swallow blank padding trailing the dropped block
                after_block = False
                if is_blank(line) or any(s in line for s in hold_substrings):
                    pending.append(line)
                    continue
                if pending:
                    out.writelines(pending)
                    pending = []
                out.write(line)
                out.flush()
            if pending:
                out.writelines(pending)
                out.flush()

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    os.dup2(write_fd, fd_num)
    os.close(write_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, fd_num)  # restore original stream; drops the pipe's write end -> EOF for the pump
        os.close(saved_fd)
        pump.join()


def best_family(model, outputfiles_basename, wl, y, y_errs, spectral_pars,
                  idx_pol, order_cube, threshold=2.0, prob=0.68, degenerate_pair=None):
    """Identify the raw-MultiNest-mode family for `model`'s already-
    completed run at `outputfiles_basename` (matching multinest_fit()'s
    own outputfiles_basename convention), and return (pooled_samples,
    pars, errs, evidence_share, winner_chi2, winner_lnZ, other_families,
    dropped, winner_evidence_lnZ): the winning family's own pooled samples
    / point-estimate (pol_idx-local order) / evidence share / chi2 /
    point-lnZ, plus other surviving families with at least
    `MIN_FAMILY_EVIDENCE_SHARE`% evidence share (top 4) and a summary of
    everything else dropped, and the winner's own volume-integrated
    log-evidence (`evidence_lnZ`, distinct from `winner_lnZ`, the point
    log-likelihood -- logsumexp'ing every family/mode's own evidence_lnZ,
    including dropped ones, reproduces MultiNest's global_evidence). Each
    `other_families` entry has the same shape as the winner ('samples',
    'pars'/'errs', 'evidence_share', 'chi2', 'lnZ', 'evidence_lnZ') so a
    caller can treat any of them as a fully-fledged candidate result (see
    app.SamplingMixin's Corner-tab family switcher).

    Parses `outputfiles_basename + 'post_separate.dat'` and
    analyzer.get_mode_stats() for each raw mode's samples/local evidence,
    undoes the cube reordering (`order_cube`), groups modes into families
    (union_find_merge() -- one family per raw mode, except
    DEGENERATE_PAIR_MODELS pairs merged under the component-swap symmetry;
    see its own docstring), pools each family's members by weight and, for
    a swap-symmetric family, relabels the pool onto one consistent side
    (relabel_swap_symmetric()), then picks the winner among families
    clearing `MIN_FAMILY_EVIDENCE_SHARE`% (or every candidate, if none do)
    by highest point log-likelihood at its own marginal median
    (`bayesian_loglike`) -- not the same as `evidence_share`, since a
    broad/multi-peaked family can have a large evidence_share while its
    marginal median sits in a low-density gap between sub-peaks.

    Raises FileNotFoundError if post_separate.dat is missing (e.g. the run
    wasn't sampled with multimodal=True) rather than silently degrading.
    """
    import pymultinest

    spec = MODELS[model]
    ndim_full = len(spec.params)
    kinds_pol = [spec.params[i].kind for i in idx_pol]
    ndim = len(idx_pol)

    def expand(pol_pars):
        full = [0.0] * ndim_full
        for i, v in zip(idx_pol, pol_pars):
            full[i] = v
        for i, v in spectral_pars.items():
            full[i] = v
        return full

    sep_path = outputfiles_basename + 'post_separate.dat'
    if not os.path.exists(sep_path):
        raise FileNotFoundError(
            f'{sep_path} not found -- was this model fit with multimodal=True?')

    with open(sep_path) as f:
        content = f.read()
    blocks = [b for b in content.split('\n\n\n') if b.strip()]

    analyzer = pymultinest.Analyzer(n_params=ndim, outputfiles_basename=outputfiles_basename)
    mode_stats = analyzer.get_mode_stats()['modes']
    assert len(mode_stats) == len(blocks), (len(mode_stats), len(blocks))
    local_logz = [m['local log-evidence'] for m in mode_stats]

    rng = np.random.default_rng(42)
    modes = []
    for block, logz in zip(blocks, local_logz):
        rows = np.array([[float(v) for v in line.split()] for line in block.strip().splitlines()])
        weight = rows[:, 0]
        pars_cube_order = rows[:, 2:2 + ndim]
        pars_pol = np.zeros_like(pars_cube_order)
        pars_pol[:, order_cube] = pars_cube_order
        modes.append(mode_summary(pars_pol, weight, logz, kinds_pol, rng))

    id_idx = mode_id_indices(kinds_pol)
    is_degenerate_pair = model in DEGENERATE_PAIR_MODELS if degenerate_pair is None else degenerate_pair
    perm = swap_perm(kinds_pol) if is_degenerate_pair else None
    groups = union_find_merge(modes, threshold, id_idx, kinds_pol, perm)

    family_logz = [logsumexp([modes[k]['logz'] for k in g]) for g in groups]
    total_logz  = logsumexp(family_logz)
    family_shares = [float(np.exp(lz - total_logz) * 100) for lz in family_logz]

    pool_rng = np.random.default_rng(123)
    candidates = []
    for group, share, lz in zip(groups, family_shares, family_logz):
        member_logz = np.array([modes[k]['logz'] for k in group])
        member_share = np.exp(member_logz - logsumexp(member_logz))
        n_per_member = np.maximum(np.round(member_share * 20000).astype(int), 1)

        parts = []
        for k, n_i in zip(group, n_per_member):
            w = modes[k]['weight']
            idx = pool_rng.choice(len(w), size=n_i, replace=True, p=w / w.sum())
            parts.append(modes[k]['pars_raw'][idx])
        pooled_samples = np.concatenate(parts, axis=0)

        if perm is not None:
            pooled_samples = relabel_swap_symmetric(pooled_samples, id_idx, kinds_pol, perm, pool_rng)

        pars, errs = median_pctl_errs(pooled_samples, prob=prob)
        for j, k in enumerate(kinds_pol):
            if k == 'X':
                med, elo, ehi = circular_median_pctl(pooled_samples[:, j], prob=prob)
                pars[j] = med
                errs[0, j] = elo
                errs[1, j] = ehi

        chi2 = family_chi2(model, expand(pars), wl, y, y_errs, ndim)
        candidates.append(dict(pooled_samples=pooled_samples, pars=pars, errs=errs,
                                evidence_share=share, chi2=chi2, evidence_lnZ=float(lz)))

    eligible = [c for c in candidates if c['evidence_share'] >= MIN_FAMILY_EVIDENCE_SHARE] or candidates
    for c in eligible:
        c['lnZ'] = float(bayesian_loglike(model, expand(c['pars']), wl, y, y_errs))

    best = max(eligible, key=lambda c: c['lnZ'])
    others = sorted((c for c in candidates if c is not best),
                     key=lambda c: -c['evidence_share'])
    significant = [c for c in others if c['evidence_share'] >= MIN_FAMILY_EVIDENCE_SHARE]
    negligible = [c for c in others if c['evidence_share'] < MIN_FAMILY_EVIDENCE_SHARE]
    other_families = [{'samples': c['pooled_samples'], 'pars': c['pars'], 'errs': c['errs'],
                        'evidence_share': c['evidence_share'],
                        'chi2': c['chi2'], 'lnZ': c['lnZ'], 'evidence_lnZ': c['evidence_lnZ']}
                       for c in significant[:4]]
    trimmed = significant[4:] + negligible
    dropped = {'count': len(trimmed),
               'evidence_share': float(sum(c['evidence_share'] for c in trimmed))}
    return (best['pooled_samples'], best['pars'], best['errs'], best['evidence_share'],
            best['chi2'], best['lnZ'], other_families, dropped, best['evidence_lnZ'])


def multinest_fit(wl, q, q_err, u, u_err, model, spectral_pars, kind_bounds,
                   outputfiles_basename, n_live_points=400,
                   sampling_efficiency=0.2, evidence_tolerance=0.5,
                   progress_callback=None, n_iter_before_update=100):
    """Fit `model` to fractional Stokes q=Q/I, u=U/I data via MultiNest
    nested sampling with multimodal mode-finding enabled, then pick the
    winning mode-family (see best_family()) as the reported point
    estimate.

    `spectral_pars` is a {index: value} map for exactly `model`'s
    `alpha`/`eps`-kind params (from a least-squares pre-fit -- see
    MainWindow.run_multinest_fit in app.py); MultiNest never samples
    them, only every other ('p','X','phi','dphi','scale'-kind) parameter
    -- see `pol_idx`. This split is a fixed property of the model, not a
    dynamic fixed/free choice: every polarization-kind parameter is always
    sampled, regardless of any Visualization-tab slider's own "fixed"
    checkbox.

    `kind_bounds` gives the prior bounds for every sampled kind:
    {'p': (lo,hi), 'scale': (lo,hi), 'phi': (lo,hi,is_log), 'dphi':
    (lo,hi,is_log)} (physical units, e.g. fractional p not percent) -- X
    has no bounds entry, it's always drawn over its full (-pi/2, pi/2]
    range via the seam-shifted reparametrization (x_from_cube). phi may be
    sampled negative (a sign is drawn separately in log-uniform mode;
    linear mode's bounds may themselves be signed); dphi may not
    (log-uniform mode never draws a sign; linear mode's lower bound is
    clamped to 0).

    `sampling_efficiency` is MultiNest's own target acceptance efficiency
    (0-1) for its ellipsoidal sampling -- lower values (~0.2-0.3, the
    default here) are recommended for a reliable Bayesian evidence
    estimate, higher values (up to ~0.8) trade that off for speed and are
    better suited to parameter estimation alone.

    `progress_callback(n_samples, logZ, logZerr)`, if given, is invoked
    periodically (every `n_iter_before_update` MultiNest iterations) via
    pymultinest's own `dump_callback` -- there's no well-defined "percent
    complete" for nested sampling ahead of time (it runs to an
    evidence-convergence criterion, not a fixed iteration budget), so this
    is the closest thing to progress available to report.

    Returns (best_pars, errs, info): `best_pars`/`errs` are full-length
    (spectral slots filled from `spectral_pars`, errs=(0,0) there); `info`
    is a dict of evidence_share/winner_chi2/winner_lnZ/global_evidence/
    global_evidence_err/other_families/dropped for display.
    """
    import pymultinest  # lazy import: keeps fitting.py importable without pymultinest installed

    spec = MODELS[model]
    ndim_full = len(spec.params)
    idx_pol = pol_idx(spec)
    kinds_pol = [spec.params[i].kind for i in idx_pol]
    ndim = len(idx_pol)

    y = np.asarray(q) + 1j * np.asarray(u)
    y_errs = np.asarray(q_err) + 1j * np.asarray(u_err)
    wl = np.asarray(wl)

    order_cube = cube_order(kinds_pol)
    kinds_cube = [kinds_pol[i] for i in order_cube]
    n_clustering_params = sum(1 for k in kinds_cube if k in ('X', 'phi'))

    def prior(cube, ndim_, nparams):
        for i in range(ndim_):
            kind = kinds_cube[i]
            if kind == 'X':
                cube[i] = x_from_cube(cube[i])
            elif kind == 'phi':
                lo, hi, is_log = kind_bounds['phi']
                if is_log:
                    u_ = cube[i]
                    if u_ < 0.5:
                        sign, u2 = -1.0, u_ / 0.5
                    else:
                        sign, u2 = 1.0, (u_ - 0.5) / 0.5
                    logval = lo + u2 * (hi - lo)
                    cube[i] = sign * 10 ** logval
                else:
                    cube[i] = lo + cube[i] * (hi - lo)
            elif kind == 'dphi':
                lo, hi, is_log = kind_bounds['dphi']
                if is_log:
                    logval = lo + cube[i] * (hi - lo)
                    cube[i] = 10 ** logval
                else:
                    lo = max(lo, 0.0)
                    cube[i] = lo + cube[i] * (hi - lo)
            else:
                lo, hi = kind_bounds[kind]
                val = lo + cube[i] * (hi - lo)
                cube[i] = min(max(val, lo), hi)

    def loglike(cube, ndim_, nparams):
        cube_vals = [cube[i] for i in range(ndim_)]
        pol_vals = [None] * ndim
        for i, local_i in enumerate(order_cube):
            pol_vals[local_i] = cube_vals[i]
        m_pars = [0.0] * ndim_full
        for i, v in zip(idx_pol, pol_vals):
            m_pars[i] = v
        for i, v in spectral_pars.items():
            m_pars[i] = v
        wres = weighted_residuals(m_pars, wl, y, y_errs, model)
        return -0.5 * float(np.sum(wres ** 2)) - float(log_norm_const(y_errs))

    def _dump(*args):
        if progress_callback is None:
            return
        try:
            n_samples = int(args[0])
            logZ = float(args[7])
            logZerr = float(args[8])
        except Exception:
            return
        progress_callback(n_samples, logZ, logZerr)

    os.makedirs(os.path.dirname(outputfiles_basename), exist_ok=True)
    with filter_fd_lines(sys.stdout.fileno(), MULTINEST_SUPPRESSED_SUBSTRINGS, MULTINEST_SUPPRESSED_HEADERS):
        pymultinest.run(
            loglike, prior, ndim,
            outputfiles_basename=outputfiles_basename,
            n_live_points=n_live_points, sampling_efficiency=sampling_efficiency,
            multimodal=True, importance_nested_sampling=False,
            n_clustering_params=n_clustering_params,
            evidence_tolerance=evidence_tolerance,
            resume=False, verbose=False,
            dump_callback=_dump if progress_callback is not None else None,
            n_iter_before_update=n_iter_before_update,
        )

    analyzer = pymultinest.Analyzer(n_params=ndim, outputfiles_basename=outputfiles_basename)
    # get_mode_stats(), not get_stats(): both carry 'global evidence'/
    # 'global evidence error' (get_stats() is just get_mode_stats() plus a
    # per-dimension marginal-statistics table this function never reads),
    # but get_stats() additionally sorts and numpy.interp's the *entire*
    # posterior column-by-column to build that table -- seconds of wasted
    # work on a run with many samples, all thrown away below.
    stats = analyzer.get_mode_stats()

    best_family_result = best_family(model, outputfiles_basename, wl, y, y_errs, spectral_pars, idx_pol, order_cube)
    return assemble_result(spec, idx_pol, spectral_pars, best_family_result,
                             stats['global evidence'], stats['global evidence error'], 2 * len(wl))


def expand_pars_errs(ndim_full, pol_idx, pol_pars, pol_errs, spectral_pars):
    """Scatter a pol_idx-local point estimate (`pol_pars`, and `pol_errs`
    in the (2,ndim) lo-row/hi-row convention median_pctl_errs/best_family
    use) back into full-length `best_pars`/`errs` lists -- spectral
    ('alpha'/'eps'-kind) slots filled from `spectral_pars`, errs=(0,0)
    there, matching what every slider/model-curve caller (ParamSlider.set_
    value, ModelPlot/StokesPlot.update_plot) expects. Used by
    assemble_result() for the winning family, and directly by app.py's
    Corner-tab family switcher for whichever other family the user picks
    from its dropdown -- both need the exact same expansion."""
    best_pars = [0.0] * ndim_full
    errs = [(0.0, 0.0)] * ndim_full
    for i, v in zip(pol_idx, pol_pars):
        best_pars[i] = v
    for i, lo, hi in zip(pol_idx, pol_errs[0], pol_errs[1]):
        errs[i] = (lo, hi)
    for i, v in spectral_pars.items():
        best_pars[i] = v
    return best_pars, errs


def assemble_result(spec, pol_idx, spectral_pars, best_family_result,
                      global_evidence, global_evidence_err, n_data):
    """Scatter best_family()'s pol_idx-local result back into full-length
    `best_pars`/`errs` (spectral slots filled from `spectral_pars`,
    errs=(0,0) there) and build the `info` dict multinest_fit() and
    load_previous_run() both return -- shared so the two entry points
    (fresh sampling vs. re-clustering an already-completed run) produce
    identical-shaped results that app.py's result handling doesn't need
    to distinguish between.

    `n_data` is `2*len(wl)` (real+imag residuals count as separate data
    points, matching fit_statistics()'s own convention for the
    least-squares path) -- used, with `len(pol_idx)` as the free-parameter
    count, to report chi2/dof/AIC/AICc/BIC in the same format
    fit_statistics() does for a least-squares fit (see
    MainWindow.format_mn_stats), computed from `winner_chi2` (already the
    *reduced* chi2 -- see family_chi2) and `winner_lnZ` (the winning
    family's own point log-likelihood -- see best_family)."""
    pooled_samples, pol_pars, pol_errs, evidence_share, winner_chi2, winner_lnZ, other_families, dropped, \
        winner_evidence_lnZ = best_family_result
    ndim_full = len(spec.params)
    n_free = len(pol_idx)

    best_pars, errs = expand_pars_errs(ndim_full, pol_idx, pol_pars, pol_errs, spectral_pars)

    dof = n_data - n_free
    chi2 = winner_chi2 * dof
    aic = 2 * n_free - 2 * winner_lnZ
    aicc = aic + (2 * n_free * (n_free + 1)) / (dof - 1) if dof > 1 else float('nan')
    bic = n_free * np.log(n_data) - 2 * winner_lnZ

    info = dict(
        evidence_share=evidence_share, winner_chi2=winner_chi2, winner_lnZ=winner_lnZ,
        winner_evidence_lnZ=winner_evidence_lnZ,
        global_evidence=global_evidence, global_evidence_err=global_evidence_err,
        other_families=other_families, dropped=dropped,
        # winner_samples: pol_idx-local order (not full-length) -- the
        # winning family's own pooled posterior, for a caller that wants
        # to draw posterior-sample curves or a corner plot (see
        # MainWindow.apply_posterior_result / build_corner_tab in
        # app.py). pol_idx says which full-model index each column is.
        winner_samples=pooled_samples, pol_idx=list(pol_idx),
        # Same field names/formulas as fit_statistics() (least-squares),
        # fed from the winning family's own chi2/lnL instead -- see
        # MainWindow.format_mn_stats, which reports both in the same
        # layout, and on the same absolute ln L scale (both ultimately go
        # through log_norm_const), so directly comparable to a
        # least-squares fit's AIC/BIC for the same data and model.
        chi2=chi2, dof=dof, chi2_red=winner_chi2, loglike=winner_lnZ,
        aic=aic, aicc=aicc, bic=bic, n_free=n_free, n_data=n_data,
    )
    return best_pars, errs, info


def load_previous_run(wl, q, q_err, u, u_err, model, spectral_pars, outputfiles_basename,
                        sampled_idx=None, degenerate_pair=None):
    """Re-run best_family()'s mode-family clustering against an already-
    completed MultiNest run at `outputfiles_basename` -- no sampling, just
    re-reads that run's own output files -- for the Sampling tab's "Load
    samples" button (app.py).

    `wl`, `q`/`q_err`, `u`/`u_err`, `spectral_pars` must match what the
    original multinest_fit() call was given (typically round-tripped
    through a sidecar file the caller saved alongside the run's own
    output at fit time -- see MainWindow.run_multinest_fit /
    load_samples_action). Same return contract as multinest_fit():
    (best_pars, errs, info).

    `sampled_idx` (optional): override for which of `model`'s own
    non-alpha/eps ("pol") param indices this particular run actually
    sampled -- defaults to all of them (pol_idx(spec)), but a run
    that also held some p/X/phi/dphi slot fixed (not just alpha/eps) --
    e.g. a qu_fit.py "no dep"/"mid" model variant, which reuses one of
    this app's own full models' formula with one or more dphi slots
    hardcoded to 0 rather than being a separately registered model --
    only ever sampled a subset; every index of `model`'s own params left
    out of `sampled_idx` must then have its own fixed value in
    `spectral_pars` (same mechanism already used for alpha/eps). See
    sampling.SamplingMixin's qu_fit.py-import path (load_samples_action).

    `degenerate_pair` (optional): override for whether the winning-family
    search treats `model` as having an exact component-swap symmetry (see
    best_family/DEGENERATE_PAIR_MODELS) -- defaults to `model in
    DEGENERATE_PAIR_MODELS`, but a `sampled_idx` override that fixes only
    one component's dphi (not both, not neither) breaks the symmetry a
    *full* run of a DEGENERATE_PAIR_MODELS member would otherwise have;
    pass False in that case (mirrors qu_fit.py's own DEGENERATE_PAIR_MODELS,
    which excludes exactly these same asymmetric "mid" variants)."""
    import pymultinest

    spec = MODELS[model]
    idx_pol = sampled_idx if sampled_idx is not None else pol_idx(spec)
    kinds_pol = [spec.params[i].kind for i in idx_pol]
    ndim = len(idx_pol)
    order_cube = cube_order(kinds_pol)

    y = np.asarray(q) + 1j * np.asarray(u)
    y_errs = np.asarray(q_err) + 1j * np.asarray(u_err)
    wl = np.asarray(wl)

    analyzer = pymultinest.Analyzer(n_params=ndim, outputfiles_basename=outputfiles_basename)
    # get_mode_stats(), not get_stats(): both carry 'global evidence'/
    # 'global evidence error' (get_stats() is just get_mode_stats() plus a
    # per-dimension marginal-statistics table this function never reads),
    # but get_stats() additionally sorts and numpy.interp's the *entire*
    # posterior column-by-column to build that table -- seconds of wasted
    # work on a run with many samples, all thrown away below.
    stats = analyzer.get_mode_stats()

    best_family_result = best_family(model, outputfiles_basename, wl, y, y_errs, spectral_pars, idx_pol, order_cube,
                                       degenerate_pair=degenerate_pair)
    return assemble_result(spec, idx_pol, spectral_pars, best_family_result,
                             stats['global evidence'], stats['global evidence error'], 2 * len(wl))
