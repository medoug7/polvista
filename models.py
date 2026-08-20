"""Faraday-depolarization models and their metadata.

"""
import numpy as np
from dataclasses import dataclass
from typing import Callable

C = 299792458.0  # m/s
H_PLANCK = 6.62607015e-34   # J s
K_BOLTZMANN = 1.380649e-23  # J/K


# ── Spectral shape: normalized source function S'(nu) ─────────────────────────
# Every model's spectral weighting (spectral_weights for two-component
# models, stokes_I's single-component branch) shapes a component's relative
# intensity via a normalized source function S'(nu), S'(nu0)=1. The standard
# choice is a power law (nu/nu0)**alpha, with nu0 auto-set to the lowest
# frequency in the currently plotted band. A log-parabola (nu/nu0)**
# (alpha+beta*ln(nu/nu0)) shares that same shared-band nu0 -- it's a
# one-parameter generalization of the power law, adding a curvature term
# beta on top of alpha rather than a new reference frequency. Two other
# alternatives instead need an explicit turnover frequency nu0 (there's no
# band edge to fall back on): the classic synchrotron self-absorbed (SSA)
# spectrum, and a thermal free-free spectrum (full Planck function x
# free-free opacity, see thermal_source), which also needs an electron
# temperature T.
#
# Module-level rather than threaded through every model func's fixed
# (x, params) signature -- there's only ever one "current"
# shape/nu0/T/beta in this single-window app, set by the Spectrum box's
# dropdown(s) (see app.MainWindow.on_spectral_shape_changed).
# SPECTRAL_SHAPE/SPECTRAL_NU0/SPECTRAL_TEMP/SPECTRAL_BETA are a
# single-component model's own shape/turnover/temperature/curvature, or a
# two-component model's component-1 shape/turnover/temperature/curvature;
# the _2 globals are component 2's own, fully independent values -- a
# two-component model may mix e.g. 'powerlaw' for one component with
# 'ssa'/'thermal'/'logparabola' for the other. SPECTRAL_TEMP(_2) is only
# ever read when the matching shape is 'thermal'; SPECTRAL_BETA(_2) only
# when it's 'logparabola'.
SPECTRAL_SHAPE = 'powerlaw'
SPECTRAL_SHAPE_2 = 'powerlaw'
SPECTRAL_NU0 = None    # MHz -- SSA/thermal turnover frequency: a
                         # single-component model's own nu0, or a
                         # two-component model's component-1 nu0 (see
                         # set_spectral_shape)
SPECTRAL_NU0_2 = None   # MHz -- two-component SSA/thermal models'
                         # component-2 turnover frequency, independent of
                         # SPECTRAL_NU0
SPECTRAL_TEMP = None    # K -- thermal shape's electron temperature: a
                         # single-component model's own T, or a
                         # two-component model's component-1 T
SPECTRAL_TEMP_2 = None  # K -- two-component thermal models' component-2
                         # temperature, independent of SPECTRAL_TEMP
SPECTRAL_BETA = None    # log-parabola's curvature index: a
                         # single-component model's own beta, or a
                         # two-component model's component-1 beta
SPECTRAL_BETA_2 = None  # two-component log-parabola models' component-2
                         # curvature index, independent of SPECTRAL_BETA


def set_spectral_shape(shape, nu0=None, shape2=None, nu0_2=None, T=None, T2=None,
                        beta=None, beta2=None):
    """Select the source function(s) S'(nu) used by every model's spectral
    weighting from here on: 'powerlaw' (the default; nu0 auto-derived per
    call, always shared between both components of a two-component model --
    see reference_nu/component_reference_nu), 'ssa' (classic synchrotron
    self-absorption; `nu0` [MHz] is then required -- there's no band edge
    to default to), 'thermal' (thermal free-free; `nu0` [MHz] and `T` [K]
    are then required -- see thermal_source), or 'logparabola' (curved
    power law sharing the same shared-band nu0 as 'powerlaw'; `beta` is
    then required -- see source_function).

    `shape`/`nu0`/`T`/`beta` is a single-component model's own
    shape/turnover frequency/temperature/curvature, or a two-component
    model's component-1 shape/turnover/temperature/curvature;
    `shape2`/`nu0_2`/`T2`/`beta2` is component 2's own, fully independent
    shape/turnover/temperature/curvature -- a two-component model's
    components need not share a spectral shape at all, and (under 'ssa' or
    'thermal') need not turn over at the same frequency (or temperature)
    either. All default to component 1's own value when not given, so
    single-component callers (and any two-component caller that hasn't
    been updated to pick component 2's shape independently) still get one
    shared shape/nu0/T/beta as before."""
    global SPECTRAL_SHAPE, SPECTRAL_SHAPE_2, SPECTRAL_NU0, SPECTRAL_NU0_2
    global SPECTRAL_TEMP, SPECTRAL_TEMP_2, SPECTRAL_BETA, SPECTRAL_BETA_2
    SPECTRAL_SHAPE = shape
    SPECTRAL_SHAPE_2 = shape2 if shape2 is not None else shape
    SPECTRAL_NU0 = nu0
    SPECTRAL_NU0_2 = nu0_2 if nu0_2 is not None else nu0
    SPECTRAL_TEMP = T
    SPECTRAL_TEMP_2 = T2 if T2 is not None else T
    SPECTRAL_BETA = beta
    SPECTRAL_BETA_2 = beta2 if beta2 is not None else beta


def band_nu0(nu, nu_min):
    """The shared power-law reference frequency: `nu_min` if given,
    otherwise the lowest frequency spanned by `nu` itself."""
    return nu_min if nu_min is not None else np.min(nu)


def component_reference_nu(nu, nu_min, nu0_override, shape):
    """The nu0 [MHz] that anchors one component's S'(nu0)=1: `nu0_override`
    (that component's own SSA/thermal turnover frequency, set via
    set_spectral_shape) when `shape` is 'ssa' or 'thermal' and it's set,
    otherwise the shared band nu0 (see band_nu0) -- i.e. only 'ssa'/
    'thermal' let a component use its own reference frequency; 'powerlaw'
    always uses the shared one. Used directly for a single-component
    model's own reference_nu, and by spectral_weights for each of a
    two-component model's components (with that component's own
    shape/nu0_override)."""
    if shape in ('ssa', 'thermal') and nu0_override is not None:
        return nu0_override
    return band_nu0(nu, nu_min)


def reference_nu(nu, nu_min=None):
    """The nu0 [MHz] that anchors S'(nu0)=1 for a single-component model:
    the SSA turnover frequency set via set_spectral_shape when that shape
    is active, otherwise the shared band nu0 (see band_nu0)."""
    return component_reference_nu(nu, nu_min, SPECTRAL_NU0, SPECTRAL_SHAPE)


def thermal_source(ratio, nu0, T):
    """Normalized thermal free-free source function
    S'(nu) = B_nu(T)*(1-e^-tau_nu) / [B_nu0(T)*(1-e^-tau0)], `ratio` =
    nu/nu0, tau0 fixed to 1 and tau_nu = (nu/nu0)**-2.1 the free-free
    opacity (folding the usual tau0 normalization into nu0, same
    convention as 'ssa' -- nu0 is where tau_nu=1). `nu0` [MHz] is only
    used to get the dimensionless Theta = h*nu0/(k_B*T) below; `T` [K] is
    the electron temperature.

    This single expression has three asymptotes, set by Theta: optically
    thick (ratio << 1) is the Rayleigh-Jeans blackbody, S' ~ ratio**2;
    optically thin and still Rayleigh-Jeans (1 << ratio << 1/Theta) is the
    classical thermal-bremsstrahlung index, S' ~ ratio**-0.1; and
    ratio >> 1/Theta (h*nu >~ k_B*T) is an exponential Wien cutoff, from
    B_nu(T) itself -- so the -0.1 window is only visible when Theta << 1,
    i.e. nu0 well below k_B*T/h.

    A nu0 placed deep in (or past) that Wien cutoff -- Theta >~ 700 -- is
    unphysical for a real thermal source, but the sliders don't forbid it
    (see app.TEMP_BOUNDS_K/NU0_BOUNDS_MULT), and QU-fitting's own nu_0
    solve (fitting.estimate_shape_2comp) can wander there mid-search even
    starting from a sane guess. planck(ratio)/planck0 = ratio**3 *
    (e^Theta-1)/(e^(Theta*ratio)-1) is computed below via the algebraic
    identity (e^a-1)/(e^b-1) = e^(a-b) * (1-e^-a)/(1-e^-b), which only ever
    evaluates e^(-x) for x>=0 (safely in [0,1]) instead of the e^(+x) that
    overflows straight to inf once Theta is that large -- and the exponent
    is clipped before the final exp() so a genuinely out-of-float64-range
    answer saturates at a large finite number instead of overflowing,
    since an inf/nan source function is a non-finite residual that
    scipy's least_squares (used by estimate_shape_2comp) rejects outright,
    aborting the fit."""
    nu0_hz = nu0 * 1e6
    theta = H_PLANCK * nu0_hz / (K_BOLTZMANN * T)
    tau = ratio ** -2.1
    bracket = -np.expm1(-tau)
    tau0 = 1.0
    bracket0 = -np.expm1(-tau0)
    theta_ratio = theta * ratio
    with np.errstate(over='ignore', invalid='ignore'):
        log_planck_ratio = (3 * np.log(ratio) + theta * (1 - ratio)
                             + np.log1p(-np.exp(-theta)) - np.log1p(-np.exp(-theta_ratio)))
    planck_ratio = np.exp(np.clip(log_planck_ratio, -700.0, 700.0))
    return np.nan_to_num(planck_ratio * bracket / bracket0, nan=0.0, posinf=1e300, neginf=0.0)


def source_function(nu, nu0, alpha, shape, T=None, beta=None):
    """Normalized source function S'(nu), S'(nu0)=1, in the given `shape`
    ('powerlaw'/'ssa'/'thermal'/'logparabola' -- always passed explicitly
    by the caller, e.g. a two-component model's own per-component
    SPECTRAL_SHAPE/SPECTRAL_SHAPE_2, since the two components of a model
    need not share one).
    'powerlaw': (nu/nu0)**alpha.
    'ssa': classic synchrotron self-absorbed spectrum (nu/nu0)**(5/2) *
    (1-e^-tau_nu)/(1-e^-tau_0), tau_0 fixed to 1 and tau_nu = tau_0*
    (nu/nu0)**(alpha-5/2) the frequency-dependent opacity -- the -5/2 offset
    is what makes alpha the actual optically-thin spectral index (nu >>
    nu0): S'(nu) there is ~ (nu/nu0)**(5/2) * tau_nu ~ (nu/nu0)**alpha.
    'thermal': thermal free-free spectrum, see thermal_source -- `alpha`
    is inert for this shape (its -0.1 optically-thin index is emergent
    from `T`, not a free parameter); `T` [K] is required.
    'logparabola': curved power law (nu/nu0)**(alpha+beta*ln(nu/nu0)) --
    alpha is still the local spectral index at nu0 (as under 'powerlaw'),
    and `beta` [dimensionless] is required; beta=0 reduces exactly to
    'powerlaw'. beta<0 gives a spectrum that peaks near nu0 and falls off
    on both sides (concave down in log-log); beta>0 diverges on both
    sides instead (concave up) -- see MODELS/spec.params for typical
    bounds. Unlike 'ssa'/'thermal', nu0 here is still the shared band edge
    (see component_reference_nu), not a free turnover of its own."""
    ratio = nu / nu0
    if shape == 'ssa':
        tau0 = 1.0
        tau_nu = tau0 * ratio ** (alpha - 2.5)
        # (1-e^-x) = -expm1(-x); tau0's constant denominator cancels the
        # sign, and expm1 keeps this well-behaved as tau_nu -> 0.
        return ratio ** 2.5 * np.expm1(-tau_nu) / np.expm1(-tau0)
    if shape == 'thermal':
        return thermal_source(ratio, nu0, T)
    if shape == 'logparabola':
        return ratio ** (alpha + beta * np.log(ratio))
    return ratio ** alpha


# ── Polarization models ───────────────────────────────────────────────────────
def burn(x, params):
    """Burn depolarization (4 pars: p0, X0, phi, dphi; trailing alpha
    is accepted but inert -- a single component's spectral index cancels
    out of p=P/I)."""
    p0, X0, phi, dphi = params[:4]
    return p0 * np.exp(-2 * dphi**2 * x**4 + 2j * (X0 + phi * x**2))


def tribble(x, params):
    """Partially resolved turbulent screen with quadratic Faraday-depth structure function (5 pars; trailing alpha inert)."""
    p0, X0, phi, dphi, s = params[:5]
    P = (np.sqrt((1 - np.exp(-s**2 / 2 - 4 * dphi**2 * x**4)) / (1 + 8 * dphi**2 * x**4 / s**2)
        + np.exp(-s**2 / 2 - 4 * dphi**2 * x**4)) * np.exp(2j * (X0 + phi * x**2)))
    return p0 * P


def partial(x, params):
    """Turbulent screen with partial coverage (5 pars; trailing alpha inert)."""
    p0, X0, phi, dphi, f = params[:5]
    return p0 * (f * np.exp(-2 * dphi**2 * x**4) * np.exp(2j * (X0 + phi * x**2)) + (1 - f) * np.exp(2j * X0))


def intern(x, params):
    """Internal uniform Faraday screen (4 pars; trailing alpha inert)."""
    p0, X0, phi, dphi = params[:4]
    return (p0 * np.exp(2j * X0) * (1 - np.exp(-(2 * dphi**2 * x**4 - 2j * phi * x**2))) / (2 * (dphi+1e-10)**2 * x**4 - 2j * phi * x**2))

def partial2(x, params):
    """internal screen with partial coverage (5 pars; trailing alpha inert)."""
    p0, X0, phi, dphi, f = params[:5]
    return p0 * np.exp(2j * X0) *(f* (1 - np.exp(-(2 * dphi**2 * x**4 - 2j * phi * x**2))) / (2 * (dphi+1e-10)**2 * x**4 - 2j * phi * x**2) + (1 - f))


def ext_term(x, p, X, phi, dphi):
    """Single external-Faraday-screen contribution -- the building block
    shared by both components of comp2RMdep/comp2mixdep."""
    return p * np.exp(2j * (X + phi * x**2)) * np.exp(-dphi**2 * x**4)


def int_term(x, p, X, phi, dphi):
    """Single internal-Faraday-screen contribution -- the building block
    shared by both components of comp2intern/comp2mixdep."""
    return p * np.exp(2j * X) * (1 - np.exp(-(2 * dphi**2 * x**4 - 2j * phi * x**2))) / (2 * (dphi+1e-10)**2 * x**4 - 2j * phi * x**2)


def spectral_weights(x, eps, alpha1, alpha2, nu_min=None):
    """w1(nu), w2(nu): each component's relative-intensity weight, i.e. its
    own I_k(nu) up to the shared overall normalization (no separate I_0
    needed -- these are shapes, amplitude 1 at each component's own
    reference frequency).

    epsilon is component 1's flux fraction at that reference frequency -- a
    directly-set slider rather than something derived from per-component
    turnover frequencies. Each weight then evolves away from it via that
    component's own source function S'(nu), in that component's own shape
    (see source_function/set_spectral_shape -- the two components need not
    share a shape):
        w1 = eps * S1'(nu),  w2 = (1-eps) * S2'(nu)
    w1+w2 is thus the (normalized) total Stokes I(nu) of the two-component
    system; see stokes_I().

    A component whose own shape is 'powerlaw' always uses the shared
    reference frequency nu0 (`nu_min` [MHz] if given, else the *lowest*
    frequency spanned by x -- the longest wavelength currently plotted);
    one whose shape is 'ssa'/'thermal' instead uses its own turnover
    frequency (and, for 'thermal', its own temperature), set via
    set_spectral_shape (see component_reference_nu). Pass an explicit
    `nu_min` to anchor the power-law case elsewhere (e.g. to loaded data's
    own nu_min so the Stokes I/Q/U display doesn't drift when the plotted
    wavelength range changes -- see MainWindow.data_nu_min)."""
    nu = C / x / 1e6  # MHz
    nu0_1 = component_reference_nu(nu, nu_min, SPECTRAL_NU0, SPECTRAL_SHAPE)
    nu0_2 = component_reference_nu(nu, nu_min, SPECTRAL_NU0_2, SPECTRAL_SHAPE_2)
    w1 = eps * source_function(nu, nu0_1, alpha1, SPECTRAL_SHAPE, T=SPECTRAL_TEMP, beta=SPECTRAL_BETA)
    w2 = (1.0 - eps) * source_function(nu, nu0_2, alpha2, SPECTRAL_SHAPE_2, T=SPECTRAL_TEMP_2, beta=SPECTRAL_BETA_2)
    return w1, w2


def spectral_combine(x, term1, term2, eps, alpha1, alpha2):
    """Blend two per-component polarization terms with independent
    power-law spectral weighting, normalized so |P| can never exceed
    max(|term1|,|term2|) regardless of alpha1/alpha2 -- i.e. spectral
    effects alone can never push the observed polarization past 100%.
    P = (w1*term1 + w2*term2) / (w1 + w2) is a convex combination of
    term1, term2 at every wavelength, so |P| <= max(|term1|, |term2|)."""
    w1, w2 = spectral_weights(x, eps, alpha1, alpha2)
    return (w1 * term1 + w2 * term2) / (w1 + w2)


def comp2RMdep(x, params):
    """Two-component external Faraday screen with turb, each component
    weighted by its own power-law spectral index (11 pars: p1,X1,phi1,
    dphi1,p2,X2,phi2,dphi2,eps,alpha1,alpha2). alpha1=alpha2=0 reduces to
    a plain eps/(1-eps) blend of the two components at every wavelength
    (see spectral_combine)."""
    p1, X1, phi1, dphi1, p2, X2, phi2, dphi2, eps, alpha1, alpha2 = params
    term1 = ext_term(x, p1, X1, phi1, dphi1)
    term2 = ext_term(x, p2, X2, phi2, dphi2)
    return spectral_combine(x, term1, term2, eps, alpha1, alpha2)


def comp2intern(x, params):
    """Two internal Faraday components, each weighted by its own power-law
    spectral index (11 pars, see comp2RMdep's docstring)."""
    p1, X1, phi1, dphi1, p2, X2, phi2, dphi2, eps, alpha1, alpha2 = params
    term1 = int_term(x, p1, X1, phi1, dphi1)
    term2 = int_term(x, p2, X2, phi2, dphi2)
    return spectral_combine(x, term1, term2, eps, alpha1, alpha2)


def comp2mixdep(x, params):
    """One internal + one external Faraday component, each weighted by its
    own power-law spectral index (11 pars, see comp2RMdep's docstring)."""
    p1, X1, phi1, dphi1, p2, X2, phi2, dphi2, eps, alpha1, alpha2 = params
    term1 = int_term(x, p1, X1, phi1, dphi1)
    term2 = ext_term(x, p2, X2, phi2, dphi2)
    return spectral_combine(x, term1, term2, eps, alpha1, alpha2)


# Each two-component model's own per-component (term1, term2) building
# blocks, in the same order as its own 11-param layout -- lets
# stokes_components() recover each component's contribution without
# duplicating comp2RMdep/comp2intern/comp2mixdep's own term selection.
COMPONENT_TERM_FUNCS = {
    comp2RMdep: (ext_term, ext_term),
    comp2intern: (int_term, int_term),
    comp2mixdep: (int_term, ext_term),
}


def evpa(fit):
    """EVPA [deg] of a complex fractional polarization array P = q+iu (e.g.
    a model's own p*e^(2i*chi) output)."""
    return np.degrees(0.5 * np.arctan2(fit.imag, fit.real))


def pol(fit):
    """Fractional linear polarization [%], |P|, of a complex fractional
    polarization array P = q+iu."""
    return 100 * np.sqrt(fit.real ** 2 + fit.imag ** 2)


def two_component_ref_wl(nu, nu_min, shape1, shape2):
    """The wavelength [m] the two-component branches of stokes_I/
    stokes_components evaluate raw_ref at: the wavelength corresponding to
    nu_min (lowest plotted frequency, or an explicit override) by default
    -- except when component 1's own shape is 'ssa' or 'thermal' (or,
    failing that, component 2's), which anchors at that component's own
    turnover nu0 instead (see stokes_I's docstring for why)."""
    if shape1 in ('ssa', 'thermal'):
        nu_ref = component_reference_nu(nu, nu_min, SPECTRAL_NU0, shape1)
    elif shape2 in ('ssa', 'thermal'):
        nu_ref = component_reference_nu(nu, nu_min, SPECTRAL_NU0_2, shape2)
    else:
        nu_ref = band_nu0(nu, nu_min)
    return C / (nu_ref * 1e6)


def stokes_I(wl, n_components, pars, nu_min=None):
    """Normalized Stokes I(nu) implied by a model's own trailing spectral
    params (the last 1 for a single-component model: alpha; the last 3
    for a two-component model: eps,alpha1,alpha2) -- no separate I_0
    needed, this is a *display/export* shape with amplitude 1 at nu_min
    (the lowest frequency spanned by wl, or an explicit override -- see
    below) for 'powerlaw'/'logparabola'. For a single component the
    spectral index has no effect on p=P/I, but it does shape the total
    intensity spectrum itself, via S'(nu) (see source_function/
    set_spectral_shape); for two components this is the same w1+w2 total
    weight that already governs how the polarization blends between them.

    Anchoring at nu_min rather than at each component's own physics-level
    reference frequency (nu0, which for 'powerlaw'/'logparabola' usually
    *is* nu_min anyway) is deliberate: it keeps this normalization matched
    to the Load Data path's own I(nu_min)=1 convention (see
    MainWindow.load_data_action).

    'ssa' and 'thermal' are the exception: each anchors at that
    component's own turnover nu0 instead (see two_component_ref_wl for the
    two-component case), not nu_min. Both source functions are already
    physically normalized to S'(nu0)=1 by construction (see
    thermal_source, and source_function's 'ssa' branch at ratio=1), so
    this is a genuine physical anchor rather than an arbitrary one -- and
    it means dragging that component's own turnover slider rescales the
    displayed curve, since nu0 is exactly where it now reads 1. (Only
    'powerlaw'/'logparabola' need the nu_min fallback: they have no
    turnover of their own to anchor at.)

    `nu_min` [MHz] defaults to the lowest frequency spanned by wl (the
    longest wavelength currently plotted); pass an explicit value (e.g. the
    loaded data's own nu_min) to anchor the normalization elsewhere."""
    nu = C / wl / 1e6  # MHz
    if nu_min is None:
        nu_min = np.min(nu)
    nu_min_arr = np.array([nu_min])

    if n_components == 1:
        alpha = pars[-1]
        nu0 = reference_nu(nu, nu_min)
        raw = source_function(nu, nu0, alpha, SPECTRAL_SHAPE, T=SPECTRAL_TEMP, beta=SPECTRAL_BETA)
        if SPECTRAL_SHAPE in ('ssa', 'thermal'):
            raw_ref = 1.0  # source_function is already S'(nu0)=1 by construction for both
        else:
            raw_ref = source_function(nu_min_arr, nu0, alpha, SPECTRAL_SHAPE, T=SPECTRAL_TEMP, beta=SPECTRAL_BETA)[0]
    else:
        eps, alpha1, alpha2 = pars[-3:]
        wl_ref = two_component_ref_wl(nu, nu_min, SPECTRAL_SHAPE, SPECTRAL_SHAPE_2)
        w1, w2 = spectral_weights(wl, eps, alpha1, alpha2, nu_min=nu_min)
        raw = w1 + w2
        w1_ref, w2_ref = spectral_weights(np.array([wl_ref]), eps, alpha1, alpha2, nu_min=nu_min)
        raw_ref = (w1_ref + w2_ref)[0]
    return raw / raw_ref


def stokes_QU(wl, model, n_components, pars, nu_min=None, I=None):
    """Physical (I_0=1-normalized) Stokes Q(nu), U(nu): model() already
    returns the fractional complex polarization P/I, so multiplying back
    by the model's own Stokes I(nu) recovers genuine Q, U (not just Q/I,
    U/I) -- e.g. for a single component with p constant in nu, Q and U
    simply trace I(nu)'s power-law shape, modulated by cos/sin(2*EVPA).
    `nu_min` [MHz] is forwarded to stokes_I() -- see there.

    `I`, if given, is that same stokes_I(wl, n_components, pars, nu_min)
    already computed by the caller -- passing it in skips recomputing it
    here. A caller that doesn't already have it can just omit it."""
    fit = model(wl, pars)
    if I is None:
        I = stokes_I(wl, n_components, pars, nu_min=nu_min)
    return fit.real * I, fit.imag * I


def stokes_components(wl, model, pars, nu_min=None):
    """Decompose a two-component model's Stokes I/Q/U into each
    component's own contribution: ((I1, Q1, U1), (I2, Q2, U2)), in the
    same nu_min=1-normalized units as stokes_I/stokes_QU, so I1+I2 equals
    stokes_I(...)'s own output and (Q1+Q2, U1+U2) equals stokes_QU(...)'s
    -- each component's own term (see COMPONENT_TERM_FUNCS) weighted by
    its own share (w1 or w2, see spectral_weights) of the same raw_ref
    normalization stokes_I anchors the total to."""
    term1_func, term2_func = COMPONENT_TERM_FUNCS[model]
    p1, X1, phi1, dphi1, p2, X2, phi2, dphi2, eps, alpha1, alpha2 = pars
    term1 = term1_func(wl, p1, X1, phi1, dphi1)
    term2 = term2_func(wl, p2, X2, phi2, dphi2)

    nu = C / wl / 1e6  # MHz
    if nu_min is None:
        nu_min = np.min(nu)
    wl_ref = two_component_ref_wl(nu, nu_min, SPECTRAL_SHAPE, SPECTRAL_SHAPE_2)  # matches stokes_I's own anchor

    w1, w2 = spectral_weights(wl, eps, alpha1, alpha2, nu_min=nu_min)
    w1_ref, w2_ref = spectral_weights(np.array([wl_ref]), eps, alpha1, alpha2, nu_min=nu_min)
    raw_ref = (w1_ref + w2_ref)[0]

    I1, I2 = w1 / raw_ref, w2 / raw_ref
    Q1, U1 = term1.real * I1, term1.imag * I1
    Q2, U2 = term2.real * I2, term2.imag * I2
    return (I1, Q1, U1), (I2, Q2, U2)


# ── Model metadata: ModelSpec registry ────────────────────────────────────────
@dataclass
class Param:
    name: str    # plain name, e.g. 'p_0' -- used for slider labels
    latex: str   # e.g. r'$p_0$' -- used for plot-facing labels
    kind: str    # 'p' | 'X' | 'phi' | 'dphi' | 'scale' -- drives slider type/bounds
    description: str = ''  # short physical-meaning blurb -- shown as a slider tooltip


@dataclass
class ModelSpec:
    func: Callable
    label: str                      # short legend/dropdown label
    title: str                      # longer name, for titles
    params: list
    bounds: tuple                   # (lower_list, upper_list)
    n_components: int = 1
    equation: str = ''              # mathtext for the complex polarization P(lambda)
    # Suggested MultiNest live-point count for the Sampling tab (see
    # sampling.SamplingMixin.build_sampling_tab/rebuild_sampling_bounds) --
    # more params (and especially the 2nd mode a two-component model's own
    # exchange symmetry introduces, see fitting.DEGENERATE_PAIR_MODELS) need
    # more live points to resolve the posterior reliably. Values mirror
    # ~/Downloads/pipe/qu_fit.py's own per-model ModelSpec.n_live_points
    # (its reference implementation this app's own QU-fitting/MultiNest
    # pipeline was distilled from -- see fitting.py's module docstring),
    # for whichever of its models this one matches; partial2 has no
    # counterpart there (qu_fit.py has no internal-screen-with-partial-
    # coverage model) so it just reuses partial's own value -- same
    # single-component, 5-param, partial-coverage structure, only an
    # internal rather than external screen.
    n_live_points: int = 1000

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def indices(self, kind):
        return [i for i, p in enumerate(self.params) if p.kind == kind]


MODELS = {}


def register(func, **kwargs):
    MODELS[func] = ModelSpec(func=func, **kwargs)


def spectral_params():
    """The 3 trailing params every two-component model appends: epsilon
    (component 1's own weight w1 at component 1's own reference frequency
    -- nu_min, the lowest frequency/longest wavelength currently plotted,
    unless component 1's own shape is 'ssa'/'thermal', in which case it's
    that component's own turnover nu0 instead; see
    component_reference_nu/spectral_weights) and each component's own
    power-law spectral index, used to weight that component's contribution
    to the total by (nu/nu0)**alpha. alpha1=alpha2=0 reduces to a plain
    eps/(1-eps) blend at every wavelength (see spectral_combine). Note
    epsilon only equals I1/(I1+I2) *at that same reference frequency* when
    component 2 also happens to equal 1 there -- true by default (both
    components sharing the band-edge nu_min under 'powerlaw'/
    'logparabola'), but not guaranteed once either component uses its own
    independent turnover."""
    return [Param('epsilon', r'$\varepsilon$', 'eps',
                  "Component 1's own weight at its own reference frequency "
                  "(the lowest plotted frequency, or its own turnover under "
                  "SSA/thermal); sets which component dominates"),
            Param('alpha1', r'$\alpha_1$', 'alpha',
                  "Spectral index of component 1's intensity: I1(nu) ∝ nu^alpha1."),
            Param('alpha2', r'$\alpha_2$', 'alpha',
                  "Spectral index of component 2's intensity: I2(nu) ∝ nu^alpha2.")]


def spectral_param_single():
    """The single trailing alpha appended to single-component models --
    shapes the model's own Stokes I(nu) (normalized to 1 at nu_min, the
    lowest frequency/longest wavelength currently plotted; see stokes_I())
    but has no effect on p=P/I itself."""
    return [Param('alpha', r'$\alpha$', 'alpha',
                  'Spectral index of total intensity: I(nu) ∝ nu^alpha. '
                  'Shapes Stokes I(nu) but has no effect on p=P/I.')]


SPECTRAL_BOUNDS_LO = [0.0, -3.0, -3.0]      # eps, alpha1, alpha2
SPECTRAL_BOUNDS_HI = [1.0, 3.0, 3.0]
SPECTRAL_BOUNDS_LO_SINGLE = [-3.0]           # alpha -- single-component only
SPECTRAL_BOUNDS_HI_SINGLE = [3.0]


register(burn,
    label='External screen with depolarization (Burn)', title='Burn depolarization',
    params=[Param('p_0', r'$p_0$', 'p', 'Intrinsic fractional polarization.'),
            Param('X_0', r'$\chi_0$', 'X', 'Intrinsic EVPA (polarization angle) at lambda=0.'),
            Param('phi', r'$\phi$', 'phi', 'Faraday depth (RM-like): sets how fast EVPA rotates with lambda^2.'),
            Param('dphi', r'$\sigma_{\phi}$', 'dphi', 'Faraday depth dispersion across the external turbulent screen; drives depolarization at long wavelengths.')] + spectral_param_single(),
    bounds=([0, -np.pi / 2, -5e6, 1e2] + SPECTRAL_BOUNDS_LO_SINGLE,
            [0.7, np.pi / 2, 5e6, 1e6] + SPECTRAL_BOUNDS_HI_SINGLE),
    n_components=1, n_live_points=500,
    equation=r'$P(\lambda)=p_0\,e^{-2\sigma_\phi^2\lambda^4}\,e^{\,2i(\chi_0+\phi\lambda^2)}$')

register(tribble,
    label='Partially resolved external screen (Tribble)', title='Tribble screen',
    params=[Param('p_0', r'$p_0$', 'p', 'Intrinsic fractional polarization.'),
            Param('X_0', r'$\chi_0$', 'X', 'Intrinsic EVPA (polarization angle) at lambda=0.'),
            Param('phi', r'$\phi$', 'phi', 'Faraday depth (RM-like): sets how fast EVPA rotates with lambda^2.'),
            Param('dphi', r'$\sigma_{\phi}$', 'dphi', 'Faraday depth dispersion across the turbulent screen; drives depolarization at long wavelengths.'),
            Param('s', r'$s$', 'scale', 'Beam-to-turbulent-cell size ratio: how well the turbulent screen is resolved (larger s = better resolved, less depolarization).')] + spectral_param_single(),
    bounds=([0, -np.pi / 2, -5e6, 0, 0] + SPECTRAL_BOUNDS_LO_SINGLE,
            [0.7, np.pi / 2, 5e6, 5e6, 10] + SPECTRAL_BOUNDS_HI_SINGLE),
    n_components=1, n_live_points=700,
    equation=r'$P(\lambda)=p_0\left[\dfrac{1-e^{-s^2/2-4\sigma_\phi^2\lambda^4}}{1+8\sigma_\phi^2\lambda^4/s^2}+e^{-s^2/2-4\sigma_\phi^2\lambda^4}\right]^{1/2}\!e^{\,2i(\chi_0+\phi\lambda^2)}$')

register(partial,
    label='Partial coverage external screen', title='Partial coverage external',
    params=[Param('p_0', r'$p_0$', 'p', 'Intrinsic fractional polarization.'),
            Param('X_0', r'$X_0$', 'X', 'Intrinsic EVPA (polarization angle) at lambda=0.'),
            Param('phi', r'$\phi$', 'phi', 'Faraday depth (RM-like): sets how fast EVPA rotates with lambda^2.'),
            Param('dphi', r'$\sigma_{\phi}$', 'dphi', 'Faraday depth dispersion across the covering part of the external screen; drives depolarization at long wavelengths.'),
            Param('f', r'$f$', 'scale', 'Covering fraction of the depolarizing screen; the remaining (1-f) of the emission passes through unchanged.')] + spectral_param_single(),
    bounds=([0, -np.pi / 2, -5e6, 0, 0] + SPECTRAL_BOUNDS_LO_SINGLE,
            [0.7, np.pi / 2, 5e6, 5e6, 1] + SPECTRAL_BOUNDS_HI_SINGLE),
    n_components=1, n_live_points=700,
    equation=r'$P(\lambda)=p_0\left[f\,e^{-2\sigma_\phi^2\lambda^4}e^{\,2i(\chi_0+\phi\lambda^2)}+(1-f)\,e^{2i\chi_0}\right]$')


register(intern,
    label='Internal screen with depolarization', title='Internal screen',
    params=[Param('p_0', r'$p_0$', 'p', 'Intrinsic fractional polarization.'),
            Param('X_0', r'$\chi_0$', 'X', 'Intrinsic EVPA (polarization angle) at lambda=0.'),
            Param('phi', r'$\phi$', 'phi', 'Faraday depth (RM-like): sets how fast EVPA rotates with lambda^2.'),
            Param('dphi', r'$\sigma_{\phi}$', 'dphi', 'Faraday depth dispersion across the internal, emission-mixed screen; drives depolarization at long wavelengths.')] + spectral_param_single(),
    bounds=([0.0, -np.pi / 2, -5e6, 0] + SPECTRAL_BOUNDS_LO_SINGLE,
            [0.7, np.pi / 2, 5e6, 5e6] + SPECTRAL_BOUNDS_HI_SINGLE),
    n_components=1, n_live_points=1000,
    equation=r'$P(\lambda)=p_0\,e^{2i\chi_0}\,\left[\frac{1-e^{-\left(2\sigma_\phi^2\lambda^4-2i\phi\lambda^2\right)}}{2\sigma_\phi^2\lambda^4-2i\phi\lambda^2}\right]$')

register(partial2,
    label='Partial coverage internal screen', title='Partial coverage internal',
    params=[Param('p_0', r'$p_0$', 'p', 'Intrinsic fractional polarization.'),
            Param('X_0', r'$X_0$', 'X', 'Intrinsic EVPA (polarization angle) at lambda=0.'),
            Param('phi', r'$\phi$', 'phi', 'Faraday depth (RM-like): sets how fast EVPA rotates with lambda^2.'),
            Param('dphi', r'$\sigma_{\phi}$', 'dphi', 'Faraday depth dispersion across the covering part of the internal screen; drives depolarization at long wavelengths.'),
            Param('f', 'f', 'scale', 'Covering fraction of the depolarizing internal screen; the remaining (1-f) of the emission passes through unchanged.')] + spectral_param_single(),
    bounds=([0, -np.pi / 2, -5e6, 0, 0] + SPECTRAL_BOUNDS_LO_SINGLE,
            [0.7, np.pi / 2, 5e6, 5e6, 1] + SPECTRAL_BOUNDS_HI_SINGLE),
    n_components=1, n_live_points=1200,  # no qu_fit.py counterpart -- reuses partial's own value (see ModelSpec.n_live_points)
    equation=r'$P(\lambda)=p_0\,e^{2i\chi_0}\left\{f\left[\frac{1-e^{-\left(2\sigma_\phi^2\lambda^4-2i\phi\lambda^2\right)}}{2\sigma_\phi^2\lambda^4-2i\phi\lambda^2}\right]+(1-f)\right\}$')


register(comp2RMdep,
    label='2 External screens', title='2 External components',
    params=[Param('p_1', r'$p_1$', 'p', "Component 1's intrinsic fractional polarization."),
            Param('X_1', r'$\chi_1$', 'X', "Component 1's intrinsic EVPA at lambda=0."),
            Param('phi1', r'$\phi_1$', 'phi', "Component 1's Faraday depth (RM-like)."),
            Param('dphi1', r'$\sigma_{\phi,1}$', 'dphi', "Component 1's Faraday depth dispersion across its external screen."),
            Param('p_2', r'$p_2$', 'p', "Component 2's intrinsic fractional polarization."),
            Param('X_2', r'$\chi_2$', 'X', "Component 2's intrinsic EVPA at lambda=0."),
            Param('phi2', r'$\phi_2$', 'phi', "Component 2's Faraday depth (RM-like)."),
            Param('dphi2', r'$\sigma_{\phi,2}$', 'dphi', "Component 2's Faraday depth dispersion across its external screen.")] + spectral_params(),
    bounds=([0, -np.pi / 2, -5e6, 1, 0, -np.pi / 2, -5e6, 1] + SPECTRAL_BOUNDS_LO,
            [0.7, np.pi / 2, 5e6, 5e6, 0.7, np.pi / 2, 5e6, 5e6] + SPECTRAL_BOUNDS_HI),
    n_components=2, n_live_points=2500,
    equation=(r'$P(\lambda)=\frac{w_1}{w_1+w_2}p_1e^{-\sigma_{\phi,1}^2\lambda^4}e^{\,2i(\chi_1+\phi_1\lambda^2)}'
              r'+\frac{w_2}{w_1+w_2}p_2e^{-\sigma_{\phi,2}^2\lambda^4}e^{\,2i(\chi_2+\phi_2\lambda^2)}$'))


register(comp2intern,
    label='2 Internal screens', title='2 internal components',
    params=[Param('p_1', r'$p_1$', 'p', "Component 1's intrinsic fractional polarization."),
            Param('X_1', r'$\chi_1$', 'X', "Component 1's intrinsic EVPA at lambda=0."),
            Param('phi1', r'$\phi_1$', 'phi', "Component 1's Faraday depth (RM-like)."),
            Param('dphi1', r'$\sigma_{\phi,1}$', 'dphi', "Component 1's Faraday depth dispersion across its internal screen."),
            Param('p_2', r'$p_2$', 'p', "Component 2's intrinsic fractional polarization, before Faraday depolarization."),
            Param('X_2', r'$\chi_2$', 'X', "Component 2's intrinsic EVPA at lambda=0."),
            Param('phi2', r'$\phi_2$', 'phi', "Component 2's Faraday depth (RM-like)."),
            Param('dphi2', r'$\sigma_{\phi,2}$', 'dphi', "Component 2's Faraday depth dispersion across its internal screen.")] + spectral_params(),
    bounds=([0, -np.pi / 2, -5e6, 1, 0, -np.pi / 2, -5e6, 1] + SPECTRAL_BOUNDS_LO,
            [0.7, np.pi / 2, 5e6, 5e6, 0.7, np.pi / 2, 5e6, 5e6] + SPECTRAL_BOUNDS_HI),
    n_components=2, n_live_points=2500,
    equation=(r'$P(\lambda)=\frac{w_1}{w_1+w_2}p_1\,e^{2i\chi_1}\left[\frac{1-e^{-\left(2\sigma_{\phi,1}^2\lambda^4-2i\phi_1\lambda^2\right)}}{2\sigma_{\phi,1}^2\lambda^4-2i\phi_1\lambda^2}\right]'
    	      r'+\frac{w_2}{w_1+w_2}p_2\,e^{2i\chi_2}\left[\frac{1-e^{-\left(2\sigma_{\phi,2}^2\lambda^4-2i\phi_2\lambda^2\right)}}{2\sigma_{\phi,2}^2\lambda^4-2i\phi_2\lambda^2}\right]$'))


register(comp2mixdep,
    label='Internal and external screens', title='Internal + external components',
    params=[Param('p_1', r'$p_1$', 'p', "Component 1's (internal screen) intrinsic fractional polarization."),
            Param('X_1', r'$\chi_1$', 'X', "Component 1's intrinsic EVPA at lambda=0."),
            Param('phi1', r'$\phi_1$', 'phi', "Component 1's Faraday depth (RM-like)."),
            Param('dphi1', r'$\sigma_{\phi,1}$', 'dphi', "Component 1's Faraday depth dispersion across its internal screen."),
            Param('p_2', r'$p_2$', 'p', "Component 2's (external screen) intrinsic fractional polarization."),
            Param('X_2', r'$\chi_2$', 'X', "Component 2's intrinsic EVPA at lambda=0."),
            Param('phi2', r'$\phi_2$', 'phi', "Component 2's Faraday depth (RM-like)."),
            Param('dphi2', r'$\sigma_{\phi,2}$', 'dphi', "Component 2's Faraday depth dispersion across its external screen.")] + spectral_params(),
    bounds=([0, -np.pi/2, -5e6, 1, 0, -np.pi/2, -5e6, 1] + SPECTRAL_BOUNDS_LO,
            [0.7, np.pi/2, 5e6, 5e6, 0.7, np.pi / 2, 5e6, 5e6] + SPECTRAL_BOUNDS_HI),
    n_components=2, n_live_points=2500,
    equation=(r'$P(\lambda)=\frac{w_1}{w_1+w_2}p_1e^{2i\chi_1}\left[\frac{1-e^{-\left(2\sigma_{\phi,1}^2\lambda^4-2i\phi_1\lambda^2\right)}}{2\sigma_{\phi,1}^2\lambda^4-2i\phi_1\lambda^2}\right]'
              r'+\frac{w_2}{w_1+w_2}p_2e^{-\sigma_{\phi,2}^2\lambda^4}e^{\,2i(\chi_2+\phi_2\lambda^2)}$'))


# ── Equation-card display: prepend the chosen S'(nu) definition(s) to a
# model's own polarization equation, and -- for two-component models -- the
# w1/w2/epsilon definition built from them, on its own line above. Built
# dynamically (not baked into ModelSpec.equation at registration time) since
# the shape(s) ('powerlaw'/'ssa') are a runtime choice -- see
# app.MainWindow.on_spectral_shape_changed.
#
# 'powerlaw' always shares one nu0 between both components of a
# two-component model (matches the Polvista paper); 'ssa' lets a component
# turn over at its own nu_{0,i} (see set_spectral_shape/
# component_reference_nu). When both components share one shape, the
# combined, unsubscripted-nu0 (powerlaw) or i-subscripted (ssa) templates
# below are used, matching the paper; when they don't (one 'powerlaw', one
# 'ssa'), full_equation instead builds each component's own S_i'(nu) line
# separately (see _s_prime_component_latex). The frequency-dependent opacity
# tau_nu isn't shown as its own term -- tau_0 is fixed to 1, so it's inlined
# directly into S'(nu) instead of introducing it as a separate symbol.
S_PRIME_POWERLAW = r"S'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha}"
S_PRIME_SSA_ONE = (r"S'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{5/2}"
                    r"\left[\dfrac{1-e^{-(\nu/\nu_0)^{\alpha-5/2}}}{1-e^{-1}}\right]")
S_PRIME_SSA_TWO = (r"S_i'(\nu)=\left(\dfrac{\nu}{\nu_{0,i}}\right)^{5/2}"
                    r"\left[\dfrac{1-e^{-(\nu/\nu_{0,i})^{\alpha_i-5/2}}}{1-e^{-1}}\right]\ \ (i=1,2)")
S_PRIME_THERMAL_ONE = (r"S'(\nu)=\dfrac{B_\nu(T)\left[1-e^{-(\nu/\nu_0)^{-2.1}}\right]}"
                        r"{B_{\nu_0}(T)\left(1-e^{-1}\right)}")
S_PRIME_THERMAL_TWO = (r"S_i'(\nu)=\dfrac{B_\nu(T_i)\left[1-e^{-(\nu/\nu_{0,i})^{-2.1}}\right]}"
                        r"{B_{\nu_{0,i}}(T_i)\left(1-e^{-1}\right)}\ \ (i=1,2)")
# Log-parabola shares 'powerlaw''s own shared band nu_0 (no per-component
# turnover), so -- like S_PRIME_POWERLAW -- one template serves both the
# single- and two-component (shared-shape) cases; only the mixed-shape
# per-component branch (_s_prime_component_latex) ever needs alpha_i/beta_i.
S_PRIME_LOGPARABOLA = r"S'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha+\beta\ln(\nu/\nu_0)}"


def _s_prime_component_latex(idx, shape):
    """S_i'(nu) LaTeX for one two-component model's own component `idx`
    (1 or 2) under its own `shape` -- used by s_prime_latex only when the
    two components don't share one shape, so each needs its own explicit
    line rather than the combined i-subscripted templates above."""
    if shape == 'ssa':
        return (r"S_{%d}'(\nu)=\left(\dfrac{\nu}{\nu_{0,%d}}\right)^{5/2}"
                 r"\left[\dfrac{1-e^{-(\nu/\nu_{0,%d})^{\alpha_{%d}-5/2}}}{1-e^{-1}}\right]"
                 % (idx, idx, idx, idx))
    if shape == 'thermal':
        return (r"S_{%d}'(\nu)=\dfrac{B_\nu(T_{%d})\left[1-e^{-(\nu/\nu_{0,%d})^{-2.1}}\right]}"
                 r"{B_{\nu_{0,%d}}(T_{%d})\left(1-e^{-1}\right)}" % (idx, idx, idx, idx, idx))
    if shape == 'logparabola':
        return (r"S_{%d}'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha_{%d}+\beta_{%d}\ln(\nu/\nu_0)}"
                 % (idx, idx, idx))
    return r"S_{%d}'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha_{%d}}" % (idx, idx)


def s_prime_latex(n_components, shape1, shape2):
    """LaTeX (no surrounding '$') for the S'(nu) source-function
    definition(s) given `shape1`/`shape2`
    ('powerlaw'/'ssa'/'thermal'/'logparabola') -- `shape2` is ignored for a
    single-component model (n_components==1: `shape1` is that model's own,
    only, shape)."""
    templates = {'ssa': (S_PRIME_SSA_ONE, S_PRIME_SSA_TWO),
                 'thermal': (S_PRIME_THERMAL_ONE, S_PRIME_THERMAL_TWO),
                 'logparabola': (S_PRIME_LOGPARABOLA, S_PRIME_LOGPARABOLA)}
    if n_components == 1:
        one, _ = templates.get(shape1, (S_PRIME_POWERLAW, None))
        return one
    if shape1 == shape2:
        _, two = templates.get(shape1, (None, S_PRIME_POWERLAW))
        return two
    return _s_prime_component_latex(1, shape1) + r"\ \ " + _s_prime_component_latex(2, shape2)


def weights_latex(shape1, shape2):
    """LaTeX (no surrounding '$') for a two-component model's w1/w2/epsilon
    line: how S'(nu) combines into each component's weight w_i and the
    epsilon that sets w1 vs w2 (Polvista paper Eq. 9-11). Each component's
    own reference frequency in the epsilon definition follows its own
    shape -- nu_0 (shared) under 'powerlaw', nu_{0,i} (its own) under
    'ssa'/'thermal' -- independently of the other component's."""
    nu0_1 = r'\nu_{0,1}' if shape1 in ('ssa', 'thermal') else r'\nu_0'
    nu0_2 = r'\nu_{0,2}' if shape2 in ('ssa', 'thermal') else r'\nu_0'
    eps_def = r"\varepsilon=\dfrac{I_1(%s)}{I_1(%s)+I_2(%s)}" % (nu0_1, nu0_1, nu0_2)
    return r"w_1=\varepsilon\,S_1'(\nu),\ \ w_2=(1-\varepsilon)\,S_2'(\nu),\ \ " + eps_def


def full_equation(spec, shape1, shape2):
    """The full (centered, possibly multi-line) equation-card LaTeX for
    `spec` at the given spectral shape(s): its own P(lambda) equation, plus
    the S'(nu) definition(s) `shape1`/`shape2` imply. `shape2` is ignored
    for a single-component model.

    Single-component model: one line, S'(nu) to the left of P(lambda)
    (S'(nu) shapes only the Stokes I spectrum there, not p=P/I itself, but
    the choice is still shown).

    Two-component model: the w1/w2/epsilon definition on its own line
    first, then S'(nu) to the left of P(lambda) on the line below (matching
    the single-component layout, since P(lambda) itself is written in
    terms of w1/w2 -- see each two-component model's own `equation`)."""
    s_prime = s_prime_latex(spec.n_components, shape1, shape2)
    body = spec.equation.strip()
    inner = body[1:-1]  # strip the surrounding '$...$'
    line2 = f'${s_prime}\\qquad\\quad {inner}$'
    if spec.n_components == 1:
        return line2
    line1 = f'${weights_latex(shape1, shape2)}$'
    return f'{line1}\n{line2}'

# Model lookup by function name (e.g. 'burn', 'comp2intern'), the identifier
# saved/loaded in "Save Model"/"Load Model" JSON files and MultiNest's own
# polvista metadata sidecar (see app.py's save_model_action/load_model_action
# and sampling.py's load_samples_action) -- MODELS itself is keyed by the
# function object, which isn't serializable.
MODELS_BY_NAME = {func.__name__: func for func in MODELS}

