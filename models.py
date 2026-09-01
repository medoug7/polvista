"""Faraday-depolarization models and their metadata.

"""
import re
from collections import OrderedDict
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from dataclasses import dataclass
from typing import Callable

C = 299792458.0  # m/s
H_PLANCK = 6.62607015e-34   # J s
K_BOLTZMANN = 1.380649e-23  # J/K


# ── Spectral shape: normalized intensity shape I'(nu) ──────────────────────────
# Every model's spectral weighting (spectral_weights for two-component
# models, stokes_I's single-component branch) shapes a component's relative
# intensity via a normalized intensity shape I'(nu), I'(nu0)=1 -- i.e. the
# emergent Stokes I(nu)/I(nu0) implied by that component's spectral choice.
# (Classical radiative-transfer notation reserves "source function" S_nu for
# j_nu/alpha_nu, the *pre-escape* quantity I_nu relaxes toward -- see
# intensity_shape's own docstring for where that bare quantity actually
# shows up, via apply_escape=False, in the custom-model per-z opacity term.)
# The standard choice is a power law (nu/nu0)**alpha, with nu0 auto-set to
# the lowest frequency in the currently plotted band. A log-parabola
# (nu/nu0)**(alpha+beta*ln(nu/nu0)) shares that same shared-band nu0 -- it's
# a one-parameter generalization of the power law, adding a curvature term
# beta on top of alpha rather than a new reference frequency. Two other
# alternatives instead need an explicit turnover frequency nu0 (there's no
# band edge to fall back on): the classic synchrotron self-absorbed (SSA)
# spectrum, and a thermal free-free spectrum (full Planck function x
# free-free opacity, see thermal_intensity_shape), which also needs an
# electron temperature T.
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
# Spectrum box's shape dropdown (app.py) and the custom-model builder's own
# preview-only Spectral-model dropdown (build_model.py): (display label,
# set_spectral_shape key). Kept here rather than in app.py so build_model.py
# can import it too without app.py<->build_model.py becoming circular.
SPECTRAL_SHAPES = [('Power-law', 'powerlaw'), ('Log-parabola', 'logparabola'), ('SSA', 'ssa'), ('Thermal', 'thermal'),]

# (lo, hi) [K] bounds for the thermal shape's electron-temperature
# sliders -- not tied to the plotted band (unlike app.NU0_BOUNDS_MULT),
# since T sets the Wien cutoff/Rayleigh-Jeans amplitude rather than a
# frequency within it. Spans cool photoionized gas (~1e2 K) through hot
# X-ray-emitting plasma (~1e8 K). Kept here rather than in app.py (like
# SPECTRAL_SHAPES above) so build_model.py's own locked "T" row (see its
# CustomModel._sync_locked_rows) can show the same bounds without a
# circular import.
TEMP_BOUNDS_K = (1e-1, 1e7)

# (lo, hi) bounds for the log-parabola shape's curvature-index (beta)
# sliders -- linear, unlike nu0/T (beta doesn't span decades). Wide enough
# to explore both a strongly peaked (beta<0) and a divergent (beta>0)
# spectrum; beta=0 (the default) reduces the shape exactly to a plain
# power law. Kept here for the same reason TEMP_BOUNDS_K is.
BETA_BOUNDS = (-2.0, 2.0)

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
    """Select the normalized intensity shape(s) I'(nu) used by every model's
    spectral weighting from here on: 'powerlaw' (the default; nu0
    auto-derived per call, always shared between both components of a
    two-component model -- see reference_nu/component_reference_nu), 'ssa'
    (classic synchrotron self-absorption; `nu0` [MHz] is then required --
    there's no band edge to default to), 'thermal' (thermal free-free;
    `nu0` [MHz] and `T` [K] are then required -- see
    thermal_intensity_shape), or 'logparabola' (curved power law sharing
    the same shared-band nu0 as 'powerlaw'; `beta` is then required -- see
    intensity_shape).

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
    """The nu0 [MHz] that anchors one component's I'(nu0)=1: `nu0_override`
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
    """The nu0 [MHz] that anchors I'(nu0)=1 for a single-component model:
    the SSA turnover frequency set via set_spectral_shape when that shape
    is active, otherwise the shared band nu0 (see band_nu0)."""
    return component_reference_nu(nu, nu_min, SPECTRAL_NU0, SPECTRAL_SHAPE)


def thermal_intensity_shape(ratio, nu0, T, apply_escape=True):
    """Normalized thermal free-free intensity shape
    I'(nu) = B_nu(T)*(1-e^-tau_nu) / [B_nu0(T)*(1-e^-tau0)], `ratio` =
    nu/nu0, tau0 fixed to 1 and tau_nu = (nu/nu0)**-2.1 the free-free
    opacity (folding the usual tau0 normalization into nu0, same
    convention as 'ssa' -- nu0 is where tau_nu=1). `nu0` [MHz] is only
    used to get the dimensionless Theta = h*nu0/(k_B*T) below; `T` [K] is
    the electron temperature.

    `apply_escape=False` returns the bare local Planck source function
    (`planck_ratio` below) alone -- B_nu(T)/B_nu0(T), the true S_nu/S_nu0
    of classical radiative-transfer notation, via Kirchhoff's law S_nu =
    B_nu(T) in LTE -- without the `(1-e^-tau_nu)/(1-e^-tau0)`
    escape-probability bracket -- see intensity_shape's own docstring for
    why a caller (models.build_custom_model's own per-z opacity) would
    want that bare source-function piece instead of this function's
    default, already-escaped, whole-slab emergent intensity ratio.

    This single expression has three asymptotes, set by Theta: optically
    thick (ratio << 1) is the Rayleigh-Jeans blackbody, I' ~ ratio**2;
    optically thin and still Rayleigh-Jeans (1 << ratio << 1/Theta) is the
    classical thermal-bremsstrahlung index, I' ~ ratio**-0.1; and
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
    since an inf/nan intensity shape is a non-finite residual that
    scipy's least_squares (used by estimate_shape_2comp) rejects outright,
    aborting the fit."""
    nu0_hz = nu0 * 1e6
    theta = H_PLANCK * nu0_hz / (K_BOLTZMANN * T)
    theta_ratio = theta * ratio
    with np.errstate(over='ignore', invalid='ignore'):
        log_planck_ratio = (3 * np.log(ratio) + theta * (1 - ratio)
                             + np.log1p(-np.exp(-theta)) - np.log1p(-np.exp(-theta_ratio)))
    planck_ratio = np.exp(np.clip(log_planck_ratio, -700.0, 700.0))
    if not apply_escape:
        return np.nan_to_num(planck_ratio, nan=0.0, posinf=1e300, neginf=0.0)
    tau = ratio ** -2.1
    bracket = -np.expm1(-tau)
    tau0 = 1.0
    bracket0 = -np.expm1(-tau0)
    return np.nan_to_num(planck_ratio * bracket / bracket0, nan=0.0, posinf=1e300, neginf=0.0)


def intensity_shape(nu, nu0, alpha, shape, T=None, beta=None, apply_escape=True):
    """Normalized intensity shape I'(nu), I'(nu0)=1, in the given `shape`
    ('powerlaw'/'ssa'/'thermal'/'logparabola' -- always passed explicitly
    by the caller, e.g. a two-component model's own per-component
    SPECTRAL_SHAPE/SPECTRAL_SHAPE_2, since the two components of a model
    need not share one).
    'powerlaw': (nu/nu0)**alpha.
    'ssa': classic synchrotron self-absorbed spectrum (nu/nu0)**(5/2) *
    (1-e^-tau_nu)/(1-e^-tau_0), tau_0 fixed to 1 and tau_nu = tau_0*
    (nu/nu0)**(alpha-5/2) the frequency-dependent opacity -- the -5/2 offset
    is what makes alpha the actual optically-thin spectral index (nu >>
    nu0): I'(nu) there is ~ (nu/nu0)**(5/2) * tau_nu ~ (nu/nu0)**alpha.
    'thermal': thermal free-free spectrum, see thermal_intensity_shape --
    `alpha` is inert for this shape (its -0.1 optically-thin index is
    emergent from `T`, not a free parameter); `T` [K] is required.
    'logparabola': curved power law (nu/nu0)**(alpha+beta*ln(nu/nu0)) --
    alpha is still the local spectral index at nu0 (as under 'powerlaw'),
    and `beta` [dimensionless] is required; beta=0 reduces exactly to
    'powerlaw'. beta<0 gives a spectrum that peaks near nu0 and falls off
    on both sides (concave down in log-log); beta>0 diverges on both
    sides instead (concave up) -- see MODELS/spec.params for typical
    bounds. Unlike 'ssa'/'thermal', nu0 here is still the shared band edge
    (see component_reference_nu), not a free turnover of its own.

    `apply_escape=False` (only meaningful for 'ssa'/'thermal' -- a no-op
    for 'powerlaw'/'logparabola', which have no tau_nu/escape-probability
    concept at all) strips the `(1-e^-tau_nu)/(1-e^-tau0)` bracket,
    returning the bare *local* source function shape alone (the true,
    classical-radiative-transfer-sense S_nu/S_nu0 = j_nu/alpha_nu, not
    I'(nu)) -- `ratio**2.5` for 'ssa' (notably alpha-independent: the
    universal synchrotron self-absorption source-function shape doesn't
    depend on the electron power-law index, only tau_nu's own
    frequency-scaling does), or the bare Planck ratio for 'thermal' (see
    thermal_intensity_shape). That bracket is the closed-form *emergent*-
    intensity solution of a uniform, unresolved slab -- i.e. it already
    *is* a line-of-sight integral, just collapsed algebraically for the
    special case of z-independent source/absorption coefficients. A caller
    building its own per-z optical depth (see build_custom_model's own
    opacity term) needs the bare S(nu) alone for Kirchhoff's law
    (alpha'(z)=j(z)/S(nu)) -- reusing the bracketed (I'(nu)) value there
    would double-apply that same escape-probability physics, once via the
    bracket and again via the newly-resolved per-z integral."""
    ratio = nu / nu0
    if shape == 'ssa':
        bare = ratio ** 2.5
        if not apply_escape:
            return bare
        tau0 = 1.0
        tau_nu = tau0 * ratio ** (alpha - 2.5)
        # (1-e^-x) = -expm1(-x); tau0's constant denominator cancels the
        # sign, and expm1 keeps this well-behaved as tau_nu -> 0.
        return bare * np.expm1(-tau_nu) / np.expm1(-tau0)
    if shape == 'thermal':
        return thermal_intensity_shape(ratio, nu0, T, apply_escape=apply_escape)
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
    component's own normalized intensity shape I'(nu), in that component's
    own shape (see intensity_shape/set_spectral_shape -- the two
    components need not share a shape):
        w1 = eps * I1'(nu),  w2 = (1-eps) * I2'(nu)
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
    w1 = eps * intensity_shape(nu, nu0_1, alpha1, SPECTRAL_SHAPE, T=SPECTRAL_TEMP, beta=SPECTRAL_BETA)
    w2 = (1.0 - eps) * intensity_shape(nu, nu0_2, alpha2, SPECTRAL_SHAPE_2, T=SPECTRAL_TEMP_2, beta=SPECTRAL_BETA_2)
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
    intensity spectrum itself, via I'(nu) (see intensity_shape/
    set_spectral_shape); for two components this is the same w1+w2 total
    weight that already governs how the polarization blends between them.

    Anchoring at nu_min rather than at each component's own physics-level
    reference frequency (nu0, which for 'powerlaw'/'logparabola' usually
    *is* nu_min anyway) is deliberate: it keeps this normalization matched
    to the Load Data path's own I(nu_min)=1 convention (see
    MainWindow.load_data_action).

    'ssa' and 'thermal' are the exception: each anchors at that
    component's own turnover nu0 instead (see two_component_ref_wl for the
    two-component case), not nu_min. Both intensity shapes are already
    physically normalized to I'(nu0)=1 by construction (see
    thermal_intensity_shape, and intensity_shape's 'ssa' branch at
    ratio=1), so this is a genuine physical anchor rather than an
    arbitrary one -- and
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
        raw = intensity_shape(nu, nu0, alpha, SPECTRAL_SHAPE, T=SPECTRAL_TEMP, beta=SPECTRAL_BETA)
        if SPECTRAL_SHAPE in ('ssa', 'thermal'):
            raw_ref = 1.0  # intensity_shape is already I'(nu0)=1 by construction for both
        else:
            raw_ref = intensity_shape(nu_min_arr, nu0, alpha, SPECTRAL_SHAPE, T=SPECTRAL_TEMP, beta=SPECTRAL_BETA)[0]
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
# Model lookup by function name (e.g. 'burn', 'comp2intern'), the identifier
# saved/loaded in "Save Model"/"Load Model" JSON files and MultiNest's own
# polvista metadata sidecar (see app.py's save_model_action/load_model_action
# and sampling.py's load_samples_action) -- MODELS itself is keyed by the
# function object, which isn't serializable. Populated by register() itself
# (below) so a custom model built at runtime (see build_custom_model) shows
# up here too, not just the ones registered at import time.
MODELS_BY_NAME = {}


def register(func, **kwargs):
    MODELS[func] = ModelSpec(func=func, **kwargs)
    MODELS_BY_NAME[func.__name__] = func


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
    but has no effect on p=P/I itself -- except for a custom model with
    SSA/thermal opacity active (see _custom_P_raw/_custom_opacity_attenuation),
    where alpha also shapes the per-z optical depth tau_nu that opacity
    term builds, and so can change p=P/I too."""
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


# ── Equation-card display: prepend the chosen I'(nu) definition(s) to a
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
# 'ssa'), full_equation instead builds each component's own I_i'(nu) line
# separately (see _i_prime_component_latex). The frequency-dependent opacity
# tau_nu isn't shown as its own term -- tau_0 is fixed to 1, so it's inlined
# directly into I'(nu) instead of introducing it as a separate symbol.
I_PRIME_POWERLAW = r"I'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha}"
I_PRIME_SSA_ONE = (r"I'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{5/2}"
                    r"\left[\dfrac{1-e^{-(\nu/\nu_0)^{\alpha-5/2}}}{1-e^{-1}}\right]")
I_PRIME_SSA_TWO = (r"I_i'(\nu)=\left(\dfrac{\nu}{\nu_{0,i}}\right)^{5/2}"
                    r"\left[\dfrac{1-e^{-(\nu/\nu_{0,i})^{\alpha_i-5/2}}}{1-e^{-1}}\right]\ \ (i=1,2)")
I_PRIME_THERMAL_ONE = (r"I'(\nu)=\dfrac{B_\nu(T)\left[1-e^{-(\nu/\nu_0)^{-2.1}}\right]}"
                        r"{B_{\nu_0}(T)\left(1-e^{-1}\right)}")
I_PRIME_THERMAL_TWO = (r"I_i'(\nu)=\dfrac{B_\nu(T_i)\left[1-e^{-(\nu/\nu_{0,i})^{-2.1}}\right]}"
                        r"{B_{\nu_{0,i}}(T_i)\left(1-e^{-1}\right)}\ \ (i=1,2)")
# Log-parabola shares 'powerlaw''s own shared band nu_0 (no per-component
# turnover), so -- like I_PRIME_POWERLAW -- one template serves both the
# single- and two-component (shared-shape) cases; only the mixed-shape
# per-component branch (_i_prime_component_latex) ever needs alpha_i/beta_i.
I_PRIME_LOGPARABOLA = r"I'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha+\beta\ln(\nu/\nu_0)}"


def _i_prime_component_latex(idx, shape):
    """I_i'(nu) LaTeX for one two-component model's own component `idx`
    (1 or 2) under its own `shape` -- used by i_prime_latex only when the
    two components don't share one shape, so each needs its own explicit
    line rather than the combined i-subscripted templates above."""
    if shape == 'ssa':
        return (r"I_{%d}'(\nu)=\left(\dfrac{\nu}{\nu_{0,%d}}\right)^{5/2}"
                 r"\left[\dfrac{1-e^{-(\nu/\nu_{0,%d})^{\alpha_{%d}-5/2}}}{1-e^{-1}}\right]"
                 % (idx, idx, idx, idx))
    if shape == 'thermal':
        return (r"I_{%d}'(\nu)=\dfrac{B_\nu(T_{%d})\left[1-e^{-(\nu/\nu_{0,%d})^{-2.1}}\right]}"
                 r"{B_{\nu_{0,%d}}(T_{%d})\left(1-e^{-1}\right)}" % (idx, idx, idx, idx, idx))
    if shape == 'logparabola':
        return (r"I_{%d}'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha_{%d}+\beta_{%d}\ln(\nu/\nu_0)}"
                 % (idx, idx, idx))
    return r"I_{%d}'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha_{%d}}" % (idx, idx)


def i_prime_latex(n_components, shape1, shape2):
    """LaTeX (no surrounding '$') for the I'(nu) normalized-intensity-shape
    definition(s) given `shape1`/`shape2`
    ('powerlaw'/'ssa'/'thermal'/'logparabola') -- `shape2` is ignored for a
    single-component model (n_components==1: `shape1` is that model's own,
    only, shape)."""
    templates = {'ssa': (I_PRIME_SSA_ONE, I_PRIME_SSA_TWO),
                 'thermal': (I_PRIME_THERMAL_ONE, I_PRIME_THERMAL_TWO),
                 'logparabola': (I_PRIME_LOGPARABOLA, I_PRIME_LOGPARABOLA)}
    if n_components == 1:
        one, _ = templates.get(shape1, (I_PRIME_POWERLAW, None))
        return one
    if shape1 == shape2:
        _, two = templates.get(shape1, (None, I_PRIME_POWERLAW))
        return two
    return _i_prime_component_latex(1, shape1) + r"\ \ " + _i_prime_component_latex(2, shape2)


def weights_latex(shape1, shape2):
    """LaTeX (no surrounding '$') for a two-component model's w1/w2/epsilon
    line: how I'(nu) combines into each component's weight w_i and the
    epsilon that sets w1 vs w2 (Polvista paper Eq. 9-11). Each component's
    own reference frequency in the epsilon definition follows its own
    shape -- nu_0 (shared) under 'powerlaw', nu_{0,i} (its own) under
    'ssa'/'thermal' -- independently of the other component's."""
    nu0_1 = r'\nu_{0,1}' if shape1 in ('ssa', 'thermal') else r'\nu_0'
    nu0_2 = r'\nu_{0,2}' if shape2 in ('ssa', 'thermal') else r'\nu_0'
    eps_def = r"\varepsilon=\dfrac{I_1(%s)}{I_1(%s)+I_2(%s)}" % (nu0_1, nu0_1, nu0_2)
    return r"w_1=\varepsilon\,I_1'(\nu),\ \ w_2=(1-\varepsilon)\,I_2'(\nu),\ \ " + eps_def


def _insert_opacity_factor(latex):
    """Splice the e^{-tau_lambda(z)} attenuation factor into a custom
    model's own P(lambda) equation text, right after its always-present
    Faraday phase factor -- shared by full_equation (main window) and the
    builder dialog's own live preview (build_model.refit_dialog_equation)
    so both show it identically whenever SSA/thermal opacity applies (see
    _custom_P_raw/_custom_opacity_attenuation for the actual computation).
    Works on either the full '$...$'-delimited line or its bare inner text
    -- the pattern matched has no trailing '$' of its own."""
    return latex.replace(r'\lambda^2}\,dz', r'\lambda^2}\,e^{-\tau_\lambda(z)}\,dz')


def full_equation(spec, shape1, shape2):
    """The full (centered, possibly multi-line) equation-card LaTeX for
    `spec` at the given spectral shape(s): its own P(lambda) equation, plus
    the I'(nu) definition(s) `shape1`/`shape2` imply. `shape2` is ignored
    for a single-component model.

    Single-component model: one line, I'(nu) to the left of P(lambda)
    (I'(nu) shapes only the Stokes I spectrum there, not p=P/I itself, but
    the choice is still shown) -- except a *custom* model with SSA/thermal
    `shape1` (see _insert_opacity_factor), where selecting that shape adds
    a genuine e^{-tau_lambda(z)} attenuation into P(lambda) itself (see
    _custom_P_raw), shown here so the choice's effect on P(lambda), not
    just I'(nu), is visible.

    Two-component model: the w1/w2/epsilon definition on its own line
    first, then I'(nu) to the left of P(lambda) on the line below (matching
    the single-component layout, since P(lambda) itself is written in
    terms of w1/w2 -- see each two-component model's own `equation`).
    Custom models are always single-component, so the opacity factor above
    never applies here.

    A custom model's own `equation` (see build_custom_model) can itself
    already be more than one line -- e.g. its own phi(z)=... definition
    above the main P(lambda) integral -- in which case every line but the
    last is shown as-is, unchanged, and only the *last* line combines with
    I'(nu) (exactly like a plain single-line equation would on its own)."""
    i_prime = i_prime_latex(spec.n_components, shape1, shape2)
    lines = spec.equation.strip().split('\n')
    inner = lines[-1].strip()[1:-1]  # strip that line's own surrounding '$...$'
    if getattr(spec.func, 'is_custom', False) and shape1 in ('ssa', 'thermal'):
        inner = _insert_opacity_factor(inner)
    main_line = f'${i_prime}\\qquad\\quad {inner}$'
    extra_lines = lines[:-1]  # e.g. a custom model's own phi(z)=... line, if any
    if spec.n_components == 1:
        return '\n'.join(extra_lines + [main_line])
    line1 = f'${weights_latex(shape1, shape2)}$'
    return '\n'.join([line1] + extra_lines + [main_line])


# ── Custom user-defined models ────────────────────────────────────────────────
# Lets a user build a new single-component model at runtime straight from the
# general, *normalized* Sokoloff et al. 1998 (eq. 1) / Burn 1966 line-of-sight
# integral
#     P(lambda) = integral_{-1}^{1} j_p(z) dz / integral_{-1}^{1} j(z) dz
# instead of picking one of the closed-form models above -- normalized by the
# total (unpolarized) emissivity so the result doesn't depend on the emitting
# region's own integration bounds, only its shape. j_p(z) = emiss(z) *
# e^{2i*phi(z)*lambda^2} is the *polarized* emissivity -- rendered as j_p(z)
# in the builder dialog's own UI -- where emiss(z) is whatever plain-text math
# expression the user types into that field (not to be confused with the
# unrelated epsilon/eps already used elsewhere for a two-component model's
# own spectral blend weight, see spectral_weights/spectral_params). j(z), the
# plain (unpolarized) emissivity used for the denominator, is that same
# emiss(z) expression with p0 forced to 1, then |...| taken of whatever's
# left (see CUSTOM_P0_SYMBOL below, _custom_P_raw and build_custom_model) --
# a magnitude rather than also substituting chi0 to 0 and integrating the
# result directly, since chi0's own e^{2i*chi0} factor (and any other
# unit-magnitude phase emiss(z) might carry, e.g. a position-dependent
# intrinsic EVPA) already has |...|=1 and so can't survive an abs() anyway,
# while integrating a merely-chi0-zeroed expression that still carries some
# *other* complex structure could leave I_lambda itself complex -- and
# dividing P(lambda) by a complex I_lambda would wrongly rotate it, not
# just rescale it.
#
# p0 (fractional polarization amplitude), chi0 (EVPA) and phi0 (Faraday-depth
# scale) are ordinary symbols emiss(z)/phi'(z) can reference directly, not
# hidden/automatic factors applied outside the integral -- but they're still
# the model's three *standard* always-present sliders (fixed
# built-in-model-matching bounds, CUSTOM_P0_BOUNDS/CUSTOM_PHI0_BOUNDS below),
# not free constants the user has to pick a Kind/bounds for themselves (see
# discover_custom_params). The builder dialog starts a fresh model's emiss(z)
# field prefilled with 'p0 * exp(2 * i * chi0)' (multiplying the always-shown,
# non-editable e^{2i*phi(z)*lambda^2} phase factor) and phi'(z) prefilled
# with 'phi0', reproducing the same physical roles p0/chi0/phi0 always had --
# amplitude, EVPA, and Faraday-depth magnitude -- just typed out explicitly
# now instead of applied invisibly, and editable/removable like any other
# part of either expression from there.
#
# phi(z) itself -- the Faraday depth actually accumulated by a photon
# emitted at position z, the thing that actually appears in the integral
# above -- is *not* what the builder dialog's phi'(z) field holds. What
# the user types there is phi'(z), a Faraday-depth *density* (proportional
# to n_e*B_parallel, same role as a rotation measure per unit length). A
# photon emitted at z travels *forward* toward the observer (z=1), so it's
# only rotated by material *ahead* of it on that remaining path, not by
# anything behind it -- phi(z) is accordingly the remaining-path integral,
#     phi(z) = integral_{z}^{1} phi'(z') dz'
# built by custom_func via a cumulative trapezoid (_cumtrapz0, run forward
# from z=-1 and then subtracted from its own total to turn that prefix sum
# into the suffix one actually needed) over the same LOS quadrature grid
# used for the P(lambda) integral itself (which spans the full [-1,1], not
# just the emitting region, precisely so there's something to integrate
# over past it -- see custom_func), not read off phi'(z) directly.
#
# emiss(z) need not be real: 'i' parses as the imaginary unit (see
# _SYMPY_GLOBAL_DICT), so a profile like 'p0*exp(2*i*chi_z*z)' encodes an
# intrinsic EVPA that itself varies with position, not just an intensity
# envelope. phi'(z) (and so phi(z)) need not be real either: since phi(z)
# is used as the angle argument of exp(2i*phi(z)*lambda^2), an imaginary
# part on phi(z) becomes a genuine exp(+-real) factor there (via
# lambda^2), i.e. a wavelength-dependent exponential damping/growth on top
# of the ordinary real rotation -- the same role a Burn-style dispersion
# term plays, e.g. phi'(z)='phi0 + i*sigma'. A plain real phi'(z) is
# unaffected either way, since its own imaginary part is already all
# zeros.
#
# The emitting and rotating regions need not span the whole [-1,1] path, or
# even overlap: emiss(z)/j(z) are only ever evaluated (and integrated) over z
# in [j_lo, j_hi] -- zero outside it -- and phi'(z) is forced to zero outside
# [p_lo, p_hi], so phi(z) only picks up whatever of the remaining path lies
# within that window. j_lo/j_hi/p_lo/p_hi are set via the builder dialog's
# own bound boxes -- default j_lo=p_lo=-1, j_hi=p_hi=1, i.e. the full path
# -- and each box takes a plain-text expression exactly like emiss(z)/
# phi'(z) themselves (see build_custom_model, parse_custom_expr): typing a
# bare number there (still every default) behaves exactly as before, but
# an expression referencing a *new* constant (e.g. j_hi='3*w') is what
# promotes that constant into one of the model's own discovered params,
# with its own slider -- a bound only becomes indirectly adjustable this
# way, through a constant it's written in terms of, never as a slider of
# its own (see discover_custom_params). Whatever the current expression
# evaluates to (fixed for a bare number; re-evaluated on every call, see
# custom_func, if a referenced constant's slider moves) is what actually
# distinguishes internal from external
# Faraday rotation here (see the module discussion this was built from):
# p_lo<=j_hi puts a genuinely co-spatial (internal, differentially-
# rotating) stretch in the middle of the emitting region, so different
# emission depths pick up different amounts of rotation before reaching
# the observer -- true Burn-slab depolarization; p_lo>=j_hi makes every
# emission point's remaining path pass through the *entire* rotating
# region (since it all lies beyond where emission stops), so phi(z)
# collapses to one shared number: a pure external screen, switched on only
# *after* all emission has stopped, rotating everything by the same amount
# regardless of where within the source it originated. phi'(z) left blank
# (treated as the constant 1 -- see parse_custom_expr) at the default
# j_lo=p_lo=-1/j_hi=p_hi=1 is accordingly already a full-path internal
# screen, phi(z)=(1-z); push p_lo up past j_hi to turn the same phi'(z)=1
# into an external one instead (phi(z)=p_hi-p_lo, the same number for
# every z<=j_hi).
#
# emiss(z)/phi'(z) can also depend on frequency/wavelength directly, not just
# position: 'nu' (frequency, MHz) and 'lambda' (wavelength, m) are both
# reserved symbols available in either expression, alongside z -- e.g.
# emiss='(nu/1000)**(-0.7)' folds a spectral index straight into the
# emissivity profile itself, on top of (or instead of) the Spectrum box's
# own overall alpha (which still applies uniformly to the total Stokes I on
# top of whatever P(lambda) this integral produces).
Z_SYMBOL = sympy.Symbol('z', real=True)
NU_SYMBOL = sympy.Symbol('nu', positive=True)      # MHz, matches spectral_weights' own convention
LAMBDA_SYMBOL = sympy.Symbol('lambda', positive=True)  # m, matches x/wl elsewhere in this module

# The three standard always-present sliders (see the module comment above),
# available for emiss(z)/phi'(z) to reference by these names exactly like
# z/nu/lambda -- parse_custom_expr's own local_dict maps 'p0'/'chi0'/'phi0'
# straight to these fixed symbols (never to an unrelated same-named free
# constant), and discover_custom_params excludes them from the set of *new*
# constants a model needs a Kind/bounds picked for, the same way it already
# excludes Z_SYMBOL/NU_SYMBOL/LAMBDA_SYMBOL.
CUSTOM_P0_SYMBOL = sympy.Symbol('p0')
CUSTOM_CHI0_SYMBOL = sympy.Symbol('chi0')
CUSTOM_PHI0_SYMBOL = sympy.Symbol('phi0')

CUSTOM_P0_BOUNDS = (0.0, 0.7)      # matches every built-in model's own p_0 bounds
CUSTOM_PHI0_BOUNDS = (-5.0e6, 5.0e6)  # matches every built-in model's own phi bounds

# Default text a freshly-opened builder dialog's emiss(z)/phi'(z) fields
# start out with -- reproduces the model's old always-applied p0*e^{2i*chi0}
# amplitude/EVPA factor and phi0-scaled density exactly, just spelled out
# explicitly now that nothing applies them automatically (see the module
# comment above). Read by build_model.py; defined here purely so the string
# a fresh dialog starts with and the physical meaning build_custom_model
# gives p0/chi0/phi0 can't drift apart.
CUSTOM_EMISS_DEFAULT = 'p0 * exp(2 * i * chi0)'
CUSTOM_PHI_DEFAULT = 'phi0'

# 'lambda' is a Python keyword (anonymous-function syntax) -- sympy's parser
# tokenizes/eval()s through Python's own grammar, so the bare word 'lambda'
# in a typed expression is a syntax error before sympy ever sees it. Rewrite
# the standalone token (not a substring of another identifier, e.g.
# 'lambda0' -- \b won't match inside it) to this parser-safe placeholder
# before parsing; local_dict below still maps it straight to a genuine
# sympy Symbol literally named 'lambda' (Symbol('lambda') is just a string
# label -- sympy itself has no problem with it), so error messages/LaTeX
# output downstream still show 'lambda' to the user, and the original typed
# text (kept verbatim everywhere else, e.g. saved model files) is untouched.
_LAMBDA_TOKEN = '__polvista_lambda__'
_LAMBDA_TOKEN_RE = re.compile(r'\blambda\b')

CUSTOM_Z_N_MIN, CUSTOM_Z_N_MAX = 200, 10000  # quadrature grid size clamp, see _custom_quad_n
CUSTOM_Z_SAMPLES_PER_CYCLE = 16  # trapz accuracy target, see _custom_quad_n
CUSTOM_Z_PROBE_N = 64  # coarse probe-grid size, see _custom_quad_n

# 'delta(...)' and 'gaussian(z0, sigma_z)' in a typed j_p(z)/phi'(z)
# expression (see _SYMPY_GLOBAL_DICT's own 'delta'/'gaussian' entries below)
# parse to a genuine sympy.DiracDelta node and a gaussian node
# respectively -- both kept symbolic through discovery and the equation card
# (so they display as an actual delta/Gaussian, see
# custom_model_equation_lines and gaussian's own _latex) -- but
# neither is something lambdify or a discrete trapezoidal grid could
# evaluate as-is: DiracDelta has no numpy/scipy printer at all (confirmed
# against sympy 1.14.0, not just this repo's pinned 1.11.1 -- lambdify
# happily emits a call to a name that was never defined, raising a plain
# NameError the instant it's evaluated), and a literal point mass is 0 at
# every generic grid node and undefined at the one that might exactly
# coincide with it anyway. _lambdify_zw substitutes both, right before
# lambdify, for the standard narrow-Gaussian regularization,
# DiracDelta(x) = lim_{sigma->0} exp(-(x/sigma)^2)/(sigma*sqrt(pi)) -- exact
# in the same sense every other trapz integral here is already only
# approximate, just at a fixed, deliberately tiny sigma rather than a true
# limit for delta(...); gaussian(z0, sigma_z) is that exact same formula
# (re-centered at z0) with sigma_z itself typed by the user instead of
# fixed, i.e. delta(x) is conceptually gaussian(x, CUSTOM_DELTA_SIGMA) --
# kept as two separate mechanisms below only so delta(...) keeps its own
# literal DiracDelta glyph in the equation card rather than gaussian(...)'s.
CUSTOM_DELTA_SIGMA = 0.001
_CUSTOM_LOS_LENGTH = 2.0  # length of the [-1,1] LOS domain -- see custom_min_n below
# The oscillation-only heuristic in _custom_quad_n has no way to know
# emiss(z)/phi'(z) themselves contain a feature this narrow (it only looks
# at how fast phi(z)*lambda^2 oscillates) -- verified empirically (see the
# conversation this was built from) that trapz integrates a sigma-wide
# Gaussian to <0.1% error, robust to exactly where the peak falls relative
# to the grid, once there's at least roughly one grid point per sigma (n =
# domain_length/sigma); CUSTOM_GAUSS_MARGIN is a safety factor on top of
# that empirically-sufficient one-sample-per-sigma minimum. Applies to both
# delta(...) (whose sigma is always exactly CUSTOM_DELTA_SIGMA, so this only
# ever needs computing once, at Define-time -- CUSTOM_DELTA_MIN_N below) and
# gaussian(z0, sigma_z) (whose sigma_z can be an arbitrary expression --
# a literal number, or referencing one of the model's own discovered
# constants -- so it can only be resolved from the actual current parameter
# values at call-time, see custom_min_n).
CUSTOM_GAUSS_MARGIN = 1.25
CUSTOM_DELTA_MIN_N = int(np.ceil(CUSTOM_GAUSS_MARGIN * _CUSTOM_LOS_LENGTH / CUSTOM_DELTA_SIGMA))


class gaussian(sympy.Function):
    """A user-typed gaussian(z0, sigma_z) call -- `nargs = 2` is what makes
    sympy itself reject any other argument count (e.g. a stray
    'gaussian(z)') with a plain TypeError, already caught by
    parse_custom_expr's own try/except alongside every other parse error.
    Kept as a literal, undissolved node through discovery
    (free_symbols already walks into self.args, same as any other
    sympy.Function -- so a symbolic z0/sigma_z, e.g. 'gaussian(0.3, w)',
    surfaces 'w' as a discovered constant with no special-casing needed) and
    the equation card (see _latex below) -- only ever substituted to its
    own normalized Gaussian form at lambdify time, see _regularize_gaussian.
    Not evaluated/simplified automatically (no `eval` classmethod defined),
    so it survives symbolic manipulation (e.g. custom_model_equation_lines'
    own p0->1/chi0->0 .subs() for j(z)) fully intact."""
    nargs = 2

    def _latex(self, printer, *args):
        z0_latex = printer._print(self.args[0])
        sigma_latex = printer._print(self.args[1])
        return r'\mathcal{N}\!\left(%s,%s\right)' % (z0_latex, sigma_latex)


def _regularize_gaussian(expr):
    """`expr` with every gaussian(z0, sigma_z) node (a `gaussian` instance)
    replaced by its own normalized-Gaussian form,
    exp(-((z-z0)/sigma_z)^2)/(sigma_z*sqrt(pi)) -- the same regularization
    _regularize_delta applies for delta(...), just re-centered at z0 and at
    whatever sigma_z the user typed instead of the fixed CUSTOM_DELTA_SIGMA.
    A plain expr with no gaussian(...) in it comes back unchanged. See
    _lambdify_zw, the only caller."""
    return expr.replace(
        lambda e: isinstance(e, gaussian),
        lambda e: sympy.exp(-((Z_SYMBOL - e.args[0]) / e.args[1]) ** 2) / (e.args[1] * sympy.sqrt(sympy.pi)))


# app.py's own update_plot() redraws both plot tabs from the same
# (wl_ext, pars) on every slider drag -- ModelPlot calls custom_func(wl_ext,
# pars) directly, and StokesPlot's stokes_QU() calls it again with the exact
# same arguments to recover Q/U from P -- and repeats that per posterior-
# sample line (see sampling.N_POSTERIOR_DRAWS) in both tabs. None of that
# redundancy is visible to custom_func's callers, so it's deduplicated here
# instead: a small content-keyed cache (x's bytes + params, not object
# identity -- wl_ext is a fresh array every redraw, so identity would never
# hit) sized to comfortably hold one redraw's worth of distinct (x, params)
# pairs -- main curve + every posterior sample, in both the wl_ext-gridded
# and (Polar Stokes mode's own resampled) wl_qu-gridded calls -- without
# evicting entries still needed later in the same redraw.
CUSTOM_FUNC_CACHE_MAX = 128


def _custom_quad_n(depth_probe, lam2_max):
    """(n, underresolved): how many LOS quadrature points custom_func needs
    for this call's wavelength range, given the accumulated Faraday depth
    phi(z) (*not* the user-typed density phi'(z) -- see build_custom_model's
    own module comment -- already integrated by the caller, via
    _cumtrapz0) evaluated on a coarse probe grid
    spanning the model's own [0, z1] (see build_custom_model), and whether
    that need exceeds CUSTOM_Z_N_MAX (in which case `n` is clamped to it
    and the returned integral should be treated as unreliable -- see
    custom_func's own last_call_underresolved).

    e^{2i*phi(z)*lambda^2} oscillates across [0,z1] roughly
    ptp(phi(z))*2*lam2_max/(2*pi) times at the longest wavelength in `x`
    (lam2_max = max(x**2)); trapz needs several samples per cycle to avoid
    aliasing (CUSTOM_Z_SAMPLES_PER_CYCLE), clamped to
    [CUSTOM_Z_N_MIN, CUSTOM_Z_N_MAX] so a pathologically large phi(z)/lambda
    combination costs more time rather than an unboundedly expensive grid
    size inside a tight likelihood loop -- at the cost of the integral
    itself becoming unreliable past that point (flagged via `underresolved`
    rather than silently returned as if it were still accurate).

    This only tracks oscillation from the *phi(z)*lambda^2 term, using
    phi(z)'s min-to-max spread on the probe grid -- a phi(z) that itself
    wiggles faster than that probe can resolve (rather than just being
    large) is a pathological case outside what any fixed-size quadrature
    here can guarantee; keep phi'(z) itself smooth/monotonic-ish.

    depth_probe can be non-finite (e.g. phi'(z) divides by a user-defined
    constant whose bounds let it reach 0) -- np.ptp of an array containing
    inf is itself inf or nan (inf-inf is an indeterminate form), which
    would otherwise reach `int(np.clip(nan, ...))` below and raise
    ValueError: cannot convert float NaN to integer. Treat that the same
    as "needs more resolution than we'll ever give it": clamp to the max
    grid size and flag underresolved, same as genuinely needing more
    points than CUSTOM_Z_N_MAX -- the integral is unreliable either way,
    and custom_func/app.py's own warning label already say so instead of
    silently trusting it."""
    phase_span = float(np.ptp(depth_probe)) * 2.0 * lam2_max
    n_cycles = phase_span / (2.0 * np.pi)
    raw_n = n_cycles * CUSTOM_Z_SAMPLES_PER_CYCLE
    if not np.isfinite(raw_n):
        return CUSTOM_Z_N_MAX, True
    n = int(np.clip(raw_n, CUSTOM_Z_N_MIN, CUSTOM_Z_N_MAX))
    return n, raw_n > CUSTOM_Z_N_MAX


def _regularize_delta(expr):
    """`expr` with every DiracDelta(arg) node replaced by the standard
    narrow-Gaussian regularization, exp(-(arg/sigma)^2)/(sigma*sqrt(pi)) at
    sigma=CUSTOM_DELTA_SIGMA (see its own module-level comment for why a
    fixed tiny sigma rather than the true distributional limit, and for the
    empirical basis behind CUSTOM_DELTA_MIN_N, the quadrature-resolution
    floor this substitution only pays off with) -- a plain expr with no
    DiracDelta in it comes back unchanged. Used by _lambdify_zw, right
    before lambdify, so every numeric caller (custom_func, preview_los_
    profiles) gets this transparently from the one place, while
    parse_custom_expr/discover_custom_params/custom_model_equation_lines
    (equation-card display) all still see the original, literal
    DiracDelta node -- sympy.latex() already renders that as a proper
    delta glyph, which is far more legible than the expanded Gaussian
    formula would be."""
    return expr.replace(
        lambda e: isinstance(e, sympy.DiracDelta),
        lambda e: sympy.exp(-(e.args[0] / CUSTOM_DELTA_SIGMA) ** 2) / (CUSTOM_DELTA_SIGMA * sympy.sqrt(sympy.pi)))


def custom_uses_delta(emiss_expr, phi_expr):
    """Whether `emiss_expr`/`phi_expr` (already parsed, see
    parse_custom_expr) reference delta(...) at all -- see CUSTOM_DELTA_MIN_N's
    own module comment for why that matters for quadrature resolution.
    Shared by build_custom_model (to size custom_func's own resolved-pass
    grid) and the builder dialog (build_model.py's refresh_preview, to size
    its own preview grid the same way -- otherwise a CUSTOM_DELTA_SIGMA-wide
    spike would be just as invisible in the preview as in an unguarded
    model)."""
    return emiss_expr.has(sympy.DiracDelta) or phi_expr.has(sympy.DiracDelta)


def _lambdify_zw(expr, const_names):
    """A vectorized numpy callable fn(z, nu, lam, p0, chi0, phi0, *consts)
    for `expr` (already parsed, see parse_custom_expr) -- shared by
    build_custom_model's own registered model function and
    preview_los_profiles's dialog-facing preview. Every one of
    nu/lam/p0/chi0/phi0 is accepted even when `expr` doesn't actually
    reference it (a plain function of z, or of neither) -- lambdify is
    fine generating an unused parameter, and it lets every caller pass a
    consistent signature regardless of which of them a given
    emiss(z)/phi'(z) actually depends on. `expr` is regularized (see
    _regularize_delta/_regularize_gaussian) before lambdify -- DiracDelta
    itself has no numpy printer and would otherwise blow up the moment this
    callable is actually invoked, not when it's built; gaussian is
    never anything but this own module's own symbolic marker to begin with,
    so it always needs substituting before numpy can touch it at all."""
    expr = _regularize_gaussian(_regularize_delta(expr))
    return sympy.lambdify((Z_SYMBOL, NU_SYMBOL, LAMBDA_SYMBOL,
                            CUSTOM_P0_SYMBOL, CUSTOM_CHI0_SYMBOL, CUSTOM_PHI0_SYMBOL,
                            *sympy.symbols(const_names)),
                           expr, modules='numpy')


def _eval_zw(fn, z, nu, lam, p0, chi0, phi0, consts):
    """fn(z, nu, lam, p0, chi0, phi0, *consts), cast to a complex array or
    scalar -- shape is left to the caller's own broadcasting (z/nu/lam are
    already consistently broadcastable column/row arrays or plain scalars;
    a subsequent numpy op against a same-shaped or scalar result broadcasts
    fine either way, so there's nothing to force-reshape here). The
    dtype=complex cast guards against an expression that lambdifies to a
    bare Python float (one referencing none of z/nu/lam/p0/chi0/phi0 at
    all, e.g. a bare numeric literal) so every caller gets a consistent
    array type back regardless of what the user's own expression happens
    to produce.

    emiss(z) may now be genuinely complex -- 'i' parses as the imaginary
    unit (see _SYMPY_GLOBAL_DICT), letting a profile like
    'exp(2*i*chi_z*z)' encode a position-dependent intrinsic EVPA -- so
    this no longer collapses to the real part itself. phi'(z) (the Faraday
    depth *density* -- see build_custom_model's own module comment) may be
    complex too -- 'i' parses the same way there -- letting a profile like
    'phi0 + i*sigma' add a wavelength-dependent (via the eventual
    e^{2i*phi(z)*lambda^2} phase term's own lambda^2) exponential
    damping/growth on top of the ordinary real rotation, the same role a
    Burn-style dispersion term plays; callers that need the real part only
    (custom_func's own coarse resolution probe, and
    preview_los_profiles's plot) take `.real` on the result themselves
    rather than that being baked in here.

    `p0`/`chi0`/`phi0`/`consts` are cast to np.float64 (not left as plain
    Python floats) before the call -- if a user-defined constant is used as
    a denominator somewhere that doesn't otherwise involve z/nu/lambda
    (e.g. emiss(z) = 'p0/w', which is w on its own once lambdified), and
    its bounds let it reach 0, lambdify's generated code does that division
    between whatever types it's handed. Plain Python floats raise
    ZeroDivisionError there; np.float64 instead follows normal IEEE-754
    semantics and quietly produces inf, which downstream code (the
    caller's own .real, the LOS integral, and ultimately widgets.py's
    axis-limit math, see _finite_bounds) is already set up to tolerate
    without crashing."""
    consts = [np.float64(c) for c in consts]
    return np.asarray(fn(z, nu, lam, np.float64(p0), np.float64(chi0), np.float64(phi0), *consts),
                       dtype=complex)


def _lambdify_bound(expr, const_names):
    """A scalar numpy callable fn(p0, chi0, phi0, *consts) for one
    j_lo/j_hi/p_lo/p_hi bound `expr` (already parsed -- see
    parse_custom_expr and discover_custom_params's own z/nu/lambda
    rejection for these four specifically) -- shared by build_custom_model's
    own registered custom_func and the builder dialog's own preview (see
    preview_custom_bounds). Same shape as _lambdify_zw but with no z/nu/lam
    argument at all: a bound can't reference the line-of-sight position or
    frequency/wavelength, so it only ever needs p0/chi0/phi0/consts. Unlike
    _lambdify_zw, no delta(...)/gaussian(...) regularization here -- a
    bound expression can't reference z, so neither could ever appear in
    one to begin with (both are always functions of z)."""
    return sympy.lambdify((CUSTOM_P0_SYMBOL, CUSTOM_CHI0_SYMBOL, CUSTOM_PHI0_SYMBOL,
                            *sympy.symbols(const_names)), expr, modules='numpy')


def _eval_bound(fn, p0, chi0, phi0, consts):
    """float(fn(p0, chi0, phi0, *consts)) -- a single j_lo/j_hi/p_lo/p_hi
    number for one set of constants. np.float64 cast on every argument
    first, same reasoning as _eval_zw's own cast (a bound that divides by
    a user-defined constant whose own bounds let it reach 0, e.g.
    j_hi='1/w', would otherwise raise ZeroDivisionError under plain Python
    floats; np.float64 quietly produces inf instead, which the masking
    comparisons downstream (z>=j_lo etc., on a z grid that never itself
    leaves [-1,1]) already tolerate without crashing). `.real` in case the
    expression happens to lambdify to a complex dtype -- a bound is always
    a real position along the line of sight, but nothing enforces that
    while parsing (e.g. a bound expression could technically reference 'i'
    even though there's no physical reason to)."""
    consts = [np.float64(c) for c in consts]
    return float(np.real(fn(np.float64(p0), np.float64(chi0), np.float64(phi0), *consts)))


def _gaussian_sigma_fns(emiss_expr, phi_expr, const_names):
    """A lambdified fn(z, nu, lam, p0, chi0, phi0, *consts) (see
    _lambdify_zw) for every gaussian(z0, sigma_z)'s own sigma_z argument
    found in `emiss_expr`/`phi_expr` (already parsed, see
    parse_custom_expr) -- one per gaussian(...) call, empty if neither
    expression has one. delta(...) doesn't need this (its own sigma is
    always exactly the fixed CUSTOM_DELTA_SIGMA, resolved once at
    Define-time via CUSTOM_DELTA_MIN_N) -- gaussian(...)'s sigma_z can be
    an arbitrary expression (a literal number, or referencing one of the
    model's own discovered constants), so the grid resolution it needs can
    only be pinned down from the actual current parameter values, at
    call-time (see _gaussian_min_n, the only consumer of this)."""
    gaussians = emiss_expr.atoms(gaussian) | phi_expr.atoms(gaussian)
    return [_lambdify_zw(g.args[1], const_names) for g in gaussians]


def _gaussian_min_n(sigma_fns, z, nu, lam, p0, chi0, phi0, consts):
    """(n, underresolved) -- the quadrature floor needed to resolve every
    gaussian(...) in `sigma_fns` (see _gaussian_sigma_fns) at these
    particular parameter values, evaluating each one's own sigma_z over the
    given `z` grid (typically custom_func's own coarse z_probe -- sigma_z
    is ordinarily a plain constant, but nothing stops a user writing one
    that itself varies with z/nu/lambda, so this checks the whole grid
    rather than assuming a single value) and taking the smallest -- the
    narrowest gaussian(...) present is what sets how fine the grid needs to
    be, same logic as CUSTOM_DELTA_MIN_N's own derivation, just evaluated
    live instead of baked in once. (0, False) if `sigma_fns` is empty. A
    non-positive or non-finite sigma_z (e.g. a slider-linked constant
    dragged to 0, or past a pole in its own defining expression) can't be
    resolved by any finite grid -- treated the same as needing more points
    than CUSTOM_Z_N_MAX can ever supply, i.e. clamped to it and flagged
    underresolved rather than raising, exactly like _custom_quad_n's own
    non-finite-depth_probe case."""
    if not sigma_fns:
        return 0, False
    min_sigma = min(
        float(np.min(np.abs(_eval_zw(fn, z, nu, lam, p0, chi0, phi0, consts).real)))
        for fn in sigma_fns)
    if not (np.isfinite(min_sigma) and min_sigma > 0.0):
        return CUSTOM_Z_N_MAX, True
    n_needed = CUSTOM_GAUSS_MARGIN * _CUSTOM_LOS_LENGTH / min_sigma
    n = int(np.clip(np.ceil(n_needed), CUSTOM_Z_N_MIN, CUSTOM_Z_N_MAX))
    return n, n_needed > CUSTOM_Z_N_MAX


def custom_preview_n(emiss_expr, phi_expr, const_names, const_values,
                      p0_prev, chi0_prev, phi0_prev, default_n=300):
    """The quadrature n the builder dialog's own preview (see
    preview_los_profiles, build_model.py's refresh_preview) should use for
    `emiss_expr`/`phi_expr` at these preview constant values --
    `default_n` unless delta(...)/gaussian(...) needs more (see
    CUSTOM_DELTA_MIN_N/CUSTOM_GAUSS_MARGIN); folds custom_uses_delta's own
    fixed floor and _gaussian_min_n's own live one into the one call the
    dialog actually needs."""
    n = CUSTOM_DELTA_MIN_N if custom_uses_delta(emiss_expr, phi_expr) else default_n
    sigma_fns = _gaussian_sigma_fns(emiss_expr, phi_expr, const_names)
    if sigma_fns:
        z_probe = np.linspace(-1.0, 1.0, CUSTOM_Z_PROBE_N)
        n_gauss, _ = _gaussian_min_n(sigma_fns, z_probe, 0.0, 0.0, p0_prev, chi0_prev, phi0_prev, const_values)
        n = max(n, n_gauss)
    return n


def _cumtrapz0(y, z):
    """Cumulative trapezoidal integral of real- or complex-valued `y` along
    its first axis against the 1-D grid `z`, with a leading 0 (same
    convention as scipy.integrate.cumulative_trapezoid(y, z, axis=0, initial=0.0), which
    this replaces) -- `y` can be 1-D (custom_func's own coarse probe pass)
    or 2-D, i.e. one column per wavelength (its full-resolution pass; see
    custom_func/preview_los_profiles, the only callers).

    Unlike np.trapz -- used for the *other* LOS integral here, P(lambda)'s
    own outer one -- this can't just return one number: phi(z) has to
    supply a different running value at *every* z along that outer
    integral's own integrand (see build_custom_model's own module comment
    for why), not a single total. That running-sum is exactly what a
    cumulative trapezoid computes, so there's no `np.trapz`-shaped
    shortcut for it -- only a choice of how to compute it. This is a plain
    np.cumsum over each segment's own trapezoid area instead of calling
    scipy's own generic-axis cumulative_trapezoid, which pays for
    dispatch/argument-checking on every call that a fixed axis-0-only
    version here doesn't need to -- custom_func calls this twice per plot
    redraw or fit likelihood evaluation (the probe pass, then the resolved
    one, which can run up to CUSTOM_Z_N_MAX points), so that overhead is
    paid often enough to be worth trimming."""
    dz = np.diff(z)
    segments = dz * (y[:-1] + y[1:]) * 0.5 if y.ndim == 1 else dz[:, None] * (y[:-1] + y[1:]) * 0.5
    out = np.empty_like(y, dtype=np.result_type(y, dz))
    out[0] = 0.0
    np.cumsum(segments, axis=0, out=out[1:])
    return out


def _custom_opacity_attenuation(emiss_fn, consts, chi0_val, phi0_val, j_lo, j_hi,
                                 z, nu, lam, emiss_den_zw, shape, nu0, alpha_val, T_val, beta_val):
    """exp(-tau(z,nu)) -- the per-z opacity attenuation envelope
    _custom_P_raw's own SSA/thermal case multiplies onto both j_p(z) and
    j(z) before they're integrated (see there), built via Kirchhoff's law
    alpha'(z) = j(z)/S(nu) with S(nu) the *bare* local source function --
    the true, classical-radiative-transfer-sense j_nu/alpha_nu, not I'(nu)
    (intensity_shape(..., apply_escape=False) -- see its own docstring for
    why the default, already-escaped I'(nu) can't be reused here without
    double-applying the same escape-probability physics this per-z
    integral is itself computing).

    tau(z,nu) = integral_z^1 alpha'(z') dz' is built the same suffix-
    integral way phi(z) is in _custom_P_raw (_cumtrapz0 + total-minus-
    prefix), from j(z,nu) = `emiss_den_zw` (already computed, and already
    masked to 0 outside [j_lo,j_hi], by the caller), then rescaled by a
    single amplitude so that tau at the emitting region's far edge
    (z=j_lo, the model's own *total* column depth) equals 1 exactly at
    nu=nu0 -- preserving the same "nu0 is where the model turns over"
    meaning every other SSA/thermal model already has (see
    intensity_shape's own docstring). That requires j(z) evaluated at
    exactly nu=nu0 too (`emiss_den_z_nu0` below), independent of whatever
    wavelengths `nu`/`lam` this call actually needs -- mirrors stokes_I's
    own raw_ref pattern (re-evaluating at a fixed reference frequency)."""
    prefix_j = _cumtrapz0(emiss_den_zw, z[:, 0])
    w_zw = prefix_j[-1:, :] - prefix_j            # W(z,nu): (n_grid, n_lambda)
    s_bare_nu = intensity_shape(nu, nu0, alpha_val, shape, T=T_val, beta=beta_val,
                                 apply_escape=False)                          # (1, n_lambda)

    nu0_arr = np.array([[nu0]])
    lam0_arr = np.array([[C / nu0 / 1e6]])
    emiss_den_z_nu0 = np.abs(_eval_zw(emiss_fn, z, nu0_arr, lam0_arr, 1.0, chi0_val, phi0_val, consts))
    emiss_den_z_nu0 = np.where((z >= j_lo) & (z <= j_hi), emiss_den_z_nu0, 0.0)
    w_total_nu0 = np.trapz(emiss_den_z_nu0[:, 0], z[:, 0])
    s_bare_nu0 = intensity_shape(nu0_arr, nu0, alpha_val, shape, T=T_val, beta=beta_val,
                                  apply_escape=False)[0, 0]

    with np.errstate(divide='ignore', invalid='ignore'):
        amplitude = s_bare_nu0 / w_total_nu0 if w_total_nu0 > 0 else 0.0
        tau_zw = np.nan_to_num(amplitude * w_zw / s_bare_nu, nan=0.0, posinf=1e300, neginf=0.0)
    return np.exp(-tau_zw)


def _custom_P_raw(emiss_fn, phi_fn, consts, p0_val, chi0_val, phi0_val,
                   x, j_lo, j_hi, p_lo, p_hi, n,
                   shape='powerlaw', nu0=None, alpha_val=None, T_val=None, beta_val=None):
    """P(x) = integral_{j_lo}^{j_hi} j_p(z) dz / integral_{j_lo}^{j_hi} j(z) dz,
    the LOS integral custom_func actually needs, evaluated on an n-point z
    grid -- j_p(z) = emiss_fn(z) * e^{2i*phi(z)*lambda^2} (`emiss_fn` is the
    user's own emiss(z) expression, referencing p0/chi0/phi0 exactly as
    typed) and j(z) = |emiss_fn(z) with p0 forced to 1| -- the *magnitude*
    of that same expression once its own polarization amplitude is removed
    (see build_custom_model's own module comment for why a magnitude,
    rather than also substituting chi0->0 and integrating the result
    directly). Both are integrated over the same masked region so the
    result only depends on emiss(z)'s own *shape*, not the raw bounds it
    happens to be evaluated over (see the module comment above for why
    that normalization matters). Pulled out of custom_func as its own
    function purely to keep that already-long resolution/masking logic
    readable -- `x` there is only ever the subset of wavelengths
    custom_func has already determined are worth resolving (see its own
    `resolvable` mask), at whatever quadrature size `n` _custom_quad_n
    decided that subset needs.

    `shape` is the live Spectrum-box shape (models.SPECTRAL_SHAPE) custom_func
    passes in; only 'ssa'/'thermal' (with `nu0`/`alpha_val`/`T_val`/`beta_val`
    -- that component's own turnover/spectral-index/temperature/curvature)
    add a genuine per-z opacity attenuation on top of the always-optically-
    thin integral above (see _custom_opacity_attenuation) -- 'powerlaw'/
    'logparabola' behave exactly as before this existed. Opacity only
    changes P(lambda)'s own *shape* when something else in the integrand
    also varies with z (an internal phi'(z), or a spatial gradient in
    j_p(z) itself) -- a spatially-uniform emissivity with a purely external
    Faraday screen cancels it out exactly between this numerator and
    denominator, same as it already does for the plain spectral index (see
    stokes_I's own docstring)."""
    z = np.linspace(-1.0, 1.0, n)[:, None]      # (n_grid, 1) -- full LOS
    lam = x[None, :]                            # (1, n_lambda)
    nu = (C / x / 1e6)[None, :]                 # (1, n_lambda) MHz
    # emiss_num_zw is left complex -- j_p(z) may use 'i' for a
    # position-dependent intrinsic EVPA (e.g. 'exp(2*i*chi_z*z)'), and that
    # phase needs to survive into the integral below.
    emiss_num_zw = _eval_zw(emiss_fn, z, nu, lam, p0_val, chi0_val, phi0_val, consts)
    emiss_num_zw = np.where((z >= j_lo) & (z <= j_hi), emiss_num_zw, 0.0)
    # j(z): `emiss_fn` evaluated again with p0 forced to 1.0 (chi0 left at
    # its real value -- see build_custom_model's own module comment for why
    # that's fine), then |...| taken of the (possibly still complex)
    # result -- a second, independent numeric call rather than something
    # derived algebraically from emiss_num_zw above, since a user-typed
    # expression need not be linear in p0 (e.g. 'p0**2*exp(2*i*chi0)+w').
    emiss_den_zw = np.abs(_eval_zw(emiss_fn, z, nu, lam, 1.0, chi0_val, phi0_val, consts))
    emiss_den_zw = np.where((z >= j_lo) & (z <= j_hi), emiss_den_zw, 0.0)
    if shape in ('ssa', 'thermal'):
        atten_zw = _custom_opacity_attenuation(
            emiss_fn, consts, chi0_val, phi0_val, j_lo, j_hi, z, nu, lam, emiss_den_zw,
            shape, nu0, alpha_val, T_val, beta_val)
        emiss_num_zw = emiss_num_zw * atten_zw
        emiss_den_zw = emiss_den_zw * atten_zw
    # phi_prime_zw is left complex too -- phi'(z) (the Faraday-depth
    # *density*) may itself use 'i', e.g. 'phi0 + i*sigma' adding a
    # wavelength-dependent exponential damping/growth on top of the
    # ordinary real rotation once it reaches the phase term below (see
    # _eval_zw's own docstring for why); a plain real phi'(z) is
    # unaffected, since its own imaginary part is already all zeros.
    phi_prime_zw = _eval_zw(phi_fn, z, nu, lam, p0_val, chi0_val, phi0_val, consts)
    phi_prime_zw = np.where((z >= p_lo) & (z <= p_hi), phi_prime_zw, 0.0)
    # phi(z), the actual Faraday depth accumulated over the *remaining*
    # path to the observer -- _cumtrapz0 along the z axis (axis 0)
    # integrates each wavelength's own column independently, using the
    # same z grid for all of them (phi'(z) can depend on nu/lambda too, so
    # each column's integrand can differ), then total-minus-prefix turns
    # that prefix (source-side) integral into the suffix (observer-side)
    # one this model actually needs (phi0, if referenced by phi'(z) at
    # all, is already baked into phi_prime_zw itself -- see the module
    # comment above for why that's now enough, unlike the old design where
    # phi0 had to modulate phi(z) after the fact).
    prefix_zw = _cumtrapz0(phi_prime_zw, z[:, 0])
    phi_zw = prefix_zw[-1:, :] - prefix_zw
    phase = 2.0 * phi_zw * lam ** 2              # (n_grid, n_lambda) via broadcasting
    numerator = np.trapz(emiss_num_zw * np.exp(1j * phase), z[:, 0], axis=0)
    denominator = np.trapz(emiss_den_zw, z[:, 0], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return numerator / denominator


def preview_los_profiles(emiss_expr, phi_expr, const_names, const_values,
                          p0_prev, chi0_prev, phi0_prev,
                          nu_mhz, lambda_m, j_lo, j_hi, p_lo, p_hi, n=300,
                          shape='powerlaw', nu0=None, alpha_val=None, T_val=None, beta_val=None):
    """(z, emiss_p_z, phi_prime_z, phi_z, emiss_p_z_attenuated) arrays over
    z in [-1,1] for the custom-model builder dialog's own preview plot,
    already masked to emiss(z)=0 outside [j_lo,j_hi] and phi'(z)=0 outside
    [p_lo,p_hi] exactly like the registered model itself (see
    build_custom_model) -- `emiss_expr`/`phi_expr` already parsed (see
    parse_custom_expr), `const_values` a plain list of numbers in the same
    order as `const_names`, and `p0_prev`/`chi0_prev`/`phi0_prev` the preview values
    to evaluate p0/chi0/phi0 at if `emiss_expr`/`phi_expr` reference them
    (they're ordinary symbols now, not applied automatically -- see the
    module comment above). `nu_mhz`/`lambda_m` are the single
    frequency/wavelength (consistent with each other -- nu_mhz =
    C/lambda_m/1e6) to preview emiss(z)/phi'(z) at if either references
    'nu'/'lambda'.

    `emiss_p_z` is emiss_expr itself -- j_p(z)'s own envelope, before the
    always-appended e^{2i*phi(z)*lambda^2} phase factor -- and may be
    complex (see _eval_zw -- a profile using 'i' for a position-dependent
    intrinsic EVPA). The denominator's own j(z) (p0->1, chi0->0
    substituted, see build_custom_model) isn't previewed here at all --
    it only ever collapses to one number, I_lambda (see build_model.py's own
    intro text for what that is), which doesn't change P(lambda)'s own
    *shape*, only its overall normalization, so there's nothing about it
    a per-z preview plot would usefully show. `phi_prime_z` (`phi_expr`
    evaluated pointwise -- the Faraday-depth *density* the user actually
    types, phi0-inclusive if referenced) and `phi_z` (its accumulated
    integral -- see build_custom_model's own module comment) are always
    real. `phi_z` is the *remaining*-path integral, integral_{z}^{1}
    phi_prime_z' dz' (not the path already traveled): a photon emitted at
    z travels forward toward the observer at z=1 through whatever rotates
    ahead of it, not through what's behind it -- see custom_func's own
    matching prefix/total-minus-prefix construction for why. Not used by
    the registered model itself -- that's custom_func, built (with its own
    adaptive-resolution quadrature grid, not this fixed n=300 preview one)
    by build_custom_model below.

    `shape`/`nu0`/`alpha_val`/`T_val`/`beta_val` mirror _custom_P_raw's own
    opacity inputs, except sourced from the builder dialog's own preview-
    only Spectral-model dropdown (build_model.py) rather than a real,
    registered model's live Spectrum-box selection -- there is no such
    model yet at Define-time. When `shape` is 'ssa'/'thermal',
    `emiss_p_z_attenuated` is `emiss_p_z` with the same per-z e^{-tau(z)}
    envelope _custom_P_raw applies (via _custom_opacity_attenuation)
    multiplied in, so the builder's own preview plot can show what
    actually gets integrated once opacity is included, next to the
    unattenuated shape as typed; otherwise it's identical to `emiss_p_z`
    (no opacity to show)."""
    z = np.linspace(-1.0, 1.0, n)
    emiss_fn = _lambdify_zw(emiss_expr, const_names)
    phi_fn = _lambdify_zw(phi_expr, const_names)
    emiss_p_z = np.broadcast_to(
        _eval_zw(emiss_fn, z, nu_mhz, lambda_m, p0_prev, chi0_prev, phi0_prev, const_values), z.shape)
    phi_prime_z = np.broadcast_to(
        _eval_zw(phi_fn, z, nu_mhz, lambda_m, p0_prev, chi0_prev, phi0_prev, const_values).real, z.shape)
    emiss_p_z = np.where((z >= j_lo) & (z <= j_hi), emiss_p_z, 0.0)
    phi_prime_z = np.where((z >= p_lo) & (z <= p_hi), phi_prime_z, 0.0)
    prefix_z = _cumtrapz0(phi_prime_z, z)
    phi_z = prefix_z[-1] - prefix_z
    if shape in ('ssa', 'thermal'):
        emiss_den_z = np.abs(_eval_zw(emiss_fn, z, nu_mhz, lambda_m, 1.0, chi0_prev, phi0_prev, const_values))
        emiss_den_z = np.where((z >= j_lo) & (z <= j_hi), emiss_den_z, 0.0)
        atten_z = _custom_opacity_attenuation(
            emiss_fn, const_values, chi0_prev, phi0_prev, j_lo, j_hi,
            z[:, None], np.array([[nu_mhz]]), np.array([[lambda_m]]), emiss_den_z[:, None],
            shape, nu0, alpha_val, T_val, beta_val)[:, 0]
        emiss_p_z_attenuated = emiss_p_z * atten_z
    else:
        emiss_p_z_attenuated = emiss_p_z
    return z, emiss_p_z, phi_prime_z, phi_z, emiss_p_z_attenuated


def preview_custom_bounds(j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr, const_names, const_values,
                           p0_prev, chi0_prev, phi0_prev):
    """(j_lo, j_hi, p_lo, p_hi): numeric bound values for the builder
    dialog's own preview (see refresh_preview in build_model.py) -- the
    four bound expressions (already parsed, see parse_custom_expr and
    discover_custom_params's own z/nu/lambda rejection for them) evaluated
    at each discovered constant's own Preview value (`const_names`/
    `const_values`, same lists preview_los_profiles itself takes) and
    p0/chi0/phi0's own fixed preview constants -- exactly analogous to
    preview_los_profiles, just for the bounds instead of j_p(z)/phi'(z)
    themselves. The registered model's own custom_func evaluates the same
    four expressions the same way, just at each constant's *current
    slider* value instead of a fixed preview one -- see build_custom_model."""
    j_lo_fn = _lambdify_bound(j_lo_expr, const_names)
    j_hi_fn = _lambdify_bound(j_hi_expr, const_names)
    p_lo_fn = _lambdify_bound(p_lo_expr, const_names)
    p_hi_fn = _lambdify_bound(p_hi_expr, const_names)
    return (_eval_bound(j_lo_fn, p0_prev, chi0_prev, phi0_prev, const_values),
            _eval_bound(j_hi_fn, p0_prev, chi0_prev, phi0_prev, const_values),
            _eval_bound(p_lo_fn, p0_prev, chi0_prev, phi0_prev, const_values),
            _eval_bound(p_hi_fn, p0_prev, chi0_prev, phi0_prev, const_values))


# The custom-model builder dialog's "Kind" dropdown for a discovered
# constant maps straight onto these Param.kind values -- a custom constant
# marked this way gets exactly the same slider behavior (log-scale depth,
# log-scale dispersion, degree-display angle, percent-display fraction, a
# plain linear number, or -- 'freq'/'wave' -- a log-scale frequency
# [MHz]/wavelength [m], for a constant meant to set the model's own
# characteristic spectral scale, e.g. a break frequency inside j_p(z)
# itself, distinct from the shared nu0 turnover the main window's own
# Spectrum box controls) as a built-in model's own p_0/X_0/phi/dphi param,
# for free, via app.ParamSlider. 'freq'/'wave' deliberately aren't the
# same kind as the Spectrum box's own 'nu0' (which works -- and is labeled
# -- in GHz, converted to MHz only by app.py's own sync_spectrum_ui right
# before it reaches set_spectral_shape): a discovered constant's value
# reaches j_p(z)/phi'(z) with no such conversion step of its own, so
# giving it 'nu0' would silently hand a GHz-valued number to an expression
# where the reserved 'nu' symbol is MHz-valued -- 'freq' avoids that by
# being MHz (matching 'nu') from end to end, with its own Param.kind so
# app.ParamSlider can tell the two apart. eps/alpha/nu0/temp still aren't
# offered here -- a custom model always gets one alpha automatically
# (spectral_param_single) and has no second component to blend or shared
# turnover of its own to duplicate.
CUSTOM_PARAM_KINDS = ('p', 'X', 'phi', 'dphi', 'scale', 'freq', 'wave')

# Names that would collide with either the reserved line-of-sight
# coordinate or the model's own always-present spectral param (alpha, see
# spectral_param_single) if a user tried to reuse one as one of their own
# constants (see discover_custom_params) -- p0/chi0/phi0 are deliberately
# *not* here: they're ordinary symbols now (see CUSTOM_P0_SYMBOL et al.
# above), parsed straight to those fixed symbols by parse_custom_expr's own
# local_dict, so they can never reach discover_custom_params as a free
# symbol needing this check at all. 'X_0' stays reserved/blocked (rather
# than also mapped) since the name a user actually types for that slider is
# 'chi0', not the model's own internal Param key.
RESERVED_CUSTOM_NAMES = {'z', 'X_0', 'alpha'}

# Restricted eval() namespace for sympy's expression parser (see
# parse_custom_expr) -- only these names (plus the reserved z, and whatever
# new symbols the user's own expression introduces) are recognized;
# '__builtins__' is explicitly emptied so a stray "__import__(...)" or
# similar in a typed expression can't reach real Python builtins through
# eval()'s implicit globals injection (parse_expr parses via eval() under
# the hood -- sympy's own docs flag this as unsafe on untrusted input
# without a locked-down namespace like this one).
_SYMPY_ALLOWED_NAMES = ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
                         'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt', 'Abs',
                         'pi', 'E', 'Heaviside', 'Piecewise', 'Min', 'Max',
                         'erf', 'floor', 'ceiling', 'sign')
_SYMPY_GLOBAL_DICT = {name: getattr(sympy, name) for name in _SYMPY_ALLOWED_NAMES}

_SYMPY_GLOBAL_DICT['abs'] = sympy.Abs
# 'delta(...)' parses to a real sympy.DiracDelta node -- see CUSTOM_DELTA_SIGMA
# above for why it's kept symbolic here and only numerically regularized
# (to a narrow Gaussian) later, right before lambdify (_lambdify_zw).
_SYMPY_GLOBAL_DICT['delta'] = sympy.DiracDelta
# 'gaussian(z0, sigma_z)' parses to a gaussian node -- see its own
# class docstring and CUSTOM_GAUSS_MARGIN above.
_SYMPY_GLOBAL_DICT['gaussian'] = gaussian
# Lowercase 'i' (not sympy's own 'I') is the imaginary unit here, matching
# the equation card's own e^{2i*chi_0} notation -- lets emiss(z) (see the
# module comment above) be genuinely complex, e.g. 'p0*exp(2*i*chi_z*z)'
# for a position-dependent intrinsic EVPA, not just a real-valued profile.
# auto_symbol (see _SYMPY_TRANSFORMATIONS below) only auto-wraps a bare
# name into a new Symbol when it isn't already in this global_dict, so
# putting it here (rather than leaving it to fall through) is what keeps a
# typed 'i' from silently becoming a spurious free constant that
# discover_custom_params would then demand bounds for.
_SYMPY_GLOBAL_DICT['i'] = sympy.I
# standard_transformations' auto_symbol/auto_number rewrite bare names/number
# literals in the typed text into Symbol(...)/Integer(...)/Float(...)/
# Rational(...) calls in the code actually eval()'d -- these have to be in
# the namespace too, or every user-typed constant name fails to parse (not
# just an intentionally-blocked one).
for _n in ('Symbol', 'Integer', 'Float', 'Rational'):
    _SYMPY_GLOBAL_DICT[_n] = getattr(sympy, _n)
_SYMPY_GLOBAL_DICT['__builtins__'] = {}
# Deliberately *not* adding implicit_multiplication_application on top of
# standard_transformations: it splits any name ending in digits into
# symbol*number (e.g. 'p0' -> Symbol('p')*0, 'phi0' -> Symbol('phi')*0)
# instead of treating it as one identifier -- exactly the naming convention
# physics constants here use (p0, phi0, dphi0, ...). Users just need an
# explicit '*' for multiplication (e.g. 'p0*exp(-z)'), which is also less
# ambiguous than implicit multiplication in general.
_SYMPY_TRANSFORMATIONS = standard_transformations

CUSTOM_MODEL_DEFS = {}  # func -> its own build_custom_model() kwargs -- see app.py save/load


class CustomModelError(Exception):
    """Raised for anything wrong with a user-typed custom-model definition
    (bad syntax, a reserved name reused as a constant, missing bounds) --
    caught by the builder dialog and shown to the user instead of
    propagating."""


def parse_custom_expr(text, what):
    """Parse one user-typed math expression (`text`, e.g.
    'exp(-((z-0.5)/w)**2)') into a sympy expression, using only the
    restricted namespace in _SYMPY_GLOBAL_DICT plus the reserved
    line-of-sight/frequency/wavelength/standard-param symbols
    z/nu/lambda/p0/chi0/phi0 -- `what` ('j_p(z)' or "phi'(z)") is just for
    the error message. A blank/whitespace-only `text` is treated as the
    constant 1 (e.g. leaving phi'(z) empty for a purely internal,
    non-rotating emitting region).

    Raises CustomModelError on anything that doesn't parse, including a
    reference to a function polvista doesn't recognize. sympy's own parser
    raises a plain NameError for that specific case -- auto_symbol
    rewrites an unrecognized 'name(...)' call into 'Function(\"name\")(...)',
    and 'Function' itself isn't in the restricted namespace -- which would
    otherwise propagate uncaught past every caller here and crash the
    whole app instead of surfacing as an on-screen warning."""
    text = text.strip() if text else ''
    if not text:
        text = '1'
    processed = _LAMBDA_TOKEN_RE.sub(_LAMBDA_TOKEN, text)
    local_dict = {'z': Z_SYMBOL, 'nu': NU_SYMBOL, _LAMBDA_TOKEN: LAMBDA_SYMBOL,
                  'p0': CUSTOM_P0_SYMBOL, 'chi0': CUSTOM_CHI0_SYMBOL, 'phi0': CUSTOM_PHI0_SYMBOL}
    try:
        expr = parse_expr(processed, local_dict=local_dict, global_dict=dict(_SYMPY_GLOBAL_DICT),
                           transformations=_SYMPY_TRANSFORMATIONS)
    except (SyntaxError, TypeError, ValueError, AttributeError) as e:
        raise CustomModelError(f"Couldn't parse {what} = '{text}': {e}") from e
    except NameError as e:
        allowed = ', '.join(sorted(list(_SYMPY_ALLOWED_NAMES))+['abs', 'gaussian', 'delta'])
        raise CustomModelError(
            f"Couldn't parse {what} = '{text}': uses a function polvista doesn't recognize. "
            f"Supported functions are: {allowed}.") from e
    # DiracDelta(arg, k) -- its k-th *derivative* -- isn't something the
    # narrow-Gaussian regularization below (see CUSTOM_DELTA_SIGMA) covers;
    # only the plain delta(arg) form (a bare point mass) is supported.
    # Caught here, right after parsing, rather than left to surface as a
    # confusing failure once _lambdify_zw tries to substitute it.
    if any(len(d.args) != 1 for d in expr.atoms(sympy.DiracDelta)):
        raise CustomModelError(
            f"Couldn't parse {what} = '{text}': delta(...) only supports the plain form, "
            "delta(arg) -- not a derivative order, delta(arg, k).")
    return expr


def discover_custom_params(emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr):
    """Every free symbol used by `emiss_expr`/`phi_expr` and/or the four
    j_lo/j_hi/p_lo/p_hi bound expressions (all already parsed, see
    parse_custom_expr) other than the reserved z/nu/lambda/p0/chi0/phi0 --
    these become the custom model's own new *discovered* parameters
    (needing a Kind and bounds picked in the builder dialog's own table),
    in sorted (deterministic) order. A bound expression referencing a new
    constant (e.g. j_hi='3*w', reusing a width `w` that also shapes
    emiss(z)) is exactly how that constant ends up with a slider -- typing
    a plain number in a bound field (the default, and everything this
    replaces) introduces nothing new here, same as before. p0/chi0/phi0
    are excluded here (not listed as discovered parameters needing a
    Kind/bounds) precisely because they're already the model's own
    standard always-present sliders regardless of whether any expression
    actually references them by name -- see the module comment above and
    build_custom_model. Raises CustomModelError if any expression reuses a
    name reserved for something else entirely (see RESERVED_CUSTOM_NAMES)
    as one of its own constants, or if a bound expression references z/nu/
    lambda -- unlike emiss(z)/phi'(z), a bound has to evaluate to a single
    number for a given set of constants, not vary along the line of sight
    or with wavelength (custom_func evaluates it once per call, before the
    line-of-sight quadrature itself even starts -- see build_custom_model)."""
    for bound_name, bound_expr in (('j_lo', j_lo_expr), ('j_hi', j_hi_expr),
                                    ('p_lo', p_lo_expr), ('p_hi', p_hi_expr)):
        disallowed = bound_expr.free_symbols & {Z_SYMBOL, NU_SYMBOL, LAMBDA_SYMBOL}
        if disallowed:
            bad_names = ', '.join(sorted(s.name for s in disallowed))
            raise CustomModelError(
                f"{bound_name} can't reference {bad_names} -- an integration bound has to "
                "evaluate to a single number for a given set of constants, not vary along the "
                "line of sight or with wavelength.")
    symbols = ((emiss_expr.free_symbols | phi_expr.free_symbols | j_lo_expr.free_symbols
                | j_hi_expr.free_symbols | p_lo_expr.free_symbols | p_hi_expr.free_symbols)
               - {Z_SYMBOL, NU_SYMBOL, LAMBDA_SYMBOL, CUSTOM_P0_SYMBOL, CUSTOM_CHI0_SYMBOL, CUSTOM_PHI0_SYMBOL})
    names = sorted(s.name for s in symbols)
    bad = [n for n in names if n in RESERVED_CUSTOM_NAMES]
    if bad:
        raise CustomModelError(
            f"'{bad[0]}' is a reserved name (z is the line-of-sight coordinate; "
            "alpha is already the model's own spectral-index slider, applied automatically; "
            "'chi0' is the name for the model's own EVPA slider, not 'X_0') -- "
            "pick another constant name.")
    return names


def _slugify(label):
    """A valid-enough Python identifier for func.__name__/MODELS_BY_NAME,
    derived from a user-typed label -- e.g. 'My Screen!' -> 'custom_my_screen'."""
    slug = re.sub(r'[^0-9a-zA-Z]+', '_', label.strip()).strip('_').lower()
    if not slug or slug[0].isdigit():
        slug = f'model_{slug}' if slug else 'model'
    return f'custom_{slug}'


def unique_custom_name(base):
    """`base` (see _slugify) if it isn't already a registered model's
    func.__name__, otherwise base with '_2', '_3', ... appended until it
    is -- MODELS_BY_NAME is keyed by name, so two custom models can't
    share one."""
    if base not in MODELS_BY_NAME:
        return base
    i = 2
    while f'{base}_{i}' in MODELS_BY_NAME:
        i += 1
    return f'{base}_{i}'


def _custom_param_latex(name):
    """Best-effort LaTeX for a user-typed constant name -- sympy.latex()
    already subscripts a trailing digit run (e.g. 'phi0' -> '\\phi_{0}') for
    any name it recognizes as a greek letter spelled out in full, and just
    italicizes anything else."""
    return f'${sympy.latex(sympy.Symbol(name))}$'


def custom_model_equation_lines(emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr):
    """(phi_line, p_line) -- the two standalone equation-card LaTeX strings
    for a custom model built from `emiss_expr`/`phi_expr` (already parsed,
    see parse_custom_expr) at the given j_lo_expr/j_hi_expr/p_lo_expr/
    p_hi_expr (also already parsed -- may be a bare number, e.g. a plain
    Integer(-1), exactly like before this replaced literal floats, or a
    genuine expression referencing a discovered constant, e.g. '3*w'):
    phi(z)'s own
    integral form (its integrand is phi_expr's own text -- phi'(z), the
    Faraday-depth density the user actually types, possibly referencing
    phi0 directly, see the module comment above -- with z renamed to the
    dummy integration variable z' for display only), and the normalized
    P(lambda) = (1/I_lambda) * integral j_p(z) dz, with j_p(z) shown
    concretely (its own form is the whole point of a custom model, so it's
    shown explicitly rather than hidden behind a placeholder symbol) but
    phi(z) referenced only by name in the phase term -- its own concrete
    form is given by the other line, so restating it again here would be
    redundant. I_lambda itself is *not* defined anywhere in either line --
    see build_model.py's own intro text for that (the one place it's
    spelled out), so this card reads as "P(lambda) is this shape,
    normalized" without cluttering the one line that matters most (what
    j_p(z) itself looks like) with I_lambda's own derivation.

    j_p(z)'s own integral runs over [j_lo,j_hi] (it's 0 outside it, so
    integrating past there would add nothing); phi(z)'s own integral
    instead runs from max(z, p_lo) up to p_hi -- a photon emitted at z is
    only rotated by material *ahead* of it on its way out, not behind it,
    and phi'(z) itself is 0 outside [p_lo,p_hi] anyway (see
    build_custom_model's own masking) -- evaluated fresh at every point
    along the outer P(lambda) integral (see custom_func's own
    prefix/total-minus-prefix construction), not a single number, except
    when p_lo>=j_hi (a pure external screen): then max(z,p_lo)=p_lo for
    every emission point z<=j_hi, so every one of them integrates the
    exact same range and phi(z) collapses to one shared total.

    Used both for build_custom_model's own registered spec.equation (via
    custom_model_equation_latex below) and by the builder dialog's own
    side-by-side equation cards (which have no built model yet to read
    spec.equation off of -- just the six parsed expressions currently
    typed).

    matplotlib mathtext (not a full TeX engine, see latex_stuff.py)
    doesn't support \\displaystyle -- plain \\int/\\dfrac render fine, just
    inline-sized."""
    z_prime = sympy.Symbol("z'", real=True)
    phi_line = (r"$\phi(z)=\int_{\max(z,\,%s)}^{%s} %s\,dz'$"
                % (sympy.latex(p_lo_expr), sympy.latex(p_hi_expr),
                   sympy.latex(phi_expr.subs(Z_SYMBOL, z_prime))))
    p_line = (r'$P(\lambda)=\dfrac{1}{I_\lambda}\int_{%s}^{%s} %s\,e^{\,2i\phi(z)\lambda^2}\,dz$'
              % (sympy.latex(j_lo_expr), sympy.latex(j_hi_expr), sympy.latex(emiss_expr)))
    return phi_line, p_line


def custom_model_equation_latex(emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr):
    """The combined two-line (phi(z) above P(lambda)) equation-card LaTeX
    for build_custom_model's own registered spec.equation -- see
    custom_model_equation_lines for what each line means; this just joins
    them for callers (app.py's own main equation card) that display a
    model's whole equation as one stacked image rather than the builder
    dialog's own side-by-side pair."""
    phi_line, p_line = custom_model_equation_lines(
        emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr)
    return f'{phi_line}\n{p_line}'


def build_custom_model(label, emiss_str, phi_str, param_specs,
                        j_lo=-1.0, j_hi=1.0, p_lo=-1.0, p_hi=1.0, title=None, name=None):
    """Parse `emiss_str`/`phi_str` (plain-text j_p(z)/phi'(z) expressions
    -- `phi_str` is the Faraday-depth *density*, not the depth itself, see
    the module comment above) and `j_lo`/`j_hi`/`p_lo`/`p_hi` (see below --
    also plain text now, e.g. '-1' or '3*w', not necessarily a bare
    float), build the resulting model function, register it into
    MODELS/MODELS_BY_NAME under a unique name, and return that function.

    The returned model's params are always p0 (fractional polarization
    amplitude), X_0 (chi_0, EVPA), phi0 (Faraday-depth scale -- see
    CUSTOM_P0_BOUNDS/CUSTOM_PHI0_BOUNDS for their fixed bounds -- same
    p_0/X_0/phi order every closed-form model above uses), then whatever
    other constants `emiss_str`/`phi_str` themselves introduce, in that
    order. p0/chi0/phi0 need no entry in `param_specs` -- they're ordinary
    symbols `emiss_str`/`phi_str` can (and, via the builder dialog's own
    defaults, normally do) reference directly by name ('p0'/'chi0'/'phi0',
    see parse_custom_expr), evaluated at their own slider's current value
    exactly like any other constant, rather than applied as an automatic
    closed-form factor the way the old design did (see the module comment
    above).

    `param_specs` is a {constant_name: (kind, lo, hi[, description])} dict
    covering every *other* free symbol `emiss_str`/`phi_str` introduce,
    i.e. excluding the reserved z/nu/lambda/p0/chi0/phi0 (see
    discover_custom_params) -- `kind` one of CUSTOM_PARAM_KINDS (picked
    via the builder dialog's own "Kind" dropdown; see its module
    docstring), `lo`/`hi` already in that kind's own physical units
    (radians for 'X', a 0-1 fraction for 'p', rad/m^2 for 'phi', as-is for
    'scale' -- i.e. the same units app.ParamSlider expects in spec.bounds
    for that kind, not necessarily what the dialog displayed to the
    user). The optional trailing `description` (the dialog's own
    Description column, see build_model.py) becomes that constant's
    Param.description -- shown as its slider's tooltip in the main window
    exactly like p0/chi0/phi0's own fixed tooltips (see app.ParamSlider)
    -- falling back to a generic "User-defined constant '<name>'." when
    omitted (every 3-element tuple, e.g. from a 'custom_definition' saved
    before this) or left blank. Plain ValueError if any constant is
    missing or its kind isn't recognized (a caller bug, not a user-input
    problem -- the builder dialog always derives param_specs from
    discover_custom_params's own output first, restricted to a valid Kind
    dropdown selection).

    `j_lo`/`j_hi`/`p_lo`/`p_hi` fix emiss(z)/j(z)=0 outside [j_lo,j_hi] and
    phi'(z)=0 outside [p_lo,p_hi] -- a structural choice fixed at
    Define-time, not itself a slider of the returned model (see the module
    comment above for what the four actually encode physically). Each is
    parsed as its own expression exactly like emiss_str/phi_str (see
    parse_custom_expr) -- typing a bare number (e.g. '-1', still every
    default here) behaves exactly as it always did, but an expression that
    references a *new* constant (e.g. j_hi='3*w', tying the emitting
    region's own extent to a width `w` that also shapes emiss(z)) is what
    turns that constant into one of the model's discovered params, with
    its own Kind/bounds/slider, same as one used directly in emiss_str/
    phi_str -- see discover_custom_params. Unlike emiss_str/phi_str, a
    bound expression can't reference z/nu/lambda (CustomModelError if it
    does): it has to evaluate to a single number for a given set of
    constants, evaluated once per custom_func call rather than varying
    along the line-of-sight quadrature or with wavelength. `name`, if
    given, is used as-is for func.__name__ (re-registering a model loaded
    from a saved 'custom_definition' block under the name it already had,
    rather than slugifying `label` fresh every time -- see app.py's
    load_model_action); otherwise one is derived from `label` via
    unique_custom_name. `title` defaults to `label`.

    Raises CustomModelError for anything wrong with any of the six
    expressions themselves (bad syntax, reserved names, a bound
    referencing z/nu/lambda)."""
    emiss_expr = parse_custom_expr(emiss_str, 'j_p(z)')
    phi_expr = parse_custom_expr(phi_str, "phi'(z)")
    # str(...) rather than assuming a string outright -- every default
    # above, and every caller that hasn't been touched since bounds were
    # bare floats (e.g. an old saved 'custom_definition' block, or a
    # not-yet-updated call site), still passes a plain float here, and
    # parse_custom_expr's own text.strip() would otherwise raise
    # AttributeError on one. Kept around (not just the parsed exprs) for
    # CUSTOM_MODEL_DEFS below, so re-editing shows the exact text typed
    # (or, for a legacy float, its own str()) rather than reformatting it.
    j_lo_str, j_hi_str, p_lo_str, p_hi_str = str(j_lo), str(j_hi), str(p_lo), str(p_hi)
    j_lo_expr = parse_custom_expr(j_lo_str, 'j_lo')
    j_hi_expr = parse_custom_expr(j_hi_str, 'j_hi')
    p_lo_expr = parse_custom_expr(p_lo_str, 'p_lo')
    p_hi_expr = parse_custom_expr(p_hi_str, 'p_hi')
    const_names = discover_custom_params(emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr)
    missing = [n for n in const_names if n not in param_specs]
    if missing:
        raise ValueError(f'No bounds given for: {", ".join(missing)}')
    bad_kind = [n for n in const_names if param_specs[n][0] not in CUSTOM_PARAM_KINDS]
    if bad_kind:
        raise ValueError(f'Unrecognized kind for: {", ".join(bad_kind)}')

    # emiss_fn is j_p(z)'s own envelope exactly as typed (may reference
    # p0/chi0 directly) -- reused for *both* the numerator (evaluated at
    # p0/chi0's real values) and the denominator (evaluated at p0=1, then
    # |...| taken -- see _custom_P_raw's own docstring for why a magnitude
    # rather than also substituting chi0->0), so there's only ever the one
    # lambdified callable, not a separately-derived symbolic expression.
    emiss_fn = _lambdify_zw(emiss_expr, const_names)
    phi_fn = _lambdify_zw(phi_expr, const_names)
    # One scalar callable per bound, closed over the same const_names as
    # emiss_fn/phi_fn -- custom_func below evaluates all four fresh on
    # every call (see _eval_bound), so a bound expression referencing a
    # constant tracks that constant's *current* slider value exactly like
    # emiss_fn/phi_fn already do, not whatever it happened to be at
    # Define-time.
    j_lo_fn = _lambdify_bound(j_lo_expr, const_names)
    j_hi_fn = _lambdify_bound(j_hi_expr, const_names)
    p_lo_fn = _lambdify_bound(p_lo_expr, const_names)
    p_hi_fn = _lambdify_bound(p_hi_expr, const_names)
    n_const = len(const_names)
    # See CUSTOM_DELTA_MIN_N above for why custom_func needs to know this --
    # the oscillation-only quadrature heuristic below has no way to detect a
    # feature this narrow on its own.
    uses_delta = custom_uses_delta(emiss_expr, phi_expr)
    # gaussian(...)'s own sigma_z, unlike delta(...)'s fixed one, can only be
    # pinned down at call-time from the actual current parameter values (see
    # _gaussian_sigma_fns/_gaussian_min_n) -- built once here regardless, so
    # custom_func itself only has to lambdify-and-evaluate, not re-discover
    # which gaussian(...) calls exist on every call.
    gauss_sigma_fns = _gaussian_sigma_fns(emiss_expr, phi_expr, const_names)
    # A photon emitted at position z travels *forward* (toward the
    # observer at z=1) through whatever Faraday-rotating material lies
    # ahead of it, so the depth it actually picks up is the *remaining*-path
    # integral, phi(z) = integral_{z}^{1} phi'(z') dz' (phi' already masked
    # to 0 outside [p_lo,p_hi], so this is exactly integral_{p_lo}^{p_hi}
    # once z<=p_lo -- i.e. a single number, the same for every emission
    # point, exactly when the whole rotating region lies beyond the
    # emitting one: a pure external screen, p_lo>=j_hi). That "beyond
    # j_hi" contribution can matter even though emiss(z)/j(z) are 0 there,
    # so the quadrature grid has to span the full [-1,1] LOS (not just
    # [j_lo,j_hi]) for the phi(z) accounting to be right -- emiss_num_zw/
    # emiss_den_zw are masked to [j_lo,j_hi] explicitly below instead of
    # the grid doing it implicitly by never reaching outside that range.
    z_probe = np.linspace(-1.0, 1.0, CUSTOM_Z_PROBE_N)

    def custom_func(x, params):
        """params: p0, X_0, phi0, then this model's own constants (in
        const_names order), then the trailing spectral alpha -- see
        build_custom_model. p0/chi0/phi0 now reach emiss_fn/phi_fn exactly
        like any other constant (whichever of them each expression
        actually references, see the module comment above) -- there's no
        separate closed-form factor applied on top of the integral any
        more, unlike the old design. j_lo/j_hi/p_lo/p_hi are re-evaluated
        from their own parsed expressions on every call (via j_lo_fn/
        j_hi_fn/p_lo_fn/p_hi_fn, see build_custom_model) rather than read
        off a fixed closure float, so a bound expression that references
        one of this model's own discovered constants tracks that
        constant's current slider value like everything else here. Sets
        custom_func.last_call_underresolved on every call (see
        _custom_quad_n) -- app.py checks this after each plot redraw to
        warn the user when a wavelength this call *did* return a computed
        value for can't be trusted (see below for the ones it doesn't even
        try).

        (x, params, spectral-shape)-keyed cache (see CUSTOM_FUNC_CACHE_MAX)
        -- checked/filled around the actual computation below,
        transparently to every caller. SPECTRAL_SHAPE/SPECTRAL_NU0/
        SPECTRAL_TEMP/SPECTRAL_BETA have to be part of the key (not just
        `params`) now too -- unlike every other model, this one's own
        result can depend on them directly, via the SSA/thermal opacity
        term (see _custom_P_raw/_custom_opacity_attenuation); without this,
        flipping the main window's Spectrum box between Power-law and SSA
        for the same model/params would keep returning the *other* shape's
        already-cached P(lambda)."""
        cache_key = (x.tobytes(), tuple(np.asarray(params, dtype=np.float64)),
                     SPECTRAL_SHAPE, SPECTRAL_NU0, SPECTRAL_TEMP, SPECTRAL_BETA)
        cached = _cache.get(cache_key)
        if cached is not None:
            _cache.move_to_end(cache_key)
            result, custom_func.last_call_underresolved = cached
            return result

        def cache_and_return(result):
            _cache[cache_key] = (result, custom_func.last_call_underresolved)
            _cache.move_to_end(cache_key)
            if len(_cache) > CUSTOM_FUNC_CACHE_MAX:
                _cache.popitem(last=False)
            return result

        p0_val = params[0]
        chi0 = params[1]
        phi0_val = params[2]
        consts = params[3:3 + n_const]
        alpha_val = params[3 + n_const]
        # j_lo/j_hi/p_lo/p_hi are typically just the Define-time literal
        # again (every _eval_bound call below a no-op past the lambdify),
        # but re-evaluated here rather than reused from the closure so a
        # bound expression referencing one of this model's own discovered
        # constants (e.g. j_hi='3*w') tracks that constant's *current*
        # slider value -- see build_custom_model's own docstring.
        j_lo = _eval_bound(j_lo_fn, p0_val, chi0, phi0_val, consts)
        j_hi = _eval_bound(j_hi_fn, p0_val, chi0, phi0_val, consts)
        p_lo = _eval_bound(p_lo_fn, p0_val, chi0, phi0_val, consts)
        p_hi = _eval_bound(p_hi_fn, p0_val, chi0, phi0_val, consts)
        if x.size:
            lam_at_max = float(x[np.argmax(x ** 2)])
            nu_at_max = C / lam_at_max / 1e6
        else:
            lam_at_max = nu_at_max = 0.0
        # phi_fn is phi'(z), the Faraday-depth *density* shape the user
        # actually typed (phi0-inclusive if it references phi0 at all, see
        # the module comment above) -- the depth itself, phi(z) =
        # integral_{z}^{1} phi'(z') dz', is accumulated here via a
        # cumulative trapezoid (run forward, then subtracted from its own
        # total to get the remaining-path/suffix integral), not read off
        # phi_fn directly.
        phi_prime_probe = np.broadcast_to(
            _eval_zw(phi_fn, z_probe, nu_at_max, lam_at_max, p0_val, chi0, phi0_val, consts).real,
            z_probe.shape)
        phi_prime_probe = np.where((z_probe >= p_lo) & (z_probe <= p_hi), phi_prime_probe, 0.0)
        prefix_probe = _cumtrapz0(phi_prime_probe, z_probe)
        depth_probe = prefix_probe[-1] - prefix_probe
        ptp_depth = float(np.ptp(depth_probe))

        result = np.full(x.shape, np.nan, dtype=complex)
        # A wavelength needing more oscillation cycles across the LOS than
        # CUSTOM_Z_N_MAX/CUSTOM_Z_SAMPLES_PER_CYCLE can ever resolve isn't
        # just slow -- any value computed for it would be aliased noise,
        # already flagged unreliable by _custom_quad_n's own
        # `underresolved` (see below). Since more oscillation cycles is
        # exactly *why* a wavelength depolarizes here in the first place
        # (differential rotation between emission depths, or dispersion,
        # cancelling itself out through the LOS integral), a wavelength
        # that can never be resolved is -- by the same physics -- already
        # deep in that cancellation regime and effectively unpolarized.
        # Truncating the array to those still-resolvable up front (a plain
        # cutoff on x**2 itself, not on any oscillatory numerical estimate
        # that could alias) is what actually keeps every call bounded by
        # CUSTOM_Z_N_MAX regardless of how wide a wavelength range or how
        # large a phi0 gets selected -- unlike trying to mask by a coarse
        # numerical |P| estimate instead, which (verified while building
        # this) can itself alias at exactly these extreme oscillation
        # counts and wrongly let a deeply-depolarized wavelength "survive",
        # right back to forcing everyone else's resolution up to match it.
        if ptp_depth > 0.0:
            lam2_afford = (CUSTOM_Z_N_MAX / CUSTOM_Z_SAMPLES_PER_CYCLE) * np.pi / ptp_depth
            resolvable = x ** 2 <= lam2_afford
        else:
            resolvable = np.ones(x.shape, dtype=bool)  # phi(z) is 0 everywhere -- nothing ever oscillates
        if not np.any(resolvable):
            custom_func.last_call_underresolved = False
            return cache_and_return(result)
        x_res = x[resolvable]

        n, underresolved = _custom_quad_n(depth_probe, float(np.max(x_res ** 2)))
        if uses_delta:
            # The phi-oscillation heuristic above has no idea emiss(z)/
            # phi'(z) themselves contain a CUSTOM_DELTA_SIGMA-wide feature --
            # force at least enough points to actually resolve it (see
            # CUSTOM_DELTA_MIN_N's own module comment for the empirical
            # basis), on top of whatever the heuristic already decided.
            n = max(n, CUSTOM_DELTA_MIN_N)
        if gauss_sigma_fns:
            # Same idea, but gaussian(...)'s own sigma_z isn't fixed --
            # re-derive the floor from whatever it actually evaluates to at
            # *this* call's params (see _gaussian_min_n), reusing the same
            # coarse z_probe grid the phi-oscillation estimate above just
            # used. A gaussian(...) dragged narrower than CUSTOM_Z_N_MAX can
            # resolve doesn't raise -- it's folded into `underresolved`
            # exactly like an unresolvable phi-oscillation case already is,
            # so the result still comes back (just flagged unreliable)
            # rather than the call failing outright.
            n_gauss, underresolved_gauss = _gaussian_min_n(
                gauss_sigma_fns, z_probe, nu_at_max, lam_at_max, p0_val, chi0, phi0_val, consts)
            n = max(n, n_gauss)
            underresolved = underresolved or underresolved_gauss
        custom_func.last_call_underresolved = underresolved

        # The full normalized result -- integral j_p(z) dz / integral j(z)
        # dz -- comes straight back from _custom_P_raw: p0/chi0 (and phi0,
        # for phi_fn) are already applied inside it, via emiss_fn/phi_fn
        # being called with their actual current values, not as a separate
        # closed-form factor the way the old design applied p0*e^{2i*chi0}
        # here. shape/nu0/T/beta are read straight off the live, shared
        # Spectrum-box globals (same ones stokes_I itself reads) rather than
        # threaded through params -- opacity is gated purely on whichever
        # shape the main window currently has selected for this model, not
        # a choice baked in at Define-time (see build_model.py's own
        # Spectral-model dropdown, which only ever previews this).
        nu_res = C / x_res / 1e6
        result[resolvable] = _custom_P_raw(
            emiss_fn, phi_fn, consts, p0_val, chi0, phi0_val, x_res, j_lo, j_hi, p_lo, p_hi, n,
            shape=SPECTRAL_SHAPE, nu0=reference_nu(nu_res),
            alpha_val=alpha_val, T_val=SPECTRAL_TEMP, beta_val=SPECTRAL_BETA)
        return cache_and_return(result)

    _cache = OrderedDict()  # see CUSTOM_FUNC_CACHE_MAX; one cache per custom model, not shared
    custom_func.last_call_underresolved = False
    custom_func.is_custom = True  # app.py's Fit!/Sampling guards key off this

    func_name = name if name is not None else unique_custom_name(_slugify(label))
    custom_func.__name__ = func_name
    custom_func.__qualname__ = func_name

    title = title or label
    # param_specs[n][3] (the dialog's own Description column, see
    # build_model.py) if present and non-blank, else a generic fallback --
    # covers both a 3-element tuple (no description column existed yet
    # when it was written, e.g. a 'custom_definition' saved before this)
    # and a 4-element one where the user simply left it blank.
    const_descriptions = {n: (param_specs[n][3] if len(param_specs[n]) > 3 and param_specs[n][3]
                               else f"User-defined constant '{n}'.")
                           for n in const_names}
    params = ([Param('p0', r'$p_0$', 'p', "Intrinsic fractional polarization amplitude -- referenced as 'p0' in j_p(z)."),
               Param('X_0', r'$\chi_0$', 'X', "Intrinsic EVPA (overall constant phase) -- referenced as 'chi0' in j_p(z)."),
               Param('phi0', r'$\phi_0$', 'phi', "Faraday-depth scale -- referenced as 'phi0' in j_p(z)/phi'(z).")]
              + [Param(n, _custom_param_latex(n), param_specs[n][0], const_descriptions[n])
                 for n in const_names]
              + spectral_param_single())
    lo = ([CUSTOM_P0_BOUNDS[0], -np.pi / 2, CUSTOM_PHI0_BOUNDS[0]]
          + [param_specs[n][1] for n in const_names] + SPECTRAL_BOUNDS_LO_SINGLE)
    hi = ([CUSTOM_P0_BOUNDS[1], np.pi / 2, CUSTOM_PHI0_BOUNDS[1]]
          + [param_specs[n][2] for n in const_names] + SPECTRAL_BOUNDS_HI_SINGLE)
    equation = custom_model_equation_latex(emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr)

    register(custom_func, label=label, title=title, params=params, bounds=(lo, hi),
             n_components=1, n_live_points=1000, equation=equation)

    CUSTOM_MODEL_DEFS[custom_func] = {
        'label': label, 'title': title, 'emiss_expr': emiss_str, 'phi_expr': phi_str,
        'param_specs': {n: list(param_specs[n]) for n in const_names},
        'j_lo': j_lo_str, 'j_hi': j_hi_str, 'p_lo': p_lo_str, 'p_hi': p_hi_str,
    }
    return custom_func


