"""Faraday-depolarization models and their metadata.

"""
import numpy as np
from dataclasses import dataclass
from typing import Callable

C = 299792458.0  # m/s


# ── Spectral shape: normalized source function S'(nu) ─────────────────────────
# Every model's spectral weighting (spectral_weights for two-component
# models, stokes_I's single-component branch) shapes a component's relative
# intensity via a normalized source function S'(nu), S'(nu0)=1. The standard
# choice is a power law (nu/nu0)**alpha, with nu0 auto-set to the lowest
# frequency in the currently plotted band; the alternative is the classic
# synchrotron self-absorbed (SSA) spectrum, which has no band edge to fall
# back on and needs an explicit turnover frequency nu0 instead.
#
# Module-level rather than threaded through every model func's fixed
# (x, params) signature -- there's only ever one "current" shape/nu0 in this
# single-window app, set by the Spectrum box's dropdown (see
# app.MainWindow.on_spectral_shape_changed).
SPECTRAL_SHAPE = 'powerlaw'
SPECTRAL_NU0 = None    # MHz -- SSA turnover frequency: a single-component
                         # model's own nu0, or a two-component model's
                         # component-1 nu0 (see set_spectral_shape)
SPECTRAL_NU0_2 = None   # MHz -- two-component SSA models' component-2
                         # turnover frequency, independent of SPECTRAL_NU0


def set_spectral_shape(shape, nu0=None, nu0_2=None):
    """Select the source function S'(nu) used by every model's spectral
    weighting from here on: 'powerlaw' (the default; nu0 auto-derived per
    call, always shared between both components of a two-component model --
    see reference_nu/component_reference_nu) or 'ssa' (classic
    synchrotron self-absorption; `nu0` [MHz] is then required -- there's no
    band edge to default to).

    `nu0` is a single-component model's own turnover frequency, or a
    two-component model's component-1 turnover frequency; `nu0_2` is
    component 2's own, independent turnover frequency -- unlike the
    power-law shape, an SSA two-component model may have each component
    turn over at a different frequency. Falls back to `nu0` when `nu0_2`
    isn't given."""
    global SPECTRAL_SHAPE, SPECTRAL_NU0, SPECTRAL_NU0_2
    SPECTRAL_SHAPE = shape
    SPECTRAL_NU0 = nu0
    SPECTRAL_NU0_2 = nu0_2 if nu0_2 is not None else nu0


def band_nu0(nu, nu_min):
    """The shared power-law reference frequency: `nu_min` if given,
    otherwise the lowest frequency spanned by `nu` itself."""
    return nu_min if nu_min is not None else np.min(nu)


def reference_nu(nu, nu_min=None):
    """The nu0 [MHz] that anchors S'(nu0)=1 for a single-component model:
    the SSA turnover frequency set via set_spectral_shape when that shape
    is active, otherwise the shared band nu0 (see band_nu0)."""
    if SPECTRAL_SHAPE == 'ssa' and SPECTRAL_NU0 is not None:
        return SPECTRAL_NU0
    return band_nu0(nu, nu_min)


def component_reference_nu(nu, nu_min, nu0_override):
    """The nu0 [MHz] that anchors one component's S'(nu0)=1 in a
    two-component model: `nu0_override` (that component's own SSA turnover
    frequency, set via set_spectral_shape) when SSA is active and it's set,
    otherwise the shared band nu0 (see band_nu0) -- i.e. only 'ssa' lets
    the two components use different reference frequencies; 'powerlaw'
    always shares one."""
    if SPECTRAL_SHAPE == 'ssa' and nu0_override is not None:
        return nu0_override
    return band_nu0(nu, nu_min)


def source_function(nu, nu0, alpha):
    """Normalized source function S'(nu), S'(nu0)=1, in the currently
    selected shape (see set_spectral_shape).
    'powerlaw': (nu/nu0)**alpha.
    'ssa': classic synchrotron self-absorbed spectrum (nu/nu0)**(5/2) *
    (1-e^-tau_nu)/(1-e^-tau_0), tau_0 fixed to 1 and tau_nu = tau_0*
    (nu/nu0)**(alpha-5/2) the frequency-dependent opacity -- the -5/2 offset
    is what makes alpha the actual optically-thin spectral index (nu >>
    nu0): S'(nu) there is ~ (nu/nu0)**(5/2) * tau_nu ~ (nu/nu0)**alpha."""
    ratio = nu / nu0
    if SPECTRAL_SHAPE == 'ssa':
        tau0 = 1.0
        tau_nu = tau0 * ratio ** (alpha - 2.5)
        # (1-e^-x) = -expm1(-x); tau0's constant denominator cancels the
        # sign, and expm1 keeps this well-behaved as tau_nu -> 0.
        return ratio ** 2.5 * np.expm1(-tau_nu) / np.expm1(-tau0)
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
    component's own source function S'(nu) (see
    source_function/set_spectral_shape):
        w1 = eps * S1'(nu),  w2 = (1-eps) * S2'(nu)
    w1+w2 is thus the (normalized) total Stokes I(nu) of the two-component
    system; see stokes_I().

    Under the power-law shape both components always share one reference
    frequency nu0 (`nu_min` [MHz] if given, else the *lowest* frequency
    spanned by x -- the longest wavelength currently plotted); under 'ssa'
    each component instead uses its own turnover frequency, set via
    set_spectral_shape (see component_reference_nu). Pass an explicit
    `nu_min` to anchor the power-law case elsewhere (e.g. to loaded data's
    own nu_min so the Stokes I/Q/U display doesn't drift when the plotted
    wavelength range changes -- see MainWindow.data_nu_min)."""
    nu = C / x / 1e6  # MHz
    nu0_1 = component_reference_nu(nu, nu_min, SPECTRAL_NU0)
    nu0_2 = component_reference_nu(nu, nu_min, SPECTRAL_NU0_2)
    w1 = eps * source_function(nu, nu0_1, alpha1)
    w2 = (1.0 - eps) * source_function(nu, nu0_2, alpha2)
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


def evpa(fit):
    """EVPA [deg] of a complex fractional polarization array P = q+iu (e.g.
    a model's own p*e^(2i*chi) output)."""
    return np.degrees(0.5 * np.arctan2(fit.imag, fit.real))


def pol(fit):
    """Fractional linear polarization [%], |P|, of a complex fractional
    polarization array P = q+iu."""
    return 100 * np.sqrt(fit.real ** 2 + fit.imag ** 2)


def stokes_I(wl, n_components, pars, nu_min=None):
    """Normalized Stokes I(nu) implied by a model's own trailing spectral
    params (the last 1 for a single-component model: alpha; the last 3
    for a two-component model: eps,alpha1,alpha2) -- no separate I_0
    needed, this is a *display/export* shape with amplitude 1 at nu_min
    (the lowest frequency spanned by wl, or an explicit override -- see
    below), regardless of shape ('powerlaw'/'ssa'). For a single component
    the spectral index has no effect on p=P/I, but it does shape the total
    intensity spectrum itself, via S'(nu) (see
    source_function/set_spectral_shape); for two components this is the
    same w1+w2 total weight that already governs how the polarization
    blends between them.

    Anchoring at nu_min rather than at each component's own physics-level
    reference frequency (nu0 for 'powerlaw', which usually *is* nu_min
    anyway; each component's own SSA turnover otherwise) is deliberate: it
    keeps this normalization matched to the Load Data path's own I(nu_min)
    =1 convention (see MainWindow.load_data_action) regardless of where an
    SSA turnover has been placed -- an SSA nu0 can sit many decades outside
    the plotted/loaded band by design (see app.NU0_BOUNDS_MULT), and
    anchoring the display there instead would let the *displayed* I(nu)
    swing to arbitrarily large or small values with no relation to what's
    actually on screen or in the loaded data. The underlying shape (via
    spectral_weights/source_function) still fully reflects nu0 -- only
    where the curve reads "1" moves.

    `nu_min` [MHz] defaults to the lowest frequency spanned by wl (the
    longest wavelength currently plotted); pass an explicit value (e.g. the
    loaded data's own nu_min) to anchor the normalization elsewhere."""
    nu = C / wl / 1e6  # MHz
    if nu_min is None:
        nu_min = np.min(nu)
    nu_min_arr = np.array([nu_min])
    wl_ref = C / (nu_min * 1e6)  # wl [m] at nu_min, for the two-component branch below

    if n_components == 1:
        alpha = pars[-1]
        nu0 = reference_nu(nu, nu_min)
        raw = source_function(nu, nu0, alpha)
        raw_ref = source_function(nu_min_arr, nu0, alpha)[0]
    else:
        eps, alpha1, alpha2 = pars[-3:]
        w1, w2 = spectral_weights(wl, eps, alpha1, alpha2, nu_min=nu_min)
        raw = w1 + w2
        w1_ref, w2_ref = spectral_weights(np.array([wl_ref]), eps, alpha1, alpha2, nu_min=nu_min)
        raw_ref = (w1_ref + w2_ref)[0]
    return raw / raw_ref


def stokes_QU(wl, model, n_components, pars, nu_min=None):
    """Physical (I_0=1-normalized) Stokes Q(nu), U(nu): model() already
    returns the fractional complex polarization P/I, so multiplying back
    by the model's own Stokes I(nu) recovers genuine Q, U (not just Q/I,
    U/I) -- e.g. for a single component with p constant in nu, Q and U
    simply trace I(nu)'s power-law shape, modulated by cos/sin(2*EVPA).
    `nu_min` [MHz] is forwarded to stokes_I() -- see there."""
    fit = model(wl, pars)
    I = stokes_I(wl, n_components, pars, nu_min=nu_min)
    return fit.real * I, fit.imag * I


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
    (component 1's flux fraction I1/(I1+I2) at nu_min, the lowest
    frequency/longest wavelength currently plotted) and each component's
    own power-law spectral index, used to weight that component's
    contribution to the total by (nu/nu_min)**alpha. alpha1=alpha2=0
    reduces to a plain eps/(1-eps) blend at every wavelength (see
    spectral_combine)."""
    return [Param('epsilon', r'$\varepsilon$', 'eps',
                  "Component 1's fraction of total intensity I1/(I1+I2) at the "
                  "lowest plotted frequency; sets which component dominates"),
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
    n_components=1,
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
    n_components=1,
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
    n_components=1,
    equation=r'$P(\lambda)=p_0\left[f\,e^{-2\sigma_\phi^2\lambda^4}e^{\,2i(\chi_0+\phi\lambda^2)}+(1-f)\,e^{2i\chi_0}\right]$')


register(intern,
    label='Internal screen with depolarization', title='Internal screen',
    params=[Param('p_0', r'$p_0$', 'p', 'Intrinsic fractional polarization.'),
            Param('X_0', r'$\chi_0$', 'X', 'Intrinsic EVPA (polarization angle) at lambda=0.'),
            Param('phi', r'$\phi$', 'phi', 'Faraday depth (RM-like): sets how fast EVPA rotates with lambda^2.'),
            Param('dphi', r'$\sigma_{\phi}$', 'dphi', 'Faraday depth dispersion across the internal, emission-mixed screen; drives depolarization at long wavelengths.')] + spectral_param_single(),
    bounds=([0.0, -np.pi / 2, -5e6, 0] + SPECTRAL_BOUNDS_LO_SINGLE,
            [0.7, np.pi / 2, 5e6, 5e6] + SPECTRAL_BOUNDS_HI_SINGLE),
    n_components=1,
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
    n_components=1,
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
    n_components=2,
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
    n_components=2,
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
    n_components=2,
    equation=(r'$P(\lambda)=\frac{w_1}{w_1+w_2}p_1e^{2i\chi_1}\left[\frac{1-e^{-\left(2\sigma_{\phi,1}^2\lambda^4-2i\phi_1\lambda^2\right)}}{2\sigma_{\phi,1}^2\lambda^4-2i\phi_1\lambda^2}\right]'
              r'+\frac{w_2}{w_1+w_2}p_2e^{-\sigma_{\phi,2}^2\lambda^4}e^{\,2i(\chi_2+\phi_2\lambda^2)}$'))


# ── Equation-card display: prepend the chosen S'(nu) definition to a model's
# own polarization equation, and -- for two-component models -- the
# w1/w2/epsilon definition built from it, on its own line above. Built
# dynamically (not baked into ModelSpec.equation at registration time) since
# the shape ('powerlaw'/'ssa') is a runtime choice -- see
# app.MainWindow.on_spectral_shape_changed.
#
# 'powerlaw' always shares one nu0 between both components of a
# two-component model (matches the Polvista paper); 'ssa' lets each component
# turn over at its own nu_{0,i} (see set_spectral_shape/
# component_reference_nu), so its two-component S' definition and epsilon
# are written with an explicit per-component subscript instead. The
# frequency-dependent opacity tau_nu isn't shown as its own term -- tau_0 is
# fixed to 1, so it's inlined directly into S'(nu) instead of introducing it
# as a separate symbol.
S_PRIME_POWERLAW = r"S'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{\alpha}"
S_PRIME_SSA_ONE = (r"S'(\nu)=\left(\dfrac{\nu}{\nu_0}\right)^{5/2}"
                    r"\left[\dfrac{1-e^{-(\nu/\nu_0)^{\alpha-5/2}}}{1-e^{-1}}\right]")
S_PRIME_SSA_TWO = (r"S_i'(\nu)=\left(\dfrac{\nu}{\nu_{0,i}}\right)^{5/2}"
                    r"\left[\dfrac{1-e^{-(\nu/\nu_{0,i})^{\alpha_i-5/2}}}{1-e^{-1}}\right]\ \ (i=1,2)")


def s_prime_latex(n_components, shape):
    """LaTeX (no surrounding '$') for the S'(nu) source-function definition
    a given `shape` ('powerlaw'/'ssa') implies."""
    if shape == 'ssa':
        return S_PRIME_SSA_TWO if n_components == 2 else S_PRIME_SSA_ONE
    return S_PRIME_POWERLAW


def weights_latex(shape):
    """LaTeX (no surrounding '$') for a two-component model's w1/w2/epsilon
    line: how S'(nu) combines into each component's weight w_i and the
    epsilon that sets w1 vs w2 (Polvista paper Eq. 9-11)."""
    if shape == 'ssa':
        eps_def = r"\varepsilon=\dfrac{I_1(\nu_{0,1})}{I_1(\nu_{0,1})+I_2(\nu_{0,2})}"
    else:
        eps_def = r"\varepsilon=\dfrac{I_1(\nu_0)}{I_1(\nu_0)+I_2(\nu_0)}"
    return r"w_1=\varepsilon\,S_1'(\nu),\ \ w_2=(1-\varepsilon)\,S_2'(\nu),\ \ " + eps_def


def full_equation(spec, shape):
    """The full (centered, possibly multi-line) equation-card LaTeX for
    `spec` at the given spectral `shape`: its own P(lambda) equation, plus
    the S'(nu) definition that shape implies.

    Single-component model: one line, S'(nu) to the left of P(lambda)
    (S'(nu) shapes only the Stokes I spectrum there, not p=P/I itself, but
    the choice is still shown).

    Two-component model: the w1/w2/epsilon definition on its own line
    first, then S'(nu) to the left of P(lambda) on the line below (matching
    the single-component layout, since P(lambda) itself is written in
    terms of w1/w2 -- see each two-component model's own `equation`)."""
    s_prime = s_prime_latex(spec.n_components, shape)
    body = spec.equation.strip()
    inner = body[1:-1]  # strip the surrounding '$...$'
    line2 = f'${s_prime}\\qquad\\quad {inner}$'
    if spec.n_components == 1:
        return line2
    line1 = f'${weights_latex(shape)}$'
    return f'{line1}\n{line2}'

# Model lookup by function name (e.g. 'burn', 'comp2intern'), the identifier
# saved/loaded in "Save Model"/"Load Model" JSON files and MultiNest's own
# polvista metadata sidecar (see app.py's save_model_action/load_model_action
# and sampling.py's load_samples_action) -- MODELS itself is keyed by the
# function object, which isn't serializable.
MODELS_BY_NAME = {func.__name__: func for func in MODELS}

