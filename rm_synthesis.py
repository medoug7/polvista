"""Pure-numpy RM synthesis / RM-CLEAN core, shared between RM_synth.py (the
standalone CLI) and the app's own "RM-synth" plot tab (see
app.MainWindow.rmsynth_from_model/_data/_measurements and
widgets.RMSynthPlot). Method reference: Brentjens & de Bruyn 2005, A&A
441, 1217; RM-CLEAN follows Heald 2009.

Deliberately has no dependency on the rest of polvista (matplotlib/PyQt5
included) -- just numpy -- so RM_synth.py can keep running as a bare
command-line tool with no GUI installed.
"""
import numpy as np

C = 299792458.0  # m/s

PHI_MAX_ALIAS_MULT = 6.0
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Absolute floor/ceiling [rad/m^2] on the alias-derived phi search/preview
# half-width (see phi_axis_half_width and default_phi_grid below) -- without
# these, a very wide wavelength band shrinks the displayed/searched phi range
# towards 0, and a very narrow one blows it up towards +inf, both with no
# limit.
PHI_HALF_WIDTH_MIN = 100.0
PHI_HALF_WIDTH_MAX = 5.0e6


def rm_transform(lambda2, values, weights, phi, lambda2_0):
    K = 1.0 / np.sum(weights)
    dl2 = (lambda2 - lambda2_0)[:, None]
    kernel = np.exp(-2j * dl2 * phi[None, :])
    return K * ((weights * values) @ kernel)


def faraday_dispersion_function(lambda2, Q, U, weights, phi, lambda2_0):
    P = Q + 1j * U
    return rm_transform(lambda2, P, weights, phi, lambda2_0)


def rmsf(lambda2, weights, phi, lambda2_0):
    ones = np.ones_like(lambda2, dtype=complex)
    return rm_transform(lambda2, ones, weights, phi, lambda2_0)


def briggs_weights(lambda2, noise_weights, robust, cell_width):
    bin_idx = np.floor((lambda2 - lambda2.min()) / cell_width).astype(int)
    cell_weight = np.zeros(bin_idx.max() + 1)
    np.add.at(cell_weight, bin_idx, noise_weights)
    f2 = (5.0 * 10.0 ** (-robust)) ** 2 / (np.sum(cell_weight ** 2) / np.sum(cell_weight))
    return noise_weights / (1.0 + f2 * cell_weight[bin_idx])


def phi_alias_scale(delta_l2):
    """180 deg EVPA-rotation alias scale pi/delta_l2 for a band spanning
    `delta_l2` [m^2] in lambda^2 -- the basis for both the default search
    half-width (default_phi_grid) and the empty-plot axis range
    (phi_axis_half_width) below."""
    return np.pi / delta_l2 if delta_l2 > 0 else np.inf


def phi_axis_half_width(lambda2_min, lambda2_max, phi_max_mult=PHI_MAX_ALIAS_MULT):
    """+/- half-width [rad/m^2] for an *empty* Faraday-spectrum plot's own
    x-axis, from just the two band-edge lambda^2 values -- no actual
    channel sampling needed yet (see default_phi_grid, which additionally
    needs the channel spacing once there's real data to search). Clipped to
    [PHI_HALF_WIDTH_MIN, PHI_HALF_WIDTH_MAX]."""
    half_width = phi_max_mult * phi_alias_scale(lambda2_max - lambda2_min)
    return float(np.clip(half_width, PHI_HALF_WIDTH_MIN, PHI_HALF_WIDTH_MAX))


def default_phi_grid(lambda2, phi_max_mult=PHI_MAX_ALIAS_MULT, phi_max_cap=None):
    l2_sorted = np.sort(lambda2)
    steps = np.diff(l2_sorted)
    steps = steps[steps > 0]
    delta_l2 = l2_sorted[-1] - l2_sorted[0]
    l2_min_step = steps.min()

    fwhm_rmsf = 2.0 * np.sqrt(3.0) / delta_l2
    phi_max_sampling = np.sqrt(3.0) / l2_min_step
    phi_alias = phi_alias_scale(delta_l2)
    if phi_max_cap is None:
        # Same [PHI_HALF_WIDTH_MIN, PHI_HALF_WIDTH_MAX] clip as
        # phi_axis_half_width, so the searched range never collapses
        # towards 0 (wide band) or blows up towards +inf (narrow band).
        phi_max_cap = np.clip(phi_max_mult * phi_alias, PHI_HALF_WIDTH_MIN, PHI_HALF_WIDTH_MAX)
    phi_max = min(phi_max_sampling, phi_max_cap)
    dphi = fwhm_rmsf / 15.0

    n = int(np.ceil(phi_max / dphi))
    phi = dphi * np.arange(-n, n + 1)
    return phi, dphi, fwhm_rmsf, phi_max_sampling, phi_alias


def rm_clean(dirty, phi, R, phi_r, gain=0.1, niter=1000, threshold=0.0):
    n_phi = len(phi)
    assert len(phi_r) == 2 * n_phi - 1
    center_r = n_phi - 1

    residual = dirty.copy()
    components = np.zeros(n_phi, dtype=complex)

    used = 0
    for used in range(1, niter + 1):
        i_peak = int(np.argmax(np.abs(residual)))
        peak_val = residual[i_peak]
        if np.abs(peak_val) <= threshold:
            used -= 1
            break
        comp = gain * peak_val
        components[i_peak] += comp
        shifted_R = R[center_r - i_peak: center_r - i_peak + n_phi]
        residual = residual - comp * shifted_R
    return components, residual, used


def measure_fwhm(x, amp, center_idx, fallback, max_offset=None):
    max_offset = len(amp) if max_offset is None else max_offset
    half = amp[center_idx]

    def half_max_crossing(step):
        i = center_idx
        for _ in range(max_offset):
            nxt = i + step
            if not (0 <= nxt < len(amp)):
                return None
            if amp[nxt] < 0.5 * half <= amp[i]:
                x0, x1 = x[i], x[nxt]
                y0, y1 = amp[i], amp[nxt]
                frac = (0.5 * half - y0) / (y1 - y0)
                return x0 + frac * (x1 - x0)
            i = nxt
        return None

    left = half_max_crossing(-1)
    right = half_max_crossing(+1)
    if left is None or right is None:
        return fallback
    return right - left


def restore(components, dphi, fwhm):
    sigma = fwhm / FWHM_TO_SIGMA
    max_half_width = max(1, (len(components) - 1) // 2)
    half_width = min(max_half_width, max(1, int(np.ceil(4 * sigma / dphi))))
    x = dphi * np.arange(-half_width, half_width + 1)
    beam = np.exp(-0.5 * (x / sigma) ** 2)
    real = np.convolve(components.real, beam, mode='same')
    imag = np.convolve(components.imag, beam, mode='same')
    return real + 1j * imag


def parabolic_peak(phi, amp, i):
    if i == 0 or i == len(amp) - 1:
        return phi[i], amp[i]
    y0, y1, y2 = amp[i - 1], amp[i], amp[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return phi[i], amp[i]
    dx = 0.5 * (y0 - y2) / denom
    dphi = phi[1] - phi[0]
    return phi[i] + dx * dphi, y1 - 0.25 * (y0 - y2) * dx


def sci_latex(value, err=None, unit=r'rad\,m^{-2}'):
    """Format `value` (optionally with `err`) as LaTeX scientific notation:
    ``(1.23 \\pm 0.04)\\times 10^{5}\\ \\mathrm{rad\\,m^{-2}}`` -- or just
    ``1.23\\times 10^{5}\\ \\mathrm{rad\\,m^{-2}}`` if no error is given.
    The common exponent is chosen from whichever of `value`/`err` is
    larger in magnitude (just `value` if no error is given), so a value
    consistent with 0 (e.g. no real Faraday rotation detected) still gets
    an exponent tied to its own error bar's actual size, rather than
    collapsing to whatever tiny floating-point residue `value` happens to
    be -- which would otherwise blow up the error's own mantissa (and,
    via decimals below, the number of digits shown) by that same
    many-orders-of-magnitude mismatch. The error's precision sets how many
    decimal places are shown (2 significant figures of the error).
    """
    ref = max(abs(value), abs(err)) if err is not None and np.isfinite(err) else abs(value)
    if ref == 0 or not np.isfinite(ref):
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(ref)))
    scale = 10.0 ** exponent
    mantissa = value / scale

    if err is not None and np.isfinite(err) and err > 0:
        err_mantissa = err / scale
        decimals = max(0, 1 - int(np.floor(np.log10(err_mantissa))))
        return (rf'$({mantissa:.{decimals}f} \pm {err_mantissa:.{decimals}f})'
                rf'\times 10^{{{exponent}}}\ \mathrm{{{unit}}}$')
    return rf'${mantissa:.2f}\times 10^{{{exponent}}}\ \mathrm{{{unit}}}$'


class FaradaySpectrumError(ValueError):
    """Raised by compute_faraday_spectrum when the input channels can't
    support RM synthesis (fewer than 2 usable points, or degenerate/zero
    lambda^2 coverage)."""


def compute_faraday_spectrum(wl, Q, U, Q_err, U_err, robust=1.0,
                              phi_max_mult=PHI_MAX_ALIAS_MULT, gain=0.1,
                              niter=1000, threshold_sigma=2.0):
    """Run RM synthesis + RM-CLEAN on one (wl [m], Q, U, Q_err, U_err) set
    of channels -- Q/U in any consistent unit (fractional polarization or
    absolute flux density; only the phase, not the amplitude scale,
    determines the recovered Faraday depth). Returns a dict of everything
    the plot needs; raises FaradaySpectrumError if there aren't enough
    usable (finite) channels to do this.
    """
    wl = np.asarray(wl, dtype=float)
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    Q_err = np.asarray(Q_err, dtype=float)
    U_err = np.asarray(U_err, dtype=float)

    finite = np.isfinite(wl) & (wl > 0) & np.isfinite(Q) & np.isfinite(U) & \
        np.isfinite(Q_err) & np.isfinite(U_err) & (Q_err > 0) & (U_err > 0)
    wl, Q, U, Q_err, U_err = wl[finite], Q[finite], U[finite], Q_err[finite], U_err[finite]
    if len(wl) < 2:
        raise FaradaySpectrumError('Need at least 2 usable spectral points to run RM synthesis.')

    lambda2 = wl ** 2
    if np.ptp(lambda2) <= 0:
        raise FaradaySpectrumError('All channels share the same wavelength -- no lambda^2 coverage to synthesize.')

    sigma = np.sqrt(0.5 * (Q_err ** 2 + U_err ** 2))
    noise_weights = 1.0 / sigma ** 2

    phi, dphi, fwhm_auto, phi_max_auto, phi_alias = default_phi_grid(lambda2, phi_max_mult=phi_max_mult)
    cell_width = np.pi / phi[-1]
    weights = briggs_weights(lambda2, noise_weights, robust, cell_width)
    lambda2_0 = np.sum(weights * lambda2) / np.sum(weights)

    n_phi = len(phi)
    phi_r = dphi * np.arange(-(n_phi - 1), n_phi)

    F_dirty = faraday_dispersion_function(lambda2, Q, U, weights, phi, lambda2_0)
    R = rmsf(lambda2, weights, phi_r, lambda2_0)

    sigma_fdf = np.sqrt(np.sum((weights * sigma) ** 2)) / np.sum(weights)
    threshold = threshold_sigma * sigma_fdf

    components, residual, n_used = rm_clean(
        F_dirty, phi, R, phi_r, gain=gain, niter=niter, threshold=threshold)

    rmsf_search = int(np.ceil(5 * fwhm_auto / dphi))
    fwhm = measure_fwhm(phi_r, np.abs(R), len(phi_r) // 2, fallback=fwhm_auto, max_offset=rmsf_search)
    F_clean = restore(components, dphi, fwhm) + residual

    i_clean_peak = int(np.argmax(np.abs(F_clean)))
    phi_peak, amp_peak = parabolic_peak(phi, np.abs(F_clean), i_clean_peak)
    snr = amp_peak / sigma_fdf if sigma_fdf > 0 else np.nan
    phi_err = fwhm / (2.0 * snr) if snr > 0 else np.nan

    return dict(phi=phi, F_dirty=F_dirty, F_clean=F_clean, components=components,
                R=R, phi_r=phi_r, fwhm=fwhm, sigma_fdf=sigma_fdf,
                phi_peak=phi_peak, amp_peak=amp_peak, phi_err=phi_err, snr=snr,
                lambda2_0=lambda2_0, n_channels=len(wl))
