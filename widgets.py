"""Qt/matplotlib widgets shared across polvista's tabs -- ParamSlider's own
small building blocks (ValueLineEdit and friends), plus the two plot
canvases (ModelPlot, StokesPlot) app.py's Visualization tab embeds. Split
out on its own so no other module has to import app.py (a large,
MainWindow-centric module) just to get these."""
import re
import colorsys

import numpy as np
import matplotlib.colors as mcolors
from PyQt5.QtWidgets import QLineEdit

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.transforms as mtransforms

from polvista.models import C, stokes_I, stokes_QU, stokes_components, evpa, pol
from polvista.measurements import RAINBOW_HUE_MAX
from polvista.rm_synthesis import sci_latex

SLIDER_STEPS = 10000  # integer resolution backing each QSlider

# Max number of segments used for the Polar Q,U view's rainbow-colored
# trajectory line (see StokesPlot.update_plot). matplotlib's LineCollection
# allocates one fresh Path object per segment on every set_segments() call
# (profiling: dominates redraw time, ~1000 Path() allocations/frame at the
# default n_points=1000, ~70% of a slider-drag frame in Polar mode) -- capped
# independently of n_points (which can run up to 3000) since a color gradient
# doesn't need anywhere near that many steps to look smooth, unlike the
# underlying curve's own point density.
QU_POLAR_MAX_SEGMENTS = 500

# Physical unit shown next to each slider's readout, keyed by Param.kind.
UNITS = {'p': '%', 'X': '°', 'phi': 'rad/m²', 'Dphi':'rad/m²', 'dphi': 'rad/m²', 'scale': '', 'alpha': '', 'eps': '',
         'nu0': 'GHz', 'temp': 'K', 'freq': 'MHz', 'wave': 'm'}
# Widest unit string, used to fix every slider's unit label to the same
# width so the value boxes above/below each other line up regardless of
# which unit (or none) a given row happens to show.
WIDEST_UNIT = max(UNITS.values(), key=len)

NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

# Sentinel for ModelPlot/StokesPlot's own posterior-sample redraw cache (see
# their update_plot methods) -- distinct from any real cache key (a tuple, or
# None when there are no samples to show) so the very first update_plot call
# always (re)populates the sample lines rather than matching by accident.
_SAMPLES_CACHE_UNSET = object()


class ValueLineEdit(QLineEdit):
    """A QLineEdit that selects all its text on focus, so clicking a
    parameter's value and typing immediately replaces it."""
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.selectAll()


# Fixed PIXEL margins for the three plot canvases below (not fractions --
# see apply_fixed_margins). Sized generously from measured worst cases (long
# negative numbers, scientific notation, wide mathtext labels) so the
# right-side rotated ylabel + tick labels are always fully visible.
# ModelPlot (p, chi vs lambda^2) and RMSynthPlot (Faraday spectrum) use this
# one directly; StokesPlot needs more left room (see STOKES_MARGINS_PX).
PLOT_MARGINS_PX = dict(left=80, right=125, bottom=95, top=40)

# StokesPlot's own margins -- wider left than PLOT_MARGINS_PX: ax_I is
# log-scaled, and a sub-decade y-range (e.g. Stokes I over a narrow band
# like Standard ALMA, where the spectral index alone doesn't swing I across
# a full decade) makes matplotlib label *minor* ticks too (e.g. '4x10^-1'),
# which are wider than the plain 'x10^n' decade labels
# _bound_yaxis_ticklabels's scilimits assumes -- and that formatter doesn't
# apply to a log axis at all (ax.ticklabel_format raises on one), so ax_I's
# ylabel needs the extra room unconditionally.
STOKES_MARGINS_PX = dict(PLOT_MARGINS_PX, left=120)

# Continuous version of measurements.py's band_colors() red->violet HSV
# sweep (same RAINBOW_HUE_MAX endpoint), used to color StokesPlot's Polar
# view (Q, U) curve by frequency so it reads on the same red=low-nu,
# violet=high-nu convention as the Measurements tab's simulated points.
QU_RAINBOW_CMAP = mcolors.ListedColormap(
    [colorsys.hsv_to_rgb(RAINBOW_HUE_MAX * t, 1.0, 1.0) for t in np.linspace(0, 1, 256)])


def apply_fixed_margins(fig, canvas, extra_adjust=None, margins=PLOT_MARGINS_PX):
    """Set fig.subplots_adjust margins from `margins` (PLOT_MARGINS_PX by
    default, STOKES_MARGINS_PX for StokesPlot), converted to the fractions
    matplotlib wants using the canvas's *current* pixel size."""

    w = max(canvas.width(), 1)
    h = max(canvas.height(), 1)
    m = margins
    left_px, right_px = m['left'], m['right']
    bottom_px, top_px = m['bottom'], m['top']
    # Guard against pathologically narrow/short windows where the fixed
    # margins alone would exceed the canvas (subplots_adjust raises if
    # left>=right) -- scale them down together so there's always at least
    # ~20% of the canvas left for the actual plotted data, rather than crash.
    if left_px + right_px > 0.8 * w:
        scale = 0.8 * w / (left_px + right_px)
        left_px, right_px = left_px * scale, right_px * scale
    if bottom_px + top_px > 0.8 * h:
        scale = 0.8 * h / (bottom_px + top_px)
        bottom_px, top_px = bottom_px * scale, top_px * scale

    fig.subplots_adjust(left=left_px / w, right=1 - right_px / w,
                        bottom=bottom_px / h, top=1 - top_px / h,
                        **(extra_adjust or {}),)


def _bound_yaxis_ticklabels(ax):
    """Force compact '10^n'-multiplier y-tick formatting once values fall
    outside [0.01, 1000) -- keeps tick-label width (and the auto-computed
    ylabel position) within PLOT_MARGINS_PX's fixed budget regardless of
    how small/large the data gets."""
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3), useMathText=True)


def _finite_bounds(arr, fallback):
    """(min, max) of the finite entries of `arr`, or `fallback` (lo, hi)
    if none are finite -- guards the axis-limit seed taken straight off a
    model curve (a custom model can divide by zero and go inf/nan) before
    set_xlim/set_ylim, which raise on a non-finite value."""
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback
    return float(finite.min()), float(finite.max())


def _clear_artists(artists):
    """Remove every artist in `artists` and return [] for reassignment --
    the tear-down-and-redraw-from-scratch pattern each draw_reference()
    below uses for its errorbar overlays."""
    for artist in artists:
        artist.remove()
    return []


# Errorbar styles shared by ModelPlot/StokesPlot's own draw_reference()
# methods: a filled black circle for a real Load Data... point, and a
# colored marker (fmt overridable to match the axis's own per-series
# marker) for a Measurements-tab simulated point.
REF_POINT_STYLE = dict(fmt='o', ms=4, color='black', ecolor='0.5', capsize=2, zorder=5)


def _meas_point_style(color, fmt='o'):
    return dict(fmt=fmt, ms=4, mew=0.5, mec='k', color=color, ecolor=color, alpha=0.85, capsize=2, zorder=6)


def _draw_errorbar_overlay(ax, x, y, style, xerr=None, yerr=None):
    return ax.errorbar(x, y, xerr=xerr, yerr=yerr, **style)


def _widen_range(lo, hi, values, err=None):
    """Widen (lo, hi) to also cover `values` (optionally +/- `err`)."""
    values = np.asarray(values)
    if err is not None:
        err = np.asarray(err)
        lo = min(lo, float(np.min(values - err)))
        hi = max(hi, float(np.max(values + err)))
    else:
        lo = min(lo, float(np.min(values)))
        hi = max(hi, float(np.max(values)))
    return lo, hi


class ModelPlot(FigureCanvas):
    #Two side-by-side axes: polarization fraction and EVPA vs lambda^2

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax_p, self.ax_x = self.fig.subplots(1, 2, gridspec_kw={'wspace': 0})
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0})
        self.ax_x.yaxis.tick_right()
        self.ax_x.yaxis.set_label_position('right')
        self.fig.text(0.5, 0.02, r'$\lambda^2$ [ mm$^2$ ]', ha='center', fontsize=16)

        # Labels/grid set once here (not in update_plot()): Axes.clear() is
        # expensive, so slider drags reuse these Line2D objects via
        # set_data() instead of clear()+replot().
        self.ax_p.set_ylabel(r'$p$ [ % ]')
        self.ax_p.grid(True)
        _bound_yaxis_ticklabels(self.ax_p)
        # labelpad=25: rotation=270 + tick_right() above put this ylabel
        # right against its own tick labels at the default pad.
        self.ax_x.set_ylabel(r'$\chi$ [ deg ]', rotation=270, labelpad=25)
        self.ax_x.grid(True)
        _bound_yaxis_ticklabels(self.ax_x)
        self.line_p, = self.ax_p.plot([], [], color='tab:blue')
        self.line_x, = self.ax_x.plot([], [], color='tab:red')
        self.xscale = None  # forces the first set_xscale('log') call to actually apply
        self.ref_data = None  # (w2, p, p_err, evpa, evpa_err) from the Load Data... button, or None
        self.ref_artists = []
        # Simulated points from the Measurements tab's Generate button (list
        # of per-band dicts) or None; kept separate from ref_data so the two
        # overlays never clobber each other.
        self.meas_bands = None
        self.meas_artists = []
        # draw_reference() rebuilds every errorbar artist from scratch (no
        # cheaper set_data() update exists for errorbar) -- too expensive to
        # run on every slider-driven update_plot() unconditionally, so this
        # flag (set by set/clear_reference_data and set/clear_measurement_
        # data) gates when update_plot() actually calls it.
        self._ref_dirty = True

        # Posterior-sample "spaghetti" overlay (see set_posterior_samples):
        # a fixed pool of faint Line2D artists, (re)created only when the
        # sample set itself changes, then refreshed via set_data() alongside
        # the main curve on every update_plot() call.
        self.posterior_samples = None  # (n, ndim_full) array, model's full param order, or None
        self.posterior_model = None    # the model func these samples belong to
        self.sample_lines_p = []
        self.sample_lines_x = []
        # A parameter-slider drag can't change the sample lines (they only
        # depend on wl_ext and the fixed posterior_samples/posterior_model),
        # so _samples_cache_key skips recomputing them when neither has
        # changed since the last update_plot() call; _samples_version
        # (bumped by set_posterior_samples) invalidates it rather than
        # comparing posterior_samples by identity.
        self._samples_cache_key = _SAMPLES_CACHE_UNSET
        self._samples_version = 0

    # how to load data
    def set_reference_data(self, w2, p, p_err, evpa, evpa_err):
        self.ref_data = (np.asarray(w2), np.asarray(p), p_err, np.asarray(evpa), evpa_err)
        self._ref_dirty = True

    def clear_reference_data(self):
        self.ref_data = None
        self._ref_dirty = True

    def set_measurement_data(self, bands):
        """`bands` is a list of per-band dicts (see measurements.py's
        generate_measurements), or None to clear."""
        self.meas_bands = bands
        self._ref_dirty = True

    def clear_measurement_data(self):
        self.meas_bands = None
        self._ref_dirty = True

    # how data is plotted
    def draw_reference(self):
        self.ref_artists = _clear_artists(self.ref_artists)
        if self.ref_data is not None:
            w2, p, p_err, evpa, evpa_err = self.ref_data
            self.ref_artists.append(_draw_errorbar_overlay(self.ax_p, w2, p, REF_POINT_STYLE, yerr=p_err))
            self.ref_artists.append(_draw_errorbar_overlay(self.ax_x, w2, evpa, REF_POINT_STYLE, yerr=evpa_err))

        self.meas_artists = _clear_artists(self.meas_artists)
        if self.meas_bands:
            for band in self.meas_bands:
                style = _meas_point_style(band['color'], fmt='D')
                self.meas_artists.append(_draw_errorbar_overlay(
                    self.ax_p, band['w2'], band['p'], style, yerr=band['p_err']))
                self.meas_artists.append(_draw_errorbar_overlay(
                    self.ax_x, band['w2'], band['evpa'], style, yerr=band['evpa_err']))

    def set_posterior_samples(self, samples, model_func):
        """(Re)build the posterior-sample line pool for `samples` (an (n,
        ndim_full) array in `model_func`'s full param order, or None to
        clear) -- creates/destroys artists (the expensive part) only here,
        not on every update_plot() call. Called after a MultiNest fit/Load
        samples, and on Reset model (with samples=None)."""
        for ln in self.sample_lines_p + self.sample_lines_x:
            ln.remove()
        self.sample_lines_p, self.sample_lines_x = [], []
        self.posterior_samples = samples
        self.posterior_model = model_func
        self._samples_version += 1
        if samples is not None:
            for _ in range(len(samples)):
                ln_p, = self.ax_p.plot([], [], color='tab:blue', alpha=0.12, lw=0.7, zorder=1)
                ln_x, = self.ax_x.plot([], [], color='tab:red', alpha=0.12, lw=0.7, zorder=1)
                self.sample_lines_p.append(ln_p)
                self.sample_lines_x.append(ln_x)
        self.draw_idle()

    def update_plot(self, wl_ext, model_func, pars, log_xscale=False):
        """Recompute p/EVPA vs lambda^2 from the model and redraw: main
        curve, posterior-sample overlay (if any), and axis limits widened
        to cover any loaded reference/measurement data."""
        fit = model_func(wl_ext, pars)
        p = pol(fit)
        evpa_vals = evpa(fit)
        w2 = wl_ext ** 2 * 1e6  # lambda^2 [mm^2]

        # Posterior-sample overlay: only actually drawn while the
        # currently-selected model matches the one these samples were fit
        # for -- switching the model dropdown blanks them (set_data([],[]))
        # without discarding them, so they silently reappear if the user
        # switches back, rather than needing an explicit clear.
        show_samples = self.posterior_samples is not None and self.posterior_model is model_func
        samples_key = (wl_ext.tobytes(), self._samples_version, model_func) if show_samples else None
        if samples_key != self._samples_cache_key:
            self._samples_cache_key = samples_key
            for i, (ln_p, ln_x) in enumerate(zip(self.sample_lines_p, self.sample_lines_x)):
                if show_samples:
                    s_fit = model_func(wl_ext, self.posterior_samples[i])
                    ln_p.set_data(w2, pol(s_fit))
                    ln_x.set_data(w2, evpa(s_fit))
                else:
                    ln_p.set_data([], [])
                    ln_x.set_data([], [])

        xscale = 'log' if log_xscale else 'linear'
        if xscale != self.xscale:
            self.ax_p.set_xscale(xscale)
            self.ax_x.set_xscale(xscale)
            self.xscale = xscale
        xlo, xhi = np.min(w2) * 0.9, np.max(w2) * 1.05
        _, p_seed_max = _finite_bounds(p, fallback=(0.0, 1e-6))
        p_max = max(p_seed_max, 1e-6)
        x_lo, x_hi = _finite_bounds(evpa_vals, fallback=(0.0, 0.0))
        # Widen the autoscale to also cover any reference data loaded via
        # the Load Data... button, so it isn't silently clipped out of view.
        if self.ref_data is not None:
            ref_w2, ref_p, _, ref_evpa, _ = self.ref_data
            xlo, xhi = min(xlo, np.min(ref_w2) * 0.9), max(xhi, np.max(ref_w2) * 1.05)
            p_max = max(p_max, np.max(ref_p))
            x_lo, x_hi = _widen_range(x_lo, x_hi, ref_evpa)
        if self.meas_bands:
            for band in self.meas_bands:
                xlo = min(xlo, np.min(band['w2']) * 0.9)
                xhi = max(xhi, np.max(band['w2']) * 1.05)
                p_max = max(p_max, np.max(band['p'] + band['p_err']))
                x_lo, x_hi = _widen_range(x_lo, x_hi, band['evpa'], band['evpa_err'])

        self.line_p.set_data(w2, p)
        self.ax_p.set_xlim(xlo, xhi)
        self.ax_p.set_ylim(0, 1.3 * p_max)

        self.line_x.set_data(w2, evpa_vals)
        self.ax_x.set_xlim(xlo, xhi)

        # Enforce a minimum 2 deg y-span so a near-flat EVPA curve (small
        # |phi|) doesn't get visually exaggerated by an overly tight autoscale.
        margin = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 0.05
        x_lo, x_hi = x_lo - margin, x_hi + margin
        if x_hi - x_lo < 2.0:
            mid = 0.5 * (x_lo + x_hi)
            x_lo, x_hi = mid - 1.0, mid + 1.0
        self.ax_x.set_ylim(x_lo, x_hi)
        if self._ref_dirty:
            self.draw_reference()
            self._ref_dirty = False
        self.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0})


def stokes_to_frac_qu(I, Q, U, I_err=None, Q_err=None, U_err=None):
    """Fractional q=Q/I, u=U/I (dimensionless, same p*cos(2EVPA)/
    p*sin(2EVPA) quantity the models themselves compute) and their
    propagated errors -- errors are None if none of I_err/Q_err/U_err
    were given, otherwise missing ones are treated as 0. Used by
    StokesPlot's Polar view so its trajectory reflects pure polarization
    state, undistorted by I's own separately-varying spectral shape --
    same error-propagation formula as app.py's load_data_action."""
    q, u = Q / I, U / I
    if I_err is None and Q_err is None and U_err is None:
        return q, u, None, None

    def frac_err(val, val_err, denom, denom_err):
        val_err = val_err if val_err is not None else 0.0
        denom_err = denom_err if denom_err is not None else 0.0
        return np.sqrt((val_err / denom) ** 2 + (val * denom_err / denom ** 2) ** 2)

    return q, u, frac_err(Q, Q_err, I, I_err), frac_err(U, U_err, I, I_err)


class StokesPlot(FigureCanvas):
    """Stokes I (left subplot) and Q, U (right subplot) vs frequency nu
    [GHz], flush against each other like ModelPlot's p/EVPA pair. The
    right subplot has two interchangeable views (`stokes_mode_combo`):
    'Spectra' plots Q(nu)/U(nu) like the left subplot; 'Polar' instead
    plots the fractional (q=Q/I, u=U/I) trajectory traced out as nu
    varies, colored red(low nu)->violet(high nu) (see stokes_to_frac_qu,
    QU_RAINBOW_CMAP) to match the Measurements tab's own band coloring.
    Switching modes rebuilds the right axis from scratch (build_qu_mode);
    everything else uses set_data()-only updates to keep slider drags
    cheap."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax_I, self.ax_QU = self.fig.subplots(1, 2, gridspec_kw={'wspace': 0})
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0}, margins=STOKES_MARGINS_PX)

        self.ax_I.set_xlabel(r'$\nu$ [ GHz ]')
        self.ax_I.set_ylabel(r'$I$ [ normalized ]')
        self.ax_I.set_yscale('log')
        self.ax_I.grid(True)
        self.line_I, = self.ax_I.plot([], [], color='black', label=r'$I$')
        # Per-component curves for two-component models (see
        # models.stokes_components) -- component 1 dashed, component 2
        # dotted, same color as the total's own solid line; left empty
        # (and so invisible) for single-component models. Spectra mode's
        # own Q1/U1/Q2/U2 counterparts are built in build_qu_mode, since
        # that axis is torn down/rebuilt on every mode switch.
        self.line_I1, = self.ax_I.plot([], [], color='black', linestyle='dashed', label=r'$I_1$')
        self.line_I2, = self.ax_I.plot([], [], color='black', linestyle='dotted', label=r'$I_2$')
        # Tracks which legend (just 'I', or 'I'/'I_1'/'I_2') ax_I currently
        # shows, so it's only rebuilt on an actual single/two-component
        # switch (see update_plot) rather than on every redraw.
        self.ax_I_two_comp = None

        self.xscale = None
        self.mode = 'Spectra'
        self.ref_data = None  # (nu_ghz, I, I_err, Q, Q_err, U, U_err) or None
        self.ref_artists_I = []
        self.ref_artists_QU = []
        # See ModelPlot's own meas_bands/meas_artists -- same pattern,
        # keyed by nu/I/Q/U instead of w2/p/evpa.
        self.meas_bands = None
        self.meas_artists_I = []
        self.meas_artists_QU = []
        # See ModelPlot's own _ref_dirty -- same "only rebuild the errorbar
        # artists when ref_data/meas_bands actually changed" gating for
        # update_plot()'s own trailing draw_reference() call. build_qu_mode
        # (mode switches) still always redraws unconditionally -- ax_QU.clear()
        # there already discards ref_artists_QU, so it must repopulate them.
        self._ref_dirty = True

        # See ModelPlot.set_posterior_samples -- same pattern here. The
        # right-hand pool (sample_lines_Q/U for Spectra, sample_lines_QU
        # for Polar) is (re)built by build_qu_mode below, sized off
        # self.posterior_samples, since it's mode-dependent.
        self.posterior_samples = None
        self.posterior_model = None
        self.sample_lines_I = []
        self.sample_lines_Q = []
        self.sample_lines_U = []
        self.sample_lines_QU = []
        self.line_Q = self.line_U = self.line_QU = None
        self.line_Q1 = self.line_U1 = self.line_Q2 = self.line_U2 = None
        # See ModelPlot's own _samples_cache_key/_samples_version -- same
        # "skip recomputing sample lines a parameter-slider drag can't
        # possibly have changed" gating, split into the wl_ext-gridded
        # sample_lines_I/Q/U (used in both modes/Spectra mode respectively)
        # and the wl_qu-resampled sample_lines_QU (Polar mode only, see
        # update_plot), since those two groups are evaluated on different
        # wavelength grids. build_qu_mode always resets both -- a mode
        # switch or a changed sample count tears down and recreates
        # whichever pool(s) it touches, which need repopulating regardless
        # of whether wl_ext/samples themselves changed.
        self._samples_cache_key = _SAMPLES_CACHE_UNSET
        self._samples_polar_cache_key = _SAMPLES_CACHE_UNSET
        self._samples_version = 0
        self.build_qu_mode()

    def build_qu_mode(self):
        """(Re)configure the right-hand axis for self.mode: static
        labels/grid/aspect, the main curve artist(s), and a posterior-
        sample line pool sized to len(self.posterior_samples). Called
        once at init, on every mode-dropdown switch, and whenever the
        posterior-sample count changes (set_posterior_samples) -- all rare
        enough that a full clear()+rebuild is fine."""
        # Freshly (re)created sample line pools always need repopulating on
        # the next update_plot(), regardless of whether wl_ext/samples
        # themselves changed since the last one.
        self._samples_cache_key = _SAMPLES_CACHE_UNSET
        self._samples_polar_cache_key = _SAMPLES_CACHE_UNSET
        self.ax_QU.clear()
        self.ax_QU.grid(True)
        # Mirror the y-axis onto the right spine (like ModelPlot.ax_x) so
        # it reads correctly against ax_I's own left-side axis across
        # their shared, gapless seam.
        self.ax_QU.yaxis.tick_right()
        self.ax_QU.yaxis.set_label_position('right')
        self.ref_artists_QU = []
        self.meas_artists_QU = []
        n_samples = len(self.posterior_samples) if self.posterior_samples is not None else 0
        if self.mode == 'Polar':
            self.ax_QU.set_xlabel(r'$q$')
            # labelpad=25, not the default ~4 -- see ModelPlot.ax_x's own
            # comment (same rotation=270 + tick_right() crowding).
            self.ax_QU.set_ylabel(r'$u$', rotation=270, labelpad=25)
            _bound_yaxis_ticklabels(self.ax_QU)
            self.line_Q = self.line_U = None
            self.line_Q1 = self.line_U1 = self.line_Q2 = self.line_U2 = None
            self.line_QU = LineCollection([], cmap=QU_RAINBOW_CMAP, norm=mcolors.Normalize(0, 1), zorder=3)
            self.ax_QU.add_collection(self.line_QU)
            self.sample_lines_Q, self.sample_lines_U = [], []
            self.sample_lines_QU = [self.ax_QU.plot([], [], color='tab:purple', alpha=0.1, lw=0.6, zorder=1)[0]
                                     for _ in range(n_samples)]
            ylim = self.ax_QU.get_ylim()
            xlim = self.ax_QU.get_xlim()
            self.ax_QU.vlines(x=0.0, ymin=-ylim[1], ymax=ylim[1], linestyle='dashed', color='k')
            self.ax_QU.hlines(y=0.0, xmin=-xlim[1], xmax=xlim[1], linestyle='dashed', color='k')
        else:
            self.ax_QU.set_xlabel(r'$\nu$ [ GHz ]')
            self.ax_QU.set_ylabel(r'$Q$, $U$ [ normalized ]', rotation=270, labelpad=25)
            _bound_yaxis_ticklabels(self.ax_QU)
            self.ax_QU.set_xscale(self.xscale or 'linear')
            self.line_QU = None
            self.line_Q, = self.ax_QU.plot([], [], color='green', label=r'$Q$')
            self.line_U, = self.ax_QU.plot([], [], color='orange', label=r'$U$')
            # Per-component Q/U (see line_I1/line_I2) -- no label, so they
            # stay out of the Q/U legend above.
            self.line_Q1, = self.ax_QU.plot([], [], color='green', linestyle='dashed')
            self.line_U1, = self.ax_QU.plot([], [], color='orange', linestyle='dashed')
            self.line_Q2, = self.ax_QU.plot([], [], color='green', linestyle='dotted')
            self.line_U2, = self.ax_QU.plot([], [], color='orange', linestyle='dotted')
            self.ax_QU.legend(loc='upper right')
            self.sample_lines_QU = []
            self.sample_lines_Q = [self.ax_QU.plot([], [], color='green', alpha=0.08, lw=0.6, zorder=1)[0]
                                    for _ in range(n_samples)]
            self.sample_lines_U = [self.ax_QU.plot([], [], color='orange', alpha=0.08, lw=0.6, zorder=1)[0]
                                    for _ in range(n_samples)]
        self.draw_reference()
        self._ref_dirty = False

    def set_posterior_samples(self, samples, model_func):
        for ln in self.sample_lines_I:
            ln.remove()
        self.sample_lines_I = []
        self.posterior_samples = samples
        self.posterior_model = model_func
        self._samples_version += 1
        if samples is not None:
            for _ in range(len(samples)):
                ln_I, = self.ax_I.plot([], [], color='black', alpha=0.08, lw=0.6, zorder=1)
                self.sample_lines_I.append(ln_I)
        self.build_qu_mode()
        self.draw_idle()

    def set_reference_data(self, nu_ghz, I, I_err, Q, Q_err, U, U_err):
        self.ref_data = (np.asarray(nu_ghz), np.asarray(I), I_err, np.asarray(Q), Q_err, np.asarray(U), U_err)
        self._ref_dirty = True

    def clear_reference_data(self):
        self.ref_data = None
        self._ref_dirty = True

    def set_measurement_data(self, bands):
        """`bands` is a list of per-band dicts (see measurements.py's
        generate_measurements), or None to clear."""
        self.meas_bands = bands
        self._ref_dirty = True

    def clear_measurement_data(self):
        self.meas_bands = None
        self._ref_dirty = True

    # draw data
    def draw_reference(self):
        self.ref_artists_I = _clear_artists(self.ref_artists_I)
        self.ref_artists_QU = _clear_artists(self.ref_artists_QU)
        if self.ref_data is not None:
            nu, I, I_err, Q, Q_err, U, U_err = self.ref_data
            self.ref_artists_I.append(_draw_errorbar_overlay(self.ax_I, nu, I, REF_POINT_STYLE, yerr=I_err))
            if self.mode == 'Polar':
                q, u, q_err, u_err = stokes_to_frac_qu(I, Q, U, I_err, Q_err, U_err)
                self.ref_artists_QU.append(_draw_errorbar_overlay(
                    self.ax_QU, q, u, REF_POINT_STYLE, xerr=q_err, yerr=u_err))
            else:
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    nu, Q, yerr=Q_err, fmt='s', ms=4, color='darkgreen', ecolor='0.5', capsize=2, zorder=5))
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    nu, U, yerr=U_err, fmt='^', ms=4, color='darkorange', ecolor='0.5', capsize=2, zorder=5))

        self.meas_artists_I = _clear_artists(self.meas_artists_I)
        self.meas_artists_QU = _clear_artists(self.meas_artists_QU)
        if self.meas_bands:
            for band in self.meas_bands:
                c = band['color']
                self.meas_artists_I.append(_draw_errorbar_overlay(
                    self.ax_I, band['nu'], band['I'], _meas_point_style(c), yerr=band['I_err']))
                if self.mode == 'Polar':
                    q, u, q_err, u_err = stokes_to_frac_qu(
                        band['I'], band['Q'], band['U'], band['I_err'], band['Q_err'], band['U_err'])
                    self.meas_artists_QU.append(_draw_errorbar_overlay(
                        self.ax_QU, q, u, _meas_point_style(c), xerr=q_err, yerr=u_err))
                else:
                    self.meas_artists_QU.append(_draw_errorbar_overlay(
                        self.ax_QU, band['nu'], band['Q'], _meas_point_style(c, fmt='s'), yerr=band['Q_err']))
                    self.meas_artists_QU.append(_draw_errorbar_overlay(
                        self.ax_QU, band['nu'], band['U'], _meas_point_style(c, fmt='^'), yerr=band['U_err']))

    def update_plot(self, wl_ext, model_func, n_components, pars, log_xscale=False, nu_min=None, mode='Spectra'):
        """Recompute and redraw Stokes I (left) and Q/U-or-polar (right)
        from the model for the current mode/log_xscale/wavelength grid,
        refresh the posterior-sample overlay, and widen/apply axis limits
        to cover any loaded reference/measurement data. Called on every
        parameter-slider tick, so hot-path cost matters (see class
        docstring)."""
        if mode != self.mode:
            self.mode = mode
            self.build_qu_mode()

        arrays = self._update_stokes_arrays(wl_ext, model_func, n_components, pars, nu_min)
        order, nu_s = arrays['order'], arrays['nu_s']

        xscale = 'log' if log_xscale else 'linear'
        if xscale != self.xscale:
            self.ax_I.set_xscale(xscale)
            if self.mode == 'Spectra':
                self.ax_QU.set_xscale(xscale)
            self.xscale = xscale

        self.line_I.set_data(nu_s, arrays['I_s'])
        if arrays['two_comp']:
            self.line_I1.set_data(nu_s, arrays['I1_s'])
            self.line_I2.set_data(nu_s, arrays['I2_s'])
        else:
            self.line_I1.set_data([], [])
            self.line_I2.set_data([], [])

        show_samples = self.posterior_samples is not None and self.posterior_model is model_func
        samples_changed = self._refresh_sample_lines_I(
            wl_ext, model_func, n_components, nu_min, order, nu_s, show_samples)

        if self.mode == 'Polar':
            q_s, u_s = self._draw_polar_mode(wl_ext, model_func, n_components, pars, nu_min, show_samples)
        else:
            q_s = u_s = None
            self._draw_spectra_mode(arrays, samples_changed, show_samples,
                                     wl_ext, model_func, n_components, nu_min, order)

        self._apply_axis_limits(arrays, q_s, u_s)

        if self._ref_dirty:
            self.draw_reference()
            self._ref_dirty = False
        self.draw_idle()

    def _update_stokes_arrays(self, wl_ext, model_func, n_components, pars, nu_min):
        """Evaluate Stokes I/Q/U (total, plus per-component for a two-
        component model) on wl_ext and sort by ascending frequency;
        returns them in a dict keyed by 'order'/'nu_s'/'I_s'/'Q_s'/'U_s'/
        'two_comp'/'I1_s'/'Q1_s'/'U1_s'/'I2_s'/'Q2_s'/'U2_s' (the *_s
        component keys are None unless two_comp). Also keeps ax_I's
        legend (I vs I/I1/I2) in sync with whether this is two-component."""
        I = stokes_I(wl_ext, n_components, pars, nu_min=nu_min)
        Q, U = stokes_QU(wl_ext, model_func, n_components, pars, nu_min=nu_min, I=I)
        nu_ghz = C / wl_ext / 1e9

        order = np.argsort(nu_ghz)
        nu_s, I_s, Q_s, U_s = nu_ghz[order], I[order], Q[order], U[order]

        two_comp = n_components == 2
        arrays = dict(order=order, nu_s=nu_s, I_s=I_s, Q_s=Q_s, U_s=U_s, two_comp=two_comp,
                      I1_s=None, Q1_s=None, U1_s=None, I2_s=None, Q2_s=None, U2_s=None)
        if two_comp:
            (I1, Q1, U1), (I2, Q2, U2) = stokes_components(wl_ext, model_func, pars, nu_min=nu_min)
            arrays.update(I1_s=I1[order], Q1_s=Q1[order], U1_s=U1[order],
                          I2_s=I2[order], Q2_s=Q2[order], U2_s=U2[order])

        if two_comp != self.ax_I_two_comp:
            self.ax_I_two_comp = two_comp
            handles = [self.line_I, self.line_I1, self.line_I2] if two_comp else [self.line_I]
            self.ax_I.legend(handles, [h.get_label() for h in handles], loc='lower right')
        return arrays

    def _refresh_sample_lines_I(self, wl_ext, model_func, n_components, nu_min, order, nu_s, show_samples):
        """Refresh the sample_lines_I posterior overlay if wl_ext/
        n_components/nu_min/model_func or the sample set itself changed
        since the last call (see ModelPlot's own _samples_cache_key for
        why this is worth skipping on a parameter-slider drag). Returns
        whether they changed -- sample_lines_Q/U share these same inputs
        (see _draw_spectra_mode), so callers reuse this to gate them too."""
        samples_key = (wl_ext.tobytes(), n_components, nu_min, self._samples_version, model_func) \
            if show_samples else None
        samples_changed = samples_key != self._samples_cache_key
        self._samples_cache_key = samples_key
        if samples_changed:
            for i, ln_I in enumerate(self.sample_lines_I):
                if show_samples:
                    s_I = stokes_I(wl_ext, n_components, self.posterior_samples[i], nu_min=nu_min)[order]
                    ln_I.set_data(nu_s, s_I)
                else:
                    ln_I.set_data([], [])
        return samples_changed

    def _draw_polar_mode(self, wl_ext, model_func, n_components, pars, nu_min, show_samples):
        """Resample the model evenly in lambda^2, draw the fractional
        (q, u) trajectory as a rainbow-colored LineCollection, and refresh
        its own posterior-sample overlay. Returns (q_s, u_s), the total
        curve's own fractional q/u (needed by _apply_axis_limits's Polar
        branch)."""
        # Resample evenly in lambda^2 -- not in lambda/log-lambda like
        # wl_ext itself (that spacing stays as-is for the I panel and
        # Spectra mode, driven by the log_xscale toggle) -- since
        # lambda^2 is the physically relevant variable for Faraday
        # rotation (same convention as ModelPlot's own p/EVPA-vs-
        # lambda^2 axis). Descending w2 <=> ascending nu, so the grid
        # walks low nu (red) -> high nu (violet) just like band_colors().
        w2_max, w2_min = wl_ext.max() ** 2, wl_ext.min() ** 2
        n_qu = min(len(wl_ext), QU_POLAR_MAX_SEGMENTS)
        w2_grid = np.linspace(w2_max, w2_min, n_qu)
        wl_qu = np.sqrt(w2_grid)
        I_qu = stokes_I(wl_qu, n_components, pars, nu_min=nu_min)
        Q_qu, U_qu = stokes_QU(wl_qu, model_func, n_components, pars, nu_min=nu_min, I=I_qu)
        q_s, u_s = Q_qu / I_qu, U_qu / I_qu
        points = np.column_stack([q_s, u_s]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        self.line_QU.set_segments(segments)
        # Color by each point's own log10(lambda^2) position within
        # [w2_min, w2_max] -- not linearly in lambda^2 like the point
        # spacing above -- since real receiver bands (see
        # measurements.band_colors) are themselves roughly log-spaced
        # in frequency; a linear-lambda^2 color scale squeezes most of
        # a wide range's high-frequency (small lambda^2) bands into a
        # sliver near one end of the rainbow, visibly desyncing the
        # curve's color from same-band measurement points' own
        # rank-assigned color. w2_grid descends as index increases
        # (large lambda^2/low nu first), so this still walks low nu
        # (red) -> high nu (violet) like band_colors() itself.
        log_w2 = np.log10(w2_grid)
        span = log_w2[0] - log_w2[-1] if len(log_w2) else 0.0
        t = (log_w2[0] - log_w2) / span if span > 0 else np.zeros_like(log_w2)
        self.line_QU.set_array(t[:-1])
        # Own cache key -- evaluated on wl_qu, not wl_ext, so it can't
        # share sample_lines_I's own cache key.
        polar_key = (wl_qu.tobytes(), n_components, nu_min, self._samples_version, model_func) \
            if show_samples else None
        polar_changed = polar_key != self._samples_polar_cache_key
        self._samples_polar_cache_key = polar_key
        if polar_changed:
            for i, ln_QU in enumerate(self.sample_lines_QU):
                if show_samples:
                    s_I = stokes_I(wl_qu, n_components, self.posterior_samples[i], nu_min=nu_min)
                    s_Q, s_U = stokes_QU(wl_qu, model_func, n_components, self.posterior_samples[i], nu_min=nu_min, I=s_I)
                    ln_QU.set_data(s_Q / s_I, s_U / s_I)
                else:
                    ln_QU.set_data([], [])
        return q_s, u_s

    def _draw_spectra_mode(self, arrays, samples_changed, show_samples,
                            wl_ext, model_func, n_components, nu_min, order):
        """Draw the raw Q(nu)/U(nu) curves (total + per-component) and
        refresh their posterior-sample overlay."""
        nu_s = arrays['nu_s']
        self.line_Q.set_data(nu_s, arrays['Q_s'])
        self.line_U.set_data(nu_s, arrays['U_s'])
        if arrays['two_comp']:
            self.line_Q1.set_data(nu_s, arrays['Q1_s'])
            self.line_U1.set_data(nu_s, arrays['U1_s'])
            self.line_Q2.set_data(nu_s, arrays['Q2_s'])
            self.line_U2.set_data(nu_s, arrays['U2_s'])
        else:
            self.line_Q1.set_data([], [])
            self.line_U1.set_data([], [])
            self.line_Q2.set_data([], [])
            self.line_U2.set_data([], [])
        if samples_changed:
            for i, (ln_Q, ln_U) in enumerate(zip(self.sample_lines_Q, self.sample_lines_U)):
                if show_samples:
                    s_Q, s_U = (a[order] for a in stokes_QU(
                        wl_ext, model_func, n_components, self.posterior_samples[i], nu_min=nu_min))
                    ln_Q.set_data(nu_s, s_Q)
                    ln_U.set_data(nu_s, s_U)
                else:
                    ln_Q.set_data([], [])
                    ln_U.set_data([], [])

    def _apply_axis_limits(self, arrays, q_s, u_s):
        """Widen and apply ax_I/ax_QU x/y limits to cover the model
        curve(s) plus any loaded reference/measurement data, following
        this.mode's own axis convention (Polar: square fractional q/u;
        Spectra: raw Q/U symmetric about 0). `q_s`/`u_s` (Polar only,
        from _draw_polar_mode) are None in Spectra mode."""
        nu_s, I_s, two_comp = arrays['nu_s'], arrays['I_s'], arrays['two_comp']

        xlo, xhi = nu_s.min() * 0.9, nu_s.max() * 1.05
        i_seed_min, i_seed_max = _finite_bounds(I_s, fallback=(1e-6, 1e-6))
        i_min = max(i_seed_min, 1e-6)
        i_max = max(i_seed_max, 1e-6)
        # Total-only Q/U extent -- what the Polar view's own limits are
        # based on (see below): it only ever draws the total curve, so
        # widening it to also cover the per-component curves (Spectra-only
        # artists) would make its square limits depend on lines it never
        # shows. q_min_spectra/q_max_spectra etc. are the Spectra view's
        # own (component-widened) counterpart.
        q_min, q_max = _finite_bounds(arrays['Q_s'], fallback=(0.0, 0.0))
        u_min, u_max = _finite_bounds(arrays['U_s'], fallback=(0.0, 0.0))
        if two_comp:
            i1_seed_min, i1_seed_max = _finite_bounds(arrays['I1_s'], fallback=(1e-6, 1e-6))
            i2_seed_min, i2_seed_max = _finite_bounds(arrays['I2_s'], fallback=(1e-6, 1e-6))
            i_min = min(i_min, max(i1_seed_min, 1e-6), max(i2_seed_min, 1e-6))
            i_max = max(i_max, i1_seed_max, i2_seed_max)
        # Widen the autoscale to also cover any reference data loaded via
        # the Load Data... button, so it isn't silently clipped out of view.
        if self.ref_data is not None:
            ref_nu, ref_I, _, ref_Q, _, ref_U, _ = self.ref_data
            xlo, xhi = min(xlo, ref_nu.min() * 0.9), max(xhi, ref_nu.max() * 1.05)
            i_min = min(i_min, max(ref_I.min(), 1e-6))
            i_max = max(i_max, ref_I.max())
            q_min, q_max = _widen_range(q_min, q_max, ref_Q)
            u_min, u_max = _widen_range(u_min, u_max, ref_U)
        if self.meas_bands:
            for band in self.meas_bands:
                xlo = min(xlo, band['nu'].min() * 0.9)
                xhi = max(xhi, band['nu'].max() * 1.05)
                i_min = min(i_min, max((band['I'] - band['I_err']).min(), 1e-6))
                i_max = max(i_max, (band['I'] + band['I_err']).max())
                q_min, q_max = _widen_range(q_min, q_max, band['Q'], band['Q_err'])
                u_min, u_max = _widen_range(u_min, u_max, band['U'], band['U_err'])

        q_min_spectra, q_max_spectra = q_min, q_max
        u_min_spectra, u_max_spectra = u_min, u_max
        if two_comp:
            q1_seed_min, q1_seed_max = _finite_bounds(arrays['Q1_s'], fallback=(0.0, 0.0))
            q2_seed_min, q2_seed_max = _finite_bounds(arrays['Q2_s'], fallback=(0.0, 0.0))
            u1_seed_min, u1_seed_max = _finite_bounds(arrays['U1_s'], fallback=(0.0, 0.0))
            u2_seed_min, u2_seed_max = _finite_bounds(arrays['U2_s'], fallback=(0.0, 0.0))
            q_min_spectra = min(q_min_spectra, q1_seed_min, q2_seed_min)
            q_max_spectra = max(q_max_spectra, q1_seed_max, q2_seed_max)
            u_min_spectra = min(u_min_spectra, u1_seed_min, u2_seed_min)
            u_max_spectra = max(u_max_spectra, u1_seed_max, u2_seed_max)

        self.ax_I.set_xlim(xlo, xhi)
        self.ax_I.set_ylim(0.7 * i_min, 1.3 * i_max)

        if self.mode == 'Polar':
            # Fractional (q, u) extent -- separate from q_min/q_max/
            # u_min/u_max above (raw Q/U, used by the Spectra view only),
            # since Polar plots q=Q/I, u=U/I instead. Based on the total
            # curve alone (+ ref/meas data, central values only -- no
            # error margin, matching q_min/q_max's own ref_data widening
            # above) -- the Polar view never draws the per-component
            # curves, so they don't belong in its own limits.
            frac_q_min, frac_q_max = _finite_bounds(q_s, fallback=(0.0, 0.0))
            frac_u_min, frac_u_max = _finite_bounds(u_s, fallback=(0.0, 0.0))
            if self.ref_data is not None:
                ref_I, ref_Q, ref_U = self.ref_data[1], self.ref_data[3], self.ref_data[5]
                ref_q, ref_u, _, _ = stokes_to_frac_qu(ref_I, ref_Q, ref_U)
                frac_q_min, frac_q_max = _widen_range(frac_q_min, frac_q_max, ref_q)
                frac_u_min, frac_u_max = _widen_range(frac_u_min, frac_u_max, ref_u)
            if self.meas_bands:
                for band in self.meas_bands:
                    band_q, band_u, band_q_err, band_u_err = stokes_to_frac_qu(
                        band['I'], band['Q'], band['U'], band['I_err'], band['Q_err'], band['U_err'])
                    frac_q_min, frac_q_max = _widen_range(frac_q_min, frac_q_max, band_q, band_q_err)
                    frac_u_min, frac_u_max = _widen_range(frac_u_min, frac_u_max, band_u, band_u_err)
            # Square numeric limits (not a square axes box -- see class
            # docstring): +/-1.25 * the larger of q's and u's own max
            # absolute extent.
            half_side = 1.25 * max(max(abs(frac_q_min), abs(frac_q_max)), max(abs(frac_u_min), abs(frac_u_max)))
            half_side = max(half_side, 1e-6)
            self.ax_QU.set_xlim(-half_side, half_side)
            self.ax_QU.set_ylim(-half_side, half_side)
        else:
            qu_max = max(abs(q_min_spectra), abs(q_max_spectra), abs(u_min_spectra), abs(u_max_spectra), 1e-6)
            self.ax_QU.set_xlim(xlo, xhi)
            self.ax_QU.set_ylim(-1.3 * qu_max, 1.3 * qu_max)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0}, margins=STOKES_MARGINS_PX)


class RMSynthPlot(FigureCanvas):
    """The 'RM-synth' tab's canvas -- the Faraday-depth spectrum |F(phi)|
    from RM synthesis + RM-CLEAN (see rm_synthesis.compute_faraday_spectrum).

    Unlike ModelPlot/StokesPlot, nothing here is wired to a slider drag or
    the wavelength-range spin boxes: it only ever changes when the user
    explicitly picks a source via the 'Synthesize from' buttons (see
    app.MainWindow.rmsynth_from_model/_data/_measurements), each of which
    calls plot_spectrum() and replaces whatever was drawn before outright.
    set_empty() is the only other entry point, used both at startup and to
    show a plausible phi axis range (from the current wavelength selection,
    see app.MainWindow.refresh_rmsynth_empty_axis) whenever nothing has
    been synthesized yet, or the last synthesized source's own data has
    been cleared (Clear data / the Measurements tab's Clear).

    A plain ax.clear() + full redraw is used throughout (unlike those other
    two canvases' careful set_data()-only updates) since redraws here are
    rare, user-initiated events, not something that has to keep up with a
    slider drag."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.subplots()
        apply_fixed_margins(self.fig, self)
        self.set_empty(1.0)

    def _style_axes(self, exponent=0):
        self.ax.set_xlabel(rf'Faraday depth $\phi$  [ $10^{{{exponent}}}$ rad m$^{{-2}}$ ]')
        self.ax.set_ylabel(r'$|F(\phi)|$  [ fractional polarization ]')
        self.ax.grid(True, linestyle='dotted')
        _bound_yaxis_ticklabels(self.ax)

    def set_empty(self, phi_half_width):
        """Reset to a blank plot spanning +/-`phi_half_width` [rad/m^2] --
        the range of Faraday depths reachable by the currently selected
        wavelength range (see rm_synthesis.phi_axis_half_width), with
        nothing plotted yet."""
        self.ax.clear()
        phi_half_width = max(phi_half_width, 1e-30)
        exponent = int(np.floor(np.log10(phi_half_width)))
        self._style_axes(exponent)
        scale = 10.0 ** exponent
        self.ax.set_xlim(-phi_half_width / scale, phi_half_width / scale)
        self.ax.set_ylim(0, 1)
        apply_fixed_margins(self.fig, self)
        self.draw_idle()

    def plot_spectrum(self, result, source_label):
        """Draw the dirty/clean Faraday spectrum from `result` (an
        rm_synthesis.compute_faraday_spectrum() return dict); `source_label`
        ('model'/'data'/'measurements') only affects the title."""
        self.ax.clear()
        ax = self.ax
        phi = result['phi']
        exponent = int(np.floor(np.log10(phi[-1]))) if phi[-1] > 0 else 0
        self._style_axes(exponent)
        scale = 10.0 ** exponent
        phi = phi / scale
        amp_dirty = np.abs(result['F_dirty'])
        amp_clean = np.abs(result['F_clean'])

        ax.plot(phi, amp_dirty, color='grey', ls='dashed', lw=1, label=r'$|F(\phi)|$ dirty')
        ax.plot(phi, amp_clean, color='tab:green', lw=1.5, label=r'$|F(\phi)|$ clean')

        nz = np.nonzero(result['components'])[0]
        if len(nz):
            markerline, _, _ = ax.stem(
                phi[nz], np.abs(result['components'][nz]), linefmt='tab:orange',
                markerfmt='o', basefmt=' ', label='CLEAN components')
            markerline.set_markersize(4)
            markerline.set_markerfacecolor('white')
            markerline.set_markeredgecolor('tab:orange')

        sigma_fdf = result['sigma_fdf']
        if sigma_fdf > 0:
            ax.axhline(sigma_fdf, color='r', lw=1, ls='dotted', label=r'$\sigma_{noise}$')

        phi_peak, phi_err, fwhm = result['phi_peak'], result['phi_err'], result['fwhm']
        ax.axvline(phi_peak / scale, color='k', lw=0.8, ls=':')
        err = phi_err if np.isfinite(phi_err) else None
        label = (r'$\phi_{\rm peak} = $' + sci_latex(phi_peak, err) + '\n'
                  + r'$R(\phi) = $' + sci_latex(fwhm))
        ax.annotate(label, xy=(0.03, 0.92), xycoords='axes fraction', color='k', va='top')

        # RMSF "beam" scale bar -- a solid black horizontal segment, length =
        # the RMSF main-lobe FWHM (in the plot's own phi units, see
        # exponent/scale above), directly below the phi_peak/RMSF text so
        # its extent reads as that RMSF value's own visual scale -- same
        # device as RM_synth.py's own plot_clean.
        beam_trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        beam_y = 0.78
        beam_x0 = phi[0] + 0.03 * (phi[-1] - phi[0])
        beam_x1 = beam_x0 + fwhm / scale
        ax.plot([beam_x0, beam_x1], [beam_y, beam_y], color='k', lw=2,
                solid_capstyle='butt', transform=beam_trans)
        cap = 0.01
        for x in (beam_x0, beam_x1):
            ax.plot([x, x], [beam_y - cap, beam_y + cap], color='k', lw=2, transform=beam_trans)

        ax.set_xlim(phi[0], phi[-1])
        ax.set_ylim(0, 1.3 * max(np.max(amp_dirty), np.max(amp_clean), 1e-12))
        ax.set_title(f'Faraday spectrum - from {source_label}')
        ax.legend(loc='upper right', fontsize=14)
        apply_fixed_margins(self.fig, self)
        self.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        apply_fixed_margins(self.fig, self)
