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

from polvista.models import C, stokes_I, stokes_QU, stokes_components, evpa, pol
from polvista.measurements import RAINBOW_HUE_MAX

SLIDER_STEPS = 10000  # integer resolution backing each QSlider

# Physical unit shown next to each slider's readout, keyed by Param.kind.
UNITS = {'p': '%', 'X': '°', 'phi': 'rad/m²', 'dphi': 'rad/m²', 'scale': '', 'alpha': '', 'eps': '',
         'nu0': 'GHz', 'temp': 'K'}
# Widest unit string, used to fix every slider's unit label to the same
# width so the value boxes above/below each other line up regardless of
# which unit (or none) a given row happens to show.
WIDEST_UNIT = max(UNITS.values(), key=len)

NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


class ValueLineEdit(QLineEdit):
    """A QLineEdit that selects all its text on focus, so clicking a
    parameter's value and typing immediately replaces it."""
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.selectAll()


# Fixed PIXEL margins for the two plot canvases below (not fractions -- see
# apply_fixed_margins). Sized generously from measured worst cases (long
# negative numbers, scientific notation, wide mathtext labels) so the
# right-side rotated ylabel + tick labels are always fully visible
PLOT_MARGINS_PX = dict(left=85, right=125, bottom=75, top=25)

# Continuous version of measurements.py's band_colors() red->violet HSV
# sweep (same RAINBOW_HUE_MAX endpoint), used to color StokesPlot's Polar
# view (Q, U) curve by frequency so it reads on the same red=low-nu,
# violet=high-nu convention as the Measurements tab's simulated points.
QU_RAINBOW_CMAP = mcolors.ListedColormap(
    [colorsys.hsv_to_rgb(RAINBOW_HUE_MAX * t, 1.0, 1.0) for t in np.linspace(0, 1, 256)])


def apply_fixed_margins(fig, canvas, extra_adjust=None):
    """Set fig.subplots_adjust margins from PLOT_MARGINS_PX, converted to
    the fractions matplotlib wants using the canvas's *current* pixel size
    """

    w = max(canvas.width(), 1)
    h = max(canvas.height(), 1)
    m = PLOT_MARGINS_PX
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

        # Static per-axes config set once here, not on every update_plot()
        # call: Axes.clear() is expensive (it tears down and rebuilds the
        # whole axis/spine/tick machinery -- profiling showed it dominates
        # redraw time), so slider drags reuse these Line2D objects via
        # set_data() instead of clear()+replot(), and everything that
        # doesn't change between frames (labels, grid) is set only once.
        self.ax_p.set_ylabel(r'$p$ [ % ]')
        self.ax_p.grid(True)
        self.ax_x.set_ylabel(r'$\chi$ [ deg ]', rotation=270, labelpad=15)
        self.ax_x.grid(True)
        self.line_p, = self.ax_p.plot([], [], color='tab:blue')
        self.line_x, = self.ax_x.plot([], [], color='tab:red')
        self.xscale = None  # forces the first set_xscale('log') call to actually apply
        self.ref_data = None  # (w2, p, p_err, evpa, evpa_err) from the Load Data... button, or None
        self.ref_artists = []
        # Simulated points from the Measurements tab's Generate button --
        # a list of per-band dicts (color, w2, p, p_err, evpa, evpa_err), or
        # None. Kept separate from ref_data (Load Data's real reference
        # points) so the two overlays never clobber each other.
        self.meas_bands = None
        self.meas_artists = []

        # Posterior-sample "spaghetti" overlay (see set_posterior_samples) --
        # a fixed pool of faint Line2D artists, created only when the
        # sample set itself changes (a MultiNest fit/Load samples/Reset
        # model), then refreshed via set_data() on every update_plot() call
        # like the main curve, so a slider drag repaints them at the
        # current wavelength range without the cost of recreating them.
        self.posterior_samples = None  # (n, ndim_full) array, model's full param order, or None
        self.posterior_model = None    # the model func these samples belong to
        self.sample_lines_p = []
        self.sample_lines_x = []

    # how to load data
    def set_reference_data(self, w2, p, p_err, evpa, evpa_err):
        self.ref_data = (np.asarray(w2), np.asarray(p), p_err, np.asarray(evpa), evpa_err)

    def clear_reference_data(self):
        self.ref_data = None

    def set_measurement_data(self, bands):
        """`bands` is a list of per-band dicts (see measurements.py's
        generate_measurements), or None to clear."""
        self.meas_bands = bands

    def clear_measurement_data(self):
        self.meas_bands = None

    # how data is plotted
    def draw_reference(self):
        for artist in self.ref_artists:
            artist.remove()
        self.ref_artists = []
        if self.ref_data is not None:
            w2, p, p_err, evpa, evpa_err = self.ref_data
            self.ref_artists.append(self.ax_p.errorbar(
                                    w2, p, yerr=p_err, fmt='o', ms=4, color='black', ecolor='0.5', capsize=2, zorder=5))
            self.ref_artists.append(self.ax_x.errorbar(
                                    w2, evpa, yerr=evpa_err, fmt='o', ms=4, color='black', ecolor='0.5', capsize=2, zorder=5))

        for artist in self.meas_artists:
            artist.remove()
        self.meas_artists = []
        if self.meas_bands:
            for band in self.meas_bands:
                c = band['color']
                self.meas_artists.append(self.ax_p.errorbar(
                    band['w2'], band['p'], yerr=band['p_err'], fmt='D', ms=4, mew=0.5, mec='k',
                    color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))
                self.meas_artists.append(self.ax_x.errorbar(
                    band['w2'], band['evpa'], yerr=band['evpa_err'], fmt='D', ms=4, mew=0.5, mec='k',
                    color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))

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
        if samples is not None:
            for _ in range(len(samples)):
                ln_p, = self.ax_p.plot([], [], color='tab:blue', alpha=0.12, lw=0.7, zorder=1)
                ln_x, = self.ax_x.plot([], [], color='tab:red', alpha=0.12, lw=0.7, zorder=1)
                self.sample_lines_p.append(ln_p)
                self.sample_lines_x.append(ln_x)
        self.draw_idle()

    def update_plot(self, wl_ext, model_func, pars, log_xscale=False):
        fit = model_func(wl_ext, pars)
        p = pol(fit)
        X = evpa(fit)
        w2 = wl_ext ** 2 * 1e6  # lambda^2 [mm^2]

        # Posterior-sample overlay: only actually drawn while the
        # currently-selected model matches the one these samples were fit
        # for -- switching the model dropdown blanks them (set_data([],[]))
        # without discarding them, so they silently reappear if the user
        # switches back, rather than needing an explicit clear.
        show_samples = self.posterior_samples is not None and self.posterior_model is model_func
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
        p_max = max(np.max(p), 1e-6)
        x_lo, x_hi = float(np.min(X)), float(np.max(X))
        # Widen the autoscale to also cover any reference data loaded via
        # the Load Data... button, so it isn't silently clipped out of view.
        if self.ref_data is not None:
            ref_w2, ref_p, _, ref_evpa, _ = self.ref_data
            xlo, xhi = min(xlo, np.min(ref_w2) * 0.9), max(xhi, np.max(ref_w2) * 1.05)
            p_max = max(p_max, np.max(ref_p))
            x_lo, x_hi = min(x_lo, float(np.min(ref_evpa))), max(x_hi, float(np.max(ref_evpa)))
        if self.meas_bands:
            for band in self.meas_bands:
                xlo = min(xlo, np.min(band['w2']) * 0.9)
                xhi = max(xhi, np.max(band['w2']) * 1.05)
                p_max = max(p_max, np.max(band['p'] + band['p_err']))
                x_lo = min(x_lo, float(np.min(band['evpa'] - band['evpa_err'])))
                x_hi = max(x_hi, float(np.max(band['evpa'] + band['evpa_err'])))

        self.line_p.set_data(w2, p)
        self.ax_p.set_xlim(xlo, xhi)
        self.ax_p.set_ylim(0, 1.3 * p_max)

        self.line_x.set_data(w2, X)
        self.ax_x.set_xlim(xlo, xhi)

        # Enforce a minimum 2 deg y-span so a near-flat EVPA curve (small
        # |phi|) doesn't get visually exaggerated by an overly tight autoscale.
        margin = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 0.05
        x_lo, x_hi = x_lo - margin, x_hi + margin
        if x_hi - x_lo < 2.0:
            mid = 0.5 * (x_lo + x_hi)
            x_lo, x_hi = mid - 1.0, mid + 1.0
        self.ax_x.set_ylim(x_lo, x_hi)
        self.draw_reference()
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
    [GHz] -- the model's own trailing spectral params (alpha for a
    single component, eps/alpha1/alpha2 for two) double as a real
    (normalized, I_0=1) Stokes I(nu) model; Q(nu), U(nu) are then
    just I(nu) scaled by the same fractional-polarization complex number
    p*e^(2i*EVPA) already used for the p/EVPA plot.

    The two subplots sit flush against each other, no gap, matching
    ModelPlot's own p/EVPA pair -- ax_QU's y-axis is mirrored onto its
    right spine (see build_qu_mode) so its tick labels don't collide with
    ax_I's across the shared middle seam.

    The right subplot has two interchangeable views, picked via the
    dropdown above it (see stokes_mode_combo): 'Spectra' plots Q(nu) and
    U(nu) like the left subplot; 'Polar' instead plots the fractional
    (q=Q/I, u=U/I) trajectory traced out as nu varies, u vs q, colored
    along its length as a continuous red(low nu)->violet(high nu) rainbow
    (see stokes_to_frac_qu and QU_RAINBOW_CMAP) matching the Measurements
    tab's own per-band coloring. Its limits are square (equal q/u numeric
    span) but the axes box itself stays whatever rectangular shape the
    flush layout gives
    it -- forcing a literal square box would fight the shared wspace=0
    layout with ax_I. Switching modes rebuilds the right axis from
    scratch (ax_QU.clear() + fresh artists) -- only happens on a rare,
    user-initiated dropdown change, not per-frame, so it doesn't need the
    set_data()-only treatment the rest of this canvas uses to keep slider
    drags cheap."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax_I, self.ax_QU = self.fig.subplots(1, 2, gridspec_kw={'wspace': 0})
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0})

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
        self.build_qu_mode()

    def build_qu_mode(self):
        """(Re)configure the right-hand axis for self.mode: static
        labels/grid/aspect, the main curve artist(s), and a posterior-
        sample line pool sized to len(self.posterior_samples). Called
        once at init, on every mode-dropdown switch, and whenever the
        posterior-sample count changes (set_posterior_samples) -- all rare
        enough that a full clear()+rebuild is fine."""
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
            self.ax_QU.set_ylabel(r'$u$', rotation=270, labelpad=15)
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
            self.ax_QU.set_ylabel(r'$Q$, $U$ [ normalized ]', rotation=270, labelpad=15)
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

    def set_posterior_samples(self, samples, model_func):
        for ln in self.sample_lines_I:
            ln.remove()
        self.sample_lines_I = []
        self.posterior_samples = samples
        self.posterior_model = model_func
        if samples is not None:
            for _ in range(len(samples)):
                ln_I, = self.ax_I.plot([], [], color='black', alpha=0.08, lw=0.6, zorder=1)
                self.sample_lines_I.append(ln_I)
        self.build_qu_mode()
        self.draw_idle()

    def set_reference_data(self, nu_ghz, I, I_err, Q, Q_err, U, U_err):
        self.ref_data = (np.asarray(nu_ghz), np.asarray(I), I_err, np.asarray(Q), Q_err, np.asarray(U), U_err)

    def clear_reference_data(self):
        self.ref_data = None

    def set_measurement_data(self, bands):
        """`bands` is a list of per-band dicts (see measurements.py's
        generate_measurements), or None to clear."""
        self.meas_bands = bands

    def clear_measurement_data(self):
        self.meas_bands = None

    # draw data
    def draw_reference(self):
        for artist in self.ref_artists_I:
            artist.remove()
        self.ref_artists_I = []
        for artist in self.ref_artists_QU:
            artist.remove()
        self.ref_artists_QU = []
        if self.ref_data is not None:
            nu, I, I_err, Q, Q_err, U, U_err = self.ref_data
            self.ref_artists_I.append(self.ax_I.errorbar(
                nu, I, yerr=I_err, fmt='o', ms=4, color='black', ecolor='0.5', capsize=2, zorder=5))
            if self.mode == 'Polar':
                q, u, q_err, u_err = stokes_to_frac_qu(I, Q, U, I_err, Q_err, U_err)
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    q, u, xerr=q_err, yerr=u_err, fmt='o', ms=4, color='black', ecolor='0.5', capsize=2, zorder=5))
            else:
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    nu, Q, yerr=Q_err, fmt='s', ms=4, color='darkgreen', ecolor='0.5', capsize=2, zorder=5))
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    nu, U, yerr=U_err, fmt='^', ms=4, color='darkorange', ecolor='0.5', capsize=2, zorder=5))

        for artist in self.meas_artists_I:
            artist.remove()
        self.meas_artists_I = []
        for artist in self.meas_artists_QU:
            artist.remove()
        self.meas_artists_QU = []
        if self.meas_bands:
            for band in self.meas_bands:
                c = band['color']
                self.meas_artists_I.append(self.ax_I.errorbar(
                    band['nu'], band['I'], yerr=band['I_err'], fmt='o', ms=4, mew=0.5, mec='k',
                    color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))
                if self.mode == 'Polar':
                    q, u, q_err, u_err = stokes_to_frac_qu(
                        band['I'], band['Q'], band['U'], band['I_err'], band['Q_err'], band['U_err'])
                    self.meas_artists_QU.append(self.ax_QU.errorbar(
                        q, u, xerr=q_err, yerr=u_err, fmt='o', ms=4, mew=0.5,
                        mec='k', color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))
                else:
                    self.meas_artists_QU.append(self.ax_QU.errorbar(
                        band['nu'], band['Q'], yerr=band['Q_err'], fmt='s', ms=4, mew=0.5, mec='k',
                        color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))
                    self.meas_artists_QU.append(self.ax_QU.errorbar(
                        band['nu'], band['U'], yerr=band['U_err'], fmt='^', ms=4, mew=0.5, mec='k',
                        color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))

    def update_plot(self, wl_ext, model_func, n_components, pars, log_xscale=False, nu_min=None, mode='Spectra'):
        if mode != self.mode:
            self.mode = mode
            self.build_qu_mode()

        I = stokes_I(wl_ext, n_components, pars, nu_min=nu_min)
        Q, U = stokes_QU(wl_ext, model_func, n_components, pars, nu_min=nu_min)
        nu_ghz = C / wl_ext / 1e9

        order = np.argsort(nu_ghz)
        nu_s, I_s, Q_s, U_s = nu_ghz[order], I[order], Q[order], U[order]

        two_comp = n_components == 2
        if two_comp:
            (I1, Q1, U1), (I2, Q2, U2) = stokes_components(wl_ext, model_func, pars, nu_min=nu_min)
            I1_s, Q1_s, U1_s = I1[order], Q1[order], U1[order]
            I2_s, Q2_s, U2_s = I2[order], Q2[order], U2[order]

        if two_comp != self.ax_I_two_comp:
            self.ax_I_two_comp = two_comp
            handles = [self.line_I, self.line_I1, self.line_I2] if two_comp else [self.line_I]
            self.ax_I.legend(handles, [h.get_label() for h in handles], loc='lower right')

        xscale = 'log' if log_xscale else 'linear'
        if xscale != self.xscale:
            self.ax_I.set_xscale(xscale)
            if self.mode == 'Spectra':
                self.ax_QU.set_xscale(xscale)
            self.xscale = xscale

        self.line_I.set_data(nu_s, I_s)
        if two_comp:
            self.line_I1.set_data(nu_s, I1_s)
            self.line_I2.set_data(nu_s, I2_s)
        else:
            self.line_I1.set_data([], [])
            self.line_I2.set_data([], [])

        show_samples = self.posterior_samples is not None and self.posterior_model is model_func
        for i, ln_I in enumerate(self.sample_lines_I):
            if show_samples:
                s_I = stokes_I(wl_ext, n_components, self.posterior_samples[i], nu_min=nu_min)[order]
                ln_I.set_data(nu_s, s_I)
            else:
                ln_I.set_data([], [])

        if self.mode == 'Polar':
            # Resample evenly in lambda^2 -- not in lambda/log-lambda like
            # wl_ext itself (that spacing stays as-is for the I panel and
            # Spectra mode, driven by the log_xscale toggle) -- since
            # lambda^2 is the physically relevant variable for Faraday
            # rotation (same convention as ModelPlot's own p/EVPA-vs-
            # lambda^2 axis). Descending w2 <=> ascending nu, so the grid
            # walks low nu (red) -> high nu (violet) just like band_colors().
            w2_max, w2_min = wl_ext.max() ** 2, wl_ext.min() ** 2
            w2_grid = np.linspace(w2_max, w2_min, len(wl_ext))
            wl_qu = np.sqrt(w2_grid)
            I_qu = stokes_I(wl_qu, n_components, pars, nu_min=nu_min)
            Q_qu, U_qu = stokes_QU(wl_qu, model_func, n_components, pars, nu_min=nu_min)
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
            for i, ln_QU in enumerate(self.sample_lines_QU):
                if show_samples:
                    s_I = stokes_I(wl_qu, n_components, self.posterior_samples[i], nu_min=nu_min)
                    s_Q, s_U = stokes_QU(wl_qu, model_func, n_components, self.posterior_samples[i], nu_min=nu_min)
                    ln_QU.set_data(s_Q / s_I, s_U / s_I)
                else:
                    ln_QU.set_data([], [])
        else:
            self.line_Q.set_data(nu_s, Q_s)
            self.line_U.set_data(nu_s, U_s)
            if two_comp:
                self.line_Q1.set_data(nu_s, Q1_s)
                self.line_U1.set_data(nu_s, U1_s)
                self.line_Q2.set_data(nu_s, Q2_s)
                self.line_U2.set_data(nu_s, U2_s)
            else:
                self.line_Q1.set_data([], [])
                self.line_U1.set_data([], [])
                self.line_Q2.set_data([], [])
                self.line_U2.set_data([], [])
            for i, (ln_Q, ln_U) in enumerate(zip(self.sample_lines_Q, self.sample_lines_U)):
                if show_samples:
                    s_Q, s_U = (a[order] for a in stokes_QU(
                        wl_ext, model_func, n_components, self.posterior_samples[i], nu_min=nu_min))
                    ln_Q.set_data(nu_s, s_Q)
                    ln_U.set_data(nu_s, s_U)
                else:
                    ln_Q.set_data([], [])
                    ln_U.set_data([], [])

        xlo, xhi = nu_s.min() * 0.9, nu_s.max() * 1.05
        i_min = max(I_s.min(), 1e-6)
        i_max = max(I_s.max(), 1e-6)
        # Total-only Q/U extent -- what the Polar view's own limits are
        # based on (see below): it only ever draws the total curve, so
        # widening it to also cover the per-component curves (Spectra-only
        # artists) would make its square limits depend on lines it never
        # shows. q_min_spectra/q_max_spectra etc. are the Spectra view's
        # own (component-widened) counterpart.
        q_min, q_max = float(Q_s.min()), float(Q_s.max())
        u_min, u_max = float(U_s.min()), float(U_s.max())
        if two_comp:
            i_min = min(i_min, max(I1_s.min(), 1e-6), max(I2_s.min(), 1e-6))
            i_max = max(i_max, I1_s.max(), I2_s.max())
        # Widen the autoscale to also cover any reference data loaded via
        # the Load Data... button, so it isn't silently clipped out of view.
        if self.ref_data is not None:
            ref_nu, ref_I, _, ref_Q, _, ref_U, _ = self.ref_data
            xlo, xhi = min(xlo, ref_nu.min() * 0.9), max(xhi, ref_nu.max() * 1.05)
            i_min = min(i_min, max(ref_I.min(), 1e-6))
            i_max = max(i_max, ref_I.max())
            q_min, q_max = min(q_min, ref_Q.min()), max(q_max, ref_Q.max())
            u_min, u_max = min(u_min, ref_U.min()), max(u_max, ref_U.max())
        if self.meas_bands:
            for band in self.meas_bands:
                xlo = min(xlo, band['nu'].min() * 0.9)
                xhi = max(xhi, band['nu'].max() * 1.05)
                i_min = min(i_min, max((band['I'] - band['I_err']).min(), 1e-6))
                i_max = max(i_max, (band['I'] + band['I_err']).max())
                q_min = min(q_min, (band['Q'] - band['Q_err']).min())
                q_max = max(q_max, (band['Q'] + band['Q_err']).max())
                u_min = min(u_min, (band['U'] - band['U_err']).min())
                u_max = max(u_max, (band['U'] + band['U_err']).max())

        q_min_spectra, q_max_spectra = q_min, q_max
        u_min_spectra, u_max_spectra = u_min, u_max
        if two_comp:
            q_min_spectra = min(q_min_spectra, Q1_s.min(), Q2_s.min())
            q_max_spectra = max(q_max_spectra, Q1_s.max(), Q2_s.max())
            u_min_spectra = min(u_min_spectra, U1_s.min(), U2_s.min())
            u_max_spectra = max(u_max_spectra, U1_s.max(), U2_s.max())

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
            frac_q_min, frac_q_max = float(q_s.min()), float(q_s.max())
            frac_u_min, frac_u_max = float(u_s.min()), float(u_s.max())
            if self.ref_data is not None:
                ref_I, ref_Q, ref_U = self.ref_data[1], self.ref_data[3], self.ref_data[5]
                ref_q, ref_u, _, _ = stokes_to_frac_qu(ref_I, ref_Q, ref_U)
                frac_q_min, frac_q_max = min(frac_q_min, ref_q.min()), max(frac_q_max, ref_q.max())
                frac_u_min, frac_u_max = min(frac_u_min, ref_u.min()), max(frac_u_max, ref_u.max())
            if self.meas_bands:
                for band in self.meas_bands:
                    band_q, band_u, band_q_err, band_u_err = stokes_to_frac_qu(
                        band['I'], band['Q'], band['U'], band['I_err'], band['Q_err'], band['U_err'])
                    frac_q_min = min(frac_q_min, (band_q - band_q_err).min())
                    frac_q_max = max(frac_q_max, (band_q + band_q_err).max())
                    frac_u_min = min(frac_u_min, (band_u - band_u_err).min())
                    frac_u_max = max(frac_u_max, (band_u + band_u_err).max())
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

        self.draw_reference()
        self.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0})
