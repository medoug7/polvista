"""
Author: Douglas Carlos, 2026

    Polvista (POLariztion VISualization Tool for Astronomy) is a PyQt5
    widget to interactively visualize polarization-fraction / EVPA 
    spectra for different Faraday-depolarization models

    
    Under the "Visualization" tab, select a model from the dropdown,
    then drag its parameter sliders to see the p(lambda) and EVPA(lambda)
    spectra update live, as well as the predicted behavior for Stokes I, Q, and U.

    In the "Measurements" tab offers some basic tools for simulating in-band
    measurements of the spectral index (alpha), the depolarization measure (D),
    and the rotation measure (RM), for a selection of real-world observatories.

    We Can load some data and perform simple least-squares regression to any
    given model throught the QU-fitting technique. 

    If pymultinest is installed, polvista will also offer a basic interface
    to perform bayesian fitting/sampling under the "Sampling" tab.
    Once a fitting is complete, we can then check the parmeter's posterior
    probability density distributions.

"""
import os
import sys


def ensure_qt_plugin_path():
    """Some conda envs ship a `PyQt5` wheel whose own Qt5/plugins directory
    is missing (QLibraryInfo.PluginsPath points at a folder that doesn't
    exist), while the real Qt platform plugins (xcb, etc.) live under the
    conda env's own `plugins/` dir (from the separate `qt-main` package).
    Without this, Qt fails with "Could not find the Qt platform plugin
    xcb" even though the plugin is present on disk. Auto-detect and set
    QT_QPA_PLATFORM_PLUGIN_PATH so the app runs without env-var wrangling."""
    if os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH'):
        return
    import PyQt5
    candidates = [
        os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins'),
        os.path.join(sys.prefix, 'plugins'),  # conda's qt-main package layout
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'platforms')):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = c
            return


ensure_qt_plugin_path()

import re
import csv
import json

import numpy as np
import matplotlib as mpl
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QComboBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTabWidget, QPushButton, QProgressBar,
    QFileDialog, QMessageBox, QAction)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from polvista.models import (
    MODELS, MODELS_BY_NAME, C, Param, stokes_I, stokes_QU, evpa, pol,
    set_spectral_shape, full_equation)
from polvista.fitting import (
    qu_fit, estimate_alpha, estimate_ssa_shape, fit_statistics, multinest_fit, load_previous_run)
from polvista.latex_stuff import latex_pixmap, fit_equation_pixmap, TexViewerDialog
from polvista.widgets import ValueLineEdit, NUMBER_RE, SLIDER_STEPS, UNITS, WIDEST_UNIT
from polvista.sampling import SamplingMixin
from polvista.measurements import MeasurementsMixin

# pymultinest (Bayesian/nested-sampling fitting) is an optional dependency --
# it's a compiled Fortran library wrapper that's a much heavier install than
# the rest of polvista's requirements, so the app must still run with plain
# least-squares fitting when it isn't present. HAS_PYMULTINEST gates every
# multinest-only UI element (e.g. the "Sampling" tab, see MainWindow) instead
# of failing at import time.
#
# The pymultinest *package* being importable doesn't mean the compiled
# libmultinest.so it wraps is actually on LD_LIBRARY_PATH -- when it isn't,
# pymultinest itself prints a diagnostic and calls sys.exit(1) instead of
# raising ImportError, so SystemExit has to be caught here too or the whole
# app dies at startup.
try:
    import pymultinest
    HAS_PYMULTINEST = True
except (ImportError, SystemExit):
    HAS_PYMULTINEST = False


# QU-fitting holds epsilon fixed at this value (see MainWindow.run_fit) --
# q, u data alone can't constrain it (see spectral_weights) and this
# app only does QU fitting so far.
FIT_FIXED_EPSILON = 0.5

# Figure default parameters
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 16

# Fixed PIXEL margins for the two plot canvases (not fractions -- see
# apply_fixed_margins). Sized generously from measured worst cases (long
# negative numbers, scientific notation, wide mathtext labels) so the
# right-side rotated ylabel + tick labels are always fully visible
PLOT_MARGINS_PX = dict(left=85, right=125, bottom=75, top=25)


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
##############################################


# Log-decade ranges for the signed Faraday-depth (phi) and turbulent term (dphi)
# sliders -- wide enough to explore the models' full behavior without
# being tied to any particular model's fit-prior bounds.
PHI_LOG_RANGE = (-6.5, 6.5)
DPHI_LOG_RANGE = (0.0, 6.5)

# Linear range for power-law spectral indices
# +/-3 covers typical AGN jet spectral indices.
ALPHA_RANGE = (-3.0, 2.0)
EPS_RANGE = (0.0, 1.0)

# Spectrum box's shape dropdown: (display label, models.set_spectral_shape key).
SPECTRAL_SHAPES = [('Power-law', 'powerlaw'), ('SSA', 'ssa')]

# Multipliers applied to the current wavelength range's own (nu_min, nu_max)
# to get the SSA turnover-frequency sliders' (lo, hi) bounds -- wide enough
# to place a turnover well outside the plotted band in either direction.
NU0_BOUNDS_MULT = (1e-3, 1e3)

# Quick-select wavelength ranges (min_mm, max_mm) for the preset dropdown.
WAVELENGTH_PRESETS = [
    ('Standard ALMA (0.75 - 3.7 mm / 80 - 350 GHz)', 0.75, 3.7),
    ('Full ALMA (0.25 - 9 mm / 30 - 1200 GHz)', 0.25, 9.0),
    ('VLA high (0.6 - 2 cm / 15 - 50 GHz)', 6.0, 20.0),
    ('VLA low (2 - 30 cm / 1 - 15 GHz)', 20.0, 300.0),
    ('Full VLA (0.6 - 30 cm / 1 - 50 GHz)', 6.0, 300.0),
    ('Full MeerKat (17 - 70 cm / 500 - 1800 MHz)', 170.0, 700.0),
    ('Full radio (0.25mm - 70 cm / 500 MHz - 1200 GHz)', 0.25, 700.0)
]


def find_column(fieldnames, *names):
    """Case/whitespace-insensitive lookup of the first of `names` present
    among `fieldnames`; returns the actual (original-case) header, or None."""
    lut = {fn.strip().casefold(): fn for fn in fieldnames}
    for name in names:
        if name in lut:
            return lut[name]
    return None


def find_freq_column(fieldnames):
    """A frequency column is recognized by name starting with 'freq' (covers
    'Frequency', 'Frequency [Hz]', 'freq_hz', ...) or equal to 'nu'."""
    for fn in fieldnames:
        key = fn.strip().casefold()
        if key.startswith('freq') or key in ('nu', 'nu [hz]'):
            return fn
    return None


def freq_unit_hz_multiplier(header):
    """Multiplier that converts a frequency column's values to Hz, inferred
    from a unit annotation in its header (e.g. 'Frequency [GHz]', 'freq_mhz',
    'nu (kHz)'). This app's own exports don't agree on a single unit --
    Save Spectra writes 'Frequency [Hz]' (save_spectra_action) while
    Generate's Export Measurements writes 'Frequency [GHz]'
    (measurements.py.export_measurements_action) -- so Load Data can't
    just assume one. Checked most-specific-first since 'hz' is a substring
    of 'ghz'/'mhz'/'khz'; defaults to Hz (multiplier 1) when no unit is
    recognized, matching the previous hard-coded assumption."""
    key = header.strip().casefold()
    for unit, mult in (('ghz', 1e9), ('mhz', 1e6), ('khz', 1e3), ('hz', 1.0)):
        if unit in key:
            return mult
    return 1.0


def find_error_column(fieldnames, base):
    """Optional per-sample error column for Stokes column `base` ('i'/'q'/'u'),
    recognized under any of the common naming conventions."""
    return find_column(fieldnames, f'{base}_err', f'{base}err', f'e{base}',
                         f'd{base}', f'sigma_{base}', f'sig{base}', f'{base}_error')


class ParamSlider(QWidget):
    """One labeled slider per model parameter.

    Internally a QSlider only understands integers, so each kind of
    parameter defines its own "working domain" (degrees for X, log10 for
    phi/dphi, percentage for p) that the integer steps are linearly mapped 
    over; `value()` decodes back to the physical
    parameter the model function expects.
    """
    valueChanged = pyqtSignal()

    def __init__(self, param, lo, hi, parent=None):
        super().__init__(parent)
        self.kind = param.kind
        self.name = param.name

        if self.kind == 'X':
            self.lo, self.hi = lo * 180 / np.pi, hi * 180 / np.pi
            default = 0.0
        elif self.kind == 'phi':
            self.lo, self.hi = PHI_LOG_RANGE
            default = 0.0
        elif self.kind == 'dphi':
            self.lo, self.hi = DPHI_LOG_RANGE
            default = 0.1
        elif self.kind == 'p':
            # worked in percent, decoded back to the 0.1 fraction the model expects
            self.lo, self.hi = lo * 100, hi * 100
            default = min(10.0, self.hi)
        elif self.kind == 'alpha':
            self.lo, self.hi = ALPHA_RANGE
            default = -0.5
        elif self.kind == 'eps':
            self.lo, self.hi = EPS_RANGE
            default = 0.5
        elif self.kind == 'nu0':
            # log-scale like phi/dphi, but a turnover frequency is never 0,
            # so a plain log10 (no dphi's continuous-through-0 offset trick)
            # over the caller-supplied (lo, hi) [GHz] is enough.
            self.lo, self.hi = np.log10(lo), np.log10(hi)
            default = 0.5 * (self.lo + self.hi)
        else:  # 'scale' -- e.g. s, f
            self.lo, self.hi = lo, hi
            default = 0.5 * (lo + hi)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # parameter names in latex format
        self.name_label = QLabel()
        self.name_label.setPixmap(latex_pixmap(param.latex))
        self.name_label.setMinimumWidth(30)
        self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Shows the parameter's physical meaning, as set in the model's registry.
        if param.description:
            self.name_label.setToolTip(param.description)

        # sliders
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(SLIDER_STEPS)
        self.slider.setMinimumWidth(120)
        self.slider.setValue(self.raw_from_working(default))
        self.slider.valueChanged.connect(self.on_slider_changed)

        # typing box for slider values
        self.value_label = ValueLineEdit()
        self.value_label.setFixedWidth(80)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.editingFinished.connect(self.on_value_edited)

        # units for parameters
        self.unit_label = QLabel(UNITS[self.kind])
        self.unit_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.unit_label.setFixedWidth(self.unit_label.fontMetrics().horizontalAdvance(WIDEST_UNIT))

        self.fix_checkbox = QCheckBox()
        self.fix_checkbox.setToolTip('Fix value during fitting')

        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.fix_checkbox)
        layout.addWidget(self.value_label)
        layout.addWidget(self.unit_label)
        self.update_label()

    def raw_from_working(self, working):
        span = self.hi - self.lo
        frac = 0.0 if span == 0 else (working - self.lo) / span
        return int(round(frac * SLIDER_STEPS))

    def working_from_raw(self, raw):
        return self.lo + (self.hi - self.lo) * raw / SLIDER_STEPS

    def decode(self, working):
        if self.kind == 'X':
            return working * np.pi / 180
        if self.kind == 'phi':
            # signed-log ("symlog") transform: 10**|w|-1 is continuous
            # through w=0 (decodes to phi=0 exactly there, unlike a plain
            # 10**working which can never reach 0) and is numerically
            # indistinguishable from a plain power-of-ten once |phi| >> 1.
            return np.sign(working) * (10 ** abs(working) - 1)
        if self.kind == 'dphi':
            # same 10**w-1 trick as phi (unsigned here since dphi >= 0):
            # continuous through w=0, decoding to dphi=0 exactly there
            # instead of a plain 10**working's floor of 1.
            return 10 ** working - 1
        if self.kind == 'p':
            return working / 100
        if self.kind == 'nu0':
            return 10 ** working
        return working  # scale, alpha, eps

    def encode(self, typed):
        """Inverse of decode: turn a number typed into the value field
        (in the same units/domain shown there -- degrees for X, percent
        for p, the physical value itself for phi/dphi/nu0, unitless for
        scale/alpha/eps) back into the slider's "working" domain."""
        if self.kind in ('X', 'p'):
            return typed  # already the working-domain value (deg / %)
        if self.kind == 'phi':
            if typed == 0:
                return 0.0
            return np.sign(typed) * np.log10(abs(typed) + 1)
        if self.kind == 'dphi':
            if typed <= 0:
                return 0.0
            return np.log10(typed + 1)
        if self.kind == 'nu0':
            return np.log10(typed) if typed > 0 else self.lo
        return typed  # scale, alpha, eps

    def value(self):
        """Physical parameter value, in the units the model function expects."""
        return self.decode(self.working_from_raw(self.slider.value()))

    def is_fixed(self):
        """Whether fix_checkbox is checked -- Fitting holds this parameter
        at its current value() instead of optimizing it; see run_fit."""
        return self.fix_checkbox.isChecked()

    def bounds(self):
        """(lo, hi) physical bounds this slider can actually reach -- may
        differ from the model's own registered spec.bounds for phi/dphi
        (which use a fixed symlog range spanning every model, not a
        per-model prior; see PHI_LOG_RANGE/DPHI_LOG_RANGE). This is what
        Fitting uses as the fit bounds, so its param_init (the slider's
        current value) is always feasible."""
        lo, hi = self.decode(self.lo), self.decode(self.hi)
        return (lo, hi) if lo <= hi else (hi, lo)

    def set_value(self, phys):
        """Move the slider to `phys` -- a physical value in the same units
        `value()` returns (e.g. loading a saved model's parameters). Goes
        through `encode`, so it's clamped to the slider's own bounds the
        same way typing a value into the box is."""
        if self.kind == 'X':
            typed = phys * 180 / np.pi
        elif self.kind == 'p':
            typed = phys * 100
        else:  # phi/dphi/scale/alpha/eps/nu0 already share value()'s domain
            typed = phys
        self.slider.setValue(self.raw_from_working(self.encode(typed)))
        self.update_label()

    def update_label(self):
        working = self.working_from_raw(self.slider.value())
        phys = self.decode(working)
        if self.kind == 'X':
            text = f'{working:.1f}'
        elif self.kind == 'p':
            text = f'{working:.2f}'
        elif self.kind in ('phi', 'dphi', 'nu0'):
            text = f'{phys:.2e}'
        else:
            text = f'{phys:.2f}'
        self.value_label.setText(text)

    def on_slider_changed(self, _raw):
        self.update_label()
        self.valueChanged.emit()

    def on_value_edited(self):
        """User typed a number into the value field and pressed Enter (or
        clicked away). Parse it, move the slider to match -- which reformats
        the text to the canonical display and fires valueChanged/replot the
        same way dragging the slider would -- so out-of-range input is
        naturally clamped to the slider's own bounds."""
        match = NUMBER_RE.search(self.value_label.text())
        if match is None:
            self.update_label()  # invalid input -- just revert the display
            return
        typed = float(match.group())
        working = self.encode(typed)
        self.slider.setValue(self.raw_from_working(working))
        # if the clamped/re-encoded slider position round-trips to the same
        # raw value the slider already had, valueChanged won't fire on its
        # own -- refresh the text either way so it's never left as raw input.
        self.update_label()


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


class StokesPlot(FigureCanvas):
    """Stokes I (left axis) and Q, U (right axis, twin) vs frequency nu [GHz]
    -- the model's own trailing spectral params (alpha for a
    single component, eps/alpha1/alpha2 for two) double as a real
    (normalized, I_0=1) Stokes I(nu) model; Q(nu), U(nu) are then
    just I(nu) scaled by the same fractional-polarization complex number
    p*e^(2i*EVPA) already used for the p/EVPA plot."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax_I = self.fig.add_subplot(111)
        self.ax_QU = self.ax_I.twinx()
        apply_fixed_margins(self.fig, self)

        self.ax_I.set_xlabel(r'$\nu$ [ GHz ]')
        self.ax_I.set_ylabel('I [ normalized ]')
        self.ax_I.tick_params(axis='y')
        self.ax_I.grid(True)
        self.ax_QU.set_ylabel('Q, U [ normalized ]', rotation=270, labelpad=15)

        self.line_I, = self.ax_I.plot([], [], color='black', label='I')
        self.line_Q, = self.ax_QU.plot([], [], color='green', label='Q', linestyle='dashed')
        self.line_U, = self.ax_QU.plot([], [], color='orange', label='U', linestyle='dashed')
        lines = [self.line_I, self.line_Q, self.line_U]
        self.ax_I.legend(lines, [l.get_label() for l in lines], loc='upper right')

        self.xscale = None
        self.ref_data = None  # (nu_ghz, I, I_err, Q, Q_err, U, U_err) or None
        self.ref_artists = []
        # See ModelPlot's own meas_bands/meas_artists -- same pattern,
        # keyed by nu/I/Q/U instead of w2/p/evpa.
        self.meas_bands = None
        self.meas_artists = []

        # See ModelPlot.set_posterior_samples -- same pattern here.
        self.posterior_samples = None
        self.posterior_model = None
        self.sample_lines_I = []
        self.sample_lines_Q = []
        self.sample_lines_U = []

    def set_posterior_samples(self, samples, model_func):
        for ln in self.sample_lines_I + self.sample_lines_Q + self.sample_lines_U:
            ln.remove()
        self.sample_lines_I, self.sample_lines_Q, self.sample_lines_U = [], [], []
        self.posterior_samples = samples
        self.posterior_model = model_func
        if samples is not None:
            for _ in range(len(samples)):
                ln_I, = self.ax_I.plot([], [], color='black', alpha=0.08, lw=0.6, zorder=1)
                ln_Q, = self.ax_QU.plot([], [], color='green', alpha=0.08, lw=0.6, zorder=1)
                ln_U, = self.ax_QU.plot([], [], color='orange', alpha=0.08, lw=0.6, zorder=1)
                self.sample_lines_I.append(ln_I)
                self.sample_lines_Q.append(ln_Q)
                self.sample_lines_U.append(ln_U)
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
        for artist in self.ref_artists:
            artist.remove()
        self.ref_artists = []
        if self.ref_data is not None:
            nu, I, I_err, Q, Q_err, U, U_err = self.ref_data
            self.ref_artists.append(self.ax_I.errorbar(
                nu, I, yerr=I_err, fmt='o', ms=4, color='black', ecolor='0.5', capsize=2, zorder=5))
            self.ref_artists.append(self.ax_QU.errorbar(
                nu, Q, yerr=Q_err, fmt='s', ms=4, color='darkgreen', ecolor='0.5', capsize=2, zorder=5))
            self.ref_artists.append(self.ax_QU.errorbar(
                nu, U, yerr=U_err, fmt='^', ms=4, color='darkorange', ecolor='0.5', capsize=2, zorder=5))

        for artist in self.meas_artists:
            artist.remove()
        self.meas_artists = []
        if self.meas_bands:
            for band in self.meas_bands:
                c = band['color']
                self.meas_artists.append(self.ax_I.errorbar(
                    band['nu'], band['I'], yerr=band['I_err'], fmt='o', ms=4, mew=0.5, mec='k',
                    color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))
                self.meas_artists.append(self.ax_QU.errorbar(
                    band['nu'], band['Q'], yerr=band['Q_err'], fmt='s', ms=4, mew=0.5, mec='k',
                    color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))
                self.meas_artists.append(self.ax_QU.errorbar(
                    band['nu'], band['U'], yerr=band['U_err'], fmt='^', ms=4, mew=0.5, mec='k',
                    color=c, ecolor=c, alpha=0.85, capsize=2, zorder=6))

    def update_plot(self, wl_ext, model_func, n_components, pars, log_xscale=False, nu_min=None):
        I = stokes_I(wl_ext, n_components, pars, nu_min=nu_min)
        Q, U = stokes_QU(wl_ext, model_func, n_components, pars, nu_min=nu_min)
        nu_ghz = C / wl_ext / 1e9

        order = np.argsort(nu_ghz)
        nu_s, I_s, Q_s, U_s = nu_ghz[order], I[order], Q[order], U[order]

        xscale = 'log' if log_xscale else 'linear'
        if xscale != self.xscale:
            self.ax_I.set_xscale(xscale)
            self.xscale = xscale

        self.line_I.set_data(nu_s, I_s)
        self.line_Q.set_data(nu_s, Q_s)
        self.line_U.set_data(nu_s, U_s)

        show_samples = self.posterior_samples is not None and self.posterior_model is model_func
        sample_lines = zip(self.sample_lines_I, self.sample_lines_Q, self.sample_lines_U)
        for i, (ln_I, ln_Q, ln_U) in enumerate(sample_lines):
            if show_samples:
                s_pars = self.posterior_samples[i]
                s_I = stokes_I(wl_ext, n_components, s_pars, nu_min=nu_min)[order]
                s_Q, s_U = (a[order] for a in stokes_QU(wl_ext, model_func, n_components, s_pars, nu_min=nu_min))
                ln_I.set_data(nu_s, s_I)
                ln_Q.set_data(nu_s, s_Q)
                ln_U.set_data(nu_s, s_U)
            else:
                ln_I.set_data([], [])
                ln_Q.set_data([], [])
                ln_U.set_data([], [])

        xlo, xhi = nu_s.min() * 0.9, nu_s.max() * 1.05
        i_max = max(I_s.max(), 1e-6)
        qu_max = max(np.abs(Q_s).max(), np.abs(U_s).max(), 1e-6)
        # Widen the autoscale to also cover any reference data loaded via
        # the Load Data... button, so it isn't silently clipped out of view.
        if self.ref_data is not None:
            ref_nu, ref_I, _, ref_Q, _, ref_U, _ = self.ref_data
            xlo, xhi = min(xlo, ref_nu.min() * 0.9), max(xhi, ref_nu.max() * 1.05)
            i_max = max(i_max, ref_I.max())
            qu_max = max(qu_max, np.abs(ref_Q).max(), np.abs(ref_U).max())
        if self.meas_bands:
            for band in self.meas_bands:
                xlo = min(xlo, band['nu'].min() * 0.9)
                xhi = max(xhi, band['nu'].max() * 1.05)
                i_max = max(i_max, (band['I'] + band['I_err']).max())
                qu_max = max(qu_max, (np.abs(band['Q']) + band['Q_err']).max(),
                              (np.abs(band['U']) + band['U_err']).max())

        self.ax_I.set_xlim(xlo, xhi)
        self.ax_I.set_ylim(0, 1.3 * i_max)
        self.ax_QU.set_ylim(-1.3 * qu_max, 1.3 * qu_max)
        self.draw_reference()
        self.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        apply_fixed_margins(self.fig, self)


class MainWindow(QMainWindow, SamplingMixin, MeasurementsMixin):
    NO_FIT_TEXT = 'Fit some data to get statistics'

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Polvista: Polarization Model Visualizer')
        #self.resize(1800, 1000) # size of the window
        self.resize(1500, 700) # size of the window

        self.sliders = []
        self.fit_data = None  # (wl, q, q_err, u, u_err, freq, I) from the Load Data... button, or None
        self.data_nu_min = None  # [MHz] loaded data's own nu_min, set by Fitting -- see run_fit

        # MultiNest (Sampling tab) result state -- see apply_posterior_result,
        # build_corner_tab, run_multinest_fit, load_samples_action.
        self.mn_worker = None
        self.mn_model = None       # model func the most recent MultiNest run/load was for
        self.posterior_samples = None
        self.posterior_model = None
        self.corner_tab = None

        # Every QMainWindow needs one "central widget"
        # QHBoxLayout arranges its children left-to-right, 
        # so this splits the window into the left
        # control panel and the right plot area added further down.
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── Left: model choice, wavelength range, parameter sliders ──────
        # QVBoxLayout stacks its children top-to-bottom -- this is the
        # left-hand control panel's layout.
        left = QWidget()
        left.setMaximumWidth(440)
        left_layout = QVBoxLayout(left)

        # set wavelength/frequency range
        # QGroupBox draws a titled border/frame around its contents --
        # used throughout this panel to visually group related controls.
        wl_box = QGroupBox('Wavelength range [mm]')
        wl_box_layout = QVBoxLayout(wl_box)

        self.wl_preset = QComboBox()
        for name, lo_mm, hi_mm in WAVELENGTH_PRESETS:
            self.wl_preset.addItem(name, (lo_mm, hi_mm))
        self.wl_preset.currentIndexChanged.connect(self.apply_wl_preset)
        wl_box_layout.addWidget(self.wl_preset)

        # QHBoxLayout here lays the min/max/n spin boxes out side by side
        # within the vertically-stacked wl_box.
        wl_row = QHBoxLayout()
        self.wl_min = QDoubleSpinBox()
        self.wl_min.setRange(0.01, 1000.0)
        self.wl_min.setDecimals(3)
        self.wl_min.setValue(0.75)
        self.wl_max = QDoubleSpinBox()
        self.wl_max.setRange(0.02, 2000.0)
        self.wl_max.setDecimals(3)
        self.wl_max.setValue(3.7)
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 2000)
        self.n_points.setValue(300)
        # Redraw the plots any time one of these three number boxes changes.
        for w in (self.wl_min, self.wl_max, self.n_points):
            w.valueChanged.connect(self.update_plot)
        wl_row.addWidget(QLabel('min'))
        wl_row.addWidget(self.wl_min)
        wl_row.addWidget(QLabel('max'))
        wl_row.addWidget(self.wl_max)
        wl_row.addWidget(QLabel('n'))
        wl_row.addWidget(self.n_points)
        wl_box_layout.addLayout(wl_row)

        self.log_xscale = QCheckBox('log x-scale')
        self.log_xscale.setChecked(True)
        self.log_xscale.stateChanged.connect(self.update_plot)
        wl_box_layout.addWidget(self.log_xscale)

        left_layout.addWidget(wl_box)

        left_layout.addWidget(QLabel('<b>Model</b>'))
        # QComboBox is the dropdown; addItem's 2nd arg is arbitrary data
        # (here the model function) retrievable later via currentData().
        self.model_combo = QComboBox()
        for func, spec in MODELS.items():
            self.model_combo.addItem(spec.label, func)
        # Qt's signal/slot mechanism: connect() wires the combo box's
        # "selection changed" signal to a method that runs whenever the
        # user picks a different model.
        self.model_combo.currentIndexChanged.connect(self.rebuild_sliders)
        left_layout.addWidget(self.model_combo)

        # spectral weighting parameters -- more relevant for two-component models
        self.spectral_box = QGroupBox()
        self.spectral_layout = QVBoxLayout(self.spectral_box)

        # Dropdown selecting the source function S'(nu) used to weight each
        # component's spectral contribution (see models.set_spectral_shape),
        # placed in the box's own top-right corner, after its "Spectrum"
        # title -- QGroupBox itself only supports a plain string title, so
        # the title is instead a plain (non-bold) QLabel in this custom
        # header row.
        spectral_header = QHBoxLayout()
        spectral_header.addWidget(QLabel('Spectrum'))
        spectral_header.addStretch(1)
        self.shape_combo = QComboBox()
        for shape_label, shape_key in SPECTRAL_SHAPES:
            self.shape_combo.addItem(shape_label, shape_key)
        self.shape_combo.currentIndexChanged.connect(self.on_spectral_shape_changed)
        spectral_header.addWidget(self.shape_combo)
        self.spectral_layout.addLayout(spectral_header)

        # Turnover-frequency nu_0 slider(s) -- log-scale, like phi/dphi,
        # since a turnover can plausibly sit many decades from the plotted
        # band -- only shown once 'SSA' is selected (see sync_nu0_ui). A
        # two-component model gets an independent nu_0 per component
        # (nu0_slider_2 only shown then); a single-component model only
        # ever needs the first. Not placed into spectral_layout here --
        # rebuild_sliders inserts nu0_container at the right spot among
        # that model's own eps/alpha sliders (epsilon, then nu_0, then
        # alpha(s); see its docstring).
        self.nu0_container = QWidget()
        nu0_container_layout = QVBoxLayout(self.nu0_container)
        nu0_container_layout.setContentsMargins(0, 0, 0, 0)
        nu0_lo, nu0_hi = self.nu0_bounds_ghz()
        default_nu0 = self.default_nu0_ghz()
        self.nu0_slider_1 = self.build_nu0_slider(
            nu0_lo, nu0_hi, default_nu0, r'$\nu_0$', 'SSA turnover frequency.')
        self.nu0_slider_2 = self.build_nu0_slider(
            nu0_lo, nu0_hi, default_nu0, r'$\nu_{0,2}$', "Component 2's SSA turnover frequency.")
        nu0_container_layout.addWidget(self.nu0_slider_1)
        nu0_container_layout.addWidget(self.nu0_slider_2)
        self.nu0_container.setVisible(False)

        self.spectral_layout.addStretch(1)

        # QScrollArea adds scrollbars around a widget that may not fit its
        # allotted space -- here it lets the (variable-length) list of
        # sliders in spectral_box scroll instead of resizing the window.
        self.spectral_scroll = QScrollArea()
        self.spectral_scroll.setWidgetResizable(True)
        self.spectral_scroll.setWidget(self.spectral_box)
        self.spectral_scroll.setMinimumHeight(140)  # fits its max of 3 rows
        self.spectral_scroll.setVisible(False)

        # the polarization-parameter sliders, in their own titled group box
        # (like spectral_box above) and scrollable area nested inside
        # param_box below -- separately from spectral_scroll, so each
        # scrolls independently
        self.pol_box = QGroupBox('Polarization')
        self.param_layout = QVBoxLayout(self.pol_box)
        self.param_layout.addStretch(1)

        pol_scroll = QScrollArea()
        pol_scroll.setWidgetResizable(True)
        pol_scroll.setWidget(self.pol_box)
        pol_scroll.setMinimumHeight(290)  # fits its max of 8 rows (2-component models)

        # param_box is just the outer, per-model container -- its title is
        # set to the current model's name in rebuild_sliders.
        self.param_box = QGroupBox()
        param_box_layout = QVBoxLayout(self.param_box)
        param_box_layout.addWidget(self.spectral_scroll)
        param_box_layout.addWidget(pol_scroll)

        # The parameter sliders live under a "Visualization" tab; a
        # "Sampling" tab (MultiNest fit options) is only added when
        # pymultinest is actually installed -- see HAS_PYMULTINEST.
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self.param_box, 'Visualization')
        self.build_measurements_tab()
        if HAS_PYMULTINEST:
            self.build_sampling_tab()  # greyed out until data is loaded, see load_data_action
        left_layout.addWidget(self.left_tabs)

        # Reset Parameters on the left, Load/Clear Data grouped together in
        # the middle, Fit! on the right -- the two addStretch(1) spacers
        # split the row into those three clusters (used to be the
        # File/Import menu's Reset Parameters/Clear Data/Load Data...
        # entries, see build_menu).
        fit_row = QHBoxLayout()
        self.reset_button = QPushButton('Reset model')
        self.reset_button.clicked.connect(self.reset_parameters)
        fit_row.addWidget(self.reset_button)

        fit_row.addStretch(1)

        self.load_data_button = QPushButton('Load data')
        self.load_data_button.clicked.connect(self.load_data_action)
        fit_row.addWidget(self.load_data_button)
        self.clear_data_button = QPushButton('Clear data')
        self.clear_data_button.clicked.connect(self.clear_data)
        fit_row.addWidget(self.clear_data_button)

        fit_row.addStretch(1)

        self.fit_button = QPushButton('Fit!')
        self.fit_button.setEnabled(False)  # only enabled once data is loaded, see load_data_action
        self.fit_button.clicked.connect(self.run_fit)
        fit_row.addWidget(self.fit_button)
        left_layout.addLayout(fit_row)

        # fit statistics
        results_box = QGroupBox('Results')
        results_layout = QVBoxLayout(results_box)

        # MultiNest sampling has no well-defined "percent complete" ahead of
        # time (it runs to an evidence-convergence criterion, not a fixed
        # iteration budget -- see fitting.multinest_fit's docstring), so
        # this is an indeterminate/"busy" bar (setRange(0,0)) rather than a
        # literal progress meter; sampling_progress_label carries the actual
        # live numbers (sample count, ln Z) fed by MultiNestWorker.progress.
        # Built unconditionally (harmless without pymultinest, just never
        # shown) -- only run_multinest_fit ever makes them visible, and
        # that's only reachable when HAS_PYMULTINEST is True (see run_fit).
        self.sampling_progress = QProgressBar()
        self.sampling_progress.setRange(0, 0)
        self.sampling_progress.setTextVisible(False)
        self.sampling_progress.setStyleSheet(
            'QProgressBar { border: 1px solid #999; border-radius: 3px; background: transparent; }'
            'QProgressBar::chunk { background-color: #2ecc71; }')
        self.sampling_progress.setVisible(False)
        results_layout.addWidget(self.sampling_progress)
        self.sampling_progress_label = QLabel('')
        self.sampling_progress_label.setStyleSheet('font-family: monospace;')
        self.sampling_progress_label.setWordWrap(True)
        self.sampling_progress_label.setVisible(False)
        results_layout.addWidget(self.sampling_progress_label)

        self.results_label = QLabel(self.NO_FIT_TEXT)
        self.results_label.setStyleSheet('font-family: monospace;')
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)
        left_layout.addWidget(results_box)

        left_layout.addStretch(1)

        root.addWidget(left)

        # ── Right: equation card, toolbar, matplotlib canvas ─────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.eq_scroll = QScrollArea()
        self.eq_scroll.setFixedHeight(130)
        self.eq_scroll.setWidgetResizable(False)
        self.eq_scroll.setAlignment(Qt.AlignCenter)
        # The equation is always re-fit (see refit_equation) to the
        # viewport size, so it should never actually need to scroll --
        # scrollbars are disabled outright rather than "as needed" so a
        # transient one-frame overflow during a resize can't flash one.
        self.eq_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.eq_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.eq_scroll.setStyleSheet('QScrollArea { background-color: white; border: 1px solid #999; }')
        self.eq_label = QLabel()
        self.eq_label.setStyleSheet('background-color: white; padding: 6px;')
        self.eq_scroll.setWidget(self.eq_label)
        right_layout.addWidget(self.eq_scroll)

        # ModelPlot/StokesPlot are matplotlib FigureCanvas subclasses, so
        # they behave like any other QWidget; NavigationToolbar is
        # matplotlib's standard pan/zoom/save toolbar wired to that canvas.
        self.canvas = ModelPlot(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        p_evpa_tab = QWidget()
        p_evpa_layout = QVBoxLayout(p_evpa_tab)
        p_evpa_layout.setContentsMargins(0, 0, 0, 0)
        p_evpa_layout.addWidget(self.toolbar)
        p_evpa_layout.addWidget(self.canvas, stretch=1)

        self.stokes_canvas = StokesPlot(self)
        self.stokes_toolbar = NavigationToolbar(self.stokes_canvas, self)
        stokes_tab = QWidget()
        stokes_layout = QVBoxLayout(stokes_tab)
        stokes_layout.setContentsMargins(0, 0, 0, 0)
        stokes_layout.addWidget(self.stokes_toolbar)
        stokes_layout.addWidget(self.stokes_canvas, stretch=1)

        # QTabWidget switches between the plot pages via clickable tabs. A
        # third "Corner plot" tab is added/removed dynamically by
        # build_corner_tab/remove_corner_tab once a MultiNest fit (or
        # Load samples) has something to show there.
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(p_evpa_tab, 'p / EVPA  vs  λ²')
        self.plot_tabs.addTab(stokes_tab, 'I, Q, U  vs  ν')
        right_layout.addWidget(self.plot_tabs, stretch=1)
        root.addWidget(right, stretch=1)

        self.build_menu()
        self.rebuild_sliders()

    def rebuild_sliders(self, *_):
        for sl in self.sliders:
            sl.setParent(None)
        self.sliders = []

        func = self.model_combo.currentData()
        spec = MODELS[func]
        lo_bounds, hi_bounds = spec.bounds

        # remove the trailing stretch, re-add sliders, then put stretch back
        self.param_layout.takeAt(self.param_layout.count() - 1)
        self.spectral_layout.takeAt(self.spectral_layout.count() - 1)
        # nu0_container isn't one of self.sliders (see _build_ui) so it
        # survives the setParent(None) loop above untouched -- pull it out
        # of spectral_layout too, so it can be re-inserted below at the
        # spot matching this model's own spectral params (epsilon, then
        # nu_0, then alpha(s) -- see the loop's comment).
        self.spectral_layout.removeWidget(self.nu0_container)

        has_spectral = False
        nu0_placed = False
        for i, param in enumerate(spec.params):
            sl = ParamSlider(param, lo_bounds[i], hi_bounds[i])
            sl.valueChanged.connect(self.update_plot)
            # self.sliders keeps the *original* spec.params order regardless of
            # which box a slider is displayed in -- update_plot() reads this
            # list positionally, matching the model function's param order.
            self.sliders.append(sl)
            if param.kind in ('alpha', 'eps'):
                # Display order: epsilon (2-component models only), then
                # nu_0 (see nu0_container -- inserted right before the
                # first alpha slider, i.e. right after epsilon when there
                # is one), then alpha(s).
                if param.kind == 'alpha' and not nu0_placed:
                    self.spectral_layout.addWidget(self.nu0_container)
                    nu0_placed = True
                self.spectral_layout.addWidget(sl)
                has_spectral = True
            else:
                self.param_layout.addWidget(sl)
        if not nu0_placed:
            self.spectral_layout.addWidget(self.nu0_container)

        self.param_layout.addStretch(1)
        self.spectral_layout.addStretch(1)
        self.param_box.setTitle(spec.title)
        self.spectral_scroll.setVisible(has_spectral)

        if hasattr(self, 'results_label'):
            self.results_label.setText(self.NO_FIT_TEXT)

        self.rebuild_sampling_bounds()

        # A model switch can change the component count (single- vs
        # two-component), which the SSA turnover-frequency row(s) need to
        # track -- see sync_nu0_ui.
        self.sync_nu0_ui()

        self.refit_equation()

        self.update_plot()

    def build_nu0_slider(self, lo_ghz, hi_ghz, default_ghz, latex, description):
        """One nu_0 ParamSlider (log-scale, like a dphi slider) for the
        Spectrum box's SSA turnover-frequency controls -- see _build_ui,
        which builds one or two of these depending on the current model's
        component count (see sync_nu0_ui). `default_ghz` is applied (and
        its own valueChanged wired up) only after construction, so building
        the slider itself can't prematurely fire the app-level handler.

        Starts *fixed* (checkbox checked) -- at its default (~1% of the
        band's own nu_min, deep in the optically-thin regime), a fit
        should start out assuming there's no turnover in view and only fit
        alpha; the user frees this slider (unchecks it) to let a fit
        jointly solve for nu_0 too (see MainWindow.fit_spectrum_lsq)."""
        sl = ParamSlider(Param('nu0', latex, 'nu0', description), lo_ghz, hi_ghz)
        sl.set_value(default_ghz)
        sl.fix_checkbox.setChecked(True)
        sl.valueChanged.connect(self.on_spectral_shape_changed)
        return sl

    def nu0_bounds_ghz(self):
        """(lo, hi) [GHz] bounds for the SSA turnover-frequency sliders:
        NU0_BOUNDS_MULT x the current wavelength range's own (nu_min,
        nu_max), wide enough to place a turnover well outside the plotted
        band in either direction. Computed once, from the band in effect
        when the Spectrum box is built -- its *default* bounds, not
        something that tracks later changes to the wavelength range (like
        PHI_LOG_RANGE/DPHI_LOG_RANGE, just not shared across every model)."""
        lo_mm = min(self.wl_min.value(), self.wl_max.value())
        hi_mm = max(self.wl_min.value(), self.wl_max.value())
        nu_min_ghz = C / (hi_mm * 1e-3) / 1e9
        nu_max_ghz = C / (lo_mm * 1e-3) / 1e9
        lo_mult, hi_mult = NU0_BOUNDS_MULT
        return lo_mult * nu_min_ghz, hi_mult * nu_max_ghz

    def default_nu0_ghz(self):
        """1% of the current wavelength range's own minimum frequency
        [GHz] -- default value for the SSA turnover-frequency sliders, both
        their initial one (see _build_ui) and the one Reset model restores
        (see reset_parameters)."""
        hi_mm = max(self.wl_min.value(), self.wl_max.value())  # longest wavelength -> nu_min
        nu_min_ghz = C / (hi_mm * 1e-3) / 1e9
        return 0.01 * nu_min_ghz

    def sync_nu0_ui(self):
        """Keep the Spectrum box's nu_0 slider(s) matched to the shape
        dropdown's current selection and the current model's component
        count, and push their values into models' global spectral-shape
        state (see models.set_spectral_shape) -- called whenever either the
        shape dropdown, a nu_0 slider, or the model itself changes."""
        shape = self.shape_combo.currentData()
        is_ssa = shape == 'ssa'
        n_components = MODELS[self.model_combo.currentData()].n_components
        self.nu0_container.setVisible(is_ssa)
        self.nu0_slider_2.setVisible(is_ssa and n_components == 2)
        self.nu0_slider_1.name_label.setPixmap(
            latex_pixmap(r'$\nu_{0,1}$' if n_components == 2 else r'$\nu_0$'))
        nu0 = self.nu0_slider_1.value() * 1e3 if is_ssa else None  # GHz -> MHz
        nu0_2 = self.nu0_slider_2.value() * 1e3 if is_ssa else None
        set_spectral_shape(shape, nu0=nu0, nu0_2=nu0_2)

    def on_spectral_shape_changed(self, *_):
        """Slot for the Spectrum box's shape dropdown and nu_0 spin
        box(es): re-sync models' spectral-shape state and redraw the
        equation card and plots to reflect it."""
        self.sync_nu0_ui()
        self.refit_equation()
        self.update_plot()

    def refit_equation(self):
        spec = MODELS[self.model_combo.currentData()]
        shape = self.shape_combo.currentData()
        eq = full_equation(spec, shape)
        viewport = self.eq_scroll.viewport().size()
        # Leave room for the label's own padding (6px/side, see its
        # stylesheet) plus a little breathing room.
        max_w = max(viewport.width() - 20, 50)
        max_h = max(viewport.height() - 20, 20)
        pixmap = fit_equation_pixmap(eq, max_w, max_h)
        self.eq_label.setPixmap(pixmap)
        self.eq_label.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'eq_scroll'):
            self.refit_equation()

    def apply_wl_preset(self, index):
        """Fill wl_min/wl_max from the selected preset's stored (lo, hi)
        data -- this itself triggers update_plot via their valueChanged.
        Also resets the Measurements tab's Bands box to that preset's own
        defaults (see measurements.DEFAULT_BANDS_BY_PRESET)."""
        lo_mm, hi_mm = self.wl_preset.itemData(index)
        self.wl_min.setValue(lo_mm)
        self.wl_max.setValue(hi_mm)
        self.apply_default_bands(self.wl_preset.itemText(index), lo_mm, hi_mm)

    def current_state(self):
        """(func, spec, pars, wl_ext) for the currently selected model,
        sliders, and wavelength range -- shared by update_plot and the
        Export actions so they can't drift out of sync."""
        func = self.model_combo.currentData()
        spec = MODELS[func]
        pars = [sl.value() for sl in self.sliders]

        wl_min_mm = min(self.wl_min.value(), self.wl_max.value())
        wl_max_mm = max(self.wl_min.value(), self.wl_max.value())
        if wl_max_mm <= wl_min_mm:
            wl_max_mm = wl_min_mm + 0.01
        # Points evenly spaced in log-space when the plot itself is log-x --
        # linspace over a wide range leaves the low end (which is where a
        # log-x plot devotes most of its width) covered by only a handful of
        # points, producing visibly faceted/jagged curves there.
        if self.log_xscale.isChecked():
            wl_ext = np.logspace(np.log10(wl_min_mm), np.log10(wl_max_mm), self.n_points.value()) * 1e-3
        else:
            wl_ext = np.linspace(wl_min_mm, wl_max_mm, self.n_points.value()) * 1e-3
        return func, spec, pars, wl_ext

    def update_plot(self, *_):
        """Redraw both plot tabs from the current model/sliders/wavelength
        range. Connected as the slot for many widgets' signals above, which
        pass along their own changed value -- `*_` just discards it since
        current_state() re-reads everything from scratch anyway."""
        func, spec, pars, wl_ext = self.current_state()
        log_xscale = self.log_xscale.isChecked()
        self.canvas.update_plot(wl_ext, func, pars, log_xscale=log_xscale)
        self.stokes_canvas.update_plot(wl_ext, func, spec.n_components, pars,
                                        log_xscale=log_xscale, nu_min=self.data_nu_min)

    # ── Menu bar ───────────────────────────────────────────────────────────
    def build_menu(self):
        # self.menuBar() gives the QMainWindow's built-in top menu bar;
        # addMenu creates one drop-down ("&File" -- the & marks Alt+F as
        # its keyboard shortcut). Each QAction is one clickable menu entry,
        # wired via triggered.connect to the method that runs it.
        menubar = self.menuBar()

        # Reset Parameters/Clear Data/Load Data used to live under a
        # dedicated File/Import menu entry each -- they're now the button
        # row beside Fit! (see fit_row in _build_ui) instead.
        import_menu = menubar.addMenu('&Import')
        load_data_action = QAction('Load Data...', self)
        load_data_action.triggered.connect(self.load_data_action)
        import_menu.addAction(load_data_action)
        load_model_action = QAction('Load Model...', self)
        load_model_action.triggered.connect(self.load_model_action)
        import_menu.addAction(load_model_action)
        

        export_menu = menubar.addMenu('&Export')
        save_model_action = QAction('Save Model...', self)
        save_model_action.triggered.connect(self.save_model_action)
        export_menu.addAction(save_model_action)
        save_spectra_action = QAction('Save Spectra...', self)
        save_spectra_action.triggered.connect(self.save_spectra_action)
        export_menu.addAction(save_spectra_action)
        save_equation_action = QAction('Save Equation...', self)
        save_equation_action.triggered.connect(self.save_equation_action)
        export_menu.addAction(save_equation_action)

        help_menu = menubar.addMenu('&Help')
        complex_pol_action = QAction('Complex polarization', self)
        complex_pol_action.triggered.connect(self.show_complex_polarization_help)
        help_menu.addAction(complex_pol_action)
        qufitting_action = QAction('QU-fitting', self)
        qufitting_action.triggered.connect(self.show_qufitting_help)
        help_menu.addAction(qufitting_action)
        about_action = QAction('About Polvista', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def reset_parameters(self):
        """ParamSlider always starts at its kind's default, so rebuilding
        the current model's sliders from scratch is a reset to defaults.
        Also drops any data-anchored Stokes I/Q/U normalization from a
        previous fit (see run_fit) back to the plot range's own nu_min --
        a reset parameters should look like a fresh model, not a fit.
        Also drops any MultiNest posterior-sample overlay and closes the
        Corner plot tab, if either is showing -- both are tied to a
        specific fit's result, which a reset discards.

        The SSA turnover-frequency sliders aren't part of self.sliders (see
        _build_ui), so rebuild_sliders alone wouldn't reset them -- move
        them back to their own default value *and* fixed state explicitly
        (see build_nu0_slider)."""
        self.data_nu_min = None
        self.clear_posterior_samples()
        self.remove_corner_tab()
        default_nu0 = self.default_nu0_ghz()
        self.nu0_slider_1.set_value(default_nu0)
        self.nu0_slider_2.set_value(default_nu0)
        self.nu0_slider_1.fix_checkbox.setChecked(True)
        self.nu0_slider_2.fix_checkbox.setChecked(True)
        self.rebuild_sliders()

    def clear_data(self):
        """Erase any reference data points loaded via the Load Data... button.
        Also closes the Corner plot tab (it was built against this data),
        but leaves any posterior-sample overlay in place -- only Reset
        model clears that, see reset_parameters."""
        self.canvas.clear_reference_data()
        self.stokes_canvas.clear_reference_data()
        self.fit_data = None
        self.data_nu_min = None
        self.remove_corner_tab()
        self.fit_button.setEnabled(False)
        self.set_sampling_tab_enabled(False)
        self.clear_measurement_points()
        self.results_label.setText(self.NO_FIT_TEXT)
        self.update_plot()

    def run_fit(self):
        """Dispatches Fit! to the Bayesian (MultiNest) path when the
        Sampling tab is the active one (and pymultinest is installed), or
        the ordinary least-squares path otherwise. `self.sampling_tab`
        only exists when HAS_PYMULTINEST is True, so the short-circuit
        `and` below matters -- it's never evaluated otherwise."""
        if HAS_PYMULTINEST and self.left_tabs.currentWidget() is self.sampling_tab:
            self.run_multinest_fit()
        else:
            self.run_lsq_fit()

    def fit_spectrum_lsq(self):
        """Least-squares fit the current model's free parameters to the
        loaded q, u data, using the sliders' current values as the initial
        guess and their bounds as the fit bounds. Shared by run_lsq_fit
        and run_multinest_fit (see there): the latter uses this as a
        pre-fit to determine alpha/epsilon, which QU-only data can't
        otherwise constrain.

        Only a QU fit: epsilon defaults to fixed at
        FIT_FIXED_EPSILON (q, u data alone can't constrain it) and alpha
        defaults to a direct regression against the loaded I(nu) data --
        estimate_alpha under 'powerlaw', or a joint (nu_0, alpha) nonlinear
        regression under 'ssa' (see estimate_ssa_shape, and
        build_nu0_slider for why nu_0 only joins that regression once its
        own slider is unchecked) -- rather than being fit alongside
        p/X/phi/dphi. For a two-component model, alpha1/alpha2 (and, under
        'ssa', nu_0,1/nu_0,2) all end up equal: there's only one real I(nu)
        dataset, so a single combined intensity regression is all QU-only
        fitting can ever be anchored to -- same reasoning FIT_FIXED_EPSILON
        already relies on.

        Any slider whose fix_checkbox is checked is additionally (or
        instead, for epsilon/alpha) held fixed at its own current value()
        rather than the above defaults or the optimizer -- letting the user
        override the automatic epsilon/alpha, or pin any other parameter
        (e.g. a known EVPA) while the rest fit around it.

        Also anchors the Stokes I/Q/U display's normalization to the loaded
        data's own nu_min (see stokes_I/stokes_QU's nu_min param), so the
        fitted curve stays aligned with the data points even if the plotted
        wavelength range is later widened or narrowed. Reset Parameters (or
        Clear Data) drops this anchor again.

        Pushes the result to the sliders itself (like the old run_fit
        always did) and returns (best_pars, stats), or None (after warning
        the user) if the fit itself raised."""
        func, spec, pars, _ = self.current_state()
        wl, q, q_err, u, u_err, freq, I = self.fit_data

        alpha_indices = spec.indices('alpha')
        if self.shape_combo.currentData() == 'ssa' and alpha_indices:
            alpha_init = self.sliders[alpha_indices[0]].value()
            nu0_init_ghz = self.nu0_slider_1.value()
            fit_nu0 = not self.nu0_slider_1.is_fixed()
            nu0_lo_ghz, nu0_hi_ghz = self.nu0_slider_1.bounds()
            alpha_est, nu0_est_ghz = estimate_ssa_shape(
                freq, I, nu0_init_ghz * 1e9, alpha_init,
                (nu0_lo_ghz * 1e9, nu0_hi_ghz * 1e9), fit_nu0)
            nu0_est_ghz = nu0_est_ghz / 1e9
            self.nu0_slider_1.set_value(nu0_est_ghz)
            self.nu0_slider_2.set_value(nu0_est_ghz)
            self.sync_nu0_ui()
        else:
            alpha_est = estimate_alpha(freq, I)

        fixed = {i: FIT_FIXED_EPSILON for i in spec.indices('eps')}
        fixed.update({i: alpha_est for i in alpha_indices})
        fixed.update({i: sl.value() for i, sl in enumerate(self.sliders) if sl.is_fixed()})
        lo = [sl.bounds()[0] for sl in self.sliders]
        hi = [sl.bounds()[1] for sl in self.sliders]

        try:
            best_pars, result = qu_fit(wl, q, q_err, u, u_err, func, pars, (lo, hi), fixed=fixed)
        except Exception as e:
            QMessageBox.warning(self, 'Fitting', f'Fit failed:\n{e}')
            return None

        self.data_nu_min = freq.min() / 1e6  # MHz, matches spectral_weights' convention
        for sl, val in zip(self.sliders, best_pars):
            sl.set_value(val)

        stats = fit_statistics(best_pars, result, wl, q, q_err, u, u_err, func)
        return best_pars, stats

    def run_lsq_fit(self):
        if self.fit_data is None:
            return
        result = self.fit_spectrum_lsq()
        if result is None:
            return
        best_pars, stats = result
        self.results_label.setText(self.format_fit_stats(stats))

    # output statistics
    def format_fit_stats(self, stats):
        aicc = f"{stats['aicc']:.4g}" if np.isfinite(stats['aicc']) else 'n/a (dof ≤ 1)'
        conv = 'yes' if stats['converged'] else 'NO -- check bounds/initial guess'
        return (
            f"χ² = {stats['chi2']:.4g}   dof = {stats['dof']}\n"
            f"χ²_red = {stats['chi2_red']:.4g}\n"
            f"ln L = {stats['loglike']:.4g}\n"
            f"AIC = {stats['aic']:.4g}   AICc = {aicc}\n"
            f"BIC = {stats['bic']:.4g}\n"
            f"converged: {conv}  ({stats['nfev']} evals)")


    def save_model_action(self):
        """Write the current model + parameter values, plus the Spectrum
        box's own state (spectral shape, and its nu_0(s) under 'ssa') --
        that state lives outside spec.params/self.sliders (see
        build_nu0_slider), so it isn't already covered by `parameters`
        below -- to a JSON file."""
        func, spec, pars, _ = self.current_state()
        # QFileDialog.getSaveFileName pops up the native "Save As" dialog;
        # it returns (chosen_path, selected_filter), or ('', ...) if the
        # user cancels -- hence the early return below.
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Model', f'{func.__name__}.json', 'JSON files (*.json)')
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        shape = self.shape_combo.currentData()
        data = {
            'model': func.__name__,
            'parameters': dict(zip(spec.param_names, pars)),
            'spectral_shape': shape,
        }
        if shape == 'ssa':
            data['nu0_ghz'] = self.nu0_slider_1.value()
            if spec.n_components == 2:
                data['nu0_2_ghz'] = self.nu0_slider_2.value()
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            # QMessageBox.warning shows a small popup dialog with an icon,
            # title, and message -- used throughout for user-facing errors.
            QMessageBox.warning(self, 'Save Model', f'Could not save model file:\n{e}')

    def save_spectra_action(self):
        """Write the current model's p/EVPA and I/Q/U spectra to a CSV."""
        func, spec, pars, wl_ext = self.current_state()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Spectra', f'{func.__name__}_spectra.csv', 'CSV files (*.csv)')
        if not path:
            return
        if not path.endswith('.csv'):
            path += '.csv'

        fit = func(wl_ext, pars)
        p_pct = pol(fit)
        evpa_deg = evpa(fit)
        # nu_min=None (the default, plot-range-derived normalization) unless
        # a fit has anchored it to the loaded data since the last Reset
        # Parameters/Clear Data -- matches whatever's currently on screen.
        I = stokes_I(wl_ext, spec.n_components, pars, nu_min=self.data_nu_min)
        Q, U = stokes_QU(wl_ext, func, spec.n_components, pars, nu_min=self.data_nu_min)
        freq_hz = C / wl_ext
        order = np.argsort(freq_hz)

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Frequency [Hz]', 'I', 'Q', 'U', 'p [%]', 'evpas [deg]'])
                for i in order:
                    writer.writerow([freq_hz[i], I[i], Q[i], U[i], p_pct[i], evpa_deg[i]])
        except OSError as e:
            QMessageBox.warning(self, 'Save Spectra', f'Could not save spectra file:\n{e}')

    def save_equation_action(self):
        """Write the current model's LaTeX equation string to a .tex file."""
        func, spec, _, _ = self.current_state()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Equation', f'{func.__name__}_equation.tex', 'TeX files (*.tex)')
        if not path:
            return
        if not path.endswith('.tex'):
            path += '.tex'
        try:
            with open(path, 'w') as f:
                f.write(full_equation(spec, self.shape_combo.currentData()) + '\n')
        except OSError as e:
            QMessageBox.warning(self, 'Save Equation', f'Could not save equation file:\n{e}')

    def load_model_action(self):
        """Read a model + parameters (+ spectral shape/nu_0) JSON file (as
        written by save_model_action) and apply it: switch the Spectrum
        box's shape dropdown and nu_0 slider(s), switch the model dropdown,
        then move each slider to the saved value.

        `spectral_shape` defaults to 'powerlaw' (and nu_0 is left alone)
        for files saved before this was tracked, so older Save Model
        output still loads the same as it always did."""
        # getOpenFileName is the read-side counterpart of getSaveFileName.
        path, _ = QFileDialog.getOpenFileName(self, 'Load Model', '', 'JSON files (*.json)')
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            func = MODELS_BY_NAME[data['model']]
            params = data['parameters']
            shape = data.get('spectral_shape', 'powerlaw')
            nu0_ghz = data.get('nu0_ghz')
            nu0_2_ghz = data.get('nu0_2_ghz', nu0_ghz)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, 'Load Model', f'Could not load model file:\n{e}')
            return

        # Set the Spectrum box's own state first -- rebuild_sliders below
        # (triggered by the model switch) re-syncs it from whatever's
        # currently selected (see sync_nu0_ui), so shape/nu_0 need to
        # already be in place before that runs.
        shape_idx = self.shape_combo.findData(shape)
        if shape_idx >= 0:
            self.shape_combo.blockSignals(True)
            self.shape_combo.setCurrentIndex(shape_idx)
            self.shape_combo.blockSignals(False)
        if nu0_ghz is not None:
            self.nu0_slider_1.set_value(nu0_ghz)
        if nu0_2_ghz is not None:
            self.nu0_slider_2.set_value(nu0_2_ghz)

        idx = self.model_combo.findData(func)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)
        self.rebuild_sliders()  # always rebuild -- setCurrentIndex above won't fire if idx is unchanged

        spec = MODELS[func]
        for sl, param in zip(self.sliders, spec.params):
            if param.name in params:
                sl.set_value(params[param.name])

    def load_data_action(self):
        """Read a Stokes I/Q/U-vs-frequency CSV, derive p/EVPA and their
        errors from it, plot it as reference points on both tabs, and stash
        it for Fitting (run_fit)."""
        path, _ = QFileDialog.getOpenFileName(self, 'Load Data', '', 'CSV files (*.csv)')
        if not path:
            return
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                freq_col = find_freq_column(fieldnames)
                i_col = find_column(fieldnames, 'i')
                q_col = find_column(fieldnames, 'q')
                u_col = find_column(fieldnames, 'u')
                if not (freq_col and i_col and q_col and u_col):
                    raise ValueError('CSV needs a Frequency column and I, Q, U columns.')
                i_err_col = find_error_column(fieldnames, 'i')
                q_err_col = find_error_column(fieldnames, 'q')
                u_err_col = find_error_column(fieldnames, 'u')
                rows = list(reader)
            if not rows:
                raise ValueError('CSV has no data rows.')

            freq = np.array([float(r[freq_col]) for r in rows]) * freq_unit_hz_multiplier(freq_col)
            I = np.array([float(r[i_col]) for r in rows])
            Q = np.array([float(r[q_col]) for r in rows])
            U = np.array([float(r[u_col]) for r in rows])
            I_err = np.array([float(r[i_err_col]) for r in rows]) if i_err_col else None
            Q_err = np.array([float(r[q_err_col]) for r in rows]) if q_err_col else None
            U_err = np.array([float(r[u_err_col]) for r in rows]) if u_err_col else None
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, 'Load Data', f'Could not load data file:\n{e}')
            return

        order = np.argsort(freq)
        freq, I, Q, U = freq[order], I[order], Q[order], U[order]
        if I_err is not None:
            I_err = I_err[order]
        if Q_err is not None:
            Q_err = Q_err[order]
        if U_err is not None:
            U_err = U_err[order]

        # p and EVPA are scale-invariant ratios of I/Q/U, so they're computed
        # from the raw values -- unaffected by the I0 normalization below.
        r2 = Q ** 2 + U ** 2
        p_pct = 100 * np.sqrt(r2) / I
        evpa_deg = np.degrees(0.5 * np.arctan2(U, Q))

        p_err_pct = evpa_err_deg = None
        if Q_err is not None or U_err is not None:
            qerr2 = Q_err ** 2 if Q_err is not None else 0.0
            uerr2 = U_err ** 2 if U_err is not None else 0.0
            sigma_r = np.sqrt(Q ** 2 * qerr2 + U ** 2 * uerr2) / np.sqrt(r2)
            sigma_p_frac = sigma_r / I
            if I_err is not None:
                sigma_p_frac = np.sqrt(sigma_p_frac ** 2 + (np.sqrt(r2) / I ** 2 * I_err) ** 2)
            p_err_pct = 100 * sigma_p_frac
            evpa_err_deg = np.degrees(0.5 * np.sqrt(Q ** 2 * uerr2 + U ** 2 * qerr2) / r2)

        # Normalize Stokes I/Q/U to I=1 at the lowest frequency, matching the
        # model curves' own I(nu_min)=1 convention (see stokes_I()). Also
        # anchor the model curves' own nu_min to this same frequency (like a
        # real Fit! does, see fit_spectrum_lsq) *immediately*, not just
        # once the user actually fits -- otherwise the plotted curve stays
        # anchored to the current wavelength range's own edge instead, and
        # visibly disagrees with these freshly-loaded, data-anchored points
        # (worse the further the plotted range's edge sits from the data's
        # own lowest frequency). Reset Parameters (or Clear Data) drops it.
        I0 = I[0]
        I_n, Q_n, U_n = I / I0, Q / I0, U / I0
        I_err_n = I_err / I0 if I_err is not None else None
        Q_err_n = Q_err / I0 if Q_err is not None else None
        U_err_n = U_err / I0 if U_err is not None else None
        self.data_nu_min = freq.min() / 1e6  # MHz, matches spectral_weights' convention

        wl = C / freq  # m
        w2 = wl ** 2 * 1e6  # lambda^2 [mm^2]
        nu_ghz = freq / 1e9

        self.canvas.set_reference_data(w2, p_pct, p_err_pct, evpa_deg, evpa_err_deg)
        self.stokes_canvas.set_reference_data(nu_ghz, I_n, I_err_n, Q_n, Q_err_n, U_n, U_err_n)

        # Fractional q=Q/I, u=U/I (per-point I, not the I0-normalized Q_n/U_n
        # above) -- the quantity the models themselves return, so it's what
        # Fitting compares against. Falls back to uniform weighting only if
        # no error columns at all were present in the CSV.
        q, u = Q / I, U / I
        if Q_err is None and U_err is None and I_err is None:
            q_err, u_err = np.ones_like(q), np.ones_like(u)
        else:
            def frac_err(val, val_err, denom, denom_err):
                val_err = val_err if val_err is not None else 0.0
                denom_err = denom_err if denom_err is not None else 0.0
                return np.sqrt((val_err / denom) ** 2 + (val * denom_err / denom ** 2) ** 2)
            q_err = frac_err(Q, Q_err, I, I_err)
            u_err = frac_err(U, U_err, I, I_err)
        # freq, I (raw, not I0-normalized) ride along for Fitting's alpha
        # estimate (see run_fit / estimate_alpha) -- computed there, not
        # here, since it should only take effect once the user clicks Fitting.
        self.fit_data = (wl, q, q_err, u, u_err, freq, I)
        self.fit_button.setEnabled(True)
        self.set_sampling_tab_enabled(True)

        self.update_plot()


    def show_complex_polarization_help(self):
        # Keep a single dialog instance around (raised on repeat clicks)
        # rather than a new one each time -- also keeps it alive, since
        # nothing else would hold a reference to a non-modal QDialog.
        if getattr(self, 'complex_pol_dialog', None) is None:
            tex_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'help', 'Complex_polarization.tex')
            self.complex_pol_dialog = TexViewerDialog(
                'Complex Polarization', tex_path, parent=self)
        self.complex_pol_dialog.show()
        self.complex_pol_dialog.raise_()
        self.complex_pol_dialog.activateWindow()


    def show_qufitting_help(self):
        if getattr(self, 'qufit_dialog', None) is None:
            tex_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'help', 'QU-fitting.tex')
            self.qufit_dialog = TexViewerDialog(
                'QU-fitting', tex_path, parent=self)
        self.qufit_dialog.show()
        self.qufit_dialog.raise_()
        self.qufit_dialog.activateWindow()


    def show_about(self):
        # QMessageBox.about is a simple info dialog (no warning icon/buttons
        # beyond OK) -- Help > About Polvista.
        box = QMessageBox(self)
        box.setWindowTitle('About Polvista')
        box.setTextFormat(Qt.RichText)
        box.setText('<b>Polvista: Polarization Model Visualizer</b><br><br>'
                    'Interactive tool to explore the Faraday effect and maybe even fit some spectropolarimetric data!<br><br>'
                    'Author: Douglas Carlos, 2026 <a href="https://github.com/medoug7/polvista">GitHub</a>')

        box.findChild(QLabel, 'qt_msgbox_label').setOpenExternalLinks(True)
        box.exec_()

    



def main():
    # Every PyQt5 app needs exactly one QApplication, created before any
    # widgets. app.exec_() starts Qt's event loop (handling clicks, redraws,
    # etc.) and blocks until the window is closed, at which point it
    # returns an exit code that sys.exit() passes back to the OS.
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
