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
from PyQt5.QtCore import Qt, pyqtSignal, QLocale, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QComboBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTabWidget, QPushButton, QProgressBar,
    QFileDialog, QMessageBox, QAction)

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from polvista.models import (
    MODELS, MODELS_BY_NAME, C, Param, stokes_I, stokes_QU, evpa, pol,
    set_spectral_shape, full_equation, build_custom_model, CUSTOM_MODEL_DEFS,
    CustomModelError, SPECTRAL_SHAPES, TEMP_BOUNDS_K, BETA_BOUNDS)
from polvista.build_model import CustomModel
from polvista.fitting import (
    qu_fit, estimate_alpha, estimate_ssa_shape, estimate_shape_2comp, fit_statistics,
    multinest_fit, load_previous_run)
from polvista.latex_stuff import latex_pixmap, fit_equation_pixmap, TexViewerDialog
from polvista.widgets import (
    ValueLineEdit, NUMBER_RE, SLIDER_STEPS, UNITS, WIDEST_UNIT, ModelPlot, StokesPlot, RMSynthPlot)
from polvista.sampling import SamplingMixin
from polvista.measurements import MeasurementsMixin
from polvista.rm_synthesis import compute_faraday_spectrum, phi_axis_half_width, FaradaySpectrumError

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
##############################################


# model_combo's "Custom model..." entry -- a distinct sentinel object (not a
# real model function) so on_model_combo_changed can tell it apart from
# MODELS_BY_NAME/MODELS' own keys and open the builder dialog instead of
# treating it as a model to plot; see open_custom_model_dialog.
CUSTOM_MODEL_SENTINEL = object()

# A QSlider's own SLIDER_STEPS resolution means almost every pixel of a drag
# fires a fresh valueChanged -- each wired (via request_update_plot, not
# update_plot directly) to a full model evaluation + matplotlib redraw, which
# for a custom model (up to a CUSTOM_Z_N_MAX-point LOS quadrature) can't
# possibly keep up with that event rate. PLOT_THROTTLE_MS caps how often
# request_update_plot actually redraws: the first event in a burst still
# redraws immediately (no perceived lag for a single discrete change, e.g. a
# spin box click), and any further events arriving before the timer elapses
# just mark the plot dirty and get coalesced into one redraw when it fires --
# so a fast drag can't queue up more model evaluations than the eye could
# ever perceive anyway. ~30 Hz is comfortably smooth to the eye while leaving
# custom models room to keep up.
PLOT_THROTTLE_MS = 33

# wl_ext's own point count (self.n_points) to use instead, only while a
# parameter slider is actively being dragged (see on_slider_drag_started/
# _ended) -- a custom model's own LOS quadrature cost scales with both its
# resolved z-grid *and* the number of wavelengths it's evaluated at, so
# capping the latter while the former is being recomputed on every throttled
# tick keeps a drag responsive even at a high n_points setting. The drag's
# final tick (sliderReleased) always forces one full-resolution redraw
# regardless, so the plotted curve never settles at this reduced grid.
DRAG_N_POINTS_CAP = 150

# Log-decade ranges for the signed Faraday-depth (phi) and turbulent term (dphi)
# sliders -- wide enough to explore the models' full behavior without
# being tied to any particular model's fit-prior bounds.
PHI_LOG_RANGE = (-6.5, 6.5)
DPHI_LOG_RANGE = (0.0, 6.5)

# Placeholder q_err/u_err for rmsynth_from_model's noiseless model curve --
# rm_synthesis.compute_faraday_spectrum only ever uses q_err/u_err in ratios
# (Briggs relative weighting) or as an overall multiplicative scale
# (sigma_fdf, the CLEAN threshold), never in absolute terms that would
# affect the recovered dirty/clean spectrum itself -- so any constant works
# equally well for "every point weighted equally", but the constant's own
# absolute size *does* set sigma_fdf/threshold's absolute size. A
# fractional-polarization curve is typically well under 1 in amplitude, so
# using 1.0 here (rather than something negligibly small) previously made
# the plotted sigma_fdf noise line -- and the CLEAN stopping threshold --
# come out comparable to or larger than the curve's own peak, even though
# there's no real noise to represent.
RMSYNTH_MODEL_WEIGHT = 1e-6

# Linear range for power-law spectral indices
# +/-3 covers typical AGN jet spectral indices.
ALPHA_RANGE = (-3.0, 2.0)
EPS_RANGE = (0.0, 1.0)

# Multipliers applied to the current wavelength range's own (nu_min, nu_max)
# to get the SSA/thermal turnover-frequency sliders' (lo, hi) bounds --
# wide enough to place a turnover well outside the plotted band in either
# direction.
NU0_BOUNDS_MULT = (1e-3, 1e3)

DEFAULT_TEMP_K = 1e4  # typical HII-region/AGN narrow-line-region electron temperature
DEFAULT_BETA = 0.0

# Quick-select wavelength ranges (min_mm, max_mm) for the preset dropdown.
# Full SPARC4 optical spans the standard (Fukugita et al. 1996) round-number
# SDSS g/r/i/z band edges -- per SPARC4's own instrument papers, which
# describe its 4 simultaneous channels as SDSS-like without republishing
# their own numbers -- as SPARC4's optical range; z's red edge is nominally
# CCD-QE-limited rather than a hard filter cutoff, so 1000 nm here is
# itself just a common round-number convention.
# Full HAWC+ spans SOFIA/HAWC+'s full filter-wheel range (Harper et al.
# 2018) -- its five broadband filters run 40-250 um, though Band B (63 um)
# has a known oversaturation issue and is left out of DEFAULT_BANDS_BY_PRESET
# below in favor of the 4 bands (A/C/D/E) actually used in science papers.
WAVELENGTH_PRESETS = [
    ('Standard ALMA (0.75 - 3.7 mm / 80 - 350 GHz)', 0.75, 3.7),
    ('Full ALMA (0.25 - 9 mm / 30 - 1200 GHz)', 0.25, 9.0),
    ('VLA high (0.6 - 2 cm / 15 - 50 GHz)', 6.0, 20.0),
    ('VLA low (2 - 30 cm / 1 - 15 GHz)', 20.0, 300.0),
    ('Full VLA (0.6 - 30 cm / 1 - 50 GHz)', 6.0, 300.0),
    ('Full GHz (0.25mm - 30 cm / 1 - 1200 GHz)', 0.25, 300),
    ('Full MeerKat (8.6 - 70 cm / 500 - 3500 MHz)', 86.0, 700.0),
    ('Full LOFAR (1.25 - 30 m / 10 - 240 MHz)', 1250, 30000),
    ('Full MHz (8.6 cm - 30 m / 10 - 3500 GHz)', 86.0, 30000),
    ('Full radio (0.25mm - 30 m / 500 MHz - 1200 GHz)', 0.25, 30000),
    ('Full HAWC+ FIR (40 - 250 um / 1.2 - 7.5 THz)', 0.04, 0.25),
    ('Full SPARC4 optical (400 - 1000 nm / 300 - 750 THz)', 4.0e-4, 1.0e-3),
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
        elif self.kind == 'temp':
            # log-scale like nu0 -- a thermal shape's Wien cutoff and
            # Rayleigh-Jeans amplitude both depend on T across many decades
            # (K, never 0).
            self.lo, self.hi = np.log10(lo), np.log10(hi)
            default = 0.5 * (self.lo + self.hi)
        elif self.kind in ('freq', 'wave'):
            # log-scale like nu0/temp -- a custom model's own characteristic
            # frequency/wavelength constant (e.g. a break frequency inside
            # j_p(z)) can plausibly span decades too, same reasoning as
            # nu0/temp -- see CUSTOM_PARAM_KINDS's own module comment in
            # models.py for why this is a separate kind from 'nu0' (units:
            # MHz/m, matching the reserved nu/lambda symbols exactly, no
            # GHz-style conversion applied anywhere else).
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
        if self.kind in ('nu0', 'temp', 'freq', 'wave'):
            return 10 ** working
        return working  # scale, alpha, eps

    def encode(self, typed):
        """Inverse of decode: turn a number typed into the value field
        (in the same units/domain shown there -- degrees for X, percent
        for p, the physical value itself for phi/dphi/nu0/temp/freq/wave,
        unitless for scale/alpha/eps) back into the slider's "working"
        domain."""
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
        if self.kind in ('nu0', 'temp', 'freq', 'wave'):
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
        else:  # phi/dphi/scale/alpha/eps/nu0/temp/freq/wave already share value()'s domain
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
        elif self.kind in ('phi', 'dphi', 'nu0', 'temp', 'freq', 'wave'):
            text = f'{phys:.2e}'
        elif self.kind == 'eps':
            text = f'{phys:.3f}'
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


class MainWindow(QMainWindow, SamplingMixin, MeasurementsMixin):
    NO_FIT_TEXT = 'Fit some data to get statistics'

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Polvista: POLarization VISualizer Tool for Astronomy')
        #self.resize(1800, 1000) # size of the window
        self.resize(1500, 700) # size of the window

        self.sliders = []
        self.fit_data = None  # (wl, q, q_err, u, u_err, freq, I) from the Load Data... button, or None
        self.data_nu_min = None  # [MHz] loaded data's own nu_min, set by Fitting -- see run_fit

        # Which source the RM-synth tab's currently plotted Faraday
        # spectrum (if any) was synthesized from -- 'model'/'data'/
        # 'measurements', or None while its plot is still the empty
        # placeholder. Lets Clear data / the Measurements tab's Clear only
        # blank the RM-synth plot when it's actually theirs to blank (see
        # clear_data/clear_measurement_points) -- a model-sourced spectrum
        # has nothing to do with loaded data or generated measurements, so
        # it survives either.
        self.rmsynth_source = None

        # MultiNest (Sampling tab) result state -- see apply_posterior_result,
        # build_corner_tab, run_multinest_fit, load_samples_action.
        self.mn_worker = None
        self.mn_model = None       # model func the most recent MultiNest run/load was for
        self.posterior_samples = None
        self.posterior_model = None
        self.corner_tab = None

        # See PLOT_THROTTLE_MS/request_update_plot and DRAG_N_POINTS_CAP/
        # on_slider_drag_started for what these back.
        self._dragging = False
        self._plot_pending = False
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._flush_update_plot)

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
        # Lower bound/decimals go down to fractions of a nm (the SDSS
        # optical presets run 4e-4 - 1e-3 mm) -- 3 decimals used to round
        # those straight to 0.000 and 0.01mm used to clamp them.
        self.wl_min.setRange(1e-4, 1000.0)
        self.wl_min.setDecimals(6)
        self.wl_min.setValue(0.75)
        self.wl_max = QDoubleSpinBox()
        self.wl_max.setRange(2e-4, 40000.0)
        self.wl_max.setDecimals(6)
        self.wl_max.setValue(3.7)
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 3000)
        self.n_points.setValue(1000)
        # Redraw the plots any time one of these three number boxes changes.
        for w in (self.wl_min, self.wl_max, self.n_points):
            w.valueChanged.connect(self.request_update_plot)
        wl_row.addWidget(QLabel('min'))
        wl_row.addWidget(self.wl_min)
        wl_row.addWidget(QLabel('max'))
        wl_row.addWidget(self.wl_max)
        wl_row.addWidget(QLabel('n'))
        wl_row.addWidget(self.n_points)
        wl_box_layout.addLayout(wl_row)

        self.log_xscale = QCheckBox('log x-scale')
        self.log_xscale.setChecked(True)
        self.log_xscale.stateChanged.connect(self.request_update_plot)
        wl_box_layout.addWidget(self.log_xscale)

        left_layout.addWidget(wl_box)

        left_layout.addWidget(QLabel('<b>Model</b>'))
        # QComboBox is the dropdown; addItem's 2nd arg is arbitrary data
        # (here the model function) retrievable later via currentData().
        self.model_combo = QComboBox()
        for func, spec in MODELS.items():
            self.model_combo.addItem(spec.label, func)
        # "Custom model..." always stays the last row (see
        # open_custom_model_dialog, which inserts newly-built custom models
        # just above it) -- picking it doesn't select a model itself, it
        # opens the builder dialog instead (see on_model_combo_changed).
        self.model_combo.addItem('Custom model...', CUSTOM_MODEL_SENTINEL)
        self._last_model_index = 0
        # Qt's signal/slot mechanism: connect() wires the combo box's
        # "selection changed" signal to a method that runs whenever the
        # user picks a different model.
        self.model_combo.currentIndexChanged.connect(self.on_model_combo_changed)
        left_layout.addWidget(self.model_combo)

        # spectral weighting parameters -- more relevant for two-component models
        self.spectral_box = QGroupBox()
        self.spectral_layout = QVBoxLayout(self.spectral_box)

        # Dropdown(s) selecting the source function S'(nu) used to weight
        # each component's spectral contribution (see
        # models.set_spectral_shape). A single-component model only ever
        # needs one, shown as a "Spectrum" header at the top of the box
        # (shape_header_single/shape_combo); a two-component model instead
        # gets one per component, each its own "Component N" header
        # (comp1_header/shape_combo_1, comp2_header/shape_combo_2) placed
        # right before that component's own alpha slider -- letting the two
        # components use different spectral shapes entirely (e.g. one
        # power-law, one SSA). rebuild_sliders picks which of these are
        # actually shown, and where, based on the current model's own
        # component count -- none of them are placed into spectral_layout
        # here.
        self.shape_header_single, self.shape_combo = self.build_spectrum_header('Spectrum')
        self.comp1_header, self.shape_combo_1 = self.build_spectrum_header('Component 1')
        self.comp2_header, self.shape_combo_2 = self.build_spectrum_header('Component 2')

        # Turnover-frequency nu_0 slider(s) -- log-scale, like phi/dphi,
        # since a turnover can plausibly sit many decades from the plotted
        # band -- only shown once its own component's shape dropdown is set
        # to 'SSA' or 'Thermal' (see sync_spectrum_ui). A two-component
        # model gets an independent nu_0 per component; a single-component
        # model only ever needs the first (reused as-is, alongside
        # shape_combo, for that case). Not placed into spectral_layout here
        # -- rebuild_sliders inserts each at the right spot among that
        # model's own eps/alpha sliders (epsilon, then component 1's own
        # nu_0/T/alpha, then component 2's; see its docstring).
        nu0_lo, nu0_hi = self.nu0_bounds_ghz()
        default_nu0 = self.default_nu0_ghz()
        self.nu0_slider_1 = self.build_nu0_slider(
            nu0_lo, nu0_hi, default_nu0, r'$\nu_0$', 'SSA/thermal turnover frequency.')
        self.nu0_slider_2 = self.build_nu0_slider(
            nu0_lo, nu0_hi, default_nu0, r'$\nu_{0,2}$', "Component 2's SSA/thermal turnover frequency.")

        # Electron-temperature T slider(s) -- log-scale (see TEMP_BOUNDS_K),
        # only shown once its own component's shape dropdown is set to
        # 'Thermal' (see sync_spectrum_ui). Same one-or-two-per-model
        # pattern as the nu_0 sliders above.
        temp_lo, temp_hi = TEMP_BOUNDS_K
        self.temp_slider_1 = self.build_temp_slider(
            temp_lo, temp_hi, DEFAULT_TEMP_K, r'$T$', 'Thermal free-free electron temperature.')
        self.temp_slider_2 = self.build_temp_slider(
            temp_lo, temp_hi, DEFAULT_TEMP_K, r'$T_2$', "Component 2's thermal free-free electron temperature.")

        # Curvature-index beta slider(s) -- linear (see BETA_BOUNDS), only
        # shown once its own component's shape dropdown is set to
        # 'Log-parabola' (see sync_spectrum_ui). Same one-or-two-per-model
        # pattern as the nu_0/T sliders above; unlike those, log-parabola
        # doesn't need its own nu_0/T (it shares the band nu_0 like a plain
        # power law), so beta is the only extra control it adds.
        beta_lo, beta_hi = BETA_BOUNDS
        self.beta_slider_1 = self.build_beta_slider(
            beta_lo, beta_hi, DEFAULT_BETA, r'$\beta$', 'Log-parabola curvature index.')
        self.beta_slider_2 = self.build_beta_slider(
            beta_lo, beta_hi, DEFAULT_BETA, r'$\beta_2$', "Component 2's log-parabola curvature index.")

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

        # Shown only for a custom model (see models.build_custom_model) whose
        # current phi(z)/wavelength-range combination outran the LOS
        # quadrature's resolution cap -- i.e. the plotted p/EVPA/Stokes
        # curves for it can't be trusted at these parameter values (see
        # update_plot, which toggles this after every redraw).
        self.custom_resolution_warning = QLabel(
            '⚠ Faraday depth too high for the current wavelength range at this '
            'resolution -- this curve may not be numerically accurate.')
        self.custom_resolution_warning.setWordWrap(True)
        self.custom_resolution_warning.setStyleSheet(
            'background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 4px;')
        self.custom_resolution_warning.setVisible(False)
        right_layout.addWidget(self.custom_resolution_warning)

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

        stokes_top_row = QHBoxLayout()
        stokes_top_row.addWidget(self.stokes_toolbar)
        stokes_top_row.addStretch(1)
        stokes_top_row.addWidget(QLabel('Q, U view:'))
        self.stokes_mode_combo = QComboBox()
        self.stokes_mode_combo.addItems(['Polar', 'Spectra'])
        self.stokes_mode_combo.currentIndexChanged.connect(self.request_update_plot)
        stokes_top_row.addWidget(self.stokes_mode_combo)
        stokes_layout.addLayout(stokes_top_row)

        stokes_layout.addWidget(self.stokes_canvas, stretch=1)

        # RM-synth tab: the Faraday-depth spectrum from RM synthesis +
        # RM-CLEAN (rm_synthesis.compute_faraday_spectrum), synthesized on
        # demand from one of three sources -- never from a parameter-slider
        # drag or the wavelength-range spin boxes, unlike p_evpa_tab/
        # stokes_tab above (see RMSynthPlot's own docstring and
        # rmsynth_from_model/_data/_measurements below).
        self.rmsynth_canvas = RMSynthPlot(self)
        self.rmsynth_toolbar = NavigationToolbar(self.rmsynth_canvas, self)
        rmsynth_tab = QWidget()
        rmsynth_layout = QVBoxLayout(rmsynth_tab)
        rmsynth_layout.setContentsMargins(0, 0, 0, 0)

        rmsynth_top_row = QHBoxLayout()
        rmsynth_top_row.addWidget(self.rmsynth_toolbar)
        rmsynth_top_row.addStretch(1)
        rmsynth_top_row.addWidget(QLabel('Synthesize from:'))
        self.rmsynth_model_button = QPushButton('model')
        self.rmsynth_model_button.setToolTip(
            "Compute the Faraday spectrum of the model curve over the full selected wavelength range.")
        self.rmsynth_model_button.clicked.connect(self.rmsynth_from_model)
        rmsynth_top_row.addWidget(self.rmsynth_model_button)
        self.rmsynth_data_button = QPushButton('data')
        self.rmsynth_data_button.setToolTip("Compute the Faraday spectrum of the loaded data.")
        self.rmsynth_data_button.setEnabled(False)  # only enabled once data is loaded, see load_data_action
        self.rmsynth_data_button.clicked.connect(self.rmsynth_from_data)
        rmsynth_top_row.addWidget(self.rmsynth_data_button)
        self.rmsynth_measurements_button = QPushButton('measurements')
        self.rmsynth_measurements_button.setToolTip("Compute the Faraday spectrum of the generated measurements.")
        self.rmsynth_measurements_button.setEnabled(False)  # only enabled once Generate has produced points
        self.rmsynth_measurements_button.clicked.connect(self.rmsynth_from_measurements)
        rmsynth_top_row.addWidget(self.rmsynth_measurements_button)
        rmsynth_layout.addLayout(rmsynth_top_row)

        rmsynth_layout.addWidget(self.rmsynth_canvas, stretch=1)

        # The empty plot's own phi axis tracks the wavelength-range spin
        # boxes (but only while there's nothing actually plotted yet, see
        # refresh_rmsynth_empty_axis) -- these are ordinary spin boxes, not
        # parameter sliders, so this doesn't conflict with the tab never
        # redrawing on a slider drag.
        self.wl_min.valueChanged.connect(self.refresh_rmsynth_empty_axis)
        self.wl_max.valueChanged.connect(self.refresh_rmsynth_empty_axis)
        self.refresh_rmsynth_empty_axis()

        # QTabWidget switches between the plot pages via clickable tabs. A
        # fourth "Corner plot" tab is added/removed dynamically by
        # build_corner_tab/remove_corner_tab once a MultiNest fit (or
        # Load samples) has something to show there.
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(p_evpa_tab, 'p / EVPA  vs  λ²')
        self.plot_tabs.addTab(stokes_tab, 'I, Q, U  vs  ν')
        self.plot_tabs.addTab(rmsynth_tab, 'RM-synth')
        right_layout.addWidget(self.plot_tabs, stretch=1)
        root.addWidget(right, stretch=1)

        self.build_menu()
        self.rebuild_sliders()

    def on_model_combo_changed(self, index):
        """model_combo's own slot: opens the custom-model builder when its
        sentinel "Custom model..." row is picked (always the last row, see
        _build_ui) instead of treating it as a model switch; otherwise
        this is a real model switch, so track it as the row to revert to
        next time the sentinel is picked (see open_custom_model_dialog) and
        rebuild the sliders as before."""
        if self.model_combo.itemData(index) is CUSTOM_MODEL_SENTINEL:
            self.open_custom_model_dialog(revert_index=self._last_model_index)
            return
        self._last_model_index = index
        self.rebuild_sliders()

    def build_custom_model_menu_action(self):
        """Models menu's "Build Custom Model..." entry -- same dialog as
        the dropdown's sentinel row, just with nothing to revert if the
        user cancels (the combo box selection isn't touched by opening it)."""
        self.open_custom_model_dialog()

    def open_custom_model_dialog(self, revert_index=None):
        """Open the "Build Custom Model" window (models.build_custom_model
        does the actual parsing/registration). If the user completes it,
        the new model is inserted into model_combo just above the "Custom
        model..." sentinel row and selected. If they cancel and this was
        opened via the dropdown's own sentinel row (`revert_index` given),
        the combo box reverts to whatever was selected before, so the
        sentinel row is never left showing as the "current" selection.

        blockSignals wraps every combo-box mutation below: inserting a row
        *before* the sentinel's own position shifts the sentinel one slot
        over, and since currentIndex still numerically points at that same
        (now-shifted) slot until setCurrentIndex actually runs, an
        unblocked insertItem alone fires a transient currentIndexChanged
        still pointing at the sentinel -- which on_model_combo_changed
        would treat as "the sentinel got picked again" and reopen this same
        dialog right back up. Blocking means rebuild_sliders has to be
        called explicitly afterward instead (same pattern already used by
        load_model_action, for the same reason). shape_combo is blocked for
        the same reason as model_combo above: firing on_spectral_shape_changed
        before rebuild_sliders has actually built this new model's own
        sliders would read stale slider state -- rebuild_sliders's own
        trailing sync_spectrum_ui/refit_equation/update_plot calls cover it
        once, in the right order, after both combo boxes are already
        settled."""
        dialog = CustomModel(self)
        if dialog.exec_() and dialog.model_func is not None:
            func = dialog.model_func
            spec = MODELS[func]
            insert_at = self.model_combo.count() - 1  # just above the sentinel
            self.model_combo.blockSignals(True)
            self.model_combo.insertItem(insert_at, spec.label, func)
            self.model_combo.setCurrentIndex(insert_at)
            self.model_combo.blockSignals(False)
            self._last_model_index = insert_at
            # Whatever Spectral model the builder dialog's own (preview-only)
            # dropdown was left on becomes this new model's *initial* live
            # Spectrum-box selection -- so opacity (if SSA/Thermal was
            # previewed) is already active the moment the model is first
            # shown, matching what its equation card/preview just showed,
            # rather than silently reverting to Power-law.
            shape_idx = self.shape_combo.findData(dialog.spectral_shape_combo.currentData())
            if shape_idx >= 0:
                self.shape_combo.blockSignals(True)
                self.shape_combo.setCurrentIndex(shape_idx)
                self.shape_combo.blockSignals(False)
            self.rebuild_sliders()
        elif revert_index is not None:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(revert_index)
            self.model_combo.blockSignals(False)
            self._last_model_index = revert_index
            self.rebuild_sliders()

    def edit_custom_model_menu_action(self):
        """Models menu's "Edit Custom Model..." entry -- only ever enabled
        (see rebuild_sliders) while the currently selected model is a
        custom one. Reopens the builder dialog pre-filled from its own
        saved definition (see models.CUSTOM_MODEL_DEFS) and its own
        current live Spectrum-box shape (see CustomModel's own
        initial_spectral_shape param -- editing shouldn't visually revert
        opacity off just because the dialog's own preview-only dropdown
        always used to start at Power-law); on success, the edited model
        *replaces* the original in place (same combo row, same
        func.__name__ -- see CustomModel's own edit_func param
        and build_custom_model's `name`) rather than adding a second entry,
        since build_custom_model always returns a fresh function object
        even when reusing the same name. The live Spectrum-box shape itself
        is untouched by this -- it's global/shared state, not something
        Edit resets."""
        func = self.model_combo.currentData()
        existing_def = CUSTOM_MODEL_DEFS.get(func)
        if existing_def is None:
            return
        dialog = CustomModel(self, existing_def=existing_def, edit_func=func,
                              initial_spectral_shape=self.shape_combo.currentData())
        if not (dialog.exec_() and dialog.model_func is not None):
            return
        new_func = dialog.model_func
        idx = self.model_combo.findData(func)
        if idx >= 0:
            self.model_combo.setItemData(idx, new_func)
            self.model_combo.setItemText(idx, MODELS[new_func].label)
        if new_func is not func:
            CUSTOM_MODEL_DEFS.pop(func, None)
            MODELS.pop(func, None)
        self.rebuild_sliders()

    def rebuild_sliders(self, *_):
        for sl in self.sliders:
            sl.setParent(None)
        self.sliders = []

        func = self.model_combo.currentData()
        spec = MODELS[func]
        lo_bounds, hi_bounds = spec.bounds
        self.edit_custom_model_action.setEnabled(func in CUSTOM_MODEL_DEFS)

        # remove the trailing stretch, re-add sliders, then put stretch back
        self.param_layout.takeAt(self.param_layout.count() - 1)
        self.spectral_layout.takeAt(self.spectral_layout.count() - 1)
        # None of the Spectrum-box header/nu_0 widgets below are one of
        # self.sliders (see _build_ui) so they survive the setParent(None)
        # loop above untouched -- pull them all out of spectral_layout too,
        # so each can be re-inserted below at the spot matching this
        # model's own component count and spectral params (see the loop's
        # comment).
        for w in (self.shape_header_single, self.comp1_header, self.comp2_header,
                  self.nu0_slider_1, self.nu0_slider_2, self.temp_slider_1, self.temp_slider_2,
                  self.beta_slider_1, self.beta_slider_2):
            self.spectral_layout.removeWidget(w)

        has_spectral = False
        two_comp = spec.n_components == 2
        if not two_comp:
            self.spectral_layout.addWidget(self.shape_header_single)
        for i, param in enumerate(spec.params):
            sl = ParamSlider(param, lo_bounds[i], hi_bounds[i])
            sl.valueChanged.connect(self.request_update_plot)
            sl.slider.sliderPressed.connect(self.on_slider_drag_started)
            sl.slider.sliderReleased.connect(self.on_slider_drag_ended)
            # self.sliders keeps the *original* spec.params order regardless of
            # which box a slider is displayed in -- update_plot() reads this
            # list positionally, matching the model function's param order.
            self.sliders.append(sl)
            if param.kind == 'eps':
                # Display order for a two-component model: epsilon, then
                # "Component 1" (its own shape dropdown, nu_0, and alpha1),
                # then "Component 2" (likewise) -- see the 'alpha' branch
                # below for where each component's own header/nu_0 land.
                self.spectral_layout.addWidget(sl)
                has_spectral = True
                if two_comp:
                    self.spectral_layout.addWidget(self.comp1_header)
            elif param.kind == 'alpha':
                has_spectral = True
                if param.name == 'alpha2':
                    self.spectral_layout.addWidget(self.nu0_slider_2)
                    self.spectral_layout.addWidget(self.temp_slider_2)
                    self.spectral_layout.addWidget(sl)
                    self.spectral_layout.addWidget(self.beta_slider_2)
                else:
                    self.spectral_layout.addWidget(self.nu0_slider_1)
                    self.spectral_layout.addWidget(self.temp_slider_1)
                    self.spectral_layout.addWidget(sl)
                    self.spectral_layout.addWidget(self.beta_slider_1)
                    if two_comp:
                        self.spectral_layout.addWidget(self.comp2_header)
            else:
                self.param_layout.addWidget(sl)

        self.param_layout.addStretch(1)
        self.spectral_layout.addStretch(1)
        self.param_box.setTitle(spec.title)
        self.spectral_scroll.setVisible(has_spectral)

        # removeWidget above only detaches a widget from the layout -- it
        # doesn't hide it, so whichever of these wasn't re-added this pass
        # (e.g. shape_header_single for a two-component model) would
        # otherwise still report itself visible, stale geometry and all.
        # Done only now, after the addWidget calls above have given
        # comp1_header/comp2_header a real parent at least once -- setting
        # a still-parentless QWidget visible makes Qt treat it as its own
        # top-level window instead, which then breaks the *next* addWidget
        # reparenting.
        self.shape_header_single.setVisible(not two_comp)
        self.comp1_header.setVisible(two_comp)
        self.comp2_header.setVisible(two_comp)

        if hasattr(self, 'results_label'):
            self.results_label.setText(self.NO_FIT_TEXT)

        self.rebuild_sampling_bounds()

        # A model switch can change the component count (single- vs
        # two-component), which the SSA/thermal turnover-frequency (and
        # thermal temperature) row(s) need to track -- see sync_spectrum_ui.
        self.sync_spectrum_ui()

        self.refit_equation()

        self.update_plot()

    def build_spectrum_header(self, label_text):
        """One (container, combo) pair for a Spectrum-box header row:
        `label_text` on the left, a Power-law/SSA/Thermal/Log-parabola shape dropdown on the
        right, wired to on_spectral_shape_changed -- used for the
        single-component "Spectrum" header and, for two-component models,
        each component's own "Component N" header (see _build_ui/
        rebuild_sliders). Returned as a standalone container (not yet
        placed in any layout) so rebuild_sliders can insert it wherever the
        current model's own component count calls for."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label_text))
        row.addStretch(1)
        combo = QComboBox()
        for shape_label, shape_key in SPECTRAL_SHAPES:
            combo.addItem(shape_label, shape_key)
        combo.currentIndexChanged.connect(self.on_spectral_shape_changed)
        row.addWidget(combo)
        return container, combo

    def build_nu0_slider(self, lo_ghz, hi_ghz, default_ghz, latex, description):
        """One nu_0 ParamSlider (log-scale, like a dphi slider) for the
        Spectrum box's SSA/thermal turnover-frequency controls -- see
        _build_ui, which builds one or two of these depending on the
        current model's component count (see sync_spectrum_ui).
        `default_ghz` is applied (and its own valueChanged wired up) only
        after construction, so building the slider itself can't
        prematurely fire the app-level handler.

        Starts *fixed* (checkbox checked) -- at its default (~1% of the
        band's own nu_min, deep in the optically-thin regime), a fit
        should start out assuming there's no turnover in view and only fit
        alpha; the user frees this slider (unchecks it) to let a fit
        jointly solve for nu_0 too (see MainWindow.fit_spectrum_lsq -- note
        this joint solve only happens for 'ssa'; a 'thermal' component's
        nu_0/T are always taken as-is from these sliders, never fit)."""
        sl = ParamSlider(Param('nu0', latex, 'nu0', description), lo_ghz, hi_ghz)
        sl.set_value(default_ghz)
        sl.fix_checkbox.setChecked(True)
        sl.valueChanged.connect(self.on_spectral_shape_changed)
        return sl

    def build_temp_slider(self, lo_k, hi_k, default_k, latex, description):
        """One T ParamSlider (log-scale, like build_nu0_slider) for the
        Spectrum box's thermal electron-temperature controls -- see
        _build_ui/sync_spectrum_ui. Starts *fixed*, like the nu_0 sliders:
        least-squares fitting never solves for a thermal component's own
        T (see fit_spectrum_lsq), so there's no "let the fit vary this"
        state to default to -- the checkbox only matters in that it keeps
        this slider visually consistent with nu_0's own default."""
        sl = ParamSlider(Param('temp', latex, 'temp', description), lo_k, hi_k)
        sl.set_value(default_k)
        sl.fix_checkbox.setChecked(True)
        sl.valueChanged.connect(self.on_spectral_shape_changed)
        return sl

    def build_beta_slider(self, lo, hi, default, latex, description):
        """One beta ParamSlider (linear, 'scale' kind -- same generic
        slider used for e.g. Tribble's s or the covering fraction f, since
        beta doesn't need nu_0/temp's log-scale) for the Spectrum box's
        log-parabola curvature controls -- see _build_ui/sync_spectrum_ui.
        Starts *fixed*, like the nu_0/T sliders, for the same reason: beta
        is never solved for during fitting (see fit_spectrum_lsq)."""
        sl = ParamSlider(Param('beta', latex, 'scale', description), lo, hi)
        sl.set_value(default)
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

    def current_shapes(self):
        """(shape1, shape2) currently selected in the Spectrum box --
        `shape1` is the single-component shape dropdown for a
        single-component model, or component 1's own dropdown for a
        two-component model; `shape2` is component 2's own dropdown,
        meaningful only for a two-component model (mirrors `shape1`
        otherwise, since a single-component model has no second
        component)."""
        n_components = MODELS[self.model_combo.currentData()].n_components
        if n_components == 2:
            return self.shape_combo_1.currentData(), self.shape_combo_2.currentData()
        shape = self.shape_combo.currentData()
        return shape, shape

    def sync_spectrum_ui(self):
        """Keep the Spectrum box's nu_0/T/beta slider(s) -- and each
        component's own alpha slider's visibility -- matched to each
        component's own shape dropdown and the current model's component
        count, and push nu_0/T/beta's values into models' global
        spectral-shape state (see models.set_spectral_shape) -- called
        whenever any shape dropdown, a nu_0/T/beta slider, or the model
        itself changes."""
        shape1, shape2 = self.current_shapes()
        spec = MODELS[self.model_combo.currentData()]
        two_comp = spec.n_components == 2
        needs_nu0_1 = shape1 in ('ssa', 'thermal')
        needs_nu0_2 = two_comp and shape2 in ('ssa', 'thermal')
        needs_temp_1 = shape1 == 'thermal'
        needs_temp_2 = two_comp and shape2 == 'thermal'
        needs_beta_1 = shape1 == 'logparabola'
        needs_beta_2 = two_comp and shape2 == 'logparabola'
        self.nu0_slider_1.setVisible(needs_nu0_1)
        self.nu0_slider_2.setVisible(needs_nu0_2)
        self.temp_slider_1.setVisible(needs_temp_1)
        self.temp_slider_2.setVisible(needs_temp_2)
        self.beta_slider_1.setVisible(needs_beta_1)
        self.beta_slider_2.setVisible(needs_beta_2)
        self.nu0_slider_1.name_label.setPixmap(
            latex_pixmap(r'$\nu_{0,1}$' if two_comp else r'$\nu_0$'))
        self.temp_slider_1.name_label.setPixmap(
            latex_pixmap(r'$T_1$' if two_comp else r'$T$'))
        self.beta_slider_1.name_label.setPixmap(
            latex_pixmap(r'$\beta_1$' if two_comp else r'$\beta$'))
        # A thermal component's own spectral index is emergent from T, not
        # a free parameter (see source_function/thermal_source) -- hide
        # its alpha slider rather than show an inert control (log-parabola
        # still uses alpha directly, alongside its own beta, so its alpha
        # slider stays visible). alpha_indices follows spec.params' own
        # order: index 0 is the single alpha for a single-component model,
        # or alpha1 (tied to shape1) for a two-component one; index 1,
        # when present, is alpha2 (shape2).
        alpha_indices = spec.indices('alpha')
        if alpha_indices:
            self.sliders[alpha_indices[0]].setVisible(shape1 != 'thermal')
            if len(alpha_indices) > 1:
                self.sliders[alpha_indices[1]].setVisible(shape2 != 'thermal')
        nu0 = self.nu0_slider_1.value() * 1e3 if needs_nu0_1 else None  # GHz -> MHz
        nu0_2 = self.nu0_slider_2.value() * 1e3 if needs_nu0_2 else None
        T = self.temp_slider_1.value() if needs_temp_1 else None
        T2 = self.temp_slider_2.value() if needs_temp_2 else None
        beta = self.beta_slider_1.value() if needs_beta_1 else None
        beta2 = self.beta_slider_2.value() if needs_beta_2 else None
        set_spectral_shape(shape1, nu0=nu0, shape2=shape2, nu0_2=nu0_2, T=T, T2=T2,
                            beta=beta, beta2=beta2)

    def on_spectral_shape_changed(self, *_):
        """Slot for the Spectrum box's shape dropdown(s) and nu_0/T spin
        box(es): re-sync models' spectral-shape state and redraw the
        equation card and plots to reflect it."""
        self.sync_spectrum_ui()
        self.refit_equation()
        self.update_plot()

    def refit_equation(self):
        spec = MODELS[self.model_combo.currentData()]
        shape1, shape2 = self.current_shapes()
        eq = full_equation(spec, shape1, shape2)
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
        n_points = min(self.n_points.value(), DRAG_N_POINTS_CAP) if self._dragging else self.n_points.value()
        if self.log_xscale.isChecked():
            wl_ext = np.logspace(np.log10(wl_min_mm), np.log10(wl_max_mm), n_points) * 1e-3
        else:
            wl_ext = np.linspace(wl_min_mm, wl_max_mm, n_points) * 1e-3
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
                                        log_xscale=log_xscale, nu_min=self.data_nu_min,
                                        mode=self.stokes_mode_combo.currentText())
        # Both canvases above already called func(wl_ext, pars) at least
        # once, which (for a custom model -- see models.build_custom_model)
        # refreshes its own last_call_underresolved flag for these exact
        # wl_ext/pars; non-custom models never set the attribute at all.
        self.custom_resolution_warning.setVisible(
            getattr(func, 'last_call_underresolved', False))

    def request_update_plot(self, *_):
        """Throttled entry point for update_plot -- see PLOT_THROTTLE_MS.
        Connected wherever update_plot used to be wired directly to a
        signal that can fire many times per second (parameter sliders,
        wavelength-range spin boxes, log-x/Stokes-mode toggles); explicit
        one-shot callers (Reset model, Load data, a finished fit, ...) still
        call update_plot() directly, since those aren't bursty and want
        their redraw immediately."""
        if self._plot_timer.isActive():
            self._plot_pending = True
            return
        self.update_plot()
        self._plot_timer.start(PLOT_THROTTLE_MS)

    def _flush_update_plot(self):
        """PLOT_THROTTLE_MS timer callback -- redraws once more, and keeps
        the timer running, only if a request_update_plot call actually
        arrived (and was coalesced) during the throttle window; otherwise
        the burst has ended and there's nothing left to flush."""
        if self._plot_pending:
            self._plot_pending = False
            self.update_plot()
            self._plot_timer.start(PLOT_THROTTLE_MS)

    def on_slider_drag_started(self):
        """See DRAG_N_POINTS_CAP -- every ParamSlider wired to it (the
        model's own parameter sliders, the ones actually worth capping)
        connects both this and on_slider_drag_ended to its own QSlider's
        sliderPressed/sliderReleased."""
        self._dragging = True

    def on_slider_drag_ended(self):
        self._dragging = False
        self._plot_pending = False  # this call's own full-res redraw supersedes any coalesced one
        self.update_plot()  # one final full-resolution redraw once the drag settles

    # ── RM-synth tab ───────────────────────────────────────────────────────
    def rmsynth_wl_bounds_m(self):
        """(wl_min, wl_max) [m] of the Wavelength range box's own current
        selection -- shared by refresh_rmsynth_empty_axis and
        rmsynth_from_model (which synthesizes over this same full range)."""
        wl_min_mm = min(self.wl_min.value(), self.wl_max.value())
        wl_max_mm = max(self.wl_min.value(), self.wl_max.value())
        return wl_min_mm * 1e-3, wl_max_mm * 1e-3

    def refresh_rmsynth_empty_axis(self, *_):
        """Keep the RM-synth tab's own empty-plot phi range matched to the
        current wavelength-range selection -- but only while nothing has
        actually been synthesized there yet (self.rmsynth_source is None).
        Once a source has been plotted, the wavelength-range spin boxes no
        longer touch that tab at all (see RMSynthPlot's own docstring)
        until Clear data/the Measurements tab's Clear (or another
        Synthesize from click) replaces it."""
        if self.rmsynth_source is not None:
            return
        wl_lo, wl_hi = self.rmsynth_wl_bounds_m()
        self.rmsynth_canvas.set_empty(phi_axis_half_width(wl_lo ** 2, wl_hi ** 2))

    def run_rmsynth(self, wl, q, u, q_err, u_err, source):
        """Shared tail end of rmsynth_from_model/_data/_measurements: run
        RM synthesis + RM-CLEAN on one (wl [m], q, u, q_err, u_err) set of
        channels and, on success, draw it -- replacing whatever the
        RM-synth tab showed before -- and remember which source it came
        from (see self.rmsynth_source)."""
        try:
            result = compute_faraday_spectrum(wl, q, u, q_err, u_err)
        except FaradaySpectrumError as e:
            QMessageBox.warning(self, 'RM synthesis', str(e))
            return
        self.rmsynth_canvas.plot_spectrum(result, source)
        self.rmsynth_source = source

    def rmsynth_from_model(self):
        """'model' button: Faraday spectrum of the model curve over the
        full selected wavelength range. q, u are the model's own fractional
        (Q/I, U/I) polarization -- func(wl, pars)'s real/imaginary parts,
        the same complex value ModelPlot's p/EVPA are derived from -- so
        this reads on the same fractional-polarization scale as
        rmsynth_from_data/_measurements below. There's no per-point noise
        for a pure curve, so every point is weighted equally, at a
        negligibly small absolute scale (see RMSYNTH_MODEL_WEIGHT) so the
        resulting sigma_fdf/CLEAN threshold reads as ~0 rather than as a
        spurious noise floor comparable to the curve's own amplitude.

        Deliberately does NOT reuse current_state()'s own wl_ext, and
        deliberately does NOT just mirror it in lambda^2 either: that grid
        is spaced (log-)uniformly in wavelength, and log spacing is
        invariant under squaring (log(lambda^2) = 2 log(lambda) is uniform
        whenever log(lambda) is), so a log-uniform-in-lambda^2 grid over
        the same endpoints would be numerically identical to it. Either
        way, over a wide band this piles most points up near the
        short-wavelength end, starving RM synthesis of the long-lambda^2-
        baseline coverage it actually needs -- which in turn fools
        rm_synthesis.measure_fwhm into picking an oversized CLEAN restoring
        beam and smearing the recovered spectrum into broad artificial
        tails. Sampling *linearly* (uniformly) in lambda^2 instead -- same
        endpoints (rmsynth_wl_bounds_m, the same full range
        refresh_rmsynth_empty_axis previews) and point count -- gives RM
        synthesis the even lambda^2 coverage it's actually sensitive to,
        regardless of the Visualization tab's own log-x display toggle
        (that toggle is about how the curve looks on screen, not what RM
        synthesis needs)."""
        func, spec, pars, _ = self.current_state()
        wl_lo, wl_hi = self.rmsynth_wl_bounds_m()
        n_points = self.n_points.value()
        l2_ext = np.linspace(wl_lo ** 2, wl_hi ** 2, n_points)
        wl_ext = np.sqrt(l2_ext)
        fit = func(wl_ext, pars)
        weight = np.full_like(wl_ext, RMSYNTH_MODEL_WEIGHT)
        self.run_rmsynth(wl_ext, fit.real, fit.imag, weight, weight, 'model')

    def rmsynth_from_data(self):
        """'data' button: Faraday spectrum of the data loaded via Load
        data -- greyed out (see load_data_action/clear_data) until some is.
        self.fit_data's own q, u, q_err, u_err are already the fractional
        (Q/I, U/I) polarization and its propagated errors (see
        load_data_action), exactly what this needs."""
        if self.fit_data is None:
            return
        wl, q, q_err, u, u_err, _freq, _I = self.fit_data
        self.run_rmsynth(wl, q, u, q_err, u_err, 'data')

    def rmsynth_from_measurements(self):
        """'measurements' button: Faraday spectrum of the Measurements
        tab's last Generate'd points -- greyed out (see
        generate_measurements/clear_measurement_points) until there are
        any. self.meas_export_rows holds absolute (I, Q, U) per point (see
        generate_measurements), so q, u and their errors are derived here
        the same way widgets.stokes_to_frac_qu/load_data_action do."""
        if not self.meas_export_rows:
            return
        rows = np.asarray(self.meas_export_rows, dtype=float)
        freq_ghz, I, I_err, Q, Q_err, U, U_err = (rows[:, i] for i in (1, 2, 3, 4, 5, 6, 7))
        wl = C / (freq_ghz * 1e9)
        q, u = Q / I, U / I
        q_err = np.sqrt((Q_err / I) ** 2 + (Q * I_err / I ** 2) ** 2)
        u_err = np.sqrt((U_err / I) ** 2 + (U * I_err / I ** 2) ** 2)
        self.run_rmsynth(wl, q, u, q_err, u_err, 'measurements')

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

        models_menu = menubar.addMenu('&Models')
        build_custom_action = QAction('Build Custom Model...', self)
        build_custom_action.triggered.connect(self.build_custom_model_menu_action)
        models_menu.addAction(build_custom_action)
        # Enabled only while the currently selected model is a custom one --
        # kept in sync by rebuild_sliders (every model switch, including
        # this action's own edit replacing the current row in place).
        self.edit_custom_model_action = QAction('Edit Custom Model...', self)
        self.edit_custom_model_action.setEnabled(False)
        self.edit_custom_model_action.triggered.connect(self.edit_custom_model_menu_action)
        models_menu.addAction(self.edit_custom_model_action)

        help_menu = menubar.addMenu('&Help')
        complex_pol_action = QAction('Complex polarization', self)
        complex_pol_action.triggered.connect(self.show_complex_polarization_help)
        help_menu.addAction(complex_pol_action)
        custom_model_action = QAction('Custom models', self)
        custom_model_action.triggered.connect(self.show_custom_model_help)
        help_menu.addAction(custom_model_action)
        qufitting_action = QAction('QU-fitting', self)
        qufitting_action.triggered.connect(self.show_qufitting_help)
        help_menu.addAction(qufitting_action)
        rmsynth_action = QAction('RM-synthesis', self)
        rmsynth_action.triggered.connect(self.show_rmsynth_help)
        help_menu.addAction(rmsynth_action)
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

        The SSA/thermal turnover-frequency, thermal temperature, and
        log-parabola curvature sliders aren't part of self.sliders (see
        _build_ui), so rebuild_sliders alone wouldn't reset them -- move
        them back to their own default value *and* fixed state explicitly
        (see build_nu0_slider/build_temp_slider/build_beta_slider)."""
        self.data_nu_min = None
        self.clear_posterior_samples()
        self.remove_corner_tab()
        default_nu0 = self.default_nu0_ghz()
        self.nu0_slider_1.set_value(default_nu0)
        self.nu0_slider_2.set_value(default_nu0)
        self.nu0_slider_1.fix_checkbox.setChecked(True)
        self.nu0_slider_2.fix_checkbox.setChecked(True)
        self.temp_slider_1.set_value(DEFAULT_TEMP_K)
        self.temp_slider_2.set_value(DEFAULT_TEMP_K)
        self.temp_slider_1.fix_checkbox.setChecked(True)
        self.temp_slider_2.fix_checkbox.setChecked(True)
        self.beta_slider_1.set_value(DEFAULT_BETA)
        self.beta_slider_2.set_value(DEFAULT_BETA)
        self.beta_slider_1.fix_checkbox.setChecked(True)
        self.beta_slider_2.fix_checkbox.setChecked(True)
        self.rebuild_sliders()

    def clear_data(self):
        """Erase any reference data points loaded via the Load Data... button
        -- generated measurements (the Measurements tab's own Clear button/
        clear_measurement_points) are a separate overlay and untouched here.
        Also closes the Corner plot tab (it was built against this data),
        but leaves any posterior-sample overlay in place -- only Reset
        model clears that, see reset_parameters.

        Also clears the RM-synth tab's own plot, but only if it was last
        synthesized from this same loaded data (see self.rmsynth_source) --
        a model- or measurements-sourced Faraday spectrum has nothing to do
        with this data and survives it."""
        self.canvas.clear_reference_data()
        self.stokes_canvas.clear_reference_data()
        self.fit_data = None
        self.data_nu_min = None
        self.remove_corner_tab()
        self.fit_button.setEnabled(False)
        self.rmsynth_data_button.setEnabled(False)
        if self.rmsynth_source == 'data':
            self.rmsynth_source = None
            self.refresh_rmsynth_empty_axis()
        self.set_sampling_tab_enabled(False)
        self.results_label.setText(self.NO_FIT_TEXT)
        self.update_plot()

    def run_fit(self):
        """Dispatches Fit! to the Bayesian (MultiNest) path when the
        Sampling tab is the active one (and pymultinest is installed), or
        the ordinary least-squares path otherwise. `self.sampling_tab`
        only exists when HAS_PYMULTINEST is True, so the short-circuit
        `and` below matters -- it's never evaluated otherwise."""
        func, _, _, _ = self.current_state()
        if getattr(func, 'is_custom', False):
            QMessageBox.information(
                self, 'Fit', "Custom models aren't supported for QU-fitting/Sampling yet -- "
                              "only the Visualization and Measurements tabs.")
            return
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
        estimate_alpha when every component's shape is 'powerlaw', or a
        joint (nu_0, alpha) nonlinear regression otherwise (see
        estimate_ssa_shape/estimate_shape_2comp, and build_nu0_slider for
        why a component's own nu_0 only joins that regression once its own
        slider is unchecked -- and only when that component's own shape is
        'ssa' to begin with, since a power-law component's reference
        frequency is always the shared band edge, never a free parameter)
        -- rather than being fit alongside p/X/phi/dphi. For a
        two-component model, alpha1/alpha2 always end up equal regardless
        of the two components' own shapes: there's only one real I(nu)
        dataset, so a single shared spectral index is all QU-only fitting
        can ever anchor two components to -- same reasoning
        FIT_FIXED_EPSILON already relies on. Each component's own nu_0 (for
        whichever of the two is 'ssa') is the exception: it's fit or pinned
        independently against that same shared-alpha regression, per its
        own nu0_slider_1/2 fixed checkbox (see estimate_shape_2comp) -- not
        forced equal to the other component's, since a fixed component's
        own turnover frequency is meaningful chosen input, not just an
        unconstrained initial guess.

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
        shape1, shape2 = self.current_shapes()
        if spec.n_components == 2 and alpha_indices and (shape1 == 'ssa' or shape2 == 'ssa'):
            alpha_init = self.sliders[alpha_indices[0]].value()
            nu0_lo_ghz, nu0_hi_ghz = self.nu0_slider_1.bounds()
            nu0_bounds_hz = (nu0_lo_ghz * 1e9, nu0_hi_ghz * 1e9)
            # Each component's own nu_0 is fit or pinned independently (per
            # its own slider's fixed checkbox), never forced equal -- alpha
            # is still shared, same reasoning FIT_FIXED_EPSILON below
            # relies on for eps (see estimate_shape_2comp). eps itself
            # mirrors the `fixed` dict built below: FIT_FIXED_EPSILON
            # unless the user has pinned the eps slider to their own value.
            # A component whose own shape is 'powerlaw' always evaluates at
            # the shared band nu0 (see models.component_reference_nu), not
            # its own (irrelevant, hidden) nu0 slider -- so its own init is
            # the band edge itself and it's never fit, regardless of that
            # slider's own fixed checkbox.
            eps_idx = spec.indices('eps')[0]
            eps_val = self.sliders[eps_idx].value() if self.sliders[eps_idx].is_fixed() else FIT_FIXED_EPSILON
            band_nu0_hz = float(freq.min())
            nu0_1_init = self.nu0_slider_1.value() * 1e9 if shape1 in ('ssa', 'thermal') else band_nu0_hz
            nu0_2_init = self.nu0_slider_2.value() * 1e9 if shape2 in ('ssa', 'thermal') else band_nu0_hz
            fit_nu0_1 = shape1 == 'ssa' and not self.nu0_slider_1.is_fixed()
            fit_nu0_2 = shape2 == 'ssa' and not self.nu0_slider_2.is_fixed()
            # A 'thermal' component's own T is never fit (see
            # build_temp_slider) -- just its current slider value, passed
            # through so estimate_shape_2comp's internal source_function
            # calls don't crash on a missing T (see thermal_source).
            T1 = self.temp_slider_1.value() if shape1 == 'thermal' else None
            T2 = self.temp_slider_2.value() if shape2 == 'thermal' else None
            # Same reasoning for a 'logparabola' component's own beta --
            # never fit (see build_beta_slider), just passed through so
            # estimate_shape_2comp's internal source_function calls don't
            # crash on a missing beta.
            beta1 = self.beta_slider_1.value() if shape1 == 'logparabola' else None
            beta2 = self.beta_slider_2.value() if shape2 == 'logparabola' else None
            alpha_est, nu0_1_est_ghz, nu0_2_est_ghz = estimate_shape_2comp(
                freq, I, eps_val, nu0_1_init, nu0_2_init, alpha_init,
                nu0_bounds_hz, fit_nu0_1, fit_nu0_2, shape1, shape2,
                T1=T1, T2=T2, beta1=beta1, beta2=beta2)
            if shape1 == 'ssa':
                self.nu0_slider_1.set_value(nu0_1_est_ghz / 1e9)
            if shape2 == 'ssa':
                self.nu0_slider_2.set_value(nu0_2_est_ghz / 1e9)
            self.sync_spectrum_ui()
        elif spec.n_components == 1 and shape1 == 'ssa' and alpha_indices:
            alpha_init = self.sliders[alpha_indices[0]].value()
            nu0_lo_ghz, nu0_hi_ghz = self.nu0_slider_1.bounds()
            nu0_bounds_hz = (nu0_lo_ghz * 1e9, nu0_hi_ghz * 1e9)
            alpha_est, nu0_est_ghz = estimate_ssa_shape(
                freq, I, self.nu0_slider_1.value() * 1e9, alpha_init,
                nu0_bounds_hz, not self.nu0_slider_1.is_fixed())
            self.nu0_slider_1.set_value(nu0_est_ghz / 1e9)
            self.sync_spectrum_ui()
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
        box's own state (each component's own spectral shape, its nu_0
        under 'ssa'/'thermal', T under 'thermal', and beta under
        'logparabola') -- that state lives outside spec.params/
        self.sliders (see build_nu0_slider/build_temp_slider/
        build_beta_slider), so it isn't already covered by `parameters`
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
        shape1, shape2 = self.current_shapes()
        data = {
            'model': func.__name__,
            'parameters': dict(zip(spec.param_names, pars)),
            'spectral_shape': shape1,
        }
        if spec.n_components == 2:
            data['spectral_shape_2'] = shape2
        if shape1 in ('ssa', 'thermal'):
            data['nu0_ghz'] = self.nu0_slider_1.value()
        if spec.n_components == 2 and shape2 in ('ssa', 'thermal'):
            data['nu0_2_ghz'] = self.nu0_slider_2.value()
        if shape1 == 'thermal':
            data['temp_k'] = self.temp_slider_1.value()
        if spec.n_components == 2 and shape2 == 'thermal':
            data['temp_2_k'] = self.temp_slider_2.value()
        if shape1 == 'logparabola':
            data['beta'] = self.beta_slider_1.value()
        if spec.n_components == 2 and shape2 == 'logparabola':
            data['beta_2'] = self.beta_slider_2.value()
        if func in CUSTOM_MODEL_DEFS:
            # eps(z)/phi(z)/bounds needed to rebuild+re-register this exact
            # model in a fresh session, where func.__name__ won't already be
            # in MODELS_BY_NAME -- see load_model_action.
            data['custom_definition'] = CUSTOM_MODEL_DEFS[func]
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
        shape1, shape2 = self.current_shapes()
        try:
            with open(path, 'w') as f:
                f.write(full_equation(spec, shape1, shape2) + '\n')
        except OSError as e:
            QMessageBox.warning(self, 'Save Equation', f'Could not save equation file:\n{e}')

    def load_model_action(self):
        """Read a model + parameters (+ per-component spectral
        shape/nu_0/T/beta) JSON file (as written by save_model_action) and
        apply it: switch the Spectrum box's shape dropdown(s) and
        nu_0/T/beta slider(s), switch the model dropdown, then move each
        slider to the saved value.

        `spectral_shape` defaults to 'powerlaw' (and nu_0/T/beta are left
        alone) for files saved before this was tracked, so older Save
        Model output still loads the same as it always did;
        `spectral_shape_2` similarly defaults to `spectral_shape` for
        files saved before the two components could have independent
        shapes."""
        # getOpenFileName is the read-side counterpart of getSaveFileName.
        path, _ = QFileDialog.getOpenFileName(self, 'Load Model', '', 'JSON files (*.json)')
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            model_name = data['model']
            if model_name not in MODELS_BY_NAME and 'custom_definition' in data:
                # This session has never seen this custom model (see
                # save_model_action) -- rebuild and register it under the
                # same name it was saved with, then insert it into the
                # dropdown just like open_custom_model_dialog does for a
                # freshly-built one.
                cd = data['custom_definition']
                # j_lo/j_hi/p_lo/p_hi replaced the old single-sided z1/z2
                # bounds -- a file saved before that change only has
                # z1/z2, so fall back to the closest equivalent under the
                # new scheme (z1 was always emiss's upper bound with an
                # implicit 0 lower bound; z2 was always phi''s lower bound
                # with an implicit 1 upper bound) rather than silently
                # reverting to the new all-purpose (0, 1) defaults.
                func = build_custom_model(
                    cd['label'], cd['emiss_expr'], cd['phi_expr'],
                    {n: tuple(v) for n, v in cd['param_specs'].items()},
                    j_lo=cd.get('j_lo', 0.0), j_hi=cd.get('j_hi', cd.get('z1', 1.0)),
                    p_lo=cd.get('p_lo', cd.get('z2', 0.0)), p_hi=cd.get('p_hi', 1.0),
                    title=cd.get('title'), name=model_name)
                insert_at = self.model_combo.count() - 1  # just above the sentinel
                self.model_combo.insertItem(insert_at, MODELS[func].label, func)
            func = MODELS_BY_NAME[model_name]
            params = data['parameters']
            shape1 = data.get('spectral_shape', 'powerlaw')
            shape2 = data.get('spectral_shape_2', shape1)
            nu0_ghz = data.get('nu0_ghz')
            nu0_2_ghz = data.get('nu0_2_ghz', nu0_ghz)
            temp_k = data.get('temp_k')
            temp_2_k = data.get('temp_2_k', temp_k)
            beta = data.get('beta')
            beta_2 = data.get('beta_2', beta)
        except (OSError, ValueError, KeyError, CustomModelError) as e:
            QMessageBox.warning(self, 'Load Model', f'Could not load model file:\n{e}')
            return

        # Set the Spectrum box's own state first -- rebuild_sliders below
        # (triggered by the model switch) re-syncs it from whatever's
        # currently selected (see sync_spectrum_ui), so shape/nu_0/T need
        # to already be in place before that runs. Both the single-component
        # dropdown and each two-component dropdown are set from the same
        # saved shape1/shape2 -- current_shapes() only ever reads whichever
        # pair matches the loaded model's own component count, so setting
        # the other is harmless (and keeps a later model switch's own
        # starting point sane).
        for combo, shape in ((self.shape_combo, shape1), (self.shape_combo_1, shape1),
                              (self.shape_combo_2, shape2)):
            idx = combo.findData(shape)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        if nu0_ghz is not None:
            self.nu0_slider_1.set_value(nu0_ghz)
        if nu0_2_ghz is not None:
            self.nu0_slider_2.set_value(nu0_2_ghz)
        if temp_k is not None:
            self.temp_slider_1.set_value(temp_k)
        if temp_2_k is not None:
            self.temp_slider_2.set_value(temp_2_k)
        if beta is not None:
            self.beta_slider_1.set_value(beta)
        if beta_2 is not None:
            self.beta_slider_2.set_value(beta_2)

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
        self.rmsynth_data_button.setEnabled(True)
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


    def show_custom_model_help(self):
        if getattr(self, 'custom_model_dialog', None) is None:
            tex_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'help', 'Custom_models.tex')
            self.custom_dialog = TexViewerDialog(
                'Custom models', tex_path, parent=self)
        self.custom_dialog.show()
        self.custom_dialog.raise_()
        self.custom_dialog.activateWindow()


    def show_qufitting_help(self):
        if getattr(self, 'qufit_dialog', None) is None:
            tex_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'help', 'QU-fitting.tex')
            self.qufit_dialog = TexViewerDialog(
                'QU-fitting', tex_path, parent=self)
        self.qufit_dialog.show()
        self.qufit_dialog.raise_()
        self.qufit_dialog.activateWindow()


    def show_rmsynth_help(self):
        if getattr(self, 'rmsynth_dialog', None) is None:
            tex_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'help', 'RM-synthesis.tex')
            self.rmsynth_dialog = TexViewerDialog(
                'RM-synthesis', tex_path, parent=self)
        self.rmsynth_dialog.show()
        self.rmsynth_dialog.raise_()
        self.rmsynth_dialog.activateWindow()


    def show_about(self):
        # QMessageBox.about is a simple info dialog (no warning icon/buttons
        # beyond OK) -- Help > About Polvista.
        box = QMessageBox(self)
        box.setWindowTitle('About Polvista')
        box.setTextFormat(Qt.RichText)
        box.setText('<b>Polvista: POLarization VISualizer Tool for Astronomy</b><br><br>'
                    'An interactive tool to explore the Faraday effect and maybe even fit or simulate some spectropolarimetric data!<br><br>'
                    'Author: Douglas Carlos, 2026 <a href="https://github.com/medoug7/polvista">GitHub</a>')

        box.findChild(QLabel, 'qt_msgbox_label').setOpenExternalLinks(True)
        box.exec_()



def main():
    # Every PyQt5 app needs exactly one QApplication, created before any
    # widgets. app.exec_() starts Qt's event loop (handling clicks, redraws,
    # etc.) and blocks until the window is closed, at which point it
    # returns an exit code that sys.exit() passes back to the OS.
    app = QApplication(sys.argv)
    # Every numeric spin box in this app (wavelength range, noise levels,
    # MultiNest efficiency/tolerance, the custom-model builder's
    # preview-lambda, ...) expects '.' as the decimal point, matching how
    # saved model/measurement files and the expression fields (including
    # the custom-model builder's own j_lo/j_hi/p_lo/p_hi bound boxes,
    # plain expression QLineEdits with no QDoubleValidator of their own)
    # already treat it -- but QDoubleSpinBox's built-in text validation
    # otherwise follows QLocale::system() by
    # default, which on a system whose locale uses ',' (e.g. pt_BR)
    # silently rejects '.' keystrokes instead of accepting them. Setting
    # the whole application's default locale once here (any QLocale-aware
    # widget not given its own explicit setLocale() falls back to this)
    # fixes every one of them at once, present and future, instead of
    # having to set it widget-by-widget at each call site.
    QLocale.setDefault(QLocale(QLocale.C))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
