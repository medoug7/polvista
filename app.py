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

    The "Sampling" tab offers a basic interface to perform bayesian
    fitting/sampling via pymultinest (an optional dependency -- Fit! on
    that tab reports an error there if it isn't installed, rather than
    hiding the tab). Once a fitting is complete, we can then check the
    parmeter's posterior probability density distributions.

"""
import os
import sys


def ensure_qt_plugin_path():
    """Point Qt at the platform-plugins dir under sys.prefix (conda's
    qt-main package layout) when the active PyQt5 wheel's own Qt5/plugins
    dir is missing one, so Qt doesn't fail to find e.g. the xcb plugin."""
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
import traceback

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from PyQt5.QtCore import Qt, pyqtSignal, QLocale, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QComboBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTabWidget, QPushButton, QProgressBar,
    QFileDialog, QMessageBox, QAction, QLineEdit, QColorDialog, QGridLayout,
    QDialog, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QColor
from matplotlib.lines import Line2D

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from polvista.models import (
    spectral_state, MODELS, MODELS_BY_NAME, C, Param, stokes_I, stokes_QU, evpa, pol,
    set_spectral_shape, get_spectral_state, full_equation, build_custom_model, CUSTOM_MODEL_DEFS,
    CustomModelError, SPECTRAL_SHAPES, TEMP_BOUNDS_K, BETA_BOUNDS,
    SCREENS, DEFAULT_SCREEN, get_composite, is_composite, get_model_by_name, LEGACY_MODELS, legacy_dphi_scale)
from polvista.build_model import CustomModel
from polvista.fitting import (
    qu_fit, estimate_alpha, estimate_spectral_shape, estimate_shape_2comp, fit_statistics,
    multinest_fit, load_previous_run, curve_statistics, expand_pars_errs)
from polvista.latex_stuff import latex_pixmap, fit_equation_pixmap, TexViewerDialog, comparison_tex_table
from polvista.widgets import (
    ValueLineEdit, NUMBER_RE, SLIDER_STEPS, UNITS, WIDEST_UNIT, ModelPlot, StokesPlot, RMSynthPlot,
    CURVE_LINESTYLES, WL_UNITS_M, set_suptitle, set_drag_mode, RESIZE_SETTLE_MS,
    PLOT_PREFS, DEFAULT_PLOT_PREFS, COMPARISON_PALETTES, apply_plot_prefs, DragHandle, CurveDropArea, sync_legend_visibility)
from polvista.sampling import SamplingMixin, warm_up_sampling_imports
from polvista.measurements import MeasurementsMixin, DEFAULT_BANDS_BY_PRESET
from polvista.rm_synthesis import compute_faraday_spectrum, phi_axis_half_width, FaradaySpectrumError

# pymultinest (Bayesian/nested-sampling fitting) is an optional dependency --
# it's a compiled Fortran library wrapper that's a much heavier install than
# the rest of polvista's requirements, so the app must still run with plain
# least-squares fitting when it isn't present. The Sampling tab is always
# built (see MainWindow); fitting.py only imports pymultinest lazily, inside
# multinest_fit/load_previous_run themselves, and sampling.py turns a failure
# there (including the SystemExit pymultinest itself raises when its compiled
# libmultinest.so isn't on LD_LIBRARY_PATH) into an on-screen error rather
# than letting it kill the app.


# QU-fitting holds epsilon fixed at this value (see MainWindow.run_fit) --
# q, u data alone can't constrain it (see spectral_weights).
FIT_FIXED_EPSILON = 0.5

def fixed_epsilons(spec):
    """{param index: value} holding a weighted model's epsilons at the
    equal-weight split (FIT_FIXED_EPSILON for two components; for n, the
    stick-breaking values 1/n, 1/(n-1), ..., 1/2 giving every component the
    same weight at its reference frequency)."""
    eps = spec.indices('eps')
    n = len(eps) + 1
    return {i: 1.0 / (n - k) for k, i in enumerate(eps)}


# Figure default parameters
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 20
##############################################


# model_combo's "Custom model..." entry -- a distinct sentinel object (not a
# real model function) so on_model_combo_changed can tell it apart from
# MODELS_BY_NAME/MODELS' own keys and open the builder dialog instead of
# treating it as a model to plot; see open_custom_model_dialog.
CUSTOM_MODEL_SENTINEL = object()

# request_update_plot's redraw throttle: the first event in a burst redraws
# immediately, further events before the timer elapses just mark the plot
# dirty and coalesce into one redraw when it fires -- otherwise a slider
# drag's near-every-pixel valueChanged rate would queue up more full model
# evaluations (esp. a custom model's LOS quadrature) than the eye could
# ever perceive anyway. ~30 Hz is comfortably smooth.
PLOT_THROTTLE_MS = 33

# wl_ext's own point count, capped to this while a slider is actively being
# dragged (see on_slider_drag_started/_ended) so a custom model's LOS-
# quadrature-per-wavelength cost stays responsive mid-drag; the drag's
# final tick (sliderReleased) always forces one full-resolution redraw.
DRAG_N_POINTS_CAP = 150

# Log-decade ranges for the signed Faraday-depth (phi) and turbulent term (dphi)
# sliders -- wide enough to explore the models' full behavior without
# being tied to any particular model's fit-prior bounds.
PHI_LOG_RANGE = (-6.5, 6.5)
DPHI_LOG_RANGE = (0.0, 6.5)

# Placeholder q_err/u_err for rmsynth_from_model's noiseless model curve.
# compute_faraday_spectrum only uses q_err/u_err in ratios (Briggs relative
# weighting) or as an overall scale (sigma_fdf/CLEAN threshold) -- any
# constant gives equal per-point weighting, but its absolute size still
# sets sigma_fdf's size, so this is small relative to a typical
# fractional-polarization curve's amplitude rather than ~1 (which made the
# plotted noise line/CLEAN threshold swamp the curve despite no real noise).
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
# Full SPHERE/IRDIS NIR spans ESO's quoted 0.95-2.32 um IRDIS range, padded
# to 2.35 um so the Ks band's own upper edge (2.332 um) fits -- its bands are
# the 4 broadband filters offered for dual-polarization imaging (de Boer et
# al. 2020), see measurements.IRDIS_ALL_BANDS.
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
    ('SKA-Mid (2 - 86 cm / 350 MHz - 15 GHz)', 20, 860.0),
    ('SKA-Low (86 cm - 6 m / 50 MHz - 350 MHz)', 860, 6000),
    ('Full radio (0.25mm - 30 m / 500 MHz - 1200 GHz)', 0.25, 30000),
    ('Full HAWC+ FIR (40 - 250 um / 1.2 - 7.5 THz)', 0.04, 0.25),
    ('Full SPHERE/IRDIS NIR (0.95 - 2.35 um / 128 - 316 THz)', 9.5e-4, 2.35e-3),
    ('Full SPARC4 optical (400 - 1000 nm / 300 - 750 THz)', 4.0e-4, 1.0e-3),
]

# Cheap consistency check: every DEFAULT_BANDS_BY_PRESET key is hand-typed
# to match one of these preset labels exactly (see that dict's own module
# comment) -- catch a typo/renamed preset here at import time rather than
# have it silently fall back to a generic single-band default.
_WAVELENGTH_PRESET_LABELS = {label for label, _lo, _hi in WAVELENGTH_PRESETS}
assert set(DEFAULT_BANDS_BY_PRESET) <= _WAVELENGTH_PRESET_LABELS, (
    f'DEFAULT_BANDS_BY_PRESET has key(s) not in WAVELENGTH_PRESETS: '
    f'{set(DEFAULT_BANDS_BY_PRESET) - _WAVELENGTH_PRESET_LABELS}')


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
    """Multiplier converting a frequency column's values to Hz, inferred
    from a unit annotation in its header (e.g. 'Frequency [GHz]',
    'freq_mhz') -- this app's own exports disagree on units (Save Spectra
    writes Hz, Export Measurements writes GHz), so Load Data can't assume
    one. Checked most-specific-first ('hz' is a substring of 'ghz' etc.);
    defaults to Hz when no unit is recognized."""
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


def load_vapola_csv(path):
    """Parse a Stokes I/Q/U-vs-frequency CSV into (freq [Hz], I, Q, U,
    I_err, Q_err, U_err), sorted by frequency -- error arrays are None
    where the CSV has no matching column. Raises ValueError on a missing
    required column or an empty file. Shared by load_data_action and
    sampling.SamplingMixin's own qu_fit.py sample import
    (_load_qu_fit_samples)."""
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

    order = np.argsort(freq)
    freq, I, Q, U = freq[order], I[order], Q[order], U[order]
    if I_err is not None:
        I_err = I_err[order]
    if Q_err is not None:
        Q_err = Q_err[order]
    if U_err is not None:
        U_err = U_err[order]
    return freq, I, Q, U, I_err, Q_err, U_err


def fractional_qu(freq, I, Q, U, I_err, Q_err, U_err):
    """(wl, q, q_err, u, u_err) -- the quantities Fitting compares against
    (see MainWindow.fit_data) -- derived from load_vapola_csv()'s raw
    columns: fractional q=Q/I, u=U/I (per-point I, so unaffected by any
    separate I0 normalization a caller applies for display). Falls back to
    uniform weighting only if no error column at all was present."""
    wl = C / freq  # m
    q, u = Q / I, U / I
    if Q_err is None and U_err is None and I_err is None:
        return wl, q, 1e-3*np.ones_like(q), u, 1e-3*np.ones_like(u)

    def frac_err(val, val_err, denom, denom_err):
        val_err = val_err if val_err is not None else 0.0
        denom_err = denom_err if denom_err is not None else 0.0
        return np.sqrt((val_err / denom) ** 2 + (val * denom_err / denom ** 2) ** 2)
    q_err = frac_err(Q, Q_err, I, I_err)
    u_err = frac_err(U, U_err, I, I_err)
    return wl, q, q_err, u, u_err


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


class PreferencesDialog(QDialog):
    """Small non-modal window for figure preferences (Preferences menu):
    curves, markers, font sizes and grid. Changes apply live to every canvas
    through MainWindow.apply_plot_preferences, which stores them in
    PLOT_PREFS; Restore puts DEFAULT_PLOT_PREFS back."""

    FONT_ROWS = (('title', 'Plot title'), ('label', 'Axis label'),
                 ('tick', 'Tick labels'), ('legend', 'Legend'))
    GRID_WHICH = (('both', 'Both'), ('major', 'Major'), ('none', 'None'))
    GRID_STYLES = (('solid', 'Solid'), ('dashed', 'Dashed'), ('dotted', 'Dotted'))
    XTICK_FORMATS = (('plain', 'Plain'), ('sci', 'Scientific notation'))

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self._refreshers = {}  # PLOT_PREFS key -> callable(value) updating its widget silently
        self.setWindowTitle('Preferences')
        layout = QVBoxLayout(self)

        curve_box = QGroupBox('Curve')
        curve_grid = QGridLayout(curve_box)
        curve_grid.addWidget(QLabel('Linewidth'), 0, 0)
        curve_grid.addWidget(self._spin('curve_width', 0.5, 10.0, 0.5), 0, 1)
        curve_grid.addWidget(QLabel('Linestyle'), 1, 0)
        style_combo = QComboBox()
        style_combo.addItems(list(CURVE_LINESTYLES))
        style_combo.currentTextChanged.connect(lambda t: self.main.apply_plot_preferences(curve_style=t))
        self._register('curve_style', style_combo, style_combo.setCurrentText)
        curve_grid.addWidget(style_combo, 1, 1)
        curve_grid.addWidget(QLabel('Color scheme'), 2, 0)
        scheme_row = QHBoxLayout()
        scheme_buttons = {}
        for value, text in (('default', 'Default'), ('custom', '')):
            button = QRadioButton(text)
            button.toggled.connect(lambda on, v=value: on and self.main.apply_plot_preferences(curve_scheme=v))
            scheme_row.addWidget(button)
            scheme_buttons[value] = button
        scheme_group = QButtonGroup(self)
        for button in scheme_buttons.values():
            scheme_group.addButton(button)
        scheme_row.addWidget(self._color_button('curve_color', curve_scheme='custom'))
        scheme_row.addStretch(1)
        curve_grid.addLayout(scheme_row, 2, 1)
        curve_grid.addWidget(QLabel('Comparison'), 3, 0)
        palette_combo = QComboBox()
        palette_combo.addItems(list(COMPARISON_PALETTES))
        palette_combo.setToolTip('Color palette of the curves added to the Comparison tab (applies to new curves).')
        palette_combo.currentTextChanged.connect(lambda t: self.main.apply_plot_preferences(comparison_palette=t))
        self._register('comparison_palette', palette_combo, palette_combo.setCurrentText)
        curve_grid.addWidget(palette_combo, 3, 1)
        curve_grid.addWidget(QLabel('Legend side'), 4, 0)
        side_row = QHBoxLayout()
        side_group = QButtonGroup(self)
        side_buttons = {}
        for value, text in (('p', 'p'), ('x', 'χ')):
            button = QRadioButton(text)
            button.toggled.connect(lambda on, v=value: on and self.main.apply_plot_preferences(legend_side=v))
            side_group.addButton(button)
            side_row.addWidget(button)
            side_buttons[value] = button
        side_row.addStretch(1)
        self._register('legend_side', side_group, lambda v: side_buttons[v].setChecked(True))
        curve_grid.addLayout(side_row, 4, 1)
        self._register('curve_scheme', scheme_group, lambda v: scheme_buttons[v].setChecked(True))
        layout.addWidget(curve_box)

        marker_box = QGroupBox('Markers')
        marker_grid = QGridLayout(marker_box)
        marker_grid.addWidget(QLabel('Format'), 0, 0)
        format_combo = QComboBox()
        for symbol, name in Line2D.markers.items():
            if isinstance(symbol, str) and symbol.strip() and symbol != 'None' and name != 'nothing':
                format_combo.addItem(f'{name} ({symbol})', symbol)
        format_combo.currentIndexChanged.connect(
            lambda i, c=format_combo: self.main.apply_plot_preferences(marker=c.itemData(i)))
        self._register('marker', format_combo, lambda v, c=format_combo: c.setCurrentIndex(max(0, c.findData(v))))
        marker_grid.addWidget(format_combo, 0, 1, 1, 2)
        marker_grid.addWidget(QLabel('Fill'), 1, 0)
        size_spin = self._spin('marker_size', 1, 30, 1, decimals=0)
        size_spin.setToolTip('Marker size')
        marker_grid.addWidget(size_spin, 1, 1)
        marker_grid.addWidget(self._color_button('marker_fill'), 1, 2)
        marker_grid.addWidget(QLabel('Contour'), 2, 0)
        edge_spin = self._spin('marker_edge_width', 0.0, 10.0, 0.1)
        edge_spin.setToolTip('Width of the marker contour and of the error lines (both take the contour color)')
        marker_grid.addWidget(edge_spin, 2, 1)
        marker_grid.addWidget(self._color_button('marker_edge_color'), 2, 2)
        layout.addWidget(marker_box)

        font_box = QGroupBox('Fontsize')
        font_grid = QGridLayout(font_box)
        for row, (key, text) in enumerate(self.FONT_ROWS):
            font_grid.addWidget(QLabel(text), row, 0)
            font_grid.addWidget(self._spin(key, 4, 60, 1, decimals=0), row, 1)
            font_grid.addWidget(self._color_button(key + '_color'), row, 2)
        layout.addWidget(font_box)

        grid_box = QGroupBox('Grid')
        grid_layout = QGridLayout(grid_box)
        grid_layout.addWidget(QLabel('Linewidth'), 0, 0)
        grid_layout.addWidget(self._spin('grid_width', 0.1, 10.0, 0.1), 0, 1)
        for row, (key, text, options) in enumerate((
                ('grid_which', 'Axis', self.GRID_WHICH), ('grid_style', 'Linestyle', self.GRID_STYLES),
                ('xtick_format', 'x-axis', self.XTICK_FORMATS)),
                start=1):
            grid_layout.addWidget(QLabel(text), row, 0)
            group = QButtonGroup(self)
            buttons = {}
            for col, (value, name) in enumerate(options, start=1):
                button = QRadioButton(name)
                button.toggled.connect(lambda on, k=key, v=value: on and self.main.apply_plot_preferences(**{k: v}))
                group.addButton(button)
                grid_layout.addWidget(button, row, col)
                buttons[value] = button

            def refresh(value, group=group, buttons=buttons):
                group.blockSignals(True)
                for button in buttons.values():
                    button.blockSignals(True)
                buttons[value].setChecked(True)
                for button in buttons.values():
                    button.blockSignals(False)
                group.blockSignals(False)
            self._refreshers[key] = refresh
        layout.addWidget(grid_box)

        bands_box = QGroupBox('Bands')
        bands_row = QHBoxLayout(bands_box)
        bands_buttons = {}
        bands_group = QButtonGroup(self)
        for value, text in ((True, 'On'), (False, 'Off')):
            button = QRadioButton(text)
            button.toggled.connect(lambda on, v=value: on and self.main.apply_plot_preferences(bands_on=v))
            bands_group.addButton(button)
            bands_row.addWidget(button)
            bands_buttons[value] = button
        self._register('bands_on', bands_group, lambda v: bands_buttons[v].setChecked(True))
        bands_row.addStretch(1)
        bands_row.addWidget(QLabel('Alpha'))
        alpha_spin = self._spin('bands_alpha', 0.0, 1.0, 0.05, decimals=2)
        alpha_spin.setToolTip('Opacity of the shaded Measurements-tab bands')
        bands_row.addWidget(alpha_spin)
        layout.addWidget(bands_box)

        restore_button = QPushButton('Restore')
        restore_button.setToolTip('Reset every figure preference to its default.')
        restore_button.clicked.connect(self.restore_defaults)
        restore_button.setAutoDefault(False)  # Enter must not trigger Restore or a color box
        ok_button = QPushButton('Ok')
        ok_button.setDefault(True)  # Enter closes the window
        ok_button.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(restore_button)
        bottom.addStretch(1)
        bottom.addWidget(ok_button)
        layout.addLayout(bottom)

        self.refresh_widgets()
        self.setFixedSize(self.sizeHint())

    def _register(self, key, widget, setter):
        def refresh(value):
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)
        self._refreshers[key] = refresh

    def _spin(self, key, lo, hi, step, decimals=1):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(lambda v: self.main.apply_plot_preferences(**{key: v}))
        self._register(key, spin, spin.setValue)
        return spin

    def _color_button(self, key, **also):
        button = QPushButton()
        button.setFixedSize(26, 26)
        button.setAutoDefault(False)  # otherwise Enter in the dialog opens this color picker

        def pick():
            color = QColorDialog.getColor(QColor(PLOT_PREFS[key]), self, 'Color')
            if color.isValid():
                button.setStyleSheet(f"background-color: {color.name()};")
                self.main.apply_plot_preferences(**{key: color.name()}, **also)
                self.refresh_widgets()  # `also` may flip another control (e.g. the color scheme)
        button.clicked.connect(pick)
        self._register(key, button, lambda v: button.setStyleSheet(f"background-color: {v};"))
        return button

    def refresh_widgets(self):
        for key, refresh in self._refreshers.items():
            refresh(PLOT_PREFS[key])

    def restore_defaults(self):
        self.main.apply_plot_preferences(**DEFAULT_PLOT_PREFS)
        self.refresh_widgets()


class MainWindow(QMainWindow, SamplingMixin, MeasurementsMixin):
    NO_FIT_TEXT = 'Fit some data to get statistics'

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Polvista: POLarization VISualizer Tool for Astronomy')
        self.resize(1500, 700) # size of the window

        self.sliders = []
        # Curves frozen by the Add curve button, oldest first -- shared by
        # reference with the p/EVPA and Stokes canvases (see __init__) and
        # listed in the Comparison tab. Each is a dict: id, func,
        # n_components, pars, spectral (models.get_spectral_state snapshot),
        # name, color ('#rrggbb'), linestyle (a CURVE_LINESTYLES key).
        self.curves = []
        self._next_curve_id = 0
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
        self.ls_worker = None      # background worker for load_samples_action, see LoadSamplesWorker
        self.corner_build_worker = None  # background worker for build_corner_tab, see CornerBuildWorker
        self.mn_model = None       # model func the most recent MultiNest run/load was for
        self.posterior_samples = None
        self.posterior_model = None
        self.corner_tab = None
        # Corner tab's own family-picker dropdown state -- see
        # SamplingMixin.build_corner_tab/on_corner_family_selected.
        self._corner_info = None
        self._corner_families = None
        # selected_idx -> already-rendered Figure, for the current
        # _corner_info's own result only -- lets switching the family
        # picker back to an already-seen family reuse that Figure instead
        # of re-running CornerBuildWorker from scratch. Reset whenever
        # _corner_info itself changes (a fresh fit/Load samples, not just a
        # family switch) -- see build_corner_tab.
        self._corner_fig_cache = {}

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

        self._build_left_panel(root)
        self._build_right_panel(root)
        self.canvas.perm_curves = self.curves
        self.stokes_canvas.perm_curves = self.curves

        self.build_menu()
        self.rebuild_sliders()
        self._ui_ready = True  # see MeasurementsMixin.on_band_edited

    def _build_left_panel(self, root):
        """Left control panel: wavelength range, model/spectrum choice,
        parameter sliders (Visualization/Measurements/Sampling tabs), and
        the Reset/Load/Clear/Fit! row + Results box. Split out of
        __init__ purely for readability -- see MainWindow's own docstring."""
        # ── Left: model choice, wavelength range, parameter sliders ──────
        # QVBoxLayout stacks its children top-to-bottom -- this is the
        # left-hand control panel's layout.
        left = QWidget()
        left.setMaximumWidth(440)
        left_layout = QVBoxLayout(left)

        # set wavelength/frequency range
        # QGroupBox draws a titled border/frame around its contents --
        # used throughout this panel to visually group related controls.
        wl_box = QGroupBox('Wavelength range')
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
        # Unit the min/max boxes above are read in (see wl_unit_m); the
        # presets themselves are stored in mm and converted when applied.
        self.wl_unit_combo = QComboBox()
        self.wl_unit_combo.addItems(list(WL_UNITS_M))
        self.wl_unit_combo.setCurrentText('mm')  # the presets/limits below start out in mm
        self._wl_unit_prev = self.wl_unit_combo.currentText()  # unit the min/max values are currently in
        self.wl_unit_combo.setToolTip('Unit of the min/max wavelengths')
        self.wl_unit_combo.currentTextChanged.connect(self.on_wl_unit_changed)
        log_row = QHBoxLayout()
        log_row.addWidget(self.log_xscale)
        log_row.addStretch(1)
        log_row.addWidget(QLabel('units'))
        log_row.addWidget(self.wl_unit_combo)
        wl_box_layout.addLayout(log_row)

        left_layout.addWidget(wl_box)

        left_layout.addWidget(QLabel('<b>Model</b>'))
        # QComboBox is the dropdown; addItem's 2nd arg is arbitrary data
        # (here the model function) retrievable later via currentData().
        self.model_combo = QComboBox()
        for func, spec in MODELS.items():
            if spec.hidden or is_composite(func):
                continue
            self.model_combo.addItem(spec.label, func)
        # One dropdown row per multi-component family. The row's own data is
        # the composite currently chosen for it (model_combo.currentData()
        # stays "the model function" everywhere); picking each component's
        # screen type or the combination scheme (see on_composite_changed)
        # swaps that data for the matching composite, built on demand.
        self.composite_state = {n: [DEFAULT_SCREEN] * n for n in (2, 3)}  # screen types, shared by both modes
        for n in (2, 3):
            for mode, word in (('class', 'classic'), ('weight', 'weighted')):
                self.model_combo.addItem(f'{n}-component model ({word})', self.composite_func(mode, n))
        # "Custom model..." always stays the last row (see
        # open_custom_model_dialog, which inserts newly-built custom models
        # just above it) -- picking it doesn't select a model itself, it
        # opens the builder dialog instead (see on_model_combo_changed).
        self.model_combo.addItem('Custom model...', CUSTOM_MODEL_SENTINEL)
        self._last_model_index = 0
        self._slider_names = []  # param names of the sliders currently shown, see rebuild_sliders
        self._eq_resize_timer = None  # see resizeEvent
        self._syncing_composite = False  # guards on_composite_changed while rebuild_sliders sets the combos
        # Qt's signal/slot mechanism: connect() wires the combo box's
        # "selection changed" signal to a method that runs whenever the
        # user picks a different model.
        self.model_combo.currentIndexChanged.connect(self.on_model_combo_changed)
        left_layout.addWidget(self.model_combo)

        # Per-component screen-type headers, placed at the top of each
        # component's own parameters (see rebuild_sliders).
        self.comp_type_headers, self.comp_type_combos = [], []
        for k in (1, 2, 3):
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 6, 0, 0)
            label = QLabel(f'<b>Component {k}</b>')
            row.addWidget(label)
            row.addStretch(1)
            combo = QComboBox()
            for code, sc in SCREENS.items():
                combo.addItem(sc['label'], code)
            combo.currentIndexChanged.connect(self.on_composite_changed)
            row.addWidget(combo)
            self.comp_type_headers.append(container)
            self.comp_type_combos.append(combo)

        # spectral weighting parameters -- more relevant for two-component models
        # Plain (frameless) container -- spectral_box and pol_box below are
        # stacked inside one shared scroll area/group box (see param_box),
        # so they read as a single box rather than two.
        self.spectral_box = QWidget()
        self.spectral_layout = QVBoxLayout(self.spectral_box)
        self.spectral_layout.setContentsMargins(0, 0, 0, 0)

        # Dropdown(s) selecting the normalized intensity shape I'(nu) used to
        # weight each component's spectral contribution (see
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
        self.comp1_header, self.shape_combo_1 = self.build_spectrum_header('Spectrum')
        self.comp2_header, self.shape_combo_2 = self.build_spectrum_header('Spectrum')
        self.comp3_header, self.shape_combo_3 = self.build_spectrum_header('Spectrum')

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
        self.nu0_slider_3 = self.build_nu0_slider(
            nu0_lo, nu0_hi, default_nu0, r'$\nu_{0,3}$', "Component 3's SSA/thermal turnover frequency.")

        # Electron-temperature T slider(s) -- log-scale (see TEMP_BOUNDS_K),
        # only shown once its own component's shape dropdown is set to
        # 'Thermal' (see sync_spectrum_ui). Same one-or-two-per-model
        # pattern as the nu_0 sliders above.
        temp_lo, temp_hi = TEMP_BOUNDS_K
        self.temp_slider_1 = self.build_temp_slider(
            temp_lo, temp_hi, DEFAULT_TEMP_K, r'$T$', 'Thermal free-free electron temperature.')
        self.temp_slider_2 = self.build_temp_slider(
            temp_lo, temp_hi, DEFAULT_TEMP_K, r'$T_2$', "Component 2's thermal free-free electron temperature.")
        self.temp_slider_3 = self.build_temp_slider(
            temp_lo, temp_hi, DEFAULT_TEMP_K, r'$T_3$', "Component 3's thermal free-free electron temperature.")

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
        self.beta_slider_3 = self.build_beta_slider(
            beta_lo, beta_hi, DEFAULT_BETA, r'$\beta_3$', "Component 3's log-parabola curvature index.")
        # Per-component views of the widgets above, index 0 = component 1
        # (also the single-component model's own shape/nu_0/T/beta).
        self.comp_headers = [self.comp1_header, self.comp2_header, self.comp3_header]
        self.shape_combos = [self.shape_combo_1, self.shape_combo_2, self.shape_combo_3]
        self.nu0_sliders = [self.nu0_slider_1, self.nu0_slider_2, self.nu0_slider_3]
        self.temp_sliders = [self.temp_slider_1, self.temp_slider_2, self.temp_slider_3]
        self.beta_sliders = [self.beta_slider_1, self.beta_slider_2, self.beta_slider_3]

        # The polarization-parameter sliders sit immediately below the
        # spectral ones, in the same box (see param_scroll below).
        self.pol_box = QWidget()
        self.param_layout = QVBoxLayout(self.pol_box)
        self.param_layout.setContentsMargins(0, 0, 0, 0)

        # QScrollArea adds scrollbars around a widget that may not fit its
        # allotted space -- here it lets the (variable-length) list of
        # sliders scroll instead of resizing the window.
        sliders_box = QWidget()
        sliders_layout = QVBoxLayout(sliders_box)
        sliders_layout.addWidget(self.spectral_box)
        sliders_layout.addWidget(self.pol_box)
        sliders_layout.addStretch(1)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setWidget(sliders_box)
        self.param_scroll.setMinimumHeight(290)  # fits its max of 8 rows (2-component models); see rebuild_sliders

        # param_box is just the outer, per-model container -- its title is
        # set to the current model's name in rebuild_sliders.
        self.param_box = QGroupBox()
        param_box_layout = QVBoxLayout(self.param_box)
        param_box_layout.addWidget(self.param_scroll)

        # "Add curve" freezes the current model as a permanent curve on the
        # p/EVPA and I/Q/U plots (see add_curve_action); the box on its left
        # shows (and lets the user override) the name it will be labeled with.
        curve_row = QHBoxLayout()
        self.load_model_button = QPushButton('Load model')
        self.load_model_button.setToolTip("Load a saved model file into the visualizer (same as Import > Load Model).")
        self.load_model_button.clicked.connect(self.load_model_action)
        curve_row.addWidget(self.load_model_button)
        self.curve_name_edit = QLineEdit()
        self.curve_name_edit.setToolTip('Name the next added curve will be labeled with.')
        self.curve_name_edit.setMaximumWidth(150)
        curve_row.addWidget(self.curve_name_edit)
        curve_row.addStretch(1)
        self.clear_curves_button = QPushButton('Clear last')
        self.clear_curves_button.setToolTip('Delete the most recently added curve.')
        self.clear_curves_button.setEnabled(False)
        self.clear_curves_button.clicked.connect(self.clear_last_curve_action)
        curve_row.addWidget(self.clear_curves_button)
        self.add_curve_button = QPushButton('Add curve')
        self.add_curve_button.setToolTip('Keep the current model as a permanent curve in all the plots.')
        self.add_curve_button.clicked.connect(self.add_curve_action)
        curve_row.addWidget(self.add_curve_button)
        param_box_layout.addLayout(curve_row)

        # The parameter sliders live under a "Visualization" tab; the
        # "Sampling" tab (MultiNest fit options) is always built -- whether
        # pymultinest itself is actually usable is only checked once Fit! is
        # pressed there (see run_fit/run_multinest_fit), not before.
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self.param_box, 'Visualization')
        self.build_comparison_tab()
        self.build_measurements_tab()
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
        # shown) -- only run_multinest_fit ever makes them visible.
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

    def _build_right_panel(self, root):
        """Right plot panel: the equation card, custom-model resolution
        warning, and the p/EVPA, Stokes I/Q/U, and RM-synth plot tabs."""
        # ── Right: equation card, toolbar, matplotlib canvas ─────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.eq_scroll = QScrollArea()
        self.eq_scroll.setFixedHeight(130)
        self.eq_scroll.setWidgetResizable(False)
        self.eq_scroll.setAlignment(Qt.AlignCenter)
        # Fixed-height card: refit_equation shrinks a short equation to fit,
        # and fits a tall multi-component one to the width, so it scrolls
        # vertically instead of squeezing the plots.
        self.eq_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.eq_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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
        # Centered suptitle box (blank by default; Load data fills it with
        # the file's name, see load_data_action).
        self.p_title_edit = self.build_title_edit(self.canvas)
        p_evpa_layout.addLayout(self.build_title_row(self.toolbar, self.p_title_edit))
        p_evpa_layout.addWidget(self.canvas, stretch=1)

        self.stokes_canvas = StokesPlot(self)
        self.stokes_toolbar = NavigationToolbar(self.stokes_canvas, self)
        stokes_tab = QWidget()
        stokes_layout = QVBoxLayout(stokes_tab)
        stokes_layout.setContentsMargins(0, 0, 0, 0)

        self.stokes_title_edit = self.build_title_edit(self.stokes_canvas)
        stokes_view = QWidget()
        stokes_view_layout = QHBoxLayout(stokes_view)
        stokes_view_layout.setContentsMargins(0, 0, 0, 0)
        stokes_view_layout.addWidget(QLabel('Q, U view:'))
        self.stokes_mode_combo = QComboBox()
        self.stokes_mode_combo.addItems(['Polar', 'Spectra'])
        self.stokes_mode_combo.currentIndexChanged.connect(self.request_update_plot)
        stokes_view_layout.addWidget(self.stokes_mode_combo)
        stokes_layout.addLayout(self.build_title_row(self.stokes_toolbar, self.stokes_title_edit, stokes_view))

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
        rmsynth_top_row.addWidget(QLabel('robust='))
        self.rmsynth_robust = QDoubleSpinBox()
        # Standard Briggs robust range: +2 -> natural (inverse-noise-
        # variance) weighting, maximizing sensitivity at the cost of a
        # broader RMSF; -2 -> uniform weighting, minimizing RMSF sidelobes/
        # broadening at the cost of sensitivity (see briggs_weights and
        # Heald 2009 Sec. 2.4, which only ever compares those two extremes).
        self.rmsynth_robust.setRange(-2.0, 2.0)
        self.rmsynth_robust.setDecimals(1)
        self.rmsynth_robust.setSingleStep(0.1)
        self.rmsynth_robust.setValue(2.0)
        self.rmsynth_robust.setToolTip(
            "Briggs robust parameter for weighting RM synthesis channels: "
            "+2 = natural weighting (best sensitivity), -2 = uniform "
            "weighting (best resolution/lowest sidelobes).")
        self.rmsynth_robust.valueChanged.connect(self.rerun_rmsynth)
        rmsynth_top_row.addWidget(self.rmsynth_robust)
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
        self.rmsynth_measurements_button.setToolTip("Compute the Faraday spectrum of the simulated measurements.")
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

    def build_title_edit(self, canvas):
        """A line edit that sets `canvas`'s suptitle as it's typed in."""
        edit = QLineEdit()
        edit.setPlaceholderText('Plot title')
        edit.setFixedWidth(260)
        edit.setAlignment(Qt.AlignCenter)
        edit.textChanged.connect(lambda text, c=canvas: self.on_title_edited(c, text))
        return edit

    @staticmethod
    def build_title_row(toolbar, title_edit, right_widget=None):
        """Toolbar on the left, the title box in the true center, and an
        optional widget on the right -- equal-stretch side columns keep the
        box centered whatever the sides' own widths."""
        row = QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setColumnStretch(0, 1)
        row.setColumnStretch(2, 1)
        row.addWidget(toolbar, 0, 0, Qt.AlignLeft)
        row.addWidget(title_edit, 0, 1)
        if right_widget is not None:
            row.addWidget(right_widget, 0, 2, Qt.AlignRight)
        return row

    def on_title_edited(self, canvas, text):
        set_suptitle(canvas.fig, text)
        canvas.draw_idle()

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
        old = dict(zip(self._slider_names, [(sl.value(), sl.is_fixed()) for sl in self.sliders]))
        self.refresh_composite_entries()
        self.rebuild_sliders()
        if is_composite(self.model_combo.currentData()) and old:
            self.restore_values(old)

    def restore_values(self, old):
        """Re-apply {param name: (value, fixed)} to every current slider whose
        name is in `old`."""
        for name, sl in zip(self._slider_names, self.sliders):
            if name in old:
                sl.set_value(old[name][0])
                sl.fix_checkbox.setChecked(old[name][1])

    def refresh_composite_entries(self):
        """Point every multi-component dropdown row at the composite for its
        mode and the family's current (shared) screen types."""
        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i)
            if is_composite(data):
                self.model_combo.setItemData(i, self.composite_func(data.composite['mode'], data.composite['n']))

    def composite_func(self, mode, n):
        """The composite model function for `mode` and the n-component
        family's current screen types (see composite_state)."""
        return get_composite(mode, tuple(self.composite_state[n]))

    def model_index_for(self, func):
        """Dropdown row for `func`, or -1. A composite lives in the row of its
        mode and component count: the shared screen types are set to `func`'s,
        and every composite row refreshed to match."""
        if not is_composite(func):
            return self.model_combo.findData(func)
        self.composite_state[func.composite['n']] = list(func.composite['codes'])
        self.refresh_composite_entries()
        return self.model_combo.findData(func)

    def on_composite_changed(self, *_):
        """A component's screen-type dropdown changed: re-point the family's
        rows at the matching composites (built and registered on demand),
        rebuild the sliders, and carry over every parameter value (and fixed
        flag) whose name survives the change."""
        func = self.model_combo.currentData()
        if not is_composite(func) or self._syncing_composite:
            return
        n = func.composite['n']
        old = dict(zip(self._slider_names, [(sl.value(), sl.is_fixed()) for sl in self.sliders]))
        self.composite_state[n] = [c.currentData() for c in self.comp_type_combos[:n]]
        self.refresh_composite_entries()
        self.rebuild_sliders()
        self.restore_values(old)

    def build_custom_model_menu_action(self):
        """Models menu's "Build Custom Model..." entry -- same dialog as
        the dropdown's sentinel row, just with nothing to revert if the
        user cancels (the combo box selection isn't touched by opening it)."""
        self.open_custom_model_dialog()

    def sync_screen_combos(self):
        """Make every component's screen-type dropdown list exactly SCREENS
        (a custom model is added there when built or loaded, see
        models.register_custom_screen), refreshing the text of one whose
        label changed. Signals are blocked: this only changes the lists, not
        what's selected."""
        for combo in self.comp_type_combos:
            combo.blockSignals(True)
            for code, sc in SCREENS.items():
                i = combo.findData(code)
                if i < 0:
                    combo.addItem(sc['label'], code)
                else:
                    combo.setItemText(i, sc['label'])
            combo.blockSignals(False)

    def register_saved_custom(self, name, cd):
        """Rebuild the custom model saved as `cd` (a 'custom_definition'
        block, see save_model_action) under `name` -- unless this session
        already has that name -- and insert it into model_combo just above
        the sentinel, exactly like open_custom_model_dialog does for a
        freshly-built one. It becomes available as a component screen too
        (see sync_screen_combos)."""
        if name in MODELS_BY_NAME:
            return
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
            title=cd.get('title'), name=name)
        insert_at = self.model_combo.count() - 1  # just above the sentinel
        self.model_combo.insertItem(insert_at, MODELS[func].label, func)
        self.sync_screen_combos()

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
            self.sync_screen_combos()
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
        # Composites using this model were rebuilt around the edited one
        # (see models.register_custom_screen).
        self.sync_screen_combos()
        self.refresh_composite_entries()
        self.rebuild_sliders()

    def rebuild_sliders(self, *_):
        for sl in self.sliders:
            sl.setParent(None)
        self.sliders = []

        func = self.model_combo.currentData()
        spec = MODELS[func]
        lo_bounds, hi_bounds = spec.bounds
        self.edit_custom_model_action.setEnabled(func in CUSTOM_MODEL_DEFS)

        # re-add sliders
        # None of the Spectrum-box header/nu_0 widgets below are one of
        # self.sliders (see _build_ui) so they survive the setParent(None)
        # loop above untouched -- pull them all out of spectral_layout too,
        # so each can be re-inserted below at the spot matching this
        # model's own component count and spectral params (see the loop's
        # comment).
        for w in ([self.shape_header_single] + self.comp_headers + self.nu0_sliders + self.temp_sliders
                  + self.beta_sliders):
            self.spectral_layout.removeWidget(w)
            self.param_layout.removeWidget(w)
        for w in self.comp_type_headers:
            self.param_layout.removeWidget(w)

        # A composite's screen-type dropdowns mirror the function actually
        # selected (e.g. after Load Model or Load samples switched it).
        composite = func.composite if is_composite(func) else None
        if composite is not None:
            self._syncing_composite = True
            try:
                for combo, code in zip(self.comp_type_combos, composite['codes']):
                    combo.setCurrentIndex(combo.findData(code))
            finally:
                self._syncing_composite = False

        multi = spec.n_components >= 2
        self._slider_names = spec.param_names
        for i, param in enumerate(spec.params):
            sl = ParamSlider(param, lo_bounds[i], hi_bounds[i])
            sl.valueChanged.connect(self.request_update_plot)
            sl.slider.sliderPressed.connect(self.on_slider_drag_started)
            sl.slider.sliderReleased.connect(self.on_slider_drag_ended)
            # self.sliders keeps the *original* spec.params order regardless of
            # which box a slider is displayed in -- update_plot() reads this
            # list positionally, matching the model function's param order.
            self.sliders.append(sl)

        # Layout. A weighted multi-component model shows only its epsilons in
        # the top (spectral) box; each component's own block (screen-type
        # dropdown, its spectrum: shape, nu_0/T, alpha, beta, then its polarization
        # sliders) sits in the parameter box. Any other model keeps its
        # alpha(s) under a "Spectrum" header in the top box, ahead of the
        # polarization sliders.
        has_spectral = False
        eps_idx = spec.indices('eps')
        alpha_idx = spec.indices('alpha')
        shown_type_headers = set()
        if not multi:
            self.spectral_layout.addWidget(self.shape_header_single)
        for i in eps_idx:
            self.spectral_layout.addWidget(self.sliders[i])
            has_spectral = True
        if not multi:
            for k, i in enumerate(alpha_idx):
                has_spectral = True
                self.spectral_layout.addWidget(self.nu0_sliders[k])
                self.spectral_layout.addWidget(self.temp_sliders[k])
                self.spectral_layout.addWidget(self.sliders[i])
                self.spectral_layout.addWidget(self.beta_sliders[k])
        spectral_idx = set(eps_idx) | set(alpha_idx)
        if composite is not None:
            for k, (_, start, size) in enumerate(composite['terms']):
                self.param_layout.addWidget(self.comp_type_headers[k])
                shown_type_headers.add(k)
                if multi:
                    self.param_layout.addWidget(self.comp_headers[k])
                    self.param_layout.addWidget(self.nu0_sliders[k])
                    self.param_layout.addWidget(self.temp_sliders[k])
                    self.param_layout.addWidget(self.sliders[alpha_idx[k]])
                    self.param_layout.addWidget(self.beta_sliders[k])
                for i in range(start, start + size):
                    self.param_layout.addWidget(self.sliders[i])
        else:
            for i, sl in enumerate(self.sliders):
                if i not in spectral_idx:
                    self.param_layout.addWidget(sl)

        self.param_box.setTitle(spec.title)
        self.spectral_box.setVisible(has_spectral)
        self.param_scroll.setMinimumHeight(430 if has_spectral else 290)  # all rows fit without scrolling

        # removeWidget above only detaches a widget from the layout -- it
        # doesn't hide it, so whichever of these wasn't re-added this pass
        # (e.g. shape_header_single for a two-component model) would
        # otherwise still report itself visible, stale geometry and all.
        # Done only now, after the addWidget calls above have given
        # comp1_header/comp2_header a real parent at least once -- setting
        # a still-parentless QWidget visible makes Qt treat it as its own
        # top-level window instead, which then breaks the *next* addWidget
        # reparenting.
        self.shape_header_single.setVisible(not multi)
        for k, header in enumerate(self.comp_headers):
            header.setVisible(multi and k < spec.n_components)
        for k, header in enumerate(self.comp_type_headers):
            header.setVisible(k in shown_type_headers)

        if hasattr(self, 'results_label'):
            self.results_label.setText(self.NO_FIT_TEXT)

        self.rebuild_sampling_bounds()
        if hasattr(self, 'curve_name_edit'):
            self.curve_name_edit.setText(self.suggested_curve_name())

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

        Starts *fixed* (checkbox checked): at its default (~1% of the
        band's own nu_min, deep in the optically-thin regime), a fit
        should assume there's no turnover in view and only fit alpha;
        unchecking it lets a fit jointly solve for nu_0 too (only for
        'ssa' -- see fit_spectrum_lsq; 'thermal' never fits nu_0/T).
        build_temp_slider/build_beta_slider start fixed for the same
        reason, since fitting never solves for T/beta at all."""
        sl = ParamSlider(Param('nu0', latex, 'nu0', description), lo_ghz, hi_ghz)
        sl.set_value(default_ghz)
        sl.fix_checkbox.setChecked(True)
        sl.valueChanged.connect(self.on_spectral_shape_changed)
        return sl

    def build_temp_slider(self, lo_k, hi_k, default_k, latex, description):
        """One T ParamSlider (log-scale) for the Spectrum box's thermal
        electron-temperature controls -- see build_nu0_slider."""
        sl = ParamSlider(Param('temp', latex, 'temp', description), lo_k, hi_k)
        sl.set_value(default_k)
        sl.fix_checkbox.setChecked(True)
        sl.valueChanged.connect(self.on_spectral_shape_changed)
        return sl

    def build_beta_slider(self, lo, hi, default, latex, description):
        """One beta ParamSlider (linear, 'scale' kind) for the Spectrum
        box's log-parabola curvature controls -- see build_nu0_slider."""
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
        lo_m, hi_m = self.rmsynth_wl_bounds_m()
        nu_min_ghz = C / hi_m / 1e9
        nu_max_ghz = C / lo_m / 1e9
        lo_mult, hi_mult = NU0_BOUNDS_MULT
        return lo_mult * nu_min_ghz, hi_mult * nu_max_ghz

    def default_nu0_ghz(self):
        """1% of the current wavelength range's own minimum frequency
        [GHz] -- default value for the SSA turnover-frequency sliders, both
        their initial one (see _build_ui) and the one Reset model restores
        (see reset_parameters)."""
        hi_m = self.rmsynth_wl_bounds_m()[1]  # longest wavelength -> nu_min
        nu_min_ghz = C / hi_m / 1e9
        return 0.01 * nu_min_ghz

    def current_shapes(self):
        """(shape1, shape2, shape3) currently selected in the Spectrum box --
        `shape1` is the single-component shape dropdown for a
        single-component model, or component 1's own dropdown for a
        multi-component one; `shape2`/`shape3` are components 2/3's own
        dropdowns, meaningful only for a model with that many weighted
        components (they mirror `shape1` otherwise)."""
        n_components = MODELS[self.model_combo.currentData()].n_components
        if n_components >= 2:
            shapes = [self.shape_combos[k].currentData() if k < n_components
                      else self.shape_combo_1.currentData() for k in range(3)]
            return tuple(shapes)
        shape = self.shape_combo.currentData()
        return shape, shape, shape

    def sync_spectrum_ui(self):
        """Keep the Spectrum box's nu_0/T/beta slider(s) -- and each
        component's own alpha slider's visibility -- matched to each
        component's own shape dropdown and the current model's component
        count, and push nu_0/T/beta's values into models' global
        spectral-shape state (see models.set_spectral_shape) -- called
        whenever any shape dropdown, a nu_0/T/beta slider, or the model
        itself changes."""
        shapes = self.current_shapes()
        spec = MODELS[self.model_combo.currentData()]
        multi = spec.n_components >= 2
        n_active = spec.n_components if multi else 1
        active = [k < n_active for k in range(3)]
        needs_nu0 = [active[k] and shapes[k] in ('ssa', 'thermal') for k in range(3)]
        needs_temp = [active[k] and shapes[k] == 'thermal' for k in range(3)]
        needs_beta = [active[k] and shapes[k] == 'logparabola' for k in range(3)]
        for k in range(3):
            self.nu0_sliders[k].setVisible(needs_nu0[k])
            self.temp_sliders[k].setVisible(needs_temp[k])
            self.beta_sliders[k].setVisible(needs_beta[k])
        self.nu0_slider_1.name_label.setPixmap(
            latex_pixmap(r'$\nu_{0,1}$' if multi else r'$\nu_0$'))
        self.temp_slider_1.name_label.setPixmap(
            latex_pixmap(r'$T_1$' if multi else r'$T$'))
        self.beta_slider_1.name_label.setPixmap(
            latex_pixmap(r'$\beta_1$' if multi else r'$\beta$'))
        # A thermal component's own spectral index is emergent from T, not
        # a free parameter (see intensity_shape/thermal_intensity_shape) -- hide
        # its alpha slider rather than show an inert control (log-parabola
        # still uses alpha directly, alongside its own beta, so its alpha
        # slider stays visible). alpha_indices follows spec.params' own
        # order: index k is component k+1's alpha (tied to shapes[k]).
        for k, ai in enumerate(spec.indices('alpha')):
            self.sliders[ai].setVisible(shapes[k] != 'thermal')
        kwargs = {}
        for k, (shape_key, nu0_key, T_key, beta_key) in enumerate(
                (('shape', 'nu0', 'T', 'beta'), ('shape2', 'nu0_2', 'T2', 'beta2'),
                 ('shape3', 'nu0_3', 'T3', 'beta3'))):
            kwargs[shape_key] = shapes[k]
            kwargs[nu0_key] = self.nu0_sliders[k].value() * 1e3 if needs_nu0[k] else None  # GHz -> MHz
            kwargs[T_key] = self.temp_sliders[k].value() if needs_temp[k] else None
            kwargs[beta_key] = self.beta_sliders[k].value() if needs_beta[k] else None
        set_spectral_shape(**kwargs)

    def on_spectral_shape_changed(self, *_):
        """Slot for the Spectrum box's shape dropdown(s) and nu_0/T spin
        box(es): re-sync models' spectral-shape state and redraw the
        equation card and plots to reflect it."""
        self.sync_spectrum_ui()
        self.refit_equation()
        self.update_plot()

    def refit_equation(self):
        spec = MODELS[self.model_combo.currentData()]
        shape1, shape2, shape3 = self.current_shapes()
        eq = full_equation(spec, shape1, shape2, shape3)
        viewport = self.eq_scroll.viewport().size()
        # Leave room for the label's own padding (6px/side, see its
        # stylesheet) plus a little breathing room.
        max_w = max(viewport.width() - 20, 50)
        # The card keeps its fixed height so it never squeezes the plots. A
        # short equation is shrunk to fit it; a multi-component one (a line
        # per component plus the weights and the combined P(lambda)) is
        # fitted to the width only and scrolled vertically.
        max_h = max(viewport.height() - 20, 20) if eq.count('\n') < 2 else 10 ** 6
        pixmap = fit_equation_pixmap(eq, max_w, max_h)
        self.eq_label.setPixmap(pixmap)
        self.eq_label.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The equation card is re-rendered (a LaTeX pixmap) only once the
        # resizing has settled, not on every event.
        if hasattr(self, 'eq_scroll'):
            if self._eq_resize_timer is None:
                self._eq_resize_timer = QTimer(self)
                self._eq_resize_timer.setSingleShot(True)
                self._eq_resize_timer.timeout.connect(self.refit_equation)
            self._eq_resize_timer.start(RESIZE_SETTLE_MS)

    # min/max spin-box limits and decimals, as set up for mm in _build_left_panel
    WL_MIN_RANGE_MM = (1e-4, 1000.0)
    WL_MAX_RANGE_MM = (2e-4, 40000.0)

    def wl_unit_m(self):
        """Meters per unit of the Wavelength range box's selected unit."""
        return WL_UNITS_M[self.wl_unit_combo.currentText()]

    def on_wl_unit_changed(self, unit):
        """Convert the min/max numbers to `unit` so they keep describing the
        same range, rescale their limits/decimals to match, and relabel the
        lambda^2 axis."""
        old_m = WL_UNITS_M[self._wl_unit_prev]
        self._wl_unit_prev = unit
        lo_mm = self.wl_min.value() * old_m / 1e-3
        hi_mm = self.wl_max.value() * old_m / 1e-3
        scale = 1e-3 / WL_UNITS_M[unit]  # mm -> unit
        decimals = {'μm': 4, 'mm': 6, 'cm': 6, 'm': 7}[unit]  # the range's lower limit is 1e-4 mm in any unit
        # One redraw below instead of one per spin-box change (and none from
        # the intermediate, clamped values).
        for box in (self.wl_min, self.wl_max):
            box.blockSignals(True)
        self.wl_min.setDecimals(decimals)
        self.wl_max.setDecimals(decimals)
        self.wl_min.setRange(self.WL_MIN_RANGE_MM[0] * scale, self.WL_MIN_RANGE_MM[1] * scale)
        self.wl_max.setRange(self.WL_MAX_RANGE_MM[0] * scale, self.WL_MAX_RANGE_MM[1] * scale)
        self.wl_min.setValue(lo_mm * scale)
        self.wl_max.setValue(hi_mm * scale)
        for box in (self.wl_min, self.wl_max):
            box.blockSignals(False)
        self.canvas.set_wl_unit(unit)
        self.refresh_rmsynth_empty_axis()
        self.update_plot()

    def apply_wl_preset(self, index):
        """Fill wl_min/wl_max from the selected preset's stored (lo, hi)
        data -- this itself triggers update_plot via their valueChanged.
        Also resets the Measurements tab's Bands box to that preset's own
        defaults (see measurements.DEFAULT_BANDS_BY_PRESET)."""
        lo_mm, hi_mm = self.wl_preset.itemData(index)
        scale = 1e-3 / self.wl_unit_m()  # presets are stored in mm
        self.wl_min.setValue(lo_mm * scale)
        self.wl_max.setValue(hi_mm * scale)
        self.apply_default_bands(self.wl_preset.itemText(index), lo_mm, hi_mm)

    def current_state(self):
        """(func, spec, pars, wl_ext) for the currently selected model,
        sliders, and wavelength range -- shared by update_plot and the
        Export actions so they can't drift out of sync."""
        func = self.model_combo.currentData()
        spec = MODELS[func]
        pars = [sl.value() for sl in self.sliders]

        unit_m = self.wl_unit_m()
        wl_min_mm = min(self.wl_min.value(), self.wl_max.value()) * unit_m / 1e-3
        wl_max_mm = max(self.wl_min.value(), self.wl_max.value()) * unit_m / 1e-3
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
        self.canvas.band_spans = self.stokes_canvas.band_spans = self.band_spans()
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

    # ── Added curves / Comparison tab ──────────────────────────────────────
    def suggested_curve_name(self, spec=None):
        """plot_name of the current model, suffixed ' 2', ' 3', ... to the
        first free number among the existing curves' names -- so a second
        Int+Ext curve is 'Int+Ext 2' and so on."""
        base = (spec or MODELS[self.model_combo.currentData()]).display_name
        taken = {c['name'] for c in self.curves}
        name, n = base, 1
        while name in taken:
            n += 1
            name = f'{base} {n}'
        return name

    def n_free_params(self, spec):
        """Free parameters of the current model for curve statistics --
        same count a Fit! would optimize: sliders not held fixed, minus eps
        and alpha, which QU data can't constrain (see fit_spectrum_lsq)."""
        return sum(1 for p, sl in zip(spec.params, self.sliders)
                   if p.kind not in ('eps', 'alpha') and not sl.is_fixed())

    def reset_curve_stats(self):
        """Re-evaluate every curve against the current data (or drop the
        stats if there's none) and rebuild the Comparison rows."""
        for c in self.curves:
            c['stats'] = self.curve_stats_vs_data(c) if self.fit_data is not None else None
        self.rebuild_comparison_rows()

    def curve_stats_vs_data(self, curve):
        wl, q, q_err, u, u_err, _freq, _I = self.fit_data
        with spectral_state(curve['spectral']):
            return curve_statistics(curve['pars'], curve['n_free'], wl, q, q_err, u, u_err, curve['func'])

    @staticmethod
    def format_curve_stats(stats):
        aicc = f"{stats['aicc']:.4g}" if np.isfinite(stats['aicc']) else 'n/a'
        return (f"χ²_k = {stats['chi2_red']:.4g}\n"
                f"ln L = {stats['loglike']:.4g}\n"
                f"AICc = {aicc}\n"
                f"BIC = {stats['bic']:.4g}")

    def next_curve_color(self):
        """The earliest color of the Preferences > Curve > Comparison palette
        not used by a current curve (so deleted curves free their color); once
        every color is in use, the earliest among the least-used ones."""
        colors = [mcolors.to_hex(c) for c in COMPARISON_PALETTES[PLOT_PREFS['comparison_palette']]]
        in_use = [mcolors.to_hex(c['color']) for c in self.curves]
        counts = [in_use.count(c) for c in colors]
        return colors[counts.index(min(counts))]

    def add_curve_action(self):
        """Freeze the current model (parameters and spectral shape) as a
        permanent curve on every p/EVPA and I/Q/U plot (not RM-synth)."""
        func, spec, pars, _ = self.current_state()
        self.append_curve(func, spec, pars, get_spectral_state(), self.n_free_params(spec),
                          self.curve_name_edit.text().strip() or self.suggested_curve_name())

    def append_curve(self, func, spec, pars, spectral, n_free, name, **extra):
        """Add a curve. Its fit statistics are computed straight away whenever
        data is loaded -- now, and again on each later Load data (see
        reset_curve_stats)."""
        curve = dict(
            id=self._next_curve_id, func=func, n_components=spec.n_components, pars=list(pars),
            spectral=spectral, n_free=n_free, stats=None, name=name,
            color=self.next_curve_color(), linestyle=PLOT_PREFS['curve_style'], linewidth=PLOT_PREFS['curve_width'],
            rev=0,  # bumped when the curve's own data changes (see widgets._curve_data_key)
            **extra)
        if self.fit_data is not None:
            curve['stats'] = self.curve_stats_vs_data(curve)
        self.curves.append(curve)
        self._next_curve_id += 1
        self.refresh_curves()

    # MultiNest-samples curves (Comparison tab's Load samples): the family's
    # marginal solution (the same point estimate the Sampling tab's sliders
    # take) drawn as the curve itself, plus `n_samples` faint posterior draws.
    MIN_FAMILY_MASS_PCT = 1.0  # families below this evidence share aren't added as curves

    def on_curve_samples_finished(self, model, best_pars, errs, info):
        """Add one curve per mode family holding at least MIN_FAMILY_MASS_PCT
        of the evidence (the winner always), each in its own palette color."""
        self._set_sampling_busy(False)
        self.cs_worker = None
        spec = MODELS[model]
        pol_idx = info['pol_idx']
        winner_errs_pol = np.array([[errs[i][0] for i in pol_idx], [errs[i][1] for i in pol_idx]])
        winner = dict(
            pars=[best_pars[i] for i in pol_idx], errs=winner_errs_pol,
            samples=info['winner_samples'], evidence_share=info['evidence_share'],
            chi2=info['winner_chi2'], lnZ=info['winner_lnZ'], evidence_lnZ=info['winner_evidence_lnZ'])
        spectral_pars = {i: best_pars[i] for i in spec.indices('alpha') + spec.indices('eps')}
        families = [winner] + [f for f in info['other_families'] if f['evidence_share'] >= self.MIN_FAMILY_MASS_PCT]
        spectral = get_spectral_state()
        for fam in families:
            extra = dict(
                func=model, family=fam, pol_idx=pol_idx, spectral_pars=spectral_pars,
                n_samples=PLOT_PREFS['curve_samples'])
            pars, extra['sample_pars'] = self._family_pars_and_samples(extra)
            extra.pop('func')
            self.append_curve(model, spec, pars, spectral, info['n_free'],
                              self.suggested_curve_name(spec), **extra)

    def _family_pars_and_samples(self, curve):
        """(full-length mean parameters, (n_samples, n_params) draws) for a
        MultiNest-samples curve dict's family -- the same expansion/draw
        recipe as stash_posterior_samples."""
        spec = MODELS[curve['func']]
        fam = curve['family']
        pol_idx, spectral_pars = curve['pol_idx'], curve['spectral_pars']
        pars, _ = expand_pars_errs(len(spec.params), pol_idx, fam['pars'], fam['errs'], spectral_pars)
        samples = fam['samples']
        rng = np.random.default_rng(0)
        n = min(curve['n_samples'], len(samples))
        subset = samples[rng.choice(len(samples), size=n, replace=False)]
        draws = np.zeros((n, len(spec.params)))
        draws[:, pol_idx] = subset
        for i, v in spectral_pars.items():
            draws[:, i] = v
        return pars, draws

    def update_curve_draws(self, curve, n_samples):
        curve['n_samples'] = n_samples
        _, curve['sample_pars'] = self._family_pars_and_samples(curve)
        curve['rev'] += 1
        self.request_update_plot()

    @staticmethod
    def curve_evidence_lines(curve):
        return [f"mass = {curve['family']['evidence_share']:.1f}%",
                f"ln Z = {curve['family']['evidence_lnZ']:.4g}"]

    def load_curve_from_file_action(self):
        """Comparison tab's Load Model: read a Save Model JSON file and add
        it straight to the curve list -- unlike Import > Load Model, the
        visualizer's own model, sliders and Spectrum box are left alone."""
        path, _ = QFileDialog.getOpenFileName(self, 'Load Model', '', 'JSON files (*.json)')
        if not path:
            return
        parsed = self.read_model_file(path)
        if parsed is None:
            return
        func, params, states = parsed
        spec = MODELS[func]
        # eps/alpha default to a neutral value if the file lacks them (older
        # saves); QU-only stats don't depend on them anyway.
        defaults = {'eps': 0.5, 'alpha': 0.0}
        try:
            pars = [params[p.name] if p.name in params else defaults[p.kind] for p in spec.params]
        except KeyError as e:
            QMessageBox.warning(self, 'Load Model', f'Model file is missing parameter {e}.')
            return
        spectral = self.spectral_kwargs(spec, states)
        n_free = sum(1 for p in spec.params if p.kind not in ('eps', 'alpha'))
        self.append_curve(func, spec, pars, spectral, n_free, self.suggested_curve_name(spec))

    def apply_plot_preferences(self, **changes):
        """Store `changes` (PLOT_PREFS keys) and restyle every canvas."""
        PLOT_PREFS.update(changes)
        check = getattr(self, 'show_bands_check', None)  # Measurements tab's Show, mirrors bands_on
        if check is not None and check.isChecked() != PLOT_PREFS['bands_on']:
            check.blockSignals(True)
            check.setChecked(PLOT_PREFS['bands_on'])
            check.blockSignals(False)
        for canvas in (self.canvas, self.stokes_canvas, self.rmsynth_canvas):
            apply_plot_prefs(canvas)
        self.update_plot()

    def on_show_bands_toggled(self, checked):
        self.apply_plot_preferences(bands_on=checked)
        if getattr(self, 'preferences_dialog', None) is not None:
            self.preferences_dialog.refresh_widgets()

    def show_preferences(self):
        if getattr(self, 'preferences_dialog', None) is None:
            self.preferences_dialog = PreferencesDialog(self)
        self.preferences_dialog.show()
        self.preferences_dialog.raise_()
        self.preferences_dialog.activateWindow()

    def clear_last_curve_action(self):
        if self.curves:
            self.curves.pop()
            self.refresh_curves()

    def clear_all_curves_action(self):
        if self.curves:
            self.curves.clear()  # in place: the canvases share this list
            self.refresh_curves()

    def remove_curve(self, curve):
        self.curves.remove(curve)
        self.refresh_curves()

    def refresh_curves(self):
        """Sync everything that follows self.curves' membership: buttons,
        the next-name box, the Comparison tab's rows/enabled state, plots."""
        self.clear_curves_button.setEnabled(bool(self.curves))
        self.clear_all_curves_button.setEnabled(bool(self.curves))
        self.export_tex_button.setEnabled(bool(self.curves))
        self.curve_name_edit.setText(self.suggested_curve_name())
        self.rebuild_comparison_rows()
        # The visualizer curve can only be hidden while there's an added
        # curve to show instead -- see on_show_main_toggled.
        if not self.curves:
            self.show_main_check.setChecked(True)
        self.show_main_check.setEnabled(bool(self.curves))
        self.update_plot()

    def build_comparison_tab(self):
        """The "Comparison" tab: one row per added curve (name, linestyle,
        color, delete) followed by its parameters -- rebuilt by
        rebuild_comparison_rows whenever the curve list changes."""
        self.comparison_tab = QWidget()
        outer = QVBoxLayout(self.comparison_tab)
        self.show_main_check = QCheckBox('Show visualizer curve')
        self.show_main_check.setChecked(True)
        self.show_main_check.setEnabled(False)  # needs at least one added curve to be unchecked, see refresh_curves
        self.show_main_check.toggled.connect(self.on_show_main_toggled)
        top_row = QHBoxLayout()
        top_row.addWidget(self.show_main_check)
        top_row.addStretch(1)
        self.show_legend_check = QCheckBox('Show legend')
        self.show_legend_check.setChecked(True)
        self.show_legend_check.toggled.connect(self.on_show_legend_toggled)
        top_row.addWidget(self.show_legend_check)
        outer.addLayout(top_row)
        self.load_curve_button = QPushButton('Load Model')
        self.load_curve_button.setToolTip(
            "Add a saved model file to the curves below, without changing the Visualization tab's model.")
        self.load_curve_button.clicked.connect(self.load_curve_from_file_action)
        self.load_curve_samples_button = QPushButton('Load samples')
        self.load_curve_samples_button.setToolTip(
            'Add a MultiNest run (mean solution plus posterior draws) to the curves below, '
            "without changing the Visualization tab's model.")
        self.load_curve_samples_button.clicked.connect(self.load_curve_samples_action)
        load_row = QHBoxLayout()
        load_row.addWidget(self.load_curve_button)
        load_row.addWidget(self.load_curve_samples_button)
        load_row.addStretch(1)
        self.clear_all_curves_button = QPushButton('Clear curves')
        self.clear_all_curves_button.setToolTip('Delete every added curve.')
        self.clear_all_curves_button.setEnabled(False)
        self.clear_all_curves_button.clicked.connect(self.clear_all_curves_action)
        load_row.addWidget(self.clear_all_curves_button)
        outer.addLayout(load_row)
        self.comparison_rows_widget = CurveDropArea()
        self.comparison_rows_widget.reordered.connect(self.move_curve)
        self.comparison_rows_layout = QVBoxLayout(self.comparison_rows_widget)
        self.comparison_rows_layout.setAlignment(Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.comparison_rows_widget)
        outer.addWidget(scroll)
        self.export_tex_button = QPushButton('Export tex')
        self.export_tex_button.setToolTip('Save the added curves as a LaTeX table: parameters, reduced chi-squared and ln Z.')
        self.export_tex_button.setEnabled(False)  # needs at least one added curve, see refresh_curves
        self.export_tex_button.clicked.connect(self.export_tex_action)
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export_row.addWidget(self.export_tex_button)
        outer.addLayout(export_row)
        self.left_tabs.addTab(self.comparison_tab, 'Comparison')

    def export_tex_action(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export tex', 'comparison.tex', 'LaTeX (*.tex)')
        if not path:
            return
        if not path.endswith('.tex'):
            path += '.tex'
        try:
            with open(path, 'w') as f:
                f.write(comparison_tex_table(self.curves))
        except OSError as e:
            QMessageBox.warning(self, 'Export tex', f'Could not save the table:\n{e}')

    def on_show_legend_toggled(self, checked):
        for canvas in (self.canvas, self.stokes_canvas, self.rmsynth_canvas):
            canvas.show_legend = checked
            sync_legend_visibility(canvas)
            canvas.draw_idle()
        self.update_plot()

    def on_show_main_toggled(self, checked):
        self.canvas.show_main = self.stokes_canvas.show_main = checked
        self.update_plot()

    def rebuild_comparison_rows(self):
        layout = self.comparison_rows_layout
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)  # off-screen immediately; deleteLater alone leaves it painted until the event loop runs
                widget.deleteLater()
        for i, curve in enumerate(self.curves):
            layout.addWidget(self.build_curve_row(curve, i))

    def move_curve(self, src, dst):
        """Drag-reorder in the Comparison tab: self.curves is kept in tab
        order (top = first in the legend, drawn last -- see
        widgets._curve_zorder)."""
        self.curves.insert(dst, self.curves.pop(src))
        self.refresh_curves()

    def build_curve_row(self, curve, index):
        box = QGroupBox()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        row.setSpacing(3)

        row.addWidget(DragHandle(index, box))

        color_button = QPushButton()
        color_button.setFixedSize(26, 26)
        color_button.setStyleSheet(f"background-color: {curve['color']};")
        color_button.setToolTip('Curve color')
        color_button.clicked.connect(lambda _=False, c=curve, b=color_button: self.pick_curve_color(c, b))
        row.addWidget(color_button)

        name_edit = QLineEdit(curve['name'])
        name_edit.textEdited.connect(lambda text, c=curve: self.on_curve_style_edited(c, name=text))
        row.addWidget(name_edit, stretch=1)

        style_combo = QComboBox()
        style_combo.addItems(list(CURVE_LINESTYLES))
        style_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        style_combo.setMinimumContentsLength(8)
        style_combo.setCurrentText(curve['linestyle'])
        style_combo.currentTextChanged.connect(lambda text, c=curve: self.on_curve_style_edited(c, linestyle=text))
        row.addWidget(style_combo)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 10.0)
        width_spin.setSingleStep(0.5)
        width_spin.setDecimals(1)
        width_spin.setValue(curve['linewidth'])
        width_spin.setFixedWidth(52)
        width_spin.setToolTip('Line width')
        width_spin.valueChanged.connect(lambda v, c=curve: self.on_curve_style_edited(c, linewidth=v))
        row.addWidget(width_spin)

        delete_button = QPushButton('X')
        delete_button.setFixedSize(26, 26)
        delete_button.setToolTip('Delete this curve')
        delete_button.clicked.connect(lambda _=False, c=curve: self.remove_curve(c))
        row.addWidget(delete_button)
        box_layout.addLayout(row)

        # Shown in the sliders' own units: p in percent, X in degrees.
        lines = []
        for par, v in zip(MODELS[curve['func']].params, curve['pars']):
            if par.kind == 'p':
                lines.append(f'{par.name} = {100 * v:.4g} %')
            elif par.kind == 'X':
                lines.append(f'{par.name} = {np.degrees(v):.4g} deg')
            else:
                lines.append(f'{par.name} = {v:.4g}')
        params_label = QLabel('\n'.join(lines))
        params_label.setStyleSheet('font-family: monospace; font-size: 8pt;')
        # Parameters on the left; on the right, top-aligned with them, the
        # curve-vs-data statistics (when data is loaded), with an Export
        # model button at the bottom right.
        body = QHBoxLayout()
        body.addWidget(params_label, alignment=Qt.AlignTop)
        body.addStretch(1)
        right = QVBoxLayout()
        right_lines = self.curve_evidence_lines(curve) if 'family' in curve else []
        if curve['stats'] is not None:
            right_lines += self.format_curve_stats(curve['stats']).split('\n')
        if right_lines:
            stats_label = QLabel('\n'.join(right_lines))
            stats_label.setStyleSheet('font-family: monospace; font-size: 8pt;')
            right.addWidget(stats_label, alignment=Qt.AlignTop)
        right.addStretch(1)
        if 'family' in curve:
            # MultiNest-samples curve: how many posterior draws to plot,
            # right above the Export button's spot.
            samples_row = QHBoxLayout()
            samples_row.addStretch(1)
            samples_row.addWidget(QLabel('# samples'))
            samples_spin = QSpinBox()
            samples_spin.setRange(0, 1000)
            samples_spin.setValue(curve['n_samples'])
            samples_spin.setFixedWidth(56)
            samples_spin.setToolTip('Number of posterior samples drawn from this mode')
            samples_spin.valueChanged.connect(lambda v, c=curve: self.update_curve_draws(c, v))
            samples_row.addWidget(samples_spin)
            right.addLayout(samples_row)
        export_button = QPushButton('Export model')
        export_button.setToolTip('Save this curve as a model file (loadable with Load Model).')
        export_button.clicked.connect(lambda _=False, c=curve: self.export_curve_action(c))
        right.addWidget(export_button, alignment=Qt.AlignRight | Qt.AlignBottom)
        body.addLayout(right)
        box_layout.addLayout(body)
        return box

    def on_curve_style_edited(self, curve, **changes):
        curve.update(changes)
        self.curve_name_edit.setText(self.suggested_curve_name())
        self.request_update_plot()

    def pick_curve_color(self, curve, button):
        color = QColorDialog.getColor(QColor(curve['color']), self, 'Curve color')
        if color.isValid():
            curve['color'] = color.name()
            button.setStyleSheet(f"background-color: {curve['color']};")
            self.request_update_plot()

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
        # Minor ticks/grid roughly double a redraw's cost; hide them for the drag.
        for canvas in (self.canvas, self.stokes_canvas):
            set_drag_mode(canvas, True)

    def on_slider_drag_ended(self):
        self._dragging = False
        for canvas in (self.canvas, self.stokes_canvas):
            set_drag_mode(canvas, False)
        self._plot_pending = False  # this call's own full-res redraw supersedes any coalesced one
        self.update_plot()  # one final full-resolution redraw once the drag settles

    # ── RM-synth tab ───────────────────────────────────────────────────────
    def rmsynth_wl_bounds_m(self):
        """(wl_min, wl_max) [m] of the Wavelength range box's own current
        selection -- shared by refresh_rmsynth_empty_axis and
        rmsynth_from_model (which synthesizes over this same full range)."""
        unit_m = self.wl_unit_m()
        return (min(self.wl_min.value(), self.wl_max.value()) * unit_m,
                max(self.wl_min.value(), self.wl_max.value()) * unit_m)

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
            result = compute_faraday_spectrum(wl, q, u, q_err, u_err, robust=self.rmsynth_robust.value())
        except FaradaySpectrumError as e:
            QMessageBox.warning(self, 'RM synthesis', str(e))
            return
        self.rmsynth_canvas.plot_spectrum(result, source)
        self.rmsynth_source = source

    def rerun_rmsynth(self, *_):
        """robust= spin box's valueChanged: re-run RM synthesis on whatever
        source is currently plotted (if any) so changing the weighting
        updates the Faraday spectrum immediately, without requiring another
        Synthesize from click. A no-op while the tab is still empty."""
        source = self.rmsynth_source
        if source == 'model':
            self.rmsynth_from_model()
        elif source == 'data':
            self.rmsynth_from_data()
        elif source == 'simulation':
            self.rmsynth_from_measurements()

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
        self.run_rmsynth(wl, q, u, q_err, u_err, 'simulation')

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

        preferences_menu = menubar.addMenu('&Preferences')
        figure_prefs_action = QAction('Figures...', self)
        figure_prefs_action.triggered.connect(self.show_preferences)
        preferences_menu.addAction(figure_prefs_action)

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
        for sl in self.nu0_sliders:
            sl.set_value(default_nu0)
            sl.fix_checkbox.setChecked(True)
        for sl in self.temp_sliders:
            sl.set_value(DEFAULT_TEMP_K)
            sl.fix_checkbox.setChecked(True)
        for sl in self.beta_sliders:
            sl.set_value(DEFAULT_BETA)
            sl.fix_checkbox.setChecked(True)
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
        # Drop the file-name titles Load data filled in, but not anything
        # the user has since typed over them.
        for edit in (self.p_title_edit, self.stokes_title_edit):
            if getattr(self, 'auto_title', None) and edit.text() == self.auto_title:
                edit.clear()
        self.auto_title = None
        had_data = self.fit_data is not None
        self.fit_data = None
        if had_data:  # repeated clicks with nothing loaded shouldn't rebuild (flicker) the Comparison rows
            self.reset_curve_stats()
        self.data_nu_min = None
        self.remove_corner_tab()
        self.fit_button.setEnabled(False)
        self.rmsynth_data_button.setEnabled(False)
        if self.rmsynth_source == 'data':
            self.rmsynth_source = None
            self.refresh_rmsynth_empty_axis()
        self.results_label.setText(self.NO_FIT_TEXT)
        self.update_plot()

    def run_fit(self):
        """Dispatches Fit! to the Bayesian (MultiNest) path when the
        Sampling tab is the active one, or the ordinary least-squares path
        otherwise. Whether pymultinest is actually usable is only checked
        by run_multinest_fit itself (via fitting.multinest_fit's lazy
        import), which reports a failure there rather than this method
        gating the choice up front."""
        func, _, _, _ = self.current_state()
        if getattr(func, 'is_custom', False):
            QMessageBox.information(
                self, 'Fit', "Custom models aren't supported for QU-fitting/Sampling yet -- "
                              "only the Visualization and Measurements tabs.")
            return
        if self.left_tabs.currentWidget() is self.sampling_tab:
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
        joint nonlinear regression otherwise (see estimate_spectral_shape/
        estimate_shape_2comp) -- rather than being fit alongside
        p/X/phi/dphi. That joint regression also picks up each shape's own
        extra parameter -- nu_0 for 'ssa'/'thermal', T for 'thermal', beta
        for 'logparabola' -- once its own slider is unchecked (see
        build_nu0_slider/build_temp_slider/build_beta_slider); a
        'powerlaw'/'logparabola' component's reference frequency is always
        the shared band edge, never a free nu_0. For a two-component
        model, alpha1/alpha2 always end up equal regardless of the two
        components' own shapes: there's only one real I(nu) dataset, so a
        single shared spectral index is all QU-only fitting can ever
        anchor two components to -- same reasoning FIT_FIXED_EPSILON
        already relies on (dropped entirely when both components are
        'thermal', where alpha is inert for both). Each component's own
        nu_0/T/beta is the exception: fit or pinned independently against
        that same shared-alpha regression, per its own slider's fixed
        checkbox -- not forced equal to the other component's, since a
        fixed component's own value is meaningful chosen input, not just
        an unconstrained initial guess.

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
        shape1, shape2, _ = self.current_shapes()
        NEEDS_REGRESSION = ('ssa', 'thermal', 'logparabola')
        if spec.n_components == 2 and alpha_indices and (shape1 in NEEDS_REGRESSION or shape2 in NEEDS_REGRESSION):
            alpha_init = self.sliders[alpha_indices[0]].value()
            nu0_lo_ghz, nu0_hi_ghz = self.nu0_slider_1.bounds()
            nu0_bounds_hz = (nu0_lo_ghz * 1e9, nu0_hi_ghz * 1e9)
            T_bounds = self.temp_slider_1.bounds()
            beta_bounds = self.beta_slider_1.bounds()
            # Each component's own nu_0/T/beta is fit or pinned
            # independently (per its own slider's fixed checkbox), never
            # forced equal -- alpha is still shared, same reasoning
            # FIT_FIXED_EPSILON below relies on for eps (see
            # estimate_shape_2comp). eps itself mirrors the `fixed` dict
            # built below: FIT_FIXED_EPSILON unless the user has pinned the
            # eps slider to their own value.
            # A component whose own shape is 'powerlaw'/'logparabola'
            # always evaluates at the shared band nu0 (see
            # models.component_reference_nu), not its own (irrelevant,
            # hidden) nu0 slider -- so its own init is the band edge itself
            # and it's never fit, regardless of that slider's own fixed
            # checkbox.
            eps_idx = spec.indices('eps')[0]
            eps_val = self.sliders[eps_idx].value() if self.sliders[eps_idx].is_fixed() else FIT_FIXED_EPSILON
            band_nu0_hz = float(freq.min())
            nu0_1_init = self.nu0_slider_1.value() * 1e9 if shape1 in ('ssa', 'thermal') else band_nu0_hz
            nu0_2_init = self.nu0_slider_2.value() * 1e9 if shape2 in ('ssa', 'thermal') else band_nu0_hz
            fit_nu0_1 = shape1 in ('ssa', 'thermal') and not self.nu0_slider_1.is_fixed()
            fit_nu0_2 = shape2 in ('ssa', 'thermal') and not self.nu0_slider_2.is_fixed()
            T1_init = self.temp_slider_1.value() if shape1 == 'thermal' else None
            T2_init = self.temp_slider_2.value() if shape2 == 'thermal' else None
            fit_T1 = shape1 == 'thermal' and not self.temp_slider_1.is_fixed()
            fit_T2 = shape2 == 'thermal' and not self.temp_slider_2.is_fixed()
            beta1_init = self.beta_slider_1.value() if shape1 == 'logparabola' else None
            beta2_init = self.beta_slider_2.value() if shape2 == 'logparabola' else None
            fit_beta1 = shape1 == 'logparabola' and not self.beta_slider_1.is_fixed()
            fit_beta2 = shape2 == 'logparabola' and not self.beta_slider_2.is_fixed()
            (alpha_est, nu0_1_est_hz, nu0_2_est_hz,
             T1_est, T2_est, beta1_est, beta2_est) = estimate_shape_2comp(
                freq, I, eps_val, nu0_1_init, nu0_2_init, alpha_init,
                nu0_bounds_hz, fit_nu0_1, fit_nu0_2, shape1, shape2,
                T1_init=T1_init, T2_init=T2_init, beta1_init=beta1_init, beta2_init=beta2_init,
                fit_T1=fit_T1, fit_T2=fit_T2, fit_beta1=fit_beta1, fit_beta2=fit_beta2,
                T_bounds=T_bounds, beta_bounds=beta_bounds)
            if shape1 in ('ssa', 'thermal'):
                self.nu0_slider_1.set_value(nu0_1_est_hz / 1e9)
            if shape2 in ('ssa', 'thermal'):
                self.nu0_slider_2.set_value(nu0_2_est_hz / 1e9)
            if shape1 == 'thermal':
                self.temp_slider_1.set_value(T1_est)
            if shape2 == 'thermal':
                self.temp_slider_2.set_value(T2_est)
            if shape1 == 'logparabola':
                self.beta_slider_1.set_value(beta1_est)
            if shape2 == 'logparabola':
                self.beta_slider_2.set_value(beta2_est)
            self.sync_spectrum_ui()
        elif spec.n_components == 1 and shape1 in NEEDS_REGRESSION and alpha_indices:
            alpha_init = self.sliders[alpha_indices[0]].value()
            nu0_lo_ghz, nu0_hi_ghz = self.nu0_slider_1.bounds()
            nu0_bounds_hz = (nu0_lo_ghz * 1e9, nu0_hi_ghz * 1e9)
            T_bounds = self.temp_slider_1.bounds()
            beta_bounds = self.beta_slider_1.bounds()
            nu0_init = self.nu0_slider_1.value() * 1e9 if shape1 in ('ssa', 'thermal') else float(freq.min())
            T_init = self.temp_slider_1.value() if shape1 == 'thermal' else None
            beta_init = self.beta_slider_1.value() if shape1 == 'logparabola' else None
            fit_nu0 = shape1 in ('ssa', 'thermal') and not self.nu0_slider_1.is_fixed()
            fit_T = shape1 == 'thermal' and not self.temp_slider_1.is_fixed()
            fit_beta = shape1 == 'logparabola' and not self.beta_slider_1.is_fixed()
            alpha_est, nu0_est_hz, T_est, beta_est = estimate_spectral_shape(
                freq, I, shape1, alpha_init, nu0_init, T_init, beta_init,
                nu0_bounds_hz, T_bounds, beta_bounds, fit_nu0, fit_T, fit_beta)
            if shape1 in ('ssa', 'thermal'):
                self.nu0_slider_1.set_value(nu0_est_hz / 1e9)
            if shape1 == 'thermal':
                self.temp_slider_1.set_value(T_est)
            if shape1 == 'logparabola':
                self.beta_slider_1.set_value(beta_est)
            self.sync_spectrum_ui()
        else:
            alpha_est = estimate_alpha(freq, I)

        fixed = fixed_epsilons(spec)
        fixed.update({i: alpha_est for i in alpha_indices})
        fixed.update({i: sl.value() for i, sl in enumerate(self.sliders) if sl.is_fixed()})
        lo = [sl.bounds()[0] for sl in self.sliders]
        hi = [sl.bounds()[1] for sl in self.sliders]

        try:
            best_pars, result = qu_fit(wl, q, q_err, u, u_err, func, pars, (lo, hi), fixed=fixed)
        except Exception as e:
            traceback.print_exc()
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
            f"χ²_k = {stats['chi2_red']:.4g}\n"
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
        data = self.model_file_data(func, spec, pars, get_spectral_state())
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            # QMessageBox.warning shows a small popup dialog with an icon,
            # title, and message -- used throughout for user-facing errors.
            QMessageBox.warning(self, 'Save Model', f'Could not save model file:\n{e}')

    def model_file_data(self, func, spec, pars, state):
        """The Save Model JSON dict for `func` at `pars`, with each component's
        spectral shape/nu_0/T/beta taken from `state` (a get_spectral_state()
        snapshot: the live Spectrum box, or a curve's frozen one)."""
        comps = [(state['shape'], state['nu0'], state['T'], state['beta']),
                 (state['shape2'], state['nu0_2'], state['T2'], state['beta2']),
                 (state['shape3'], state['nu0_3'], state['T3'], state['beta3'])]
        data = {
            'model': func.__name__,
            'parameters': dict(zip(spec.param_names, pars)),
        }
        for k in range(max(spec.n_components, 1)):
            shape, nu0, T, beta = comps[k]
            keys = self.MODEL_FILE_KEYS[k]
            data[keys['shape']] = shape
            if shape in ('ssa', 'thermal') and nu0 is not None:
                data[keys['nu0']] = nu0 / 1e3  # MHz -> GHz
            if shape == 'thermal' and T is not None:
                data[keys['temp']] = T
            if shape == 'logparabola' and beta is not None:
                data[keys['beta']] = beta
        if func in CUSTOM_MODEL_DEFS:
            # eps(z)/phi(z)/bounds needed to rebuild+re-register this exact
            # model in a fresh session, where func.__name__ won't already be
            # in MODELS_BY_NAME -- see load_model_action.
            data['custom_definition'] = CUSTOM_MODEL_DEFS[func]
        elif is_composite(func):
            # Same, for each custom model used as a component screen.
            defs = {code: CUSTOM_MODEL_DEFS[SCREENS[code]['func']]
                    for code in func.composite['codes'] if SCREENS[code].get('custom')}
            if defs:
                data['custom_definitions'] = defs
        return data

    def export_curve_action(self, curve):
        """Comparison tab's Export model: write this curve (model, parameters,
        spectral shapes) as a Save Model JSON file."""
        func = curve['func']
        default = re.sub(r'[^\w.+-]+', '_', curve['name']) or func.__name__
        path, _ = QFileDialog.getSaveFileName(self, 'Export Model', f'{default}.json', 'JSON files (*.json)')
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        data = self.model_file_data(func, MODELS[func], curve['pars'], curve['spectral'])
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            QMessageBox.warning(self, 'Export Model', f'Could not save model file:\n{e}')

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
        shape1, shape2, shape3 = self.current_shapes()
        try:
            with open(path, 'w') as f:
                f.write(full_equation(spec, shape1, shape2, shape3) + '\n')
        except OSError as e:
            QMessageBox.warning(self, 'Save Equation', f'Could not save equation file:\n{e}')

    # JSON keys of each component's own spectral state in a Save Model file
    # (component 1's are the original, un-numbered ones).
    MODEL_FILE_KEYS = [
        dict(shape='spectral_shape', nu0='nu0_ghz', temp='temp_k', beta='beta'),
        dict(shape='spectral_shape_2', nu0='nu0_2_ghz', temp='temp_2_k', beta='beta_2'),
        dict(shape='spectral_shape_3', nu0='nu0_3_ghz', temp='temp_3_k', beta='beta_3'),
    ]

    @staticmethod
    def spectral_kwargs(spec, states):
        """models.set_spectral_shape kwargs for a loaded model's per-component
        `states` (dicts of shape/nu0_ghz/temp_k/beta, see read_model_file),
        keeping a value only where that component's own shape uses it."""
        n = spec.n_components if spec.n_components >= 2 else 1
        out = {}
        for k, (shape_key, nu0_key, T_key, beta_key) in enumerate(
                (('shape', 'nu0', 'T', 'beta'), ('shape2', 'nu0_2', 'T2', 'beta2'),
                 ('shape3', 'nu0_3', 'T3', 'beta3'))):
            st = states[k] if k < n else states[0]
            shape = st['shape']
            out[shape_key] = shape
            out[nu0_key] = st['nu0_ghz'] * 1e3 if shape in ('ssa', 'thermal') and st['nu0_ghz'] is not None else None  # GHz -> MHz
            out[T_key] = st['temp_k'] if shape == 'thermal' else None
            out[beta_key] = st['beta'] if shape == 'logparabola' else None
        return out

    def read_model_file(self, path):
        """Parse a Save Model JSON file (registering and inserting into the
        dropdown a custom model this session hasn't seen), returning (func,
        params, states), or None (after warning) if it can't be loaded; `states`
        is one dict per component (up to 3) with its shape/nu0_ghz/temp_k/beta. Shared by
        load_model_action and load_curve_from_file_action."""
        try:
            with open(path) as f:
                data = json.load(f)
            model_name = data['model']
            # A custom model this session has never seen (see
            # save_model_action) -- rebuilt and registered under the same
            # name it was saved with, whether it's the model itself or a
            # component of a saved multi-component one.
            if model_name not in MODELS_BY_NAME and 'custom_definition' in data:
                self.register_saved_custom(model_name, data['custom_definition'])
            for code, cd in data.get('custom_definitions', {}).items():
                self.register_saved_custom(code, cd)
            func = get_model_by_name(model_name)
            params = dict(data['parameters'])
            if model_name in LEGACY_MODELS:
                # Saved from a legacy two-component model (now a composite):
                # its external sigma_phi used the old exp(-sigma^2 lambda^4).
                for i, scale in legacy_dphi_scale(func).items():
                    name = MODELS[func].params[i].name
                    if name in params:
                        params[name] = params[name] * scale
            states = []
            for k, keys in enumerate(self.MODEL_FILE_KEYS):
                first = states[0] if states else None
                states.append(dict(
                    shape=data.get(keys['shape'], first['shape'] if first else 'powerlaw'),
                    nu0_ghz=data.get(keys['nu0'], first['nu0_ghz'] if first else None),
                    temp_k=data.get(keys['temp'], first['temp_k'] if first else None),
                    beta=data.get(keys['beta'], first['beta'] if first else None)))
        except (OSError, ValueError, KeyError, CustomModelError) as e:
            QMessageBox.warning(self, 'Load Model', f'Could not load model file:\n{e}')
            return None
        return func, params, states

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
        parsed = self.read_model_file(path)
        if parsed is None:
            return
        func, params, states = parsed

        # Set the Spectrum box's own state first -- rebuild_sliders below
        # (triggered by the model switch) re-syncs it from whatever's
        # currently selected (see sync_spectrum_ui), so shape/nu_0/T need
        # to already be in place before that runs. Both the single-component
        # dropdown and each two-component dropdown are set from the same
        # saved shape1/shape2 -- current_shapes() only ever reads whichever
        # pair matches the loaded model's own component count, so setting
        # the other is harmless (and keeps a later model switch's own
        # starting point sane).
        combos = [(self.shape_combo, states[0])] + list(zip(self.shape_combos, states))
        for combo, st in combos:
            idx = combo.findData(st['shape'])
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        for k, st in enumerate(states):
            if st['nu0_ghz'] is not None:
                self.nu0_sliders[k].set_value(st['nu0_ghz'])
            if st['temp_k'] is not None:
                self.temp_sliders[k].set_value(st['temp_k'])
            if st['beta'] is not None:
                self.beta_sliders[k].set_value(st['beta'])

        idx = self.model_index_for(func)
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
            freq, I, Q, U, I_err, Q_err, U_err = load_vapola_csv(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, 'Load Data', f'Could not load data file:\n{e}')
            return

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

        # Both plots get the file's name as their title, minus the extension
        # and 'VAPOLA', with underscores shown as spaces.
        stem = os.path.splitext(os.path.basename(path))[0]
        self.auto_title = ' '.join(stem.replace('VAPOLA', '').replace('_', ' ').split())
        self.p_title_edit.setText(self.auto_title)
        self.stokes_title_edit.setText(self.auto_title)

        self.canvas.set_reference_data(w2, p_pct, p_err_pct, evpa_deg, evpa_err_deg)
        self.stokes_canvas.set_reference_data(nu_ghz, I_n, I_err_n, Q_n, Q_err_n, U_n, U_err_n)

        # Fractional q=Q/I, u=U/I (per-point I, not the I0-normalized Q_n/U_n
        # above) -- the quantity the models themselves return, so it's what
        # Fitting compares against. freq, I (raw, not I0-normalized) ride
        # along for Fitting's alpha estimate (see run_fit / estimate_alpha)
        # -- computed there, not here, since it should only take effect once
        # the user clicks Fitting.
        wl, q, q_err, u, u_err = fractional_qu(freq, I, Q, U, I_err, Q_err, U_err)
        self.fit_data = (wl, q, q_err, u, u_err, freq, I)
        self.reset_curve_stats()
        self.fit_button.setEnabled(True)
        self.rmsynth_data_button.setEnabled(True)

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
    # Off the critical path of the user's first Fit!/Load samples/family
    # pick -- see warm_up_sampling_imports's own docstring for why that
    # first use would otherwise pay this cost instead, uninterruptibly.
    warm_up_sampling_imports()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
