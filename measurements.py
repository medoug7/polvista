"""The "Measurements" tab: simulate an RM/depolarization measurement of the
currently selected model in one or more observing bands, and fit the
per-band depolarization (dp/dnu) and RM (dEVPA/dlambda^2) from the
simulated points.

MeasurementsMixin is mixed into app.py's MainWindow, the same way
sampling.SamplingMixin is -- see that module's docstring for why.
"""
import colorsys
import csv

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
    QScrollArea, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QSizePolicy)

from polvista.models import C, stokes_I
from polvista.latex_stuff import latex_pixmap

# Upper end of the red->violet HSV hue sweep band_colors() spans, in
# fractional hue (0=red). Stops short of 1.0 (which would wrap back through
# magenta to red) so the sweep reads as a physical rainbow's own red-to-
# violet order rather than a full color-wheel loop.
RAINBOW_HUE_MAX = 0.8


def band_colors(n):
    """n distinct per-band point colors, low frequency (band 0) red through
    high frequency (band n-1) violet, evenly spread across the bands in
    between -- matches how a real rainbow orders wavelength.

    n=1 is pure red (there's no "highest band" to anchor violet to); n>=2
    spans the full red..violet hue range regardless of n, so e.g. n=3 is
    red/green/violet and n=2 is just the two endpoints, red/violet."""
    if n <= 1:
        return [(1.0, 0.0, 0.0)] * n
    return [colorsys.hsv_to_rgb(RAINBOW_HUE_MAX * i / (n - 1), 1.0, 1.0) for i in range(n)]


# Pixel size of the Spectral parameters box's own per-band color swatch
# (see band_marker_pixmap) -- deliberately close to the polarization plot's
# own per-band marker size (ms=4 points, see app.ModelPlot.draw_reference)
# once Qt's device-independent-pixel scaling is accounted for.
MARKER_SIZE = 12


def band_marker_pixmap(color, size=MARKER_SIZE):
    """Small filled-diamond swatch matching the polarization/EVPA and
    Stokes plots' own per-band point marker (fmt='D', mec='k', mew=0.5 --
    see app.ModelPlot.draw_reference and app.StokesPlot's own analogous
    draw), so the Spectral parameters box's per-band results can be
    visually tied back to those plotted points. `color` is a
    matplotlib-style (r, g, b) tuple in 0..1, as returned by band_colors."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    r, g, b = (round(c * 255) for c in color)
    painter.setBrush(QBrush(QColor(r, g, b)))
    painter.setPen(QPen(QColor('black'), 1))
    half = size / 2
    painter.drawPolygon(QPolygonF([QPointF(half, 0), QPointF(size, half),
                                    QPointF(half, size), QPointF(0, half)]))
    painter.end()
    return pixmap


# Error floor for the weighted linear fits below -- keeps the fit's weights
# finite (1/err^2) even if a noise field is set to exactly 0.
ERR_FLOOR = 1e-8


def weighted_linfit(x, y, y_err):
    """Weighted least-squares fit y = a + b*x. Returns (a, a_err, b, b_err),
    or None if there are fewer than 2 points or x is degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    w = 1.0 / np.clip(np.asarray(y_err, dtype=float), ERR_FLOOR, None) ** 2
    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)
    delta = S * Sxx - Sx ** 2
    if delta == 0:
        return None
    b = (S * Sxy - Sx * Sy) / delta
    a = (Sxx * Sy - Sx * Sxy) / delta
    return float(a), float(np.sqrt(Sxx / delta)), float(b), float(np.sqrt(S / delta))


def format_sci_latex(value, err):
    """Mathtext '(v \\pm e)\\times10^{exp}' fragment (no surrounding '$',
    no symbol/units of its own -- see format_band_result_latex, which
    wraps this into a full '$symbol=...$' line), exponent picked from
    value's own magnitude (or err's, if value happens to be 0)."""
    ref = abs(value) if value != 0 else (abs(err) if err != 0 else 1.0)
    exp = int(np.floor(np.log10(ref)))
    scale = 10.0 ** exp
    return rf'({value / scale:.2f}\pm{err / scale:.2f})\times10^{{{exp}}}'


def format_band_result_latex(band_label, alpha, alpha_err, dep, dep_err, rm, rm_err):
    """One band's results block: a plain-text header line (`band_label`)
    followed by three mathtext lines (alpha, then Dep, then RM -- alpha is
    the in-band spectral index, a weighted log(I) vs log(nu) slope fit
    alongside the existing Dep/RM fits; see generate_measurements).

    Mathtext (matplotlib's, via latex_pixmap) freely mixes plain text and
    '$...$' math within one string -- lines with no '$' at all just render
    as ordinary text -- so the whole block (all bands concatenated) renders
    as a single pixmap; see generate_measurements."""
    return (
        f'{band_label}\n'
        rf'        $\alpha={alpha:.2f}\pm{alpha_err:.2f}$' '\n'
        rf'        $\mathrm{{Dep}}={format_sci_latex(dep, dep_err)}\ \mathrm{{GHz^{{-1}}}}$' '\n'
        rf'        $\mathrm{{RM}}={format_sci_latex(rm, rm_err)}\ \mathrm{{rad\,m^{{-2}}}}$' '\n')


def model_fit_pinned(model_func, wl_arr, pars, wl_pivot):
    """model_func(wl_arr, pars), with any two-component spectral-mixing
    pivot (see models.spectral_weights: the reference frequency
    alpha1/alpha2 weight I1/I2 relative to) pinned to `wl_pivot` instead
    of each call silently deriving its own pivot from min(wl_arr).

    Model functions take no explicit pivot/nu_min argument -- they always
    use the lowest frequency (longest wavelength) present in whatever
    array they're called on. Evaluating a handful of SPW frequencies
    on their own therefore picks a different pivot than the model curves
    themselves do (which are evaluated across the whole plotted
    wavelength range at once), and the two silently diverge whenever
    alpha1 != alpha2. Appending `wl_pivot` as an extra sample -- the
    longest wavelength in the *plotted* range, so also the longest here,
    as long as no band's own frequency falls outside that range -- forces
    the model's own np.min() to resolve to the same pivot the curves use;
    that extra sample is then dropped from the result.

    No-op (beyond one wasted extra evaluation) for single-component
    models, which don't reference any pivot at all."""
    wl_extended = np.concatenate([np.asarray(wl_arr, dtype=float), [wl_pivot]])
    return model_func(wl_extended, pars)[:-1]


def propagate_pol_errors(I, Q, U, I_err, Q_err, U_err):
    """Standard error propagation from independent Gaussian (I, Q, U)
    noise to fractional polarization p=P/I [dimensionless] and EVPA
    [rad] -- same formulas app.py's load_data_action uses for real,
    loaded I/Q/U data, applied here to simulated points instead."""
    r2 = Q ** 2 + U ** 2
    p_frac = np.sqrt(r2) / I
    evpa_rad = 0.5 * np.arctan2(U, Q)
    sigma_r = np.sqrt(Q ** 2 * Q_err ** 2 + U ** 2 * U_err ** 2) / np.sqrt(r2)
    p_frac_err = np.sqrt((sigma_r / I) ** 2 + (np.sqrt(r2) / I ** 2 * I_err) ** 2)
    evpa_rad_err = 0.5 * np.sqrt(Q ** 2 * U_err ** 2 + U ** 2 * Q_err ** 2) / r2
    return p_frac, p_frac_err, evpa_rad, evpa_rad_err


# Label of the Visualization tab's own default wavelength-range preset (see
# app.WAVELENGTH_PRESETS) -- both the key DEFAULT_BANDS_BY_PRESET uses for
# its curated ALMA entry and the one build_measurements_tab seeds the
# Bands box with at startup, matching that preset's own default selection.
STANDARD_ALMA_LABEL = 'Standard ALMA (0.75 - 3.7 mm / 80 - 350 GHz)'

# ── Real receiver bands, by facility ────────────────────────────────────
# (center_ghz, bw_ghz, n_spw) triples sourced from each observatory's own
# published receiver frequency ranges -- centers are the geometric mean of
# each band's own range (except ALMA 3/6/7, which reuse the Standard ALMA
# preset's literature-standard polarization tuning frequencies so that
# entry and the fuller ALMA list below agree); bandwidths are a realistic
# per-band continuum setup (ALMA: one 8 GHz baseband pair, its own standard
# continuum bandwidth; VLA: up to a 2 GHz baseband, capped narrower for the
# bands too narrow to fit one; MeerKAT: the receiver's own full bandwidth,
# split into n_spw=4 the same way every other band here is).
#
# ALMA -- all 10 receiver bands, https://www.almaobservatory.org/en/about-alma/how-alma-works/technologies/receivers/
ALMA_BAND1 = (43.0, 8.0, 4)      # Band 1:  35-50 GHz
ALMA_BAND2 = (78.0, 8.0, 4)      # Band 2:  67-90 GHz
ALMA_BAND3 = (93.3, 14.0, 4)     # Band 3:  84-116 GHz
ALMA_BAND4 = (145.0, 8.0, 4)     # Band 4:  125-163 GHz
ALMA_BAND5 = (185.0, 8.0, 4)     # Band 5:  163-211 GHz
ALMA_BAND6 = (220.3, 16.0, 4)    # Band 6:  211-275 GHz
ALMA_BAND7 = (342.6, 14.0, 4)    # Band 7:  275-373 GHz
ALMA_BAND8 = (440.0, 8.0, 4)     # Band 8:  385-500 GHz
ALMA_BAND9 = (650.0, 8.0, 4)     # Band 9:  602-720 GHz
ALMA_BAND10 = (870.0, 8.0, 4)    # Band 10: 787-950 GHz
ALMA_ALL_BANDS = [ALMA_BAND1, ALMA_BAND2, ALMA_BAND3, ALMA_BAND4, ALMA_BAND5,
                    ALMA_BAND6, ALMA_BAND7, ALMA_BAND8, ALMA_BAND9, ALMA_BAND10]

# VLA -- L through Q band, https://science.nrao.edu/facilities/vla/docs/manuals/propvla/frequency-bands-and-samplers
VLA_L = (1.4, 0.6, 4)       # L band:  1-2 GHz
VLA_S = (2.8, 1.2, 4)       # S band:  2-4 GHz
VLA_C = (5.7, 2.0, 4)       # C band:  4-8 GHz
VLA_X = (9.8, 2.0, 4)       # X band:  8-12 GHz
VLA_KU = (14.7, 2.0, 4)     # Ku band: 12-18 GHz
VLA_K = (21.8, 2.0, 4)      # K band:  18-26.5 GHz
VLA_KA = (32.5, 2.0, 4)     # Ka band: 26.5-40 GHz
VLA_Q = (44.7, 2.0, 4)      # Q band:  40-50 GHz
VLA_LOW_BANDS = [VLA_L, VLA_S, VLA_C, VLA_X]
VLA_HIGH_BANDS = [VLA_K, VLA_KA, VLA_Q]
VLA_ALL_BANDS = [VLA_L, VLA_S, VLA_C, VLA_X, VLA_KU, VLA_K, VLA_KA, VLA_Q]

# MeerKAT -- UHF, L, and S band, https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/277315585/MeerKAT+specifications
MEERKAT_UHF = (0.77, 0.54, 4)   # UHF band: 544-1087 MHz
MEERKAT_L = (1.21, 0.86, 4)     # L band:   856-1711 MHz
MEERKAT_S = (2.47, 1.75, 4)     # S band:   1750-3499 MHz
MEERKAT_ALL_BANDS = [MEERKAT_UHF, MEERKAT_L, MEERKAT_S]

# LOFAR -- LBA and HBA band, https://www.aanda.org/articles/aa/full_html/2013/08/aa20873-12/aa20873-12.html
LOFAR_LBA = (0.05, 0.08, 4)   # 10 - 90 MHz
LOFAR_HBA = (0.18, 0.12, 4)     # 120 - 240 MHz
LOFAR_ALL_BANDS = [LOFAR_LBA, LOFAR_HBA]

# SDSS-like optical filters (g, r, i, z) -- SPARC4's own 4 simultaneous
# channels, per its instrument papers -- using the same round-number
# Fukugita et al. 1996 band edges as app.WAVELENGTH_PRESETS; center is
# the geometric mean of each band's own nu_min/nu_max, bw = nu_max - nu_min.
SDSS_G = (639200.0, 204400.0, 4)   # g band: 400-550 nm / 545-749 THz
SDSS_R = (483200.0, 116800.0, 4)   # r band: 550-700 nm / 428-545 THz
SDSS_I = (388700.0, 75600.0, 4)    # i band: 700-850 nm / 353-428 THz
SDSS_Z = (325200.0, 52900.0, 4)    # z band: 850-1000 nm / 300-353 THz
SDSS_ALL_BANDS = [SDSS_G, SDSS_R, SDSS_I, SDSS_Z]

# SOFIA/HAWC+'s far-IR bands, per Harper et al. 2018 (center_um, bw_um):
# A=53/8.7, C=89/17.0, D=154/34.0, E=214/44.0 -- Band B (63 um) is left out,
# it has a known oversaturation issue and isn't used in science papers (see
# app.WAVELENGTH_PRESETS). center/bw here follow the same nu_min/nu_max
# geometric-mean/difference convention as SDSS_G etc. above, converting
# each band's own (symmetric-in-wavelength) filter edges to frequency.
HAWC_A = (5675.6, 934.8, 4)     # Band A: 53 um (48.65-57.35 um)
HAWC_C = (3383.9, 649.3, 4)     # Band C: 89 um (80.5-97.5 um)
HAWC_D = (1958.7, 435.1, 4)     # Band D: 154 um (137-171 um)
HAWC_E = (1408.4, 291.1, 4)     # Band E: 214 um (192-236 um)
HAWC_ALL_BANDS = [HAWC_A, HAWC_C, HAWC_D, HAWC_E]

# Full radio combines ALMA + VLA (its own two lowest-frequency facilities
# above) into one default list. ALMA Band 1 (35-50 GHz) and VLA's Q band
# (40-50 GHz) overlap 40-50 GHz -- Band 1 is dropped and VLA_Q kept for
# that stretch, per the rest of ALMA (Bands 2-10, from 67 GHz up) never
# overlapping VLA's own top end.
FULL_GHZ_BANDS = VLA_ALL_BANDS + [ALMA_BAND2, ALMA_BAND3, ALMA_BAND4,
                                     ALMA_BAND5, ALMA_BAND6, ALMA_BAND7,
                                     ALMA_BAND8, ALMA_BAND9, ALMA_BAND10]

FULL_MHZ_BANDS = LOFAR_ALL_BANDS + MEERKAT_ALL_BANDS


FULL_RADIO_BANDS = FULL_MHZ_BANDS + VLA_ALL_BANDS + [ALMA_BAND2, ALMA_BAND3, ALMA_BAND4,
                                     ALMA_BAND5, ALMA_BAND6, ALMA_BAND7,
                                     ALMA_BAND8, ALMA_BAND9, ALMA_BAND10]

# Curated (center_ghz, bw_ghz, n_spw) defaults, keyed by the wavelength-
# preset label they belong to -- auto-populated into the Bands box when
# that preset is chosen from the Visualization tab's dropdown (see
# MainWindow.apply_wl_preset / MeasurementsMixin.apply_default_bands).
# Presets with no entry at all here (none, as of this writing) would
# instead fall back to a single generic band spanning their own range --
# see default_bands_for_preset.
DEFAULT_BANDS_BY_PRESET = {
    STANDARD_ALMA_LABEL: [ALMA_BAND3, ALMA_BAND6, ALMA_BAND7],
    'Full ALMA (0.25 - 9 mm / 30 - 1200 GHz)': ALMA_ALL_BANDS,
    'VLA high (0.6 - 2 cm / 15 - 50 GHz)': VLA_HIGH_BANDS,
    'VLA low (2 - 30 cm / 1 - 15 GHz)': VLA_LOW_BANDS,
    'Full VLA (0.6 - 30 cm / 1 - 50 GHz)': VLA_ALL_BANDS,
    'Full GHz (0.25mm - 30 cm / 1 - 1200 GHz)': FULL_GHZ_BANDS,
    'Full MeerKat (8.6 - 70 cm / 500 - 3500 MHz)': MEERKAT_ALL_BANDS,
    'Full LOFAR (1.25 - 30 m / 10 - 240 MHz)': LOFAR_ALL_BANDS,
    'Full MHz (8.6 cm - 30 m / 10 - 3500 GHz)': FULL_MHZ_BANDS,
    'Full radio (0.25mm - 30 m / 500 MHz - 1200 GHz)': FULL_RADIO_BANDS,
    'Full SPARC4 optical (400 - 1000 nm / 300 - 750 THz)': SDSS_ALL_BANDS,
    'Full HAWC+ FIR (40 - 250 um / 1.2 - 7.5 THz)': HAWC_ALL_BANDS,
}


def default_bands_for_preset(label, lo_mm, hi_mm):
    """(center_ghz, bw_ghz, n_spw) triples to auto-populate the Bands box
    with for wavelength-range preset `label` (whose own stored range is
    `lo_mm`-`hi_mm` mm) -- the curated list in DEFAULT_BANDS_BY_PRESET if
    one exists, otherwise a single band spanning the preset's own range
    (geometric-mean center frequency, 20% fractional bandwidth)."""
    if label in DEFAULT_BANDS_BY_PRESET:
        return DEFAULT_BANDS_BY_PRESET[label]
    lo_ghz, hi_ghz = C / (hi_mm * 1e-3) / 1e9, C / (lo_mm * 1e-3) / 1e9
    center = float(np.sqrt(lo_ghz * hi_ghz))
    bw = max(0.2 * center, 0.5)
    return [(center, bw, 4)]


class BandRow(QWidget):
    """One 'Bands' row: central frequency, bandwidth (both GHz), and number
    of spectral windows (SPW) that subdivide the band -- Generate places one
    point per SPW, at that sub-band's own center frequency."""
    removed = pyqtSignal(object)

    def __init__(self, center_ghz=93.3, bw_ghz=14.0, n_spw=4, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        # No setSuffix()/wide ranges baked into the box itself -- the unit
        # is its own QLabel outside the spin box, and each box is pinned to
        # a narrow fixed width, so a full row (both frequencies + SPW +
        # remove button) fits the left panel's width without the Bands
        # scroll area ever needing a horizontal scrollbar.
        center_tip = 'Central (average) frequency of this band.'
        bw_tip = 'Total bandwidth of this band.'
        spw_tip = 'Number of spectral windows (SPW) the band is split into'

        self.center = QDoubleSpinBox()
        # Upper bound reaches into optical frequencies (SDSS g/r/i/z run
        # ~300,000-750,000 GHz) -- 2000 GHz used to clamp those.
        self.center.setRange(0.01, 1000000.0)
        self.center.setDecimals(2)
        self.center.setFixedWidth(80)
        self.center.setValue(center_ghz)
        self.center.setToolTip(center_tip)

        self.bw = QDoubleSpinBox()
        self.bw.setRange(0.001, 1000000.0)
        self.bw.setDecimals(2)
        self.bw.setFixedWidth(72)
        self.bw.setValue(bw_ghz)
        self.bw.setToolTip(bw_tip)

        self.n_spw = QSpinBox()
        self.n_spw.setRange(1, 64)
        self.n_spw.setFixedWidth(40)
        self.n_spw.setValue(n_spw)
        self.n_spw.setToolTip(spw_tip)

        self.remove_button = QPushButton('×')
        self.remove_button.setFixedWidth(22)
        self.remove_button.setToolTip('Remove this band')
        self.remove_button.clicked.connect(lambda: self.removed.emit(self))

        center_label, bw_label, spw_label = QLabel(), QLabel(), QLabel('SPW')
        center_label.setPixmap(latex_pixmap(r'$\bar{\nu}$'))
        center_label.setToolTip(center_tip)
        bw_label.setPixmap(latex_pixmap(r'$\Delta \nu$'))
        bw_label.setToolTip(bw_tip)
        #spw_label.setPixmap(latex_pixmap('SPW'))
        spw_label.setToolTip(spw_tip)

        layout.addWidget(center_label)
        layout.addWidget(self.center)
        layout.addWidget(QLabel('GHz'))
        layout.addWidget(bw_label)
        layout.addWidget(self.bw)
        layout.addWidget(QLabel('GHz'))
        layout.addWidget(spw_label)
        layout.addWidget(self.n_spw)
        layout.addWidget(self.remove_button)

    def spw_freqs_ghz(self):
        """One frequency per SPW [GHz], at the center of each of `n_spw`
        equal sub-divisions of [center-bw/2, center+bw/2]."""
        c, bw, n = self.center.value(), self.bw.value(), self.n_spw.value()
        lo = c - bw / 2
        step = bw / n
        return lo + step * (np.arange(n) + 0.5)


class MeasurementsMixin:
    """Measurements-tab (band definitions, noise, Generate) UI and
    orchestration, mixed into app.MainWindow -- see module docstring."""

    def build_measurements_tab(self):
        self.measurements_tab = QWidget()
        outer_layout = QVBoxLayout(self.measurements_tab)

        # ── Bands box ──────────────────────────────────────────────────
        bands_box = QGroupBox('Bands')
        bands_layout = QVBoxLayout(bands_box)

        self.bands_container = QWidget()
        self.bands_container_layout = QVBoxLayout(self.bands_container)
        self.bands_container_layout.addStretch(1)

        self.bands_scroll = QScrollArea()
        self.bands_scroll.setWidgetResizable(True)
        self.bands_scroll.setWidget(self.bands_container)
        bands_layout.addWidget(self.bands_scroll, stretch=1)

        add_band_row = QHBoxLayout()
        self.add_band_button = QPushButton('+ Add band')
        self.add_band_button.clicked.connect(lambda: self.add_band_row())
        add_band_row.addWidget(self.add_band_button)
        add_band_row.addStretch(1)
        bands_layout.addLayout(add_band_row)

        bands_layout.addWidget(QLabel('Noise (1σ, % of Stokes I)'))

        # Label left-aligned, spinbox+unit right-aligned against the box's
        # own right border (the stretch between them absorbs the gap).
        i_row = QHBoxLayout()
        i_row.addWidget(QLabel('Stokes I:'))
        i_row.addStretch(1)
        self.meas_i_noise = QDoubleSpinBox()
        self.meas_i_noise.setRange(0.0, 100.0)
        self.meas_i_noise.setDecimals(3)
        self.meas_i_noise.setValue(0.25)
        i_row.addWidget(self.meas_i_noise)
        i_row.addWidget(QLabel('%'))
        bands_layout.addLayout(i_row)

        qu_row = QHBoxLayout()
        qu_row.addWidget(QLabel('Stokes Q, U:'))
        qu_row.addStretch(1)
        self.meas_qu_noise = QDoubleSpinBox()
        self.meas_qu_noise.setRange(0.0, 100.0)
        self.meas_qu_noise.setDecimals(3)
        self.meas_qu_noise.setValue(0.03)
        qu_row.addWidget(self.meas_qu_noise)
        qu_row.addWidget(QLabel('%'))
        bands_layout.addLayout(qu_row)

        # Clear/Generate on the left, sharing that half of the row evenly
        # (Expanding size policy so they actually grow to fill it, not just
        # sit at their natural size with dead space after them); Export on
        # the right, pinned to the box's own right edge within the other
        # half rather than stretched -- the two addLayout/addWidget calls
        # below share stretch=1 each, splitting gen_row itself into equal
        # left/right halves.
        gen_row = QHBoxLayout()
        left_group = QHBoxLayout()
        self.generate_button = QPushButton('Generate')
        self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.generate_button.clicked.connect(self.generate_measurements)
        left_group.addWidget(self.generate_button)
        self.clear_meas_button = QPushButton('Clear')
        self.clear_meas_button.setToolTip('Remove the currently plotted simulated measurement points.')
        self.clear_meas_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_meas_button.clicked.connect(self.clear_measurement_points)
        left_group.addWidget(self.clear_meas_button)
        gen_row.addLayout(left_group, 1)
        self.export_meas_button = QPushButton('Export')
        self.export_meas_button.setToolTip('Save the currently plotted simulated measurement points to a CSV.')
        self.export_meas_button.setEnabled(False)  # only enabled once Generate has produced points
        self.export_meas_button.clicked.connect(self.export_measurements_action)
        gen_row.addWidget(self.export_meas_button, 1, Qt.AlignRight)
        bands_layout.addLayout(gen_row)

        outer_layout.addWidget(bands_box)

        # ── Spectral parameters box ───────────────────────────────────
        spectral_params_box = QGroupBox('Spectral parameters')
        spectral_params_layout = QVBoxLayout(spectral_params_box)

        # One row per band once Generate has run (see set_spectral_params_bands):
        # a small colored marker matching that band's own point marker on the
        # Visualization tab's plots (app.ModelPlot.draw_reference), followed
        # by that band's results. self.spectral_params_placeholder (plain,
        # markerless status text) is the sole row before the first Generate.
        self.spectral_params_placeholder = QLabel('Generate some points to see per-band fits.')
        self.spectral_params_placeholder.setStyleSheet('font-family: monospace;')
        self.spectral_params_placeholder.setWordWrap(True)

        self.spectral_params_container = QWidget()
        self.spectral_params_rows_layout = QVBoxLayout(self.spectral_params_container)
        self.spectral_params_rows_layout.setAlignment(Qt.AlignTop)
        self.spectral_params_rows_layout.addWidget(self.spectral_params_placeholder)
        self.spectral_params_band_rows = []  # dynamic per-band row widgets, see set_spectral_params_bands

        spectral_params_scroll = QScrollArea()
        spectral_params_scroll.setWidgetResizable(True)
        spectral_params_scroll.setWidget(self.spectral_params_container)
        spectral_params_layout.addWidget(spectral_params_scroll)

        outer_layout.addWidget(spectral_params_box, stretch=1)

        self.left_tabs.addTab(self.measurements_tab, 'Measurements')

        # Raw per-band rows behind the last Generate'd points, kept around
        # only for export_measurements_action -- see generate_measurements.
        self.meas_export_rows = None

        # Seed with the Standard ALMA preset's own default bands -- matches
        # the Visualization tab's wavelength-range dropdown, which also
        # starts on that preset (see app.WAVELENGTH_PRESETS[0] and
        # apply_default_bands, which re-does this whenever the dropdown
        # selection changes).
        self.band_rows = []
        for center_ghz, bw_ghz, n_spw in DEFAULT_BANDS_BY_PRESET[STANDARD_ALMA_LABEL]:
            self.add_band_row(center_ghz, bw_ghz, n_spw)

        # Default height fits exactly this many rows (3, from the seeding
        # above) without a scrollbar -- more bands than that scroll, so the
        # box can't keep eating into the Spectral parameters box below it.
        # activate() first so sizeHint() reflects the 3 rows just added
        # (otherwise it can under-measure by a few px, just enough to force
        # a needless scrollbar -- which then steals width from the
        # viewport and re-triggers the horizontal scrolling this layout is
        # meant to avoid); the frame width and a small buffer cover the
        # scroll area's own border, also not part of that sizeHint.
        self.bands_container_layout.activate()
        content_h = self.bands_container.sizeHint().height()
        frame = 2 * self.bands_scroll.frameWidth()
        self.bands_scroll.setFixedHeight(content_h + frame + 6)

    def add_band_row(self, center_ghz=93.3, bw_ghz=14.0, n_spw=4):
        row = BandRow(center_ghz, bw_ghz, n_spw)
        row.removed.connect(self.remove_band_row)
        self.bands_container_layout.insertWidget(self.bands_container_layout.count() - 1, row)
        self.band_rows.append(row)

    def apply_default_bands(self, label, lo_mm, hi_mm):
        """Reset the Bands box to preset `label`'s own defaults (see
        DEFAULT_BANDS_BY_PRESET/default_bands_for_preset) -- called from
        MainWindow.apply_wl_preset, i.e. only when the Visualization tab's
        wavelength-range dropdown itself is changed, not from manually
        editing wl_min/wl_max."""
        if not hasattr(self, 'band_rows'):
            return  # Measurements tab not built yet -- mid app startup
        for row in list(self.band_rows):
            self.remove_band_row(row)
        for center_ghz, bw_ghz, n_spw in default_bands_for_preset(label, lo_mm, hi_mm):
            self.add_band_row(center_ghz, bw_ghz, n_spw)

    def remove_band_row(self, row):
        self.band_rows.remove(row)
        row.setParent(None)

    def set_spectral_params_message(self, text):
        """Show a plain, markerless status line in the Spectral parameters
        box (no bands yet, or none defined) and drop any per-band rows from
        the last Generate."""
        for row in self.spectral_params_band_rows:
            row.setParent(None)
        self.spectral_params_band_rows = []
        self.spectral_params_placeholder.setText(text)
        self.spectral_params_placeholder.setVisible(True)

    def set_spectral_params_bands(self, entries):
        """Replace the Spectral parameters box's contents with one row per
        band: `entries` is a list of (color, latex_text) pairs, in the same
        per-band order as generate_measurements' own band_colors -- each
        row gets a small colored marker matching that band's own point
        marker on the Visualization tab's plots (see
        app.ModelPlot.draw_reference's fmt='D', mec='k' points), followed
        by that band's rendered results."""
        for row in self.spectral_params_band_rows:
            row.setParent(None)
        self.spectral_params_band_rows = []
        self.spectral_params_placeholder.setVisible(False)

        for color, text in entries:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setAlignment(Qt.AlignTop)

            marker = QLabel()
            marker.setPixmap(band_marker_pixmap(color))
            marker.setFixedSize(MARKER_SIZE, MARKER_SIZE)
            marker.setContentsMargins(0, 3, 4, 0)  # nudge down to align with the text's first line
            row_layout.addWidget(marker)

            text_label = QLabel()
            text_label.setPixmap(latex_pixmap(text))
            row_layout.addWidget(text_label)
            row_layout.addStretch(1)

            self.spectral_params_rows_layout.addWidget(row)
            self.spectral_params_band_rows.append(row)

    def clear_measurement_points(self):
        """Remove any Generate'd points from both plot canvases and reset
        the Spectral parameters box -- wired to the Measurements tab's own
        Clear button.

        Also clears the RM-synth tab's own plot, but only if it was last
        synthesized from these same measurements (see self.rmsynth_source
        and app.MainWindow.clear_data's analogous handling of loaded data)
        -- a model- or data-sourced Faraday spectrum survives this."""
        self.canvas.clear_measurement_data()
        self.stokes_canvas.clear_measurement_data()
        self.set_spectral_params_message('Generate some points to see per-band fits.')
        self.meas_export_rows = None
        self.export_meas_button.setEnabled(False)
        self.rmsynth_measurements_button.setEnabled(False)
        if self.rmsynth_source == 'measurements':
            self.rmsynth_source = None
            self.refresh_rmsynth_empty_axis()
        self.update_plot()

    def generate_measurements(self):
        """Simulate one noisy (p, EVPA, I, Q, U) point per SPW of every
        defined band, from the current model/sliders, overlay them on the
        Visualization tab's p/EVPA and I/Q/U plots, and fit each band's own
        depolarization (dp/dnu) and RM (dEVPA/dlambda^2)."""
        func, spec, pars, wl_ext = self.current_state()

        if not self.band_rows:
            self.canvas.clear_measurement_data()
            self.stokes_canvas.clear_measurement_data()
            self.set_spectral_params_message('No bands defined.')
            self.update_plot()
            return

        band_freqs = [row.spw_freqs_ghz() for row in self.band_rows]

        # Assign each band a color by its own rank in center frequency, not
        # by row/list order (bands need not be listed low-to-high) -- so the
        # red->violet rainbow sweep (band_colors) always tracks frequency
        # the way a real rainbow does, regardless of the order bands were
        # added in.
        freq_rank = np.argsort([row.center.value() for row in self.band_rows]).argsort()
        rainbow = band_colors(len(self.band_rows))
        row_colors = [rainbow[r] for r in freq_rank]

        # Pin every band's own pivot -- both the Stokes I/Q/U amplitude
        # normalization (models.stokes_I's nu_min) and, for two-component
        # models, the alpha1/alpha2 spectral-mixing weight (see
        # model_fit_pinned) -- to the exact same pivot the currently
        # displayed model curves themselves use. Without this, each band
        # would derive its own local pivot from just its own handful of
        # frequencies, which silently diverges from the curves' own (in
        # both Stokes I and, whenever alpha1 != alpha2, p/EVPA too) -- see
        # model_fit_pinned's docstring.
        #
        # Real loaded data (MainWindow.load_data_action) has priority for
        # that shared pivot: its own I(nu_min)=1 normalization is computed
        # once at load time and never rescales, so if any data is loaded,
        # the pivot must stay anchored to *its* own nu_min, not whatever
        # frequencies these freshly-generated bands happen to span --
        # otherwise a band reaching below the loaded data's own lowest
        # frequency would drag self.data_nu_min down with it and visibly
        # desync the model curves from the already-plotted real points.
        # Only derive a fresh pivot from the currently plotted wavelength
        # range (wl_ext's own lowest frequency) when there's no loaded data
        # to defer to.
        #
        # Either way this overwrites self.data_nu_min the same way a real
        # Fit! anchors it (see MainWindow.fit_spectrum_lsq), so the Stokes
        # I/Q/U curve itself keeps using this pivot on every redraw, not
        # just for the points generated right now.
        if self.fit_data is not None:
            freq_loaded = self.fit_data[5]  # (wl, q, q_err, u, u_err, freq, I)
            nu_min = float(np.min(freq_loaded)) / 1e6  # MHz
            wl_pivot = C / (nu_min * 1e6)  # wl [m] at that same nu_min
        else:
            wl_pivot = float(np.max(wl_ext))
            nu_min = C / wl_pivot / 1e6  # MHz, matches stokes_I's own nu_min convention
        self.data_nu_min = nu_min

        i_noise = self.meas_i_noise.value()    # % of Stokes I
        qu_noise = self.meas_qu_noise.value()  # % of Stokes I

        rng = np.random.default_rng()

        p_bands, i_bands = [], []
        export_rows = []
        results_lines = []

        for b, (row, freqs_ghz) in enumerate(zip(self.band_rows, band_freqs)):
            n = len(freqs_ghz)
            wl_arr = C / (freqs_ghz * 1e9)  # m

            I_true = stokes_I(wl_arr, spec.n_components, pars, nu_min=nu_min)
            fit_true = model_fit_pinned(func, wl_arr, pars, wl_pivot)
            Q_true, U_true = fit_true.real * I_true, fit_true.imag * I_true

            # Noise is injected directly into I, Q, U (each a fixed % of the
            # model's own local Stokes I) -- p and EVPA are never given
            # their own noise, they're derived from the noisy I/Q/U by the
            # same error propagation a real reduction pipeline would use
            # (propagate_pol_errors), same as app.py's own Load Data path.
            I_err = i_noise / 100.0 * np.abs(I_true)
            Q_err = qu_noise / 100.0 * np.abs(I_true)
            U_err = qu_noise / 100.0 * np.abs(I_true)

            I_obs = I_true + rng.normal(0, I_err, n)
            Q_obs = Q_true + rng.normal(0, Q_err, n)
            U_obs = U_true + rng.normal(0, U_err, n)

            p_frac, p_frac_err, X_obs_rad, X_err_rad = propagate_pol_errors(
                I_obs, Q_obs, U_obs, I_err, Q_err, U_err)
            p_obs, p_err = 100.0 * p_frac, 100.0 * p_frac_err
            X_obs_deg, X_err_deg = np.degrees(X_obs_rad), np.degrees(X_err_rad)

            color = row_colors[b]
            w2 = wl_arr ** 2 * 1e6  # mm^2, matches ModelPlot's own convention
            p_bands.append(dict(color=color, w2=w2, p=p_obs, p_err=p_err,
                                 evpa=X_obs_deg, evpa_err=X_err_deg))
            i_bands.append(dict(color=color, nu=freqs_ghz, I=I_obs, I_err=I_err,
                                 Q=Q_obs, Q_err=Q_err, U=U_obs, U_err=U_err))
            for i in range(n):
                export_rows.append([b + 1, freqs_ghz[i], I_obs[i], I_err[i], Q_obs[i], Q_err[i],
                                     U_obs[i], U_err[i], p_obs[i], p_err[i], X_obs_deg[i], X_err_deg[i]])

            band_label = f'Band {b + 1} ({row.center.value():.3g} GHz, {n} SPW)'
            if n < 2:
                results_lines.append((color, f'{band_label}\n  need ≥ 2 SPWs to fit a slope'))
                continue

            lam2 = wl_arr ** 2  # m^2
            # In-band spectral index: a weighted log(I) vs log(nu) slope,
            # independent of (and not to be confused with) the model's own
            # alpha1/alpha2 sliders -- this is what a real observer would
            # actually measure from just this band's own noisy I points.
            alpha_fit = weighted_linfit(np.log(freqs_ghz), np.log(np.abs(I_obs)), I_err / np.abs(I_obs))
            dep_fit = weighted_linfit(freqs_ghz, p_frac, p_frac_err)
            rm_fit = weighted_linfit(lam2, X_obs_rad, X_err_rad)
            if alpha_fit is None or dep_fit is None or rm_fit is None:
                results_lines.append((color, f'{band_label}\n  fit failed (degenerate points)'))
                continue

            _, _, alpha_slope, alpha_slope_err = alpha_fit
            _, _, dep_slope, dep_slope_err = dep_fit
            _, _, rm_slope, rm_slope_err = rm_fit
            results_lines.append((color, format_band_result_latex(
                band_label, alpha_slope, alpha_slope_err, dep_slope, dep_slope_err, rm_slope, rm_slope_err)))

        self.canvas.set_measurement_data(p_bands)
        self.stokes_canvas.set_measurement_data(i_bands)
        self.set_spectral_params_bands(results_lines)
        self.meas_export_rows = export_rows
        self.export_meas_button.setEnabled(True)
        self.rmsynth_measurements_button.setEnabled(True)
        self.update_plot()

    def export_measurements_action(self):
        """Write the currently plotted simulated measurement points (one
        row per SPW, across every band) to a CSV -- the Measurements tab's
        counterpart to app.py's Save Spectra action, but for Generate's
        points rather than the continuous model curve."""
        if not self.meas_export_rows:
            return
        func, _spec, _pars, _wl_ext = self.current_state()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Measurements', f'{func.__name__}_measurements.csv', 'CSV files (*.csv)')
        if not path:
            return
        if not path.endswith('.csv'):
            path += '.csv'

        # Normalize I/Q/U (and their errors) so I=1 at the lowest frequency
        # point being saved, matching how Load Data normalizes real data
        # points before plotting them (see app.py.load_data_action's own
        # I0 = I[0] after sorting by frequency). p/EVPA are scale-invariant
        # ratios, so they're written out untouched, same as there.
        rows = self.meas_export_rows
        min_row = min(rows, key=lambda r: r[1])
        I0 = min_row[2]
        export_rows = [
            [band, freq, I / I0, I_err / I0, Q / I0, Q_err / I0, U / I0, U_err / I0,
             p, p_err, evpa_deg, evpa_err]
            for band, freq, I, I_err, Q, Q_err, U, U_err, p, p_err, evpa_deg, evpa_err in rows]

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Band', 'Frequency [GHz]', 'I', 'I_err', 'Q', 'Q_err', 'U', 'U_err',
                                  'p [%]', 'p_err [%]', 'EVPA [deg]', 'EVPA_err [deg]'])
                writer.writerows(export_rows)
        except OSError as e:
            QMessageBox.warning(self, 'Export Measurements', f'Could not save measurements file:\n{e}')
