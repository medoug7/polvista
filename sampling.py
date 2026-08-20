"""The "Sampling" tab: Bayesian (MultiNest) fitting UI and the Corner plot
tab it produces.

SamplingMixin is mixed into app.py's MainWindow (`class MainWindow(QMainWindow,
SamplingMixin)`) rather than composed as a separate widget/controller object,
since its methods reach deep into MainWindow's own state (sliders, canvases,
results box, plot_tabs) the same way the rest of MainWindow's methods do --
splitting this file out is purely about app.py's size, not about drawing a
real ownership boundary.
"""
import os

import numpy as np
import matplotlib as mpl
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QComboBox, QVBoxLayout, QHBoxLayout,
    QScrollArea, QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
    QPushButton, QFileDialog, QMessageBox, QStyle, QStyleOptionSlider,
    QStylePainter)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from polvista.models import MODELS, MODELS_BY_NAME
from polvista.fitting import multinest_fit, load_previous_run
from polvista.latex_stuff import latex_pixmap
from polvista.widgets import ValueLineEdit, NUMBER_RE, SLIDER_STEPS, UNITS, WIDEST_UNIT

# How many of a MultiNest run's winning-family posterior samples to draw as
# faint "spaghetti" lines behind the best-fit curve (see
# SamplingMixin.apply_posterior_result / app.ModelPlot.set_posterior_samples)
# -- enough to show the posterior's spread without cluttering the plot or
# costing much per redraw (each one is a full model() evaluation, refreshed
# on every slider drag).
N_POSTERIOR_DRAWS = 40

# Qualitative palette for non-winning mode families in the Corner plot tab
# (see SamplingMixin.build_corner_tab), assigned in evidence-rank order --
# mirrors qu_fit.py's corner_plot(); the winning family is always 'b'.
FAMILY_PALETTE = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def format_sci_title(latex, value, lo, hi):
    """'$\\phi=(1.23^{+0.45}_{-0.32})\\times10^{5}$'-style corner-panel
    title for a phi/dphi parameter: `value`/`lo`/`hi` (lo/hi already the
    *magnitudes* of the lower/upper uncertainty, not absolute bounds --
    see fitting.median_pctl_errs) in the parameter's own raw (unscaled,
    rad/m^2) units, with an exponent picked from `value`'s own magnitude
    so the mantissa lands in [1, 10) -- unlike the fixed 10^5 scaling
    every other phi/dphi panel content uses (see build_corner_tab's
    scale_map), which suits the shared plotted axis range but not a
    single title's own value, which can land anywhere from ~10^2 to
    ~10^6 depending on the parameter and the fit."""
    exp = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    scale = 10.0 ** exp
    v, l, h = value / scale, lo / scale, hi / scale
    name = latex.strip('$')
    return fr'${name}=({v:.2f}^{{+{h:.2f}}}_{{-{l:.2f}}})\times10^{{{exp}}}$'


class RangeSlider(QWidget):
    """A two-handle horizontal slider spanning integer raw values
    [0, steps], used by the Sampling tab (see SamplingBoundsRow) to pick a
    parameter's lower/upper prior bound -- ParamSlider only has one handle,
    for a single value. Qt has no built-in dual-handle slider, so this
    reuses the current style's own slider groove/handle drawing (via
    QStyle.CC_Slider) at two positions instead of hand-painting one."""
    valuesChanged = pyqtSignal(int, int)  # (lo_raw, hi_raw)

    def __init__(self, steps=SLIDER_STEPS, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.lo = 0
        self.hi = steps
        self.active = None  # 'lo' | 'hi' | None -- handle currently being dragged
        self.setMinimumHeight(24)
        self.setMinimumWidth(120)

    def values(self):
        return self.lo, self.hi

    def setValues(self, lo, hi):
        lo = max(0, min(int(round(lo)), self.steps))
        hi = max(0, min(int(round(hi)), self.steps))
        if lo > hi:
            lo, hi = hi, lo
        if (lo, hi) == (self.lo, self.hi):
            return
        self.lo, self.hi = lo, hi
        self.update()
        self.valuesChanged.emit(lo, hi)

    def sizeHint(self):
        return QSize(160, 24)

    def style_option(self):
        opt = QStyleOptionSlider()
        opt.initFrom(self)
        opt.minimum = 0
        opt.maximum = self.steps
        opt.orientation = Qt.Horizontal
        return opt

    def groove_rect(self):
        opt = self.style_option()
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)

    def handle_rect(self, value):
        opt = self.style_option()
        opt.sliderPosition = value
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

    def pos_to_value(self, x):
        groove = self.groove_rect()
        handle = self.handle_rect(0)
        usable = max(groove.width() - handle.width(), 1)
        frac = (x - groove.x() - handle.width() / 2) / usable
        frac = min(max(frac, 0.0), 1.0)
        return int(round(frac * self.steps))

    def paintEvent(self, event):
        painter = QStylePainter(self)
        groove_opt = self.style_option()
        groove_opt.subControls = QStyle.SC_SliderGroove
        painter.drawComplexControl(QStyle.CC_Slider, groove_opt)
        for val in (self.lo, self.hi):
            handle_opt = self.style_option()
            handle_opt.subControls = QStyle.SC_SliderHandle
            handle_opt.sliderPosition = val
            painter.drawComplexControl(QStyle.CC_Slider, handle_opt)

    def mousePressEvent(self, event):
        val = self.pos_to_value(event.x())
        # drag whichever handle is nearer the click
        self.active = 'lo' if abs(val - self.lo) <= abs(val - self.hi) else 'hi'
        self.drag_to(val)

    def mouseMoveEvent(self, event):
        if self.active is not None:
            self.drag_to(self.pos_to_value(event.x()))

    def mouseReleaseEvent(self, event):
        self.active = None

    def drag_to(self, val):
        if self.active == 'lo':
            self.setValues(val, self.hi)
        else:
            self.setValues(self.lo, val)


# Sampling-tab bounds sliders: generic (subscript-free) labels per parameter
# *kind*, since two-component models share one bounds row per kind rather
# than one per individual parameter -- see SamplingMixin.rebuild_sampling_bounds.
# X isn't included -- it has no bounds row at all, see there.
SAMPLING_KIND_LATEX = {'p': r'$p$', 'phi': r'$\phi$', 'dphi': r'$\sigma_{\phi}$'}

# Default prior bounds for the p bounds row, in the same physical units
# shown there (%). phi/dphi have no such default beyond "the whole
# slider" -- see SAMPLING_LINEAR_RANGE below.
SAMPLING_P_DEFAULT = (0.0, 30.0)     # %

# phi/dphi bounds rows can sample either linearly (physical rad/m^2,
# signed) or log-uniformly (log10 of the magnitude -- the sign is drawn
# separately at fit time, see qu_fit.py's multinest_fit prior()). These are
# the two extents the single shared "Log-uniform" checkbox above both rows
# (see SamplingMixin.rebuild_sampling_bounds) switches their sliders between;
# SAMPLING_LINEAR_RANGE also doubles as the linear mode's default (lo, hi),
# i.e. unconstrained until the user narrows it.
SAMPLING_LINEAR_RANGE = (-5e6, 5e6)
SAMPLING_LOG_RANGE = (0.0, 6.7)


class SamplingBoundsRow(QWidget):
    """One Sampling-tab row: a dual-handle RangeSlider plus lo/hi readouts,
    setting a parameter kind's lower/upper prior bound for MultiNest's
    initial sampling. Two-component models share a single row per kind
    (e.g. one 'p' row bounds both p_1 and p_2) since nothing so far
    constrains the two components separately.

    phi/dphi rows are log-capable: MainWindow toggles both between a linear
    (rad/m^2) and log10-magnitude range together via set_log_mode(), driven
    by a single checkbox shared above both rows rather than one per row --
    see SAMPLING_LINEAR_RANGE/SAMPLING_LOG_RANGE."""
    boundsChanged = pyqtSignal()

    def __init__(self, kind, latex, unit, extent, default, log_capable=False, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.log_capable = log_capable
        self.log_flag = False
        if log_capable:
            self.linear_extent = extent
            self.log_extent = SAMPLING_LOG_RANGE
        else:
            self.extent = extent

        # Two lines per row -- the slider on top, its lo/hi readouts below
        # -- rather than one long line, since a dual-handle slider plus two
        # value boxes doesn't fit the left panel's ~420px width on one line
        # the way ParamSlider's single value box does.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QHBoxLayout()
        self.name_label = QLabel()
        self.name_label.setPixmap(latex_pixmap(latex))
        self.name_label.setMinimumWidth(30)
        self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top.addWidget(self.name_label)

        self.slider = RangeSlider(SLIDER_STEPS)
        self.slider.valuesChanged.connect(self.on_slider_changed)
        # ~2/3 as wide as a stretch-filled slider would be -- this row's
        # value boxes+unit sit on their own line below (see `bottom`), so
        # the slider doesn't need to span the full row width to stay legible.
        self.slider.setMaximumWidth(230)
        top.addWidget(self.slider)
        top.addStretch(1)
        outer.addLayout(top)

        bottom = QHBoxLayout()
        bottom.addSpacing(30)  # align under the slider, past name_label's width

        self.lo_edit = ValueLineEdit()
        self.lo_edit.setFixedWidth(70)
        self.lo_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lo_edit.editingFinished.connect(self.on_lo_edited)
        self.hi_edit = ValueLineEdit()
        self.hi_edit.setFixedWidth(70)
        self.hi_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.hi_edit.editingFinished.connect(self.on_hi_edited)
        bottom.addWidget(self.lo_edit)
        bottom.addWidget(QLabel('-'))
        bottom.addWidget(self.hi_edit)

        self.unit_label = QLabel(unit)
        self.unit_label.setFixedWidth(self.unit_label.fontMetrics().horizontalAdvance(WIDEST_UNIT))
        bottom.addWidget(self.unit_label)

        bottom.addStretch(1)
        outer.addLayout(bottom)

        self.set_values(*default)

    @property
    def extent_now(self):
        if self.log_capable:
            return self.log_extent if self.log_flag else self.linear_extent
        return self.extent

    def raw_from_working(self, working):
        lo, hi = self.extent_now
        span = hi - lo
        frac = 0.0 if span == 0 else (working - lo) / span
        return int(round(frac * SLIDER_STEPS))

    def working_from_raw(self, raw):
        lo, hi = self.extent_now
        return lo + (hi - lo) * raw / SLIDER_STEPS

    def values(self):
        """(lo, hi) bounds, in the units shown -- % for p, rad/m^2 for
        phi/dphi in linear mode, or log10(rad/m^2) (decades, 0..6.7) for
        phi/dphi in log-uniform mode -- see set_log_mode."""
        raw_lo, raw_hi = self.slider.values()
        return self.working_from_raw(raw_lo), self.working_from_raw(raw_hi)

    def is_log(self):
        return self.log_capable and self.log_flag

    def set_values(self, lo, hi):
        self.slider.setValues(self.raw_from_working(lo), self.raw_from_working(hi))
        self.update_labels()

    def set_log_mode(self, is_log):
        """Switch a log-capable row between its linear and log10-magnitude
        extent, called externally by MainWindow's single shared "Log-uniform"
        checkbox (see rebuild_sampling_bounds) -- there's no principled way
        to carry a linear-mode selection over into log-magnitude space (or
        back), so switching modes just resets the row to that mode's full
        extent, the same "unconstrained until narrowed" default it starts
        with."""
        if not self.log_capable or is_log == self.log_flag:
            return
        self.log_flag = is_log
        lo, hi = self.extent_now
        self.set_values(lo, hi)
        self.boundsChanged.emit()

    def format(self, val):
        if self.kind in ('phi', 'dphi'):
            # Log-uniform mode's val is already the small (0..6.7) log10
            # exponent -- plain decimal reads better there; linear mode's
            # val is a raw rad/m^2 value up to +-5e6, where scientific
            # notation is more legible.
            return f'{val:.2f}' if self.is_log() else f'{val:.1e}'
        return f'{val:.2f}'

    def update_labels(self):
        lo, hi = self.values()
        self.lo_edit.setText(self.format(lo))
        self.hi_edit.setText(self.format(hi))

    def on_slider_changed(self, _lo_raw, _hi_raw):
        self.update_labels()
        self.boundsChanged.emit()

    def on_lo_edited(self):
        match = NUMBER_RE.search(self.lo_edit.text())
        if match is None:
            self.update_labels()
            return
        _, hi = self.values()
        self.set_values(float(match.group()), hi)
        self.boundsChanged.emit()

    def on_hi_edited(self):
        match = NUMBER_RE.search(self.hi_edit.text())
        if match is None:
            self.update_labels()
            return
        lo, _ = self.values()
        self.set_values(lo, float(match.group()))
        self.boundsChanged.emit()


class MultiNestWorker(QThread):
    """Runs fitting.multinest_fit() in a background thread so the UI stays
    responsive while MultiNest samples -- a real fit can take anywhere from
    seconds to minutes, and pymultinest.run() itself is a single blocking
    call with no way to yield control back to Qt's event loop.

    `progress`'s emit() is called from this thread (via
    fitting.multinest_fit()'s own dump_callback), but pyqtSignal delivers
    it to slots connected in the main thread via a queued connection
    automatically -- no manual locking needed on the caller's side."""
    progress = pyqtSignal(int, float, float)     # n_samples, logZ, logZerr
    finished_ok = pyqtSignal(list, list, dict)    # best_pars, errs, info
    failed = pyqtSignal(str)

    def __init__(self, kwargs, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self):
        try:
            best_pars, errs, info = multinest_fit(
                progress_callback=self.progress.emit, **self.kwargs)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(best_pars, errs, info)


# The corner figure's on-screen dpi is left at matplotlib's own default --
# raising it would inflate the (point-sized) tick/label/title text relative
# to the panels themselves, undoing the ndim-scaled font sizing in
# build_corner_tab. A PNG saved at that same screen dpi is fine at the size
# it's embedded at but turns blocky once viewed at full size/zoomed into
# later (more so for denser, higher-ndim plots, which cram more panels into
# the same on-screen pixel footprint -- see build_corner_tab's `size`
# comment), so CornerToolbar's Save button renders at this higher dpi
# instead, independent of whatever dpi the embedded canvas itself is using.
CORNER_SAVE_DPI = 300


class CornerToolbar(NavigationToolbar):
    """NavigationToolbar2QT for the Corner plot tab whose Save button
    exports at CORNER_SAVE_DPI regardless of the embedded figure's own
    (screen-matched) dpi -- see CORNER_SAVE_DPI.

    Goes through the 'savefig.dpi' rcParam rather than setting
    self.canvas.figure.dpi directly: the latter *should* be equivalent
    (Figure.savefig()'s own default dpi is 'figure', i.e. "use the
    figure's own dpi"), but on a figure whose canvas has never been
    drawn at a numeric dpi before (true here -- see build_corner_tab's
    own Agg probe, which never triggers a real draw at the canvas's
    *own* dpi), that 'figure' string fails to resolve to the just-set
    value on the very next savefig call and silently saves at the old
    dpi instead. Setting the rcParam sidesteps that resolution step
    entirely -- Figure.savefig() reads it directly."""
    def save_figure(self, *args):
        original_dpi = mpl.rcParams['savefig.dpi']
        mpl.rcParams['savefig.dpi'] = CORNER_SAVE_DPI
        try:
            super().save_figure(*args)
        finally:
            mpl.rcParams['savefig.dpi'] = original_dpi


class SamplingMixin:
    """Sampling-tab (MultiNest) UI/orchestration and the Corner plot tab,
    mixed into app.MainWindow -- see module docstring."""

    def build_sampling_tab(self):
        """Build the "Sampling" tab (MultiNest fit options: output dir, #
        live points, sampling efficiency, evidence tolerance, prior-bounds
        rows, Load samples) and add it to self.left_tabs. Only called from
        MainWindow.__init__ when HAS_PYMULTINEST is True."""
        self.sampling_tab = QWidget()
        sampling_layout = QVBoxLayout(self.sampling_tab)

        outdir_row = QHBoxLayout()
        outdir_row.addWidget(QLabel('Output directory:'))
        self.sampling_outdir_edit = QLineEdit()
        self.sampling_outdir_edit.setReadOnly(True)
        outdir_row.addWidget(self.sampling_outdir_edit, stretch=1)
        self.sampling_outdir_button = QPushButton('Browse...')
        self.sampling_outdir_button.clicked.connect(self.choose_sampling_outdir)
        outdir_row.addWidget(self.sampling_outdir_button)
        sampling_layout.addLayout(outdir_row)

        live_tip = ('Number of live points MultiNest maintains during nested sampling.\n'
                    'More live points resolve the posterior (and multiple modes) more finely \n'
                    'and give a more reliable evidence estimate, at the cost of more \n'
                    'likelihood evaluations (slower)')
        live_row = QHBoxLayout()
        # Plain text, like every other row's own label, with just the
        # n_live symbol itself rendered as mathtext (a second QLabel,
        # pixmap-only) tacked on at the end -- same split used for a
        # slider's own name_label/unit_label pair (see ParamSlider),
        # rather than rendering the whole "Number of live points" phrase
        # as one mathtext image.
        live_label = QLabel('Number of live points')
        live_label.setToolTip(live_tip)
        live_row.addWidget(live_label)
        live_symbol = QLabel()
        live_symbol.setPixmap(latex_pixmap(r'$n_\mathrm{live}$'))
        live_symbol.setToolTip(live_tip)
        live_row.addWidget(live_symbol)
        live_row.addStretch(1)
        self.sampling_n_live = QSpinBox()
        self.sampling_n_live.setRange(1, 10000)
        # Seeded from the current model's own preset (see
        # rebuild_sampling_bounds, which keeps this in sync on every later
        # model switch too) rather than one fixed default -- a 2-component
        # model's extra params (and its exchange-symmetry double mode, see
        # fitting.DEGENERATE_PAIR_MODELS) need more live points to resolve
        # reliably than a single-component one.
        self.sampling_n_live.setValue(MODELS[self.model_combo.currentData()].n_live_points)
        self.sampling_n_live.setToolTip(live_tip)
        live_row.addWidget(self.sampling_n_live)
        sampling_layout.addLayout(live_row)

        eff_tip = ('Target acceptance efficiency (0-1) for MultiNest\'s ellipsoidal sampling.\n'
                   'Lower values (~0.2-0.3, the recommended range for a reliable Bayesian evidence)\n '
                   'sample more conservatively; higher values (up to ~0.8) run faster but are \n'
                   'only recommended for quick the parameter estimates, not reliable evidences.')
        eff_row = QHBoxLayout()
        eff_label = QLabel('Sampling efficiency:')
        eff_label.setToolTip(eff_tip)
        eff_row.addWidget(eff_label)
        eff_row.addStretch(1)
        self.sampling_efficiency = QDoubleSpinBox()
        self.sampling_efficiency.setRange(0.01, 1)
        self.sampling_efficiency.setDecimals(2)
        self.sampling_efficiency.setSingleStep(0.01)
        self.sampling_efficiency.setValue(0.2)
        self.sampling_efficiency.setToolTip(eff_tip)
        eff_row.addWidget(self.sampling_efficiency)
        sampling_layout.addLayout(eff_row)

        tol_tip = ('Stopping criterion: MultiNest halts once its estimate of the\n'
                   'remaining (unsampled) evidence contribution falls below this\n'
                   'tolerance. Smaller values sample longer but converge the\n'
                   'evidence estimate more tightly.')
        tol_row = QHBoxLayout()
        tol_label = QLabel('Evidence tolerance:')
        tol_label.setToolTip(tol_tip)
        tol_row.addWidget(tol_label)
        tol_row.addStretch(1)
        self.sampling_evidence_tol = QDoubleSpinBox()
        self.sampling_evidence_tol.setRange(0.01, 1)
        self.sampling_evidence_tol.setDecimals(2)
        self.sampling_evidence_tol.setSingleStep(0.01)
        self.sampling_evidence_tol.setValue(0.5)
        self.sampling_evidence_tol.setToolTip(tol_tip)
        tol_row.addWidget(self.sampling_evidence_tol)
        sampling_layout.addLayout(tol_row)

        # Prior-bounds sliders -- one row per parameter *kind* (see
        # rebuild_sampling_bounds): two-component models share a single
        # row across both components, since nothing so far constrains
        # them separately. Spectral params (alpha, eps) are excluded --
        # QU data alone can't constrain them, so fitting holds epsilon
        # fixed at 0.5 instead of sampling it (see app.FIT_FIXED_EPSILON).
        sampling_layout.addWidget(QLabel('<b>Parameter bounds</b>'))
        self.sampling_bounds_box = QWidget()
        self.sampling_bounds_layout = QVBoxLayout(self.sampling_bounds_box)
        self.sampling_bounds_layout.setSpacing(8)
        self.sampling_bounds_layout.addStretch(1)
        sampling_bounds_scroll = QScrollArea()
        sampling_bounds_scroll.setWidgetResizable(True)
        sampling_bounds_scroll.setWidget(self.sampling_bounds_box)
        sampling_layout.addWidget(sampling_bounds_scroll, stretch=1)

        # Bottom-left of the tab (the bounds scroll area above already
        # claims all remaining vertical space via stretch=1, so this
        # row lands at the very bottom).
        load_samples_row = QHBoxLayout()
        self.sampling_load_button = QPushButton('Load samples')
        self.sampling_load_button.setToolTip(
            "Load a previous MultiNest run's output, re-cluster its\n"
            'modes, and show the winning family (parameters, posterior\n'
            'overlay, corner plot) -- same as a fresh Fit! run.')
        self.sampling_load_button.clicked.connect(self.load_samples_action)
        load_samples_row.addWidget(self.sampling_load_button)
        load_samples_row.addStretch(1)
        sampling_layout.addLayout(load_samples_row)

        self.left_tabs.addTab(self.sampling_tab, 'Sampling')
        self.left_tabs.setTabToolTip(
            self.left_tabs.indexOf(self.sampling_tab),
            'Load data first (Load data button) to enable Bayesian sampling.')
        # MultiNest fits against loaded q/u data (see run_multinest_fit) --
        # greyed out until Load data succeeds, see MainWindow.load_data_action
        # / clear_data.
        self.set_sampling_tab_enabled(False)

    def set_sampling_tab_enabled(self, enabled):
        """Enable/disable (grey out) the Sampling tab. No-op if pymultinest
        isn't installed (no Sampling tab was ever built, see HAS_PYMULTINEST)."""
        if not hasattr(self, 'sampling_tab'):
            return
        idx = self.left_tabs.indexOf(self.sampling_tab)
        if idx == -1:
            return
        if not enabled and self.left_tabs.currentIndex() == idx:
            self.left_tabs.setCurrentWidget(self.param_box)  # back to Visualization
        self.left_tabs.setTabEnabled(idx, enabled)

    def choose_sampling_outdir(self):
        directory = QFileDialog.getExistingDirectory(
            self, 'Select output directory', self.sampling_outdir_edit.text() or os.getcwd())
        if directory:
            self.sampling_outdir_edit.setText(directory)

    def rebuild_sampling_bounds(self):
        """(Re)build the Sampling tab's prior-bounds rows for the current
        model -- one SamplingBoundsRow per parameter *kind* present, in the
        fixed order p, scale (e.g. s/f), phi, dphi -- skipping spectral
        kinds (alpha, eps, held at FIT_FIXED_EPSILON/unused) and X (always
        sampled over its full -90..90 deg range, so it gets no bounds row
        at all). Two-component models collapse duplicate kinds down to a
        single shared row (see SamplingBoundsRow).

        phi and dphi share one "Log-uniform" checkbox, shown once above
        both rows rather than per-row, that switches them both between a
        linear and log10-magnitude sampling range together.

        Also resets the # of live points spin box to the new model's own
        preset (see build_sampling_tab/models.ModelSpec.n_live_points),
        same as the bounds rows themselves -- any manual override the user
        had set for the *previous* model doesn't carry over, matching how
        the bounds rows below are unconditionally rebuilt from scratch too
        rather than trying to preserve a per-model customization across a
        model switch."""
        if not hasattr(self, 'sampling_bounds_layout'):
            return  # HAS_PYMULTINEST is False -- no Sampling tab was built

        for row in getattr(self, 'sampling_bounds_rows', {}).values():
            row.setParent(None)
        self.sampling_bounds_rows = {}
        if hasattr(self, 'sampling_log_checkbox'):
            self.sampling_log_checkbox.setParent(None)
            del self.sampling_log_checkbox

        spec = MODELS[self.model_combo.currentData()]
        lo_bounds, hi_bounds = spec.bounds
        self.sampling_n_live.setValue(spec.n_live_points)

        self.sampling_bounds_layout.takeAt(self.sampling_bounds_layout.count() - 1)

        first_by_kind = {}
        for i, param in enumerate(spec.params):
            if param.kind in ('alpha', 'eps', 'X') or param.kind in first_by_kind:
                continue
            first_by_kind[param.kind] = (param, lo_bounds[i], hi_bounds[i])

        # 'p' and any 'scale' kind(s) first, then phi/dphi (with their
        # shared log-uniform checkbox) last, regardless of the model's own
        # params order.
        kind_order = sorted(first_by_kind, key=lambda k: {'p': 0, 'phi': 2, 'dphi': 2}.get(k, 1))

        has_phi_or_dphi = any(k in ('phi', 'dphi') for k in kind_order)
        if has_phi_or_dphi:
            self.sampling_log_checkbox = QCheckBox('Log-uniform (φ, σ_φ)')
            self.sampling_log_checkbox.setToolTip(
                'Sample φ and σ_φ log-uniformly in magnitude (sign drawn '
                'separately) instead of uniformly in rad/m².')
            # Log-uniform is the sensible default for phi/dphi, whose
            # physically plausible range spans several decades -- a plain
            # linear prior over that range wastes almost all its mass at
            # the largest magnitudes.
            self.sampling_log_checkbox.setChecked(True)
            self.sampling_log_checkbox.stateChanged.connect(self.on_sampling_log_toggled)

        for kind in kind_order:
            if kind in ('phi', 'dphi') and hasattr(self, 'sampling_log_checkbox') \
                    and self.sampling_bounds_layout.indexOf(self.sampling_log_checkbox) == -1:
                self.sampling_bounds_layout.addWidget(self.sampling_log_checkbox)
            param, lo, hi = first_by_kind[kind]
            row = self.make_sampling_bounds_row(kind, param, lo, hi)
            self.sampling_bounds_layout.addWidget(row)
            self.sampling_bounds_rows[kind] = row

        # Rows are always freshly built in linear mode (SamplingBoundsRow's
        # own default) regardless of the checkbox above -- sync them to it
        # now that they exist, same handler the checkbox's own toggle uses.
        if has_phi_or_dphi:
            self.on_sampling_log_toggled(None)

        self.sampling_bounds_layout.addStretch(1)

    def on_sampling_log_toggled(self, _state):
        is_log = self.sampling_log_checkbox.isChecked()
        for kind in ('phi', 'dphi'):
            row = self.sampling_bounds_rows.get(kind)
            if row is not None:
                row.set_log_mode(is_log)

    def make_sampling_bounds_row(self, kind, param, lo, hi):
        if kind == 'p':
            return SamplingBoundsRow('p', SAMPLING_KIND_LATEX['p'], '%',
                                      (lo * 100, hi * 100), SAMPLING_P_DEFAULT)
        if kind in ('phi', 'dphi'):
            return SamplingBoundsRow(kind, SAMPLING_KIND_LATEX[kind], UNITS[kind],
                                      SAMPLING_LINEAR_RANGE, SAMPLING_LINEAR_RANGE, log_capable=True)
        # 'scale' -- e.g. s (Tribble), f (partial coverage): no specific
        # default given, so it starts unconstrained across the model's own
        # registered bounds.
        return SamplingBoundsRow(kind, param.latex, UNITS['scale'], (lo, hi), (lo, hi))

    def run_multinest_fit(self):
        """Bayesian (MultiNest) fit for the Sampling tab's Fit! button.

        First runs the ordinary least-squares pre-fit (fit_spectrum_lsq)
        -- this fits alpha/epsilon (which QU-only data can't otherwise
        constrain) and pushes an immediate baseline result to the sliders
        and Results box, so both are already meaningful (and saveable)
        before the -- typically much slower -- MultiNest run even starts.

        MultiNest itself then always samples every p/X/phi/dphi/scale-kind
        parameter of the model (never alpha/epsilon, and regardless of any
        individual slider's own 'fixed' checkbox -- see
        fitting.multinest_fit's docstring), using the Sampling tab's own
        prior bounds (self.sampling_bounds_rows / sampling_log_checkbox)
        and settings (# live points, evidence tolerance, output dir), run
        in a background MultiNestWorker thread so the UI stays responsive.
        On completion (on_mn_finished) it runs the same mode-family
        clustering qu_fit.py does to pick the best solution family, and
        pushes that family's parameters into the sliders -- and so the
        plots -- exactly like the least-squares path already does."""
        if self.fit_data is None:
            return
        outdir = self.sampling_outdir_edit.text()
        if not outdir:
            QMessageBox.warning(self, 'Sampling', 'Choose an output directory first.')
            return

        result = self.fit_spectrum_lsq()
        if result is None:
            return
        best_pars_lsq, stats = result
        self.results_label.setText(
            'Pre-fit spectrum (α, ε):\n' + self.format_fit_stats(stats) + '\n\nSampling...')

        func, spec, _, _ = self.current_state()
        wl, q, q_err, u, u_err, freq, I = self.fit_data
        spectral_pars = {i: best_pars_lsq[i] for i in spec.indices('alpha') + spec.indices('eps')}

        is_log = hasattr(self, 'sampling_log_checkbox') and self.sampling_log_checkbox.isChecked()
        kind_bounds = {}
        for kind, row in self.sampling_bounds_rows.items():
            lo, hi = row.values()
            if kind == 'p':
                kind_bounds['p'] = (lo / 100.0, hi / 100.0)  # % -> fraction
            elif kind in ('phi', 'dphi'):
                kind_bounds[kind] = (lo, hi, is_log)
            else:
                kind_bounds[kind] = (lo, hi)

        basename = os.path.join(outdir, func.__name__, 'mn_')

        # Sidecar saved alongside MultiNest's own output files -- lets Load
        # samples (load_samples_action) later reconstruct this exact run
        # (model, the alpha/epsilon values held fixed, and the q/u data it
        # was fit against) without needing that data to still be loaded.
        os.makedirs(os.path.dirname(basename), exist_ok=True)
        np.savez(basename + 'polvista_meta.npz',
                  model=func.__name__,
                  spectral_idx=np.array(list(spectral_pars.keys())),
                  spectral_val=np.array(list(spectral_pars.values())),
                  wl=wl, q=q, q_err=q_err, u=u, u_err=u_err)

        self.fit_button.setEnabled(False)
        self.load_data_button.setEnabled(False)
        self.clear_data_button.setEnabled(False)
        self.sampling_load_button.setEnabled(False)
        self.sampling_progress.setVisible(True)
        self.sampling_progress_label.setVisible(True)
        self.sampling_progress_label.setText('Starting MultiNest...')

        self.mn_model = func
        self.mn_worker = MultiNestWorker(dict(
            wl=wl, q=q, q_err=q_err, u=u, u_err=u_err, model=func,
            spectral_pars=spectral_pars, kind_bounds=kind_bounds,
            outputfiles_basename=basename,
            n_live_points=self.sampling_n_live.value(),
            sampling_efficiency=self.sampling_efficiency.value(),
            evidence_tolerance=self.sampling_evidence_tol.value(),
        ))
        self.mn_worker.progress.connect(self.on_mn_progress)
        self.mn_worker.finished_ok.connect(self.on_mn_finished)
        self.mn_worker.failed.connect(self.on_mn_failed)
        self.mn_worker.start()

    def on_mn_progress(self, n_samples, logZ, logZerr):
        self.sampling_progress_label.setText(
            f'{n_samples} samples evaluated -- ln Z = {logZ:.4g} ± {logZerr:.2g}')

    def on_mn_finished(self, best_pars, errs, info):
        self.sampling_progress.setVisible(False)
        self.sampling_progress_label.setVisible(False)
        self.fit_button.setEnabled(True)
        self.load_data_button.setEnabled(True)
        self.clear_data_button.setEnabled(True)
        self.sampling_load_button.setEnabled(True)
        for sl, val in zip(self.sliders, best_pars):
            sl.set_value(val)
        self.results_label.setText(self.format_mn_stats(info))
        self.apply_posterior_result(self.mn_model, best_pars, errs, info)
        self.mn_worker = None

    def on_mn_failed(self, msg):
        self.sampling_progress.setVisible(False)
        self.sampling_progress_label.setVisible(False)
        self.fit_button.setEnabled(True)
        self.load_data_button.setEnabled(True)
        self.clear_data_button.setEnabled(True)
        self.sampling_load_button.setEnabled(True)
        QMessageBox.warning(self, 'Sampling', f'MultiNest fit failed:\n{msg}')
        self.mn_worker = None

    def format_mn_stats(self, info):
        """Same field set/format as format_fit_stats() (chi2, dof,
        chi2_red, ln L, AIC, AICc, BIC -- see fitting.assemble_result,
        which computes them the same way fitting.fit_statistics() does for
        least-squares, from the winning family's own chi2/lnL), so the
        Results box reads the same regardless of which fit method was
        used. The winning-family share and a combined other-families/
        dropped-modes line come first (mode-family information -- not
        applicable to least-squares -- grouped together right below the
        winner line, ahead of the shared stats), and MultiNest's own
        global evidence trails at the end."""
        others = info['other_families']
        dropped = info['dropped']
        aicc = f"{info['aicc']:.4g}" if np.isfinite(info['aicc']) else 'n/a (dof ≤ 1)'

        family_bits = []
        if others:
            family_bits.append(f"{len(others)} other family(ies)")
        if dropped['count']:
            family_bits.append(f"{dropped['count']} negligible mode(s) dropped "
                                f"({dropped['evidence_share']:.2g}% evidence)")

        lines = [f"winning family: {info['evidence_share']:.2f}% evidence share"]
        if family_bits:
            lines.append('; '.join(family_bits))
        lines += [
            f"χ² = {info['chi2']:.4g}   dof = {info['dof']}",
            f"χ²_red = {info['chi2_red']:.4g}",
            f"ln L = {info['loglike']:.4g}",
            f"AIC = {info['aic']:.4g}   AICc = {aicc}",
            f"BIC = {info['bic']:.4g}",
            f"global ln Z = {info['global_evidence']:.4g} ± {info['global_evidence_err']:.3g}",
        ]
        return '\n'.join(lines)

    # ── MultiNest result plumbing (posterior overlay + Corner plot tab) ──────
    # Shared by a fresh fit (on_mn_finished) and Load samples
    # (load_samples_action), which both end up with the same `info` shape
    # (see fitting.multinest_fit / fitting.load_previous_run).
    def apply_posterior_result(self, model, best_pars, errs, info):
        """Stash a random subset of the winning family's posterior (for the
        faint spaghetti overlay on the Visualization tab's plots -- see
        app.ModelPlot/StokesPlot.set_posterior_samples) and (re)build the
        Corner plot tab from the full result."""
        spec = MODELS[model]
        pol_idx = info['pol_idx']
        spectral_pars = {i: best_pars[i] for i in spec.indices('alpha') + spec.indices('eps')}
        winner_samples = info['winner_samples']  # (n, len(pol_idx)), pol_idx-local order

        rng = np.random.default_rng(0)
        n_draws = min(N_POSTERIOR_DRAWS, len(winner_samples))
        subset = winner_samples[rng.choice(len(winner_samples), size=n_draws, replace=False)]

        full_samples = np.zeros((n_draws, len(spec.params)))
        full_samples[:, pol_idx] = subset
        for i, v in spectral_pars.items():
            full_samples[:, i] = v

        self.posterior_samples = full_samples
        self.posterior_model = model
        self.canvas.set_posterior_samples(full_samples, model)
        self.stokes_canvas.set_posterior_samples(full_samples, model)

        self.build_corner_tab(model, best_pars, errs, info)
        self.update_plot()

    def clear_posterior_samples(self):
        self.posterior_samples = None
        self.posterior_model = None
        self.canvas.set_posterior_samples(None, None)
        self.stokes_canvas.set_posterior_samples(None, None)

    def build_corner_tab(self, model, best_pars, errs, info):
        """(Re)build the 'Corner plot' tab: the winning family's own pooled
        posterior (blue, opaque) plus up to 4 other mode families overlaid
        in their own color at alpha=0.5 (if MultiNest found more than one),
        with the winning point-estimate drawn as dashed truths= crosshairs
        and a legend of each family's evidence share/ln Z/chi2 -- mirrors
        qu_fit.py's corner_plot(). Removed by Reset model or Clear data
        (see remove_corner_tab)."""
        try:
            import corner
            import matplotlib.colors as mcolors
            import matplotlib.patches as mpatches
        except ImportError:
            QMessageBox.warning(self, 'Corner plot',
                                 "The 'corner' package isn't installed -- skipping the corner plot.")
            return

        self.remove_corner_tab()

        spec = MODELS[model]
        pol_idx = info['pol_idx']
        kinds_pol = [spec.params[i].kind for i in pol_idx]
        labels = [spec.params[i].latex for i in pol_idx]
        scale_map = {'p': 100.0, 'X': 180.0 / np.pi, 'phi': 1e-5, 'dphi': 1e-5, 'scale': 1.0}
        scales = np.array([scale_map[k] for k in kinds_pol])

        winner_samples = info['winner_samples'] * scales[np.newaxis, :]
        truths = [best_pars[i] * s for i, s in zip(pol_idx, scales)]
        other_families = info['other_families']
        dropped = info['dropped']

        # A bare (non-pyplot) Figure -- corner.corner() builds its KxK axes
        # grid directly onto it (Figure.subplots()) when passed an empty
        # one, so this never touches pyplot's global figure registry.
        #
        # The figure's *inch* size barely matters once embedded: Qt's
        # FigureCanvasQT.resizeEvent() calls figure.set_size_inches() on
        # every resize to match the actual on-screen widget size, at a
        # fixed dpi -- so whatever figsize is picked here gets shrunk (or
        # grown) to fit the tab's real, roughly ndim-independent pixel
        # footprint regardless. Font sizes are in points, not inches, so
        # they *don't* shrink along with it -- more panels (larger ndim)
        # squeezed into that same fixed on-screen space means each panel
        # gets proportionally less room while text stays full size, which
        # is what caused two-component (ndim=8) corner plots to overlap
        # badly even though single-component (ndim=4) ones looked fine.
        # Scaling label/title/tick fontsize down as ndim grows past 4
        # keeps each panel's text-to-panel-size ratio roughly constant
        # instead of only fixing it for whatever ndim happened to be
        # tested.
        ndim = len(pol_idx)
        size = max(2.2 * ndim + 1.5, 6.0)
        fig = Figure(figsize=(size, size))

        scale = min(1.0, 4.0 / ndim)
        base_fs = mpl.rcParams['font.size']
        label_fs = max(6.0, base_fs * scale)
        title_fs = max(6.0, base_fs * 1.2 * scale)
        tick_fs = max(5.0, base_fs * scale)
        max_n_ticks = 5 if ndim <= 5 else (4 if ndim <= 7 else 3)
        label_kwargs = {'fontsize': label_fs}
        title_kwargs = {'fontsize': title_fs}
        # corner.corner() places each bottom-row axis label at -0.3 axes-
        # fraction below its own axes (ax.xaxis.set_label_coords) -- too
        # close to the tick numbers directly above it once those ticks
        # have any real height, so the two overlap. This isn't actually an
        # ndim-driven effect (measured overlap was, if anything, *worse*
        # at ndim=4 than ndim=8: our tick/label fontsizes already scale
        # together with panel size -- see the ndim comment above -- so the
        # ratio between "gap available" and "tick height" stays roughly
        # constant across model sizes), so a single flat labelpad works
        # for every model rather than needing its own ndim formula; 0.13
        # was picked by rendering both a 4- and an 8-parameter model and
        # measuring the actual tick-label/axis-label pixel gap until
        # neither overlapped, with a few pixels to spare.
        labelpad = 0.15

        ranked_colors = [FAMILY_PALETTE[i % len(FAMILY_PALETTE)] for i in range(len(other_families))]
        # Weakest-evidence family drawn first, winner last, so an
        # overlapping higher-evidence contour is never hidden underneath a
        # lower-evidence one.
        draw_order = sorted(zip(other_families, ranked_colors), key=lambda fc: fc[0]['evidence_share'])

        for family, color in draw_order:
            fam_samples = family['samples'] * scales[np.newaxis, :]
            faded_color = mcolors.to_rgba(color, alpha=0.5)
            corner.corner(
                fam_samples, fig=fig, labels=labels, plot_datapoints=False,
                color=faded_color, levels=(0.5, 0.8, 0.95),
                hist_kwargs={'color': color, 'alpha': 0.5},
                contour_kwargs={'colors': 'k', 'alpha': 0.5},
                fill_contours=True, smooth=1.0, show_titles=False,
                label_kwargs=label_kwargs, max_n_ticks=max_n_ticks, labelpad=labelpad)

        corner.corner(
            winner_samples, fig=fig, show_titles=True, labels=labels,
            plot_datapoints=False, color='b', levels=(0.5, 0.8, 0.95),
            hist_kwargs={'color': 'b'}, contour_kwargs={'colors': 'k'},
            fill_contours=True, smooth=1.0, title_fmt='.2f',
            label_kwargs=label_kwargs, title_kwargs=title_kwargs,
            max_n_ticks=max_n_ticks, labelpad=labelpad)

        for ax in fig.axes:
            ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        # phi/dphi's raw physical scale (~1e4-1e6 rad/m^2) doesn't suit a
        # fixed '10^5'-scaled decimal (see scale_map above, which exists
        # for the *panel contents* -- histograms/contours need one shared
        # scale per axis to be legible at all) -- their titles instead get
        # their own per-parameter exponent picked from that title's own
        # center value, overwriting whatever corner.corner()'s generic
        # title_fmt wrote for just those two diagonal panels (fig.axes'
        # flattened K*K layout puts panel (i,i) at index i*ndim+i -- see
        # corner.core._get_fig_axes, which reads it back the same way).
        for local_i, kind in enumerate(kinds_pol):
            if kind in ('phi', 'dphi'):
                orig_i = pol_idx[local_i]
                lo, hi = errs[orig_i]
                title = format_sci_title(labels[local_i], best_pars[orig_i], lo, hi)
                fig.axes[local_i * ndim + local_i].set_title(title, **title_kwargs)

        corner.overplot_lines(fig, truths, color='k', linestyle='--', alpha=0.4)
        corner.overplot_points(fig, [truths], marker='s', color='k', alpha=0.4)

        if other_families:
            legend_handles = [mpatches.Patch(
                color='b',
                label=fr"mass={info['evidence_share']:.1f}%, $\ln Z$={info['winner_lnZ']:.2f}, "
                      fr"$\chi_\nu^2$={info['winner_chi2']:.2f} (winner)")]
            for family, color in zip(other_families, ranked_colors):
                legend_handles.append(mpatches.Patch(
                    color=color, alpha=0.5,
                    label=fr"mass={family['evidence_share']:.1f}%, $\ln Z$={family['lnZ']:.2f}, "
                          fr"$\chi_\nu^2$={family['chi2']:.2f}"))
            if dropped['count'] > 0:
                legend_handles.append(mpatches.Patch(
                    color='none', label=f"+{dropped['count']} more, {dropped['evidence_share']:.1f}% combined"))
            fig.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)

        # corner.corner() picks its own left/bottom/top margins from a
        # fixed formula (based on its own internal `dim`, unaware of our
        # ndim-scaled font sizes above), sized for its own default fonts --
        # too tight once our larger ndim=4 fonts are in play, and (in the
        # other direction) far too generous once our smaller ndim=8 fonts
        # shrink to fit, wasting panel space on blank margin that grows
        # right along with the figure's own inch size (see `size` above)
        # even as the actual text needing room for shrinks. Rather than
        # hand-tuning another ndim-keyed formula, measure what's actually
        # rendered (tick labels, axis labels, the phi/dphi titles just
        # overwritten above, the legend) via a throwaway Agg renderer, and
        # grow the margins by exactly however far that content overflows
        # the current figure bounds -- self-correcting for any ndim, font
        # size, or label content instead of guessing.
        #
        # pad_in itself (the safety margin added *beyond* that measured
        # overflow) grows a little past ndim=4 rather than staying flat:
        # denser plots pack more (smaller, tighter-spaced) tick/axis
        # labels along each edge, so the same flat pad reads as
        # comfortable headroom at ndim=4 but noticeably tighter once
        # there's more of that small text lined up along it -- barely
        # different for a 4-parameter model like burn, a bit more
        # breathing room for an 8-parameter one like comp2RMdep.
        probe = FigureCanvasAgg(fig)
        probe.draw()
        bbox = fig.get_tightbbox(probe.get_renderer())
        pad_in = 0.06 + 0.025 * max(0, ndim - 4)
        sp = fig.subplotpars
        fig.subplots_adjust(
            left=sp.left + max(0.0, pad_in - bbox.x0) / size,
            bottom=sp.bottom + max(0.0, pad_in - bbox.y0) / size,
            top=sp.top - max(0.0, bbox.y1 - size + pad_in) / size,
            right=sp.right - max(0.0, bbox.x1 - size + pad_in) / size)

        canvas = FigureCanvas(fig)
        toolbar = CornerToolbar(canvas, self)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, stretch=1)

        self.corner_tab = tab
        self.plot_tabs.addTab(tab, 'Corner plot')
        self.plot_tabs.setCurrentWidget(tab)

    def remove_corner_tab(self):
        tab = self.corner_tab
        if tab is None:
            return
        idx = self.plot_tabs.indexOf(tab)
        if idx != -1:
            self.plot_tabs.removeTab(idx)
        tab.deleteLater()
        self.corner_tab = None

    def load_samples_action(self):
        """Load a previous MultiNest run's output (picked via a directory
        dialog), re-run the same mode-family clustering fitting.load_previous_run
        does after a fresh fit, and apply the result exactly like a fresh
        MultiNest fit would (sliders, posterior overlay, Corner plot tab --
        see apply_posterior_result).

        The chosen directory must contain the 'mn_polvista_meta.npz' sidecar
        run_multinest_fit saves alongside its own MultiNest output (model
        name, the alpha/epsilon values held fixed during that run, and the
        exact q/u data it was fit against) -- this makes Load samples fully
        self-contained, independent of whatever data (if any) happens to be
        currently loaded via Load data."""
        start_dir = self.sampling_outdir_edit.text() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(self, 'Load MultiNest samples', start_dir)
        if not directory:
            return

        meta_path = os.path.join(directory, 'mn_polvista_meta.npz')
        if not os.path.exists(meta_path):
            QMessageBox.warning(
                self, 'Load samples',
                f'No polvista metadata found in:\n{directory}\n\n'
                "(expected 'mn_polvista_meta.npz', written by this app's own Sampling-tab "
                'Fit! runs -- pick the folder one was saved to.)')
            return

        try:
            meta = np.load(meta_path)
            model_name = str(meta['model'])
            if model_name not in MODELS_BY_NAME:
                raise ValueError(f"unknown model '{model_name}' in metadata")
            model = MODELS_BY_NAME[model_name]
            spectral_pars = {int(i): float(v) for i, v in zip(meta['spectral_idx'], meta['spectral_val'])}
            wl, q, q_err, u, u_err = meta['wl'], meta['q'], meta['q_err'], meta['u'], meta['u_err']

            basename = os.path.join(directory, 'mn_')
            best_pars, errs, info = load_previous_run(wl, q, q_err, u, u_err, model, spectral_pars, basename)
        except Exception as e:
            QMessageBox.warning(self, 'Load samples', f'Failed to load samples:\n{e}')
            return

        index = self.model_combo.findData(model)
        if index != -1:
            self.model_combo.setCurrentIndex(index)  # triggers rebuild_sliders if it changed
        for sl, val in zip(self.sliders, best_pars):
            sl.set_value(val)
        self.results_label.setText(self.format_mn_stats(info))
        self.apply_posterior_result(model, best_pars, errs, info)
