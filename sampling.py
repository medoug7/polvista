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
import threading
import traceback

import numpy as np
import matplotlib as mpl
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QComboBox, QVBoxLayout, QHBoxLayout,
    QScrollArea, QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
    QPushButton, QFileDialog, QMessageBox, QStyle, QStyleOptionSlider,
    QStylePainter)

import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from polvista.models import MODELS, MODELS_BY_NAME
from polvista.fitting import multinest_fit, load_previous_run, expand_pars_errs, estimate_alpha, pol_idx, wrap_pm_halfpi
from polvista.latex_stuff import latex_pixmap
from polvista.widgets import ValueLineEdit, NUMBER_RE, SLIDER_STEPS, UNITS, WIDEST_UNIT


# ── qu_fit.py sample import (load_samples_action's fallback for a directory
# with no mn_polvista_meta.npz sidecar) ─────────────────────────────────────
# Maps a ~/Downloads/pipe/qu_fit.py MultiNest output directory name (the
# reference pipeline this app's Sampling tab was distilled from -- see
# fitting.py's module docstring) to (this app's own model name, the
# param indices qu_fit.py held fixed at 0 rather than sampling, and a
# fitting.load_previous_run degenerate_pair override -- None keeps the
# default; see its docstring). The two-component entries map to this app's
# *_legacy models (qu_fit.py's plain-sum convention, not this app's own
# eps-weighted spectral_combine -- see comp2RMdep_legacy's docstring), and
# _load_qu_fit_samples re-derives alpha/epsilon (never sampled by
# qu_fit.py) the same way a fresh Sampling-tab pre-fit would. '2_Int'
# (comp2intern) is unsupported: qu_fit.py's own MODELS registration for it
# is internally inconsistent (6 vs. 8 params), so there's no way to tell
# which convention a real output directory would use.
QU_FIT_MODEL_MAP = {
    'Burn':          ('burn',               [],     None),
    'Internal':      ('intern',             [],     None),
    'Partial':       ('partial',            [],     None),
    'Tribble':       ('tribble',            [],     None),
    '2_Ext':         ('comp2RMdep_legacy',  [3, 7], None),
    '2_Ext_dep':     ('comp2RMdep_legacy',  [],     None),
    '2_Ext_dep1':    ('comp2RMdep_legacy',  [7],    False),
    'Int_Ext':       ('comp2mixdep_legacy', [3, 7], None),
    'Int_Ext_dep':   ('comp2mixdep_legacy', [],     True),
    'Int_Ext_dep1':  ('comp2mixdep_legacy', [7],    False),
    'Int_Ext_dep2':  ('comp2mixdep_legacy', [3],    False),
}


def warm_up_sampling_imports():
    """Import pymultinest and corner on a throwaway background thread,
    called once from app.main() right after the main window is shown, so
    their surprisingly heavy cold-import cost (pymultinest pulls in
    matplotlib.pyplot's ~100 Artist-subclass registrations; corner pulls in
    scipy -- neither releases the GIL) doesn't stall the UI the first time
    a background QThread (MultiNestWorker/LoadSamplesWorker/
    CornerBuildWorker) needs them. Both remain lazily imported elsewhere
    too, so a polvista install missing either optional dependency still
    doesn't fail at startup -- ImportError here is silently swallowed the
    same way those call sites already handle it."""
    def _warm():
        try:
            import pymultinest  # noqa: F401
        except ImportError:
            pass
        try:
            import corner  # noqa: F401
            import matplotlib.colors  # noqa: F401
            import matplotlib.patches  # noqa: F401
        except ImportError:
            pass
    threading.Thread(target=_warm, daemon=True).start()


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
FAMILY_PALETTE = ['#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                   '#999999', '#e41a1c', '#dede00']

# Gap (points, not a fraction of the panel -- see build_corner_tab's own
# use of this) between each panel's rotated tick numbers and its own axis
# label (x on the bottom row, y on the left column). Both x and y share
# one value since matplotlib's own native, tick-aware auto-positioning
# (build_corner_tab re-enables it, undoing corner.corner()'s own fixed
# axes-*fraction* placement) measures the actual rendered tick-label
# extent itself at every redraw -- unlike that fraction-based placement,
# a single points value here already reads the same regardless of ndim,
# font scaling, or how many digits a given fit's tick numbers happen to
# have, so there's no separate x/y or per-model tuning to do.
CORNER_LABELPAD_PT = 3.0

# dpi for fit_corner_label_ink_pad's own throwaway probe render -- kept far
# below CORNER_SAVE_DPI (600, used only for the actual exported PNG)
# because that function's own result is expressed in *points*, not pixels
# (see its docstring), so it's dpi-independent: this only trades a little
# sub-pixel precision in the measured ink gap (still well under a point at
# this dpi) for a probe render that's over an order of magnitude cheaper --
# significant since build_corner_tab's own figures can be a foot or more
# across (see `size` there) and this probe redraws the whole thing.
CORNER_INK_PROBE_DPI = 150


def sci_exponent(value):
    """Power-of-ten exponent such that value/10**exponent lands in [1, 10)
    (0 for value==0) -- used by build_corner_tab to auto-scale each phi/
    dphi parameter's own corner column/row (axis label and panel content
    alike) by its own magnitude instead of a fixed 10^5, and by
    format_sci_title for that same panel's title, so both agree."""
    return int(np.floor(np.log10(abs(value)))) if value != 0 else 0


def format_sci_title(latex, value, lo, hi, exp):
    """'$\\phi=(1.23^{+0.45}_{-0.32})\\times10^{5}$'-style corner-panel
    title for a phi/dphi parameter: `value`/`lo`/`hi` (lo/hi already the
    *magnitudes* of the lower/upper uncertainty, not absolute bounds --
    see fitting.median_pctl_errs) in the parameter's own raw (unscaled,
    rad/m^2) units, scaled by 10**`exp` (see sci_exponent -- build_corner_tab
    passes the same exponent it used to scale this panel's plotted samples,
    so the title's mantissa and the panel's own axis agree)."""
    scale = 10.0 ** exp
    v, l, h = value / scale, lo / scale, hi / scale
    name = latex.strip('$')
    return fr'${name}=({v:.2f}^{{+{h:.2f}}}_{{-{l:.2f}}})\times10^{{{exp}}}$'


def fit_diagonal_titles(fig, ndim):
    """Left-align (instead of corner.corner()'s default centered) whichever
    of `fig`'s `ndim` diagonal-panel titles is currently too wide for its
    own panel (measured via a forced draw), so it overflows only into the
    always-empty upper-triangle space beside it. Must be re-run after
    every resize (see build_corner_tab's canvas.mpl_connect call) since
    Qt's FigureCanvasQT.resizeEvent() re-fits the figure's inches to the
    Corner tab's real on-screen footprint while font point-sizes stay
    fixed, so whether a title fits can change with the window size. Each
    call re-checks every title from scratch (resetting any previous flip)
    rather than assuming the last call's answer still holds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i in range(ndim):
        ax = fig.axes[i * ndim + i]
        title = ax.title
        if not title.get_text():
            continue
        title.set_horizontalalignment('center')
        title.set_x(0.5)
        if title.get_window_extent(renderer).width > ax.get_window_extent(renderer).width:
            title.set_horizontalalignment('left')
            title.set_x(0.0)


def fit_corner_margins(fig, ndim):
    """Grow `fig`'s outer left/bottom/top/right margins by however far its
    actually-rendered content (tick/axis labels, phi/dphi titles, family
    legend) overflows the figure bounds, rather than trusting
    corner.corner()'s own fixed dim-based formula (sized for its default,
    unscaled fonts). Must be re-run after every resize (see
    build_corner_tab's canvas.mpl_connect call) for the same reason
    fit_diagonal_titles is -- a margin fraction generous enough at
    construction can leave too little absolute room once the embedded
    canvas reaches its real, typically smaller and non-square, displayed
    size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    w_in, h_in = fig.get_size_inches()
    # Denser plots pack more (smaller, tighter-spaced) tick/axis labels
    # along each edge, so the same flat pad reads as comfortable headroom
    # at ndim=4 but noticeably tighter once there's more of that small
    # text lined up along it -- barely different for a 4-parameter model
    # like burn, a bit more breathing room for an 8-parameter one like
    # comp2RMdep.
    pad_in = 0.06 + 0.025 * max(0, ndim - 4)
    sp = fig.subplotpars
    fig.subplots_adjust(
        left=sp.left + max(0.0, pad_in - bbox.x0) / w_in,
        bottom=sp.bottom + max(0.0, pad_in - bbox.y0) / h_in,
        top=sp.top - max(0.0, bbox.y1 - h_in + pad_in) / h_in,
        right=sp.right - max(0.0, bbox.x1 - w_in + pad_in) / w_in)


def fit_corner_label_ink_pad(fig, bottom_axes, left_axes):
    """Nudge each axis-label's own `labelpad` so the *visible ink* of
    every bottom-row x-label sits a consistent distance below its tick
    numbers (likewise left-column y-labels), on top of fig.align_xlabels/
    align_ylabels's own alignment -- those already place every label at
    the exact same nominal-bounding-box anchor, but how much empty box a
    mathtext string reserves above/beside its own ink varies with content
    (e.g. phi/dphi's "$\\times10^{n}$" suffix vs. a plain "$p_1$"), so the
    visible gap can still differ between labels in the same row/column.
    Measures that ink gap once via a throwaway Agg render at
    CORNER_INK_PROBE_DPI and folds the difference into `labelpad` (points,
    dpi-independent -- no re-run needed on resize, unlike
    fit_corner_margins/fit_diagonal_titles)."""
    orig_dpi = fig.dpi
    probe = FigureCanvasAgg(fig)
    try:
        fig.dpi = CORNER_INK_PROBE_DPI
        probe.draw()
        renderer = probe.get_renderer()
        arr = np.asarray(renderer.buffer_rgba())
        H = arr.shape[0]
        pt_per_px = 72.0 / CORNER_INK_PROBE_DPI

        def ink_gap(text, side):
            bbox = text.get_window_extent(renderer)
            x0, x1 = int(np.floor(bbox.x0)), int(np.ceil(bbox.x1))
            y0, y1 = int(np.floor(bbox.y0)), int(np.ceil(bbox.y1))
            r0, r1 = H - y1, H - y0
            crop = arr[max(r0, 0):max(r1, 0), max(x0, 0):max(x1, 0), :3]
            if crop.size == 0:
                return 0.0
            ink = crop.min(axis=2) < 200
            if side == 'top':
                rows = np.where(ink.any(axis=1))[0]
                return float(rows.min()) if rows.size else 0.0
            cols = np.where(ink.any(axis=0))[0]
            return float(crop.shape[1] - 1 - cols.max()) if cols.size else 0.0

        x_gaps = {ax: ink_gap(ax.xaxis.label, 'top') for ax in bottom_axes}
        y_gaps = {ax: ink_gap(ax.yaxis.label, 'right') for ax in left_axes}
        x_target = max(x_gaps.values(), default=0.0)
        y_target = max(y_gaps.values(), default=0.0)
        for ax, gap in x_gaps.items():
            ax.xaxis.labelpad = CORNER_LABELPAD_PT + (x_target - gap) * pt_per_px
        for ax, gap in y_gaps.items():
            ax.yaxis.labelpad = CORNER_LABELPAD_PT + (y_target - gap) * pt_per_px
    finally:
        fig.dpi = orig_dpi


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


class _KwargsWorker(QThread):
    """Shared boilerplate for the Sampling tab's background-thread
    workers: __init__ stores `kwargs` for `_call()` to use, and run() logs
    the exception traceback (helps debug a real crash from the UI alone)
    and emits `failed` with its text if `_call()` raises `catch_types`,
    or `finished_ok` with whatever `_call()` returns otherwise. Subclasses
    implement `_call()`, declare their own `finished_ok`/`failed` signals
    (signatures differ per worker), and override `catch_types` if
    SystemExit needs catching too (see MultiNestWorker)."""
    catch_types = (Exception,)

    def __init__(self, kwargs, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self):
        try:
            result = self._call()
        except self.catch_types as e:
            traceback.print_exc()
            self.failed.emit(str(e))
            return
        if isinstance(result, tuple):
            self.finished_ok.emit(*result)
        else:
            self.finished_ok.emit(result)


class MultiNestWorker(_KwargsWorker):
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
    # SystemExit included: pymultinest itself calls sys.exit(1) (rather
    # than raising ImportError) when its compiled libmultinest.so isn't on
    # LD_LIBRARY_PATH -- see app.py's module docstring. Without catching it
    # here that would escape this thread uncaught and leave the UI stuck
    # mid-fit instead of reporting the failure.
    catch_types = (Exception, SystemExit)

    def _call(self):
        return multinest_fit(progress_callback=self.progress.emit, **self.kwargs)


class LoadSamplesWorker(_KwargsWorker):
    """Runs fitting.load_previous_run() in a background thread so the UI
    stays responsive while it re-reads a completed MultiNest run's output
    files and re-clusters them into mode families (load_samples_action) --
    a few seconds' worth of disk I/O and numpy work with no natural
    progress fraction, hence no `progress` signal (unlike MultiNestWorker):
    the caller just shows an indeterminate busy bar for the duration."""
    finished_ok = pyqtSignal(list, list, dict)   # best_pars, errs, info
    failed = pyqtSignal(str)
    catch_types = (Exception, SystemExit)  # same pymultinest sys.exit(1) risk as MultiNestWorker

    def _call(self):
        return load_previous_run(**self.kwargs)


def unwrap_x_samples(samples, pars, kinds_pol):
    """Copy of `samples` (N, ndim, pol_idx-local order) with every X-kind
    (EVPA) column shifted by its own period (pi) onto the branch centered
    on that same family's own `pars[i]` (its circular median, from
    fitting.circular_median_pctl/best_family) instead of the fixed
    (-pi/2, pi/2] domain every raw MultiNest sample is drawn in.

    Needed before handing samples to corner.corner(): X has no natural
    "zero" -- a family whose true EVPA sits near the domain's own hard
    +-90 deg edge has samples split across that seam (e.g. some near
    -89 deg, some near +89 deg, the *same* physical angle mod pi) even
    though fitting.best_family()'s own circular_median_pctl already
    reports the correct, unbiased point estimate for it. Left as-is,
    corner.corner() -- which has no notion of X's periodicity -- plots
    that single physical cluster as two disconnected blobs (looking
    exactly like a genuine second solution) and computes its own
    diagonal-panel title from plain (non-circular) percentiles of the
    raw column, which for a seam-split column comes out nonsensically
    wide (spanning nearly the whole +-90 deg domain) and inconsistent
    with `pars`/`errs`'s own correct value. Shifting each family's own
    X columns onto *that family's* median first makes both the plotted
    cluster and corner's own auto-computed title agree with the number
    already used everywhere else (sliders, results box, model curve)."""
    samples = samples.copy()
    for i, k in enumerate(kinds_pol):
        if k == 'X':
            samples[:, i] = pars[i] + wrap_pm_halfpi(samples[:, i] - pars[i])
    return samples


def _scale_and_labels(kinds_pol, labels, sel):
    """Per-parameter plot scale factors and axis labels for
    render_corner_figure: fixed factors for 'p'/'X'/'scale', but phi/dphi
    each get their own exponent picked from `sel`'s (the family being
    drawn with show_titles=True) own point estimate, so a phi/dphi
    panel's numbers read at a legible size regardless of that parameter's
    raw ~1e2-1e6 rad/m^2 scale. Returns (scales, plot_labels, phi_exps)."""
    scale_map = {'p': 100.0, 'X': 180.0 / np.pi, 'scale': 1.0}
    phi_exps = {i: sci_exponent(sel['pars'][i]) for i, k in enumerate(kinds_pol) if k in ('phi', 'dphi')}
    scales = np.array([10.0 ** -phi_exps[i] if i in phi_exps else scale_map[k]
                        for i, k in enumerate(kinds_pol)])
    plot_labels = [
        (fr"{lbl.strip('$')}\;(\times10^{{{phi_exps[i]}}})" if i in phi_exps and phi_exps[i] != 0
         else lbl.strip('$'))
        for i, lbl in enumerate(labels)]
    plot_labels = [fr'${lbl}$' for lbl in plot_labels]
    return scales, plot_labels, phi_exps


def _draw_family_layer(fig, families, draw_order, color_by_k, kinds_pol, scales, plot_labels,
                        label_kwargs, max_n_ticks, labelpad):
    """Overlay every non-selected family's own posterior (in `draw_order`,
    weakest-evidence first so a higher-evidence contour is never hidden
    underneath) onto `fig` at alpha=0.5 in its own color -- the selected
    family itself is drawn separately, with show_titles=True, by
    render_corner_figure."""
    import corner
    import matplotlib.colors as mcolors
    for k in draw_order:
        fam_samples = unwrap_x_samples(families[k]['samples'], families[k]['pars'], kinds_pol) * scales[np.newaxis, :]
        color = color_by_k[k]
        faded_color = mcolors.to_rgba(color, alpha=0.5)
        corner.corner(
            fam_samples, fig=fig, labels=plot_labels, plot_datapoints=False,
            color=faded_color, levels=(0.5, 0.8, 0.95),
            hist_kwargs={'color': color, 'alpha': 0.5},
            contour_kwargs={'colors': 'k', 'alpha': 0.5,'linewidths': 0.2},
            fill_contours=True, smooth=1.0, show_titles=False,
            label_kwargs=label_kwargs, max_n_ticks=max_n_ticks, labelpad=labelpad)


def _restore_native_label_positioning(fig, ndim):
    """Undo corner.corner()'s own fixed axes-fraction label placement
    (each of its ax.xaxis/yaxis.set_label_coords() calls flips that
    Axis's private `_autolabelpos` off -- matplotlib-internal, version-
    sensitive API) and restore each label's default transform, so
    matplotlib's own native `_update_label_position` takes back over: it
    measures the real rendered tick-label-plus-spine bbox and places the
    label CORNER_LABELPAD_PT points beyond it, self-correcting for ndim/
    font size/tick digit count/resizing instead of corner's single
    guessed offset. Also aligns each bottom/left group's labels to the
    group's widest tick-label bbox (fig.align_xlabels/align_ylabels) and
    folds each label's own ink-vs-nominal-box gap into its labelpad
    (fit_corner_label_ink_pad) so the visible gap is uniform too."""
    bottom_axes = [fig.axes[(ndim - 1) * ndim + j] for j in range(ndim)]
    left_axes = [fig.axes[i * ndim] for i in range(1, ndim)]
    for ax in bottom_axes:
        ax.xaxis.labelpad = CORNER_LABELPAD_PT
        ax.xaxis._autolabelpos = True
        ax.xaxis.label.set_transform(mtransforms.blended_transform_factory(
            ax.transAxes, mtransforms.IdentityTransform()))
    for ax in left_axes:
        ax.yaxis.labelpad = CORNER_LABELPAD_PT
        ax.yaxis._autolabelpos = True
        ax.yaxis.label.set_transform(mtransforms.blended_transform_factory(
            mtransforms.IdentityTransform(), ax.transAxes))

    fig.align_xlabels(bottom_axes)
    fig.align_ylabels(left_axes)
    fit_corner_label_ink_pad(fig, bottom_axes, left_axes)
    return bottom_axes, left_axes


def _build_family_legend(fig, families, dropped, selected_idx, color_by_k):
    """Legend patch per surviving family (evidence share/ln L/chi2, with a
    'winner'/'shown' suffix as applicable) plus a combined entry for
    everything `dropped` -- only added when there's more than one family
    or something was dropped, matching render_corner_figure's own
    single-family/nothing-dropped case (no legend needed)."""
    if not (len(families) > 1 or dropped['count'] > 0):
        return
    import matplotlib.patches as mpatches
    legend_handles = []
    for k, family in enumerate(families):
        bits = []
        if k == 0:
            bits.append('winner')
        if k == selected_idx and selected_idx != 0:
            bits.append('shown')
        suffix = f" ({', '.join(bits)})" if bits else ''
        legend_handles.append(mpatches.Patch(
            color=color_by_k[k], alpha=(1.0 if k == selected_idx else 0.5),
            label=fr"mass={family['evidence_share']:.1f}%, $\ln L$={family['lnZ']:.2f}, "
                  fr"$\chi_\nu^2$={family['chi2']:.2f}{suffix}"))
    if dropped['count'] > 0:
        legend_handles.append(mpatches.Patch(
            color='none', label=f"+{dropped['count']} more, {dropped['evidence_share']:.1f}% combined"))
    fig.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)


def render_corner_figure(kinds_pol, labels, families, dropped, selected_idx, ndim):
    """Draw `selected_idx`'s family's own pooled posterior (blue, opaque,
    with titled diagonal panels) plus every other surviving mode family
    overlaid in its own color at alpha=0.5, with that family's own
    point-estimate drawn as dashed truths= crosshairs and a legend of each
    family's evidence share/ln Z/chi2 -- mirrors qu_fit.py's corner_plot().
    Returns the finished matplotlib Figure, already margin/title/label-pad
    fitted at its construction size (see fit_corner_margins/
    fit_diagonal_titles) -- everything build_corner_tab needs to embed it
    into a Qt canvas, which is the only step this doesn't do.

    Deliberately Qt-free (a bare Figure via the Agg-only FigureCanvasAgg,
    never FigureCanvasQTAgg) so CornerBuildWorker can run this on a
    background thread: corner.corner()'s own KDE/contour computation
    (repeated once per surviving family) plus the two full re-draws
    fit_corner_margins/fit_diagonal_titles each force to measure real
    rendered geometry easily add up to several seconds for a many-parameter
    model with several mode families, which would otherwise freeze the
    whole UI -- both right after a fresh fit/Load samples and on every
    later family-picker change (see build_corner_tab, its only caller)."""
    import corner

    sel = families[selected_idx]
    scales, plot_labels, phi_exps = _scale_and_labels(kinds_pol, labels, sel)

    sel_samples = unwrap_x_samples(sel['samples'], sel['pars'], kinds_pol) * scales[np.newaxis, :]
    truths = [p * s for p, s in zip(sel['pars'], scales)]

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
    # corner.corner() itself only ever gets 0 here: build_corner_tab
    # replaces its axis-label placement with matplotlib's own native,
    # tick-aware auto-positioning right after the last corner.corner()
    # call below (see CORNER_LABELPAD_PT), so whatever this adds on
    # top of corner's own fixed -0.3 axes-fraction offset would just
    # be thrown away regardless of its value.
    labelpad = 0.0

    # Color assignment: the selected family is always blue/opaque; every
    # other family gets a FAMILY_PALETTE color assigned in evidence-
    # share-descending order (so, as before selected_idx became
    # pickable, the highest-evidence non-selected family is always
    # 'tab:orange', etc. -- this just no longer assumes the winner
    # (index 0) is always the selected one).
    non_selected_desc = sorted((k for k in range(len(families)) if k != selected_idx),
                                key=lambda k: -families[k]['evidence_share'])
    color_by_k = {selected_idx: 'b'}
    for rank, k in enumerate(non_selected_desc):
        color_by_k[k] = FAMILY_PALETTE[rank % len(FAMILY_PALETTE)]

    # Weakest-evidence family drawn first, selected family last, so an
    # overlapping higher-evidence contour is never hidden underneath a
    # lower-evidence one.
    draw_order = sorted(non_selected_desc, key=lambda k: families[k]['evidence_share'])

    _draw_family_layer(fig, families, draw_order, color_by_k, kinds_pol, scales, plot_labels,
                        label_kwargs, max_n_ticks, labelpad)

    corner.corner(
        sel_samples, fig=fig, show_titles=True, labels=plot_labels,
        plot_datapoints=False, color='b', levels=(0.5, 0.8, 0.95),
        hist_kwargs={'color': 'b'}, contour_kwargs={'colors': 'k','linewidths': 0.2},
        fill_contours=True, smooth=1.0, title_fmt='.2f',
        label_kwargs=label_kwargs, title_kwargs=title_kwargs,
        max_n_ticks=max_n_ticks, labelpad=labelpad)

    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    _restore_native_label_positioning(fig, ndim)

    # corner.corner() re-applies its own fixed wspace=hspace=0.05
    # gutter (a fraction of each panel's own width/height) on every
    # call, so this has to run after the last one above rather than
    # once up front. Zeroing it here -- before the title-overflow and
    # margin measurements below, both of which need the real final
    # panel geometry to measure against -- packs the KxK grid edge to
    # edge; safe since corner.corner() already hides every tick label
    # except the bottom row's and left column's (see corner.core), so
    # there's no interior label left to collide with a touching
    # neighbor.
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # phi/dphi's raw physical scale (~1e2-1e6 rad/m^2) doesn't suit a
    # single title-wide formula -- their titles get their own
    # per-parameter exponent (the same one `scales`/`plot_labels` above
    # already picked for that whole column/row, so title and axis
    # agree), overwriting whatever corner.corner()'s generic title_fmt
    # wrote for just those two diagonal panels (fig.axes' flattened
    # K*K layout puts panel (i,i) at index i*ndim+i -- see
    # corner.core._get_fig_axes, which reads it back the same way).
    # `sel` (not always the winner, since selected_idx may not be 0)
    # is the family show_titles=True was rendered for just above, so
    # its own pars/errs are what these titles must match.
    for local_i, kind in enumerate(kinds_pol):
        if kind in ('phi', 'dphi'):
            lo, hi = sel['errs'][0, local_i], sel['errs'][1, local_i]
            title = format_sci_title(labels[local_i], sel['pars'][local_i], lo, hi, phi_exps[local_i])
            fig.axes[local_i * ndim + local_i].set_title(title, **title_kwargs)

    corner.overplot_lines(fig, truths, color='k', linestyle='--', alpha=0.4)
    corner.overplot_points(fig, [truths], marker='s', color='k', alpha=0.4)

    _build_family_legend(fig, families, dropped, selected_idx, color_by_k)

    # Approximate first pass at the figure's current (construction-
    # time) size -- see fit_corner_margins's/fit_diagonal_titles's own
    # docstrings for why this alone isn't reliable once actually
    # embedded, and build_corner_tab's own canvas.mpl_connect for the
    # real fix, applied once this figure is actually embedded in a Qt
    # canvas back on the main thread. A bare Figure (as built above) has
    # no canvas of its own yet -- both functions need one to draw with,
    # so give it a throwaway Agg one; the real Qt canvas built later
    # replaces it as fig.canvas regardless. Margins first, then titles,
    # since the former decides each panel's actual width, which the
    # latter checks titles against.
    FigureCanvasAgg(fig)
    fit_corner_margins(fig, ndim)
    fit_diagonal_titles(fig, ndim)

    return fig


class CornerBuildWorker(_KwargsWorker):
    """Runs render_corner_figure() in a background thread so the UI stays
    responsive while it draws -- unlike MultiNestWorker/LoadSamplesWorker
    (which mainly hide a blocking library/disk-I/O call), the work here is
    CPU-bound Python/numpy/matplotlib the whole way through: corner.corner()'s
    own KDE/contour computation (once per surviving mode family) plus the
    two full re-draws render_corner_figure forces to measure real rendered
    geometry (see fit_corner_margins/fit_diagonal_titles) easily add up to
    several seconds for a many-parameter model with several families --
    long enough to freeze the whole UI if run on the main thread, which
    used to happen both right after a fresh fit/Load samples and on every
    later family-picker change (build_corner_tab is this worker's only
    caller, for both cases)."""
    finished_ok = pyqtSignal(object)   # fig
    failed = pyqtSignal(str)

    def _call(self):
        return render_corner_figure(**self.kwargs)


# The corner figure's on-screen dpi is left at matplotlib's own default --
# raising it would inflate the (point-sized) tick/label/title text relative
# to the panels themselves, undoing the ndim-scaled font sizing in
# build_corner_tab. A PNG saved at that same screen dpi is fine at the size
# it's embedded at but turns blocky once viewed at full size/zoomed into
# later (more so for denser, higher-ndim plots, which cram more panels into
# the same on-screen pixel footprint -- see build_corner_tab's `size`
# comment), so CornerToolbar's Save button renders at this higher dpi
# instead, independent of whatever dpi the embedded canvas itself is using.
CORNER_SAVE_DPI = 600


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

    def _set_sampling_busy(self, busy, message=''):
        """Toggle the Sampling tab's busy state -- fit/load/clear buttons
        and the progress bar/label -- shared by every MultiNest fit/Load
        samples/corner-build entry and exit point. `message` sets the
        progress label's text while busy (ignored when busy=False)."""
        self.fit_button.setEnabled(not busy)
        self.load_data_button.setEnabled(not busy)
        self.clear_data_button.setEnabled(not busy)
        self.sampling_load_button.setEnabled(not busy)
        self.sampling_progress.setVisible(busy)
        self.sampling_progress_label.setVisible(busy)
        if busy:
            self.sampling_progress_label.setText(message)

    def build_sampling_tab(self):
        """Build the "Sampling" tab (MultiNest fit options: output dir, #
        live points, sampling efficiency, evidence tolerance, prior-bounds
        rows, Load samples) and add it to self.left_tabs. Always called
        from MainWindow.__init__, regardless of whether pymultinest is
        actually installed -- see module docstring."""
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
        sampling_layout.addWidget(QLabel('<b>Parameter priors</b>'))
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
            'Load data first (Load data button) to enable Fit!.')
        # The tab itself is always accessible (e.g. Load samples doesn't
        # need any data currently loaded) -- only Fit! is gated on loaded
        # data, via the shared self.fit_button (see MainWindow.load_data_action
        # / clear_data), same as the Visualization tab's own fit.

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
            return  # Sampling tab not built yet

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

        self._set_sampling_busy(True, 'Starting MultiNest...')

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
        self._set_sampling_busy(False)
        for sl, val in zip(self.sliders, best_pars):
            sl.set_value(val)
        self.results_label.setText(self.format_mn_stats(info))
        self.apply_posterior_result(self.mn_model, best_pars, errs, info)
        self.mn_worker = None

    def on_mn_failed(self, msg):
        self._set_sampling_busy(False)
        QMessageBox.warning(self, 'Sampling', f'MultiNest fit failed:\n{msg}')
        self.mn_worker = None

    def _format_fit_stats_block(self, chi2, dof, chi2_red, loglike, aic, aicc, bic,
                                  evidence_lnZ, global_evidence, global_evidence_err):
        """Shared chi2/dof/AIC/AICc/BIC/evidence-lnZ lines used by both
        format_mn_stats (the winning family) and format_family_stats (any
        other family), so the Results box reads identically regardless of
        which one is shown -- same field set/formulas as
        fitting.fit_statistics()'s own least-squares report."""
        aicc_str = f'{aicc:.4g}' if np.isfinite(aicc) else 'n/a (dof ≤ 1)'
        return [
            f"χ² = {chi2:.4g}   dof = {dof}",
            f"χ²_red = {chi2_red:.4g}",
            f"ln L = {loglike:.4g}",
            f"AIC = {aic:.4g}   AICc = {aicc_str}",
            f"BIC = {bic:.4g}",
            f"evidence ln Z (this family) = {evidence_lnZ:.4g}",
            f"global ln Z (whole model) = {global_evidence:.4g} ± {global_evidence_err:.3g}",
        ]

    def format_mn_stats(self, info):
        """Results-box text for the winning family: a winning-family-share
        header (plus a combined other-families/dropped-modes line, if any)
        followed by the shared stats block (see _format_fit_stats_block)."""
        others = info['other_families']
        dropped = info['dropped']

        family_bits = []
        if others:
            family_bits.append(f"{len(others)} other family(ies)")
        if dropped['count']:
            family_bits.append(f"{dropped['count']} negligible mode(s) dropped "
                                f"({dropped['evidence_share']:.2g}% evidence)")

        lines = [f"winning family: {info['evidence_share']:.2f}% evidence share"]
        if family_bits:
            lines.append('; '.join(family_bits))
        lines += self._format_fit_stats_block(
            info['chi2'], info['dof'], info['chi2_red'], info['loglike'],
            info['aic'], info['aicc'], info['bic'], info['winner_evidence_lnZ'],
            info['global_evidence'], info['global_evidence_err'])
        return '\n'.join(lines)

    # ── MultiNest result plumbing (posterior overlay + Corner plot tab) ──────
    # Shared by a fresh fit (on_mn_finished) and Load samples
    # (load_samples_action), which both end up with the same `info` shape
    # (see fitting.multinest_fit / fitting.load_previous_run).
    def apply_posterior_result(self, model, best_pars, errs, info):
        """Stash a random subset of the winning family's posterior (for the
        faint spaghetti overlay on the Visualization tab's plots -- see
        app.ModelPlot/StokesPlot.set_posterior_samples) and (re)build the
        Corner plot tab from the full result. The Corner tab's own family
        dropdown (see build_corner_tab) defaults to this same winning
        family (selected_idx=0) -- our winner-selection criteria don't
        change, only which family is *displayed* can, via that dropdown."""
        spec = MODELS[model]
        pol_idx = info['pol_idx']
        spectral_pars = {i: best_pars[i] for i in spec.indices('alpha') + spec.indices('eps')}
        self.stash_posterior_samples(model, pol_idx, info['winner_samples'], spectral_pars)
        self.build_corner_tab(model, best_pars, errs, info)
        self.update_plot()

    def stash_posterior_samples(self, model, pol_idx, pol_samples, spectral_pars):
        """(Re)build the faint spaghetti-overlay posterior draws (see
        app.ModelPlot/StokesPlot.set_posterior_samples) from `pol_samples`
        (pol_idx-local order, e.g. info['winner_samples'] or a Corner-tab
        `other_families` entry's own 'samples') -- shared by
        apply_posterior_result (the winning family) and
        on_corner_family_selected (whichever family the Corner tab's
        dropdown currently has picked)."""
        spec = MODELS[model]
        rng = np.random.default_rng(0)
        n_draws = min(N_POSTERIOR_DRAWS, len(pol_samples))
        subset = pol_samples[rng.choice(len(pol_samples), size=n_draws, replace=False)]

        full_samples = np.zeros((n_draws, len(spec.params)))
        full_samples[:, pol_idx] = subset
        for i, v in spectral_pars.items():
            full_samples[:, i] = v

        self.posterior_samples = full_samples
        self.posterior_model = model
        self.canvas.set_posterior_samples(full_samples, model)
        self.stokes_canvas.set_posterior_samples(full_samples, model)

    def clear_posterior_samples(self):
        self.posterior_samples = None
        self.posterior_model = None
        self.canvas.set_posterior_samples(None, None)
        self.stokes_canvas.set_posterior_samples(None, None)

    def build_corner_tab(self, model, best_pars, errs, info, selected_idx=0):
        """(Re)build the 'Corner plot' tab: `selected_idx`'s family's own
        pooled posterior (blue, opaque, with titled diagonal panels) plus
        every other surviving mode family overlaid in its own color at
        alpha=0.5, with that family's own point-estimate drawn as dashed
        truths= crosshairs and a legend of each family's evidence share/
        ln Z/chi2 -- mirrors qu_fit.py's corner_plot(). `selected_idx` is
        an index into `families` (see below): 0 is always our actual
        winning family (the one apply_posterior_result's own selection
        criteria picked, per SamplingMixin's module docstring -- that
        choice never changes), 1.. are info['other_families'] in their own
        already-evidence-descending order. A top-right dropdown
        (corner_family_combo, populated in _on_corner_figure_ready) lets
        the user switch `selected_idx` (see on_corner_family_selected) to
        inspect any other non-discarded family without disturbing which
        one is actually the winner. Removed by Reset model or Clear data
        (see remove_corner_tab).

        `selected_idx`'s own Figure is cached (_corner_fig_cache, keyed by
        selected_idx, reset whenever `info` itself changes -- a genuinely
        new result, not just a family switch) -- so flipping the family
        picker back to one already seen for this same result just
        re-embeds that Figure instead of rebuilding it from scratch.

        Otherwise, the actual drawing (render_corner_figure -- every
        corner.corner() call plus the margin/title/label-pad refit passes)
        runs in a background CornerBuildWorker rather than here: for a
        many-parameter model with several surviving families that's
        several seconds of pure-Python/numpy/matplotlib work, long enough
        to freeze the whole UI if run inline -- this returns immediately
        once that worker is started, and _on_corner_figure_ready finishes
        the (fast, genuinely Qt-bound) job of embedding its finished
        Figure once it's done. The previous corner tab, if any, is left in
        place and interactive until then, rather than torn down up front."""
        try:
            import corner  # noqa: F401 -- availability probe; render_corner_figure does the real import
        except ImportError:
            QMessageBox.warning(self, 'Corner plot',
                                 "The 'corner' package isn't installed -- skipping the corner plot.")
            return

        spec = MODELS[model]
        pol_idx = info['pol_idx']
        ndim = len(pol_idx)
        kinds_pol = [spec.params[i].kind for i in pol_idx]
        labels = [spec.params[i].latex for i in pol_idx]

        # families[0] is always the actual winner -- reconstructed here
        # (rather than stored as its own 'family' dict by best_family()
        # itself) purely because best_pars/errs/info is the shape every
        # other caller (sliders, Results box, ...) already works with; see
        # the docstring above for why index 0 is special. families[1:] are
        # info['other_families'] verbatim -- both already carry the same
        # 'pars'/'errs' (pol_idx-local, (2,ndim) lo/hi errs)/'samples'/
        # 'evidence_share'/'chi2'/'lnZ'/'evidence_lnZ' shape (see
        # fitting.best_family).
        winner_errs_pol = np.array([[errs[i][0] for i in pol_idx],
                                     [errs[i][1] for i in pol_idx]])
        winner_family = dict(
            pars=[best_pars[i] for i in pol_idx], errs=winner_errs_pol,
            samples=info['winner_samples'], evidence_share=info['evidence_share'],
            chi2=info['winner_chi2'], lnZ=info['winner_lnZ'],
            evidence_lnZ=info['winner_evidence_lnZ'])
        families = [winner_family] + info['other_families']
        dropped = info['dropped']
        selected_idx = max(0, min(selected_idx, len(families) - 1))

        if info is not self._corner_info:
            # A genuinely new result (fresh fit/Load samples), not just
            # on_corner_family_selected re-requesting a different
            # selected_idx of the *same* one (it passes this same `info`
            # object straight through) -- _corner_fig_cache's Figures are
            # for the old result and no longer apply.
            self._corner_fig_cache = {}

        # Cached so on_corner_family_selected (the dropdown's own handler)
        # can rebuild this tab and the Visualization-tab model curves for
        # a newly-picked family without re-clustering MultiNest's raw
        # output -- everything it needs is already sitting in `families`.
        self._corner_model = model
        self._corner_best_pars = best_pars
        self._corner_errs = errs
        self._corner_info = info
        self._corner_families = families
        self._corner_selected_idx = selected_idx
        self._corner_ndim = ndim

        if self.corner_build_worker is not None:
            # A previous build (a fresh fit/Load samples, or an earlier
            # family pick) is still rendering when this one was requested
            # -- self._corner_* above has already moved past whatever it
            # was building, so let it run to completion (Agg rendering
            # isn't cheaply interruptible) but drop its result on arrival
            # instead of embedding a stale figure over this newer request.
            # Cleared (not just disconnected) so a *second* still-cache-hit
            # request right behind this one doesn't try to disconnect an
            # already-disconnected worker below (PyQt raises on that).
            self.corner_build_worker.finished_ok.disconnect(self._on_corner_figure_ready)
            self.corner_build_worker.failed.disconnect(self._on_corner_figure_failed)
            self.corner_build_worker = None

        # selected_idx already rendered for this same result (a family the
        # user switched to before, e.g. switching back to the winner) --
        # reuse that Figure instead of re-running CornerBuildWorker (whose
        # corner.corner() calls plus the margin/title refit passes take
        # several seconds for a many-parameter model) from scratch.
        cached_fig = self._corner_fig_cache.get(selected_idx)
        if cached_fig is not None:
            self._on_corner_figure_ready(cached_fig)
            return

        self._set_sampling_busy(True, 'Building corner plot...')
        if getattr(self, 'corner_family_combo', None) is not None:
            self.corner_family_combo.setEnabled(False)

        self.corner_build_worker = CornerBuildWorker(dict(
            kinds_pol=kinds_pol, labels=labels, families=families,
            dropped=dropped, selected_idx=selected_idx, ndim=ndim))
        self.corner_build_worker.finished_ok.connect(self._on_corner_figure_ready)
        self.corner_build_worker.failed.connect(self._on_corner_figure_failed)
        self.corner_build_worker.start()

    def _on_corner_build_done(self):
        """Shared tail of _on_corner_figure_ready/_on_corner_figure_failed:
        undoes build_corner_tab's own busy state (progress bar, disabled
        buttons) regardless of whether the render succeeded."""
        self._set_sampling_busy(False)
        if getattr(self, 'corner_family_combo', None) is not None:
            self.corner_family_combo.setEnabled(True)
        self.corner_build_worker = None

    def _on_corner_figure_ready(self, fig):
        """CornerBuildWorker.finished_ok handler: embed the already-drawn
        `fig` (built entirely off the main thread -- see
        render_corner_figure) into a fresh Qt canvas/toolbar/family-picker
        and swap it in as the Corner plot tab. This is the only genuinely
        Qt-bound step of a corner-tab rebuild, and a fast one, unlike the
        rendering itself -- so it's fine to run it here on the main thread."""
        self._on_corner_build_done()

        # Captured before remove_corner_tab(), which -- as the "fully
        # clear any Corner-tab result" path Reset model/Clear data also
        # use it for -- blanks self._corner_info/_corner_families as one
        # of its own side effects; restored below since here that result
        # is still current, just moving into a freshly-built tab/combo.
        ndim = self._corner_ndim
        families = self._corner_families
        info = self._corner_info
        selected_idx = self._corner_selected_idx

        if families is None:
            # Reset model or Clear data ran (see their own remove_corner_tab()
            # calls) while this fig was still rendering in the background --
            # neither disables Reset model the way build_corner_tab disables
            # Fit!/Load data/Clear data/the family combo for its own
            # duration, so that race is reachable. Whatever this fig shows
            # is for a result the user has already discarded -- drop it
            # rather than resurrecting a Corner tab (and stale sliders, via
            # a now-impossible on_corner_family_selected replay) they just
            # asked to clear.
            return

        self.remove_corner_tab()
        self._corner_info = info
        self._corner_families = families
        self._corner_fig_cache[selected_idx] = fig  # see build_corner_tab's own cache-hit check

        canvas = FigureCanvas(fig)

        # Re-run both the margin fit and the diagonal-title fit (in that
        # order -- the former decides each panel's actual on-screen width,
        # which the latter checks titles against) once Qt actually gives
        # this canvas its real on-screen size; the passes render_corner_figure
        # already did, at the figure's construction-time size, are only
        # ever a first guess (see both functions' own docstrings).
        def _refit_corner(event):
            fit_corner_margins(fig, ndim)
            fit_diagonal_titles(fig, ndim)
        canvas.mpl_connect('resize_event', _refit_corner)

        toolbar = CornerToolbar(canvas, self)

        # Top-right family picker: switches which (non-discarded) family
        # is drawn opaque/blue above -- and, via on_corner_family_selected,
        # the model curves on the p/EVPA and Stokes I,Q,U tabs -- without
        # touching which family our own selection criteria consider the
        # winner (families[0], always labeled 'Winner' here regardless of
        # what's currently picked). Built fresh (like the rest of this
        # tab) on every call; setCurrentIndex() is set *before* connecting
        # currentIndexChanged so repopulating it here never re-triggers
        # on_corner_family_selected itself.
        self.corner_family_combo = QComboBox()
        self.corner_family_combo.setToolTip(
            "Pick which family to inspect")
        for k, family in enumerate(families):
            text = (f"Winner ({family['evidence_share']:.1f}% mass)" if k == 0
                    else f"Family {k + 1} ({family['evidence_share']:.1f}% mass)")
            self.corner_family_combo.addItem(text)
        self.corner_family_combo.setCurrentIndex(selected_idx)
        self.corner_family_combo.currentIndexChanged.connect(self.on_corner_family_selected)

        top_row = QHBoxLayout()
        top_row.addWidget(toolbar, stretch=1)
        top_row.addWidget(QLabel('Solution:'))
        top_row.addWidget(self.corner_family_combo)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top_row)
        layout.addWidget(canvas, stretch=1)

        self.corner_tab = tab
        self.plot_tabs.addTab(tab, 'Corner plot')
        self.plot_tabs.setCurrentWidget(tab)

    def _on_corner_figure_failed(self, msg):
        self._on_corner_build_done()
        QMessageBox.warning(self, 'Corner plot', f'Failed to build corner plot:\n{msg}')

    def on_corner_family_selected(self, k):
        """corner_family_combo's currentIndexChanged handler: move the
        Visualization-tab sliders (and so the p/EVPA and Stokes I,Q,U
        model curves) and the posterior spaghetti-overlay draws to family
        `k`'s own point estimate/samples, update the Results box to match,
        and rebuild the Corner tab with `k` as the new selected_idx --
        same cached model/best_pars/errs/info build_corner_tab last saw,
        no re-clustering needed. A no-op if the Corner tab isn't currently
        showing a result (shouldn't happen -- this is only wired to a
        combo box that lives inside the Corner tab itself)."""
        info = getattr(self, '_corner_info', None)
        if info is None:
            return
        model, best_pars, errs = self._corner_model, self._corner_best_pars, self._corner_errs
        families = self._corner_families
        k = max(0, min(k, len(families) - 1))
        fam = families[k]

        spec = MODELS[model]
        pol_idx = info['pol_idx']
        spectral_pars = {i: best_pars[i] for i in spec.indices('alpha') + spec.indices('eps')}
        full_pars, _ = expand_pars_errs(len(spec.params), pol_idx, fam['pars'], fam['errs'], spectral_pars)
        for sl, val in zip(self.sliders, full_pars):
            sl.set_value(val)
        self.stash_posterior_samples(model, pol_idx, fam['samples'], spectral_pars)

        self.results_label.setText(
            self.format_mn_stats(info) if k == 0 else self.format_family_stats(k, fam, info))

        self.build_corner_tab(model, best_pars, errs, info, selected_idx=k)
        self.update_plot()

    def format_family_stats(self, k, family, info):
        """Results-box text for a non-winning family picked from the
        Corner tab's dropdown: same shared stats block as format_mn_stats
        (see _format_fit_stats_block), recomputed from `family`'s own
        chi2/lnZ (dof/n_free/n_data are shared across every family of the
        same run, so those come from `info`), behind a header clearly
        marking it as not the winner."""
        dof, n_free, n_data = info['dof'], info['n_free'], info['n_data']
        chi2 = family['chi2'] * dof
        aic = 2 * n_free - 2 * family['lnZ']
        aicc = aic + (2 * n_free * (n_free + 1)) / (dof - 1) if dof > 1 else float('nan')
        bic = n_free * np.log(n_data) - 2 * family['lnZ']
        lines = [f"family #{k + 1} (shown; not the winning family): "
                 f"{family['evidence_share']:.2f}% evidence share"]
        lines += self._format_fit_stats_block(
            chi2, dof, family['chi2'], family['lnZ'], aic, aicc, bic,
            family['evidence_lnZ'], info['global_evidence'], info['global_evidence_err'])
        return '\n'.join(lines)

    def remove_corner_tab(self):
        # _corner_info/_corner_families are cleared unconditionally --
        # not only when there's a tab widget to tear down -- since
        # Reset model/Clear data can also land here (see their own
        # callers) while a build_corner_tab() is still rendering in the
        # background with no widget embedded yet (self.corner_tab is
        # still None from before that build started); _on_corner_figure_ready
        # relies on _corner_families going to None here to notice its own
        # result was discarded out from under it and drop the now-stale
        # figure instead of resurrecting a tab for it.
        self._corner_info = None
        self._corner_families = None
        tab = self.corner_tab
        if tab is None:
            return
        idx = self.plot_tabs.indexOf(tab)
        if idx != -1:
            self.plot_tabs.removeTab(idx)
        tab.deleteLater()
        self.corner_tab = None
        # corner_family_combo is a child of `tab` (see _on_corner_figure_ready)
        # and so is destroyed along with it above -- drop this reference too,
        # or the hasattr/getattr guards elsewhere would still find a Python
        # attribute pointing at an already-deleted C/C++ QComboBox.
        self.corner_family_combo = None

    def load_samples_action(self):
        """Load a previous MultiNest run's output (picked via a directory
        dialog), re-run the same mode-family clustering fitting.load_previous_run
        does after a fresh fit, and apply the result exactly like a fresh
        MultiNest fit would (sliders, posterior overlay, Corner plot tab --
        see apply_posterior_result).

        The chosen directory ideally contains the 'mn_polvista_meta.npz'
        sidecar run_multinest_fit saves alongside its own MultiNest output
        (model name, the alpha/epsilon values held fixed during that run,
        and the exact q/u data it was fit against) -- this makes Load
        samples fully self-contained, independent of whatever data (if
        any) happens to be currently loaded via Load data. Failing that,
        falls back to _load_qu_fit_samples: a plain ~/Downloads/pipe/
        qu_fit.py MultiNest run (no polvista sidecar at all) recognized by
        its own directory name -- see QU_FIT_MODEL_MAP."""
        start_dir = self.sampling_outdir_edit.text() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(self, 'Load MultiNest samples', start_dir)
        if not directory:
            return

        meta_path = os.path.join(directory, 'mn_polvista_meta.npz')
        sampled_idx = degenerate_pair = None
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path)
                model_name = str(meta['model'])
                if model_name not in MODELS_BY_NAME:
                    raise ValueError(f"unknown model '{model_name}' in metadata")
                model = MODELS_BY_NAME[model_name]
                spectral_pars = {int(i): float(v) for i, v in zip(meta['spectral_idx'], meta['spectral_val'])}
                wl, q, q_err, u, u_err = meta['wl'], meta['q'], meta['q_err'], meta['u'], meta['u_err']
            except (Exception, SystemExit) as e:
                # SystemExit included: see MultiNestWorker.run -- this runs on
                # the main thread, so letting it escape uncaught would exit the
                # whole app instead of just reporting the failure here.
                QMessageBox.warning(self, 'Load samples', f'Failed to load samples:\n{e}')
                return
        else:
            result = self._load_qu_fit_samples(directory)
            if result is None:
                return
            model, spectral_pars, wl, q, q_err, u, u_err, sampled_idx, degenerate_pair = result

        basename = os.path.join(directory, 'mn_')

        self._set_sampling_busy(True, 'Loading samples...')

        self.mn_model = model
        self.ls_worker = LoadSamplesWorker(dict(
            wl=wl, q=q, q_err=q_err, u=u, u_err=u_err, model=model,
            spectral_pars=spectral_pars, outputfiles_basename=basename,
            sampled_idx=sampled_idx, degenerate_pair=degenerate_pair,
        ))
        self.ls_worker.finished_ok.connect(self.on_load_samples_finished)
        self.ls_worker.failed.connect(self.on_load_samples_failed)
        self.ls_worker.start()

    def _load_qu_fit_samples(self, directory):
        """Fall back for a directory with no mn_polvista_meta.npz sidecar --
        a plain qu_fit.py MultiNest run instead of one from this app's own
        Sampling tab. Recognized by directory name (QU_FIT_MODEL_MAP) and
        reconstructed into the same (model, spectral_pars, wl, q, q_err, u,
        u_err) shape the sidecar path provides, plus (sampled_idx,
        degenerate_pair) for fitting.load_previous_run's own overrides.
        wl/q/q_err/u/u_err come from the parent folder's plot_inputs.npz
        (qu_fit.py's own record of the fit data, reproducing its error
        propagation exactly -- re-deriving q/u from a CSV wouldn't); a CSV
        is still needed for the Stokes I(nu) values plot_inputs.npz lacks
        (for this run's alpha estimate), tried first at the conventional
        assets/data/<source>_<year>_VAPOLA.csv path (parsed from
        `directory`'s two parent folder names), falling back to a
        file-picker. Returns None (after warning the user) on any
        recognized failure; otherwise (model, spectral_pars, wl, q, q_err,
        u, u_err, sampled_idx, degenerate_pair)."""
        name = os.path.basename(os.path.normpath(directory))
        entry = QU_FIT_MODEL_MAP.get(name)
        if entry is None or not os.path.exists(os.path.join(directory, 'mn_.txt')):
            QMessageBox.warning(
                self, 'Load samples',
                f'No polvista metadata found in:\n{directory}\n\n'
                "(expected 'mn_polvista_meta.npz', written by this app's own Sampling-tab "
                "Fit! runs -- pick the folder one was saved to. This folder also isn't a "
                "recognized qu_fit.py model output directory.)")
            return None
        model_name, fixed_zero_idx, degenerate_pair = entry
        model = MODELS_BY_NAME[model_name]
        spec = MODELS[model]

        parent = os.path.dirname(os.path.normpath(directory))
        plot_inputs_path = os.path.join(parent, 'plot_inputs.npz')
        if not os.path.exists(plot_inputs_path):
            QMessageBox.warning(
                self, 'Load samples',
                f"No plot_inputs.npz found in:\n{parent}\n\n"
                "(qu_fit.py's own record of the q/u data this directory's model was "
                'fit against -- needed to re-read it.)')
            return None
        plot_inputs = np.load(plot_inputs_path)
        wl, q, q_err, u, u_err = (plot_inputs['wl'], plot_inputs['qs'], plot_inputs['qers'],
                                    plot_inputs['us'], plot_inputs['uers'])

        year = os.path.basename(parent)
        source = os.path.basename(os.path.dirname(parent))
        default_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'assets', 'data',
            f'{source}_{year}_VAPOLA.csv')
        csv_path = default_csv if os.path.isfile(default_csv) else None
        if csv_path is None:
            csv_path, _ = QFileDialog.getOpenFileName(
                self, f'Stokes I/Q/U data for {source} {year}', parent, 'CSV files (*.csv)')
            if not csv_path:
                return None

        # Deferred: app.py imports this module (SamplingMixin) at module
        # load time, so importing app.py back at *this* module's own top
        # level would be circular -- safe here since by call time (a user
        # actually clicking Load samples) app.py is already fully loaded.
        from polvista.app import load_vapola_csv, FIT_FIXED_EPSILON
        try:
            freq, I, _Q, _U, _I_err, _Q_err, _U_err = load_vapola_csv(csv_path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, 'Load samples', f'Failed to load data CSV:\n{csv_path}\n\n{e}')
            return None

        # Standard spectral parameters, same as a fresh Sampling-tab fit's
        # own pre-fit (app.fit_spectrum_lsq) would default to: powerlaw
        # alpha estimated from this source's own I(nu) data (shared by both
        # components for a two-component model -- see estimate_alpha's own
        # docstring for why only one shared value is ever recoverable from
        # QU-only fitting), epsilon fixed at FIT_FIXED_EPSILON. Both inert
        # for a *_legacy model's own P(lambda) (legacy_sum_combine ignores
        # them) -- only stokes_I()'s own I(nu) display shape still uses them.
        alpha_val = estimate_alpha(freq, I)
        spectral_pars = {i: alpha_val for i in spec.indices('alpha')}
        eps_indices = spec.indices('eps')
        if eps_indices:
            spectral_pars[eps_indices[0]] = FIT_FIXED_EPSILON
        for i in fixed_zero_idx:
            spectral_pars[i] = 0.0
        sampled_idx = [i for i in pol_idx(spec) if i not in fixed_zero_idx]

        return model, spectral_pars, wl, q, q_err, u, u_err, sampled_idx, degenerate_pair

    def on_load_samples_finished(self, best_pars, errs, info):
        self._set_sampling_busy(False)

        model = self.mn_model
        index = self.model_combo.findData(model)
        if index == -1:
            # A hidden model (ModelSpec.hidden -- currently only the
            # *_legacy qu_fit.py-import variants, see QU_FIT_MODEL_MAP)
            # isn't in the dropdown yet; insert it just above the "Custom
            # model..." sentinel row, same as a freshly-built custom model
            # (see app.MainWindow.open_custom_model_dialog), so
            # setCurrentIndex below can select it and rebuild_sliders picks
            # it up like any other model from then on.
            index = self.model_combo.count() - 1
            self.model_combo.insertItem(index, MODELS[model].label, model)
        self.model_combo.setCurrentIndex(index)  # triggers rebuild_sliders if it changed
        for sl, val in zip(self.sliders, best_pars):
            sl.set_value(val)
        self.results_label.setText(self.format_mn_stats(info))
        self.apply_posterior_result(model, best_pars, errs, info)
        self.ls_worker = None

    def on_load_samples_failed(self, msg):
        self._set_sampling_busy(False)
        QMessageBox.warning(self, 'Load samples', f'Failed to load samples:\n{msg}')
        self.ls_worker = None
