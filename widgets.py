"""Qt/matplotlib widgets shared across polvista's tabs -- ParamSlider's own
small building blocks (ValueLineEdit and friends), plus the two plot
canvases (ModelPlot, StokesPlot) app.py's Visualization tab embeds. Split
out on its own so no other module has to import app.py (a large,
MainWindow-centric module) just to get these."""
import re
import time
import functools
import colorsys

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.transforms as mtransforms
from matplotlib.ticker import NullFormatter, NullLocator

from matplotlib import mathtext as _mathtext

# matplotlib parses every mathtext string (all the log-axis tick labels, axis
# labels and legend entries) through an lru_cache of only 50 entries. These
# canvases have more distinct strings than that, so each full draw re-parsed
# most of them -- about half of a redraw. Keep a cache big enough for all.
MATHTEXT_CACHE_SIZE = 2048
_mathtext.MathTextParser._parse_cached = functools.lru_cache(MATHTEXT_CACHE_SIZE)(
    _mathtext.MathTextParser._parse_cached.__wrapped__)

from polvista.models import C, stokes_I, stokes_QU, stokes_components, evpa, pol, spectral_state
from polvista.measurements import RAINBOW_HUE_MAX
from polvista.rm_synthesis import sci_latex

# Default figure preferences (Preferences menu): the values polvista starts
# with and that Restore falls back to. Font sizes in points; grid_which is
# 'both' (major + faint minor), 'major' or 'none'; grid_style is a
# CURVE_LINESTYLES key for curve_style, a matplotlib linestyle name for
# grid_style; curve_scheme is 'default' (each curve its own color) or 'custom'
# (all of the visualizer's model curves in curve_color); curve_samples is the
# number of posterior draws a Comparison-tab samples curve starts with;
# ylim_err_factor_p/_evpa are how many max-errors beyond the data the p/EVPA
# y-limits extend when points are loaded; EVPA points spanning more than
# evpa_full_span degrees get the fixed y-limits evpa_full_ylim instead; bands_on shades the
# Measurements tab's bands on the plots at opacity bands_alpha; marker is a matplotlib marker symbol.
DEFAULT_PLOT_PREFS = {
    'title': 27, 'label': 25, 'tick': 22, 'legend': 20,
    'title_color': '#000000', 'label_color': '#000000',
    'tick_color': '#000000', 'legend_color': '#000000',
    'grid_which': 'both', 'grid_style': 'solid', 'grid_width': 1.0,
    'curve_width': 2.5, 'curve_style': 'solid', 'curve_scheme': 'default', 'curve_color': '#000000', 'comparison_palette': 'polvista',
    'curve_samples': 0,
    'marker': 'o', 'marker_size': 4, 'marker_fill': '#000000',
    'marker_edge_width': 1.5, 'marker_edge_color': '#000000',
    'bands_on': False, 'bands_alpha': 0.3,
    'ylim_err_factor_p': 8, 'ylim_err_factor_evpa': 4,
    'evpa_full_span': 170, 'evpa_full_ylim': (-100, 100),
}

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
PLOT_MARGINS_PX = dict(left=80, right=150, bottom=95, top=40)

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


def set_suptitle(fig, text):
    """Create-or-update `fig`'s centered suptitle (blank text hides it) --
    sized to fit PLOT_MARGINS_PX's fixed top margin."""
    if getattr(fig, '_polvista_suptitle', None) is None:
        fig._polvista_suptitle = fig.suptitle(text, fontsize=PLOT_PREFS['title'], color=PLOT_PREFS['title_color'],
                                           y=0.995, va='top')
    else:
        fig._polvista_suptitle.set_text(text)


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

    # Horizontal centering offset measured by CenterMixin (px, >0 moves the
    # axes right), kept inside the margins so neither goes negative.
    shift = getattr(canvas, '_center_shift', 0.0)
    shift = max(-(left_px - 4), min(right_px - 4, shift))
    left_px, right_px = left_px + shift, right_px - shift

    fig.subplots_adjust(left=left_px / w, right=1 - right_px / w,
                        bottom=bottom_px / h, top=1 - top_px / h,
                        **(extra_adjust or {}),)


def _bound_yaxis_ticklabels(ax):
    """Force compact '10^n'-multiplier y-tick formatting once values fall
    outside [0.01, 1000) -- keeps tick-label width (and the auto-computed
    ylabel position) within PLOT_MARGINS_PX's fixed budget regardless of
    how small/large the data gets."""
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3), useMathText=True)


# Live figure preferences, changed from the Preferences menu (see app.PreferencesDialog);
# Restore puts DEFAULT_PLOT_PREFS (top of this file) back.
PLOT_PREFS = dict(DEFAULT_PLOT_PREFS)


def sync_rcparams():
    """Mirror the font preferences into rcParams so axes, labels and legends
    created (or clear()ed) later pick them up."""
    p = PLOT_PREFS
    mpl.rcParams['axes.titlesize'] = p['title']
    mpl.rcParams['axes.titlecolor'] = p['title_color']
    mpl.rcParams['axes.labelsize'] = p['label']
    mpl.rcParams['axes.labelcolor'] = p['label_color']
    mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = p['tick']
    mpl.rcParams['xtick.labelcolor'] = mpl.rcParams['ytick.labelcolor'] = p['tick_color']
    mpl.rcParams['legend.labelcolor'] = p['legend_color']


sync_rcparams()


# Color palettes offered for the curves added to the Comparison tab
# (Preferences > Curve > Comparison): qu_fit.py's model-line colors, then
# three standard matplotlib qualitative palettes (dropdown order).
COMPARISON_PALETTES = {
    'polvista': ['darkred', 'orange', 'limegreen', 'dodgerblue', 'indigo', 'magenta', 'teal'],
    'tab10': list(mpl.colormaps['tab10'].colors),
    'Set1': list(mpl.colormaps['Set1'].colors),
    'Accent': list(mpl.colormaps['Accent'].colors),
}


# Default colors of the visualizer's own model curves, by canvas attribute
# name (component lines share their total's color).
MAIN_CURVE_COLORS = {'line_p': 'tab:blue', 'line_x': 'tab:red',
                     **dict.fromkeys(('line_I', 'line_I1', 'line_I2', 'line_I3'), 'black'),
                     **dict.fromkeys(('line_Q', 'line_Q1', 'line_Q2', 'line_Q3'), 'green'),
                     **dict.fromkeys(('line_U', 'line_U1', 'line_U2', 'line_U3'), 'orange')}


COMPONENT_LINES = ('line_I1', 'line_I2', 'line_I3', 'line_Q1', 'line_Q2', 'line_Q3',
                   'line_U1', 'line_U2', 'line_U3')


def main_curve_color(name):
    return PLOT_PREFS['curve_color'] if PLOT_PREFS['curve_scheme'] == 'custom' else MAIN_CURVE_COLORS[name]


def main_curve_style(name, components=False):
    """Line kwargs (color, width, style) for the visualizer's own model
    curve `name` (a canvas attribute such as 'line_p'); the per-component
    lines of a multi-component model keep their dashed/dotted looks."""
    style = dict(color=main_curve_color(name), linewidth=PLOT_PREFS['curve_width'])
    if not components:
        style['linestyle'] = CURVE_LINESTYLES[PLOT_PREFS['curve_style']]
    return style


def draw_band_spans(canvas, targets):
    """(Re)draw the shaded Measurements-tab bands (canvas.band_spans, in GHz)
    on `targets`, a list of (ax, x_of_ghz). Rebuilt only when what it
    depends on changed, so a slider drag doesn't pay for it."""
    on, alpha = PLOT_PREFS['bands_on'], PLOT_PREFS['bands_alpha']
    sig = (on, alpha, tuple(canvas.band_spans), tuple((id(ax), x_of(1.0)) for ax, x_of in targets))
    if sig == canvas._bands_sig:
        return
    canvas._bands_sig = sig
    _remove_lines(canvas._band_artists)
    canvas._band_artists = []
    if not on:
        return
    for ax, x_of in targets:
        for lo, hi, color in canvas.band_spans:
            canvas._band_artists.append(ax.axvspan(x_of(lo), x_of(hi), color=color, alpha=alpha, lw=0, zorder=0.5))


def apply_grid(ax):
    """(Re)draw `ax`'s grid per PLOT_PREFS. Minor lines only show where the
    axis has minor ticks (style_grid turns them on)."""
    which, ls, lw = PLOT_PREFS['grid_which'], PLOT_PREFS['grid_style'], PLOT_PREFS['grid_width']
    ax.grid(False, which='both')
    if which in ('both', 'major'):
        ax.grid(True, which='major', linestyle=ls, linewidth=lw)
    if which == 'both' and getattr(ax, '_polvista_minor', None) is None:
        ax.grid(True, which='minor', alpha=0.25, linestyle=ls, linewidth=lw)


def apply_plot_prefs(canvas):
    """Push PLOT_PREFS onto `canvas`'s existing axes and onto rcParams (so
    axes rebuilt later by clear() pick the sizes up too), then redraw."""
    p = PLOT_PREFS
    sync_rcparams()
    canvas.legend_fontsize = p['legend']
    for ax in canvas.fig.axes:
        ax.title.set_fontsize(p['title'])
        ax.title.set_color(p['title_color'])
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(p['label'])
            label.set_color(p['label_color'])
        ax.tick_params(which='both', labelsize=p['tick'], labelcolor=p['tick_color'])
        for offset in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            offset.set_fontsize(p['tick'])
            offset.set_color(p['tick_color'])
        apply_grid(ax)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(p['legend'])
                text.set_color(p['legend_color'])
    for name in MAIN_CURVE_COLORS:
        line = getattr(canvas, name, None)
        if line is not None:
            line.set_color(main_curve_color(name))
            line.set_linewidth(p['curve_width'])
            if name not in COMPONENT_LINES:
                line.set_linestyle(CURVE_LINESTYLES[p['curve_style']])
    canvas._bands_sig = None
    if hasattr(canvas, '_ref_dirty'):
        canvas._ref_dirty = True  # data points are rebuilt with the new marker style
    if getattr(canvas, 'xlabel_text', None) is not None:
        canvas.xlabel_text.set_fontsize(p['label'])
        canvas.xlabel_text.set_color(p['label_color'])
    suptitle = getattr(canvas.fig, '_polvista_suptitle', None)
    if suptitle is not None:
        suptitle.set_fontsize(p['title'])
        suptitle.set_color(p['title_color'])
    canvas.draw_idle()


def set_minor_grid(ax, on):
    """Hide (on=False) or restore (on=True) `ax`'s minor ticks and grid --
    their cost dominates a redraw on a log axis, so a slider drag turns them
    off for its duration. The locators are put back exactly as they were."""
    saved = getattr(ax, '_polvista_minor', None)
    if on:
        if saved is not None:
            ax.xaxis.set_minor_locator(saved[0])
            ax.yaxis.set_minor_locator(saved[1])
            ax._polvista_minor = None
            apply_grid(ax)
    elif saved is None:
        ax._polvista_minor = (ax.xaxis.get_minor_locator(), ax.yaxis.get_minor_locator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(False, which='minor')


def set_drag_mode(canvas, dragging):
    """Fast-redraw mode for a slider drag: minor ticks/grid off and blitting
    (see BlitDragMixin) while `dragging`, both back to normal afterwards.
    Only touches axes style_grid has set up; the caller redraws."""
    for ax in canvas.fig.axes:
        if hasattr(ax, '_polvista_minor'):
            set_minor_grid(ax, not dragging)
    if hasattr(canvas, 'blit_begin'):
        canvas.blit_begin() if dragging else canvas.blit_end()


# Extra room, as a fraction of the span, added on the side a limit is
# widened toward during a drag (see _widen_only).
BLIT_HEADROOM = 0.15


def _widen_only(ax):
    """Replace `ax`'s set_xlim/set_ylim with versions that only ever widen
    the current limits, with BLIT_HEADROOM extra room on the widened side
    (multiplicative on a log axis). A blitted drag needs limits that hold
    still: every change forces a full redraw, and with headroom a steady
    drag in one direction needs one only every few frames."""
    def make(set_fn, get_fn, log_fn):
        def setter(lo, hi=None, **kwargs):
            if hi is None:
                lo, hi = lo
            cur_lo, cur_hi = get_fn()
            if lo >= cur_lo and hi <= cur_hi:
                return cur_lo, cur_hi
            new_lo, new_hi = min(lo, cur_lo), max(hi, cur_hi)
            if log_fn() == 'log' and new_lo > 0:
                f = (new_hi / new_lo) ** BLIT_HEADROOM
                new_lo, new_hi = (new_lo / f if lo < cur_lo else new_lo), (new_hi * f if hi > cur_hi else new_hi)
            else:
                span = new_hi - new_lo
                new_lo, new_hi = (new_lo - BLIT_HEADROOM * span if lo < cur_lo else new_lo), \
                                 (new_hi + BLIT_HEADROOM * span if hi > cur_hi else new_hi)
            return set_fn(new_lo, new_hi)
        return setter
    ax.set_xlim = make(ax.set_xlim, ax.get_xlim, ax.get_xscale)
    ax.set_ylim = make(ax.set_ylim, ax.get_ylim, ax.get_yscale)


QU_POLAR_MIN_SPAN = 0.1  # smallest q (and u) range the Polar view shows [%]


def _min_span(lo, hi, span):
    """(lo, hi), widened symmetrically about its center to at least `span`."""
    if hi - lo >= span:
        return lo, hi
    mid = 0.5 * (lo + hi)
    return mid - 0.5 * span, mid + 0.5 * span


# Window resizing: redraws are throttled to one per RESIZE_LIVE_MS while the
# size is still changing, and the full-quality draw happens once it has been
# quiet for RESIZE_SETTLE_MS.
RESIZE_LIVE_MS = 60
RESIZE_SETTLE_MS = 150


class ResizeFastMixin:
    """Cheap window resizing for a FigureCanvas. Every Qt resize event used
    to queue a full redraw; while the size is still changing this instead
    applies the margins (cheap), keeps the minor ticks/grid off, redraws at
    most every RESIZE_LIVE_MS, and ignores every other draw request. Once no
    resize event has arrived for RESIZE_SETTLE_MS the minor grid comes back
    and one full draw is made (CenterMixin re-measures then too).

    Subclasses implement `_apply_margins()` and call `_resize_setup()` once
    the figure exists."""
    _resizing = False
    _last_live_draw = 0.0

    def _apply_margins(self):
        raise NotImplementedError

    def _resize_setup(self):
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._finish_resize)

    def draw_idle(self, *args, **kwargs):
        if self._resizing:
            return  # _finish_resize draws once the size settles
        super().draw_idle(*args, **kwargs)

    def resizeEvent(self, event):
        if not self._resizing:
            self._resizing = True
            for ax in self.fig.axes:
                if hasattr(ax, '_polvista_minor'):
                    set_minor_grid(ax, False)
        super().resizeEvent(event)  # resizes the figure (its own draw request is ignored above)
        self._apply_margins()
        now = time.monotonic()
        if self.isVisible() and now - self._last_live_draw >= RESIZE_LIVE_MS / 1000:
            self._last_live_draw = now
            self.draw()
        self._settle_timer.start(RESIZE_SETTLE_MS)

    def _finish_resize(self):
        self._resizing = False
        if not getattr(self, '_blit_active', False):
            for ax in self.fig.axes:
                if hasattr(ax, '_polvista_minor'):
                    set_minor_grid(ax, True)
        self._center_sig = None
        self.draw_idle()


class CenterMixin:
    """Keep a two-panel canvas's decorations (tick labels, axis labels,
    legends) horizontally centered. The fixed pixel margins budget for the
    worst case on each side, so the content is only centered by luck: with
    typical tick labels it sat a few tens of pixels off. After each settled
    draw whose tick/axis label text changed, the axes' real horizontal
    extent is measured and the whole axes box is shifted by half the
    left-right difference in free space (`_center_shift`, read by
    apply_fixed_margins). Changes under CENTER_TOL_PX are ignored and at most
    CENTER_MAX_ITERS corrections run in a row, so it can't oscillate. Skipped
    during a blitted drag, whose limits and margins have to hold still; the
    release redraw re-centers.

    Subclasses implement `_apply_margins()` and call `_install_centering()`
    once the figure exists."""
    _center_shift = 0.0
    _center_sig = None
    _center_iters = 0
    CENTER_TOL_PX = 2.0
    CENTER_MAX_ITERS = 3

    def _apply_margins(self):
        raise NotImplementedError

    def _install_centering(self):
        self.mpl_connect('draw_event', self._on_draw_center)

    def _on_draw_center(self, event):
        if self._blit_active or self._resizing:
            return
        axes = self.fig.axes
        texts = tuple(t.get_text() for ax in axes for t in ax.get_xticklabels() + ax.get_yticklabels()) \
            + tuple(ax.yaxis.label.get_text() for ax in axes)
        sig = (self.width(), texts)
        if sig == self._center_sig:
            return
        self._center_sig = sig
        bbox = mtransforms.Bbox.union([ax.get_tightbbox(event.renderer) for ax in axes])
        free_left = bbox.x0  # Axes.get_tightbbox is already in display pixels
        free_right = self.fig.bbox.width - bbox.x1
        delta = (free_right - free_left) / 2
        if abs(delta) < self.CENTER_TOL_PX:
            self._center_iters = 0
            return
        if self._center_iters >= self.CENTER_MAX_ITERS:
            return
        self._center_iters += 1
        self._center_shift += delta
        self._apply_margins()
        self._center_sig = None  # measure again after the redraw
        self.draw_idle()


class BlitDragMixin:
    """Blitted redraws for a FigureCanvas while a parameter slider is being
    dragged. The first frame of a drag (and any frame that changes the axis
    limits, the canvas size or the set of moving artists) is a full draw with
    the moving artists left out, from which the static background -- axes,
    ticks, grid, data, added curves, samples, legends -- is captured; every
    other frame restores that background and draws just the moving artists
    (`_blit_artists`), about 9x cheaper. Legends are cached as pixels and put
    back on top of the moving lines, since redrawing their mathtext every
    frame costs more than everything else; a line passing under one is
    therefore hidden there rather than faintly visible through its frame,
    until the full redraw on release. A canvas on a hidden tab is skipped.

    Subclasses implement `_blit_artists()` and end update_plot with
    `self._present()` in place of draw_idle()."""
    _blit_active = False
    _blit_state = None

    def _blit_artists(self):
        """[(axes, artist)] for everything the slider moves."""
        raise NotImplementedError

    def blit_begin(self):
        self._blit_active = True
        self._blit_state = None
        for ax in self.fig.axes:
            _widen_only(ax)

    def blit_end(self):
        if self._blit_state is not None:
            for _, art in self._blit_state['arts']:
                art.set_animated(False)
        self._blit_active = False
        self._blit_state = None
        for ax in self.fig.axes:
            ax.__dict__.pop('set_xlim', None)
            ax.__dict__.pop('set_ylim', None)

    def _present(self):
        """What update_plot does in place of draw_idle(): a plain deferred
        draw normally, a blitted frame during a drag."""
        if not self._blit_active:
            self.draw_idle()
            return
        if not self.isVisible():
            return
        arts = self._blit_artists()
        legends = [lg for lg in (ax.get_legend() for ax in self.fig.axes) if lg is not None]
        key = (self.width(), self.height(), tuple(id(a) for _, a in arts), tuple(id(lg) for lg in legends),
               tuple((ax.get_xlim(), ax.get_ylim()) for ax in self.fig.axes))
        state = self._blit_state
        if state is None or state['key'] != key:
            if state is not None:
                for _, art in state['arts']:
                    art.set_animated(False)
            for _, art in arts:
                art.set_animated(True)
            self.draw()  # everything but the moving artists
            state = self._blit_state = dict(
                key=key, arts=arts, bg=self.copy_from_bbox(self.fig.bbox),
                patches=[self.copy_from_bbox(lg.get_window_extent().expanded(1.02, 1.02)) for lg in legends])
        self.restore_region(state['bg'])
        for ax, art in state['arts']:
            ax.draw_artist(art)
        for patch in state['patches']:
            self.restore_region(patch)
        self.blit(self.fig.bbox)


def style_grid(ax):
    """Major grid at full strength, minor grid fainter and unlabeled. Must be
    re-applied after set_xscale/set_yscale/clear(), which reset the locators
    and formatters (a log axis otherwise labels its minor ticks)."""
    ax._polvista_minor = None
    ax.minorticks_on()
    apply_grid(ax)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


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


# Wavelength units offered by the Wavelength range box -> meters per unit.
WL_UNITS_M = {'mm': 1e-3, 'cm': 1e-2, 'm': 1.0}

# Line styles offered for Add curve curves (Comparison tab dropdown).
CURVE_LINESTYLES = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dash-dot': '-.',
    'dash double-dot': (0, (6, 2, 1, 2, 1, 2)),
    'dot double-dash': (0, (1, 2, 6, 2, 6, 2)),
}


def _curve_signature(curves):
    """Hashable (name, color, linestyle) per curve -- what a legend/line
    style sync needs to notice changing, as opposed to the curves' data."""
    return tuple((c['name'], c['color'], c['linestyle'], c['linewidth']) for c in curves)


def _curve_data_key(curves):
    """What a curve's data (as opposed to its style) depends on: its id,
    plus `rev`, bumped whenever a MultiNest-samples curve switches family
    or redraws its samples."""
    return tuple((c['id'], c['rev']) for c in curves)


def _style_curve_line(line, curve, label):
    line.set_color(curve['color'])
    line.set_linestyle(CURVE_LINESTYLES[curve['linestyle']])
    line.set_linewidth(curve['linewidth'])
    line.set_label(label)


def _remove_lines(lines):
    """Like _clear_artists, but tolerates lines whose axis was already
    clear()ed out from under them (StokesPlot.build_qu_mode)."""
    for ln in lines:
        try:
            ln.remove()
        except (ValueError, NotImplementedError, AttributeError):
            pass


def _style_sample_lines(lines, curve):
    """Faint posterior-sample lines of a MultiNest-samples curve -- same
    look as the Visualization tab's own spaghetti overlay (low alpha, thin),
    in the curve's color."""
    for ln in lines:
        ln.set_color(curve['color'])
        ln.set_alpha(0.12)
        ln.set_linewidth(0.7)
        ln.set_label('_nolegend_')


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
def ref_point_style(**overrides):
    """Errorbar style of a Load Data... point, from the marker preferences
    (Preferences > Figures)."""
    p = PLOT_PREFS
    style = dict(fmt=p['marker'], ms=p['marker_size'], mfc=p['marker_fill'], color=p['marker_fill'],
                 mec=p['marker_edge_color'], mew=p['marker_edge_width'], ecolor=p['marker_edge_color'],
                 elinewidth=p['marker_edge_width'], capthick=p['marker_edge_width'], capsize=2, zorder=5)
    style.update(overrides)
    return style


def _qu_point_style(fmt, color):
    """Q/U spectra points keep their own marker and fill (they tell the two
    Stokes apart); size and contour follow the preferences."""
    return ref_point_style(fmt=fmt, mfc=color, color=color)


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


class ModelPlot(BlitDragMixin, CenterMixin, ResizeFastMixin, FigureCanvas):
    #Two side-by-side axes: polarization fraction and EVPA vs lambda^2

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax_p, self.ax_x = self.fig.subplots(1, 2, gridspec_kw={'wspace': 0})
        self._apply_margins()
        self._install_centering()
        self._resize_setup()
        self.ax_x.yaxis.tick_right()
        self.ax_x.yaxis.set_label_position('right')
        self.xlabel_text = self.fig.text(0.5, 0.05, r'$\lambda^2$ [ mm$^2$ ]', ha='center',
                                    fontsize=PLOT_PREFS['label'], color=PLOT_PREFS['label_color'])
        self.w2_scale = 1.0  # lambda^2 [mm^2] -> the selected wavelength unit's square, see set_wl_unit

        # Labels/grid set once here (not in update_plot()): Axes.clear() is
        # expensive, so slider drags reuse these Line2D objects via
        # set_data() instead of clear()+replot().
        self.ax_p.set_ylabel(r'$p$ [ % ]')
        style_grid(self.ax_p)
        _bound_yaxis_ticklabels(self.ax_p)
        # labelpad=25: rotation=270 + tick_right() above put this ylabel
        # right against its own tick labels at the default pad.
        self.ax_x.set_ylabel(r'$\chi$ [ deg ]', rotation=270, labelpad=45)
        style_grid(self.ax_x)
        _bound_yaxis_ticklabels(self.ax_x)
        self.line_p, = self.ax_p.plot([], [], **main_curve_style('line_p'))
        self.line_x, = self.ax_x.plot([], [], **main_curve_style('line_x'))
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

        # Permanent curves from the Add curve button: MainWindow.curves, a
        # list shared by reference (dicts with func/n_components/pars/
        # spectral/name/color/linestyle/id). Their data is re-evaluated only
        # when wl_ext or the set of curve ids changes; style/legend syncs
        # are cheap and run on every update_plot (see _refresh_permanent).
        self.band_spans = []  # [(nu_lo, nu_hi, color)] GHz, set by MainWindow.update_plot
        self._band_artists = []
        self._bands_sig = None
        self.show_main = True  # whether the visualizer's own curve is drawn, see MainWindow.on_show_main_toggled
        self.legend_fontsize = PLOT_PREFS['legend'] 
        self.perm_curves = []
        self.perm_lines_p = []
        self.perm_lines_x = []
        self.perm_sample_lines = []  # per curve: its faint sample lines on both axes
        self._perm_arrays = []
        self._perm_cache_key = None
        self._perm_style_sig = None

    def _refresh_permanent(self, wl_ext, w2):
        """Rebuild the permanent lines if wl_ext or the curve set changed,
        sync their style/legend, and return their (p, evpa) arrays for axis
        autoscaling."""
        key = (wl_ext.tobytes(), _curve_data_key(self.perm_curves))
        if key != self._perm_cache_key:
            self._perm_cache_key = key
            self._perm_style_sig = None
            self.perm_lines_p = _clear_artists(self.perm_lines_p)
            self.perm_lines_x = _clear_artists(self.perm_lines_x)
            for lines in self.perm_sample_lines:
                _remove_lines(lines)
            self.perm_sample_lines = []
            self._perm_arrays = []
            for c in self.perm_curves:
                with spectral_state(c['spectral']):
                    fit = c['func'](wl_ext, c['pars'])
                p_c, x_c = pol(fit), evpa(fit)
                self._perm_arrays.append((p_c, x_c))
                sample_lines = []
                for sample in c.get('sample_pars', ()):
                    with spectral_state(c['spectral']):
                        s_fit = c['func'](wl_ext, sample)
                    sample_lines.append(self.ax_p.plot(w2, pol(s_fit), zorder=3)[0])
                    sample_lines.append(self.ax_x.plot(w2, evpa(s_fit), zorder=3)[0])
                self.perm_sample_lines.append(sample_lines)
                ln_p, = self.ax_p.plot(w2, p_c, zorder=4)
                ln_x, = self.ax_x.plot(w2, x_c, zorder=4)
                self.perm_lines_p.append(ln_p)
                self.perm_lines_x.append(ln_x)
        sig = _curve_signature(self.perm_curves) + (self.legend_fontsize,)
        if sig != self._perm_style_sig:
            self._perm_style_sig = sig
            for c, ln_p, ln_x, samples in zip(self.perm_curves, self.perm_lines_p, self.perm_lines_x,
                                              self.perm_sample_lines):
                _style_curve_line(ln_p, c, c['name'])
                _style_curve_line(ln_x, c, '_nolegend_')
                _style_sample_lines(samples, c)
            legend = self.ax_p.get_legend()
            if legend is not None:
                legend.remove()
            if self.perm_lines_p:
                self.ax_p.legend(handles=self.perm_lines_p, loc='best', fontsize=self.legend_fontsize)
        return self._perm_arrays

    def set_wl_unit(self, unit):
        """Show lambda^2 in `unit`^2 (a WL_UNITS_M key). Everything fed in
        from outside (reference data, measurement bands) stays in mm^2 and
        is scaled by w2_scale when drawn."""
        self.w2_scale = (1e-3 / WL_UNITS_M[unit]) ** 2
        self.xlabel_text.set_text(rf'$\lambda^2$ [ {unit}$^2$ ]')
        self._ref_dirty = True

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
            w2 = w2 * self.w2_scale
            self.ref_artists.append(_draw_errorbar_overlay(self.ax_p, w2, p, ref_point_style(), yerr=p_err))
            self.ref_artists.append(_draw_errorbar_overlay(self.ax_x, w2, evpa, ref_point_style(), yerr=evpa_err))

        self.meas_artists = _clear_artists(self.meas_artists)
        if self.meas_bands:
            for band in self.meas_bands:
                style = _meas_point_style(band['color'], fmt='D')
                self.meas_artists.append(_draw_errorbar_overlay(
                    self.ax_p, band['w2'] * self.w2_scale, band['p'], style, yerr=band['p_err']))
                self.meas_artists.append(_draw_errorbar_overlay(
                    self.ax_x, band['w2'] * self.w2_scale, band['evpa'], style, yerr=band['evpa_err']))

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
        w2 = wl_ext ** 2 * 1e6 * self.w2_scale  # lambda^2 [mm^2], or the selected unit's square

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
            style_grid(self.ax_p)
            style_grid(self.ax_x)
            self.xscale = xscale
        xlo, xhi = np.min(w2) * 0.9, np.max(w2) * 1.05
        perm = self._refresh_permanent(wl_ext, w2)
        # A hidden visualizer curve (Show visualizer curve unchecked) must
        # not influence the limits either -- seed them from the first added
        # curve instead (the box can only be unchecked while one exists).
        seed_p, seed_x = (p, evpa_vals) if self.show_main or not perm else perm[0]
        _, p_seed_max = _finite_bounds(seed_p, fallback=(0.0, 1e-6))
        p_max = max(p_seed_max, 1e-6)
        x_lo, x_hi = _finite_bounds(seed_x, fallback=(0.0, 0.0))
        # Widen the autoscale to also cover any reference data loaded via
        # the Load Data... button, so it isn't silently clipped out of view.
        for p_c, x_c in perm:
            p_max = max(p_max, _finite_bounds(p_c, fallback=(0.0, 0.0))[1])
            x_lo, x_hi = _widen_range(x_lo, x_hi, x_c)
        if self.ref_data is not None:
            ref_w2, ref_p, _, ref_evpa, _ = self.ref_data
            ref_w2 = ref_w2 * self.w2_scale
            xlo, xhi = min(xlo, np.min(ref_w2) * 0.9), max(xhi, np.max(ref_w2) * 1.05)
            p_max = max(p_max, np.max(ref_p))
            x_lo, x_hi = _widen_range(x_lo, x_hi, ref_evpa)
        if self.meas_bands:
            for band in self.meas_bands:
                xlo = min(xlo, np.min(band['w2']) * self.w2_scale * 0.9)
                xhi = max(xhi, np.max(band['w2']) * self.w2_scale * 1.05)
                p_max = max(p_max, np.max(band['p'] + band['p_err']))
                x_lo, x_hi = _widen_range(x_lo, x_hi, band['evpa'], band['evpa_err'])

        # With data or simulated measurements loaded, the y-limits follow the
        # points alone: [min(y) - k max(yerr), max(y) + k max(yerr)] with k = PLOT_PREFS['ylim_err_factor_p'] for p (lower limit
        # clipped at 0) and ['ylim_err_factor_evpa'] for the EVPA.
        p_ylim = x_ylim = None
        p_pts, p_errs, x_pts, x_errs = [], [], [], []
        if self.ref_data is not None:
            _, ref_p, ref_p_err, ref_evpa, ref_evpa_err = self.ref_data
            p_pts.append(ref_p), p_errs.append(ref_p_err), x_pts.append(ref_evpa), x_errs.append(ref_evpa_err)
        for band in self.meas_bands or ():
            p_pts.append(band['p']), p_errs.append(band['p_err'])
            x_pts.append(band['evpa']), x_errs.append(band['evpa_err'])
        if p_pts:
            def data_ylim(pts, errs, k):
                y, e = np.concatenate(pts), np.concatenate(errs)
                finite = np.isfinite(y)
                err = np.max(e[np.isfinite(e)], initial=0.0)
                return float(np.min(y[finite])) - k * err, float(np.max(y[finite])) + k * err
            if any(np.isfinite(np.concatenate(p_pts))):
                lo, hi = data_ylim(p_pts, p_errs, PLOT_PREFS['ylim_err_factor_p'])
                p_ylim = (max(lo, 0.0), hi)
            if any(np.isfinite(np.concatenate(x_pts))):
                x_ylim = data_ylim(x_pts, x_errs, PLOT_PREFS['ylim_err_factor_evpa'])
                x_all = np.concatenate(x_pts)
                x_all = x_all[np.isfinite(x_all)]
                if np.ptp(x_all) > PLOT_PREFS['evpa_full_span']:
                    x_ylim = PLOT_PREFS['evpa_full_ylim']  # points spread over (nearly) the full EVPA range

        self.line_p.set_data(w2, p)
        self.ax_p.set_xlim(xlo, xhi)
        self.ax_p.set_ylim(*(p_ylim if p_ylim and p_ylim[1] > p_ylim[0] else (0, 1.3 * p_max)))

        self.line_x.set_data(w2, evpa_vals)
        self.ax_x.set_xlim(xlo, xhi)

        # Enforce a minimum 2 deg y-span so a near-flat EVPA curve (small
        # |phi|) doesn't get visually exaggerated by an overly tight autoscale.
        margin = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 0.05
        x_lo, x_hi = x_lo - margin, x_hi + margin
        if x_hi - x_lo < 2.0:
            mid = 0.5 * (x_lo + x_hi)
            x_lo, x_hi = mid - 1.0, mid + 1.0
        self.ax_x.set_ylim(*(x_ylim if x_ylim and x_ylim[1] > x_ylim[0] else (x_lo, x_hi)))
        self.line_p.set_visible(self.show_main)
        self.line_x.set_visible(self.show_main)
        if self._ref_dirty:
            self.draw_reference()
            self._ref_dirty = False
        to_w2 = lambda nu_ghz: (C / (nu_ghz * 1e6)) ** 2 * self.w2_scale  # GHz -> lambda^2 in the axis unit
        draw_band_spans(self, [(self.ax_p, to_w2), (self.ax_x, to_w2)])
        self._present()

    def _blit_artists(self):
        return [(self.ax_p, self.line_p), (self.ax_x, self.line_x)]

    def _apply_margins(self):
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0})


def stokes_to_frac_qu(I, Q, U, I_err=None, Q_err=None, U_err=None):
    """q=Q/I, u=U/I in percent (the fractions are the same p*cos(2EVPA)/
    p*sin(2EVPA) quantity the models themselves compute) and their
    propagated errors -- errors are None if none of I_err/Q_err/U_err
    were given, otherwise missing ones are treated as 0. Used by
    StokesPlot's Polar view so its trajectory reflects pure polarization
    state, undistorted by I's own separately-varying spectral shape --
    same error-propagation formula as app.py's load_data_action."""
    q, u = 100.0 * Q / I, 100.0 * U / I
    if I_err is None and Q_err is None and U_err is None:
        return q, u, None, None

    def frac_err(val, val_err, denom, denom_err):
        val_err = val_err if val_err is not None else 0.0
        denom_err = denom_err if denom_err is not None else 0.0
        return np.sqrt((val_err / denom) ** 2 + (val * denom_err / denom ** 2) ** 2)

    return q, u, 100.0 * frac_err(Q, Q_err, I, I_err), 100.0 * frac_err(U, U_err, I, I_err)


class StokesPlot(BlitDragMixin, CenterMixin, ResizeFastMixin, FigureCanvas):
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
        self._apply_margins()
        self._install_centering()
        self._resize_setup()

        self.ax_I.set_xlabel(r'$\nu$ [ GHz ]')
        self.ax_I.set_ylabel(r'$I$ [ normalized ]')
        self.ax_I.set_yscale('log')
        style_grid(self.ax_I)
        self.line_I, = self.ax_I.plot([], [], label=r'$I$', **main_curve_style('line_I'))
        # Per-component curves for two-component models (see
        # models.stokes_components) -- component 1 dashed, component 2
        # dotted, same color as the total's own solid line; left empty
        # (and so invisible) for single-component models. Spectra mode's
        # own Q1/U1/Q2/U2 counterparts are built in build_qu_mode, since
        # that axis is torn down/rebuilt on every mode switch.
        self.line_I1, = self.ax_I.plot([], [], linestyle='dashed', label=r'$I_1$', **main_curve_style('line_I1', True))
        self.line_I2, = self.ax_I.plot([], [], linestyle='dotted', label=r'$I_2$', **main_curve_style('line_I2', True))
        self.line_I3, = self.ax_I.plot([], [], linestyle='dashdot', label=r'$I_3$', **main_curve_style('line_I3', True))  # 3-component models
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
        self.line_Q1 = self.line_U1 = self.line_Q2 = self.line_U2 = self.line_Q3 = self.line_U3 = None
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

        # Permanent curves from the Add curve button -- see ModelPlot's own
        # perm_curves. ax_I lines persist; the ax_QU ones are torn down with
        # that axis on every build_qu_mode, so they rebuild via the cache key.
        self.band_spans = []  # see ModelPlot.band_spans
        self._band_artists = []
        self._bands_sig = None
        self.show_main = True  # see ModelPlot.show_main
        self.legend_fontsize = PLOT_PREFS['legend']
        self.perm_curves = []
        self.perm_lines_I = []
        self.perm_lines_QU = []  # per curve: [line] (Polar) or [Q line, U line] (Spectra)
        self.perm_sample_lines = []  # per curve: its faint sample lines on ax_I and ax_QU
        self._perm_arrays = []
        self._perm_cache_key = None
        self._perm_style_sig = None
        self.build_qu_mode()

    def _sample_lines(self, c, wl_ext, nu_min, order, nu_s):
        """Faint lines for a MultiNest-samples curve's drawn samples (none
        for an ordinary curve): I on ax_I, Q/U (Spectra) or the fractional
        q-u trajectory (Polar) on ax_QU."""
        lines = []
        func, n_comp = c['func'], c['n_components']
        if self.mode == 'Polar':
            w2_grid = np.linspace(wl_ext.max() ** 2, wl_ext.min() ** 2, min(len(wl_ext), QU_POLAR_MAX_SEGMENTS))
            wl_qu = np.sqrt(w2_grid)
        with spectral_state(c['spectral']):
            for sample in c.get('sample_pars', ()):
                s_I = stokes_I(wl_ext, n_comp, sample, nu_min=nu_min)
                lines += self.ax_I.plot(nu_s, s_I[order], zorder=3)
                if self.mode == 'Polar':
                    s_Iq = stokes_I(wl_qu, n_comp, sample, nu_min=nu_min)
                    s_Q, s_U = stokes_QU(wl_qu, func, n_comp, sample, nu_min=nu_min, I=s_Iq)
                    lines += self.ax_QU.plot(100 * s_Q / s_Iq, 100 * s_U / s_Iq, zorder=3)
                else:
                    s_Q, s_U = stokes_QU(wl_ext, func, n_comp, sample, nu_min=nu_min, I=s_I)
                    lines += self.ax_QU.plot(nu_s, s_Q[order], zorder=3)
                    lines += self.ax_QU.plot(nu_s, s_U[order], zorder=3)
        return lines

    def _refresh_permanent(self, wl_ext, nu_min, nu_s, order):
        """Rebuild the permanent lines if their inputs changed, sync their
        style, and return per-curve dicts of I/Q/U (nu-sorted) and, in
        Polar mode, q/u arrays for axis autoscaling. The ax_I legend is
        built by _update_legend_I."""
        key = (wl_ext.tobytes(), nu_min, self.mode, _curve_data_key(self.perm_curves))
        if key != self._perm_cache_key:
            self._perm_cache_key = key
            self._perm_style_sig = None
            self.perm_lines_I = _clear_artists(self.perm_lines_I)
            for lines in self.perm_lines_QU:
                _remove_lines(lines)
            self.perm_lines_QU = []
            for lines in self.perm_sample_lines:
                _remove_lines(lines)
            self.perm_sample_lines = []
            self._perm_arrays = []
            for c in self.perm_curves:
                func, n_comp, pars = c['func'], c['n_components'], c['pars']
                with spectral_state(c['spectral']):
                    I = stokes_I(wl_ext, n_comp, pars, nu_min=nu_min)
                    Q, U = stokes_QU(wl_ext, func, n_comp, pars, nu_min=nu_min, I=I)
                    arr = dict(I=I[order], Q=Q[order], U=U[order])
                    if self.mode == 'Polar':
                        w2_grid = np.linspace(wl_ext.max() ** 2, wl_ext.min() ** 2,
                                              min(len(wl_ext), QU_POLAR_MAX_SEGMENTS))
                        wl_qu = np.sqrt(w2_grid)
                        I_qu = stokes_I(wl_qu, n_comp, pars, nu_min=nu_min)
                        Q_qu, U_qu = stokes_QU(wl_qu, func, n_comp, pars, nu_min=nu_min, I=I_qu)
                        arr['q'], arr['u'] = 100 * Q_qu / I_qu, 100 * U_qu / I_qu
                sample_lines = self._sample_lines(c, wl_ext, nu_min, order, nu_s)
                self.perm_sample_lines.append(sample_lines)
                ln_I, = self.ax_I.plot(nu_s, arr['I'], zorder=4)
                self.perm_lines_I.append(ln_I)
                if self.mode == 'Polar':
                    lines = self.ax_QU.plot(arr['q'], arr['u'], zorder=4)
                else:
                    lines = (self.ax_QU.plot(nu_s, arr['Q'], zorder=4)
                             + self.ax_QU.plot(nu_s, arr['U'], zorder=4))
                self.perm_lines_QU.append(lines)
                self._perm_arrays.append(arr)
        sig = _curve_signature(self.perm_curves) + (self.ax_I_two_comp, self.legend_fontsize)
        if sig != self._perm_style_sig:
            self._perm_style_sig = sig
            for c, ln_I, lines, samples in zip(self.perm_curves, self.perm_lines_I, self.perm_lines_QU,
                                               self.perm_sample_lines):
                _style_sample_lines(samples, c)
                _style_curve_line(ln_I, c, c['name'])
                for k, ln in enumerate(lines):
                    # Spectra mode: only the Q line carries the curve's legend entry
                    _style_curve_line(ln, c, c['name'] if k == 0 else '_nolegend_')
            # Legends: the usual I/I1/I2 and Q/U ones only while no curve has
            # been added; once one has, just the curves' own labels.
            if self.perm_curves:
                handles_I = self.perm_lines_I
            else:
                handles_I = ([self.line_I, self.line_I1, self.line_I2] + ([self.line_I3] if self.ax_I_two_comp == 3 else [])
                             if self.ax_I_two_comp else [self.line_I])
            self.ax_I.legend(handles_I, [h.get_label() for h in handles_I], loc='lower left',
                             fontsize=self.legend_fontsize)
            if self.mode == 'Spectra':
                legend = self.ax_QU.get_legend()
                if legend is not None:
                    legend.remove()
                handles_QU = ([lines[0] for lines in self.perm_lines_QU] if self.perm_curves
                              else [self.line_Q, self.line_U])
                self.ax_QU.legend(handles_QU, [h.get_label() for h in handles_QU], loc='upper right',
                                  fontsize=self.legend_fontsize)
        return self._perm_arrays

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
        self._bands_sig = None  # the clear() above dropped ax_QU's shaded bands
        apply_grid(self.ax_QU)
        # Mirror the y-axis onto the right spine (like ModelPlot.ax_x) so
        # it reads correctly against ax_I's own left-side axis across
        # their shared, gapless seam.
        self.ax_QU.yaxis.tick_right()
        self.ax_QU.yaxis.set_label_position('right')
        self.ref_artists_QU = []
        self.meas_artists_QU = []
        self._perm_cache_key = None  # the permanent Q/U lines went with the clear() above
        self.perm_lines_QU = []
        n_samples = len(self.posterior_samples) if self.posterior_samples is not None else 0
        if self.mode == 'Polar':
            self.ax_QU.set_xlabel(r'$q$ [ % ]')
            # labelpad=25, not the default ~4 -- see ModelPlot.ax_x's own
            # comment (same rotation=270 + tick_right() crowding).
            self.ax_QU.set_ylabel(r'$u$ [ % ]', rotation=270, labelpad=45)
            # Percent values: plain tick labels, never an x10^n offset above the axis.
            self.ax_QU.ticklabel_format(axis='both', style='plain', useOffset=False)
            self.line_Q = self.line_U = None
            self.line_Q1 = self.line_U1 = self.line_Q2 = self.line_U2 = self.line_Q3 = self.line_U3 = None
            self.line_QU = LineCollection([], cmap=QU_RAINBOW_CMAP, norm=mcolors.Normalize(0, 1), zorder=3)
            self.ax_QU.add_collection(self.line_QU)
            self.sample_lines_Q, self.sample_lines_U = [], []
            self.sample_lines_QU = [self.ax_QU.plot([], [], color='tab:purple', alpha=0.1, lw=0.6, zorder=1)[0]
                                     for _ in range(n_samples)]
            # axvline/axhline span the whole axes whatever limits update_plot
            # sets later (vlines/hlines froze their extent at the limits of this moment).
            self.ax_QU.axvline(0.0, linestyle='dashed', color='k')
            self.ax_QU.axhline(0.0, linestyle='dashed', color='k')
        else:
            self.ax_QU.set_xlabel(r'$\nu$ [ GHz ]')
            self.ax_QU.set_ylabel(r'$Q$, $U$ [ normalized ]', rotation=270, labelpad=45)
            _bound_yaxis_ticklabels(self.ax_QU)
            self.ax_QU.set_xscale(self.xscale or 'linear')
            style_grid(self.ax_QU)
            self.line_QU = None
            self.line_Q, = self.ax_QU.plot([], [], label=r'$Q$', **main_curve_style('line_Q'))
            self.line_U, = self.ax_QU.plot([], [], label=r'$U$', **main_curve_style('line_U'))
            # Per-component Q/U (see line_I1/line_I2) -- no label, so they
            # stay out of the Q/U legend above.
            self.line_Q1, = self.ax_QU.plot([], [], linestyle='dashed', **main_curve_style('line_Q1', True))
            self.line_U1, = self.ax_QU.plot([], [], linestyle='dashed', **main_curve_style('line_U1', True))
            self.line_Q2, = self.ax_QU.plot([], [], linestyle='dotted', **main_curve_style('line_Q2', True))
            self.line_U2, = self.ax_QU.plot([], [], linestyle='dotted', **main_curve_style('line_U2', True))
            self.line_Q3, = self.ax_QU.plot([], [], linestyle='dashdot', **main_curve_style('line_Q3', True))
            self.line_U3, = self.ax_QU.plot([], [], linestyle='dashdot', **main_curve_style('line_U3', True))
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
            self.ref_artists_I.append(_draw_errorbar_overlay(self.ax_I, nu, I, ref_point_style(), yerr=I_err))
            if self.mode == 'Polar':
                q, u, q_err, u_err = stokes_to_frac_qu(I, Q, U, I_err, Q_err, U_err)
                self.ref_artists_QU.append(_draw_errorbar_overlay(
                    self.ax_QU, q, u, ref_point_style(), xerr=q_err, yerr=u_err))
            else:
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    nu, Q, yerr=Q_err, **_qu_point_style('s', 'darkgreen')))
                self.ref_artists_QU.append(self.ax_QU.errorbar(
                    nu, U, yerr=U_err, **_qu_point_style('^', 'darkorange')))

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
            style_grid(self.ax_I)
            if self.mode == 'Spectra':
                self.ax_QU.set_xscale(xscale)
                style_grid(self.ax_QU)
            self.xscale = xscale

        self.line_I.set_data(nu_s, arrays['I_s'])
        for k, ln in enumerate((self.line_I1, self.line_I2, self.line_I3), start=1):
            if arrays['two_comp'] and k <= arrays['n_comp']:
                ln.set_data(nu_s, arrays[f'I{k}_s'])
            else:
                ln.set_data([], [])

        show_samples = self.posterior_samples is not None and self.posterior_model is model_func
        samples_changed = self._refresh_sample_lines_I(
            wl_ext, model_func, n_components, nu_min, order, nu_s, show_samples)

        if self.mode == 'Polar':
            q_s, u_s = self._draw_polar_mode(wl_ext, model_func, n_components, pars, nu_min, show_samples)
        else:
            q_s = u_s = None
            self._draw_spectra_mode(arrays, samples_changed, show_samples,
                                     wl_ext, model_func, n_components, nu_min, order)

        arrays['perm'] = self._refresh_permanent(wl_ext, nu_min, nu_s, order)
        self._apply_axis_limits(arrays, q_s, u_s)
        for ln in (self.line_I, self.line_I1, self.line_I2, self.line_I3, self.line_Q, self.line_U, self.line_Q1,
                   self.line_U1, self.line_Q2, self.line_U2, self.line_Q3, self.line_U3, self.line_QU):
            if ln is not None:
                ln.set_visible(self.show_main)

        if self._ref_dirty:
            self.draw_reference()
            self._ref_dirty = False
        identity = lambda nu_ghz: nu_ghz
        draw_band_spans(self, [(self.ax_I, identity)] + ([(self.ax_QU, identity)] if self.mode == 'Spectra' else []))
        self._present()

    def _blit_artists(self):
        arts = [(self.ax_I, ln) for ln in (self.line_I, self.line_I1, self.line_I2, self.line_I3)]
        qu = ([self.line_QU] if self.mode == 'Polar' else
              [self.line_Q, self.line_U, self.line_Q1, self.line_U1, self.line_Q2, self.line_U2,
               self.line_Q3, self.line_U3])
        return arts + [(self.ax_QU, ln) for ln in qu if ln is not None]

    def _update_stokes_arrays(self, wl_ext, model_func, n_components, pars, nu_min):
        """Evaluate Stokes I/Q/U (total, plus per-component for a two-
        component model) on wl_ext and sort by ascending frequency;
        returns them in a dict keyed by 'order'/'nu_s'/'I_s'/'Q_s'/'U_s'/
        'two_comp'/'I1_s'/'Q1_s'/'U1_s'/'I2_s'/'Q2_s'/'U2_s' (the *_s
        component keys are None unless two_comp). Also keeps ax_I's
        legend (I vs I/I1/I2[/I3]) in sync with the component count."""
        I = stokes_I(wl_ext, n_components, pars, nu_min=nu_min)
        Q, U = stokes_QU(wl_ext, model_func, n_components, pars, nu_min=nu_min, I=I)
        nu_ghz = C / wl_ext / 1e9

        order = np.argsort(nu_ghz)
        nu_s, I_s, Q_s, U_s = nu_ghz[order], I[order], Q[order], U[order]

        two_comp = n_components >= 2
        arrays = dict(order=order, nu_s=nu_s, I_s=I_s, Q_s=Q_s, U_s=U_s, two_comp=two_comp,
                      n_comp=n_components if two_comp else 1,
                      **{f'{a}{k}_s': None for k in (1, 2, 3) for a in 'IQU'})
        if two_comp:
            for k, (I_k, Q_k, U_k) in enumerate(stokes_components(wl_ext, model_func, pars, nu_min=nu_min), start=1):
                arrays.update({f'I{k}_s': I_k[order], f'Q{k}_s': Q_k[order], f'U{k}_s': U_k[order]})

        n_curves = n_components if two_comp else 0  # per-component curves drawn: 0, 2 or 3
        if n_curves != self.ax_I_two_comp:
            self.ax_I_two_comp = n_curves
            self._perm_style_sig = None  # legend rebuilt by _refresh_permanent
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
        q_s, u_s = 100 * Q_qu / I_qu, 100 * U_qu / I_qu  # percent
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
                    ln_QU.set_data(100 * s_Q / s_I, 100 * s_U / s_I)
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
        for k, (ln_Q, ln_U) in enumerate(((self.line_Q1, self.line_U1), (self.line_Q2, self.line_U2),
                                          (self.line_Q3, self.line_U3)), start=1):
            if arrays['two_comp'] and k <= arrays['n_comp']:
                ln_Q.set_data(nu_s, arrays[f'Q{k}_s'])
                ln_U.set_data(nu_s, arrays[f'U{k}_s'])
            else:
                ln_Q.set_data([], [])
                ln_U.set_data([], [])
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
        Spectra: raw Q/U over their own extent). `q_s`/`u_s` (Polar only,
        from _draw_polar_mode) are None in Spectra mode."""
        perm = arrays['perm']
        if not self.show_main and perm:
            # Hidden visualizer curve: seed every limit from the first added
            # curve instead (see ModelPlot.update_plot's own note).
            arrays = dict(arrays, I_s=perm[0]['I'], Q_s=perm[0]['Q'], U_s=perm[0]['U'], two_comp=False)
            if q_s is not None:
                q_s, u_s = perm[0]['q'], perm[0]['u']
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
            for k in range(1, arrays['n_comp'] + 1):
                ik_min, ik_max = _finite_bounds(arrays[f'I{k}_s'], fallback=(1e-6, 1e-6))
                i_min = min(i_min, max(ik_min, 1e-6))
                i_max = max(i_max, ik_max)
        for arr in arrays['perm']:
            i_lo, i_hi = _finite_bounds(arr['I'], fallback=(1e-6, 1e-6))
            i_min, i_max = min(i_min, max(i_lo, 1e-6)), max(i_max, i_hi)
            q_min, q_max = _widen_range(q_min, q_max, arr['Q'])
            u_min, u_max = _widen_range(u_min, u_max, arr['U'])
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
            for k in range(1, arrays['n_comp'] + 1):
                qk_min, qk_max = _finite_bounds(arrays[f'Q{k}_s'], fallback=(0.0, 0.0))
                uk_min, uk_max = _finite_bounds(arrays[f'U{k}_s'], fallback=(0.0, 0.0))
                q_min_spectra, q_max_spectra = min(q_min_spectra, qk_min), max(q_max_spectra, qk_max)
                u_min_spectra, u_max_spectra = min(u_min_spectra, uk_min), max(u_max_spectra, uk_max)

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
            for arr in arrays['perm']:
                frac_q_min, frac_q_max = _widen_range(frac_q_min, frac_q_max, arr['q'])
                frac_u_min, frac_u_max = _widen_range(frac_u_min, frac_u_max, arr['u'])
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
            # Each axis over its own extent (not symmetric about 0, and no
            # longer square): a margin of 10% of the span on each side.
            q_pad = 0.1 * max(frac_q_max - frac_q_min, 1e-6)
            u_pad = 0.1 * max(frac_u_max - frac_u_min, 1e-6)
            # Never narrower than QU_POLAR_MIN_SPAN, so a (nearly) point-like
            # trajectory isn't magnified into noise -- like ModelPlot's
            # minimum EVPA span.
            self.ax_QU.set_xlim(*_min_span(frac_q_min - q_pad, frac_q_max + q_pad, QU_POLAR_MIN_SPAN))
            self.ax_QU.set_ylim(*_min_span(frac_u_min - u_pad, frac_u_max + u_pad, QU_POLAR_MIN_SPAN))
        else:
            # Tight to the Q/U extent (all curves, data and error bars), not
            # symmetric about 0: a margin of 10% of the span on each side.
            y_lo, y_hi = min(q_min_spectra, u_min_spectra), max(q_max_spectra, u_max_spectra)
            pad = 0.1 * max(y_hi - y_lo, 1e-6)
            self.ax_QU.set_xlim(xlo, xhi)
            self.ax_QU.set_ylim(y_lo - pad, y_hi + pad)

    def _apply_margins(self):
        apply_fixed_margins(self.fig, self, extra_adjust={'wspace': 0.0}, margins=STOKES_MARGINS_PX)


class RMSynthPlot(ResizeFastMixin, FigureCanvas):
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
        self._apply_margins()
        self._resize_setup()
        self.set_empty(1.0)

    def _style_axes(self, exponent=0):
        self.ax.set_xlabel(rf'Faraday depth $\phi$  [ $10^{{{exponent}}}$ rad m$^{{-2}}$ ]')
        self.ax.set_ylabel(r'$|F(\phi)|$  [ fractional polarization ]')
        apply_grid(self.ax)
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
        ax.legend(loc='upper right', fontsize=PLOT_PREFS['legend'])
        apply_fixed_margins(self.fig, self)
        self.draw_idle()

    def _apply_margins(self):
        apply_fixed_margins(self.fig, self)
