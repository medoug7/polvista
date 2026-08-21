"""LaTeX/math rendering for polvista: mathtext-rendered parameter labels and
model equations (`latex_pixmap`, `fit_equation_pixmap`), and the
help/*.tex viewer (`TexViewerDialog`, backed by a real pdflatex toolchain).

Split out of app.py to keep the GUI module focused on the app itself.
"""
import os
import io
import re
import html
import glob
import base64
import shutil
import tempfile
import functools
import subprocess

from PyQt5.QtCore import Qt, QBuffer, QIODevice
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextBrowser, QShortcut)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


LATEX_DPI = 200
SCREEN_DPI = 96  # Qt's nominal logical DPI for point-sized widgets/fonts



@functools.lru_cache(maxsize=128)
def latex_pixmap(latex, fontsize=13, facecolor=None, pad_inches=0.02, multialignment='left', dpi=LATEX_DPI):
    """Rasterize a mathtext string (e.g. r'$\\phi_1$') to a QPixmap, using
    matplotlib's built-in mathtext renderer (same STIX font as the plots) so
    slider/equation labels don't need a real LaTeX install or Qt-side math
    support. `facecolor=None` gives a transparent background (slider
    labels); pass e.g. 'white' for an opaque card (the equation display).
    `multialignment` ('left'/'center'/'right') only matters for a multi-line
    (embedded '\\n') `latex` string -- it controls how the shorter lines are
    aligned relative to the widest one (e.g. the equation card's stacked
    spectral-definition + P(lambda) lines, centered relative to each other).
    `dpi` defaults to the module's baseline LATEX_DPI, but a caller that
    knows it's rendering for a specific screen (e.g. a HiDPI display, where
    LATEX_DPI/SCREEN_DPI alone under-supplies pixels) can pass a higher
    value -- see `pixmap.setDevicePixelRatio(dpi / SCREEN_DPI)` below,
    which is what actually makes the extra raster pixels legible rather
    than just enlarging the on-screen image.

    Mathtext parsing of a complex nested expression is genuinely slow (tens
    of ms), so this is memorized -- every (model, slider, dpi) label string
    is static for the app's lifetime, so re-rendering it on every model
    switch is pure waste. Safe to cache: QPixmap is copy-on-write and a
    cached one can be handed to multiple QLabels without them affecting
    each other."""
    fig = Figure()
    if facecolor is None:
        fig.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor(facecolor)

    text = fig.text(0, 0, latex, fontsize=fontsize, multialignment=multialignment)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    bbox = text.get_window_extent(renderer=canvas.get_renderer())
    width_in = max(bbox.width / fig.dpi, 0.05)
    height_in = max(bbox.height / fig.dpi, 0.05)
    fig.set_size_inches(width_in, height_in)

    buf = io.BytesIO()
    savefig_kwargs = dict(format='png', dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    if facecolor is None:
        savefig_kwargs['transparent'] = True
    else:
        savefig_kwargs['facecolor'] = facecolor
    fig.savefig(buf, **savefig_kwargs)
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue(), 'PNG')
    pixmap.setDevicePixelRatio(dpi / SCREEN_DPI)
    return pixmap


def fit_equation_pixmap(latex, max_width_px, max_height_px, fontsize_max=24, fontsize_min=16,
                          multialignment='center'):
    """Render `latex` as large as possible (capped at fontsize_max) while
    fitting within max_width_px x max_height_px, down to fontsize_min -- so
    the equation card never needs a scrollbar, however small the window.
    `multialignment` centers a multi-line equation's shorter line(s)
    relative to the widest one by default -- see latex_pixmap.

    Mathtext size scales ~linearly with fontsize for a fixed string, so
    instead of shrinking one point at a time (up to 12 renders, each a slow
    mathtext parse), render once at fontsize_max, extrapolate the fontsize
    that would fit both dimensions, and do a single corrective render --
    at most 2 renders total. Not memoized itself (max_width/height change
    on every resize), but it's built on top of the memoized `latex_pixmap`,
    which is -- so re-fitting to a previously-seen fontsize is still cheap."""
    pixmap = latex_pixmap(latex, fontsize=fontsize_max, facecolor='white', pad_inches=0.05,
                            multialignment=multialignment)
    width = pixmap.width() / pixmap.devicePixelRatio()
    height = pixmap.height() / pixmap.devicePixelRatio()
    scale = min(max_width_px / width, max_height_px / height, 1.0)
    if scale >= 1.0:
        return pixmap
    target = max(fontsize_min, int(fontsize_max * scale))
    if target >= fontsize_max:
        return pixmap
    return latex_pixmap(latex, fontsize=target, facecolor='white', pad_inches=0.05,
                          multialignment=multialignment)


math_snippet_cache = {}

# Fallback DPI for standalone `render_math_batch` calls. `TexViewerDialog`
# doesn't use this default -- it always passes an explicit dpi computed by
# `target_math_dpi` for the current zoom and the real screen's
# devicePixelRatio, so formulas are rasterized at exactly the density the
# screen needs rather than a guessed fixed value.
EQ_DPI = 300


def numbered_page_files(tmp_dir, prefix):
    """glob.glob's results for '<prefix>-<N>.png' sort lexicographically
    (page-10 before page-2), which silently scrambles formula order past
    the 9th one -- sort by the actual page number instead."""
    files = glob.glob(os.path.join(tmp_dir, prefix + '-*.png'))

    def page_num(path):
        m = re.search(r'-(\d+)\.png$', path)
        return int(m.group(1)) if m else 0
    return sorted(files, key=page_num)


def render_math_batch(math_sources, dpi=EQ_DPI):
    """Render a list of raw LaTeX math strings (no surrounding $) to
    transparent-background QPixmaps, one real pdflatex + pdftocairo pass for
    the whole batch rather than one process per formula -- each source is
    wrapped in its own `preview` environment, which (with the `tightpage`
    option) makes pdflatex emit one tightly-cropped PDF page per formula, so
    a single compile is enough regardless of how many formulas there are.

    Already-seen sources (keyed on (source, dpi), globally across calls) are
    served from `math_snippet_cache` and skipped in the batch -- callers
    re-render the same help doc on every zoom change, and re-running
    pdflatex for formulas that haven't changed would be pure waste."""
    results = [None] * len(math_sources)
    to_render = []
    for i, src in enumerate(math_sources):
        key = (src, dpi)
        cached = math_snippet_cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            to_render.append((i, src))
    if not to_render:
        return results

    tmp_dir = tempfile.mkdtemp(prefix='polvista_eq_')
    try:
        tex_file = os.path.join(tmp_dir, 'batch.tex')
        pieces = [
            r'\documentclass[11pt]{article}',
            r'\usepackage[active,tightpage]{preview}',
            r'\usepackage{amsmath, amssymb}',
            r'\PreviewEnvironment{preview}',
            r'\begin{document}',
        ]
        for _, src in to_render:
            pieces.append(r'\begin{preview}$' + src + r'$\end{preview}')
        pieces.append(r'\end{document}')
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pieces) + '\n')

        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'batch.tex'],
            cwd=tmp_dir, capture_output=True, text=True, timeout=60)
        pdf_file = os.path.join(tmp_dir, 'batch.pdf')
        if result.returncode != 0 or not os.path.isfile(pdf_file):
            raise RuntimeError(result.stdout[-2000:] or result.stderr[-2000:])

        subprocess.run(
            ['pdftocairo', '-png', '-transp', '-r', str(dpi), pdf_file, os.path.join(tmp_dir, 'page')],
            check=True, capture_output=True, timeout=60)
        page_files = numbered_page_files(tmp_dir, 'page')
        if len(page_files) != len(to_render):
            raise RuntimeError(
                f'Expected {len(to_render)} rendered formulas, got {len(page_files)}')
        for (i, src), page_file in zip(to_render, page_files):
            pixmap = QPixmap(page_file)
            math_snippet_cache[(src, dpi)] = pixmap
            results[i] = pixmap
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


# Private-use-area sentinel for inline placeholder tokens, chosen so it can
# never collide with real document text and (unlike a NUL byte) is a
# perfectly ordinary character as far as Qt's C++ QString conversion is
# concerned.
MARK = '\uE000'
BLOCK_TOKEN_RE = re.compile(re.escape(MARK) + r'B(\d+)' + re.escape(MARK))
IMG_TOKEN_RE = re.compile(re.escape(MARK) + r'I(\d+)' + re.escape(MARK))

# Recognizes the specific LaTeX subset the help/*.tex files are written in:
# \section{}/\subsection{} headings, \begin{equation}\label{}...\end{equation}
# (numbered, referenceable via \ref{}), \begin{center}$...$\end{center}
# (unnumbered display math), inline $...$ math, \ref{}, and bare \\ line
# breaks. Anything else is left as literal (escaped) text.
TEX_TOKEN_RE = re.compile(
    r'\\section\{(?P<section>[^}]*)\}'
    r'|\\subsection\{(?P<subsection>[^}]*)\}'
    r'|\\begin\{equation\}\s*(?:\\label\{(?P<eqlabel>[^}]*)\})?(?P<eqbody>.*?)\\end\{equation\}'
    r'|\\begin\{center\}(?P<centerbody>.*?)\\end\{center\}'
    r'|\$(?P<inline>[^$]*)\$'
    r'|\\ref\{(?P<reflabel>[^}]*)\}'
    r'|\\\\',
    re.DOTALL)

EQUATION_LABEL_RE = re.compile(
    r'\\begin\{equation\}\s*(?:\\label\{(?P<label>[^}]*)\})?', re.DOTALL)


def tex_to_html_source(tex_path):
    """Parse a help/*.tex file into (intermediate, block_html, math_sources):
    `intermediate` is the document as escaped HTML text with block-level
    constructs (headings, equations, centered math) replaced by \uE000B<n>\uE000
    tokens and inline math replaced by \uE000I<n>\uE000 tokens; `block_html`
    maps each block token's id to its HTML (itself possibly containing image
    tokens); `math_sources` is the ordered list of raw LaTeX math strings
    referenced by the image tokens, ready for `render_math_batch`.

    \\ref{} is resolved against every \\label{} in the document regardless of
    processing order, so forward references (to an equation defined later in
    the file) work the same as they would in real LaTeX."""
    with open(tex_path, encoding='utf-8') as f:
        body = f.read()

    labels = {}
    for i, m in enumerate(EQUATION_LABEL_RE.finditer(body), start=1):
        label = m.group('label')
        if label:
            labels[label] = i

    math_sources = []
    block_html = {}
    out = []
    pos = 0
    eq_number = 0
    next_block_id = 0

    for m in TEX_TOKEN_RE.finditer(body):
        out.append(html.escape(body[pos:m.start()]))
        pos = m.end()

        if m.group('section') is not None:
            block_html[next_block_id] = ('heading', 2, html.escape(m.group('section')))
            out.append(f'{MARK}B{next_block_id}{MARK}')
            next_block_id += 1
        elif m.group('subsection') is not None:
            block_html[next_block_id] = ('heading', 3, html.escape(m.group('subsection')))
            out.append(f'{MARK}B{next_block_id}{MARK}')
            next_block_id += 1
        elif m.group('eqbody') is not None:
            eq_number += 1
            idx = len(math_sources)
            math_sources.append(m.group('eqbody').strip())
            block_html[next_block_id] = (
                '<table width="100%" style="margin:10px 0;"><tr>'
                '<td width="12%"></td>'
                f'<td align="center">{MARK}I{idx}{MARK}</td>'
                f'<td width="12%" align="right">&nbsp;&nbsp;({eq_number})</td>'
                '</tr></table>'
            )
            out.append(f'{MARK}B{next_block_id}{MARK}')
            next_block_id += 1
        elif m.group('centerbody') is not None:
            idx = len(math_sources)
            math_sources.append(m.group('centerbody').strip().strip('$'))
            block_html[next_block_id] = f'<p align="center">{MARK}I{idx}{MARK}</p>'
            out.append(f'{MARK}B{next_block_id}{MARK}')
            next_block_id += 1
        elif m.group('inline') is not None:
            idx = len(math_sources)
            math_sources.append(m.group('inline'))
            out.append(f'{MARK}I{idx}{MARK}')
        elif m.group('reflabel') is not None:
            number = labels.get(m.group('reflabel'))
            out.append(f'({number})' if number else '(?)')
        # else: bare \\ line break -- dropped; blank-line paragraph
        # splitting in `assemble_html` already provides the visual break.

    out.append(html.escape(body[pos:]))
    return ''.join(out), block_html, math_sources


def pixmap_to_img_tag(pixmap, device_pixel_ratio, inline):
    """HTML <img> tag embedding `pixmap` as a base64 PNG data URI.

    `pixmap` is assumed to already be rasterized at exactly
    `target_math_dpi(zoom, device_pixel_ratio)` (see that function) for the
    zoom level currently in effect, i.e. its pixel count is
    `device_pixel_ratio` times the size it should actually occupy on
    screen -- so the CSS width/height set here is simply the raster size
    divided back down by `device_pixel_ratio`. A `data:` URI has no way to
    carry Qt's own devicePixelRatio metadata (unlike a QPixmap handed
    straight to a QLabel), so this manual division is what stands in for
    it; get it wrong (e.g. by using a fixed assumed density instead of the
    screen's real one) and Qt has to resample the image to fit the CSS box,
    which is what caused the soft/blurry equations before this."""
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, 'PNG')
    b64 = base64.b64encode(bytes(buf.data())).decode('ascii')
    width = pixmap.width() / device_pixel_ratio
    height = pixmap.height() / device_pixel_ratio
    valign = ' style="vertical-align:-15%;"' if inline else ''
    return (f'<img src="data:image/png;base64,{b64}" '
            f'width="{width:.1f}" height="{height:.1f}"{valign}>')


def target_math_dpi(zoom, device_pixel_ratio, dpi_floor=72):
    """The pdflatex/pdftocairo render DPI that makes a formula's raster
    pixel count exactly match the physical screen pixels it will actually
    occupy at the given zoom level -- i.e. no resampling needed in either
    direction, at any zoom or screen density, without just picking an
    ever-higher fixed DPI and hoping it's enough.

    Derivation: a formula's intrinsic size is fixed in TeX points; at
    zoom=1 it should appear on screen at the standard 96px/72pt scaling, so
    its target CSS size is pt_size * 96/72 * zoom, and the physical pixels
    that covers is that times device_pixel_ratio. Since pdftocairo's -r
    option rasterizes directly from pt size (raster_px = pt_size *
    dpi/72), setting dpi = 96 * zoom * device_pixel_ratio makes
    raster_px equal that same physical-pixel target exactly."""
    return max(dpi_floor, round(96 * zoom * device_pixel_ratio))


def render_block(block, zoom, pixmaps, device_pixel_ratio):
    """Render one block_html entry to HTML at the given zoom level. Headings
    use an inline style computed here (rather than the external stylesheet
    used for p/td) because Qt's rich-text engine gives <h1>-<h6> a baked-in
    default size that a stylesheet rule for the tag doesn't reliably
    override -- an inline style attribute has the highest CSS specificity
    there is, so it always wins."""
    if isinstance(block, tuple) and block[0] == 'heading':
        _, level, title = block
        px = round((20 if level == 2 else 17) * zoom)
        margin_top = round((20 if level == 2 else 16) * zoom)
        return (f'<p style="font-family:sans-serif; font-size:{px}px; '
                f'font-weight:bold; margin-top:{margin_top}px; margin-bottom:{round(6 * zoom)}px;">'
                f'{title}</p>')
    return IMG_TOKEN_RE.sub(
        lambda m: pixmap_to_img_tag(pixmaps[int(m.group(1))], device_pixel_ratio, inline=False), block)


def assemble_html(intermediate, block_html, pixmaps, zoom, device_pixel_ratio):
    """Expand an `tex_to_html_source` result into final HTML at the given
    zoom level: block tokens become their standalone element (heading,
    numbered-equation table, ...), remaining prose is split into <p>
    paragraphs on blank lines, and every image token (inline or inside a
    block) is replaced with an actual sized <img> tag. `pixmaps` must
    already be rendered at a raster density matching `device_pixel_ratio`
    for the given zoom (see `TexViewerDialog.apply_zoom`) -- this function
    only lays them out, it doesn't re-rasterize anything."""
    def render_images(s, inline):
        return IMG_TOKEN_RE.sub(
            lambda m: pixmap_to_img_tag(pixmaps[int(m.group(1))], device_pixel_ratio, inline), s)

    # BLOCK_TOKEN_RE has one capturing group, so split() alternates plain
    # text with captured block ids: [text, id, text, id, ..., text].
    html_parts = []
    for i, part in enumerate(BLOCK_TOKEN_RE.split(intermediate)):
        if i % 2 == 1:
            html_parts.append(render_block(block_html[int(part)], zoom, pixmaps, device_pixel_ratio))
            continue
        for para in re.split(r'\n\s*\n', part):
            para = render_images(para, inline=True)
            para = re.sub(r'\s+', ' ', para).strip()
            if para:
                html_parts.append(f'<p>{para}</p>')
    return ''.join(html_parts)


class TexViewerDialog(QDialog):
    """Non-modal window that renders a help/*.tex file as real, selectable
    rich text (via `tex_to_html_source`/`assemble_html`) in a QTextBrowser,
    with math typeset by a real LaTeX toolchain (`render_math_batch`) and
    embedded as images -- used by Help menu entries pointing at a
    help/*.tex writeup.

    QTextBrowser gives native mouse-drag text selection and Ctrl+C copy for
    free; zoom (buttons, Ctrl+=/Ctrl+-/Ctrl+0) scales the base font size and
    re-renders the embedded equation images at a matching size, so text and
    math stay proportional to each other as the zoom level changes -- and,
    since that re-render targets the real screen's devicePixelRatio (see
    `target_math_dpi`), stay crisp at any zoom level and on any display,
    rather than just being a fixed-resolution image stretched to fit."""

    BASE_FONT_PX = 15
    DEFAULT_ZOOM = 1.3
    ZOOM_STEP = 1.15
    ZOOM_MIN = 0.5
    ZOOM_MAX = 3.0

    def __init__(self, title, tex_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 900)
        self.zoom = self.DEFAULT_ZOOM
        self.render_error = None
        self.math_sources = []
        self.pixmaps = []
        self.eq_dpi = None  # dpi the currently-cached self.pixmaps were rendered at

        screen = self.screen() if hasattr(self, 'screen') else None
        self.device_pixel_ratio = screen.devicePixelRatio() if screen else QApplication.primaryScreen().devicePixelRatio()

        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.addStretch()
        zoom_out_btn = QPushButton('−')
        zoom_out_btn.setFixedWidth(32)
        zoom_out_btn.clicked.connect(lambda: self.set_zoom(self.zoom / self.ZOOM_STEP))
        toolbar.addWidget(zoom_out_btn)
        self.zoom_label = QLabel(f'{round(self.zoom * 100)}%')
        self.zoom_label.setFixedWidth(48)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.zoom_label)
        zoom_in_btn = QPushButton('+')
        zoom_in_btn.setFixedWidth(32)
        zoom_in_btn.clicked.connect(lambda: self.set_zoom(self.zoom * self.ZOOM_STEP))
        toolbar.addWidget(zoom_in_btn)
        reset_btn = QPushButton('Reset')
        reset_btn.clicked.connect(lambda: self.set_zoom(self.DEFAULT_ZOOM))
        toolbar.addWidget(reset_btn)
        layout.addLayout(toolbar)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout.addWidget(self.browser)

        try:
            self.intermediate, self.block_html, self.math_sources = tex_to_html_source(tex_path)
        except Exception as e:
            self.render_error = f'Could not render {os.path.basename(tex_path)}:\n\n{e}'

        self.apply_zoom()

        for keys, factor in ((QKeySequence.ZoomIn, self.ZOOM_STEP),
                             (QKeySequence.ZoomOut, 1 / self.ZOOM_STEP)):
            QShortcut(keys, self, activated=functools.partial(self.step_zoom, factor))
        QShortcut(QKeySequence('Ctrl+0'), self, activated=lambda: self.set_zoom(self.DEFAULT_ZOOM))

    def step_zoom(self, factor):
        self.set_zoom(self.zoom * factor)

    def set_zoom(self, zoom):
        self.zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, zoom))
        self.zoom_label.setText(f'{round(self.zoom * 100)}%')
        self.apply_zoom()

    def apply_zoom(self):
        scrollbar = self.browser.verticalScrollBar()
        fraction = scrollbar.value() / scrollbar.maximum() if scrollbar.maximum() else 0.0

        if self.render_error is not None:
            self.browser.setHtml(f'<pre>{html.escape(self.render_error)}</pre>')
            return

        # Re-render the formulas whenever the zoom level changes the target
        # dpi -- `render_math_batch` caches by (source, dpi), so toggling
        # back to a previously-seen zoom level is still free.
        eq_dpi = target_math_dpi(self.zoom, self.device_pixel_ratio)
        if eq_dpi != self.eq_dpi:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.pixmaps = render_math_batch(self.math_sources, dpi=eq_dpi)
                self.eq_dpi = eq_dpi
            except Exception as e:
                self.render_error = f'Could not render formulas:\n\n{e}'
                self.browser.setHtml(f'<pre>{html.escape(self.render_error)}</pre>')
                return
            finally:
                QApplication.restoreOverrideCursor()

        # Qt's rich-text CSS engine doesn't reliably cascade a `body { ... }`
        # font-size down to descendant tags -- p/td need their own explicit
        # rule or they silently keep whatever size they last had (which is
        # what made the body text look stuck). Headings are sized via an
        # inline style instead (see `render_block`), since Qt's baked-in
        # <h1>-<h6> default sizing wins over a stylesheet rule for the tag.
        body_px = round(self.BASE_FONT_PX * self.zoom)
        css = f'p, td {{ font-family: sans-serif; font-size: {body_px}px; margin: 8px 0; text-align: justify; }}'
        self.browser.document().setDefaultStyleSheet(css)
        self.browser.document().setDefaultFont(self.browser.font())
        body_html = assemble_html(self.intermediate, self.block_html, self.pixmaps,
                                    self.zoom, self.device_pixel_ratio)
        self.browser.setHtml(body_html)

        scrollbar = self.browser.verticalScrollBar()
        scrollbar.setValue(round(fraction * scrollbar.maximum()))
