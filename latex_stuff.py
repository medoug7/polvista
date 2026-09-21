"""LaTeX/math rendering for polvista: mathtext-rendered parameter labels and
model equations (`latex_pixmap`, `fit_equation_pixmap`), and the
help/*.tex viewer (`TexViewerDialog`, backed by a real pdflatex toolchain).

Split out of app.py to keep the GUI module focused on the app itself.

Implementation notes (Qt quirks affecting several functions below, noted
once here instead of per-function):
  - A `data:` image URI carries no DPI/devicePixelRatio metadata, unlike a
    QPixmap handed straight to a QLabel -- `pixmap_to_img_tag` divides the
    CSS width/height back down by the ratio itself so Qt doesn't have to
    resample (and blur) the image. `target_math_dpi` picks the render DPI
    that makes this come out pixel-exact at the current zoom/screen.
  - Qt's rich-text engine doesn't reliably cascade a `body { font-size }`
    rule down to `<p>`/`<td>`, and gives `<h1>`-`<h6>` a baked-in default
    size that a stylesheet rule for the tag can't override -- see
    `TexViewerDialog.apply_zoom`'s CSS and `render_block`'s inline heading
    style, which work around each respectively.
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
import collections

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
    """Rasterize a mathtext string (e.g. r'$\\phi_1$') to a QPixmap via
    matplotlib's mathtext renderer, so slider/equation labels don't need a
    real LaTeX install. `facecolor=None` gives a transparent background
    (slider labels); pass e.g. 'white' for an opaque card. `multialignment`
    controls how a multi-line `latex` string's shorter lines align relative
    to the widest one. `dpi` lets a caller rendering for a HiDPI screen ask
    for more raster pixels than LATEX_DPI/SCREEN_DPI alone would supply.

    Memoized: mathtext parsing is slow (tens of ms) and label strings are
    static for the app's lifetime; QPixmap is copy-on-write so a cached one
    is safe to hand to multiple QLabels."""
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
    # This ratio is what makes the extra raster pixels at a higher `dpi`
    # actually legible, rather than just enlarging the on-screen image.
    pixmap.setDevicePixelRatio(dpi / SCREEN_DPI)
    return pixmap


def fit_equation_pixmap(latex, max_width_px, max_height_px, fontsize_max=24, fontsize_min=16,
                          multialignment='center'):
    """Render `latex` as large as possible (capped at fontsize_max, floored
    at fontsize_min) while fitting within max_width_px x max_height_px, so
    the equation card never needs a scrollbar. Not memoized itself (the
    box size changes on every resize), but built on the memoized
    `latex_pixmap`, so re-fitting to a previously-seen fontsize is cheap."""
    pixmap = latex_pixmap(latex, fontsize=fontsize_max, facecolor='white', pad_inches=0.05,
                            multialignment=multialignment)
    width = pixmap.width() / pixmap.devicePixelRatio()
    height = pixmap.height() / pixmap.devicePixelRatio()
    scale = min(max_width_px / width, max_height_px / height, 1.0)
    if scale >= 1.0:
        return pixmap
    # Mathtext size scales ~linearly with fontsize, so extrapolate the
    # fitting size from one render and do a single corrective render,
    # rather than shrinking one point at a time (up to 12 slow re-renders).
    target = max(fontsize_min, int(fontsize_max * scale))
    if target >= fontsize_max:
        return pixmap
    return latex_pixmap(latex, fontsize=target, facecolor='white', pad_inches=0.05,
                          multialignment=multialignment)


# Bounded like latex_pixmap's own lru_cache(maxsize=128) -- without a cap
# this would grow for the app's whole lifetime, since every zoom change
# re-keys every visible formula by its new dpi.
MATH_SNIPPET_CACHE_MAXSIZE = 512
math_snippet_cache = collections.OrderedDict()

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
    """Render raw LaTeX math strings (no surrounding $) to transparent
    QPixmaps in one pdflatex + pdftocairo pass (each wrapped in a
    `preview`/`tightpage` environment, so pdftocairo emits one tightly-
    cropped page per formula). Results already seen at this (source, dpi)
    are served from `math_snippet_cache` and skipped in the batch."""
    results = [None] * len(math_sources)
    to_render = []
    for i, src in enumerate(math_sources):
        key = (src, dpi)
        cached = math_snippet_cache.get(key)
        if cached is not None:
            math_snippet_cache.move_to_end(key)
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
            key = (src, dpi)
            math_snippet_cache[key] = pixmap
            math_snippet_cache.move_to_end(key)
            if len(math_snippet_cache) > MATH_SNIPPET_CACHE_MAXSIZE:
                math_snippet_cache.popitem(last=False)
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
    `intermediate` is escaped HTML text with block-level constructs
    (headings, equations, centered math) replaced by \uE000B<n>\uE000 tokens
    and inline math by \uE000I<n>\uE000 tokens; `block_html` maps each block
    token's id to its HTML; `math_sources` is the ordered list of raw LaTeX
    strings the image tokens reference, ready for `render_math_batch`.
    `\\ref{}` resolves against every `\\label{}` regardless of order, so
    forward references work as in real LaTeX."""
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
    """HTML <img> tag embedding `pixmap` as a base64 PNG data URI, sized
    down from its raster pixels by `device_pixel_ratio` (see this module's
    top-of-file implementation notes on why -- `pixmap` must already be
    rasterized at `target_math_dpi(zoom, device_pixel_ratio)`)."""
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
    """Render DPI that makes a formula's raster pixel count exactly match
    the physical screen pixels it will occupy at this zoom/density, so
    neither direction needs resampling."""
    # pt_size * 96/72 * zoom is the target CSS size at this zoom (96px/72pt
    # is the standard scaling); * device_pixel_ratio converts that to
    # physical pixels. pdftocairo's -r rasterizes as pt_size * dpi/72, so
    # this dpi makes its raster_px land exactly on that physical-pixel target.
    return max(dpi_floor, round(96 * zoom * device_pixel_ratio))


def render_block(block, zoom, pixmaps, device_pixel_ratio):
    """Render one block_html entry to HTML at the given zoom level (see
    this module's top-of-file notes on why headings get an inline style
    here rather than the external p/td stylesheet)."""
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
    """Expand a `tex_to_html_source` result into final HTML at the given
    zoom level: block tokens become their standalone element, remaining
    prose splits into `<p>` paragraphs on blank lines, and every image
    token is replaced with a sized `<img>` tag. `pixmaps` must already be
    rendered at a raster density matching `device_pixel_ratio` for this
    zoom (see `TexViewerDialog.apply_zoom`); this only lays them out."""
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
    """Non-modal window rendering a help/*.tex file as selectable rich text
    in a QTextBrowser (via `tex_to_html_source`/`assemble_html`), with math
    typeset by a real LaTeX toolchain and embedded as images. Zoom (buttons,
    Ctrl+=/Ctrl+-/Ctrl+0) scales the base font and re-renders the equation
    images to match, staying crisp at any zoom/screen density (see
    `target_math_dpi`)."""

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

        self._load_source(tex_path)
        self.apply_zoom()

        for keys, factor in ((QKeySequence.ZoomIn, self.ZOOM_STEP),
                             (QKeySequence.ZoomOut, 1 / self.ZOOM_STEP)):
            QShortcut(keys, self, activated=functools.partial(self.step_zoom, factor))
        QShortcut(QKeySequence('Ctrl+0'), self, activated=lambda: self.set_zoom(self.DEFAULT_ZOOM))

    def _load_source(self, tex_path):
        try:
            self.intermediate, self.block_html, self.math_sources = tex_to_html_source(tex_path)
        except Exception as e:
            self.render_error = f'Could not render {os.path.basename(tex_path)}:\n\n{e}'

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

        # Re-render only if zoom changed the target dpi; render_math_batch's
        # own (source, dpi) cache makes revisiting a zoom level free.
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

        # p/td need their own explicit font-size rule (see module notes);
        # headings are sized separately via render_block's inline style.
        body_px = round(self.BASE_FONT_PX * self.zoom)
        css = f'p, td {{ font-family: sans-serif; font-size: {body_px}px; margin: 8px 0; text-align: justify; }}'
        self.browser.document().setDefaultStyleSheet(css)
        self.browser.document().setDefaultFont(self.browser.font())
        body_html = assemble_html(self.intermediate, self.block_html, self.pixmaps,
                                    self.zoom, self.device_pixel_ratio)
        self.browser.setHtml(body_html)

        scrollbar = self.browser.verticalScrollBar()
        scrollbar.setValue(round(fraction * scrollbar.maximum()))
