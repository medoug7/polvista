"""CustomModelDialog -- the "Build Custom Model" window (see app.py's Models
menu action / "Custom model..." dropdown entry). Lets a user type j_lambda(z)
(the emissivity) and phi'(z) (the Faraday-depth *density* -- see models.
build_custom_model's own module comment for why this isn't phi(z) itself)
expressions for the general Sokoloff et al. 1998 (eq. 1) / Burn 1966
line-of-sight integral (see models.build_custom_model), discover the new
constants those expressions introduce, pick each one's Kind (which
auto-sets its bounds to that kind's own usual range -- see KIND_DEFS) and a
preview value to evaluate it at, preview j_lambda(z)/phi'(z)/phi(z) over the
line of sight, and register the result as a new selectable model.
j_lambda(z)'s own frequency dependence beyond whatever it references directly
(the 'nu'/'lambda' symbols, see the intro text below) isn't set here --
that's the main window's existing Spectrum box (Power-law/SSA/Thermal/
Log-parabola + alpha), applied on top exactly as it is for every other
single-component model, once the custom model is selected there."""
import functools
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QDialogButtonBox, QMessageBox, QHeaderView, QSizePolicy, QTextBrowser)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from polvista.models import (
    C, build_custom_model, custom_model_equation_lines, discover_custom_params,
    parse_custom_expr, preview_los_profiles, CustomModelError)
from polvista.latex_stuff import LATEX_DPI, fit_equation_pixmap, latex_pixmap, pixmap_to_img_tag

DEFAULT_PREVIEW_LAMBDA_MM = 3.0  # fallback when parent has no wl_min/wl_max to derive one from

# COL_* -- the constants table's own column layout, shared by every method
# below that reads/writes a row.
COL_NAME, COL_KIND, COL_LO, COL_HI, COL_PREVIEW = range(5)

# Kind dropdown label -> (Param.kind, display<->physical conversion,
# default (lo, hi) and preview value *in display units*). 'Fraction'/'Angle'
# are shown/typed in the same user-facing units app.ParamSlider itself
# displays for 'p'/'X' (percent, degrees) even though the physical value
# threaded through to the model (and stored in spec.bounds) is a 0-1
# fraction / radians -- 'Depth'/'Dispersion'/'Number' have no such split,
# display units are the physical ones. Preview defaults are deliberately
# away from 0 where a typical expression would divide by the constant (a
# width, say) -- the bound-*midpoint* this dialog used to preview at was 0
# by construction for any kind symmetric about 0, which is exactly the
# singularity a width-like constant hits.
KIND_DEFS = {
    'Fraction': dict(model_kind='p', to_phys=lambda d: d / 100.0, to_disp=lambda p: p * 100.0,
                      bounds_disp=(0.0, 100.0), preview_disp=10.0),
    'Angle': dict(model_kind='X', to_phys=lambda d: d * np.pi / 180.0, to_disp=lambda p: p * 180.0 / np.pi,
                   bounds_disp=(-90.0, 90.0), preview_disp=0.0),
    'Depth': dict(model_kind='phi', to_phys=lambda d: d, to_disp=lambda p: p,
                   bounds_disp=(-5.0e6, 5.0e6), preview_disp=300.0),
    # Same log-scale slider app.ParamSlider gives a built-in model's own
    # dphi (sigma_phi -- Faraday depth *dispersion* across a turbulent/
    # blended screen, see models.py's burn/tribble/partial): always >= 0,
    # unlike 'Depth', so bounds_disp has no negative side to mirror.
    'Dispersion': dict(model_kind='dphi', to_phys=lambda d: d, to_disp=lambda p: p,
                        bounds_disp=(0.0, 5.0e6), preview_disp=300.0),
    'Number': dict(model_kind='scale', to_phys=lambda d: d, to_disp=lambda p: p,
                    bounds_disp=(-10.0, 10.0), preview_disp=1.0),
}
MODEL_KIND_TO_LABEL = {d['model_kind']: label for label, d in KIND_DEFS.items()}
DEFAULT_KIND_LABEL = 'Number'


def _intro_math(latex, dpi, fontsize=13):
    """<img> tag rendering `latex` via the same mathtext pipeline
    ParamSlider uses for its own parameter-name labels (`latex_stuff.
    latex_pixmap`, rendered here with a transparent background and wrapped
    for inline use by `latex_stuff.pixmap_to_img_tag` -- the same two
    functions TexViewerDialog composes for its own inline math, just at a
    fixed size since this QLabel never zooms).

    `dpi` -- see `_intro_html` -- is threaded through to `latex_pixmap` so
    the raster actually has enough pixels for the screen it's shown on;
    `pixmap_to_img_tag` is what turns that oversampling into on-screen
    sharpness rather than just a bigger image (it sizes the <img>'s CSS
    box down by the same devicePixelRatio latex_pixmap attached, so more
    dpi means more pixels *within* the same displayed size, not a bigger
    image)."""
    pixmap = latex_pixmap(latex, fontsize=fontsize, facecolor=None, dpi=dpi)
    return pixmap_to_img_tag(pixmap, pixmap.devicePixelRatio(), inline=True)


def _intro_html(dpi):
    """Rich-text HTML for the dialog's intro QTextBrowser (see
    CustomModelDialog.__init__) -- same wording as the plain-text version
    this replaced, but every equation/variable name is a real
    mathtext-rendered image instead of an approximating unicode character,
    matching how ParamSlider labels the model's own parameters elsewhere in
    the app.

    `dpi` (see CustomModelDialog.__init__, which derives it from the
    dialog's actual screen) is the same guarantee-crisp-on-any-display
    trick TexViewerDialog uses for the Help-menu tutorials (its own
    `target_math_dpi`) -- scale the render resolution to the real screen's
    devicePixelRatio instead of a single fixed DPI that's only right for
    one density of display."""
    # Pre-render every math run into a plain variable -- a raw string
    # containing a backslash can't appear inside an f-string's {...}
    # expression part (SyntaxError on Python < 3.12), so these can't just
    # be inlined as m(r'...') calls in the f-strings below.
    m = functools.partial(_intro_math, dpi=dpi)
    equation = m(r'$P(\lambda)=p_0\,e^{\,2i\chi_0}\int_0^1 j_\lambda(z)\,'
                 r'e^{\,2i\,\phi(z)\,\lambda^2}\,dz$', fontsize=14)
    depth_formula = m(r"$\phi(z)=\phi_0 \int_{\max(z,\,\mathrm{lower})}^{\mathrm{upper}}\phi'(z')\,dz'$",
                       fontsize=14)
    z_range, z_source, z_observer = m(r'$z\in[0,1]$'), m(r'$z=0$'), m(r'$z=1$')
    j_lambda, z = m(r'$j_\lambda(z)$'), m(r'$z$')
    phi_prime = m(r"$\phi'(z)$")
    nu, lam = m(r'$\nu$'), m(r'$\lambda$')
    p0, phi0, w, chi0 = m(r'$p_0$'), m(r'$\phi_0$'), m(r'$w$'), m(r'$\chi_0$')
    i_unit = m(r'$i$')
    evpa_example = m(r'$e^{2i\chi(z)}$')
    return (
        '<p>Define a model from the general Faraday-rotation line-of-sight integral:</p>'
        f'<p align="center">{equation}</p>'
        f'<p>over a normalized line of sight {z_range} running the way light travels: '
        f'{z_source} at the source, {z_observer} at the observer. {p0} and {chi0} sit outside '
        f'both integrals entirely, as their own always-present sliders -- never typed into either '
        f'field below -- so a typical simple screen needs neither field to mention them at all.</p>'
        f'<p>{j_lambda} is the emissivity profile (zero outside its own lower/upper bound, set '
        f'below) -- real for a plain intensity profile, or complex (e.g. {evpa_example}) to give '
        f'the source a position-dependent intrinsic EVPA.</p>'
        f'<p>{phi_prime} is an unscaled Faraday-depth <i>density</i> shape (zero outside its own '
        f'lower/upper bound, set below), not the depth itself -- the actual Faraday depth is its '
        f'accumulated integral, scaled by {phi0}\'s own always-present slider, {depth_formula} -- '
        f'so a typical screen can leave {phi_prime} as a bare shape (e.g. \'1\' or \'z\') and set '
        f'the actual rad/m² magnitude with {phi0} instead of typing it into the field.</p>'
        f'<p>Where the two ranges overlap, different emission depths pick up '
        f'different amounts of rotation before reaching the observer -- internal, differential '
        f'Faraday rotation.</p>'
        f'<p>Where {phi_prime}\'s own range lies entirely beyond {j_lambda}\'s, '
        f'every emission point instead sees the exact same total rotation -- a pure external '
        f'screen.</p>'
        f'<p>Use {z}, {nu} (frequency, MHz), {lam} (wavelength, m), {i_unit} (the imaginary '
        f'unit), and any new constant name (e.g. {w}) with explicit \'*\' for products -- those '
        f'do need a Kind and bounds picked below. Leaving either field above blank means the '
        f'constant 1.</p>'
    )


def _math_field_label(latex, dpi, fontsize=13):
    """QLabel showing `latex` as a rendered mathtext image -- same
    latex_pixmap pipeline (and the same per-screen `dpi`, see
    CustomModelDialog.__init__) as the intro text's inline math and
    ParamSlider's own parameter-name labels elsewhere in the app -- used
    for the j_lambda(z)/phi'(z) row labels below instead of plain
    approximating text."""
    label = QLabel()
    label.setPixmap(latex_pixmap(latex, fontsize=fontsize, facecolor=None, dpi=dpi))
    return label


def _bound_edit(default):
    """A plain QLineEdit (not a QDoubleSpinBox -- no up/down buttons) for
    one z1/z2-style integration-bound value, restricted to [0,1] at 3
    decimal places by its QDoubleValidator. `default` is both the initial
    text and (via _bound_value) the fallback used while the box is
    transiently empty/invalid mid-edit."""
    edit = QLineEdit(f'{default:.3f}')
    edit.setValidator(QDoubleValidator(0.0, 1.0, 3, edit))
    edit.setFixedWidth(60)
    edit.setAlignment(Qt.AlignRight)
    return edit


class CustomModelDialog(QDialog):
    """On accept (Define/Update Model), self.model_func holds the newly
    built and registered model function (see models.build_custom_model);
    None if the dialog was cancelled. `existing_def`, if given, pre-fills
    every field and the preview instead of starting blank -- either
    re-opening a custom model previously loaded from a saved
    'custom_definition' JSON block (see app.py's load_model_action), or
    editing one already selected in the current session (see
    app.py's edit_custom_model_menu_action).

    `edit_func`, if given (the currently-registered function this dialog is
    editing -- app.py's Models menu "Edit Custom Model..." entry, not the
    plain "Custom model..."/"Build Custom Model..." paths, which always
    leave it None), makes on_define() pass edit_func.__name__ through to
    build_custom_model's own `name` so the edited model *replaces* the
    original registration in place (same MODELS_BY_NAME key) instead of
    minting a new one alongside it -- app.py's caller still has to swap the
    combo row's own itemData/text over to the freshly-returned function
    object itself, since build_custom_model always returns a brand new
    closure even when reusing the same name."""

    def __init__(self, parent=None, existing_def=None, edit_func=None):
        super().__init__(parent)
        self._edit_func = edit_func
        self.setWindowTitle('Edit Custom Model' if edit_func is not None else 'Build Custom Model')
        # Just a width floor here -- the real minimum *and* initial shown
        # size are set from the layout's own minimumSize()/sizeHint() at
        # the very end of __init__ instead of a hardcoded guess, once every
        # widget actually exists. A hardcoded guess is exactly what
        # previously let the preview canvas's own hard 320x320 floor (see
        # self.canvas below) overflow into the Define/Cancel button row
        # instead of the window sizing itself (and refusing to shrink below
        # what's needed) to fit it.
        self.setMinimumWidth(980)
        self.model_func = None

        layout = QVBoxLayout(self)

        # Render the intro's math images at a DPI matched to this dialog's
        # actual screen -- same guarantee-crisp-on-any-display approach
        # TexViewerDialog uses for the Help-menu tutorials (its own
        # target_math_dpi). self.screen() only reflects the real screen once
        # the widget has a window handle, which it does not yet at this
        # point in a fresh QDialog's __init__ -- same fallback
        # TexViewerDialog uses for the same reason.
        screen = self.screen() if hasattr(self, 'screen') else None
        device_pixel_ratio = screen.devicePixelRatio() if screen else QApplication.primaryScreen().devicePixelRatio()
        intro_dpi = round(LATEX_DPI * device_pixel_ratio)

        # QTextBrowser rather than a word-wrapped QLabel -- same rich-text/
        # inline-math-image rendering (see TexViewerDialog in latex_stuff.py,
        # the Help-menu tutorial viewer this now matches), but scrollable
        # internally instead of forcing the whole dialog taller to fit every
        # paragraph at once. Fixed height keeps that box a small, constant
        # slice of the dialog regardless of content length; the border
        # matches the app's other framed HTML panels (see eq_scroll in
        # app.py).
        intro = QTextBrowser()
        intro.setOpenExternalLinks(False)
        intro.setStyleSheet(
            "QTextBrowser { font-size: 12pt; border: 1px solid #999; }")
        intro.setFixedHeight(160)
        intro.setHtml(_intro_html(intro_dpi))
        layout.addWidget(intro)

        form = QFormLayout()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText('e.g. My screen')

        emiss_bound_tip = ('j_λ(z) is forced to 0 outside [lower, upper] -- the emitting region '
                            'is confined to that range.')
        self.emiss_edit = QLineEdit()
        self.emiss_edit.setPlaceholderText('e.g. exp(-((z-0.5)/w)**2)')
        self.j_lo_edit = _bound_edit(0.0)
        self.j_hi_edit = _bound_edit(1.0)
        for edit in (self.j_lo_edit, self.j_hi_edit):
            edit.setToolTip(emiss_bound_tip)
            edit.editingFinished.connect(self.refresh_preview)
        emiss_row = QHBoxLayout()
        emiss_row.addWidget(self.emiss_edit, stretch=1)
        emiss_row.addWidget(QLabel('for'))
        emiss_row.addWidget(self.j_lo_edit)
        emiss_row.addWidget(QLabel('≤ z ≤'))
        emiss_row.addWidget(self.j_hi_edit)

        phi_bound_tip = ("φ'(z) is forced to 0 outside [lower, upper], so the accumulated "
                          'Faraday depth φ(z) only picks up rotation from within that range.')
        self.phi_edit = QLineEdit()
        self.phi_edit.setPlaceholderText('e.g. w*z')
        self.p_lo_edit = _bound_edit(0.0)
        self.p_hi_edit = _bound_edit(1.0)
        for edit in (self.p_lo_edit, self.p_hi_edit):
            edit.setToolTip(phi_bound_tip)
            edit.editingFinished.connect(self.refresh_preview)
        phi_row = QHBoxLayout()
        phi_row.addWidget(self.phi_edit, stretch=1)
        phi_row.addWidget(QLabel('for'))
        phi_row.addWidget(self.p_lo_edit)
        phi_row.addWidget(QLabel('≤ z ≤'))
        phi_row.addWidget(self.p_hi_edit)

        form.addRow('Model name:', self.label_edit)
        form.addRow(_math_field_label(r'$j_\lambda(z)=$', intro_dpi), emiss_row)
        form.addRow(_math_field_label(r"$\phi'(z)=$", intro_dpi), phi_row)
        layout.addLayout(form)

        parse_row = QHBoxLayout()
        self.parse_button = QPushButton('Parse')
        self.parse_button.setToolTip(
            "Discover the constants used in j_λ(z)/φ'(z), (re)build the table below "
            '(keeping any existing row\'s Kind/bounds/preview value for a name that survives), '
            'and refresh the preview plots.')
        self.parse_button.clicked.connect(self.on_parse)
        parse_row.addWidget(self.parse_button)
        parse_row.addStretch(1)
        layout.addLayout(parse_row)

        # Live rendered-LaTeX preview of the two equations this model would
        # get (see models.custom_model_equation_lines) -- phi(z) and
        # P(lambda) side by side rather than one stacked image, each fit to
        # its own half of the row (see refit_dialog_equation) with no
        # scrolling (unlike app.py's own single-equation main-window card,
        # which shows one already-combined multi-line image via
        # fit_equation_pixmap -- there's nothing to lay out side by side
        # there since it's just one model at a time, already selected).
        # QSizePolicy.Ignored on the horizontal axis is what lets each
        # label shrink below its currently-shown pixmap's own width instead
        # of that pixmap's size forcing the row wider than the dialog --
        # refreshed alongside the eps/phi plots below (see refresh_preview)
        # and re-fit to each panel's own size on every dialog resize (see
        # resizeEvent).
        eq_row = QHBoxLayout()
        self.phi_eq_label = QLabel()
        self.p_eq_label = QLabel()
        for eq_label in (self.phi_eq_label, self.p_eq_label):
            eq_label.setAlignment(Qt.AlignCenter)
            eq_label.setFixedHeight(80)
            eq_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            eq_label.setStyleSheet('background-color: white; padding: 6px; border: 1px solid #999;')
            eq_row.addWidget(eq_label, stretch=1)
        layout.addLayout(eq_row)

        content_row = QHBoxLayout()

        table_col = QVBoxLayout()
        table_col.addWidget(QLabel('Discovered constants:'))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['Constant', 'Kind', 'Lower bound', 'Upper bound', 'Preview value'])
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(COL_KIND, QHeaderView.ResizeToContents)
        self.table.setToolTip(
            'Kind auto-fills Lower/Upper bound and Preview value with that kind\'s usual range -- '
            'all three stay editable afterward. Preview value (only, not the bounds) is what the '
            "j_λ(z)/φ'(z)/φ(z) plots to the right are evaluated at.")
        self.table.itemChanged.connect(self.on_item_changed)
        table_col.addWidget(self.table)
        content_row.addLayout(table_col, stretch=1)

        preview_col = QVBoxLayout()
        preview_col.addWidget(QLabel('j_λ(z), φ\'(z)/φ(z) preview (at each constant’s Preview value):'))
        preview_lambda_row = QHBoxLayout()
        preview_lambda_row.addWidget(QLabel('Preview λ [mm]:'))
        self.preview_lambda_spin = QDoubleSpinBox()
        self.preview_lambda_spin.setRange(1e-4, 1.0e5)
        self.preview_lambda_spin.setDecimals(4)
        self.preview_lambda_spin.setValue(self._default_preview_lambda_mm(parent))
        self.preview_lambda_spin.setToolTip(
            "Wavelength j_λ(z)/phi'(z) are evaluated at for this preview, if either references "
            "'nu' or 'lambda' -- irrelevant otherwise.")
        self.preview_lambda_spin.valueChanged.connect(self.refresh_preview)
        preview_lambda_row.addWidget(self.preview_lambda_spin)
        preview_lambda_row.addStretch(1)
        preview_col.addLayout(preview_lambda_row)
        # Height is 80% of width's own proportion (3.2 vs. an original 4.0)
        # -- tall enough for two stacked panels to stay readable, short
        # enough that the panel actually fits above the Define/Cancel row
        # instead of its own x-axis getting clipped against it.
        self.fig = Figure(figsize=(4.2, 3.2))
        self.canvas = FigureCanvas(self.fig)
        # A stretch factor alone only distributes *extra* space -- with
        # table_col competing for the same row and no minimum size of its
        # own, the canvas could otherwise get laid out at only a few
        # pixels tall before the dialog's first real resize settles,
        # collapsing _apply_preview_margins's own pixel margins down to a
        # near-zero plot area. Setting an explicit floor guarantees it
        # always gets real room.
        self.canvas.setMinimumSize(320, 256)
        self.ax_emiss, self.ax_phi = self.fig.subplots(2, 1, sharex=True)
        self._style_preview_axes()
        preview_col.addWidget(self.canvas)
        content_row.addLayout(preview_col, stretch=1)

        layout.addLayout(content_row)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.define_button = QPushButton('Update Model' if edit_func is not None else 'Define Model')
        self.define_button.clicked.connect(self.on_define)
        self.define_button.setEnabled(False)
        self.buttons.addButton(self.define_button, QDialogButtonBox.AcceptRole)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        if existing_def:
            self.label_edit.setText(existing_def.get('label', ''))
            self.emiss_edit.setText(existing_def.get('emiss_expr', ''))
            self.phi_edit.setText(existing_def.get('phi_expr', ''))
            # Parse first (populates the constants table to match j_λ(z)/
            # phi''s own text) *then* set the bound boxes -- QLineEdit's
            # setText() (unlike a spinbox's setValue()) doesn't itself fire
            # editingFinished, so nothing below re-runs the preview until
            # the explicit refresh_preview() at the end of this block; that
            # still needs the table already populated with every constant
            # j_λ(z)/phi'(z) actually reference, or it can't even build the
            # preview's lambdified callables.
            self.on_parse()
            # j_lo/j_hi/p_lo/p_hi replaced the old single-sided z1/z2
            # bounds -- fall back to the closest equivalent (z1 was always
            # emiss's upper bound with an implicit 0 lower bound; z2 was
            # always phi''s lower bound with an implicit 1 upper bound) for
            # a definition saved before that change, rather than silently
            # reverting a truncated model to the new all-purpose (0, 1)
            # defaults.
            self.j_lo_edit.setText(f"{existing_def.get('j_lo', 0.0):.3f}")
            self.j_hi_edit.setText(f"{existing_def.get('j_hi', existing_def.get('z1', 1.0)):.3f}")
            self.p_lo_edit.setText(f"{existing_def.get('p_lo', existing_def.get('z2', 0.0)):.3f}")
            self.p_hi_edit.setText(f"{existing_def.get('p_hi', 1.0):.3f}")
            specs = existing_def.get('param_specs', {})
            for row in range(self.table.rowCount()):
                name = self.table.item(row, COL_NAME).text()
                if name not in specs:
                    continue
                model_kind, lo_phys, hi_phys = specs[name]
                label = MODEL_KIND_TO_LABEL.get(model_kind, DEFAULT_KIND_LABEL)
                combo = self.table.cellWidget(row, COL_KIND)
                idx = combo.findText(label)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
                d = KIND_DEFS[label]
                self.table.blockSignals(True)
                self.table.item(row, COL_LO).setText(str(d['to_disp'](lo_phys)))
                self.table.item(row, COL_HI).setText(str(d['to_disp'](hi_phys)))
                self.table.blockSignals(False)
            self.refresh_preview()

        # Both the real minimum size *and* the initial shown size come from
        # the layout's own computed sizes here, now that every widget
        # actually exists. setMinimumHeight is what stops the user manually
        # shrinking the window back down past the point the preview
        # canvas's own hard 320x320 floor (see self.canvas above) can no
        # longer fit above the Define/Cancel row.
        self.setMinimumHeight(self.layout().minimumSize().height())
        self.resize(self.sizeHint())

    @staticmethod
    def _default_preview_lambda_mm(parent):
        """The geometric mean of the main window's own currently-plotted
        wavelength range [mm] (a physically relevant default if j_λ(z)/
        phi'(z) reference 'nu'/'lambda' at all), or DEFAULT_PREVIEW_LAMBDA_MM
        if `parent` isn't a MainWindow with that range available (e.g. a
        standalone test)."""
        try:
            return float(np.sqrt(parent.wl_min.value() * parent.wl_max.value()))
        except AttributeError:
            return DEFAULT_PREVIEW_LAMBDA_MM

    @staticmethod
    def _bound_value(edit, fallback):
        """float(edit.text()), or `fallback` if the box is transiently
        empty/invalid mid-edit (its QDoubleValidator allows an
        'Intermediate' state, e.g. an empty string, while the user is
        still typing)."""
        try:
            return float(edit.text())
        except ValueError:
            return fallback

    def _bounds(self):
        """(j_lo, j_hi, p_lo, p_hi) parsed from the four bound boxes."""
        return (self._bound_value(self.j_lo_edit, 0.0), self._bound_value(self.j_hi_edit, 1.0),
                self._bound_value(self.p_lo_edit, 0.0), self._bound_value(self.p_hi_edit, 1.0))

    # Fixed PIXEL margins for the stacked emiss/phi preview axes (see
    # apply_fixed_margins in widgets.py for the same left/right/bottom/top
    # -> subplots_adjust fraction technique, used here instead of
    # tight_layout() specifically so hspace can be pinned to exactly 0 --
    # tight_layout() recomputes every spacing itself, including hspace,
    # so anything it's given would just get overridden back to whatever
    # it thinks the tick/label geometry needs).
    PREVIEW_MARGINS_PX = dict(left=65, right=15, bottom=65, top=10)

    def _apply_preview_margins(self):
        """subplots_adjust the two stacked preview axes to PREVIEW_MARGINS_PX
        with hspace=0 -- ax_emiss's own x tick labels are hidden (see
        _style_preview_axes) precisely so the two panels can sit flush
        against each other without redundant sharex labels caught in the
        gap."""
        w = max(self.canvas.width(), 1)
        h = max(self.canvas.height(), 1)
        m = self.PREVIEW_MARGINS_PX
        left_px, right_px = m['left'], m['right']
        bottom_px, top_px = m['bottom'], m['top']
        if left_px + right_px > 0.8 * w:
            scale = 0.8 * w / (left_px + right_px)
            left_px, right_px = left_px * scale, right_px * scale
        if bottom_px + top_px > 0.8 * h:
            scale = 0.8 * h / (bottom_px + top_px)
            bottom_px, top_px = bottom_px * scale, top_px * scale
        self.fig.subplots_adjust(left=left_px / w, right=1 - right_px / w,
                                  bottom=bottom_px / h, top=1 - top_px / h, hspace=0.0)

    def _style_preview_axes(self):
        self.ax_emiss.set_ylabel(r'$j_\lambda(z)$')
        self.ax_phi.set_ylabel(r"$\phi(z),\ \phi'(z)$")
        self.ax_phi.set_xlabel('z  (0 = source  →  1 = observer)', fontsize=14)
        self.ax_emiss.set_xlim(0.0, 1.0)
        self.ax_emiss.set_ylim(0.0,)
        self.ax_phi.set_xlim(0.0, 1.0)
        # sharex=True already keeps the two x-axes in sync -- hide the top
        # panel's own tick labels so hspace=0 (see _apply_preview_margins)
        # doesn't jam them into the gap against ax_phi's plot area.
        self.ax_emiss.tick_params(labelbottom=False)
        self.ax_emiss.grid(True)
        self.ax_phi.grid(True)

    def refit_dialog_equation(self, emiss_expr, phi_expr):
        """Re-render the two live equation previews (see models.
        custom_model_equation_lines) to fit each one's own panel -- called
        from refresh_preview (whenever j_λ(z)/φ'(z)/the bound boxes
        change) and from resizeEvent (whenever the dialog itself is
        resized), matching app.py's own MainWindow.refit_equation."""
        phi_line, p_line = custom_model_equation_lines(emiss_expr, phi_expr, *self._bounds())
        for eq_label, eq in ((self.phi_eq_label, phi_line), (self.p_eq_label, p_line)):
            max_w = max(eq_label.width() - 20, 50)
            max_h = max(eq_label.height() - 20, 20)
            eq_label.setPixmap(fit_equation_pixmap(eq, max_w, max_h))

    def resizeEvent(self, event):
        """Keep the equation preview fit to its panel, and the eps/phi
        preview plots' own margins fit to *their* panel, across a dialog
        resize -- _apply_preview_margins only knows the canvas's size as
        of the moment it's called, so without this it stays stale at
        whatever (likely too-small, pre-layout) size the canvas had at the
        first draw, leaving the plots visibly not filling the space Qt
        actually gave them."""
        super().resizeEvent(event)
        try:
            emiss_expr = parse_custom_expr(self.emiss_edit.text(), 'j_lambda(z)')
            phi_expr = parse_custom_expr(self.phi_edit.text(), "phi'(z)")
        except CustomModelError:
            pass
        else:
            self.refit_dialog_equation(emiss_expr, phi_expr)
        self._apply_preview_margins()
        self.canvas.draw_idle()

    def _row_state(self, row):
        """(kind_label, lo_text, hi_text, preview_text) currently shown in
        `row` -- used by on_parse to carry a surviving name's row forward
        across a re-parse."""
        combo = self.table.cellWidget(row, COL_KIND)
        return (combo.currentText(), self.table.item(row, COL_LO).text(),
                self.table.item(row, COL_HI).text(), self.table.item(row, COL_PREVIEW).text())

    def _default_row_state(self):
        d = KIND_DEFS[DEFAULT_KIND_LABEL]
        lo, hi = d['bounds_disp']
        return DEFAULT_KIND_LABEL, str(lo), str(hi), str(d['preview_disp'])

    def _set_row(self, row, name, kind_label, lo_text, hi_text, preview_text):
        name_item = QTableWidgetItem(name)
        name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(row, COL_NAME, name_item)

        combo = QComboBox()
        combo.addItems(list(KIND_DEFS.keys()))
        idx = combo.findText(kind_label)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.currentIndexChanged.connect(lambda _checked, r=row: self.on_kind_changed(r))
        self.table.setCellWidget(row, COL_KIND, combo)

        self.table.setItem(row, COL_LO, QTableWidgetItem(lo_text))
        self.table.setItem(row, COL_HI, QTableWidgetItem(hi_text))
        self.table.setItem(row, COL_PREVIEW, QTableWidgetItem(preview_text))

    def on_parse(self):
        """Parse both expressions, (re)populate the constants table -- rows
        for names that survive re-parsing keep their existing Kind/bounds/
        preview value, new names start at Kind=Number -- and refresh the
        preview plot. The Define Model button only unlocks once this
        succeeds at least once."""
        try:
            emiss_expr = parse_custom_expr(self.emiss_edit.text(), 'j_lambda(z)')
            phi_expr = parse_custom_expr(self.phi_edit.text(), "phi'(z)")
            names = discover_custom_params(emiss_expr, phi_expr)
        except CustomModelError as e:
            QMessageBox.warning(self, 'Build Custom Model', str(e))
            self.define_button.setEnabled(False)
            return

        old_state = {self.table.item(row, COL_NAME).text(): self._row_state(row)
                     for row in range(self.table.rowCount())}

        self.table.blockSignals(True)
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            kind_label, lo_txt, hi_txt, prev_txt = old_state.get(name, self._default_row_state())
            self._set_row(row, name, kind_label, lo_txt, hi_txt, prev_txt)
        self.table.blockSignals(False)

        self.define_button.setEnabled(True)
        self.refresh_preview()

    def on_kind_changed(self, row):
        """A row's Kind dropdown changed -- auto-fill its Lower/Upper bound
        and Preview value with that kind's own default range (still
        editable afterward), then refresh the preview."""
        combo = self.table.cellWidget(row, COL_KIND)
        d = KIND_DEFS[combo.currentText()]
        lo, hi = d['bounds_disp']
        self.table.blockSignals(True)
        self.table.item(row, COL_LO).setText(str(lo))
        self.table.item(row, COL_HI).setText(str(hi))
        self.table.item(row, COL_PREVIEW).setText(str(d['preview_disp']))
        self.table.blockSignals(False)
        self.refresh_preview()

    def on_item_changed(self, item):
        """Only a Preview value edit needs to redraw the preview -- Lower/
        Upper bound edits only matter once the model is actually defined."""
        if item.column() == COL_PREVIEW:
            self.refresh_preview()

    def _table_param_specs(self):
        """{name: (model_kind, lo_physical, hi_physical)} from the table's
        current Kind/bounds, or None (with a warning dialog already shown)
        if any row's bounds don't parse as lo < hi."""
        specs = {}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, COL_NAME).text()
            label = self.table.cellWidget(row, COL_KIND).currentText()
            d = KIND_DEFS[label]
            try:
                lo_disp = float(self.table.item(row, COL_LO).text())
                hi_disp = float(self.table.item(row, COL_HI).text())
            except ValueError:
                QMessageBox.warning(self, 'Build Custom Model', f"'{name}': bounds must be numbers.")
                return None
            if not lo_disp < hi_disp:
                QMessageBox.warning(self, 'Build Custom Model',
                                     f"'{name}': lower bound must be less than upper bound.")
                return None
            specs[name] = (d['model_kind'], d['to_phys'](lo_disp), d['to_phys'](hi_disp))
        return specs

    def refresh_preview(self):
        """Redraw j_λ(z) and phi'(z)/phi(z) over z in [0,1] using each
        row's own Preview value (converted from that row's display units
        to the physical units the model itself would see) -- called after
        on_parse, whenever a Kind dropdown changes, and whenever a Preview
        value cell is edited (see on_item_changed)."""
        try:
            emiss_expr = parse_custom_expr(self.emiss_edit.text(), 'j_lambda(z)')
            phi_expr = parse_custom_expr(self.phi_edit.text(), "phi'(z)")
        except CustomModelError:
            return
        # The equation cards only need the parsed expressions and the
        # bound boxes -- no numeric constant values -- so refresh them now,
        # before any of the numeric-preview logic below that can still
        # legitimately fail (e.g. a Preview value not yet filled in for a
        # brand new row).
        self.refit_dialog_equation(emiss_expr, phi_expr)
        names, values = [], []
        try:
            for row in range(self.table.rowCount()):
                label = self.table.cellWidget(row, COL_KIND).currentText()
                preview_disp = float(self.table.item(row, COL_PREVIEW).text())
                names.append(self.table.item(row, COL_NAME).text())
                values.append(KIND_DEFS[label]['to_phys'](preview_disp))
        except ValueError:
            return

        lambda_m = self.preview_lambda_spin.value() * 1e-3
        nu_mhz = C / lambda_m / 1e6
        try:
            # The table can be out of sync with j_λ(z)/phi''s own current
            # text (e.g. a bound box nudged, or Preview value edited,
            # before ever clicking Parse on freshly-typed text) --
            # `names` may then be missing a constant the expression
            # actually references, which only surfaces once sympy's
            # lambdified callable is actually called and hits an
            # undefined name.
            z, emiss_z, phi_prime_z, phi_z = preview_los_profiles(
                emiss_expr, phi_expr, names, values, nu_mhz, lambda_m, *self._bounds())
        except (NameError, TypeError, ValueError):
            return
        self.ax_emiss.clear()
        self.ax_phi.clear()
        # j_lambda(z) may now be complex (a profile using 'i' for a
        # position-dependent intrinsic EVPA, e.g. 'exp(2*i*chi_z*z)') --
        # .real/.imag are safe on a plain-real array too (imag is then all
        # zeros), so only draw the second line when there's an actual
        # imaginary part to show, keeping every existing real-only model's
        # preview pixel-identical to before this.
        has_imag = np.any(np.abs(emiss_z.imag) > 1e-12 * max(1.0, np.max(np.abs(emiss_z))))
        self.ax_emiss.plot(z, emiss_z.real, color='limegreen', label='Re' if has_imag else None)
        if has_imag:
            self.ax_emiss.plot(z, emiss_z.imag, color='limegreen', linestyle='dotted', label='Im')
            self.ax_emiss.legend(fontsize=8, loc='best')
        self.ax_phi.plot(z, phi_z, color='magenta', label=r"$\phi$")
        self.ax_phi.plot(z, phi_prime_z, color='magenta', linestyle='dotted', label=r"$\phi'$")
        self.ax_phi.legend(fontsize=8, loc='best')
        self.ax_phi.hlines(0, xmin=0, xmax=1, linestyle='dashed',color='k')
        self._style_preview_axes()
        self._apply_preview_margins()
        # A synchronous draw() (not draw_idle()'s deferred repaint) so the
        # preview always reflects what was just typed/selected the moment
        # this returns, regardless of the surrounding event-loop state --
        # this is an infrequently-updated dialog, not a slider-drag hot
        # loop, so there's no redraw-coalescing benefit to give up.
        self.canvas.draw()

    def on_define(self):
        label = self.label_edit.text().strip() or 'Custom model'
        param_specs = self._table_param_specs()
        if param_specs is None:
            return
        edit_name = self._edit_func.__name__ if self._edit_func is not None else None
        j_lo, j_hi, p_lo, p_hi = self._bounds()
        try:
            self.model_func = build_custom_model(
                label, self.emiss_edit.text(), self.phi_edit.text(), param_specs,
                j_lo=j_lo, j_hi=j_hi, p_lo=p_lo, p_hi=p_hi, name=edit_name)
        except (CustomModelError, ValueError) as e:
            QMessageBox.warning(self, 'Build Custom Model', str(e))
            return
        self.accept()
