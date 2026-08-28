"""CustomModel -- the "Build Custom Model" window (see app.py's Models
menu action / "Custom model..." dropdown entry). Lets a user type j_p(z)
(the polarized emissivity envelope -- may reference the standard p0/chi0
sliders directly, see models.build_custom_model's own module comment) and
phi'(z) (the Faraday-depth *density*, may reference phi0 -- see models.
build_custom_model's own module comment for why this isn't phi(z) itself)
expressions for the general, normalized Sokoloff et al. 1998 (eq. 1) /
Burn 1966 line-of-sight integral (see models.build_custom_model), discover
the new constants those expressions introduce (beyond p0/chi0/phi0), pick
each one's Kind (which auto-sets its bounds to that kind's own usual range
-- see KIND_DEFS) and a preview value to evaluate it at, preview
j_p(z)/phi'(z)/phi(z) over the line of sight, and register the result
as a new selectable model. j_p(z)'s own frequency dependence beyond
whatever it references directly (the 'nu'/'lambda' symbols, see the intro
text below) isn't set here -- that's the main window's existing Spectrum
box (Power-law/SSA/Thermal/Log-parabola + alpha), applied on top exactly
as it is for every other single-component model, once the custom model is
selected there."""
import functools
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QDialogButtonBox, QMessageBox, QHeaderView, QSizePolicy, QTextBrowser)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from polvista.models import (
    C, build_custom_model, custom_model_equation_lines, custom_preview_n, discover_custom_params,
    parse_custom_expr, preview_los_profiles, preview_custom_bounds, CustomModelError,
    CUSTOM_EMISS_DEFAULT, CUSTOM_PHI_DEFAULT, SPECTRAL_SHAPES, _insert_opacity_factor,
    CUSTOM_P0_BOUNDS, CUSTOM_PHI0_BOUNDS, SPECTRAL_BOUNDS_LO_SINGLE, SPECTRAL_BOUNDS_HI_SINGLE,
    TEMP_BOUNDS_K, BETA_BOUNDS)
from polvista.latex_stuff import LATEX_DPI, fit_equation_pixmap, latex_pixmap, pixmap_to_img_tag

DEFAULT_PREVIEW_LAMBDA_MM = 3.0  # fallback when parent has no wl_min/wl_max to derive one from

# p0/chi0/phi0's own preview values -- j_p(z)/phi'(z) can now reference
# these three standard sliders directly (see models.py's own module comment
# above build_custom_model), but the builder dialog itself has no slider
# for them (those only exist once a model is actually registered and
# selected in the main window) -- so the j_p(z)/j(z)/phi'(z)/phi(z) preview
# plot (see refresh_preview) needs *some* fixed value to evaluate them at.
# Matches KIND_DEFS's own 'Fraction'/'Angle'/'Depth' preview_disp defaults
# (physical units: a 0-1 fraction, radians, rad/m^2) -- the same preview
# magnitudes p0/chi0/phi0 would get if they were ever offered as one of
# those Kinds themselves.
# nu0/alpha/T/beta's own preview values -- same reasoning as CUSTOM_P0_PREVIEW
# et al. above, but for the builder's own preview-only Spectral-model
# dropdown (see _build_ui): when it's set to SSA/Thermal, refresh_preview
# needs *some* fixed turnover-frequency/spectral-index/temperature/curvature
# to evaluate models.preview_los_profiles's own opacity term at, since (like
# p0/chi0/phi0) there are no real sliders for these yet at Define-time --
# the real values only exist once this model is registered and selected in
# the main window, where its actual opacity behavior is governed by the main
# window's own live Spectrum box, not this dropdown (see the dropdown's own
# tooltip). CUSTOM_NU0_PREVIEW [MHz] is a mid-band-ish placeholder (~10 cm);
# CUSTOM_ALPHA_PREVIEW is a typical optically-thin synchrotron index;
# CUSTOM_TEMP_PREVIEW [K] a typical HII-region electron temperature;
# CUSTOM_BETA_PREVIEW unused by SSA/Thermal (only 'logparabola', which this
# dropdown never previews opacity for) but still passed through for symmetry.
CUSTOM_NU0_PREVIEW = 3000.0
CUSTOM_ALPHA_PREVIEW = -0.7
CUSTOM_TEMP_PREVIEW = 1e4
CUSTOM_BETA_PREVIEW = 0.0

CUSTOM_P0_PREVIEW = 0.10
CUSTOM_CHI0_PREVIEW = 0.0
CUSTOM_PHI0_PREVIEW = 300.0

# COL_* -- the constants table's own column layout, shared by every method
# below that reads/writes a row.
COL_NAME, COL_KIND, COL_LO, COL_HI, COL_PREVIEW, COL_DESC = range(6)

# Background for a locked row's own non-editable cells (see _set_locked_row)
# -- visibly distinct from a discovered constant's plain white/editable
# cells, so a greyed-out Kind/bound/Description doesn't look like it could
# be typed into.
LOCKED_ROW_BG = QColor(224, 224, 224)

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
    'Dispersion': dict(model_kind='dphi', to_phys=lambda d: d, to_disp=lambda p: p,
                        bounds_disp=(0.0, 5.0e6), preview_disp=300.0),
    'Number': dict(model_kind='scale', to_phys=lambda d: d, to_disp=lambda p: p,
                    bounds_disp=(-10.0, 10.0), preview_disp=1.0),
    # Broad, "exotic spectral behavior"-friendly ranges (many decades, log
    # slider -- see app.ParamSlider's own 'freq'/'wave' branches) for a
    # constant meant to set the model's own characteristic frequency/
    # wavelength scale (e.g. a break frequency inside j_p(z)) -- MHz/m to
    # match the reserved nu/lambda symbols' own units exactly, no
    # conversion applied anywhere downstream (see CUSTOM_PARAM_KINDS's own
    # module comment in models.py for why this isn't just the 'nu0' kind).
    'Frequency': dict(model_kind='freq', to_phys=lambda d: d, to_disp=lambda p: p,
                       bounds_disp=(1.0, 1.0e7), preview_disp=3000.0),
    'Wavelength': dict(model_kind='wave', to_phys=lambda d: d, to_disp=lambda p: p,
                        bounds_disp=(1.0e-5, 1.0e3), preview_disp=0.003),
}
MODEL_KIND_TO_LABEL = {d['model_kind']: label for label, d in KIND_DEFS.items()}
DEFAULT_KIND_LABEL = 'Number'

# The always-present p0/chi0/phi0 sliders (see models.py's own module
# comment above build_custom_model) plus the Spectrum box's own
# nu0/alpha/T/beta -- shown as locked rows at the top of the constants
# table (see CustomModel._sync_locked_rows) purely for reference (their
# Kind/bounds/Description are fixed elsewhere -- CUSTOM_P0_BOUNDS, the
# main window's own SSA/Thermal sliders, etc. -- editing them here
# wouldn't do anything, so those cells are non-editable), with a genuinely
# editable Preview value that -- unlike a real discovered constant's --
# feeds refresh_preview directly in place of a fixed module constant, so
# the builder's own preview plot/equation can be explored at different
# p0/chi0/phi0/nu0/alpha/T/beta without editing code.
#
# `lo_disp`/`hi_disp` None means "no single fixed pair to show" -- nu0's
# own real bounds are dynamically derived from the current wavelength
# range (see app.NU0_BOUNDS_MULT), not a fixed constant, so its row just
# shows an em dash instead of duplicating that calculation here.
#
# `relevant(shape)` decides whether this row is shown at all for the
# Spectral-model dropdown's current selection -- mirrors app.py's own
# rebuild_sliders visibility exactly (nu0 needs SSA/Thermal, T needs
# Thermal, beta needs Log-parabola, alpha is hidden only for Thermal,
# where source_function's own alpha argument is inert).
LOCKED_PARAM_ROWS = [
    dict(name='p0', kind_text='Fraction', to_phys=KIND_DEFS['Fraction']['to_phys'],
         lo_disp=CUSTOM_P0_BOUNDS[0] * 100.0, hi_disp=CUSTOM_P0_BOUNDS[1] * 100.0,
         preview_default=CUSTOM_P0_PREVIEW * 100.0,
         description="Intrinsic fractional polarization amplitude.", relevant=lambda shape: True),
    dict(name='chi0', kind_text='Angle', to_phys=KIND_DEFS['Angle']['to_phys'],
         lo_disp=-90.0, hi_disp=90.0, preview_default=CUSTOM_CHI0_PREVIEW * 180.0 / np.pi,
         description="Intrinsic EVPA (overall constant phase).", relevant=lambda shape: True),
    dict(name='phi0', kind_text='Depth', to_phys=lambda d: d,
         lo_disp=CUSTOM_PHI0_BOUNDS[0], hi_disp=CUSTOM_PHI0_BOUNDS[1], preview_default=CUSTOM_PHI0_PREVIEW,
         description="Faraday-depth scale.", relevant=lambda shape: True),
    dict(name='nu0', kind_text='Frequency [MHz]', to_phys=lambda d: d,
         lo_disp=None, hi_disp=None, preview_default=CUSTOM_NU0_PREVIEW,
         description="SSA/Thermal turnover frequency (set by the main window's own "
                      "wavelength-range-derived bounds).", relevant=lambda shape: shape in ('ssa', 'thermal')),
    dict(name='alpha', kind_text='Spectral index', to_phys=lambda d: d,
         lo_disp=SPECTRAL_BOUNDS_LO_SINGLE[0], hi_disp=SPECTRAL_BOUNDS_HI_SINGLE[0],
         preview_default=CUSTOM_ALPHA_PREVIEW,
         description="Total-intensity spectral index: I(nu) ∝ nu^alpha.",
         relevant=lambda shape: shape != 'thermal'),
    dict(name='T', kind_text='Temperature [K]', to_phys=lambda d: d,
         lo_disp=TEMP_BOUNDS_K[0], hi_disp=TEMP_BOUNDS_K[1], preview_default=CUSTOM_TEMP_PREVIEW,
         description="Thermal free-free electron temperature.", relevant=lambda shape: shape == 'thermal'),
    dict(name='beta', kind_text='Curvature', to_phys=lambda d: d,
         lo_disp=BETA_BOUNDS[0], hi_disp=BETA_BOUNDS[1], preview_default=CUSTOM_BETA_PREVIEW,
         description="Log-parabola curvature index.", relevant=lambda shape: shape == 'logparabola'),
]


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
    CustomModel.__init__) -- same wording as the plain-text version
    this replaced, but every equation/variable name is a real
    mathtext-rendered image instead of an approximating unicode character,
    matching how ParamSlider labels the model's own parameters elsewhere in
    the app.

    `dpi` (see CustomModel.__init__, which derives it from the
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
    equation1 = m(r'$P(\lambda)= \frac{1}{J}\int_{-1}^{1} j_p(z)\,dz$', fontsize=18)
    equation2 = m(r'$J=\int_{-1}^{1} j(z)\,dz$', fontsize=16)

    depth_formula = m(r"$\phi(z)=\int_{\max(z, \mathrm{lower})}^{\mathrm{upper}}\phi'(z') dz'$",
                       fontsize=14)

    z_range, z_source, z_observer = m(r'$z\in[-1,1]$'), m(r'$z=-1$'), m(r'$z=1$')
    j_p, j_plain, z = m(r'$j_p(z)$'), m(r'$j(z)$'), m(r'$z$')
    phase_factor = m(r'$e^{2i\phi(z)\lambda^2}$')
    phi_prime = m(r"$\phi'(z)$")
    nu, lam = m(r'$\nu$'), m(r'$\lambda$')
    p0, phi0, w, chi0 = m(r'$p_0$'), m(r'$\phi_0$'), m(r'$w$'), m(r'$\chi_0$')
    i_unit = m(r'$i$')
    evpa_example = m(r'$e^{2i\chi(z)}$')
    delta_example = m(r'$\delta(z-z_0)$')
    gaussian_example = m(r'$\mathcal{G}(z;\,z_0,\,\sigma_z)$')
    opacity_factor = m(r'$e^{-\tau_\lambda(z)}$')
    return (
        '<p>Define a model from the general, normalized Faraday-rotation line-of-sight '
        'integral:</p>'
        f'<p align="center">{equation1}&nbsp;&nbsp;&nbsp;&nbsp;{equation2}</p>'
        f'<p>over a normalized line of sight {z_range} running the way light travels: '
        f'{z_source} at the source, {z_observer} at the observer.</p>'
        f'<p>{j_p} is the <i>polarized</i> emissivity: typed below as its own amplitude/EVPA/'
        f'shape (default \'p0 * exp(2 * i * chi0)\'), multiplying the always-present {phase_factor} '
        f'phase factor shown to its right (not itself editable) -- zero outside its own '
        f'lower/upper bound, set below. {p0} (fractional polarization amplitude) and {chi0} (EVPA) '
        f'are, like every other constant here, their own always-present sliders -- but now typed '
        f'directly into this field rather than applied automatically, so a screen that needs '
        f'neither to vary with position can simply leave that default as-is. The field may be '
        f'complex beyond just {chi0} too (e.g. {evpa_example}), giving the source its own '
        f'position-dependent intrinsic EVPA.</p>'
        f'<p>{j_plain} -- the plain, unpolarized emissivity implied by the same shape -- is that '
        f'field with {p0} forced to 1 and the magnitude taken of whatever is left ({chi0}\'s own '
        f'phase, or any other position-dependent phase the shape might carry, has magnitude 1 '
        f'either way and so cannot survive that). {equation2} is its own integral, normalizing '
        f'{equation1}\'s result so it depends only on {j_p}\'s own <i>shape</i>, not the arbitrary '
        f'bounds it happens to be integrated over.</p>'
        f'<p>{phi_prime} is a Faraday-depth <i>density</i> shape (zero outside its own lower/upper '
        f'bound, set below), not the depth itself -- the actual Faraday depth is its accumulated '
        f'integral, {depth_formula}. {phi0} (Faraday-depth scale) is, like {p0}/{chi0}, its own '
        f'always-present slider -- typed directly into this field too (default \'phi0\') -- so a '
        f'typical screen can leave {phi_prime} as \'phi0\' or scale a bare shape (e.g. \'phi0*z\') '
        f'to set the actual rad/m² magnitude.</p>'
        f'<p>Where the two ranges overlap, different emission depths pick up '
        f'different amounts of rotation before reaching the observer -- internal, differential '
        f'Faraday rotation.</p>'
        f'<p>Where {phi_prime}\'s own range lies entirely beyond {j_p}\'s, '
        f'every emission point instead sees the exact same total rotation -- a pure external '
        f'screen.</p>'
        f'<p>Each lower/upper bound field is usually just a plain number (every default here), '
        f'but may instead be an expression referencing a new constant (e.g. {w} again, so the '
        f'emitting region\'s own extent tracks a width that also shapes {j_p}) -- that constant '
        f'then gets a Kind/bounds/slider below exactly like one used directly in {j_p}/{phi_prime} '
        f'themselves.</p>'
        f'<p>The Spectral model dropdown above previews (not sets -- it always follows the main '
        f'window\'s own live Spectrum box once this model is selected there) whether this model '
        f'picks up opacity: SSA/Thermal add a genuine per-z {opacity_factor} attenuation multiplying '
        f'{j_p} (built from {j_plain}\'s own shape via Kirchhoff\'s law, not typed separately), shown '
        f'in the equation card and the {j_p} preview below; Power-law/Log-parabola never do.</p>'
        f'<p>Use {z}, {nu} (frequency, MHz), {lam} (wavelength, m), {i_unit} (the imaginary '
        f'unit), and any new constant name (e.g. {w}) with explicit \'*\' for products -- those '
        f'do need a Kind and bounds picked below. Leaving either field above blank means the '
        f'constant 1. delta(arg) (e.g. {delta_example}, for a thin/concentrated feature) is also '
        f'available -- realized internally as a very narrow, fixed-width Gaussian rather than a '
        f'literal point mass, since the latter can\'t be evaluated on any numerical grid. '
        f'gaussian(z0, sigma_z) (e.g. {gaussian_example}) is the same idea with a width you '
        f'choose yourself -- either a plain number, or a new constant name (with its own Kind/'
        f'bounds, so it becomes a live slider) -- both z0 and sigma_z can be either.</p>'
    )


def _math_field_label(latex, dpi, fontsize=13):
    """QLabel showing `latex` as a rendered mathtext image -- same
    latex_pixmap pipeline (and the same per-screen `dpi`, see
    CustomModel.__init__) as the intro text's inline math and
    ParamSlider's own parameter-name labels elsewhere in the app -- used
    for the j_p(z)/phi'(z) row labels below instead of plain
    approximating text."""
    label = QLabel()
    label.setPixmap(latex_pixmap(latex, fontsize=fontsize, facecolor=None, dpi=dpi))
    return label


def _bound_edit(default_text):
    """A plain QLineEdit for one j_lo/j_hi/p_lo/p_hi integration-bound
    field -- a free-form math expression (see parse_custom_expr), exactly
    like emiss_edit/phi_edit, not restricted to a bare float any more. A
    plain number (e.g. '-1', still every default here) behaves exactly as
    it always did; an expression referencing a *new* constant (e.g.
    '3*w') is what promotes that constant into a discovered param with its
    own slider -- see models.discover_custom_params. `default_text` is
    just the field's initial text."""
    return QLineEdit(default_text)


def _bound_text(value):
    """Text for a j_lo/j_hi/p_lo/p_hi bound box's own existing_def value
    (see CustomModel.__init__) -- `value` verbatim if it's already a
    string expression (every model built since bounds could be
    expressions), or `f'{value:.2f}'` if it's still a bare float (a
    'custom_definition' saved before that -- see models.CUSTOM_MODEL_DEFS)."""
    return value if isinstance(value, str) else f'{value:.2f}'


class CustomModel(QDialog):
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
    closure even when reusing the same name.

    `initial_spectral_shape`, if given, is the key (see SPECTRAL_SHAPES)
    the (preview-only) Spectral-model dropdown opens on instead of always
    defaulting to 'powerlaw' -- app.py's edit_custom_model_menu_action
    passes the main window's own live Spectrum-box shape for the model
    being edited, so re-opening it to edit doesn't visually revert opacity
    off (in the equation card/preview here) when it's actually still on
    for this model in the main window. Left None for the plain "Custom
    model..."/"Build Custom Model..." paths, which have no existing model
    -- and so no live shape -- to inherit one from."""

    def __init__(self, parent=None, existing_def=None, edit_func=None, initial_spectral_shape=None):
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
        # Stashed for refit_dialog_equation, which re-renders emiss_phase_label
        # at this same DPI whenever the Spectral-model dropdown changes.
        self._intro_dpi = intro_dpi

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

        self.spectral_shape_combo = QComboBox()
        for shape_label, shape_key in SPECTRAL_SHAPES:
            self.spectral_shape_combo.addItem(shape_label, shape_key)
        self.spectral_shape_combo.setCurrentIndex(
            self.spectral_shape_combo.findData(initial_spectral_shape or 'powerlaw'))
        self.spectral_shape_combo.setToolTip(
            "Preview only -- picking SSA/Thermal here previews the e^{-τ_λ(z)} opacity term this "
            'model will pick up whenever it is selected in the main window *and* the main window\'s '
            "own Spectrum box is set to SSA/Thermal there (that live selection is what actually "
            "controls it, not this dropdown -- see the intro text above). Power-law/Log-parabola "
            'have no opacity term at all, matching every other model.')
        self.spectral_shape_combo.currentIndexChanged.connect(self.refresh_preview)
        # Sharing a row with Model name (rather than its own form row) keeps
        # it visually secondary to the name field -- a preview-only choice,
        # not a peer of j_p(z)/φ'(z) below -- while still being reachable
        # without scrolling. label_edit's own stretch=1 is the row's only
        # stretch, so it claims every pixel up to the "Spectral model:"
        # label rather than sharing that space with a separate spacer.
        name_row = QHBoxLayout()
        name_row.addWidget(self.label_edit, stretch=1)
        name_row.addWidget(QLabel('Spectral model:'))
        name_row.addWidget(self.spectral_shape_combo)

        emiss_bound_tip = ('j_p(z) is forced to 0 outside [lower, upper] -- the emitting region '
                            'is confined to that range. Usually a plain number, but may be an '
                            "expression (e.g. '3*w') referencing a new constant, which then gets "
                            'its own Kind/bounds/slider below once Parsed, same as one used '
                            'directly in j_p(z)/φ\'(z).')
        self.emiss_edit = QLineEdit()
        self.emiss_edit.setText(CUSTOM_EMISS_DEFAULT)
        #self.emiss_edit.setPlaceholderText('e.g. p0*exp(2*i*chi0)*exp(-((z-0.5)/w)**2)')
        self.j_lo_edit = _bound_edit('-1')
        self.j_hi_edit = _bound_edit('1')
        for edit in (self.j_lo_edit, self.j_hi_edit):
            edit.setFixedWidth(70)
            edit.setToolTip(emiss_bound_tip)
            edit.editingFinished.connect(self.refresh_preview)
        # The e^{2i*phi(z)*lambda^2} phase factor is never itself part of
        # j_p(z)'s editable text -- it's a fixed, always-present multiplier
        # shown here purely for context (see models.py's own module comment
        # above build_custom_model) -- only the amplitude/EVPA/shape to its
        # left is user-editable. Not a plain QLabel built once: it also
        # gains the same e^{-τ_λ(z)} opacity factor the equation cards below
        # do (see refit_dialog_equation/_emiss_phase_latex) whenever the
        # Spectral-model dropdown is SSA/Thermal, so this stays in sync
        # with them as that dropdown changes.
        self.emiss_phase_label = _math_field_label(self._emiss_phase_latex(), intro_dpi)
        emiss_row = QHBoxLayout()
        emiss_row.addWidget(self.emiss_edit, stretch=1)
        emiss_row.addWidget(self.emiss_phase_label)
        emiss_row.addWidget(QLabel('for'))
        emiss_row.addWidget(self.j_lo_edit)
        emiss_row.addWidget(QLabel('≤ z ≤'))
        emiss_row.addWidget(self.j_hi_edit)

        phi_bound_tip = ("φ'(z) is forced to 0 outside [lower, upper], so the accumulated "
                          'Faraday depth φ(z) only picks up rotation from within that range. '
                          "Usually a plain number, but may be an expression (e.g. '3*w') "
                          'referencing a new constant, which then gets its own Kind/bounds/'
                          'slider below once Parsed, same as one used directly in j_p(z)/φ\'(z).')
        self.phi_edit = QLineEdit()
        self.phi_edit.setText(CUSTOM_PHI_DEFAULT)
        self.phi_edit.setPlaceholderText('e.g. phi0*z')
        self.p_lo_edit = _bound_edit('-1')
        self.p_hi_edit = _bound_edit('1')
        for edit in (self.p_lo_edit, self.p_hi_edit):
            edit.setFixedWidth(70)
            edit.setToolTip(phi_bound_tip)
            edit.editingFinished.connect(self.refresh_preview)
        phi_row = QHBoxLayout()
        phi_row.addWidget(self.phi_edit, stretch=1)
        phi_row.addWidget(QLabel('for'))
        phi_row.addWidget(self.p_lo_edit)
        phi_row.addWidget(QLabel('≤ z ≤'))
        phi_row.addWidget(self.p_hi_edit)

        form.addRow('Model name:', name_row)
        form.addRow(_math_field_label(r'$j_p(z)\ =$', intro_dpi), emiss_row)
        form.addRow(_math_field_label(r"$\phi'(z)\ =$", intro_dpi), phi_row)
        layout.addLayout(form)

        parse_row = QHBoxLayout()
        self.parse_button = QPushButton('Parse')
        self.parse_button.setToolTip(
            "Discover the constants used in j_p(z)/φ'(z) and the four bound boxes (other than "
            "the standard p0/chi0/phi0 sliders), (re)build the table below "
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
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ['Constant', 'Kind', 'Lower bound', 'Upper bound', 'Preview value', 'Description'])
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(COL_KIND, QHeaderView.ResizeToContents)
        # Description is free text (what a constant represents physically,
        # e.g. 'Gaussian FWHM of the emitting shell') and typically the
        # longest column by far -- Stretch is what actually lets it use
        # whatever width the other, ResizeToContents/interactive columns
        # don't need, rather than sitting at some arbitrary fixed guess.
        self.table.horizontalHeader().setSectionResizeMode(COL_DESC, QHeaderView.Stretch)
        #self.table.setToolTip(
        #    'Kind auto-fills Lower/Upper bound and Preview value with that kind\'s usual range -- '
        #    'all three stay editable afterward. Preview value (only, not the bounds) is what the '
        #    "j_p(z)/φ'(z)/φ(z) plots to the right are evaluated at. Description is free text -- "
        #    "shown as that constant's own slider tooltip once selected in the main window.")
        self.table.itemChanged.connect(self.on_item_changed)
        # How many of the table's own rows, starting at 0, are currently
        # the locked p0/chi0/phi0/nu0/alpha/T/beta rows (see
        # LOCKED_PARAM_ROWS) rather than a discovered constant -- kept in
        # sync by _sync_locked_rows, read by on_parse/_table_param_specs/
        # refresh_preview to know where the discovered-constant rows
        # actually start. p0/chi0/phi0 are always relevant, so this is
        # never 0 once _sync_locked_rows below has run at least once.
        self._n_locked_rows = 0
        self._sync_locked_rows()
        table_col.addWidget(self.table)
        content_row.addLayout(table_col, stretch=1)

        preview_col = QVBoxLayout()
        preview_col.addWidget(QLabel('j_p(z), φ\'(z)/φ(z) preview (at each constant’s Preview value):'))
        preview_lambda_row = QHBoxLayout()
        preview_lambda_row.addWidget(QLabel('Preview λ [mm]:'))
        self.preview_lambda_spin = QDoubleSpinBox()
        self.preview_lambda_spin.setRange(1e-4, 1.0e5)
        self.preview_lambda_spin.setDecimals(4)
        self.preview_lambda_spin.setValue(self._default_preview_lambda_mm(parent))
        self.preview_lambda_spin.setToolTip(
            "Wavelength j_p(z)/phi'(z) are evaluated at for this preview, if either references "
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
            # j_lo/j_hi/p_lo/p_hi replaced the old single-sided z1/z2
            # bounds -- fall back to the closest equivalent (z1 was always
            # emiss's upper bound with an implicit 0 lower bound; z2 was
            # always phi''s lower bound with an implicit 1 upper bound) for
            # a definition saved before that change, rather than silently
            # reverting a truncated model to the new all-purpose (0, 1)
            # defaults. _bound_text handles either a fresh string
            # expression or a legacy bare float the same way a saved file
            # from before bounds could be expressions would still have.
            self.j_lo_edit.setText(_bound_text(existing_def.get('j_lo', -1.0)))
            self.j_hi_edit.setText(_bound_text(existing_def.get('j_hi', existing_def.get('z1', 1.0))))
            self.p_lo_edit.setText(_bound_text(existing_def.get('p_lo', existing_def.get('z2', -1.0))))
            self.p_hi_edit.setText(_bound_text(existing_def.get('p_hi', 1.0)))
            # Parse *after* every field (including the four bound boxes
            # above) is set, not before -- on_parse now has to discover
            # constants out of the bound expressions too (see
            # models.discover_custom_params), so it needs their saved text
            # in place first, not still showing the fresh-dialog default.
            # QLineEdit's setText() (unlike a spinbox's setValue()) doesn't
            # itself fire editingFinished, so nothing above re-runs the
            # preview until the explicit refresh_preview() at the end of
            # this block; that still needs the table already populated
            # with every constant j_p(z)/phi'(z)/the four bounds actually
            # reference, or it can't even build the preview's lambdified
            # callables.
            self.on_parse()
            specs = existing_def.get('param_specs', {})
            for row in range(self.table.rowCount()):
                name = self.table.item(row, COL_NAME).text()
                if name not in specs:
                    continue
                # *rest -- a saved spec has no Description element at all
                # if it predates that column (see models.build_custom_model);
                # description then just stays '', same as a fresh row.
                model_kind, lo_phys, hi_phys, *rest = specs[name]
                description = rest[0] if rest else ''
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
                self.table.item(row, COL_DESC).setText(description)
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
        wavelength range [mm] (a physically relevant default if j_p(z)/
        phi'(z) reference 'nu'/'lambda' at all), or DEFAULT_PREVIEW_LAMBDA_MM
        if `parent` isn't a MainWindow with that range available (e.g. a
        standalone test)."""
        try:
            return float(np.sqrt(parent.wl_min.value() * parent.wl_max.value()))
        except AttributeError:
            return DEFAULT_PREVIEW_LAMBDA_MM

    def _bound_exprs(self):
        """(j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr) parsed (see
        models.parse_custom_expr) from the four bound boxes -- may raise
        CustomModelError, exactly like parsing emiss_edit/phi_edit's own
        text, since a bound box is now just as free-form."""
        return (parse_custom_expr(self.j_lo_edit.text(), 'j_lo'),
                parse_custom_expr(self.j_hi_edit.text(), 'j_hi'),
                parse_custom_expr(self.p_lo_edit.text(), 'p_lo'),
                parse_custom_expr(self.p_hi_edit.text(), 'p_hi'))

    # Fixed PIXEL margins for the stacked emiss/phi preview axes (see
    # apply_fixed_margins in widgets.py for the same left/right/bottom/top
    # -> subplots_adjust fraction technique, used here instead of
    # tight_layout() specifically so hspace can be pinned to exactly 0 --
    # tight_layout() recomputes every spacing itself, including hspace,
    # so anything it's given would just get overridden back to whatever
    # it thinks the tick/label geometry needs).
    PREVIEW_MARGINS_PX = dict(left=85, right=15, bottom=65, top=10)

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
        self.ax_emiss.set_ylabel(r'$j_p(z)$')
        self.ax_phi.set_ylabel(r"$\phi(z),\ \phi'(z)$")
        self.ax_phi.set_xlabel('z  (-1 = source far-side  →  1 = observer)', fontsize=14)
        self.ax_emiss.set_xlim(-1.0, 1.0)
        #self.ax_emiss.set_ylim(0.0,)
        self.ax_phi.set_xlim(-1.0, 1.0)
        # sharex=True already keeps the two x-axes in sync -- hide the top
        # panel's own tick labels so hspace=0 (see _apply_preview_margins)
        # doesn't jam them into the gap against ax_phi's plot area.
        self.ax_emiss.tick_params(labelbottom=False)
        self.ax_emiss.grid(True)
        self.ax_phi.grid(True)

    def _emiss_phase_latex(self):
        """LaTeX for emiss_phase_label -- the fixed, always-present
        multiplier shown to the right of the editable j_p(z) field: the
        Faraday phase factor, plus the same e^{-τ_λ(z)} opacity factor
        the equation cards below gain (see refit_dialog_equation/
        models._insert_opacity_factor) whenever the Spectral-model
        dropdown (preview-only, see its own tooltip) is SSA/Thermal."""
        latex = r'\cdot\ e^{2i\phi(z)\lambda^2}'
        if self.spectral_shape_combo.currentData() in ('ssa', 'thermal'):
            latex += r'\,e^{-\tau_\lambda(z)}'
        return f'${latex}$'

    def refit_dialog_equation(self, emiss_expr, phi_expr):
        """Re-render the two live equation previews (see models.
        custom_model_equation_lines) to fit each one's own panel, and the
        j_p(z) row's own fixed phase-factor label (see
        _emiss_phase_latex) -- called from refresh_preview (whenever
        j_p(z)/φ'(z)/the bound boxes, or the Spectral-model dropdown,
        change) and from resizeEvent (whenever the dialog itself is
        resized), matching app.py's own MainWindow.refit_equation. The
        bound expressions render exactly as typed (a bare number as that
        number, e.g. '3*w' as that fraction) -- see
        custom_model_equation_lines. When the Spectral-model dropdown
        (preview-only, see its own tooltip) is SSA/Thermal, splices in the
        same e^{-τ_λ(z)} opacity factor full_equation adds to a registered
        model's own equation card under the same condition (see
        models._insert_opacity_factor)."""
        try:
            j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr = self._bound_exprs()
        except CustomModelError:
            return
        phi_line, p_line = custom_model_equation_lines(
            emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr)
        if self.spectral_shape_combo.currentData() in ('ssa', 'thermal'):
            p_line = _insert_opacity_factor(p_line)
        self.emiss_phase_label.setPixmap(latex_pixmap(self._emiss_phase_latex(), fontsize=13, facecolor=None,
                                                        dpi=self._intro_dpi))
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
            emiss_expr = parse_custom_expr(self.emiss_edit.text(), 'j_p(z)')
            phi_expr = parse_custom_expr(self.phi_edit.text(), "phi'(z)")
        except CustomModelError:
            pass
        else:
            self.refit_dialog_equation(emiss_expr, phi_expr)
        self._apply_preview_margins()
        self.canvas.draw_idle()

    def _row_state(self, row):
        """(kind_label, lo_text, hi_text, preview_text, desc_text)
        currently shown in `row` -- used by on_parse to carry a surviving
        name's row forward across a re-parse."""
        combo = self.table.cellWidget(row, COL_KIND)
        return (combo.currentText(), self.table.item(row, COL_LO).text(),
                self.table.item(row, COL_HI).text(), self.table.item(row, COL_PREVIEW).text(),
                self.table.item(row, COL_DESC).text())

    def _default_row_state(self):
        d = KIND_DEFS[DEFAULT_KIND_LABEL]
        lo, hi = d['bounds_disp']
        return DEFAULT_KIND_LABEL, str(lo), str(hi), str(d['preview_disp']), ''

    def _set_row(self, row, name, kind_label, lo_text, hi_text, preview_text, desc_text=''):
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
        # Free text describing what this constant represents physically --
        # purely informational, doesn't feed the preview or the model
        # itself (see on_item_changed) -- surfaced as this constant's own
        # slider tooltip once the model is registered and selected in the
        # main window (see models.build_custom_model, app.ParamSlider).
        self.table.setItem(row, COL_DESC, QTableWidgetItem(desc_text))

    def _set_locked_row(self, row, param, preview_text):
        """One locked row (see LOCKED_PARAM_ROWS) at `row` -- every column
        but Preview value is non-editable (Qt.ItemIsEnabled only, no
        Qt.ItemIsEditable) *and* visibly greyed (LOCKED_ROW_BG), since
        Kind/bounds/Description are fixed elsewhere for these, unlike a
        real discovered constant's own row (see _set_row) -- an editable-
        looking cell that silently did nothing on edit would be worse than
        not offering it at all. `preview_text` is carried over from the
        row's own previous text by _sync_locked_rows, or that param's own
        preview_default the first time it appears."""
        name_item = QTableWidgetItem(param['name'])
        name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(row, COL_NAME, name_item)

        kind_item = QTableWidgetItem(param['kind_text'])
        kind_item.setFlags(Qt.ItemIsEnabled)
        self.table.setItem(row, COL_KIND, kind_item)

        lo_text = '—' if param['lo_disp'] is None else str(param['lo_disp'])
        hi_text = '—' if param['hi_disp'] is None else str(param['hi_disp'])
        for col, text in ((COL_LO, lo_text), (COL_HI, hi_text), (COL_DESC, param['description'])):
            cell = QTableWidgetItem(text)
            cell.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, col, cell)

        # Full text on hover for whichever cell the Description column's
        # own Stretch width doesn't have room to show in full (see
        # _build_ui's own setSectionResizeMode(COL_DESC, ...) comment).
        self.table.item(row, COL_DESC).setToolTip(param['description'])

        for col in (COL_NAME, COL_KIND, COL_LO, COL_HI, COL_DESC):
            self.table.item(row, col).setBackground(LOCKED_ROW_BG)

        # Deliberately left with the default (editable, white-background)
        # flags/look, unlike every other cell in this row -- see
        # LOCKED_PARAM_ROWS's own module comment for why this one
        # genuinely drives refresh_preview.
        self.table.setItem(row, COL_PREVIEW, QTableWidgetItem(preview_text))

    def _sync_locked_rows(self):
        """(Re)build the table's locked p0/chi0/phi0/nu0/alpha/T/beta rows
        (see LOCKED_PARAM_ROWS) at the very top of the table to match
        which ones are relevant for the Spectral-model dropdown's current
        selection -- called once from __init__ and again at the top of
        every refresh_preview (so switching that dropdown adds/removes
        nu0/alpha/T/beta rows live). Idempotent when the relevant set
        hasn't actually changed (the common case -- most refresh_preview
        calls are from an unrelated edit) -- an early return there is what
        keeps a user's own Preview-value edit on a locked row from being
        silently reset back to that param's default on every keystroke
        elsewhere in the dialog; only an actual dropdown change reaches
        the rebuild below, and even then a still-relevant row's own
        current Preview text survives into its rebuilt row."""
        shape = self.spectral_shape_combo.currentData()
        wanted = [p for p in LOCKED_PARAM_ROWS if p['relevant'](shape)]
        wanted_names = [p['name'] for p in wanted]
        current_names = [self.table.item(row, COL_NAME).text() for row in range(self._n_locked_rows)]
        if current_names == wanted_names:
            return

        old_preview = {self.table.item(row, COL_NAME).text(): self.table.item(row, COL_PREVIEW).text()
                        for row in range(self._n_locked_rows)}

        self.table.blockSignals(True)
        for _ in range(self._n_locked_rows):
            self.table.removeRow(0)
        for i, param in enumerate(wanted):
            self.table.insertRow(i)
            preview_text = old_preview.get(param['name'], str(param['preview_default']))
            self._set_locked_row(i, param, preview_text)
        self._n_locked_rows = len(wanted)
        self.table.blockSignals(False)

    def _locked_row_values(self):
        """{name: physical_value} for every currently-shown locked row
        (p0/chi0/phi0 always present; nu0/alpha/T/beta only when relevant
        to the current Spectral-model dropdown selection -- see
        LOCKED_PARAM_ROWS/_sync_locked_rows), read from each row's own
        (possibly user-edited) Preview cell and converted to physical
        units via that param's own to_phys. Plain ValueError (uncaught) if
        any doesn't parse as a number -- refresh_preview's own try/except
        already handles that the same way a malformed discovered-
        constant's Preview cell is handled."""
        values = {}
        for row in range(self._n_locked_rows):
            name = self.table.item(row, COL_NAME).text()
            param = next(p for p in LOCKED_PARAM_ROWS if p['name'] == name)
            disp = float(self.table.item(row, COL_PREVIEW).text())
            values[name] = param['to_phys'](disp)
        return values

    def on_parse(self):
        """Parse j_p(z)/φ'(z) and the four bound boxes, (re)populate the
        constants table -- rows for names that survive re-parsing keep
        their existing Kind/bounds/preview value, new names start at
        Kind=Number -- and refresh the preview plot. A constant used only
        inside a bound expression (e.g. j_hi='3*w') shows up here exactly
        like one used directly in j_p(z)/φ'(z) -- see
        models.discover_custom_params. The Define Model button only
        unlocks once this succeeds at least once."""
        try:
            emiss_expr = parse_custom_expr(self.emiss_edit.text(), 'j_p(z)')
            phi_expr = parse_custom_expr(self.phi_edit.text(), "phi'(z)")
            j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr = self._bound_exprs()
            names = discover_custom_params(emiss_expr, phi_expr, j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr)
        except CustomModelError as e:
            QMessageBox.warning(self, 'Build Custom Model', str(e))
            self.define_button.setEnabled(False)
            return

        # Rows [0, self._n_locked_rows) are the locked p0/chi0/phi0/nu0/
        # alpha/T/beta rows (see LOCKED_PARAM_ROWS/_sync_locked_rows) --
        # untouched here, only the discovered-constant rows after them are
        # rebuilt.
        old_state = {self.table.item(row, COL_NAME).text(): self._row_state(row)
                     for row in range(self._n_locked_rows, self.table.rowCount())}

        self.table.blockSignals(True)
        self.table.setRowCount(self._n_locked_rows + len(names))
        for i, name in enumerate(names):
            row = self._n_locked_rows + i
            kind_label, lo_txt, hi_txt, prev_txt, desc_txt = old_state.get(name, self._default_row_state())
            self._set_row(row, name, kind_label, lo_txt, hi_txt, prev_txt, desc_txt)
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
        """{name: (model_kind, lo_physical, hi_physical, description)} from
        the table's current Kind/bounds/Description, or None (with a
        warning dialog already shown) if any row's bounds don't parse as
        lo < hi. `description` (the Description column's free text,
        possibly '') becomes that constant's Param.description -- see
        models.build_custom_model. Skips the locked p0/chi0/phi0/nu0/alpha/
        T/beta rows (see LOCKED_PARAM_ROWS/_sync_locked_rows) -- those
        aren't discovered constants at all (p0/chi0/phi0 are already
        always-present sliders build_custom_model adds on its own;
        nu0/alpha/T/beta aren't even part of param_specs), just shown here
        for reference with an editable Preview value."""
        specs = {}
        for row in range(self._n_locked_rows, self.table.rowCount()):
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
            description = self.table.item(row, COL_DESC).text().strip()
            specs[name] = (d['model_kind'], d['to_phys'](lo_disp), d['to_phys'](hi_disp), description)
        return specs

    def refresh_preview(self):
        """Redraw j_p(z) and phi'(z)/phi(z) over z in [-1,1] using each
        row's own Preview value (converted from that row's display units
        to the physical units the model itself would see) -- called after
        on_parse, whenever a Kind dropdown or the Spectral-model dropdown
        changes, and whenever a Preview value cell is edited (see
        on_item_changed). _sync_locked_rows runs first (cheap/idempotent
        when nothing needs to change, see its own docstring) so a
        Spectral-model dropdown change's nu0/alpha/T/beta row adds/removes
        are reflected before anything below reads the table."""
        self._sync_locked_rows()
        try:
            emiss_expr = parse_custom_expr(self.emiss_edit.text(), 'j_p(z)')
            phi_expr = parse_custom_expr(self.phi_edit.text(), "phi'(z)")
            j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr = self._bound_exprs()
        except CustomModelError:
            return
        # The equation cards only need the parsed expressions -- no numeric
        # constant values -- so refresh them now, before any of the
        # numeric-preview logic below that can still legitimately fail
        # (e.g. a Preview value not yet filled in for a brand new row).
        self.refit_dialog_equation(emiss_expr, phi_expr)
        try:
            locked = self._locked_row_values()
        except ValueError:
            return
        names, values = [], []
        try:
            for row in range(self._n_locked_rows, self.table.rowCount()):
                label = self.table.cellWidget(row, COL_KIND).currentText()
                preview_disp = float(self.table.item(row, COL_PREVIEW).text())
                names.append(self.table.item(row, COL_NAME).text())
                values.append(KIND_DEFS[label]['to_phys'](preview_disp))
        except ValueError:
            return

        lambda_m = self.preview_lambda_spin.value() * 1e-3
        nu_mhz = C / lambda_m / 1e6
        p0_prev, chi0_prev, phi0_prev = locked['p0'], locked['chi0'], locked['phi0']
        try:
            # The table can be out of sync with j_p(z)/phi''s (or a bound
            # box's) own current text (e.g. a bound box nudged, or Preview
            # value edited, before ever clicking Parse on freshly-typed
            # text) -- `names` may then be missing a constant an
            # expression actually references, which only surfaces once
            # sympy's lambdified callable is actually called and hits an
            # undefined name. p0/chi0/phi0/nu0/alpha/T/beta -- read from
            # the table's own locked rows above, not a fixed module
            # constant any more -- aren't in `names`/`values` at all:
            # they're the model's own standard sliders (or, for nu0/alpha/
            # T/beta, the Spectrum box's), not discovered constants, but
            # j_p(z)/phi'(z)/the four bounds can still reference p0/chi0/
            # phi0 directly (see models.py's own module comment above
            # build_custom_model).
            #
            # preview_los_profiles's own default n=300 is nowhere near
            # enough to make a delta(...)/gaussian(...)'s own narrow spike
            # visible (or even numerically present) -- bump it the same way
            # build_custom_model's own custom_func does whenever either
            # field uses one (see custom_preview_n).
            j_lo, j_hi, p_lo, p_hi = preview_custom_bounds(
                j_lo_expr, j_hi_expr, p_lo_expr, p_hi_expr, names, values,
                p0_prev, chi0_prev, phi0_prev)
            preview_n = custom_preview_n(emiss_expr, phi_expr, names, values,
                                          p0_prev, chi0_prev, phi0_prev)
            spectral_shape = self.spectral_shape_combo.currentData()
            z, emiss_p_z, phi_prime_z, phi_z, emiss_p_z_attenuated = preview_los_profiles(
                emiss_expr, phi_expr, names, values,
                p0_prev, chi0_prev, phi0_prev,
                nu_mhz, lambda_m, j_lo, j_hi, p_lo, p_hi, n=preview_n,
                shape=spectral_shape, nu0=locked.get('nu0', CUSTOM_NU0_PREVIEW),
                alpha_val=locked.get('alpha', CUSTOM_ALPHA_PREVIEW),
                T_val=locked.get('T', CUSTOM_TEMP_PREVIEW), beta_val=locked.get('beta', CUSTOM_BETA_PREVIEW))
        except (NameError, TypeError, ValueError):
            return
        self.ax_emiss.clear()
        self.ax_phi.clear()
        # j_p(z) may now be complex (a profile using 'i' for a
        # position-dependent intrinsic EVPA, e.g. 'exp(2*i*chi_z*z)') --
        # .real/.imag are safe on a plain-real array too (imag is then all
        # zeros), so only draw the second line when there's an actual
        # imaginary part to show, keeping every existing real-only model's
        # preview pixel-identical to before this. j(z), the denominator's
        # own unpolarized shape, isn't plotted here at all: it only ever
        # collapses to one number, J (see the equation cards above), so a
        # per-z preview of it wouldn't show anything P(lambda)'s own shape
        # actually depends on -- only its overall normalization.
        #
        # When the (preview-only, see its own tooltip) Spectral-model
        # dropdown is SSA/Thermal, `j_p^0(z)` (unattenuated, as typed) and
        # `j_p(z)` (attenuated by the same e^{-τ_λ(z)} envelope
        # models._custom_opacity_attenuation applies to the real, registered
        # model) are both drawn, so the *magnitude* of opacity's effect is
        # visible here, not just its presence in the equation card above.
        show_opacity = spectral_shape in ('ssa', 'thermal')
        has_imag = np.any(np.abs(emiss_p_z.imag) > 1e-12 * max(1.0, np.max(np.abs(emiss_p_z))))
        unatten_label = r'$j_p^0(z)$' if show_opacity else None
        self.ax_emiss.plot(z, emiss_p_z.real, color='limegreen',
                            label=(f'{unatten_label} (Re)' if (unatten_label and has_imag)
                                   else (unatten_label or ('Re' if has_imag else None))))
        if has_imag:
            self.ax_emiss.plot(z, emiss_p_z.imag, color='limegreen', linestyle='dotted',
                                label=(f'{unatten_label} (Im)' if unatten_label else 'Im'))
        if show_opacity:
            self.ax_emiss.plot(z, emiss_p_z_attenuated.real, color='darkgreen',
                                label=(r'$j_p(z)$ (Re)' if has_imag else r'$j_p(z)$'))
            if has_imag:
                self.ax_emiss.plot(z, emiss_p_z_attenuated.imag, color='darkgreen', linestyle='dotted',
                                    label=r'$j_p(z)$ (Im)')
        if has_imag or show_opacity:
            self.ax_emiss.legend(fontsize=8, loc='best')
        self.ax_phi.plot(z, phi_z, color='magenta', label=r"$\phi$")
        self.ax_phi.plot(z, phi_prime_z, color='magenta', linestyle='dotted', label=r"$\phi'$")
        self.ax_phi.legend(fontsize=8, loc='best')
        self.ax_phi.hlines(0, xmin=-1, xmax=1, linestyle='dashed',color='k')
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
        try:
            self.model_func = build_custom_model(
                label, self.emiss_edit.text(), self.phi_edit.text(), param_specs,
                j_lo=self.j_lo_edit.text(), j_hi=self.j_hi_edit.text(),
                p_lo=self.p_lo_edit.text(), p_hi=self.p_hi_edit.text(), name=edit_name)
        except (CustomModelError, ValueError) as e:
            QMessageBox.warning(self, 'Build Custom Model', str(e))
            return
        self.accept()
