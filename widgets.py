"""Small Qt widget primitives shared between app.py's parameter sliders
(ParamSlider) and sampling.py's prior-bounds sliders (SamplingBoundsRow) --
split out on its own so neither module has to import the other just to get
these."""
import re

from PyQt5.QtWidgets import QLineEdit

SLIDER_STEPS = 10000  # integer resolution backing each QSlider

# Physical unit shown next to each slider's readout, keyed by Param.kind.
UNITS = {'p': '%', 'X': '°', 'phi': 'rad/m²', 'dphi': 'rad/m²', 'scale': '', 'alpha': '', 'eps': '',
         'nu0': 'GHz'}
# Widest unit string, used to fix every slider's unit label to the same
# width so the value boxes above/below each other line up regardless of
# which unit (or none) a given row happens to show.
WIDEST_UNIT = max(UNITS.values(), key=len)

NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


class ValueLineEdit(QLineEdit):
    """A QLineEdit that selects all its text on focus, so clicking a
    parameter's value and typing immediately replaces it."""
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.selectAll()
