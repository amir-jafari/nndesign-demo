"""Shared layout helpers for the framework-style demos of chapters 3, 6 and 10.

These demos (gradient descent vs Adam, the TensorFlow and the PyTorch
introductions) carry many more controls than the classic demos, so they use the
wider control panel first introduced by chapter 7: the blue divider is moved to
x = DIVIDER_X, the figure fills the left column and the controls are ordinary Qt
widgets in a (scrollable) panel on the right, under the chapter logo and the
short description. Everything else -- title, logo, "Chapter N", the blue lines
-- is the standard NNDLayout chrome.

All rectangles are in the 700 x 680 design coordinates of NNDLayout and are
scaled with the window like every other geometry in the package.
"""

import csv
import os

import numpy as np
from matplotlib import ticker
from PyQt6 import QtCore, QtGui, QtWidgets

from nndesigndemos.nndesign_layout import NNDLayout, JumpSlider
from nndesigndemos.get_package_path import PACKAGE_PATH

# ---------------------------------------------------------------- geometry
DIVIDER_X = 428                      # x of the blue vertical line
TITLE_RECT = (128, 25, 300, 100)     # title, right-aligned with the end of the horizontal line
DESC_RECT = (448, 26, 128, 108)      # description, beside the logo, clear of the title's end
ICON_RECT = (578, 16, 120, 100)      # chapter logo: top-right corner, level with the title band
PANEL_RECT = (438, 135, 259, 505)    # control panel, with a margin from the divider
PLOT_FULL = (5, 100, 419, 570)       # a figure that owns the whole left column
PLOT_TOP = (5, 100, 419, 400)        # a figure with a text widget underneath ...
TEXT_BOTTOM = (5, 500, 419, 170)     # ... this one

FS = 8   # one font size for every title / axis label / tick label / legend in the figures


def plain_log_ticks(axis):
    """Label a log axis with plain numbers (0.1, 1, ...) instead of large mathtext powers."""
    axis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(ticker.NullFormatter())

DATA_PATH = os.path.join(PACKAGE_PATH, "Data", "SampleDF.csv")


# ---------------------------------------------------------------- fonts
def scaled_pt(demo, pt):
    """Point size scaled the way NNDLayout.make_label scales its fonts."""
    if demo.running_on_windows or demo.running_on_linux:
        return max(6, int(pt * 0.78 * (demo.w_ratio + demo.h_ratio) / 2))
    return max(6, int(pt * (demo.w_ratio + demo.h_ratio) / 2))


def ui_font(demo, pt=11, bold=False):
    font = QtGui.QFont("Arial", scaled_pt(demo, pt))
    font.setBold(bold)
    return font


def mono_font(demo, pt=9):
    font = QtGui.QFont("Menlo", scaled_pt(demo, pt))
    font.setFamilies(["Menlo", "Consolas", "DejaVu Sans Mono", "Courier New"])
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font


# ---------------------------------------------------------------- data
def load_sample_df():
    """FVC (input P) and Percent (target T) columns of SampleDF.csv.

    The chapters read the file with pandas; the standard-library csv module is
    enough here. Falls back to synthetic data so the demos always run.
    """
    try:
        with open(DATA_PATH, newline="") as fh:
            rows = list(csv.DictReader(fh))
        P = np.array([float(r["FVC"]) for r in rows])
        T = np.array([float(r["Percent"]) for r in rows])
        return P, T, "SampleDF.csv: {} rows".format(len(P))
    except Exception as exc:
        rng = np.random.default_rng(0)
        P = rng.uniform(800, 5000, 100)
        T = 0.025 * P + rng.normal(0, 8, 100) + 10
        return P, T, "SampleDF.csv missing ({}) - using fake data".format(exc.__class__.__name__)


# ---------------------------------------------------------------- widgets
class LabeledSlider(QtWidgets.QWidget):
    """A slider with its name and current value written above it.

    lo, hi, val are in user units. With ``decimals`` > 0 the slider carries a
    fixed-point value; with ``log=True`` lo and hi are base-10 exponents and the
    value is 10 ** position (for learning rates).
    """

    valueChanged = QtCore.pyqtSignal()

    def __init__(self, demo, name, lo, hi, val, decimals=0, log=False, suffix="", fmt=None):
        super().__init__()
        self.name, self.decimals, self.log, self.suffix, self.fmt = name, decimals, log, suffix, fmt
        self._k = 10 ** decimals
        self.lbl = QtWidgets.QLabel()
        self.lbl.setFont(ui_font(demo, 10))
        self.sld = JumpSlider(QtCore.Qt.Orientation.Horizontal)
        if log:
            self.sld.setRange(int(round(lo * 100)), int(round(hi * 100)))
            self.sld.setValue(int(round(np.log10(val) * 100)))
        else:
            self.sld.setRange(int(round(lo * self._k)), int(round(hi * self._k)))
            self.sld.setValue(int(round(val * self._k)))
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.lbl)
        lay.addWidget(self.sld)
        self.sld.valueChanged.connect(self._on_change)
        self.refresh_label()

    def _on_change(self):
        self.refresh_label()
        self.valueChanged.emit()

    def text(self):
        v = self.value()
        if self.fmt:
            return self.fmt.format(v)
        if self.log or self.decimals:
            return "{:.5g}".format(v)
        return str(v)

    def refresh_label(self):
        self.lbl.setText("{}: <b>{}{}</b>".format(self.name, self.text(), self.suffix))

    def value(self):
        if self.log:
            return 10 ** (self.sld.value() / 100)
        v = self.sld.value() / self._k
        return v if self.decimals else int(v)

    def set_value(self, v):
        """Move the slider without emitting valueChanged."""
        raw = np.log10(v) * 100 if self.log else v * self._k
        blocked = self.sld.blockSignals(True)
        self.sld.setValue(int(round(raw)))
        self.sld.blockSignals(blocked)
        self.refresh_label()

    def set_max(self, hi):
        blocked = self.sld.blockSignals(True)
        self.sld.setMaximum(max(self.sld.minimum(), int(round(hi * self._k))))
        self.sld.blockSignals(blocked)
        self.refresh_label()

    def set_range(self, lo, hi, val=None):
        blocked = self.sld.blockSignals(True)
        self.sld.setRange(int(round(lo * self._k)), int(round(hi * self._k)))
        if val is not None:
            self.sld.setValue(int(round(val * self._k)))
        self.sld.blockSignals(blocked)
        self.refresh_label()

    def set_name(self, name):
        self.name = name
        self.refresh_label()


class PanelDemo(NNDLayout):
    """Base class of the framework-style demos.

    Subclasses set ``title``, ``chapter`` and ``description`` and build their
    widgets with the helpers below; the standard chrome is placed here.
    """

    title = ""
    chapter = 2
    description = ""

    def __init__(self, w_ratio, h_ratio, dpi):
        super(PanelDemo, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)
        self.xl2 = DIVIDER_X   # widen the control panel (before fill_chapter)
        self.fill_chapter(self.title, self.chapter, self.description,
                          PACKAGE_PATH + "Logo/book2_logos/{}.svg".format(self.chapter), None, 2,
                          description_coords=DESC_RECT)
        # fill_chapter places the title relative to the divider, which would put
        # it on top of the "DESIGN: DEEP LEARNING" label here; right-align it
        # with the end of the horizontal line instead.
        self.label3.setGeometry(TITLE_RECT[0] * self.w_ratio, TITLE_RECT[1] * self.h_ratio,
                                TITLE_RECT[2] * self.w_ratio, TITLE_RECT[3] * self.h_ratio)
        self.label3.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # Header band: description beside the logo (chapter 7 style), logo at the right edge.
        self.icon1.setGeometry(ICON_RECT[0] * self.w_ratio, ICON_RECT[1] * self.h_ratio,
                               ICON_RECT[2] * self.w_ratio, ICON_RECT[3] * self.h_ratio)
        font = self.label5.font()
        font.setPointSize(scaled_pt(self, 12))
        self.label5.setFont(font)
        self.label5.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label5.setWordWrap(True)   # safety net for wider fonts on other platforms

    # ---- figure ----
    def make_figure(self, rect=PLOT_FULL, plot_number=1):
        """A matplotlib figure (constrained layout) filling ``rect``.

        Returns (figure, canvas); they are also stored as self.figure / self.canvas
        (self.figure2 / self.canvas2, ... for other plot numbers) like make_plot.
        """
        self.make_plot(plot_number, rect)
        suffix = "" if plot_number == 1 else str(plot_number)
        figure, canvas = getattr(self, "figure" + suffix), getattr(self, "canvas" + suffix)
        figure.set_layout_engine("constrained")
        return figure, canvas

    # ---- panel ----
    def make_panel(self, widget, rect=PANEL_RECT):
        """Place ``widget`` as the control panel, inside a vertical scroll area."""
        widget.setFont(ui_font(self, 11))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.set_layout(rect, scroll, margins=(0, 0, 0, 0))
        return scroll

    def place(self, rect, widget, margins=(0, 0, 0, 0)):
        """Put any widget at a design-coordinate rectangle."""
        self.set_layout(rect, widget, margins=margins)

    def group(self, title, pt=11):
        """A bold-titled QGroupBox; returns (box, its QVBoxLayout)."""
        box = QtWidgets.QGroupBox(title)
        box.setFont(ui_font(self, pt, bold=True))
        lay = QtWidgets.QVBoxLayout(box)
        lay.setSpacing(3)
        lay.setContentsMargins(6, 4, 6, 6)
        return box, lay

    def label(self, text, pt=10, bold=False, wrap=False):
        lbl = QtWidgets.QLabel(text)
        lbl.setFont(ui_font(self, pt, bold))
        lbl.setWordWrap(wrap)
        return lbl

    def slider(self, name, lo, hi, val, slot=None, **kwargs):
        sld = LabeledSlider(self, name, lo, hi, val, **kwargs)
        if slot is not None:
            sld.valueChanged.connect(slot)
        return sld

    def combo(self, items, slot=None, pt=10):
        cb = QtWidgets.QComboBox()
        cb.setFont(ui_font(self, pt))
        cb.addItems(list(items))
        if slot is not None:
            cb.currentIndexChanged.connect(slot)
        return cb

    def checkbox(self, text, checked, slot=None, pt=10):
        cb = QtWidgets.QCheckBox(text)
        cb.setFont(ui_font(self, pt))
        cb.setChecked(checked)
        if slot is not None:
            cb.stateChanged.connect(slot)
        return cb

    def button(self, text, slot=None, bold=False, pt=10, mono=False):
        btn = QtWidgets.QPushButton(text)
        btn.setFont(mono_font(self, pt) if mono else ui_font(self, pt, bold))
        if slot is not None:
            btn.clicked.connect(slot)
        return btn

    def text_box(self, pt=9, wrap=False, min_height=None):
        """A read-only monospace QTextEdit for readouts; scrolls on its own."""
        box = QtWidgets.QTextEdit()
        box.setReadOnly(True)
        box.setFont(mono_font(self, pt))
        box.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth if wrap
                            else QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        box.document().setDocumentMargin(2)
        if min_height is not None:
            box.setMinimumHeight(int(min_height * self.h_ratio))
        return box
