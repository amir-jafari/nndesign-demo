from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class PoslinDecisionRegionsBase(NNDLayout):
    # Slider integer range (-SLIDER_MAX, SLIDER_MAX) maps to (-SLIDER_MAX/SLIDER_SCALE, +...) real values,
    # i.e. -5 .. 5 in steps of 0.25.
    SLIDER_MAX = 20
    SLIDER_SCALE = 4

    # Grid geometry (logical px) for the parameter sliders below the plot.
    HEADER_Y = 488
    ROW_Y0 = 524
    ROW_PITCH = 36
    LABEL_H = 16
    SLIDER_H = 20
    SLIDER_W = 85
    COL_X = {"w1_0": 22, "w1_1": 110, "w2": 210, "b1": 310, "b2": 410}

    def __init__(self, w_ratio, h_ratio, dpi, post_title):
        super(PoslinDecisionRegionsBase, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(f"Poslin Decision Regions {post_title}", 2, "\nAlter the network's\n"
                                                        "parameters by dragging\nthe slide bars.\n\n"
                                                        "The plot updates as\nyou drag.\n\n"
                                                        "Click [Set Default] to\nrestore original values.\n\n"
                                                        "Choose the output transfer\nfunction f below.",
                          PACKAGE_PATH + "Chapters/2_D/Logo_Ch_2.svg", PACKAGE_PATH + "Figures/poslinNet2Ddemo.svg", 2,
                          description_coords=(535, 130, 450, 220), icon_coords=(45, 78, 420, 168), icon_rescale=True)

        # Default values for network parameters
        self.default_w1 = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
        self.default_w2 = [[-1], [-1], [-1], [-1]]
        self.default_b1 = [[-1], [3], [1], [1]]
        self.default_b2 = [[5]]

        # Column headers, centered over each bracket
        for name, text in (("w1", "W1"), ("w2", "W2"), ("b1", "b1"), ("b2", "b2")):
            x, w = self._bracket_span(name)
            self.make_label("label_" + name, text, (x, self.HEADER_Y, w, 30), font_size=18)
            getattr(self, "label_" + name).setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Sliders laid out like the matrices they represent. Each entry is (slider, label).
        self.sliders_w1 = [[self._make_param_slider(f"slider_w1_{i}{j}", self.default_w1[i][j],
                                                    self.COL_X[f"w1_{j}"], self._row_y(i))
                            for j in range(2)] for i in range(4)]
        self.sliders_w2 = [[self._make_param_slider(f"slider_w2_{i}0", self.default_w2[i][0],
                                                    self.COL_X["w2"], self._row_y(i))] for i in range(4)]
        self.sliders_b1 = [[self._make_param_slider(f"slider_b1_{i}0", self.default_b1[i][0],
                                                    self.COL_X["b1"], self._row_y(i))] for i in range(4)]
        self.sliders_b2 = [[self._make_param_slider("slider_b2_00", self.default_b2[0][0],
                                                    self.COL_X["b2"], self._row_y(0))]]
        self.all_sliders = [s for mat in (self.sliders_w1, self.sliders_w2, self.sliders_b1, self.sliders_b2)
                            for row in mat for s in row]

        # Set Default button
        self.make_button("default_button", "Set Default",
                         (self.x_chapter_button, 430, self.w_chapter_button, self.h_chapter_button),
                         self.set_default_values)

        self.combobox_funcs = [self.poslin, self.hardlim, self.hardlims, self.purelin, self.satlin, self.satlins, self.logsig, self.tansig]
        self.combobox_funcs_str = ["poslin", "hardlim", "hardlims", "purelin", "satlin", "satlins", "logsig", "tansig"]
        self.make_combobox(1, self.combobox_funcs_str, (self.x_chapter_usual, 370, self.w_chapter_slider, 50), self.change_transfer_f, "label_f", "f")
        self.func1 = self.poslin

    def _row_y(self, row):
        return self.ROW_Y0 + row * self.ROW_PITCH

    def _bracket_span(self, name):
        """(x, width) of the bracket around matrix `name` ('w1', 'w2', 'b1', 'b2')."""
        pad = 4
        if name == "w1":
            x0, x1 = self.COL_X["w1_0"], self.COL_X["w1_1"]
        else:
            x0 = x1 = self.COL_X[name]
        return x0 - pad, (x1 + self.SLIDER_W + pad) - (x0 - pad)

    def _make_param_slider(self, attr_name, default, x, y):
        """Create one parameter slider with its value label just above it. Returns (slider, label)."""
        label_name = "label_" + attr_name
        self.make_slider(attr_name, QtCore.Qt.Orientation.Horizontal, (-self.SLIDER_MAX, self.SLIDER_MAX),
                         QtWidgets.QSlider.TickPosition.TicksBelow, self.SLIDER_SCALE, int(default * self.SLIDER_SCALE),
                         (x, y + self.LABEL_H, self.SLIDER_W, self.SLIDER_H), self.update_values,
                         label_name, self._fmt(default), (x, y, self.SLIDER_W, self.LABEL_H), label_font_size=12,
                         margins=(0, 0, 0, 0))
        label = getattr(self, label_name)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return getattr(self, attr_name), label

    @staticmethod
    def _fmt(value):
        return ("%.2f" % value).rstrip("0").rstrip(".") if value != int(value) else str(int(value))

    def _matrix_values(self, sliders):
        """Read a matrix-shaped list of (slider, label) pairs into a numpy array, refreshing the labels."""
        values = []
        for row in sliders:
            row_values = []
            for slider, label in row:
                v = slider.value() / self.SLIDER_SCALE
                label.setText(self._fmt(v))
                row_values.append(v)
            values.append(row_values)
        return np.array(values, dtype=float)

    def set_default_values(self):
        """Reset all sliders to their default values and update the graph once."""
        defaults = [(self.sliders_w1, self.default_w1), (self.sliders_w2, self.default_w2),
                    (self.sliders_b1, self.default_b1), (self.sliders_b2, self.default_b2)]
        for sliders, default in defaults:
            for row_s, row_d in zip(sliders, default):
                for (slider, _), value in zip(row_s, row_d):
                    slider.blockSignals(True)
                    slider.setValue(int(value * self.SLIDER_SCALE))
                    slider.blockSignals(False)
        self.update_values()

    def paintEvent(self, event):
        super(PoslinDecisionRegionsBase, self).paintEvent(event)
        painter = QtGui.QPainter()
        painter.begin(self)
        pen = QtGui.QPen(QtGui.QColor("black"), 2, QtCore.Qt.PenStyle.SolidLine)
        painter.setPen(pen)

        y1 = self.HEADER_Y + 15  # top bar level with the header text, so it reads like  ⌐ W1 ¬
        y2 = self._row_y(3) + self.LABEL_H + self.SLIDER_H + 4
        for name in ("w1", "w2", "b1", "b2"):  # W1 (4x2), W2 (4x1), b1 (4x1), b2 (1x1)
            x, w = self._bracket_span(name)
            self.paint_bracket(painter, x, y1, y2, w)

        painter.end()

    def change_transfer_f(self, idx):
        self.func1 = self.combobox_funcs[idx]
        self.graph()

    def update_values(self):
        self.w1 = self._matrix_values(self.sliders_w1)   # 4x2
        self.w2 = self._matrix_values(self.sliders_w2)   # 4x1
        self.b1 = self._matrix_values(self.sliders_b1)   # 4x1
        self.b2 = self._matrix_values(self.sliders_b2).reshape(-1)  # (1,)
        self.graph()
