from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


t = np.arange(0, 5.1, 0.1)


class GrossbergLayer1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(GrossbergLayer1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(25, 150, 450, 450))

        self.fill_chapter("Grossberg Layer 1", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 5)
        self.axis.set_ylim(-5, 5)
        self.axis.plot([0] * 20, np.linspace(-5, 5, 20), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot(np.linspace(0, 5, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Net inputs n(1), n(2)")
        self.axis.set_title("Response")
        self.lines1, self.lines2 = [], []

        self.label_input_pos = QtWidgets.QLabel(self)
        self.label_input_pos.setText("Input p(1): 1.00")
        self.label_input_pos.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_input_pos.setGeometry(self.x_chapter_slider_label * self.w_ratio, 200 * self.h_ratio,
                                         150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_input_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_input_pos.setRange(0, 100)
        self.slider_input_pos.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_input_pos.setTickInterval(1)
        self.slider_input_pos.setValue(10)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 230 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_input_pos)
        self.wid3.setLayout(self.layout3)

        self.label_input_neg = QtWidgets.QLabel(self)
        self.label_input_neg.setText("Input p(2): 0.00")
        self.label_input_neg.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_input_neg.setGeometry(self.x_chapter_slider_label * self.w_ratio, 270 * self.h_ratio,
                                         150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_input_neg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_input_neg.setRange(0, 100)
        self.slider_input_neg.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_input_neg.setTickInterval(1)
        self.slider_input_neg.setValue(0)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 300 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_input_neg)
        self.wid4.setLayout(self.layout4)

        self.label_bias_pos = QtWidgets.QLabel(self)
        self.label_bias_pos.setText("Bias b+: 1.00")
        self.label_bias_pos.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_bias_pos.setGeometry(self.x_chapter_slider_label * self.w_ratio, 340 * self.h_ratio,
                                        150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_bias_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bias_pos.setRange(0, 50)
        self.slider_bias_pos.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_bias_pos.setTickInterval(1)
        self.slider_bias_pos.setValue(10)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(self.x_chapter_usual * self.w_ratio, 370 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout5.addWidget(self.slider_bias_pos)
        self.wid5.setLayout(self.layout5)

        self.label_bias_neg = QtWidgets.QLabel(self)
        self.label_bias_neg.setText("Bias b-: 0.00")
        self.label_bias_neg.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_bias_neg.setGeometry(self.x_chapter_slider_label * self.w_ratio, 410 * self.h_ratio,
                                        150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_bias_neg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bias_neg.setRange(0, 50)
        self.slider_bias_neg.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_bias_neg.setTickInterval(1)
        self.slider_bias_neg.setValue(0)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(self.x_chapter_usual * self.w_ratio, 440 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout5.addWidget(self.slider_bias_neg)
        self.wid5.setLayout(self.layout5)

        self.label_tcte = QtWidgets.QLabel(self)
        self.label_tcte.setText("Time Constant: 1.00")
        self.label_tcte.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_tcte.setGeometry(self.x_chapter_slider_label * self.w_ratio, 480 * self.h_ratio, 600 * self.w_ratio, 100 * self.h_ratio)
        self.slider_tcte = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tcte.setRange(1, 50)
        self.slider_tcte.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_tcte.setTickInterval(1)
        self.slider_tcte.setValue(10)
        self.wid7 = QtWidgets.QWidget(self)
        self.layout7 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid7.setGeometry(self.x_chapter_usual * self.w_ratio, 510 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout7.addWidget(self.slider_tcte)
        self.wid7.setLayout(self.layout7)

        self.clear_button = QtWidgets.QPushButton("Clear", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 580 * self.h_ratio,
                                      self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.random_button = QtWidgets.QPushButton("Random", self)
        self.random_button.setStyleSheet("font-size:13px")
        self.random_button.setGeometry(self.x_chapter_button * self.w_ratio, 630 * self.h_ratio,
                                       self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.random_button.clicked.connect(self.on_random)

        self.slider_input_pos.valueChanged.connect(self.graph)
        self.slider_input_neg.valueChanged.connect(self.graph)
        self.slider_bias_pos.valueChanged.connect(self.graph)
        self.slider_bias_neg.valueChanged.connect(self.graph)
        self.slider_tcte.valueChanged.connect(self.graph)
        self.do_graph = True

        self.graph()

    def layer1(self, t, y):
        return [(-y[0] + (self.bp - y[0]) * self.pp - (y[0] + self.bn) * self.pn) / self.e,
                (-y[1] + (self.bp - y[1]) * self.pn - (y[1] + self.bn) * self.pp) / self.e]

    def graph(self):
        if self.do_graph:
            self.pp = self.slider_input_pos.value() / 10
            self.pn = self.slider_input_neg.value() / 10
            self.bp = self.slider_bias_pos.value() / 10
            self.bn = self.slider_bias_neg.value() / 10
            self.e = self.slider_tcte.value() / 10
            self.label_input_pos.setText("Input p(1): " + str(round(self.pp, 2)))
            self.label_input_neg.setText("Input p(2): " + str(round(self.pn, 2)))
            self.label_bias_pos.setText("Bias b+: " + str(round(self.bp, 2)))
            self.label_bias_neg.setText("Bias b- " + str(round(self.bn, 2)))
            self.label_tcte.setText("Tme Constant: " + str(round(self.e, 2)))
            r = ode(self.layer1).set_integrator("zvode")
            r.set_initial_value([0, 0], 0)
            t1 = 5
            dt = 0.1
            out_1, out_2 = [], []
            while r.successful() and r.t < t1:
                out = r.integrate(r.t + dt)
                out_1.append(out[0])
                out_2.append(out[1])
            while len(self.lines1) > 1:
                self.lines1.pop(0).remove()
            while len(self.lines2) > 1:
                self.lines2.pop(0).remove()
            for line in self.lines1:
                # line.set_color("gray")
                line.set_alpha(0.5)
            for line in self.lines2:
                # line.set_color("gray")
                line.set_alpha(0.5)
            self.lines1.append(self.axis.plot(t, out_1, color="red")[0])
            self.lines2.append(self.axis.plot(t, out_2, color="green")[0])
            self.canvas.draw()

    def on_clear(self):
        while len(self.lines1) > 1:
            self.lines1.pop(0).remove()
        while len(self.lines2) > 1:
            self.lines2.pop(0).remove()
        self.canvas.draw()

    def on_random(self):
        self.do_graph = False
        self.slider_input_pos.setValue(np.random.uniform(0, 1) * 100)
        self.slider_input_neg.setValue(np.random.uniform(0, 1) * 100)
        self.slider_bias_pos.setValue(np.random.uniform(0, 1) * 50)
        self.slider_bias_neg.setValue(np.random.uniform(0, 1) * 50)
        self.do_graph = True
        self.slider_tcte.setValue(np.random.uniform(0, 1) * 50)
