from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


t = np.arange(0, 0.21, 0.01)


class OrientingSubsystem(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(OrientingSubsystem, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(25, 150, 450, 450))

        self.fill_chapter("Orienting Subsystem", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.21)
        self.axis.set_ylim(-1, 1)
        self.axis.plot(np.linspace(0, 0.21, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Reset a0")
        self.axis.set_title("Response")
        self.lines = []

        self.label_input_pos = QtWidgets.QLabel(self)
        self.label_input_pos.setText("Input p(1): 1")
        self.label_input_pos.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_input_pos.setGeometry(self.x_chapter_slider_label * self.w_ratio, 200 * self.h_ratio,
                                         150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_input_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_input_pos.setRange(0, 1)
        self.slider_input_pos.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_input_pos.setTickInterval(1)
        self.slider_input_pos.setValue(1)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 230 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_input_pos)
        self.wid3.setLayout(self.layout3)

        self.label_input_neg = QtWidgets.QLabel(self)
        self.label_input_neg.setText("Input p(2): 1")
        self.label_input_neg.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_input_neg.setGeometry(self.x_chapter_slider_label * self.w_ratio, 270 * self.h_ratio,
                                         150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_input_neg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_input_neg.setRange(0, 1)
        self.slider_input_neg.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_input_neg.setTickInterval(1)
        self.slider_input_neg.setValue(1)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 300 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_input_neg)
        self.wid4.setLayout(self.layout4)

        self.label_bias_pos = QtWidgets.QLabel(self)
        self.label_bias_pos.setText("Input a1(1): 1")
        self.label_bias_pos.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_bias_pos.setGeometry(self.x_chapter_slider_label * self.w_ratio, 340 * self.h_ratio,
                                        150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_bias_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bias_pos.setRange(0, 1)
        self.slider_bias_pos.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_bias_pos.setTickInterval(1)
        self.slider_bias_pos.setValue(1)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(self.x_chapter_usual * self.w_ratio, 370 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout5.addWidget(self.slider_bias_pos)
        self.wid5.setLayout(self.layout5)

        self.label_bias_neg = QtWidgets.QLabel(self)
        self.label_bias_neg.setText("Input a1(2): 0")
        self.label_bias_neg.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_bias_neg.setGeometry(self.x_chapter_slider_label * self.w_ratio, 410 * self.h_ratio,
                                        150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_bias_neg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bias_neg.setRange(0, 1)
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
        self.label_tcte.setText("+W0 Elements: 3.00")
        self.label_tcte.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_tcte.setGeometry(self.x_chapter_slider_label * self.w_ratio, 480 * self.h_ratio, 600 * self.w_ratio, 100 * self.h_ratio)
        self.slider_tcte = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tcte.setRange(1, 50)
        self.slider_tcte.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_tcte.setTickInterval(1)
        self.slider_tcte.setValue(30)
        self.wid7 = QtWidgets.QWidget(self)
        self.layout7 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid7.setGeometry(self.x_chapter_usual * self.w_ratio, 510 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout7.addWidget(self.slider_tcte)
        self.wid7.setLayout(self.layout7)

        self.label_tcte1 = QtWidgets.QLabel(self)
        self.label_tcte1.setText("-W0 Elements: 4.00")
        self.label_tcte1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_tcte1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 540 * self.h_ratio, 600 * self.w_ratio,
                                    100 * self.h_ratio)
        self.slider_tcte1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tcte1.setRange(1, 50)
        self.slider_tcte1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_tcte1.setTickInterval(1)
        self.slider_tcte1.setValue(40)
        self.wid7 = QtWidgets.QWidget(self)
        self.layout7 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid7.setGeometry(self.x_chapter_usual * self.w_ratio, 570 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout7.addWidget(self.slider_tcte1)
        self.wid7.setLayout(self.layout7)

        self.clear_button = QtWidgets.QPushButton("Clear", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 630 * self.h_ratio,
                                      self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.slider_input_pos.valueChanged.connect(self.graph)
        self.slider_input_neg.valueChanged.connect(self.graph)
        self.slider_bias_pos.valueChanged.connect(self.graph)
        self.slider_bias_neg.valueChanged.connect(self.graph)
        self.slider_tcte.valueChanged.connect(self.graph)
        self.slider_tcte1.valueChanged.connect(self.graph)

        self.graph()

    def shunt(self, t, y):
        return (-y + (1 - y) * self.A * (self.p[0, 0] + self.p[1, 0]) - (y + 1) * self.B * (self.a[0, 0] + self.a[1, 0])) / 0.1

    def graph(self):
        self.pp = self.slider_input_pos.value()
        self.pn = self.slider_input_neg.value()
        self.bp = self.slider_bias_pos.value()
        self.bn = self.slider_bias_neg.value()
        self.A = self.slider_tcte.value() / 10
        self.B = self.slider_tcte1.value() / 10
        self.label_input_pos.setText("Input p(1): " + str(self.pp))
        self.label_input_neg.setText("Input p(2): " + str(self.pn))
        self.label_bias_pos.setText("Input a1(1): " + str(self.bp))
        self.label_bias_neg.setText("Input a1(2): " + str(self.bn))
        self.label_tcte.setText("+W0 Elements: " + str(round(self.A, 2)))
        self.label_tcte1.setText("-W0 Elements: " + str(round(self.B, 2)))
        self.p = np.array([[self.pp], [self.pn]])
        self.a = np.array([[self.bp], [self.bn]])
        r = ode(self.shunt).set_integrator("zvode")
        r.set_initial_value(np.array([0, 0]), 0)
        t1 = 0.21
        dt = 0.01
        out = []
        while r.successful() and r.t < t1:
            out.append(r.integrate(r.t + dt)[0])
        while len(self.lines) > 3:
            self.lines.pop(0).remove()
        for line in self.lines:
            line.set_color("gray")
            line.set_alpha(0.5)
        self.lines.append(self.axis.plot(t, out, color="red")[0])
        self.canvas.draw()

    def on_clear(self):
        while len(self.lines) > 1:
            self.lines.pop(0).remove()
        self.canvas.draw()
