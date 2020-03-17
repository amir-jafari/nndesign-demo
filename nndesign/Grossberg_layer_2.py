from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


t = np.arange(0, 0.51, 0.01)


class GrossbergLayer2(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(GrossbergLayer2, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(25, 90, 450, 450))

        self.fill_chapter("Grossberg Layer 1", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.bp, self.bn, self.e = 1, 0, 0.1

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.51)
        self.axis.set_ylim(0, 1)
        self.axis.plot([0] * 10, np.linspace(0, 1, 10), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot([0.25] * 10, np.linspace(0, 1, 10), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot(np.linspace(0, 0.5, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Net inputs n2(1), n2(2)")
        self.axis.set_title("Response")
        self.lines1, self.lines2 = [], []

        self.label_input_pos = QtWidgets.QLabel(self)
        self.label_input_pos.setText("Input a1(1): 1.00")
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
        self.label_input_neg.setText("Input a1(2): 0.00")
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

        self.w_11 = QtWidgets.QLineEdit()
        self.w_11.setText("0.9")
        self.w_11.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_11)
        self.wid3.setLayout(self.layout3)

        self.w_12 = QtWidgets.QLineEdit()
        self.w_12.setText("0.45")
        self.w_11.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_12)
        self.wid3.setLayout(self.layout3)

        self.w_21 = QtWidgets.QLineEdit()
        self.w_21.setText("0.45")
        self.w_21.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_21)
        self.wid3.setLayout(self.layout3)

        self.w_22 = QtWidgets.QLineEdit()
        self.w_22.setText("0.9")
        self.w_22.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_22)
        self.wid3.setLayout(self.layout3)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.f2, self.purelin, self.f3, self.f4]
        self.comboBox1_functions_str = ['(10n^2)/(1 + n^2)', "purelin", '10n^2', '1 - exp(-n)']
        self.comboBox1.addItems(self.comboBox1_functions_str)
        self.func1 = self.f2
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("Transfer function")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12))
        self.label_f.setGeometry(self.x_chapter_slider_label * self.w_ratio, 480 * self.h_ratio,
                                 150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)
        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)

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

        self.run_button = QtWidgets.QPushButton("Update", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.graph)

        self.slider_input_pos.valueChanged.connect(self.graph)
        self.slider_input_neg.valueChanged.connect(self.graph)
        self.do_graph = True

        self.graph()

    @staticmethod
    def f2(n):
        return 10 * n ** 2 / (1 + n ** 2)

    @staticmethod
    def f3(n):
        return 10 * n ** 2

    @staticmethod
    def f4(n):
        return 1 - np.exp(-n)

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions[idx]
        self.graph()

    def layer2(self, t, y):
        i1 = np.dot(self.W2[0, :], self.p).item()
        i2 = np.dot(self.W2[1, :], self.p).item()
        y = np.array([[y[0]], [y[1]]])
        a = self.func1(y)
        y_out = np.zeros(y.shape)
        y_out[0, 0] = (-y[0, 0] + (self.bp - y[0, 0]) * (a[0, 0] + i1) - (y[0, 0] + self.bn) * a[1, 0]) / self.e
        y_out[1, 0] = (-y[1, 0] + (self.bp - y[1, 0]) * (a[1, 0] + i2) - (y[1, 0] + self.bn) * a[0, 0]) / self.e
        return y_out

    def graph(self):
        if self.do_graph:
            self.pp = self.slider_input_pos.value() / 10
            self.pn = self.slider_input_neg.value() / 10
            self.label_input_pos.setText("Input a1(1): " + str(round(self.pp, 2)))
            self.label_input_neg.setText("Input a1(2): " + str(round(self.pn, 2)))
            w11, w12 = float(self.w_11.text()), float(self.w_12.text())
            w21, w22 = float(self.w_21.text()), float(self.w_22.text())
            self.W2 = np.array([[w11, w12], [w21, w22]])
            self.p = np.array([[self.pp], [self.pn]])
            r1 = ode(self.layer2).set_integrator("zvode")
            r1.set_initial_value(np.array([[0], [0]]), 0)
            t1 = 0.26
            dt = 0.01
            out_11, out_21 = [], []
            while r1.successful() and r1.t < t1:
                out = r1.integrate(r1.t + dt)
                out_11.append(out[0, 0])
                out_21.append(out[1, 0])
            self.p = np.array([[0], [0]])
            r2 = ode(self.layer2).set_integrator("zvode")
            r2.set_initial_value(np.array([[out_11[-1]], [out_21[-1]]]), 0.26)
            t2 = 0.51
            out_12, out_22 = [], []
            while r2.successful() and r2.t < t2:
                out = r2.integrate(r2.t + dt)
                out_12.append(out[0, 0])
                out_22.append(out[1, 0])
            out_1, out_2 = out_11 + out_12, out_21 + out_22
            while len(self.lines1) > 1:
                self.lines1.pop(0).remove()
            while len(self.lines2) > 1:
                self.lines2.pop(0).remove()
            for line in self.lines1:
                # line.set_color("gray")
                line.set_alpha(0.2)
            for line in self.lines2:
                # line.set_color("gray")
                line.set_alpha(0.2)
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
        self.do_graph = True
        self.slider_input_neg.setValue(np.random.uniform(0, 1) * 100)
