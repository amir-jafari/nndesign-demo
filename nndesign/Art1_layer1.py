from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


t = np.arange(0, 0.2, 0.01)


class ART1Layer1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(ART1Layer1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(25, 90, 450, 450))

        self.fill_chapter("ART1 Layer 1", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.bp, self.bn, self.e = 1, 0, 0.1

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.2)
        self.axis.set_ylim(-0.5, 0.5)
        self.axis.plot([0] * 10, np.linspace(-0.5, 0.5, 10), color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Net inputs n1(1), n1(2)")
        self.axis.set_title("Response")
        self.lines1, self.lines2 = [], []

        self.label_input_pos = QtWidgets.QLabel(self)
        self.label_input_pos.setText("Input p(1): 0")
        self.label_input_pos.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_input_pos.setGeometry(self.x_chapter_slider_label * self.w_ratio, 200 * self.h_ratio,
                                         150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_input_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_input_pos.setRange(0, 1)
        self.slider_input_pos.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_input_pos.setTickInterval(1)
        self.slider_input_pos.setValue(0)
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
        self.label_bias_pos.setText("Bias b+: 1.00")
        self.label_bias_pos.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_bias_pos.setGeometry(self.x_chapter_slider_label * self.w_ratio, 340 * self.h_ratio,
                                        150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_bias_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bias_pos.setRange(0, 30)
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
        self.label_bias_neg.setText("Bias b-: 1.50")
        self.label_bias_neg.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_bias_neg.setGeometry(self.x_chapter_slider_label * self.w_ratio, 410 * self.h_ratio,
                                        150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_bias_neg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bias_neg.setRange(0, 30)
        self.slider_bias_neg.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_bias_neg.setTickInterval(1)
        self.slider_bias_neg.setValue(15)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(self.x_chapter_usual * self.w_ratio, 440 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout5.addWidget(self.slider_bias_neg)
        self.wid5.setLayout(self.layout5)

        self.w_11 = QtWidgets.QLineEdit()
        self.w_11.setText("1")
        self.w_11.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_11)
        self.wid3.setLayout(self.layout3)

        self.w_12 = QtWidgets.QLineEdit()
        self.w_12.setText("1")
        self.w_11.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_12)
        self.wid3.setLayout(self.layout3)

        self.w_21 = QtWidgets.QLineEdit()
        self.w_21.setText("0")
        self.w_21.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_21)
        self.wid3.setLayout(self.layout3)

        self.w_22 = QtWidgets.QLineEdit()
        self.w_22.setText("1")
        self.w_22.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.w_22)
        self.wid3.setLayout(self.layout3)

        self.clear_button = QtWidgets.QPushButton("Clear", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 580 * self.h_ratio,
                                      self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.run_button = QtWidgets.QPushButton("Update", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 540 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.graph)

        self.slider_input_pos.valueChanged.connect(self.graph)
        self.slider_input_neg.valueChanged.connect(self.graph)
        self.slider_bias_pos.valueChanged.connect(self.graph)
        self.slider_bias_neg.valueChanged.connect(self.graph)

        self.graph()

    def layer1(self, t, y):
        y = np.array([[y[0]], [y[1]]])
        y_out = np.zeros(y.shape)
        y_out[0, 0] = (-y[0, 0] + (self.bp - y[0, 0]) * (self.p[0, 0] * self.W2[0, 1]) - (y[0, 0] + self.bn)) / 0.1
        y_out[1, 0] = (-y[1, 0] + (self.bp - y[1, 0]) * (self.p[1, 0] * self.W2[1, 1]) - (y[1, 0] + self.bn)) / 0.1
        return y_out

    def graph(self):
        self.pp = self.slider_input_pos.value()
        self.pn = self.slider_input_neg.value()
        self.label_input_pos.setText("Input p(1): " + str(self.pp))
        self.label_input_neg.setText("Input p(2): " + str(self.pn))
        self.bp = self.slider_bias_pos.value() / 10
        self.bn = self.slider_bias_neg.value() / 10
        self.label_bias_pos.setText("Bias b+: " + str(round(self.bp, 2)))
        self.label_bias_neg.setText("Bias b-: " + str(round(self.bn, 2)))
        w11, w12 = float(self.w_11.text()), float(self.w_12.text())
        w21, w22 = float(self.w_21.text()), float(self.w_22.text())
        for idx, param in enumerate([w11, w12, w21, w22]):
            if param not in [0, 1]:
                if idx == 0:
                    w11 = 0
                    self.w_11.setText("0")
                elif idx == 1:
                    w12 = 0
                    self.w_12.setText("0")
                elif idx == 2:
                    w21 = 0
                    self.w_21.setText("0")
                else:
                    w22 = 0
                    self.w_22.setText("0")
        self.W2 = np.array([[w11, w21], [w12, w22]])
        self.p = np.array([[self.pp], [self.pn]])
        r1 = ode(self.layer1).set_integrator("zvode")
        r1.set_initial_value(np.array([[0], [0]]), 0)
        t1 = 0.2
        dt = 0.01
        out_1, out_2 = [], []
        while r1.successful() and r1.t < t1:
            out = r1.integrate(r1.t + dt)
            out_1.append(out[0, 0])
            out_2.append(out[1, 0])
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
