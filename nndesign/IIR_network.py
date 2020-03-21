from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.signal import lfilter

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class IIRNetwork(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(IIRNetwork, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(15, 300, 500, 370))

        self.fill_chapter("IIR Network", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions_str = ["square", 'sine']
        self.comboBox1.addItems(self.comboBox1_functions_str)
        self.func1 = "square"
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 490 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 520 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.comboBox2 = QtWidgets.QComboBox(self)
        self.freq = 1 / 12
        self.comboBox2_divs = ["1/16", '1/14', '1/12', '1/10', '1/8']
        self.comboBox2.addItems(self.comboBox2_divs)
        self.comboBox2.setCurrentIndex(2)
        self.label_div = QtWidgets.QLabel(self)
        self.label_div.setText("frequency")
        self.label_div.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_div.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 560 * self.h_ratio,
                                   150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 590 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox2)
        self.wid2.setLayout(self.layout2)

        self.comboBox3 = QtWidgets.QComboBox(self)
        self.autoscale = False
        self.comboBox3.addItems(["No", "Yes"])
        self.label_autoscale = QtWidgets.QLabel(self)
        self.label_autoscale.setText("Autoscale")
        self.label_autoscale.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_autoscale.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 420 * self.h_ratio,
                                         150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 450 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox3)
        self.wid2.setLayout(self.layout2)

        self.label_w0 = QtWidgets.QLabel(self)
        self.label_w0.setText("iW(0): 0.5")
        self.label_w0.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w0.setGeometry(self.x_chapter_slider_label * self.w_ratio, 200 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_w0 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w0.setRange(-20, 20)
        self.slider_w0.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w0.setTickInterval(1)
        self.slider_w0.setValue(5)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 230 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_w0)
        self.wid3.setLayout(self.layout3)

        self.label_w1 = QtWidgets.QLabel(self)
        self.label_w1.setText("lW(1): -0.5")
        self.label_w1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 270 * self.h_ratio, 150 * self.w_ratio,
                                  100 * self.h_ratio)
        self.slider_w1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1.setRange(-20, 20)
        self.slider_w1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1.setTickInterval(1)
        self.slider_w1.setValue(-5)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 300 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_w1)
        self.wid3.setLayout(self.layout3)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.comboBox2.currentIndexChanged.connect(self.change_freq)
        self.comboBox3.currentIndexChanged.connect(self.change_autoscale)
        self.slider_w0.valueChanged.connect(self.graph)
        self.slider_w1.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot
        if not self.autoscale:
            a.set_xlim(0, 25)
            a.set_ylim(-6, 6)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        # a.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.plot(np.linspace(0, 25, 50), [0] * 50, color="red", linestyle="--", linewidth=0.2)
        # a.plot(np.linspace(-2, 2, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        # a.set_xlabel("$p$")
        # a.xaxis.set_label_coords(1, -0.025)
        # a.set_ylabel("$a$")
        # a.yaxis.set_label_coords(-0.025, 1)

        if self.func1 == "square":
            if self.freq == 1 / 16:
                p = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
            elif self.freq == 1 / 14:
                p = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
            elif self.freq == 1 / 12:
                p = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
            elif self.freq == 1 / 10:
                p = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
            elif self.freq == 1 / 8:
                p = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
        else:
            p = np.sin(np.arange(0, 24, 1) * 2 * np.pi * self.freq)

        weight_0 = self.slider_w0.value() / 10
        weight_1 = self.slider_w1.value() / 10
        self.label_w0.setText("iW(0): " + str(weight_0))
        self.label_w1.setText("lW(1): " + str(weight_1))

        a0, a_1, t, t1 = 0, 0, list(range(1, len(p) + 1)), list(range(len(p) + 1))
        num = np.array([weight_0])
        den = np.array([1, weight_1])
        zi = np.array([a0])
        A = lfilter(num, den, p, zi=zi)

        a.scatter(t, p, color="white", marker="o", edgecolor="red")
        a.scatter(t1, [a0] + list(A[0]), color="blue", marker=".", s=[1]*len(t1))
        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-4, 4)
        # a.set_ylim(-2, 2)
        # add grid and axes
        # a.grid(True, which='both')
        # a.axhline(y=0, color='k')
        # a.axvline(x=0, color='k')
        self.canvas.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions_str[idx]
        self.graph()

    def change_freq(self, idx):
        self.freq = eval(self.comboBox2_divs[idx])
        self.graph()

    def change_autoscale(self, idx):
        self.autoscale = True if idx == 1 else False
        self.graph()
