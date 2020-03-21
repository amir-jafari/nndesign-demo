from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.signal import lfilter

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class DynamicDerivatives(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(DynamicDerivatives, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Dynamic Derivatives", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 300 * self.h_ratio, 250 * self.w_ratio, 200 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(250 * self.w_ratio, 300 * self.h_ratio, 250 * self.w_ratio, 200 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)

        self.figure3 = Figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(15 * self.w_ratio, 480 * self.h_ratio, 250 * self.w_ratio, 200 * self.h_ratio)
        self.layout3.addWidget(self.canvas3)
        self.wid3.setLayout(self.layout3)

        self.figure4 = Figure()
        self.canvas4 = FigureCanvas(self.figure4)
        self.toolbar4 = NavigationToolbar(self.canvas4, self)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(250 * self.w_ratio, 480 * self.h_ratio, 250 * self.w_ratio, 200 * self.h_ratio)
        self.layout4.addWidget(self.canvas4)
        self.wid4.setLayout(self.layout4)

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

        """self.comboBox3 = QtWidgets.QComboBox(self)
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
        self.wid2.setLayout(self.layout2)"""

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
        # self.comboBox3.currentIndexChanged.connect(self.change_autoscale)
        self.slider_w0.valueChanged.connect(self.graph)
        self.slider_w1.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a4 = self.figure.add_subplot(1, 1, 1)
        a = self.figure2.add_subplot(1, 1, 1)
        a2 = self.figure3.add_subplot(1, 1, 1)
        a3 = self.figure4.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot
        a2.clear()
        a3.clear()
        a4.clear()
        a.set_xlim(0, 25)
        a2.set_xlim(0, 25)
        a3.set_xlim(0, 25)
        a4.set_xlim(0, 25)
        a4.set_title("Incremental Response iw + 0.1", fontsize=10)
        a.set_title("Incremental Response lw + 0.1", fontsize=10)
        a3.set_title("Derivative with respect to iw", fontsize=10)
        a2.set_title("Derivative with respect to lw", fontsize=10)
        # if not self.autoscale:
        #     a.set_xlim(0, 25)
        #     a.set_ylim(-6, 6)

        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        # a.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        a2.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        a3.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        a4.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
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
        # a.scatter(t, p, color="white", marker="o", edgecolor="red")
        a.scatter(t1, [a0] + list(A[0]), color="blue", marker="x")
        lw111 = weight_1
        iw11 = weight_0 + 0.1
        num = np.array([iw11])
        den = np.array([1, lw111])
        a1 = lfilter(num, den, p, zi=zi)
        a.scatter(t1, [a0] + list(a1[0]), color="blue", marker=".", s=[1] * len(t1))

        da_diw_0 = 0
        da_diw = lfilter(np.array([1]), den, p, zi=np.array([da_diw_0]))
        a2.scatter(t1, [da_diw_0] + list(da_diw[0]), color="white", marker="D", edgecolor="blue")
        a2.scatter(t, p, color="white", marker="s", edgecolor="black", s=[8]*len(t))

        da_dlw_0 = 0
        ad = np.array([a0] + list(A[0])[:-1])
        da_dlw = lfilter(np.array([1]), den, ad, zi=np.array([da_dlw_0]))
        a3.scatter(t1, [da_dlw_0] + list(da_dlw[0]), color="white", marker="D", edgecolor="blue")
        a3.scatter(t, ad, color="white", marker="s", edgecolor="black", s=[8]*len(t))

        a4.scatter(t1, [a0] + list(A[0]), color="blue", marker="x")
        lw111 = weight_1 + .1
        iw11 = weight_0
        num = np.array([iw11])
        den = np.array([1, lw111])
        a1 = lfilter(num, den, p, zi=zi)
        a4.scatter(t1, [a0] + list(a1[0]), color="blue", marker=".", s=[1] * len(t1))

        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-4, 4)
        # a.set_ylim(-2, 2)
        # add grid and axes
        # a.grid(True, which='both')
        # a.axhline(y=0, color='k')
        # a.axvline(x=0, color='k')
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions_str[idx]
        self.graph()

    def change_freq(self, idx):
        self.freq = eval(self.comboBox2_divs[idx])
        self.graph()

    # def change_autoscale(self, idx):
    #     self.autoscale = True if idx == 1 else False
    #     self.graph()
