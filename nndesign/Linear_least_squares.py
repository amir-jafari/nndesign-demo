from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


randseq = [-0.7616, -1.0287, 0.5348, -0.8102, -1.1690, 0.0419, 0.8944, 0.5460, -0.9345, 0.0754,
           -0.7616, -1.0287, 0.5348, -0.8102, -1.1690, 0.0419, 0.8944, 0.5460, -0.9345, 0.0754]


class LinearLeastSquares(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(LinearLeastSquares, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(20, 100, 450, 300))

        self.fill_chapter("Linear Least Squares", 17, " Alter the network's parameters\n by dragging the slide bars",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_info=False, show_pic=False)  # TODO: Change icons

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(20 * self.w_ratio, 390 * self.h_ratio, 450 * self.w_ratio, 140 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)
        self.canvas2.draw()

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.addItems(["Yes", "No"])
        self.comboBox1.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 640 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.auto_bias = True
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("Auto Bias")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 600 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 620 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.label_w1_1 = QtWidgets.QLabel(self)
        self.label_w1_1.setText("w1(1, 1): -2")
        self.label_w1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 120 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_1.setRange(-20, -10)
        self.slider_w1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_1.setTickInterval(1)
        self.slider_w1_1.setValue(-20)
        self.wid_w1_1 = QtWidgets.QWidget(self)
        self.layout_w1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 140 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1_1.addWidget(self.slider_w1_1)
        self.wid_w1_1.setLayout(self.layout_w1_1)

        self.label_w1_2 = QtWidgets.QLabel(self)
        self.label_w1_2.setText("Hidden Neurons: 5")
        self.label_w1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 170 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_2.setRange(2, 9)
        self.slider_w1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_2.setTickInterval(1)
        self.slider_w1_2.setValue(5)
        self.wid_w1_2 = QtWidgets.QWidget(self)
        self.layout_w1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 200 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1_2.addWidget(self.slider_w1_2)
        self.wid_w1_2.setLayout(self.layout_w1_2)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("b: 1.67")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(self.x_chapter_slider_label * self.w_ratio, 240 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(10, 1000)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(167)
        self.wid_b = QtWidgets.QWidget(self)
        self.layout_b = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b.setGeometry(self.x_chapter_usual * self.w_ratio, 270 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b.addWidget(self.slider_b)
        self.wid_b.setLayout(self.layout_b)

        self.label_b1_2 = QtWidgets.QLabel(self)
        self.label_b1_2.setText("Number of Points: 10")
        self.label_b1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 310 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_2.setRange(2, 20)
        self.slider_b1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_2.setTickInterval(1)
        self.slider_b1_2.setValue(10)
        self.wid_b1_2 = QtWidgets.QWidget(self)
        self.layout_b1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 340 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b1_2.addWidget(self.slider_b1_2)
        self.wid_b1_2.setLayout(self.layout_b1_2)

        self.label_w2_1 = QtWidgets.QLabel(self)
        self.label_w2_1.setText("Regularization: 0.0")
        self.label_w2_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 380 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w2_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_1.setRange(0, 10)
        self.slider_w2_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_1.setTickInterval(1)
        self.slider_w2_1.setValue(0)

        self.wid_w2_1 = QtWidgets.QWidget(self)
        self.layout_w2_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_1.setGeometry(self.x_chapter_usual * self.w_ratio, 410 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w2_1.addWidget(self.slider_w2_1)
        self.wid_w2_1.setLayout(self.layout_w2_1)

        self.label_w2_2 = QtWidgets.QLabel(self)
        self.label_w2_2.setText("Stdev Noise: 0.0")
        self.label_w2_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 450 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w2_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_2.setRange(0, 10)
        self.slider_w2_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_2.setTickInterval(1)
        self.slider_w2_2.setValue(0)
        self.wid_w2_2 = QtWidgets.QWidget(self)
        self.layout_w2_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_2.setGeometry(self.x_chapter_usual * self.w_ratio, 480 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w2_2.addWidget(self.slider_w2_2)
        self.wid_w2_2.setLayout(self.layout_w2_2)

        self.label_b2 = QtWidgets.QLabel(self)
        self.label_b2.setText("Function Frequency: 0.50")
        self.label_b2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b2.setGeometry((self.x_chapter_slider_label - 30) * self.w_ratio, 520 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b2.setRange(25, 100)
        self.slider_b2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b2.setTickInterval(1)
        self.slider_b2.setValue(50)
        self.wid_b2 = QtWidgets.QWidget(self)
        self.layout_b2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b2.setGeometry(self.x_chapter_usual * self.w_ratio, 550 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b2.addWidget(self.slider_b2)
        self.wid_b2.setLayout(self.layout_b2)

        self.label_fp = QtWidgets.QLabel(self)
        self.label_fp.setText("Function Phase: 90")
        self.label_fp.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_fp.setGeometry(220 * self.w_ratio, 540 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_fp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_fp.setRange(0, 360)
        self.slider_fp.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_fp.setTickInterval(1)
        self.slider_fp.setValue(90)
        self.wid_fp = QtWidgets.QWidget(self)
        self.layout_fp = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_fp.setGeometry(200 * self.w_ratio, 570 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_fp.addWidget(self.slider_fp)
        self.wid_fp.setLayout(self.layout_fp)

        self.comboBox1.currentIndexChanged.connect(self.change_auto_bias)
        self.slider_w1_1.valueChanged.connect(self.graph)
        self.slider_w1_2.valueChanged.connect(self.graph)
        self.slider_b.valueChanged.connect(self.graph)
        self.slider_b1_2.valueChanged.connect(self.graph)
        self.slider_w2_1.valueChanged.connect(self.graph)
        self.slider_w2_2.valueChanged.connect(self.graph)
        self.slider_b2.valueChanged.connect(self.graph)
        self.slider_fp.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        axis = self.figure.add_subplot(1, 1, 1)
        axis.clear()  # Clear the plot
        axis.set_xlim(-2, 2)
        axis.set_ylim(-2, 4)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        axis.set_xticks([-2, -1, 0, 1])
        axis.set_yticks([-2, -1, 0, 1, 2, 3])
        axis.plot(np.linspace(-2, 2, 10), [0]*10, color="black", linestyle="--", linewidth=0.2)
        axis.set_xlabel("$p$")
        axis.xaxis.set_label_coords(1, -0.025)
        axis.set_ylabel("$a^2$")
        axis.yaxis.set_label_coords(-0.025, 1)

        axis2 = self.figure2.add_subplot(1, 1, 1)
        axis2.clear()  # Clear the plot
        axis2.set_xlim(-2, 2)
        axis2.set_ylim(0, 1)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        axis2.set_xticks([-2, -1, 0, 1])
        axis2.set_yticks([0, 0.5])
        axis2.set_xlabel("$p$")
        axis2.xaxis.set_label_coords(1, -0.025)
        axis2.set_ylabel("$a^1$")
        axis2.yaxis.set_label_coords(-0.025, 1)

        # ax.set_xticks(major_ticks)
        # ax.set_xticks(minor_ticks, minor=True)
        # ax.set_yticks(major_ticks)
        # ax.set_yticks(minor_ticks, minor=True)
        #
        # # And a corresponding grid
        # ax.grid(which='both')
        #
        # # Or if you want different settings for the grids:
        # ax.grid(which='minor', alpha=0.2)
        # ax.grid(which='major', alpha=0.5)

        w1_1 = self.slider_w1_1.value() / 10
        bias = self.slider_b.value() / 100
        S1 = self.slider_w1_2.value()
        n_points = self.slider_b1_2.value()
        ro = self.slider_w2_1.value() / 10
        sigma = self.slider_w2_2.value() / 10
        freq = self.slider_b2.value() / 100
        phase = self.slider_fp.value()

        self.label_w1_1.setText("w1(1, 1): " + str(w1_1))
        self.label_b.setText("b: " + str(round(bias, 2)))
        self.label_w1_2.setText("Hidden Neurons: " + str(S1))
        self.label_b1_2.setText("Number of Points: " + str(n_points))
        self.label_w2_1.setText("Regularization: " + str(ro))
        self.label_w2_2.setText("Stdev Noise: " + str(sigma))
        self.label_b2.setText("Function Frequency: " + str(freq))
        self.label_fp.setText("Function Phase: " + str(phase))

        d1 = (2 - -2) / (n_points - 1)
        p = np.arange(-2, 2 + 0.0001, d1)
        t = np.sin(2 * np.pi * (freq * p + phase / 360)) + 1 + sigma * np.array(randseq[:len(p)])
        delta = (2 - -2) / (S1 - 1)
        if self.auto_bias:
            bias = 1.6652 / delta
            self.slider_b.setValue(bias * 100)
            self.label_b.setText("b: " + str(bias))
        total = 2 - -2
        W1 = (np.arange(-2, 2 + 0.0001, delta) + w1_1 - -2).T.reshape(-1, 1)
        b1 = bias * np.ones(W1.shape)
        Q = len(p)
        pp = np.repeat(p.reshape(1, -1), S1, 0)
        n1 = np.abs(pp - np.dot(W1, np.ones((1, Q)))) * np.dot(b1, np.ones((1, Q)))
        a1 = np.exp(-n1 ** 2)
        Z = np.vstack((a1, np.ones((1, Q))))
        x = np.dot(np.linalg.pinv(np.dot(Z, Z.T) + ro * np.eye(S1 + 1)), np.dot(Z, t.T))
        W2, b2 = x[:-1].T, x[-1]
        a2 = np.dot(W2, a1) + b2
        p2 = np.arange(-2, 2 + total / 100, total / 100)
        Q2 = len(p2)
        pp2 = np.repeat(p2.reshape(1, -1), S1, 0)
        n12 = np.abs(pp2 - np.dot(W1, np.ones((1, Q2)))) * np.dot(b1, np.ones((1, Q2)))
        a12 = np.exp(-n12 ** 2)
        a22 = np.dot(W2, a12) + b2
        t_exact = np.sin(2 * np.pi * (freq * p2 + phase / 360)) + 1
        temp = np.vstack((np.dot(W2.T, np.ones((1, Q2)) * a12), b2 * np.ones((1, Q2))))

        axis.scatter(p, t, color="white", edgecolor="black")
        for i in range(len(temp)):
            axis.plot(p2, temp[i], linestyle="--", color="black", linewidth=0.5)
        axis.plot(p2, t_exact, color="blue", linewidth=2)
        axis.plot(p2, a22, color="red", linewidth=1)
        for i in range(len(a12)):
            axis2.plot(p2, a12[i], color="black")

        self.canvas.draw()
        self.canvas2.draw()

    def change_auto_bias(self, idx):
        self.auto_bias = idx == 0
        self.graph()
