from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class NetworkFunctionRadial(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(NetworkFunctionRadial, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(10, 400, 500, 250))

        self.fill_chapter("Network Function", 11, " Alter the network's parameters\n by dragging the slide bars",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_info=False, show_pic=False)  # TODO: Change icons

        # self.label_eq = QtWidgets.QLabel(self)
        # self.label_eq.setText("a = purelin(w2 * tansig(w1 * p + b1) + b2))")
        # self.label_eq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_eq.setGeometry(180 * self.w_ratio, 270 * self.h_ratio, (self.w_chapter_slider + 100) * self.w_ratio, 50 * self.h_ratio)

        self.label_w1_1 = QtWidgets.QLabel(self)
        self.label_w1_1.setText("w1_1")
        self.label_w1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 120 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_1.setRange(-40, 40)
        self.slider_w1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_1.setTickInterval(10)
        self.slider_w1_1.setValue(-10)

        self.wid_w1_1 = QtWidgets.QWidget(self)
        self.layout_w1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 140 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1_1.addWidget(self.slider_w1_1)
        self.wid_w1_1.setLayout(self.layout_w1_1)

        self.label_w1_2 = QtWidgets.QLabel(self)
        self.label_w1_2.setText("w1_2")
        self.label_w1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 170 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_2.setRange(-40, 40)
        self.slider_w1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_2.setTickInterval(10)
        self.slider_w1_2.setValue(10)

        self.wid_w1_2 = QtWidgets.QWidget(self)
        self.layout_w1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 200 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1_2.addWidget(self.slider_w1_2)
        self.wid_w1_2.setLayout(self.layout_w1_2)

        self.label_b1_1 = QtWidgets.QLabel(self)
        self.label_b1_1.setText("b1_1")
        self.label_b1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 240 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_1.setRange(-40, 40)
        self.slider_b1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_1.setTickInterval(10)
        self.slider_b1_1.setValue(20)

        self.wid_b1_1 = QtWidgets.QWidget(self)
        self.layout_b1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 270 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b1_1.addWidget(self.slider_b1_1)
        self.wid_b1_1.setLayout(self.layout_b1_1)

        self.label_b1_2 = QtWidgets.QLabel(self)
        self.label_b1_2.setText("b1_2")
        self.label_b1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 310 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_2.setRange(-40, 40)
        self.slider_b1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_2.setTickInterval(10)
        self.slider_b1_2.setValue(20)

        self.wid_b1_2 = QtWidgets.QWidget(self)
        self.layout_b1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 340 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b1_2.addWidget(self.slider_b1_2)
        self.wid_b1_2.setLayout(self.layout_b1_2)

        self.label_w2_1 = QtWidgets.QLabel(self)
        self.label_w2_1.setText("w2_1")
        self.label_w2_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 380 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w2_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_1.setRange(-20, 20)
        self.slider_w2_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_1.setTickInterval(10)
        self.slider_w2_1.setValue(10)

        self.wid_w2_1 = QtWidgets.QWidget(self)
        self.layout_w2_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_1.setGeometry(self.x_chapter_usual * self.w_ratio, 410 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w2_1.addWidget(self.slider_w2_1)
        self.wid_w2_1.setLayout(self.layout_w2_1)

        self.label_w2_2 = QtWidgets.QLabel(self)
        self.label_w2_2.setText("w2_2")
        self.label_w2_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 450 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w2_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_2.setRange(-20, 20)
        self.slider_w2_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_2.setTickInterval(10)
        self.slider_w2_2.setValue(10)

        self.wid_w2_2 = QtWidgets.QWidget(self)
        self.layout_w2_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_2.setGeometry(self.x_chapter_usual * self.w_ratio, 480 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w2_2.addWidget(self.slider_w2_2)
        self.wid_w2_2.setLayout(self.layout_w2_2)

        self.label_b2 = QtWidgets.QLabel(self)
        self.label_b2.setText("b1_2")
        self.label_b2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 520 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b2.setRange(-20, 20)
        self.slider_b2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b2.setTickInterval(10)
        self.slider_b2.setValue(0)

        self.wid_b2 = QtWidgets.QWidget(self)
        self.layout_b2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b2.setGeometry(self.x_chapter_usual * self.w_ratio, 550 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b2.addWidget(self.slider_b2)
        self.wid_b2.setLayout(self.layout_b2)

        self.slider_w1_1.valueChanged.connect(self.graph)
        self.slider_w1_2.valueChanged.connect(self.graph)
        self.slider_b1_1.valueChanged.connect(self.graph)
        self.slider_b1_2.valueChanged.connect(self.graph)
        self.slider_w2_1.valueChanged.connect(self.graph)
        self.slider_w2_2.valueChanged.connect(self.graph)
        self.slider_b2.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot
        a.set_xlim(-5, 5)
        a.set_ylim(0, 1)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        a.set_xticks([-5, 0, 4])
        a.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        a.plot([0]*10, np.linspace(-2, 2, 10), color="black", linestyle="--", linewidth=0.2)
        a.plot(np.linspace(-2, 2, 10), [0]*10, color="black", linestyle="--", linewidth=0.2)
        a.set_xlabel("$p$")
        a.xaxis.set_label_coords(1, -0.025)
        a.set_ylabel("$a$")
        a.yaxis.set_label_coords(-0.025, 1)

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

        weight1_1 = self.slider_w1_1.value() / 10
        weight1_2 = self.slider_w1_2.value() / 10
        bias1_1 = self.slider_b1_1.value() / 10
        bias1_2 = self.slider_b1_2.value() / 10
        weight2_1 = self.slider_w2_1.value() / 10
        weight2_2 = self.slider_w2_2.value() / 10
        bias2 = self.slider_b2.value() / 10

        self.label_w1_1.setText("w1_1: " + str(weight1_1))
        self.label_w1_2.setText("w1_2: " + str(weight1_2))
        self.label_b1_1.setText("b1_1: " + str(bias1_1))
        self.label_b1_2.setText("b1_2: " + str(bias1_2))
        self.label_w2_1.setText("w2_1: " + str(weight2_1))
        self.label_w2_2.setText("w2_2: " + str(weight2_2))
        self.label_b2.setText("b2: " + str(bias2))

        weight_1, bias_1 = np.array([[weight1_1, weight1_2]]), np.array([[bias1_1, bias1_2]])
        weight_2, bias_2 = np.array([[weight2_1], [weight2_2]]), np.array([[bias2]])

        p = np.arange(-4, 4, 0.01)
        # a = W2(1)*exp(-((p-W1(1)).*b1(1)).^2) + W2(2)*exp(-((p-W1(2)).*b1(2)).^2) + b2
        out = weight_2[0, 0] * np.exp(-((p - weight_1[0, 0]) * bias_1[0, 0]) ** 2)
        out += weight_2[1, 0] * np.exp(-((p - weight_1[0, 1]) * bias_1[0, 1]) ** 2) + bias_2[0, 0]

        a.plot(p, out.reshape(-1), markersize=3, color="red")
        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-2, 2)
        # a.set_ylim(-2, 2)
        # add grid and axes
        # a.grid(True, which='both')
        # a.axhline(y=0, color='k')
        # a.axvline(x=0, color='k')
        self.canvas.draw()