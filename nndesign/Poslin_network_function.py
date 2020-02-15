from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class PoslinNetworkFunction(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(PoslinNetworkFunction, self).__init__(w_ratio, h_ratio, main_menu=2)

        self.fill_chapter("Poslin Network Function", 2, "",
                          PACKAGE_PATH + "Chapters/2_D/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2_D/poslinNet_new.svg", icon_move_left=120)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.poslin, self.purelin, self.hardlim, self.hardlims, self.satlin, self.satlins, self.logsig, self.tansig]
        self.comboBox1.addItems(["Poslin", "Purelin", 'Hardlim', 'Hardlims', 'Satlin', 'Satlins', 'LogSig', 'TanSig'])
        self.comboBox1.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 580 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.func1 = self.poslin
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 570 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 590 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.label_w1_1 = QtWidgets.QLabel(self)
        self.label_w1_1.setText("w1_1")
        self.label_w1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 100 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_w1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_1.setRange(-1, 1)
        self.slider_w1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_1.setTickInterval(1)
        self.slider_w1_1.setValue(1)

        self.wid_w1_1 = QtWidgets.QWidget(self)
        self.layout_w1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 130 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_w1_1.addWidget(self.slider_w1_1)
        self.wid_w1_1.setLayout(self.layout_w1_1)

        self.label_w1_2 = QtWidgets.QLabel(self)
        self.label_w1_2.setText("w1_2")
        self.label_w1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 170 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_w1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_2.setRange(-3, 3)
        self.slider_w1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_2.setTickInterval(1)
        self.slider_w1_2.setValue(2)

        self.wid_w1_2 = QtWidgets.QWidget(self)
        self.layout_w1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 200 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_w1_2.addWidget(self.slider_w1_2)
        self.wid_w1_2.setLayout(self.layout_w1_2)

        self.label_b1_1 = QtWidgets.QLabel(self)
        self.label_b1_1.setText("b1_1")
        self.label_b1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 240 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_b1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_1.setRange(-2, 2)
        self.slider_b1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_1.setTickInterval(1)
        self.slider_b1_1.setValue(0)

        self.wid_b1_1 = QtWidgets.QWidget(self)
        self.layout_b1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 270 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_b1_1.addWidget(self.slider_b1_1)
        self.wid_b1_1.setLayout(self.layout_b1_1)

        self.label_b1_2 = QtWidgets.QLabel(self)
        self.label_b1_2.setText("b1_2")
        self.label_b1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 310 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_b1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_2.setRange(-1, 1)
        self.slider_b1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_2.setTickInterval(1)
        self.slider_b1_2.setValue(-1)

        self.wid_b1_2 = QtWidgets.QWidget(self)
        self.layout_b1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 340 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_b1_2.addWidget(self.slider_b1_2)
        self.wid_b1_2.setLayout(self.layout_b1_2)

        self.label_w2_1 = QtWidgets.QLabel(self)
        self.label_w2_1.setText("w2_1")
        self.label_w2_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 380 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_w2_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_1.setRange(-5, 5)
        self.slider_w2_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_1.setTickInterval(1)
        self.slider_w2_1.setValue(-1)

        self.wid_w2_1 = QtWidgets.QWidget(self)
        self.layout_w2_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_1.setGeometry(self.x_chapter_usual * self.w_ratio, 410 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_w2_1.addWidget(self.slider_w2_1)
        self.wid_w2_1.setLayout(self.layout_w2_1)

        self.label_w2_2 = QtWidgets.QLabel(self)
        self.label_w2_2.setText("w2_2")
        self.label_w2_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 450 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_w2_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_2.setRange(-3, 3)
        self.slider_w2_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_2.setTickInterval(1)
        self.slider_w2_2.setValue(2)

        self.wid_w2_2 = QtWidgets.QWidget(self)
        self.layout_w2_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_2.setGeometry(self.x_chapter_usual * self.w_ratio, 480 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_w2_2.addWidget(self.slider_w2_2)
        self.wid_w2_2.setLayout(self.layout_w2_2)

        self.label_b2 = QtWidgets.QLabel(self)
        self.label_b2.setText("b1_2")
        self.label_b2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 540 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b2.setRange(-2, 2)
        self.slider_b2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b2.setTickInterval(1)
        self.slider_b2.setValue(0)
        self.wid_b2 = QtWidgets.QWidget(self)
        self.layout_b2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b2.setGeometry(self.x_chapter_usual * self.w_ratio, 570 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b2.addWidget(self.slider_b2)
        self.wid_b2.setLayout(self.layout_b2)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
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

        weight1_1 = self.slider_w1_1.value()
        weight1_2 = self.slider_w1_2.value()
        bias1_1 = self.slider_b1_1.value()
        bias1_2 = self.slider_b1_2.value()
        weight2_1 = self.slider_w2_1.value()
        weight2_2 = self.slider_w2_2.value()
        bias2 = self.slider_b2.value()

        self.label_w1_1.setText("w1_1: " + str(weight1_1))
        self.label_w1_2.setText("w1_2: " + str(weight1_2))
        self.label_b1_1.setText("b1_1: " + str(bias1_1))
        self.label_b1_2.setText("b1_2: " + str(bias1_2))
        self.label_w2_1.setText("w2_1: " + str(weight2_1))
        self.label_w2_2.setText("w2_2: " + str(weight2_2))
        self.label_b2.setText("b2: " + str(bias2))

        weight_1, bias_1 = np.array([[weight1_1, weight1_2]]), np.array([[bias1_1, bias1_2]])
        weight_2, bias_2 = np.array([[weight2_1], [weight2_2]]), np.array([[bias2]])

        p = np.arange(-4, 4, 0.1)
        func = np.vectorize(self.func1)
        out = np.dot(func(np.dot(p.reshape(-1, 1), weight_1) + bias_1), weight_2) + bias_2

        a.plot(p, out.reshape(-1), markersize=3, color="red")
        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-2, 2)
        # a.set_ylim(-2, 2)
        # add grid and axes
        a.grid(True, which='both')
        a.axhline(y=0, color='k')
        a.axvline(x=0, color='k')
        self.canvas.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions[idx]
        self.graph()

    @staticmethod
    def hardlim(x):
        if x < 0:
            return 0
        else:
            return 1

    @staticmethod
    def hardlims(x):
        if x < 0:
            return -1
        else:
            return 1

    @staticmethod
    def poslin(x):
        if x < 0:
            return 0
        else:
            return x

    @staticmethod
    def purelin(x):
        return x

    @staticmethod
    def satlin(x):
        if x < 0:
            return 0
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    def satlins(x):
        if x < -1:
            return 0
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    def logsig(x):
        return 1 / (1 + math.e ** (-x))

    @staticmethod
    def tansig(x):
        return 2 / (1 + math.e ** (-2 * x)) - 1
