from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class TwoInputNeuron(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(TwoInputNeuron, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Two input neuron", 2, " Alter the weight and bias\n and input by dragging the\n triangular"
                                                 " shaped indictors.\n \n Pick the transfer function\n with the F menu.\n "
                                                 "\n Watch the change\n to the  neuron function\n and its  output.",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.purelin, self.hardlim, self.hardlims, self.satlin, self.satlins, self.logsig, self.tansig]
        self.comboBox1.addItems(["Purelin", 'Hardlim', 'Hardlims', 'Satlin', 'Satlins', 'LogSig', 'TanSig'])
        self.func = self.purelin
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 550 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.label_p1 = QtWidgets.QLabel(self)
        self.label_p1.setText("p1")
        self.label_p1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_p1.setGeometry(25 * self.w_ratio, 220 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.slider_p1 = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider_p1.setRange(-10, 10)
        self.slider_p1.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider_p1.setTickInterval(1)
        self.slider_p1.setValue(0)
        self.wid_p1 = QtWidgets.QWidget(self)
        self.layout_p1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_p1.setGeometry(30 * self.w_ratio, 220 * self.h_ratio, 50 * self.w_ratio, 100 * self.h_ratio)
        self.layout_p1.addWidget(self.slider_p1)
        self.wid_p1.setLayout(self.layout_p1)

        self.label_w1 = QtWidgets.QLabel(self)
        self.label_w1.setText("w1")
        self.label_w1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1.setGeometry(125 * self.w_ratio, 230 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.label_w1_ = QtWidgets.QLabel(self)
        self.label_w1_.setText("w1")
        self.label_w1_.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_.setGeometry(155 * self.w_ratio, 100 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.slider_w1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1.setRange(-20, 20)
        self.slider_w1.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider_w1.setTickInterval(1)
        self.slider_w1.setValue(10)
        self.wid_w1 = QtWidgets.QWidget(self)
        self.layout_w1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1.setGeometry(90 * self.w_ratio, 150 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1.addWidget(self.slider_w1)
        self.wid_w1.setLayout(self.layout_w1)

        self.label_p2 = QtWidgets.QLabel(self)
        self.label_p2.setText("p2")
        self.label_p2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_p2.setGeometry(25 * self.w_ratio, 335 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.slider_p2 = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider_p2.setRange(-10, 10)
        self.slider_p2.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider_p2.setTickInterval(1)
        self.slider_p2.setValue(0)
        self.wid_p2 = QtWidgets.QWidget(self)
        self.layout_p2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_p2.setGeometry(30 * self.w_ratio, 335 * self.h_ratio, 50 * self.w_ratio, 100 * self.h_ratio)
        self.layout_p2.addWidget(self.slider_p2)
        self.wid_p2.setLayout(self.layout_p2)

        self.label_w2 = QtWidgets.QLabel(self)
        self.label_w2.setText("w2")
        self.label_w2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2.setGeometry(130 * self.w_ratio, 310 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.label_w2_ = QtWidgets.QLabel(self)
        self.label_w2_.setText("w2")
        self.label_w2_.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_.setGeometry(155 * self.w_ratio, 470 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.slider_w2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2.setRange(-20, 20)
        self.slider_w2.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider_w2.setTickInterval(1)
        self.slider_w2.setValue(-10)
        self.wid_w2 = QtWidgets.QWidget(self)
        self.layout_w2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2.setGeometry(90 * self.w_ratio, 500 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.layout_w2.addWidget(self.slider_w2)
        self.wid_w2.setLayout(self.layout_w2)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("b")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(210 * self.w_ratio, 340 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.label_b_ = QtWidgets.QLabel(self)
        self.label_b_.setText("b")
        self.label_b_.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b_.setGeometry(315 * self.w_ratio, 470 * self.h_ratio, 30 * self.w_ratio, 100 * self.h_ratio)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(-20, 20)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(0)
        self.wid_b = QtWidgets.QWidget(self)
        self.layout_b = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b.setGeometry(240 * self.w_ratio, 500 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.layout_b.addWidget(self.slider_b)
        self.wid_b.setLayout(self.layout_b)

        self.label_n = QtWidgets.QLabel(self)
        self.label_n.setText("n: 0.0")
        self.label_n.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_n.setGeometry(310 * self.w_ratio, 310 * self.h_ratio, 50 * self.w_ratio, 100 * self.h_ratio)
        self.slider_n = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider_n.setRange(-60, 60)
        self.slider_n.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider_n.setValue(0)
        self.wid_n = QtWidgets.QWidget(self)
        self.layout_n = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_n.setGeometry(260 * self.w_ratio, 280 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.layout_n.addWidget(self.slider_n)
        self.wid_n.setLayout(self.layout_n)

        self.label_a = QtWidgets.QLabel(self)
        self.label_a.setText("a: 0.0")
        self.label_a.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a.setGeometry(420 * self.w_ratio, 310 * self.h_ratio, 50 * self.w_ratio, 200 * self.h_ratio)
        self.slider_a = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider_a.setRange(-60, 60)
        self.slider_a.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider_a.setValue(0)
        self.wid_a = QtWidgets.QWidget(self)
        self.layout_a = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_a.setGeometry(470 * self.w_ratio, 230 * self.h_ratio, 150 * self.w_ratio, 200 * self.h_ratio)
        self.layout_a.addWidget(self.slider_a)
        self.wid_a.setLayout(self.layout_a)

        self.icon3 = QtWidgets.QLabel(self)
        self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nn2d2.svg").pixmap(400 * self.w_ratio, 300 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon3.setGeometry(75 * self.w_ratio, 200 * self.h_ratio, 400 * self.w_ratio, 300 * self.h_ratio)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.slider_p1.valueChanged.connect(self.slide)
        self.slider_p2.valueChanged.connect(self.slide)
        self.slider_w1.valueChanged.connect(self.slide)
        self.slider_w2.valueChanged.connect(self.slide)
        self.slider_b.valueChanged.connect(self.slide)

    def slide(self):
        p_1 = float(self.slider_p1.value() / 10)
        w_1 = float(self.slider_w1.value() / 10)
        p_2 = float(self.slider_p2.value() / 10)
        w_2 = float(self.slider_w2.value() / 10)
        b = float(self.slider_b.value() / 10)
        n = w_1 * p_1 + w_2 * p_2 + b
        self.slider_n.setValue(n * 10)
        self.label_n.setText("n: {}".format(round(n, 2)))
        a = self.func(n)
        self.slider_a.setValue(a * 10)
        self.label_a.setText("a: {}".format(round(a, 2)))

    def change_transfer_function(self, idx):
        self.func = self.comboBox1_functions[idx]
        self.slide()

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
