from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class OneInputNeuron(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(OneInputNeuron, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("One input neuron", 2, " Alter the weight and bias\n and input by dragging the\n triangular"
                                                 " shaped indictors.\n \n Pick the transfer function\n with the F menu.\n "
                                                 "\n Watch the change\n to the  neuron function\n and its  output.",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.purelin, self.hardlim, self.hardlims, self.satlin, self.satlins, self.logsig, self.tansig]
        self.comboBox1_functions_str = ["purelin", 'hardlim', 'hardlims', 'satlin', 'satlins', 'logsig', 'tansig']
        self.comboBox1.addItems(self.comboBox1_functions_str)
        self.func1 = self.purelin
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 550 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        self.label_eq = QtWidgets.QLabel(self)
        self.label_eq.setText("a = purelin(w * p + b)")
        self.label_eq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_eq.setGeometry((self.x_chapter_slider_label - 30) * self.w_ratio, 350 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        self.label_w = QtWidgets.QLabel(self)
        self.label_w.setText("w")
        self.label_w.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w.setGeometry(self.x_chapter_slider_label * self.w_ratio, 400 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_w = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w.setRange(-30, 30)
        self.slider_w.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w.setTickInterval(1)
        self.slider_w.setValue(1)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("b")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(self.x_chapter_slider_label * self.w_ratio, 470 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(-30, 30)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(0)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_w)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_b)
        self.wid4.setLayout(self.layout4)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.slider_w.valueChanged.connect(self.graph)
        self.slider_b.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot
        a.set_xlim(-2, 2)
        a.set_ylim(-2, 2)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        a.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.plot([0] * 10, np.linspace(-2, 2, 10), color="black", linestyle="--", linewidth=0.2)
        a.plot(np.linspace(-2, 2, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        a.set_xlabel("$p$")
        a.xaxis.set_label_coords(1, -0.025)
        a.set_ylabel("$a$")
        a.yaxis.set_label_coords(-0.025, 1)

        weight = self.slider_w.value() / 10
        bias = self.slider_b.value() / 10
        self.label_w.setText("w: " + str(weight))
        self.label_b.setText("b: " + str(bias))
        p = np.arange(-4, 4, 0.1)
        func = np.vectorize(self.func1)
        out = func(np.dot(weight, p) + bias)

        a.plot(p, out, markersize=3, color="red")
        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-4, 4)
        # a.set_ylim(-2, 2)
        # add grid and axes
        # a.grid(True, which='both')
        # a.axhline(y=0, color='k')
        # a.axvline(x=0, color='k')
        self.canvas.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions[idx]
        self.label_eq.setText("a = {}(w * p + b)".format(self.comboBox1_functions_str[idx]))
        self.graph()
