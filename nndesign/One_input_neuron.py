from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class OneInputNeuron(NNDLayout):
    def __init__(self):
        super(OneInputNeuron, self).__init__(main_menu=1)

        self.fill_chapter("One input neuron", 2, "Alter the weight and bias\n and input by dragging the\n triangular"
                                                 " shaped indictors.\n \n Pick the transfer function\n with the F menu.\n "
                                                 "\n Watch the change\n to the  neuron function\n and its  output.",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.purelin, self.hardlim, self.hardlims, self.satlin, self.satlins, self.logsig, self.tansig]
        self.comboBox1.addItems(["Purelin", 'Hardlim', 'Hardlims', 'Satlin', 'Satlins', 'LogSig', 'TanSig'])
        self.func1 = self.purelin
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry(775, 550, 150, 100)

        self.label_w = QtWidgets.QLabel(self)
        self.label_w.setText("w")
        self.label_w.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w.setGeometry(775, 400, 150, 100)
        self.slider_w = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w.setRange(-3, 3)
        self.slider_w.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w.setTickInterval(1)
        self.slider_w.setValue(1)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("b")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(775, 470, 150, 100)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(-3, 3)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(0)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(710, 580, 150, 100)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(710, 430, 150, 100)
        self.layout3.addWidget(self.slider_w)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(710, 500, 150, 100)
        self.layout4.addWidget(self.slider_b)
        self.wid4.setLayout(self.layout4)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.slider_w.valueChanged.connect(self.graph)
        self.slider_b.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot

        weight = self.slider_w.value()
        bias = self.slider_b.value()
        self.label_w.setText("w: " + str(weight))
        self.label_b.setText("b: " + str(bias))
        p = np.arange(-4, 4, 0.1)
        func = np.vectorize(self.func1)
        out = func(np.dot(weight, p) + bias)

        a.plot(p, out, markersize=3, color="red")
        # Setting limits so that the point moves instead of the plot.
        a.set_xlim(-4, 4)
        a.set_ylim(-2, 2)
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
