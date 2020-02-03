from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


def logsigmoid(n):
    return 1 / (1 + np.exp(-n))


def logsigmoid_der(n):
    return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))


def purelin(n):
    return n


def purelin_der(n):
    return np.array([1]).reshape(n.shape)


class FunctionApproximation(NNDLayout):
    def __init__(self):
        super(FunctionApproximation, self).__init__(main_menu=1)

        self.fill_chapter("Function Approximation", 11, "Click the train button to train the log-sig ...",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")  # TODO: Logo and Icon

        self.S1 = 4
        self.diff = 1
        self.p = np.linspace(-2, 2, 100)
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.random_state = 0
        self.init_params()

        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.2, left=0.1)
        self.axes.set_xlim(-2, 2)
        self.axes.set_ylim(0, 2)
        self.axes.tick_params(labelsize=8)
        self.axes.set_xlabel("Input", fontsize=10)
        self.axes.set_ylabel("Target", fontsize=10)
        self.data_to_approx, = self.axes.plot([], label="Function to Approximate")
        self.net_approx, = self.axes.plot([], label="Network Approximation")
        self.axes.legend(loc='lower center', fontsize=8, framealpha=0.9, numpoints=1, ncol=3,
                         bbox_to_anchor=(0, -.24, 1, -.280), mode='expand')
        self.axes.set_title("Function Approximation")
        self.plot_f()

        self.label_s1 = QtWidgets.QLabel(self)
        self.label_s1.setText("Number of Hidden Neurons S1: 4")
        self.label_s1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_s1.setGeometry(710, 400, 150, 100)
        self.slider_s1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_s1.setRange(1, 9)
        self.slider_s1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_s1.setTickInterval(1)
        self.slider_s1.setValue(4)

        self.label_diff = QtWidgets.QLabel(self)
        self.label_diff.setText("Difficulty index: 1")
        self.label_diff.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_diff.setGeometry(710, 470, 150, 100)
        self.slider_diff = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_diff.setRange(1, 9)
        self.slider_diff.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_diff.setTickInterval(1)
        self.slider_diff.setValue(1)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(710, 430, 150, 100)
        self.layout3.addWidget(self.slider_s1)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(710, 500, 150, 100)
        self.layout4.addWidget(self.slider_diff)
        self.wid4.setLayout(self.layout4)

        self.slider_s1.valueChanged.connect(self.slide)
        self.slider_diff.valueChanged.connect(self.slide)

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(705, 600, 190, 30)
        self.run_button.clicked.connect(self.on_run)

    def slide(self):
        slider_s1 = self.slider_s1.value()
        if self.S1 != slider_s1:
            self.S1 = slider_s1
            self.init_params()
        self.diff = self.slider_diff.value()
        self.label_s1.setText("Number of Hidden Neurons S1: {}".format(self.S1))
        self.label_diff.setText("Difficulty Index: {}".format(self.diff))
        self.f_to_approx = lambda p: 1 + np.sin(np.pi * p * self.diff / 5)
        self.plot_f()

    def init_params(self):
        np.random.seed(self.random_state)
        self.W1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.b1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (1, self.S1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def plot_f(self):
        self.data_to_approx.set_data(self.p, 1 + np.sin(np.pi * self.p * self.diff / 5))
        self.canvas.draw()

    def f_to_approx(self, p):
        return 1 + np.sin(np.pi * p * self.diff / 5)

    def on_run(self):
        n_epochs, alpha = 100, 0.01
        # Training Loop
        for _ in range(n_epochs):
            # Array of the errors for each sample
            error = np.array([])
            nn_output = []
            # Updating parameters for each sample
            for sample in self.p:
                # Propagates the input forward
                # Reshapes input as 1x1
                a0 = sample.reshape(-1, 1)
                # Hidden Layer's Net Input
                n1 = np.dot(self.W1, a0) + self.b1
                #  Hidden Layer's Transformation
                a1 = logsigmoid(n1)
                # Output Layer's Net Input
                n2 = np.dot(self.W2, a1) + self.b2
                # Output Layer's Transformation
                a = purelin(n2)  # (a2 = a)
                nn_output.append(a)

                # Back-propagates the sensitivities
                # Compares our NN's output with the real value
                e = self.f_to_approx(a0) - a
                error = np.append(error, e)
                # Output Layer
                F2_der = np.diag(purelin_der(n2).reshape(-1))
                s = -2 * np.dot(F2_der, e)  # (s2 = s)
                # Hidden Layer
                F1_der = np.diag(logsigmoid_der(n1).reshape(-1))
                s1 = np.dot(F1_der, np.dot(self.W2.T, s))

                # Updates the weights and biases
                # Hidden Layer
                self.W1 += -alpha * np.dot(s1, a0.T)
                self.b1 += -alpha * s1
                # Output Layer
                self.W2 += -alpha * np.dot(s, a1.T)
                self.b2 += -alpha * s

            self.plot_approx(np.array(nn_output))

    def plot_approx(self, approx):
        self.net_approx.set_data(self.p, approx)
        self.canvas.draw()

    def net_forward(self):
        nn_output = np.array([])
        for sample in self.p:
            a0 = sample.reshape(-1, 1)
            # Hidden Layer's Transformation
            n1 = np.dot(self.W1, a0) + self.b1
            a1 = logsigmoid(n1)
            # Output Layer's Transformation
            n2 = np.dot(self.W2, a1) + self.b2
            nn_output = np.append(nn_output, purelin(n2)[0][0])
        return nn_output
