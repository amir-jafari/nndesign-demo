# https://stackoverflow.com/questions/28001655/draggable-line-with-draggable-points

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


POS = 1
NEG = 0


def hardlim(n):
    return 0 if n < 0 else 1


class DecisionBoundaries(NNDLayout):
    def __init__(self):
        super(DecisionBoundaries, self).__init__(main_menu=1)

        self.fill_chapter("Decision Boundaries", 4, "On the plot, click the\n Primary mouse button\n to add a positive class, and\n"
                                                    " the secondary mouse button\n to add a negative class \n Modify the parameters' values"
                                                    " and see how the error changes",
                          PACKAGE_PATH + "Chapters/4/Logo_Ch_4.svg", PACKAGE_PATH + "Chapters/4/Percptron1.svg")  # TODO: get icon

        self.w = np.ones((2,))
        self.w[1] = -1
        self.b = np.zeros((1,))

        self.data = []
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.2, left=0.1)
        self.axes.set_xlim(-5, 5)
        self.axes.set_ylim(-5, 5)
        self.axes.tick_params(labelsize=8)
        self.axes.set_xlabel("$p^1$", fontsize=10)
        self.axes.set_ylabel("$p^2$", fontsize=10)
        self.pos_line, = self.axes.plot([], 'mo', label="Positive Class")
        self.neg_line, = self.axes.plot([], 'cs', label="Negative Class")
        self.decision, = self.axes.plot([], 'r-', label="Decision Boundary")
        self.weight_vector = self.axes.quiver([0], [0], [1], [-1], scale=21, label="Weight vector")
        self.axes.legend(loc='lower center', fontsize=8, framealpha=0.9, numpoints=1, ncol=3,
                         bbox_to_anchor=(0, -.24, 1, -.280), mode='expand')
        self.axes.set_title("Single Neuron Perceptron")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.label_w1 = QtWidgets.QLabel(self)
        self.label_w1.setText("w1: 1.0")
        self.label_w1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1.setGeometry(775, 330, 150, 100)
        self.slider_w1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1.setRange(-100, 100)
        self.slider_w1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1.setSingleStep(1)
        self.slider_w1.setTickInterval(1)
        self.slider_w1.setValue(10)

        self.label_w2 = QtWidgets.QLabel(self)
        self.label_w2.setText("w2: -1.0")
        self.label_w2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2.setGeometry(775, 400, 150, 100)
        self.slider_w2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2.setRange(-100, 100)
        self.slider_w2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2.setSingleStep(1)
        self.slider_w2.setTickInterval(1)
        self.slider_w2.setValue(-10)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("b: 0.0")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(775, 470, 150, 100)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(-100, 100)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setSingleStep(1)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(0)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(710, 360, 150, 100)
        self.layout3.addWidget(self.slider_w1)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(710, 430, 150, 100)
        self.layout4.addWidget(self.slider_w2)
        self.wid4.setLayout(self.layout4)

        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(710, 500, 150, 100)
        self.layout5.addWidget(self.slider_b)
        self.wid5.setLayout(self.layout5)

        self.label_error = QtWidgets.QLabel(self)
        self.label_error.setText("Error: ---")
        self.label_error.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_error.setGeometry(775, 550, 150, 100)

        self.undo_click_button = QtWidgets.QPushButton("Undo Last Mouse Click", self)
        self.undo_click_button.setStyleSheet("font-size:13px")
        self.undo_click_button.setGeometry(705, 670, 190, 30)
        self.undo_click_button.clicked.connect(self.on_undo_mouseclick)

        self.clear_button = QtWidgets.QPushButton("Clear Data", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(705, 620, 190, 30)
        self.clear_button.clicked.connect(self.on_clear)

        self.slider_w1.valueChanged.connect(self.slide)
        self.slider_w2.valueChanged.connect(self.slide)
        self.slider_b.valueChanged.connect(self.slide)

        self.draw_decision_boundary()

    def slide(self):
        weight1 = float(self.slider_w1.value()) / 10
        weight2 = float(self.slider_w2.value()) / 10
        bias = float(self.slider_b.value()) / 10
        self.w[0], self.w[1], self.b[0] = weight1, weight2, bias
        self.label_w1.setText("w1: " + str(weight1))
        self.label_w2.setText("w2: " + str(weight2))
        self.label_b.setText("b: " + str(bias))
        self.draw_decision_boundary()
        self.compute_error()

    def on_mouseclick(self, event):
        """Add an item to the plot"""
        if event.xdata != None and event.xdata != None:
            self.data.append((event.xdata, event.ydata, POS if event.button == 1 else NEG))
            self.draw_data()
            self.compute_error()

    def draw_data(self):
        self.pos_line.set_data([x[0] for x in self.data if x[2] == POS], [y[1] for y in self.data if y[2] == POS])
        self.neg_line.set_data([x[0] for x in self.data if x[2] == NEG], [y[1] for y in self.data if y[2] == NEG])
        self.canvas.draw()

    def draw_decision_boundary(self):
        lim = self.axes.get_xlim()
        X = np.linspace(lim[0], lim[1], 101)
        Y = self.find_decision_boundary(X)
        self.decision.set_data(X, Y)
        self.weight_vector.set_UVC(self.w[0], self.w[1])
        self.canvas.draw()

    def find_decision_boundary(self, x):
        """Returns the corresponding y value for the input x on the decision
        boundary"""
        return -(x * self.w[0] + self.b) / (self.w[1] if self.w[1] != 0 else .000001)

    def run_forward(self, p):
        """Given an input of dimension R, run the network"""
        return hardlim(self.w.dot(p) + self.b)

    def compute_error(self):
        if self.data:
            all_t_hat = np.array([self.run_forward(np.array(xy[0:2])) for xy in self.data])
            error = abs(np.array([t[2] for t in self.data]) - all_t_hat).sum()
            self.label_error.setText("Error: {}".format(error))
        else:
            self.label_error.setText("Error: ---")

    def on_clear(self):
        self.data = []
        self.compute_error()
        self.draw_data()

    def clear_decision_boundary(self):
        self.decision.set_data([], [])
        self.canvas.draw()

    def on_undo_mouseclick(self):
        if self.data:
            self.data.pop()
            self.draw_data()
        self.compute_error()
