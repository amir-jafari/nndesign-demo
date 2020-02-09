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
    def __init__(self, w_ratio, h_ratio):
        super(DecisionBoundaries, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("Decision Boundaries", 4, " On the plot, click the\n Primary mouse button\n to add a positive class, and\n"
                                                    " the secondary mouse button\n to add a negative class \n Modify the parameters' values\n"
                                                    " and see how the error changes",
                          PACKAGE_PATH + "Chapters/4/Logo_Ch_4.svg", PACKAGE_PATH + "Chapters/4/Percptron1.svg")  # TODO: get icon

        self.w = np.ones((2,))
        self.w[1] = -1
        self.b = np.zeros((1,))

        self.data, self.data_missclasified = [], []
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.2, left=0.1)
        self.axes.set_xlim(-5, 5)
        self.axes.set_ylim(-5, 5)
        self.axes.tick_params(labelsize=8)
        self.axes.set_xlabel("$p^1$", fontsize=10)
        self.axes.xaxis.set_label_coords(0.5, 0.1)
        self.axes.set_ylabel("$p^2$", fontsize=10)
        self.axes.yaxis.set_label_coords(-0.05, 0.5)
        self.pos_line, = self.axes.plot([], 'mo', label="Positive Class")
        self.neg_line, = self.axes.plot([], 'cs', label="Negative Class")
        self.miss_line_pos, = self.axes.plot([], 'ro')
        self.miss_line_neg, = self.axes.plot([], 'rs')
        self.decision, = self.axes.plot([], 'r-', label="Decision Boundary")
        self.weight_vector = self.axes.quiver([0], [0], [1], [-1], scale=21, label="Weight vector")
        self.axes.legend(loc='lower center', fontsize=8, framealpha=0.9, numpoints=1, ncol=2,
                         bbox_to_anchor=(0, -.28, 1, -.280), mode='expand')
        self.axes.set_title("Single Neuron Perceptron")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        # latex_w = self.mathTex_to_QPixmap("$W = [1.0 -1.0]$", 6)
        # latex_w = self.mathTex_to_QPixmap(r"$W = \begin{bmatrix}1.0 & -1.0\end{bmatrix}$", 6)
        latex_w = self.mathTex_to_QPixmap("$W = [1.0 \quad -1.0]$", 5)
        self.latex_w = QtWidgets.QLabel(self)
        self.latex_w.setPixmap(latex_w)
        self.latex_w.setGeometry((self.x_chapter_usual + 15) * self.w_ratio, 260 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        # self.label_w1 = QtWidgets.QLabel(self)
        # self.label_w1.setText("w1: 1.0")
        # self.label_w1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_w1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 260 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_w1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1.setRange(-100, 100)
        self.slider_w1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1.setSingleStep(1)
        self.slider_w1.setTickInterval(1)
        self.slider_w1.setValue(10)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 300 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_w1)
        self.wid3.setLayout(self.layout3)

        # self.label_w2 = QtWidgets.QLabel(self)
        # self.label_w2.setText("w2: -1.0")
        # self.label_w2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_w2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 330 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_w2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2.setRange(-100, 100)
        self.slider_w2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2.setSingleStep(1)
        self.slider_w2.setTickInterval(1)
        self.slider_w2.setValue(-10)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 350 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_w2)
        self.wid4.setLayout(self.layout4)

        latex_b = self.mathTex_to_QPixmap("$b = [0.0]$", 5)
        self.latex_b = QtWidgets.QLabel(self)
        self.latex_b.setPixmap(latex_b)
        self.latex_b.setGeometry((self.x_chapter_usual + 30) * self.w_ratio, 390 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        # self.label_b = QtWidgets.QLabel(self)
        # self.label_b.setText("b: 0.0")
        # self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_b.setGeometry(self.x_chapter_slider_label * self.w_ratio, 400 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(-100, 100)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setSingleStep(1)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(0)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(self.x_chapter_usual * self.w_ratio, 425 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout5.addWidget(self.slider_b)
        self.wid5.setLayout(self.layout5)

        self.undo_click_button = QtWidgets.QPushButton("Undo Last Mouse Click", self)
        self.undo_click_button.setStyleSheet("font-size:13px")
        self.undo_click_button.setGeometry(self.x_chapter_button * self.w_ratio, 500 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.undo_click_button.clicked.connect(self.on_undo_mouseclick)

        self.clear_button = QtWidgets.QPushButton("Clear Data", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 530 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.label_error = QtWidgets.QLabel(self)
        self.label_error.setText("Error: ---")
        self.label_error.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_error.setGeometry(self.x_chapter_slider_label * self.w_ratio, 530 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        self.slider_w1.valueChanged.connect(self.slide_w1)
        self.slider_w2.valueChanged.connect(self.slide_w2)
        self.slider_b.valueChanged.connect(self.slide_b)

        # self.slider_w1.valueChanged.connect(self.slide)
        # self.slider_w2.valueChanged.connect(self.slide)
        # self.slider_b.valueChanged.connect(self.slide)

        self.draw_decision_boundary()

    def slide_w1(self):  # slide
        self.w[0] = float(self.slider_w1.value()) / 10
        # self.w[1] = float(self.slider_w2.value()) / 10
        # self.b[0] = float(self.slider_b.value()) / 10
        # self.label_w1.setText("w1: " + str(self.w[0]))
        # self.label_w2.setText("w2: " + str(self.w[1]))
        # self.label_b.setText("b: " + str(self.b[0]))
        self.latex_w.setPixmap(self.mathTex_to_QPixmap("$W = [{} \quad {}]$".format(self.w[0], self.w[1]), 5))
        # self.latex_b.setPixmap(self.mathTex_to_QPixmap("$b = [{}]$".format(self.b[0]), 5))
        self.draw_decision_boundary()
        self.compute_error()

    def slide_w2(self):
        self.w[1] = float(self.slider_w2.value()) / 10
        self.latex_w.setPixmap(self.mathTex_to_QPixmap("$W = [{} \quad {}]$".format(self.w[0], self.w[1]), 5))
        self.draw_decision_boundary()
        self.compute_error()

    def slide_b(self):
        self.b[0] = float(self.slider_b.value()) / 10
        self.latex_b.setPixmap(self.mathTex_to_QPixmap("$b = [{}]$".format(self.b[0]), 5))
        self.draw_decision_boundary()
        self.compute_error()

    def on_mouseclick(self, event):
        """Add an item to the plot"""
        if event.xdata != None and event.xdata != None:
            self.data.append((event.xdata, event.ydata, POS if event.button == 1 else NEG))
            self.compute_error()
            self.draw_data()

    def draw_data(self):
        # self.pos_line.set_data([x[0] for x in self.data if x[2] == POS], [y[1] for y in self.data if y[2] == POS])
        # self.neg_line.set_data([x[0] for x in self.data if x[2] == NEG], [y[1] for y in self.data if y[2] == NEG])
        # self.pos_line.set_data([x[0] for x in self.data if x[2] == POS], [y[1] for y in self.data if y[2] == POS])
        # self.neg_line.set_data([x[0] for x in self.data if x[2] == NEG], [y[1] for y in self.data if y[2] == NEG])
        data_pos, data_neg, data_miss_pos, data_miss_neg = [], [], [], []
        for xy, miss in zip(self.data, self.data_missclasified):
            if miss:
                # self.miss_line.set_data([xy[0]], [xy[1]])
                if xy[2] == 1:
                    data_miss_pos.append(xy)
                elif xy[2] == 0:
                    data_miss_neg.append(xy)
            else:
                if xy[2] == 1:
                    data_pos.append(xy)
                    # self.pos_line.set_data([xy[0]], [xy[1]])
                elif xy[2] == 0:
                    data_neg.append(xy)
                    # self.neg_line.set_data([xy[0]], [xy[1]])
        self.pos_line.set_data([xy[0] for xy in data_pos], [xy[1] for xy in data_pos])
        self.neg_line.set_data([xy[0] for xy in data_neg], [xy[1] for xy in data_neg])
        self.miss_line_pos.set_data([xy[0] for xy in data_miss_pos], [xy[1] for xy in data_miss_pos])
        self.miss_line_neg.set_data([xy[0] for xy in data_miss_neg], [xy[1] for xy in data_miss_neg])
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
            # all_t_hat = np.array([self.run_forward(np.array(xy[0:2])) for xy in self.data])
            # error = abs(np.array([t[2] for t in self.data]) - all_t_hat).sum()
            self.data_missclasified, error = [], 0
            for xy in self.data:
                t_hat = self.run_forward(np.array(xy[0:2]))
                if t_hat != xy[2]:
                    self.data_missclasified.append(True)
                    error += 1
                else:
                    self.data_missclasified.append(False)
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
            self.compute_error()
            self.draw_data()
