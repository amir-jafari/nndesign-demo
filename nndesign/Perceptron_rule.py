from PyQt5 import QtWidgets, QtGui, QtCore
import random
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


POS = 1
NEG = 0


def hardlim(n):
    return 0 if n < 0 else 1


class PerceptronRule(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(PerceptronRule, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("Perceptron rule", 4, " On the plot, click the\n Primary mouse button\n to add a positive class.\n"
                                                " Secondary mouse button\n to add a negative class.\n Then click Train  ",
                          PACKAGE_PATH + "Chapters/4/Logo_Ch_4.svg", PACKAGE_PATH + "Chapters/4/Percptron1.svg")

        self.data, self.data_missclasified = [], []
        self.total_epochs = 0
        self.R = 2  # Num input dimensions
        self.S = 1  # Num neurons

        # Add a plot
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.2, left=0.1)
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 10)
        self.axes.tick_params(labelsize=10)
        self.axes.set_xlabel("$p^1$", fontsize=10)
        self.axes.xaxis.set_label_coords(0.5, 0.1)
        self.axes.set_ylabel("$p^2$", fontsize=10)
        self.axes.yaxis.set_label_coords(-0.05, 0.5)
        self.pos_line, = self.axes.plot([], 'mo', label="Positive Class")
        self.neg_line, = self.axes.plot([], 'cs', label="Negative Class")
        self.miss_line_pos, = self.axes.plot([], 'ro')
        self.miss_line_neg, = self.axes.plot([], 'rs')
        self.highlight_data, = self.axes.plot([], "*", markersize=16)
        self.decision, = self.axes.plot([], 'r-', label="Decision Boundary")
        self.axes.legend(loc='lower left', fontsize=8, numpoints=1, ncol=3, bbox_to_anchor=(-0.1, -.24, 1.1, -.280))
        self.axes.set_title("Single Neuron Perceptron")
        self.canvas.draw()
        # Add event handler for a mouseclick in the plot
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.epoch_label = QtWidgets.QLabel(self)
        self.epoch_label.setText("Epochs so far: 0")
        self.epoch_label.setFixedHeight(20)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 600 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.epoch_label)
        self.wid2.setLayout(self.layout2)

        self.error_label = QtWidgets.QLabel(self)
        self.error_label.setText("Error: ---")
        self.error_label.setFixedHeight(20)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry((self.x_chapter_usual + 10) * self.w_ratio, 540 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.error_label)
        self.wid4.setLayout(self.layout4)

        self.warning_label = QtWidgets.QLabel(self)
        self.warning_label.setText("")
        # self.warning_label.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.warning_label.setGeometry((self.x_chapter_usual + 10) * self.w_ratio, 520 * self.h_ratio, 300 * self.w_ratio, 100 * self.h_ratio)

        self.epr_label = QtWidgets.QLabel("Epochs to run")
        wid6 = QtWidgets.QWidget(self)
        layout6 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        wid6.setGeometry((self.x_chapter_usual + 5) * self.w_ratio, 480 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        layout6.addWidget(self.epr_label)
        wid6.setLayout(layout6)
        self.epochs_per_run = QtWidgets.QComboBox(self)
        self.epochs_per_run.addItems(["1", "10", "100", "1000"])
        self.epochs_per_run.setCurrentIndex(0)
        wid7 = QtWidgets.QWidget(self)
        layout7 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        wid7.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        layout7.addWidget(self.epochs_per_run)
        wid7.setLayout(layout7)

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 460 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

        self.run_button = QtWidgets.QPushButton("Learn", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run_3)

        self.rerun_button = QtWidgets.QPushButton("Reset to Start", self)
        self.rerun_button.setStyleSheet("font-size:13px")
        self.rerun_button.setGeometry(self.x_chapter_button * self.w_ratio, 380 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.rerun_button.clicked.connect(self.on_reset)

        self.undo_click_button = QtWidgets.QPushButton("Undo Last Mouse Click", self)
        self.undo_click_button.setStyleSheet("font-size:13px")
        self.undo_click_button.setGeometry(self.x_chapter_button * self.w_ratio, 340 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.undo_click_button.clicked.connect(self.on_undo_mouseclick)

        self.clear_button = QtWidgets.QPushButton("Clear Data", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 300 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.ani = None
        self.initialize_weights()
        self.learn = None

    def draw_data(self):

        data_pos, data_neg, data_miss_pos, data_miss_neg = [], [], [], []
        for xy, miss in zip(self.data, self.data_missclasified):
            if miss:
                if xy[2] == 1:
                    data_miss_pos.append(xy)
                elif xy[2] == 0:
                    data_miss_neg.append(xy)
            else:
                if xy[2] == 1:
                    data_pos.append(xy)
                elif xy[2] == 0:
                    data_neg.append(xy)
        self.pos_line.set_data([xy[0] for xy in data_pos], [xy[1] for xy in data_pos])
        self.neg_line.set_data([xy[0] for xy in data_neg], [xy[1] for xy in data_neg])
        self.miss_line_pos.set_data([xy[0] for xy in data_miss_pos], [xy[1] for xy in data_miss_pos])
        self.miss_line_neg.set_data([xy[0] for xy in data_miss_neg], [xy[1] for xy in data_miss_neg])
        # self.canvas.draw()

    def draw_decision_boundary(self):
        lim = self.axes.get_xlim()
        X = np.linspace(lim[0], lim[1], 101)
        Y = self.find_decision_boundary(X)
        self.decision.set_data(X, Y)
        # self.canvas.draw()

    def clear_decision_boundary(self):
        self.decision.set_data([], [])
        # self.canvas.draw()

    def on_mouseclick(self, event):
        if self.ani:
            self.ani.event_source.stop()
        """Add an item to the plot"""
        if event.xdata != None and event.xdata != None:
            self.data.append((event.xdata, event.ydata, POS if event.button == 1 else NEG))
        self.data_missclasified, error = [], 0
        for xy in self.data:
            t_hat = self.run_forward(np.array(xy[0:2]))
            if t_hat != xy[2]:
                self.data_missclasified.append(True)
                error += 1
            else:
                self.data_missclasified.append(False)
        self.draw_data()
        self.canvas.draw()

    def on_clear(self):
        if self.ani:
            self.ani.event_source.stop()
        self.data = []
        self.clear_decision_boundary()
        self.initialize_weights()
        self.total_epochs = 0
        self.update_run_status()
        self.draw_data()
        self.canvas.draw()

    def update_run_status(self):
        if self.total_epochs == 0:
            self.epoch_label.setText("Epochs so far: 0")
            self.error_label.setText("Error: ---")
        else:
            self.epoch_label.setText("Epochs so far: {}".format(self.total_epochs))
            self.error_label.setText("Error: {}".format(self.total_error))

    def on_run(self):

        if self.ani:
            self.ani.event_source.stop()

        if int(self.epochs_per_run.currentText()) > 1:

            if len(self.data) < 2:
                self.warning_label.setText("Please select at least two\ndata points before training")
            else:
                if len(np.unique([cls[2] for cls in self.data])) == 1:
                    self.warning_label.setText("Please select at least one\ndata point of each class")
                else:
                    self.warning_label.setText("")
                    # Do 10 epochs
                    for epoch in range(int(self.epochs_per_run.currentText())):

                        # training = self.data.copy()
                        # np.random.shuffle(training)
                        for d in self.data:
                            self.total_epochs += 1
                            self.train_one_iteration(np.array(d[0:2]), d[2])

                        # Calculate the error for the epoch
                        self.all_t_hat = np.array([self.run_forward(np.array(xy[0:2])) for xy in self.data])
                        self.total_error = abs(np.array([t[2] for t in self.data]) - self.all_t_hat).sum()

                        if self.total_error == 0:
                            break

                    self.update_run_status()
                    self.draw_decision_boundary()
                    self.data_missclasified, error = [], 0
                    for xy in self.data:
                        t_hat = self.run_forward(np.array(xy[0:2]))
                        if t_hat != xy[2]:
                            self.data_missclasified.append(True)
                            error += 1
                        else:
                            self.data_missclasified.append(False)
                    self.draw_data()
                    self.canvas.draw()

        else:

            self.learn = False
            self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init,
                                     frames=len(self.data) * 2 + 1,
                                     interval=1000, repeat=False, blit=True)
            self.canvas.draw()

    def on_run_2(self):
        if self.ani:
            self.ani.event_source.stop()
        self.learn = False
        self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=len(self.data) * 2 + 1,
                                 interval=1000, repeat=False, blit=True)
        self.canvas.draw()

    def on_run_3(self):
        if self.ani:
            self.ani.event_source.stop()
        self.learn = True
        random.shuffle(self.data)
        self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=3,
                                 interval=1000, repeat=False, blit=True)
        self.canvas.draw()

    def animate_init(self):
        # self.pos_line.set_data([], [])
        # self.neg_line.set_data([], [])
        # self.miss_line_pos.set_data([], [])
        # self.miss_line_neg.set_data([], [])
        # self.decision.set_data([], [])
        self.all_t_hat = np.array([self.run_forward(np.array(xy[0:2])) for xy in self.data])
        self.total_error = abs(np.array([t[2] for t in self.data]) - self.all_t_hat).sum()
        # self.canvas.draw()
        return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data

    def on_animate(self, idx):
        """ GD version """

        if len(self.data) < 2:
            self.warning_label.setText("Please select at least two\ndata points before training")
            return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data
        else:
            if len(np.unique([cls[2] for cls in self.data])) == 1:
                self.warning_label.setText("Please select at least one\ndata point of each class")
                return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data
            else:
                self.warning_label.setText("")

                if self.total_error == 0:
                    return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data

                # training = self.data.copy()
                # np.random.shuffle(training)
                # for d in self.data:
                    # self.train_one_iteration(np.array(d[0:2]), d[2])
                # if idx > len(self.data) - 1:
                #     return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision
                # else:
                if self.learn and idx == 2:
                    self.highlight_data.set_data([], [])
                    return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data
                if idx == len(self.data) * 2:
                    self.highlight_data.set_data([], [])
                    return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data
                else:
                    if idx % 2 == 0:
                        self.highlight_data.set_data([self.data[int(idx / 2)][0]], [self.data[int(idx / 2)][1]])
                    else:
                        self.train_one_iteration(np.array(self.data[idx // 2][0:2]), self.data[idx // 2][2])
                        self.total_epochs += 1

                # Calculate the error for the epoch
                self.all_t_hat = np.array([self.run_forward(np.array(xy[0:2])) for xy in self.data])
                self.total_error = abs(np.array([t[2] for t in self.data]) - self.all_t_hat).sum()

                self.update_run_status()
                self.draw_decision_boundary()
                self.data_missclasified, error = [], 0
                for xy in self.data:
                    t_hat = self.run_forward(np.array(xy[0:2]))
                    if t_hat != xy[2]:
                        self.data_missclasified.append(True)
                        error += 1
                    else:
                        self.data_missclasified.append(False)
                self.draw_data()

                self.epoch_label.setText("Epochs so far: {}".format(self.total_epochs))
                self.error_label.setText("Error: {}".format(self.total_error))

                return self.pos_line, self.neg_line, self.miss_line_pos, self.miss_line_neg, self.decision, self.highlight_data

    def on_reset(self):
        if self.ani:
            self.ani.event_source.stop()
        self.initialize_weights()
        self.total_epochs = 0
        self.update_run_status()
        self.clear_decision_boundary()
        self.canvas.draw()

    def on_undo_mouseclick(self):
        if self.ani:
            self.ani.event_source.stop()
        if self.data:
            self.data.pop()
            self.draw_data()
            self.canvas.draw()

    def run_forward(self, p):
        """Given an input of dimension R, run the network"""
        return hardlim(self.Weights.dot(p) + self.bias)

    def train_one_iteration(self, p, t):
        """Given one input of dimension R and its target, perform one training iteration.
        Update the weights and biases using the Perceptron learning Rule."""

        t_hat = self.run_forward(p)
        self.error = t - t_hat

        # Adjust weights and bias based on the error from this iteration
        self.Weights = self.Weights + self.error * p.T
        self.bias = self.bias + self.error
        return self.error

    def find_decision_boundary(self, x):
        """Returns the corresponding y value for the input x on the decision
        boundary"""
        return -(x * self.Weights[0] + self.bias) / \
               (self.Weights[1] if self.Weights[1] != 0 else .000001)

    def initialize_weights(self):
        if self.ani:
            self.ani.event_source.stop()
        self.Weights = (np.random.random(self.R) - 0.5) * 20
        self.bias = (np.random.random(self.S) - 0.5) * 20
