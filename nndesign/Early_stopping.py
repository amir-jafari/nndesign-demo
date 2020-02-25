from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


max_epoch = 1000

T = 2
pp0 = np.linspace(-1, 1, 201)
tt0 = np.sin(2 * np.pi * pp0 / T)

pp = np.linspace(-0.95, 0.95, 20)
p = np.linspace(-1, 1, 21)


def logsigmoid(n):
    return 1 / (1 + np.exp(-n))


def logsigmoid_der(n):
    return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))


def purelin(n):
    return n


def purelin_der(n):
    return np.array([1]).reshape(n.shape)


class EarlyStopping(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(EarlyStopping, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Steepest Descent", 9, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.train_error, self.error_train = [], None
        self.test_error, self.error_test = [], None
        self.ani_1, self.ani_2, self.ani_3 = None, None, None
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.S1, self.random_state = 10, 42
        np.random.seed(self.random_state)
        self.tt, self.t = None, None

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Function F", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-1, 1)
        self.axes_1.set_ylim(-1.5, 1.5)
        self.axes_1.plot(pp0, np.sin(2 * np.pi * pp0 / T))
        self.net_approx, = self.axes_1.plot([], linestyle="--")
        self.train_points, = self.axes_1.plot([], marker='*', label="Train", linestyle="")
        self.test_points, = self.axes_1.plot([], marker='.', label="Test", linestyle="")
        self.axes_1.legend()
        self.canvas.draw()

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation Fa", fontdict={'fontsize': 10})
        self.train_e, = self.axes_2.plot([], linestyle='-', color="b")
        self.test_e, = self.axes_2.plot([], linestyle='-', color="r")
        self.axes_2.set_xlim(0, max_epoch)
        self.axes_2.set_ylim(0, 20)
        self.canvas2.draw()

        self.nsd = 1
        self.label_nsd = QtWidgets.QLabel(self)
        self.label_nsd.setText("Noise standard deviation: 1")
        self.label_nsd.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_nsd.setGeometry((self.x_chapter_slider_label - 50) * self.w_ratio, 250 * self.h_ratio,
                                   self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_nsd = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_nsd.setRange(0, 30)
        self.slider_nsd.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_nsd.setTickInterval(1)
        self.slider_nsd.setValue(10)
        self.wid_nsd = QtWidgets.QWidget(self)
        self.layout_nsd = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_nsd.setGeometry(self.x_chapter_usual * self.w_ratio, 280 * self.h_ratio,
                                 self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_nsd.addWidget(self.slider_nsd)
        self.wid_nsd.setLayout(self.layout_nsd)

        self.animation_speed = 500
        self.label_anim_speed = QtWidgets.QLabel(self)
        self.label_anim_speed.setText("Animation Delay: 500 ms")
        self.label_anim_speed.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_anim_speed.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 350 * self.h_ratio,
                                          self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_anim_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_anim_speed.setRange(0, 6)
        self.slider_anim_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_anim_speed.setTickInterval(1)
        self.slider_anim_speed.setValue(5)
        self.wid_anim_speed = QtWidgets.QWidget(self)
        self.layout_anim_speed = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_anim_speed.setGeometry(self.x_chapter_usual * self.w_ratio, 380 * self.h_ratio,
                                        self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_anim_speed.addWidget(self.slider_anim_speed)
        self.wid_anim_speed.setLayout(self.layout_anim_speed)

        self.slider_nsd.valueChanged.connect(self.slide)
        self.slider_anim_speed.valueChanged.connect(self.slide)
        self.plot_train_test_data()

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 500 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def animate_init_1(self):
        self.train_e, = self.axes_2.plot([], linestyle='--')
        return self.train_e,

    def animate_init_2(self):
        self.test_e, = self.axes_2.plot([], linestyle='-', color="r")
        return self.test_e,

    def animate_init_3(self):
        self.net_approx.set_data([], [])
        return self.net_approx,

    def on_animate_1(self, idx):
        self.error_train, self.error_test = self.train()
        self.train_error.append(self.error_train)
        self.train_e.set_data(list(range(len(self.train_error))), self.train_error)
        print("---", self.error_train)
        return self.train_e,

    def on_animate_2(self, idx):
        self.test_error.append(self.error_test)
        self.test_e.set_data(list(range(len(self.test_error))), self.test_error)
        print(self.error_test)
        return self.test_e,

    def on_animate_3(self, idx):
        nn_output = []
        for sample, target in zip(pp0, tt0):
            a, n2, n1, a1, a0 = self.forward(sample)
            nn_output.append(a)
        self.net_approx.set_data(pp0, nn_output)
        return self.net_approx,

    def on_run(self):
        self.init_params()
        if self.ani_1:
            self.ani_1.event_source.stop()
        if self.ani_2:
            self.ani_2.event_source.stop()
        if self.ani_3:
            self.ani_3.event_source.stop()
        self.train_error, self.test_error = [], []
        self.train_e.set_data([], [])
        self.test_e.set_data([], [])
        self.net_approx.set_data([], [])
        self.canvas.draw()
        self.canvas2.draw()
        self.run_animation()

    def run_animation(self):
        self.ani_1 = FuncAnimation(self.figure2, self.on_animate_1, init_func=self.animate_init_1, frames=max_epoch,
                                   interval=self.animation_speed, repeat=False, blit=True)
        self.ani_2 = FuncAnimation(self.figure2, self.on_animate_2, init_func=self.animate_init_2, frames=max_epoch,
                                   interval=self.animation_speed, repeat=False, blit=True)
        self.ani_3 = FuncAnimation(self.figure, self.on_animate_3, init_func=self.animate_init_3, frames=max_epoch,
                                   interval=self.animation_speed, repeat=False, blit=True)

    def slide(self):
        np.random.seed(self.random_state)
        self.nsd = float(self.slider_nsd.value() / 10)
        self.label_nsd.setText("Noise standard deviation: " + str(self.nsd))
        self.plot_train_test_data()
        self.animation_speed = int(self.slider_anim_speed.value()) * 100
        self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
        if self.train_error:
            if self.ani_1:
                self.ani_1.event_source.stop()
            if self.ani_2:
                self.ani_2.event_source.stop()
            if self.ani_3:
                self.ani_3.event_source.stop()
            self.train_error, self.test_error = [], []
            self.net_approx.set_data([], [])
            self.canvas.draw()
            self.canvas2.draw()
            self.run_animation()

    def plot_train_test_data(self):
        self.tt = np.sin(2 * np.pi * pp / T) + np.random.uniform(-2, 2, pp.shape) * 0.2 * self.nsd
        self.train_points.set_data(pp, self.tt)
        self.t = np.sin(2 * np.pi * p / T) + np.random.uniform(-2, 2, p.shape) * 0.2 * self.nsd
        self.test_points.set_data(p, self.t)

    def init_params(self):
        np.random.seed(self.random_state)
        self.W1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.b1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (1, self.S1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def forward(self, sample):
        a0 = sample.reshape(-1, 1)
        # Hidden Layer's Net Input
        n1 = np.dot(self.W1, a0) + self.b1
        #  Hidden Layer's Transformation
        a1 = logsigmoid(n1)
        # Output Layer's Net Input
        n2 = np.dot(self.W2, a1) + self.b2
        # Output Layer's Transformation
        return purelin(n2), n2, n1, a1, a0

    def train(self):
        alpha = 0.03

        error_train = []
        for sample, target in zip(pp, self.tt):
            a, n2, n1, a1, a0 = self.forward(sample)
            e = target - a
            error_train.append(e)
            # error = np.append(error, e)
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

        error_test = []
        for sample, target in zip(p, self.t):
            a, n2, n1, a1, a0 = self.forward(sample)
            e = target - a
            error_test.append(e)

        return np.sum(np.abs(error_train)), np.sum(np.abs(error_test))
