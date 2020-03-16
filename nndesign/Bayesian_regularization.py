from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


max_epoch = 101

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


class BayesianRegularization(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(BayesianRegularization, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Bayesian Regularization", 9, "",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.train_error, self.error_train = [], None
        self.test_error, self.error_test = [], None
        self.gamma_list = []
        self.ani_1, self.ani_2, self.ani_3 = None, None, None
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.random_state = 42
        np.random.seed(self.random_state)
        self.tt, self.t = None, None

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Function F", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-1, 1)
        self.axes_1.set_ylim(-1.5, 1.5)
        self.axes_1_blue_line, = self.axes_1.plot([], [], color="blue")
        self.net_approx, = self.axes_1.plot([], linestyle="--", color="red")
        self.train_points, = self.axes_1.plot([], marker='*', label="Train", linestyle="")
        # self.test_points, = self.axes_1.plot([], marker='.', label="Test", linestyle="")
        self.axes_1.legend()
        self.canvas.draw()

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation Fa", fontdict={'fontsize': 10})
        self.train_e, = self.axes_2.plot([], [], linestyle='-', color="blue", label="train error")
        self.test_e, = self.axes_2.plot([], [], linestyle='-', color="black", label="test error")
        self.gamma, = self.axes_2.plot([], [], linestyle='-', color="red", label="gamma")
        self.axes_2.legend()
        self.axes_2.set_xlim(0, max_epoch)
        self.axes_2.set_ylim(-0.1, 20)
        self.canvas2.draw()

        self.nsd = 1
        self.label_nsd = QtWidgets.QLabel(self)
        self.label_nsd.setText("Noise standard deviation: 1")
        self.label_nsd.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_nsd.setGeometry((self.x_chapter_slider_label - 50) * self.w_ratio, 100 * self.h_ratio,
                                   self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_nsd = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_nsd.setRange(0, 30)
        self.slider_nsd.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_nsd.setTickInterval(1)
        self.slider_nsd.setValue(10)
        self.wid_nsd = QtWidgets.QWidget(self)
        self.layout_nsd = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_nsd.setGeometry(self.x_chapter_usual * self.w_ratio, 130 * self.h_ratio,
                                 self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_nsd.addWidget(self.slider_nsd)
        self.wid_nsd.setLayout(self.layout_nsd)

        self.animation_speed = 100
        self.label_anim_speed = QtWidgets.QLabel(self)
        self.label_anim_speed.setText("Animation Delay: 100 ms")
        self.label_anim_speed.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_anim_speed.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 200 * self.h_ratio,
                                          self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_anim_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_anim_speed.setRange(0, 6)
        self.slider_anim_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_anim_speed.setTickInterval(1)
        self.slider_anim_speed.setValue(1)
        self.wid_anim_speed = QtWidgets.QWidget(self)
        self.layout_anim_speed = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_anim_speed.setGeometry(self.x_chapter_usual * self.w_ratio, 230 * self.h_ratio,
                                        self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_anim_speed.addWidget(self.slider_anim_speed)
        self.wid_anim_speed.setLayout(self.layout_anim_speed)

        self.S1 = 20
        self.label_S1 = QtWidgets.QLabel(self)
        self.label_S1.setText("# Hidden Neurons: 20")
        self.label_S1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_S1.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 300 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_S1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_S1.setRange(2, 40)
        self.slider_S1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_S1.setTickInterval(1)
        self.slider_S1.setValue(20)
        self.wid_S1 = QtWidgets.QWidget(self)
        self.layout_S1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_S1.setGeometry(self.x_chapter_usual * self.w_ratio, 330 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_S1.addWidget(self.slider_S1)
        self.wid_S1.setLayout(self.layout_S1)

        self.n_points = 21
        self.label_n_points = QtWidgets.QLabel(self)
        self.label_n_points.setText("# Data Points: 21")
        self.label_n_points.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_n_points.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 400 * self.h_ratio,
                                        self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_n_points = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_n_points.setRange(10, 40)
        self.slider_n_points.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_n_points.setTickInterval(1)
        self.slider_n_points.setValue(21)
        self.wid_n_points = QtWidgets.QWidget(self)
        self.layout_n_points = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_n_points.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio,
                                      self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_n_points.addWidget(self.slider_n_points)
        self.wid_n_points.setLayout(self.layout_n_points)

        self.freq = 1
        self.label_freq = QtWidgets.QLabel(self)
        self.label_freq.setText("Frequency: 1.00")
        self.label_freq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_freq.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 500 * self.h_ratio,
                                    self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_freq = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_freq.setRange(50, 400)
        self.slider_freq.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_freq.setTickInterval(1)
        self.slider_freq.setValue(100)
        self.wid_freq = QtWidgets.QWidget(self)
        self.layout_freq = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_freq.setGeometry(self.x_chapter_usual * self.w_ratio, 530 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_freq.addWidget(self.slider_freq)
        self.wid_freq.setLayout(self.layout_freq)

        self.slider_nsd.valueChanged.connect(self.slide)
        self.slider_anim_speed.valueChanged.connect(self.slide)
        self.slider_S1.valueChanged.connect(self.slide)
        self.slider_n_points.valueChanged.connect(self.slide)
        self.slider_freq.valueChanged.connect(self.slide)
        self.plot_train_test_data()

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 620 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)
        self.init_params()
        self.full_batch = False

        self.pp = np.linspace(-0.95, 0.95, self.n_points)

    def animate_init_1(self):
        self.init_params()
        self.error_goal_reached = False
        self.a1 = self.tansig(np.dot(self.W1, self.pp.reshape(1, -1)) + self.b1)
        self.a2 = self.purelin(np.dot(self.W2, self.a1) + self.b2)
        self.e = self.tt.reshape(1, -1) - self.a2
        self.error_prev = np.dot(self.e, self.e.T).item()
        self.gamk = self.S1 * 3 + 1
        if self.error_prev == 0:
            self.beta = 1
        else:
            self.beta = (self.e.shape[1] - self.gamk) / (2 * self.error_prev)
        if self.beta <= 0:
            self.beta = 1
        self.reg = 0
        for param in [self.W1, self.b1, self.W2, self.b2]:
            self.reg += np.dot(param.reshape(1, -1), param.reshape(-1, 1)).item()
        self.alpha = self.gamk / (2 * self.reg)
        self.f1 = self.beta * self.error_prev + self.alpha * self.reg
        self.mu = 10
        self.RS = self.S1 * 1
        self.RS1 = self.RS + 1
        self.RSS = self.RS + self.S1
        self.RSS1 = self.RSS + 1
        self.RSS2 = self.RSS + self.S1 * 1
        self.RSS3 = self.RSS2 + 1
        self.RSS4 = self.RSS2 + 1
        self.ii = np.eye(self.RSS4)
        self.train_e.set_data([], [])
        self.test_e.set_data([], [])
        self.gamma.set_data([], [])
        self.net_approx.set_data([], [])
        return self.train_e, self.test_e, self.gamma

    def animate_init_2(self):
        self.net_approx.set_data([], [])
        return self.net_approx,

    def on_animate_1(self, idx):
        self.error_train, self.error_test, gamma = self.train_v2()
        self.gamma_list.append(gamma)
        self.train_error.append(self.error_train)
        self.train_e.set_data(list(range(len(self.train_error))), self.train_error)
        self.test_error.append(self.error_test)
        self.test_e.set_data(list(range(len(self.test_error))), self.test_error)
        self.gamma.set_data(list(range(len(self.gamma_list))), self.gamma_list)
        return self.train_e, self.test_e, self.gamma

    def on_animate_2(self, idx):
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
        self.net_approx.set_data([], [])
        self.train_error, self.test_error, self.gamma_list = [], [], []
        self.canvas.draw()
        self.canvas2.draw()
        self.run_animation()

    def run_animation(self):
        if self.ani_1:
            self.ani_1.event_source.stop()
        if self.ani_2:
            self.ani_2.event_source.stop()
        self.ani_1 = FuncAnimation(self.figure2, self.on_animate_1, init_func=self.animate_init_1, frames=max_epoch,
                                   interval=self.animation_speed, repeat=False, blit=True)
        self.ani_2 = FuncAnimation(self.figure, self.on_animate_2, init_func=self.animate_init_2, frames=max_epoch,
                                   interval=self.animation_speed, repeat=False, blit=True)

    def slide(self):
        self.init_params()
        np.random.seed(self.random_state)
        self.nsd = float(self.slider_nsd.value() / 10)
        self.label_nsd.setText("Noise standard deviation: " + str(self.nsd))
        self.plot_train_test_data()
        self.animation_speed = int(self.slider_anim_speed.value()) * 100
        self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
        self.S1 = int(self.slider_S1.value())
        self.label_S1.setText("# Hidden Neurons: " + str(self.S1))
        self.n_points = int(self.slider_n_points.value())
        self.label_n_points.setText("# Data Points: " + str(self.n_points))
        self.freq = int(self.slider_freq.value()) / 100
        self.label_freq.setText("Frequency: " + str(self.freq))
        if self.ani_1:
            self.ani_1.event_source.stop()
        if self.ani_2:
            self.ani_2.event_source.stop()
        self.train_error, self.test_error, self.gamma_list = [], [], []
        self.net_approx.set_data([], [])
        self.canvas.draw()
        self.canvas2.draw()
        self.run_animation()

    def plot_train_test_data(self):
        self.axes_1_blue_line.set_data(pp0, np.sin(2 * np.pi * pp0 * self.freq / T))
        self.pp = np.linspace(-0.95, 0.95, self.n_points)
        self.tt = np.sin(2 * np.pi * self.pp * self.freq / T) + np.random.uniform(-2, 2, self.pp.shape) * 0.2 * self.nsd
        self.train_points.set_data(self.pp, self.tt)
        self.t = np.sin(2 * np.pi * p * self.freq / T) + np.random.uniform(-2, 2, p.shape) * 0.2 * self.nsd
        # self.test_points.set_data(p, self.t)

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
        #  Hidden Layer's Transformation
        a1 = self.tansig(n1)
        # Output Layer's Net Input
        n2 = np.dot(self.W2, a1) + self.b2
        # Output Layer's Transformation
        return purelin(n2), n2, n1, a1, a0

    def train(self):
        alpha = 0.03

        error_train, dw1, db1, dw2, db2 = [], 0, 0, 0, 0
        for sample, target in zip(self.pp, self.tt):
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

            if self.full_batch:
                dw1 += np.dot(s1, a0.T)
                db1 += s1
                dw2 += np.dot(s, a1.T)
                db2 += s
            else:
                # Updates the weights and biases
                # Hidden Layer
                self.W1 += -alpha * np.dot(s1, a0.T)
                self.b1 += -alpha * s1
                # Output Layer
                self.W2 += -alpha * np.dot(s, a1.T)
                self.b2 += -alpha * s

        if self.full_batch:
            # Updates the weights and biases
            # Hidden Layer
            self.W1 += -alpha * dw1
            self.b1 += -alpha * db1
            # Output Layer
            self.W2 += -alpha * dw2
            self.b2 += -alpha * db2

        error_test = []
        for sample, target in zip(p, self.t):
            a, n2, n1, a1, a0 = self.forward(sample)
            e = target - a
            error_test.append(e)

        return np.sum(np.abs(error_train)), np.sum(np.abs(error_test))

    def train_v2(self):

        self.a1 = np.kron(self.a1, np.ones((1, 1)))
        d2 = self.lin_delta(self.a2)
        d1 = self.tan_delta(self.a1, d2, self.W2)
        jac1 = self.marq(np.kron(self.pp.reshape(1, -1), np.ones((1, 1))), d1)
        jac2 = self.marq(self.a1, d2)
        jac = np.hstack((jac1, d1.T))
        jac = np.hstack((jac, jac2))
        jac = np.hstack((jac, d2.T))
        je = np.dot(jac.T, self.e.T)

        grad = np.sqrt(np.dot(je.T, je)).item()
        if grad < 1e-8:
            error_test = []
            for sample, target in zip(p, self.t):
                a, n2, n1, a1, a0 = self.forward(sample)
                e = target - a
                error_test.append(e)
            return self.error_prev, np.sum(np.abs(error_test)), self.gamk

        jj = np.dot(jac.T, jac)
        w = np.copy(self.W1.reshape(-1, 1))
        for param in [self.b1, self.W2, self.b2]:
            w = np.vstack((w, param.reshape(-1, 1)))

        while self.mu < 1e10:

            dw = -np.dot(np.linalg.inv(self.beta * jj + (self.mu + self.alpha) * self.ii),
                         self.beta * je + self.alpha * w)
            dW1 = dw[:self.RS]
            db1 = dw[self.RS:self.RSS]
            dW2 = dw[self.RSS:self.RSS2].reshape(1, -1)
            db2 = dw[self.RSS2].reshape(1, 1)

            self.a1 = self.tansig(np.dot((self.W1 + dW1), self.pp.reshape(1, -1)) + self.b1 + db1)
            self.a2 = self.purelin(np.dot((self.W2 + dW2), self.a1) + self.b2 + db2)
            self.e = self.tt.reshape(1, -1) - self.a2
            error = np.dot(self.e, self.e.T).item()
            reg = 0
            for param in [self.W1 + dW1, self.b1 + db1, self.W2 + dW2, self.b2 + db2]:
                reg += np.dot(param.reshape(1, -1), param.reshape(-1, 1)).item()
            f2 = self.beta * error + self.alpha * reg

            if f2 < self.f1:
                self.W1 += dW1
                self.b1 += db1
                self.W2 += dW2
                self.b2 += db2
                self.error_prev = error
                self.reg = reg
                w = np.copy(self.W1.reshape(-1, 1))
                for param in [self.b1, self.W2, self.b2]:
                    w = np.vstack((w, param.reshape(-1, 1)))
                self.mu /= 2
                if self.mu < 1e-20:
                    self.mu = 1e-20
                break
            self.mu *= 2

        self.gamk = self.S1 * 3 + 1 - self.alpha * np.trace(np.linalg.inv(self.beta * jj + self.ii * self.alpha))
        if self.reg == 0:
            self.aplha = 1
        else:
            self.alpha = self.gamk / (2 * self.reg)
        if self.error_prev == 0:
            self.beta = 1
        else:
            self.beta = (self.e.shape[1] - self.gamk) / (2 * self.error_prev)
        self.f1 = self.beta * self.error_prev + self.alpha * self.reg

        if self.error_prev <= 0:
            if self.error_goal_reached:
                print("Error goal reached!")
                self.error_goal_reached = None
            error_test = []
            for sample, target in zip(p, self.t):
                a, n2, n1, a1, a0 = self.forward(sample)
                e = target - a
                error_test.append(e)
            return self.error_prev, np.sum(np.abs(error_test)), self.gamk

        error_test = []
        for sample, target in zip(p, self.t):
            a, n2, n1, a1, a0 = self.forward(sample)
            e = target - a
            error_test.append(e)
        return self.error_prev, np.sum(np.abs(error_test)), self.gamk
