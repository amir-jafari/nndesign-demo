from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


def logsigmoid(n):
    return 1 / (1 + np.exp(-n))


def logsigmoid_stable(n):
    n = np.clip(n, -100, 100)
    return 1 / (1 + np.exp(-n))


def logsigmoid_der(n):
    return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))


def purelin(n):
    return n


def purelin_der(n):
    return np.array([1]).reshape(n.shape)


def lin_delta(a, d=None, w=None):
    na, ma = a.shape
    if d is None and w is None:
        return -np.kron(np.ones((1, ma)), np.eye(na))
    else:
        return np.dot(w.T, d)


def log_delta(a, d=None, w=None):
    s1, _ = a.shape
    if d is None and w is None:
        return -np.kron((1 - a) * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
    else:
        return (1 - a) * a * np.dot(w.T, d)


def marq(p, d):
    s, _ = d.shape
    r, _ = p.shape
    return np.kron(p.T, np.ones((1, s))) * np.kron(np.ones((1, r)), d.T)


mu_initial = 0.01
mingrad = 0.001


class FunctionApproximation(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(FunctionApproximation, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("Function Approximation", 11, "Click the train button to train\n the log-sig ...",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")  # TODO: Logo and Icon

        self.S1 = 4
        self.diff = 1
        self.p = np.linspace(-2, 2, 100)
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.mu = None
        self.ani = None
        self.random_state = 0
        self.init_params()
        self.error_prev, self.ii = 1000, None
        self.RS, self.RS1, self.RSS, self.RSS1 = None, None, None, None
        self.RSS2, self.RSS3, self.RSS4 = None, None, None

        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.2, left=0.1)
        self.axes.set_xlim(-2, 2)
        self.axes.set_ylim(0, 2)
        self.axes.tick_params(labelsize=8)
        self.axes.set_xlabel("Input", fontsize=10)
        self.axes.xaxis.set_label_coords(0.5, 0.1)
        self.axes.set_ylabel("Target", fontsize=10)
        self.axes.yaxis.set_label_coords(0.05, 0.5)
        self.data_to_approx, = self.axes.plot([], label="Function to Approximate")
        self.net_approx, = self.axes.plot([], label="Network Approximation")
        self.axes.legend(loc='lower center', fontsize=8, framealpha=0.9, numpoints=1, ncol=3,
                         bbox_to_anchor=(0, -.24, 1, -.280), mode='expand')
        self.axes.set_title("Function Approximation")
        self.plot_f()

        self.label_s1 = QtWidgets.QLabel(self)
        self.label_s1.setText("Number of Hidden Neurons S1: 4")
        self.label_s1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_s1.setGeometry((self.x_chapter_slider_label - 60) * self.w_ratio, 400 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_s1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_s1.setRange(1, 9)
        self.slider_s1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_s1.setTickInterval(1)
        self.slider_s1.setValue(4)

        self.label_diff = QtWidgets.QLabel(self)
        self.label_diff.setText("Difficulty index: 1")
        self.label_diff.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_diff.setGeometry((self.x_chapter_slider_label - 10) * self.w_ratio, 470 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_diff = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_diff.setRange(1, 9)
        self.slider_diff.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_diff.setTickInterval(1)
        self.slider_diff.setValue(1)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_s1)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_diff)
        self.wid4.setLayout(self.layout4)

        self.slider_s1.valueChanged.connect(self.slide)
        self.slider_diff.valueChanged.connect(self.slide)

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 600 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def slide(self):
        self.error_prev = 1000
        if self.ani:
            self.ani.event_source.stop()
        slider_s1 = self.slider_s1.value()
        if self.S1 != slider_s1:
            self.S1 = slider_s1
            self.init_params()
        self.diff = self.slider_diff.value()
        self.label_s1.setText("Number of Hidden Neurons S1: {}".format(self.S1))
        self.label_diff.setText("Difficulty Index: {}".format(self.diff))
        self.f_to_approx = lambda p: 1 + np.sin(np.pi * p * self.diff / 5)
        self.net_approx.set_data([], [])
        self.plot_f()

    def init_params(self):
        # np.random.seed(self.random_state)
        self.W1 = 2 * np.random.uniform(0, 1, (self.S1, 1)) - 1
        self.b1 = 2 * np.random.uniform(0, 1, (self.S1, 1)) - 1
        self.W2 = 2 * np.random.uniform(0, 1, (1, self.S1)) - 1
        self.b2 = 2 * np.random.uniform(0, 0, (1, 1)) - 1

    def plot_f(self):
        self.data_to_approx.set_data(self.p, 1 + np.sin(np.pi * self.p * self.diff / 5))
        self.canvas.draw()

    def f_to_approx(self, p):
        return 1 + np.sin(np.pi * p * self.diff / 5)

    # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    def on_run(self):
        if self.ani:
            self.ani.event_source.stop()
        n_epochs = 500
        self.ani = FuncAnimation(self.figure, self.on_animate_v2, init_func=self.animate_init_v2, frames=n_epochs,
                                 interval=20, repeat=False, blit=True)

    def animate_init(self):
        self.net_approx.set_data([], [])
        return self.net_approx,

    def animate_init_v2(self):
        self.p = self.p.reshape(1, -1)
        self.mu = mu_initial
        self.RS = self.S1 * 1
        self.RS1 = self.RS + 1
        self.RSS = self.RS + self.S1
        self.RSS1 = self.RSS + 1
        self.RSS2 = self.RSS + self.S1 * 1
        self.RSS3 = self.RSS2 + 1
        self.RSS4 = self.RSS2 + 1
        self.ii = np.eye(self.RSS4)
        self.net_approx.set_data([], [])
        return self.net_approx,

    def on_animate_v2(self, idx):

        a1 = logsigmoid_stable(np.dot(self.W1, self.p) + self.b1)
        a2 = purelin(np.dot(self.W2, a1) + self.b2)
        e = self.f_to_approx(self.p) - a2
        error = np.dot(e, e.T).item()

        if error <= 0.005:
            print("Error goal reached!")
            self.net_approx.set_data(self.p.reshape(-1), a2.reshape(-1))
            return self.net_approx,

        self.mu /= 10

        while error >= self.error_prev:
            try:

                a1 = np.kron(a1, np.ones((1, 1)))
                d2 = lin_delta(a2)
                d1 = log_delta(a1, d2, self.W2)
                jac1 = marq(np.kron(self.p, np.ones((1, 1))), d1)  # Does it need a row of 0s?
                jac2 = marq(a1, d2)
                jac = np.hstack((jac1, d1.T))
                jac = np.hstack((jac, jac2))
                jac = np.hstack((jac, d2.T))
                je = np.dot(jac.T, e.T)

                grad = np.sqrt(np.dot(je.T, je)).item()
                if grad < mingrad:
                    self.net_approx.set_data(self.p.reshape(-1), a2.reshape(-1))
                    return self.net_approx,

                # Can't get this operation to produce the same results as MATLAB...
                dw = -np.dot(np.linalg.inv(np.dot(jac.T, jac) + self.mu * self.ii), je)
                self.W1 += dw[:self.RS]
                self.b1 += dw[self.RS:self.RSS]
                self.W2 += dw[self.RSS:self.RSS2].reshape(1, -1)
                self.b2 += dw[self.RSS2].reshape(1, 1)

                a1 = logsigmoid_stable(np.dot(self.W1, self.p) + self.b1)
                a2 = purelin(np.dot(self.W2, a1) + self.b2)
                e = self.f_to_approx(self.p) - a2
                error = np.dot(e, e.T).item()

                self.mu *= 10
                if self.mu > 1e10:
                    break

            except Exception as e:
                if str(e) == "Singular matrix":
                    self.mu *= 10
                else:
                    raise e

        if error < self.error_prev:
            self.error_prev = error

        self.net_approx.set_data(self.p.reshape(-1), a2.reshape(-1))
        return self.net_approx,

    def on_animate(self, idx):
        alpha = 0.03
        nn_output = []
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
        self.net_approx.set_data(self.p, nn_output)
        return self.net_approx,
