from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from scipy.io import loadmat
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

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

W1 = np.array([[10], [10]])
b1 = np.array([[-5], [5]])
W2 = np.array([[1, 1]])
b2 = np.array([-1])
P = np.arange(-2, 2.1, 0.1).reshape(1, -1)
A1 = logsigmoid(np.dot(W1, P) + b1)
T = logsigmoid(np.dot(W2, A1) + b2)
tau1 = 1 - 0.618
delta = 0.32
tol = 0.03 / 20
scale = 2
b_max = 26
n = 2


class ConjugateGradient(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(ConjugateGradient, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Conjugate Gradient", 9, " Click anywhere to start an\n initial guess. The gradient\n descent path will be shown\n"
                                                               " Modify the learning rate\n by moving the slide bar",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)  # TODO: Change icons

        self.W1, self.b1 = np.array([[10], 10]), np.array([[-5], [5]])
        self.W2, self.b2 = np.array([[1, 1]]), np.array([[-1]])
        self.lr, self.epochs = None, None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(50 * self.w_ratio, 200 * self.h_ratio, 450 * self.w_ratio, 450 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.axes = self.figure.add_subplot(1, 1, 1)
        self.path, = self.axes.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.ani, self.event = None, None

        self.pair_of_params = 1
        self.pair_params = [["W1(1, 1)", "W2(1, 1)"], ["W1(1, 1)", "b1(1)"], ["b1(1)", "b1(2)"]]
        self.plot_data()

        self.x, self.y = None, None

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.addItems(["W1(1, 1), W2(1, 1)", 'W1(1, 1), b1(1)', 'b1(1), b1(2)'])
        self.label_combo = QtWidgets.QLabel(self)
        self.label_combo.setText("Pair of parameters")
        self.label_combo.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_combo.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 550 * self.h_ratio,
                                     150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.animation_speed = 100
        self.label_anim_speed = QtWidgets.QLabel(self)
        self.label_anim_speed.setText("Animation Delay: 100 ms")
        self.label_anim_speed.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_anim_speed.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 460 * self.h_ratio,
                                          self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_anim_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_anim_speed.setRange(0, 6)
        self.slider_anim_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_anim_speed.setTickInterval(1)
        self.slider_anim_speed.setValue(1)
        self.wid_anim_speed = QtWidgets.QWidget(self)
        self.layout_anim_speed = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_anim_speed.setGeometry(self.x_chapter_usual * self.w_ratio, 490 * self.h_ratio,
                                        self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_anim_speed.addWidget(self.slider_anim_speed)
        self.wid_anim_speed.setLayout(self.layout_anim_speed)

        self.comboBox1.currentIndexChanged.connect(self.change_pair_of_params)
        self.slider_anim_speed.valueChanged.connect(self.slide)
        self.canvas.draw()

    def change_pair_of_params(self, idx):
        self.pair_of_params = idx + 1
        self.init_params()
        self.plot_data()

    def plot_data(self):
        self.x_data = []
        self.y_data = []
        self.path.set_data(self.x_data, self.y_data)
        while self.axes.collections:
            for collection in self.axes.collections:
                collection.remove()
        f_data = loadmat("nndbp{}.mat".format(self.pair_of_params))
        x1, y1 = np.meshgrid(f_data["x1"], f_data["y1"])
        self.axes.contour(x1, y1, f_data["E1"], list(f_data["levels"].reshape(-1)))
        if self.pair_of_params == 1:
            self.axes.set_xlim(-5, 15)
            self.axes.set_ylim(-5, 15)
            self.axes.set_xticks([-5, 0, 5, 10])
            self.axes.set_yticks([-5, 0, 5, 10])
        elif self.pair_of_params == 2:
            self.axes.set_xlim(-10, 30)
            self.axes.set_ylim(-20, 10)
            self.axes.set_xticks([-10, 0, 10, 20])
            self.axes.set_yticks([-20, -15, -10, -5, 0, 5])
        elif self.pair_of_params == 3:
            self.axes.set_xlim(-10, 10)
            self.axes.set_ylim(-10, 10)
            self.axes.set_xticks([-10, -5, 0, 5])
            self.axes.set_xticks([-10, -5, 0, 5])
        self.axes.set_xlabel(self.pair_params[self.pair_of_params - 1][0], fontsize=8)
        self.axes.xaxis.set_label_coords(0.95, -0.025)
        self.axes.set_ylabel(self.pair_params[self.pair_of_params - 1][1], fontsize=8)
        self.axes.yaxis.set_label_coords(-0.025, 0.95)
        self.canvas.draw()

    def slide(self):
        self.animation_speed = int(self.slider_anim_speed.value()) * 100
        self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
        if self.x_data:
            if self.ani:
                self.ani.event_source.stop()
            self.path.set_data([], [])
            self.x_data, self.y_data = [self.x_data[0]], [self.y_data[0]]
            self.canvas.draw()
            self.run_animation(self.event)

    def animate_init(self):
        self.path.set_data(self.x_data, self.y_data)
        self.n1 = np.dot(self.W1, P) + self.b1
        self.a1 = logsigmoid(self.n1)
        self.n2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = logsigmoid(self.n2)
        self.e = T - self.a2
        self.fa = np.sum(self.e * self.e)
        self.D2 = self.a2 * (1 - self.a2) * self.e
        self.D1 = self.a1 * (1 - self.a1) * np.dot(self.W2.T, self.D2)
        self.gW1 = np.dot(self.D1, P.T)
        self.gb1 = np.dot(self.D1, np.ones((self.D1.shape[1], 1)))
        self.gW2 = np.dot(self.D2, self.a1.T)
        self.gb2 = np.dot(self.D2, np.ones((self.D1.shape[1], 1)))
        if self.pair_of_params == 1:
            self.nrmo = self.gW1[0, 0] ** 2 + self.gW2[0, 0] ** 2
        elif self.pair_of_params == 2:
            self.nrmo = self.gW1[0, 0] ** 2 + self.gb1[0] ** 2
        elif self.pair_of_params == 3:
            self.nrmo = self.gb1[0] ** 2 + self.gb1[1] ** 2
        self.nrmrt = np.sqrt(self.nrmo)
        self.dW1_old, self.db1_old, self.dW2_old, self.db2_old = self.gW1, self.gb1, self.gW2, self.gb2
        self.dW1, self.db1, self.dW2, self.db2 = self.gW1 / self.nrmrt, self.gb1 / self.nrmrt, self.gW2 / self.nrmrt, self.gb2 / self.nrmrt
        return self.path,

    def on_animate(self, idx):

        a = 0
        aold = 0
        b = delta
        faold = self.fa

        W1t, b1t, W2t, b2t = np.copy(self.W1), np.copy(self.b1), np.copy(self.W2), np.copy(self.b2)
        if self.pair_of_params == 1:
            self.x, self.y = self.W1[0, 0] + b * self.dW1[0, 0], self.W2[0, 0] + b * self.dW2[0, 0]
            W1t[0, 0] = self.x
            W2t[0, 0] = self.y
        elif self.pair_of_params == 2:
            self.x, self.y = self.W1[0, 0] + b * self.dW1[0, 0], self.b1[0] + b * self.db1[0]
            W1t[0, 0] = self.x
            b1t[0] = self.y
        elif self.pair_of_params == 3:
            self.x, self.y = self.b1[0] + b * self.db1[0], self.b1[1] + b * self.db1[1]
            b1t[0] = self.x
            b1t[1] = self.y
        n1 = np.dot(W1t, P) + b1t
        a1 = logsigmoid(n1)
        n2 = np.dot(W2t, a1) + b2t
        a2 = logsigmoid(n2)
        e = T - a2
        fb = np.sum(e * e)

        while self.fa > fb and b < b_max:
            aold = a
            faold = self.fa
            self.fa = fb
            a = b
            b *= scale
            if self.pair_of_params == 1:
                self.x, self.y = self.W1[0, 0] + b * self.dW1[0, 0], self.W2[0, 0] + b * self.dW2[0, 0]
                W1t[0, 0] = self.x
                W2t[0, 0] = self.y
            elif self.pair_of_params == 2:
                self.x, self.y = self.W1[0, 0] + b * self.dW1[0, 0], self.b1[0] + b * self.db1[0]
                W1t[0, 0] = self.x
                b1t[0] = self.y
            elif self.pair_of_params == 3:
                self.x, self.y = self.b1[0] + b * self.db1[0], self.b1[1] + b * self.db1[1]
                b1t[0] = self.x
                b1t[1] = self.y
            n1 = np.dot(W1t, P) + b1t
            a1 = logsigmoid(n1)
            n2 = np.dot(W2t, a1) + b2t
            a2 = logsigmoid(n2)
            e = T - a2
            fb = np.sum(e * e)
        a = aold
        self.fa = faold

        c = a + tau1 * (b - a)
        if self.pair_of_params == 1:
            self.x, self.y = self.W1[0, 0] + c * self.dW1[0, 0], self.W2[0, 0] + c * self.dW2[0, 0]
            W1t[0, 0] = self.x
            W2t[0, 0] = self.y
        elif self.pair_of_params == 2:
            self.x, self.y = self.W1[0, 0] + c * self.dW1[0, 0], self.b1[0] + c * self.db1[0]
            W1t[0, 0] = self.x
            b1t[0] = self.y
        elif self.pair_of_params == 3:
            self.x, self.y = self.b1[0] + c * self.db1[0], self.b1[1] + c * self.db1[1]
            b1t[0] = self.x
            b1t[1] = self.y
        n1 = np.dot(W1t, P) + b1t
        a1 = logsigmoid(n1)
        n2 = np.dot(W2t, a1) + b2t
        a2 = logsigmoid(n2)
        e = T - a2
        fc = np.sum(e * e)

        d = b - tau1 * (b - a)
        if self.pair_of_params == 1:
            self.x, self.y = self.W1[0, 0] + d * self.dW1[0, 0], self.W2[0, 0] + d * self.dW2[0, 0]
            W1t[0, 0] = self.x
            W2t[0, 0] = self.y
        elif self.pair_of_params == 2:
            self.x, self.y = self.W1[0, 0] + d * self.dW1[0, 0], self.b1[0] + d * self.db1[0]
            W1t[0, 0] = self.x
            b1t[0] = self.y
        elif self.pair_of_params == 3:
            self.x, self.y = self.b1[0] + d * self.db1[0], self.b1[1] + d * self.db1[1]
            b1t[0] = self.x
            b1t[1] = self.y
        n1 = np.dot(W1t, P) + b1t
        a1 = logsigmoid(n1)
        n2 = np.dot(W2t, a1) + b2t
        a2 = logsigmoid(n2)
        e = T - a2
        fd = np.sum(e * e)

        while b - a > tol:
            if (fc < fd and fb >= np.min([self.fa, fc, fd])) or self.fa < np.min([fb, fc, fd]):
                b = d
                d = c
                fb = fd
                c = a + tau1 * (b - a)
                fd = fc
                if self.pair_of_params == 1:
                    self.x, self.y = self.W1[0, 0] + c * self.dW1[0, 0], self.W2[0, 0] + c * self.dW2[0, 0]
                    W1t[0, 0] = self.x
                    W2t[0, 0] = self.y
                elif self.pair_of_params == 2:
                    self.x, self.y = self.W1[0, 0] + c * self.dW1[0, 0], self.b1[0] + c * self.db1[0]
                    W1t[0, 0] = self.x
                    b1t[0] = self.y
                elif self.pair_of_params == 3:
                    self.x, self.y = self.b1[0] + c * self.db1[0], self.b1[1] + c * self.db1[1]
                    b1t[0] = self.x
                    b1t[1] = self.y
                n1 = np.dot(W1t, P) + b1t
                a1 = logsigmoid(n1)
                n2 = np.dot(W2t, a1) + b2t
                a2 = logsigmoid(n2)
                e = T - a2
                fc = np.sum(e * e)
            else:
                a = c
                c = d
                self.fa = fc
                d = b - tau1 * (b - a)
                fc = fd
                if self.pair_of_params == 1:
                    self.x, self.y = self.W1[0, 0] + d * self.dW1[0, 0], self.W2[0, 0] + d * self.dW2[0, 0]
                    W1t[0, 0] = self.x
                    W2t[0, 0] = self.y
                elif self.pair_of_params == 2:
                    self.x, self.y = self.W1[0, 0] + d * self.dW1[0, 0], self.b1[0] + d * self.db1[0]
                    W1t[0, 0] = self.x
                    b1t[0] = self.y
                elif self.pair_of_params == 3:
                    self.x, self.y = self.b1[0] + d * self.db1[0], self.b1[1] + d * self.db1[1]
                    b1t[0] = self.x
                    b1t[1] = self.y
                n1 = np.dot(W1t, P) + b1t
                a1 = logsigmoid(n1)
                n2 = np.dot(W2t, a1) + b2t
                a2 = logsigmoid(n2)
                e = T - a2
                fd = np.sum(e * e)
        if self.pair_of_params == 1:
            self.x, self.y = self.W1[0, 0] + a * self.dW1[0, 0], self.W2[0, 0] + a * self.dW2[0, 0]
            self.W1[0, 0] = self.x
            self.W2[0, 0] = self.y
        elif self.pair_of_params == 2:
            self.x, self.y = self.W1[0, 0] + a * self.dW1[0, 0], self.b1[0] + a * self.db1[0]
            self.W1[0, 0] = self.x
            self.b1[0] = self.y
        elif self.pair_of_params == 3:
            self.x, self.y = self.b1[0] + a * self.db1[0], self.b1[1] + a * self.db1[1]
            self.b1[0] = self.x
            self.b1[1] = self.y
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.n1 = np.dot(self.W1, P) + self.b1
        self.a1 = logsigmoid(self.n1)
        self.n2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = logsigmoid(self.n2)
        self.e = T - self.a2
        self.D2 = self.a2 * (1 - self.a2) * self.e
        self.D1 = self.a1 * (1 - self.a1) * np.dot(self.W2.T, self.D2)
        self.gW1 = np.dot(self.D1, P.T)
        self.gb1 = np.dot(self.D1, np.ones((self.D1.shape[1], 1)))
        self.gW2 = np.dot(self.D2, self.a1.T)
        self.gb2 = np.dot(self.D2, np.ones((self.D1.shape[1], 1)))
        if self.pair_of_params == 1:
            nrmn = self.gW1[0, 0] ** 2 + self.gW2[0, 0] ** 2
        elif self.pair_of_params == 2:
            nrmn = self.gW1[0, 0] ** 2 + self.gb1[0] ** 2
        elif self.pair_of_params == 3:
            nrmn = self.gb1[0] ** 2 + self.gb1[1] ** 2
        if idx % n == 0:
            Z = nrmn / self.nrmo
        else:
            Z = 0
        self.dW1_old = self.gW1 + self.dW1_old * Z
        self.db1_old = self.gb1 + self.db1_old * Z
        self.dW2_old = self.gW2 + self.dW2_old * Z
        self.db2_old = self.gb2 + self.db2_old * Z
        self.nrmo = nrmn
        if self.pair_of_params == 1:
            nrm = np.sqrt(self.dW1_old[0, 0] ** 2 + self.dW2_old[0, 0] ** 2)
        elif self.pair_of_params == 2:
            nrm = np.sqrt(self.dW1_old[0, 0] ** 2 + self.db1_old[0] ** 2)
        elif self.pair_of_params == 3:
            nrm = np.sqrt(self.db1_old[0] ** 2 + self.db1_old[1] ** 2)
        self.dW1, self.db1, self.dW2, self.db2 = self.dW1_old / nrm, self.db1_old / nrm, self.dW2_old / nrm, self.db2_old / nrm

        self.path.set_data(self.x_data, self.y_data)
        return self.path,

    def on_mouseclick(self, event):
        self.init_params()
        self.event = event
        if self.ani:
            self.ani.event_source.stop()
        self.path.set_data([], [])
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.run_animation(event)

    def run_animation(self, event):
        if event.xdata != None and event.xdata != None:
            self.x_data, self.y_data = [event.xdata], [event.ydata]
            self.x, self.y = event.xdata, event.ydata
            if self.pair_of_params == 1:
                self.W1[0, 0], self.W2[0, 0] = self.x, self.y
            elif self.pair_of_params == 2:
                self.W1[0, 0], self.b1[0, 0] = self.x, self.y
            elif self.pair_of_params == 3:
                self.b1[0, 0], self.b1[1, 0] = self.x, self.y
            self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=15,
                                     interval=self.animation_speed, repeat=False, blit=True)

    def init_params(self):
        self.W1, self.b1 = np.array([[10.], [10.]]), np.array([[-5.], [5.]])
        self.W2, self.b2 = np.array([[1., 1.]]), np.array([[-1.]])
