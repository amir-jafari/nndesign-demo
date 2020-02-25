from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
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


P = np.linspace(-2, 2)

x, y = np.linspace(-4, 0+(0.2*22), 22, endpoint=False), np.linspace(-2, 0+(4/31*17), 17, endpoint=False)
X, Y = np.meshgrid(x, y)

a, b, c = np.array([[2, 0], [0, 50]]), np.array([0, 0]), 0
max_epoch = 50

F = (a[0, 0] * X ** 2 + a[0, 1] + a[1, 0] * X * Y + a[1, 1] * Y ** 2) / 2 + b[0] * X + b[1] * Y + c

x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0,
              0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
y = np.copy(x)
X, Y = np.meshgrid(x, y)
F = (Y - X) ** 4 + 8 * X * Y - X + Y + 3
F[F < 0] = 0
F[F > 12] = 12

xs = np.linspace(-2, 2, 100)
ys = np.linspace(-2, 2, 100)
XX, YY = np.meshgrid(xs, ys)
FF = (YY - XX) ** 4 + 8 * XX * YY - XX + YY + 3
FF[FF < 0] = 0
FF[FF > 12] = 12


class SteepestDescentBackprop1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(SteepestDescentBackprop1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Steepest Descent for Quadratic", 9, " Click anywhere to start an\n initial guess. The gradient\n descent path will be shown\n"
                                                               " Modify the learning rate\n by moving the slide bar",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")  # TODO: Change icons

        self.W1, self.b1 = np.array([[10], 10]), np.array([[-5], [5]])
        self.W2, self.b2 = np.array([[1, 1]]), np.array([[-1]])

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(260 * self.w_ratio, 400 * self.h_ratio, 260 * self.w_ratio, 260 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.axes2 = Axes3D(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(10 * self.w_ratio, 400 * self.h_ratio, 260 * self.w_ratio, 260 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)

        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.contour(X, Y, F)
        self.axes.set_xlim(-2, 2)
        self.axes.set_ylim(-2, 2)
        self.path, = self.axes.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.ani, self.event = None, None

        self.axes2.set_title("Function", fontdict={'fontsize': 10})
        """w11, w12 = np.linspace(-5, 15, 100), np.linspace(-5, 15, 100)
        b11, b12 = np.linspace(-5, 15, 100), np.linspace(-5, 15, 100)
        W11, W12 = np.meshgrid(w11, w12)
        B11, B12 = np.meshgrid(b11, b12)
        A11 = logsigmoid(np.dot(W11.reshape((100, 100, 1)), P.reshape(1, -1)) + B11[..., None])
        A12 = logsigmoid(np.dot(W12.reshape((100, 100, 1)), P.reshape(1, -1)) + B12[..., None])
        w21, w22 = np.linspace(-5, 15, 100), np.linspace(-5, 15, 100)
        W21, W22 = np.meshgrid(w21, w22)
        B2 = np.copy(B11)
        A2 = logsigmoid(
            np.dot(W21.reshape((100, 100, 1)), A11.reshape(1, -1)) +
            np.dot(W22.reshape((100, 100, 1)), A12.reshape(1, -1)) + B2[..., None]
        )
        E = self.f_to_approx(P) - A2
        # FF = self.forward()
        self.axes2.plot_surface(W11, W12, E)"""
        self.axes2.view_init(30, 60)
        self.canvas2.draw()

        self.lr = 0.03
        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("lr: 0.001")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(self.x_chapter_slider_label * self.w_ratio, 250 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 6)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(3)
        self.wid_lr = QtWidgets.QWidget(self)
        self.layout_lr = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_lr.setGeometry(self.x_chapter_usual * self.w_ratio, 280 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_lr.addWidget(self.slider_lr)
        self.wid_lr.setLayout(self.layout_lr)

        self.animation_speed = 200
        self.label_anim_speed = QtWidgets.QLabel(self)
        self.label_anim_speed.setText("Animation Delay: 200 ms")
        self.label_anim_speed.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_anim_speed.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 350 * self.h_ratio,
                                          self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_anim_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_anim_speed.setRange(0, 6)
        self.slider_anim_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_anim_speed.setTickInterval(1)
        self.slider_anim_speed.setValue(2)
        self.wid_anim_speed = QtWidgets.QWidget(self)
        self.layout_anim_speed = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_anim_speed.setGeometry(self.x_chapter_usual * self.w_ratio, 380 * self.h_ratio,
                                        self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_anim_speed.addWidget(self.slider_anim_speed)
        self.wid_anim_speed.setLayout(self.layout_anim_speed)

        self.slider_lr.valueChanged.connect(self.slide)
        self.slider_anim_speed.valueChanged.connect(self.slide)
        self.canvas.draw()

    def slide(self):
        self.lr = float(self.slider_lr.value()/100)
        self.label_lr.setText("lr: " + str(self.lr))
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
        self.path, = self.axes.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        return self.path,

    def on_animate(self, idx):
        gradient = np.dot(a, np.array([self.x, self.y])) + b.T
        self.x -= self.lr * gradient[0]
        self.y -= self.lr * gradient[1]
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.path.set_data(self.x_data, self.y_data)
        return self.path,

    def on_mouseclick(self, event):
        self.event = event
        if self.ani:
            self.ani.event_source.stop()
        # self.ani = None
        self.path.set_data([], [])
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.run_animation(event)

    def run_animation(self, event):
        if event.xdata != None and event.xdata != None:
            self.x_data, self.y_data = [event.xdata], [event.ydata]
            self.x, self.y = event.xdata, event.ydata
            self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=max_epoch,
                                     interval=self.animation_speed, repeat=False, blit=True)

    def forward(self, W11, W12, p):
        n1 = np.dot(W11, p) + self.b1
        #  Hidden Layer's Transformation
        a1 = logsigmoid(n1)
        # Output Layer's Net Input
        n2 = np.dot(self.W2, a1) + self.b2
        # Output Layer's Transformation
        a = logsigmoid(n2)  # (a2 = a)

        # Back-propagates the sensitivities
        # Compares our NN's output with the real value
        e = self.f_to_approx(p) - a

        return e

    def f_to_approx(self, p):
        return 1 + np.sin(np.pi * p * 4 / 5)
