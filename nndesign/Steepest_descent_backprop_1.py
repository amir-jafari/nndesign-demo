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


class SteepestDescentBackprop1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(SteepestDescentBackprop1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Steepest Descent for Quadratic", 9, " Click anywhere to start an\n initial guess. The gradient\n descent path will be shown\n"
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
        self.path, = self.axes.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.ani, self.event = None, None
        self.axes2.set_title("Sum Sq. Error", fontdict={'fontsize': 10})
        
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
        self.label_anim_speed.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 350 * self.h_ratio,
                                          self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_anim_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_anim_speed.setRange(0, 6)
        self.slider_anim_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_anim_speed.setTickInterval(1)
        self.slider_anim_speed.setValue(1)
        self.wid_anim_speed = QtWidgets.QWidget(self)
        self.layout_anim_speed = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_anim_speed.setGeometry(self.x_chapter_usual * self.w_ratio, 380 * self.h_ratio,
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
        while self.axes2.collections:
            for collection in self.axes2.collections:
                collection.remove()
        f_data = loadmat("nndbp{}.mat".format(self.pair_of_params))
        x1, y1 = np.meshgrid(f_data["x1"], f_data["y1"])
        x2, y2 = np.meshgrid(f_data["x2"], f_data["y2"])
        self.axes.contour(x1, y1, f_data["E1"], list(f_data["levels"].reshape(-1)))
        self.axes2.plot_surface(x2, y2, f_data["E2"], color="blue")
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
        self.axes2.set_xlabel(self.pair_params[self.pair_of_params - 1][0])
        self.axes2.set_ylabel(self.pair_params[self.pair_of_params - 1][1])
        # self.axes2.view_init(30, 60)
        self.canvas.draw()
        self.canvas2.draw()

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
        return self.path,

    def on_animate(self, idx):

        n1 = np.dot(self.W1, P) + self.b1
        a1 = logsigmoid(n1)
        n2 = np.dot(self.W2, a1) + self.b2
        a2 = logsigmoid(n2)

        e = T - a2

        D2 = a2 * (1 - a2) * e
        D1 = a1 * (1 - a1) * np.dot(self.W2.T, D2)
        dW1 = np.dot(D1, P.T) * self.lr
        db1 = np.dot(D1, np.ones((D1.shape[1], 1))) * self.lr
        dW2 = np.dot(D2, a1.T) * self.lr
        db2 = np.dot(D2, np.ones((D2.shape[1], 1))) * self.lr

        if self.pair_of_params == 1:
            self.W1[0, 0] += dW1[0, 0]
            self.W2[0, 0] += dW2[0, 0]
            self.x, self.y = self.W1[0, 0], self.W2[0, 0]
        elif self.pair_of_params == 2:
            self.W1[0, 0] += dW1[0, 0]
            self.b1[0, 0] += db1[0, 0]
            self.x, self.y = self.W1[0, 0], self.b1[0, 0]
        elif self.pair_of_params == 3:
            self.b1[0, 0] += db1[0, 0]
            self.b1[1, 0] += db1[1, 0]
            self.x, self.y = self.b1[0, 0], self.b1[1, 0]

        self.x_data.append(self.x)
        self.y_data.append(self.y)
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
                self.lr, self.epochs = 3.5, 1000
            elif self.pair_of_params == 2:
                self.W1[0, 0], self.b1[0, 0] = self.x, self.y
                self.lr, self.epochs = 25, 300
            elif self.pair_of_params == 3:
                self.b1[0, 0], self.b1[1, 0] = self.x, self.y
                self.lr, self.epochs = 25, 60
            self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=self.epochs,
                                     interval=self.animation_speed, repeat=False, blit=True)

    def init_params(self):
        self.W1, self.b1 = np.array([[10.], [10.]]), np.array([[-5.], [5.]])
        self.W2, self.b2 = np.array([[1., 1.]]), np.array([[-1.]])
