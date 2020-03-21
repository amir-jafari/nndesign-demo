from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.signal import lfilter

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class RecurrentNetworkTraining(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(RecurrentNetworkTraining, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Recurrent Network Training", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 300 * self.h_ratio, 250 * self.w_ratio, 200 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(250 * self.w_ratio, 300 * self.h_ratio, 250 * self.w_ratio, 350 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)

        self.figure3 = Figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(15 * self.w_ratio, 480 * self.h_ratio, 250 * self.w_ratio, 200 * self.h_ratio)
        self.layout3.addWidget(self.canvas3)
        self.wid3.setLayout(self.layout3)

        self.label_w0 = QtWidgets.QLabel(self)
        self.label_w0.setText("iW(0): 0.5")
        self.label_w0.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w0.setGeometry(self.x_chapter_slider_label * self.w_ratio, 200 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_w0 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w0.setRange(-20, 20)
        self.slider_w0.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w0.setTickInterval(1)
        self.slider_w0.setValue(5)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 230 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_w0)
        self.wid3.setLayout(self.layout3)

        self.label_w1 = QtWidgets.QLabel(self)
        self.label_w1.setText("lW(1): 0.5")
        self.label_w1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 270 * self.h_ratio, 150 * self.w_ratio,
                                  100 * self.h_ratio)
        self.slider_w1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1.setRange(-20, 20)
        self.slider_w1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1.setTickInterval(1)
        self.slider_w1.setValue(5)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 300 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_w1)
        self.wid3.setLayout(self.layout3)

        self.animation_speed = 200
        self.label_anim_speed = QtWidgets.QLabel(self)
        self.label_anim_speed.setText("Animation Delay: 0 ms")
        self.label_anim_speed.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_anim_speed.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 340 * self.h_ratio,
                                          self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_anim_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_anim_speed.setRange(0, 6)
        self.slider_anim_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_anim_speed.setTickInterval(1)
        self.slider_anim_speed.setValue(2)
        self.wid_anim_speed = QtWidgets.QWidget(self)
        self.layout_anim_speed = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_anim_speed.setGeometry(self.x_chapter_usual * self.w_ratio, 370 * self.h_ratio,
                                        self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_anim_speed.addWidget(self.slider_anim_speed)
        self.wid_anim_speed.setLayout(self.layout_anim_speed)

        self.slider_anim_speed.valueChanged.connect(self.slide)
        # self.slider_w0.valueChanged.connect(self.graph)
        # self.slider_w1.valueChanged.connect(self.graph)

        self.graph()

    def slide(self):
        if self.ani2:
            self.ani2.event_source.stop()
        if self.ani1:
            self.ani1.event_source.stop()
        self.steepest_descent_data.set_data([], [])
        self.net_approx.set_data([], [])
        self.canvas.draw()
        self.canvas2.draw()
        self.animation_speed = int(self.slider_anim_speed.value()) * 100
        self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
        self.x_data, self.y_data = [self.x_data[0]], [self.y_data[0]]
        self.run_animation()

    def graph(self):

        self.a = self.figure.add_subplot(1, 1, 1)
        self.a2 = self.figure2.add_subplot(1, 1, 1)
        a3 = Axes3D(self.figure3)
        self.a.clear()  # Clear the plot
        self.a2.clear()
        a3.clear()
        self.a.set_xlim(0, 20)
        # self.a2.set_xlim(-2, 2)
        a3.set_xlim(-2, 2)
        a3.set_ylim(-2, 2)
        # a3.set_zlim(-2, 2)
        self.a.set_title("Input and Target Sequences", fontsize=10)
        self.a2.set_title("Steepest Descent Trajectory", fontsize=10)
        a3.set_title("Performance Surface", fontsize=10)
        self.a.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)

        self.iw = self.slider_w0.value() / 10
        self.lw = self.slider_w1.value() / 10
        self.animation_speed = int(self.slider_anim_speed.value()) * 100
        self.label_w0.setText("iW(0): " + str(self.iw))
        self.label_w1.setText("lW(1): " + str(self.lw))
        self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")

        self.P = np.array([0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, -0.75, -0.75, -0.75, -0.75, 0.55, 0.55, 0.55, -0.25, -0.25, -0.25, -0.25, -0.25])
        self.a0, t = 0, list(range(1, len(self.P) + 1))
        self.T = self.forward(self.iw, self.lw, self.P, self.a0)

        xx, yy = np.arange(-2, 2.1, 0.1), np.arange(-2, 2.1, 0.1)
        num = len(self.P)
        self.error = np.zeros((len(xx), len(yy)))
        j = 0
        for y1 in yy:
            y2 = self.forward(xx, y1, self.P, self.a0).T
            i = 0
            for x1 in xx:
                e = self.T - y2[i, :]
                self.error[i, j] = np.sum(e * e) / num
                i += 1
            j += 1
        self.X, self.Y = np.meshgrid(xx, yy)

        self.a.scatter([0] + t, [self.a0] + list(self.P), color="white", marker=".", edgecolor="blue")
        self.a.scatter([0] + t, [self.a0] + list(self.T), color="blue", marker=".", s=[1] * (len(t) + 1))
        self.a2.contour(self.X, self.Y, self.error.T, levels=[0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        a3.plot_surface(self.X, self.Y, self.error.T)

        self.ani1, self.ani2 = None, None
        self.x_data, self.y_data = [], []
        self.steepest_descent_data, = self.a2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.net_approx, = self.a.plot([], linestyle='none', marker="+", color="blue", label="Network approximation")
        self.canvas2.mpl_connect('button_press_event', self.on_mouseclick)

        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    def forward(self, iw, lw, p, a0):
        if type(iw) != np.ndarray:
            y = [self.tansig(iw * p[0] + lw * a0)]
            for i in range(len(p) - 1):
                y.append(self.tansig(iw * p[i + 1] + lw * y[i]))
            return np.array(y)
        else:
            y = np.zeros((len(p), len(iw)))
            y[0, :] = self.tansig(iw * p[0] + lw * a0)
            for i in range(len(p) - 1):
                y[i + 1, :] = self.tansig(iw * p[i + 1] + lw * y[i, :])
            return y

    def gradient(self, iw, lw, p, yt, a0):
        y = [self.tansig(iw * p[0] + lw * a0)]
        e = [yt[0] - y[0]]
        df = 1 - y[0] ** 2
        dy_diw = df * p[0]
        dy_dlw = df * a0
        gw = -2 * e[0] * np.array([[dy_diw], [dy_dlw]])
        for i in range(len(p) - 1):
            y.append(self.tansig(iw * p[i + 1] + lw * y[i]))
            df = 1 - y[i + 1] ** 2
            dy_diw = df * (p[i + 1] + lw * dy_diw)
            dy_dlw = df * (y[i] + lw * dy_dlw)
            e.append(yt[i + 1] - y[i + 1])
            gw = gw - 2 * e[i + 1] * np.array([[dy_diw], [dy_dlw]])
        return gw

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.x_data = [event.xdata]
            self.y_data = [event.ydata]
            self.net_approx.set_data([], [])
            self.canvas.draw()
            self.canvas2.draw()
            self.run_animation()

    def run_animation(self):
        if self.ani1:
            self.ani1.event_source.stop()
        if self.ani2:
            self.ani2.event_source.stop()
        self.ani2 = FuncAnimation(self.figure2, self.on_animate, init_func=self.animate_init, frames=100,
                                  interval=self.animation_speed, repeat=False, blit=False)
        self.ani1 = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=100,
                                  interval=self.animation_speed, repeat=False, blit=False)
        self.canvas.draw()
        self.canvas2.draw()

    def animate_init(self):
        self.steepest_descent_data.set_data(self.x_data, self.y_data)
        A2 = self.forward(self.x_data[-1], self.y_data[-1], self.P, self.a0)
        self.net_approx.set_data([0] + list(range(1, len(self.P) + 1)), [self.a0] + list(A2))
        E = self.T - A2
        gw = self.gradient(self.x_data[-1], self.y_data[-1], self.P, self.T, self.a0)
        self.dW = -gw
        self.nrmrt = np.sqrt(np.dot(gw.T, gw))
        return self.steepest_descent_data, self.net_approx

    def on_animate(self, idx):

        if self.nrmrt < 0.2:
            return self.steepest_descent_data, self.net_approx

        self.x_data.append(self.x_data[-1] + 0.1 * self.dW[0, 0])
        self.y_data.append(self.y_data[-1] + 0.1 * self.dW[1, 0])
        A2 = self.forward(self.x_data[-1], self.y_data[-1], self.P, self.a0)
        E = self.T - A2
        gw = self.gradient(self.x_data[-1], self.y_data[-1], self.P, self.T, self.a0)
        self.nrmrt = np.sqrt(np.dot(gw.T, gw))
        self.dW = -gw
        self.a2.set_xlim(min([-2, min(self.x_data)]), max([2, max(self.x_data)]))
        self.a2.set_ylim(min([-2, min(self.y_data)]), max([2, max(self.y_data)]))
        self.steepest_descent_data.set_data(self.x_data, self.y_data)
        self.net_approx.set_data([0] + list(range(1, len(self.P) + 1)), [self.a0] + list(A2))
        return self.steepest_descent_data, self.net_approx
