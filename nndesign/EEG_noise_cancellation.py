from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


N, f, max_t = 3, 60, 0.5
s = N * f
ts = s * max_t + 1
A, theta, k = 0.1, np.pi/2, 0.2
signal = k * (2 * np.random.uniform(0, 1, (1, int(ts))) - 1)
i = np.arange(ts).reshape(1, -1)
noise = 1.2 * np.sin(2 * np.pi * (i - 1) / N + theta)
filtered_noise = A * 1.20 * np.sin(2 * np.pi * (i-1) / N + theta)
delayed_noise = np.array([list(noise.reshape(-1)), [0] + list(noise.reshape(-1))[:-1]])
noisy_signal = signal + filtered_noise

w = np.array([0, -2])
time = np.arange(1, ts + 1) / ts * max_t

P = delayed_noise
T = noisy_signal[:]
A = 2 * np.dot(P, P.T)
d = -2 * np.dot(P, T.T)
c = np.dot(T, T.T)

x = np.linspace(-2.1, 2.1, 30)
y = np.copy(x)
X, Y = np.meshgrid(x, y)
F = (A[0, 0] * X ** 2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y ** 2) / 2 + d[0, 0] * X + d[1, 0] * Y + c


class EEGNoiseCancellation(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(EEGNoiseCancellation, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("EEG Noise Cancellation", 10, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.x_data, self.y_data = [], []
        self.ani_1, self.ani_2, self.event, self.x, self.y = None, None, None, None, None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(20 * self.w_ratio, 120 * self.h_ratio, 480 * self.w_ratio, 270 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(20 * self.w_ratio, 390 * self.h_ratio, 270 * self.w_ratio, 270 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Original (blue) and Estimated (red) Signals", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(0, 0.5)
        self.axes_1.set_ylim(-2, 2)
        self.signal, = self.axes_1.plot([], linestyle='--', label="Original Signal", color="blue")
        self.signal.set_data(time, signal)
        self.signal_approx, = self.axes_1.plot([], linestyle='-', label="Approx Signal", color="red")
        self.canvas.draw()

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.contour(X, Y, F)
        self.axes_2.set_title("Adaptive Weights", fontdict={'fontsize': 10})
        self.path_2, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.w1_data, self.w2_data = [], []
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.canvas2.draw()

        self.lr = 0.2
        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("lr: 0.2")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(self.x_chapter_slider_label * self.w_ratio, 250 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 15)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(2)
        self.wid_lr = QtWidgets.QWidget(self)
        self.layout_lr = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_lr.setGeometry(self.x_chapter_usual * self.w_ratio, 280 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_lr.addWidget(self.slider_lr)
        self.wid_lr.setLayout(self.layout_lr)

        self.mc = 0
        self.label_mc = QtWidgets.QLabel(self)
        self.label_mc.setText("mc: 0")
        self.label_mc.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_mc.setGeometry(self.x_chapter_slider_label * self.w_ratio, 300 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_mc = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_mc.setRange(0, 10)
        self.slider_mc.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_mc.setTickInterval(1)
        self.slider_mc.setValue(0)
        self.wid_mc = QtWidgets.QWidget(self)
        self.layout_mc = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_mc.setGeometry(self.x_chapter_usual * self.w_ratio, 330 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_mc.addWidget(self.slider_mc)
        self.wid_mc.setLayout(self.layout_mc)

        self.w, self.e = None, None
        self.canvas2.mpl_connect('button_press_event', self.on_mouseclick)
        self.slider_lr.valueChanged.connect(self.slide)
        self.slider_mc.valueChanged.connect(self.slide)

    def slide(self):
        self.lr = float(self.slider_lr.value() / 10)
        self.label_lr.setText("lr: " + str(self.lr))
        self.mc = float(self.slider_mc.value() / 10)
        self.label_mc.setText("mc: " + str(self.mc))
        if self.w1_data:
            if self.ani_2:
                self.ani_2.event_source.stop()
            if self.ani_1:
                self.ani_1.event_source.stop()
            self.path_2.set_data([], [])
            self.signal_approx.set_data([], [])
            self.w1_data, self.w2_data = [self.w1_data[0]], [self.w2_data[0]]
            # e_temp = self.e[0]
            self.e = np.zeros((int(ts),))
            # self.e[0] = e_temp
            self.canvas.draw()
            self.canvas2.draw()
            self.run_animation()

    def animate_init_1(self):
        self.signal_approx, = self.axes_1.plot([], linestyle='-', label="Approx Signal", color="red")
        return self.signal_approx,

    def animate_init_2(self):
        self.path_2, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        return self.path_2,

    def on_animate_1(self, idx):
        self.signal_approx.set_data(time[:idx + 1], self.e[:idx + 1])
        return self.signal_approx,

    def on_animate_2(self, idx):

        a = np.dot(self.w, P[:, idx])
        self.e[idx] = T[0, idx] - a
        self.w = self.mc * self.w + (1 - self.mc) * self.lr * self.e[idx] * P[:, idx].T
        self.w1_data.append(self.w[0])
        self.w2_data.append(self.w[1])
        self.path_2.set_data(self.w1_data, self.w2_data)
        return self.path_2,

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.w = np.array([event.xdata, event.ydata])
            self.e = np.zeros((int(ts),))
            self.event = event
            if self.ani_1:
                self.ani_1.event_source.stop()
            if self.ani_2:
                self.ani_2.event_source.stop()
            self.signal_approx.set_data([], [])
            self.path_2.set_data([], [])
            self.w1_data, self.w2_data = [self.w[0]], [self.w[1]]
            self.canvas.draw()
            self.canvas2.draw()
            self.run_animation()
            self.canvas.draw()
            self.canvas2.draw()

    def run_animation(self):
        self.ani_2 = FuncAnimation(self.figure2, self.on_animate_2, init_func=self.animate_init_2, frames=int(ts),
                                   interval=100, repeat=False, blit=True)
        self.ani_1 = FuncAnimation(self.figure, self.on_animate_1, init_func=self.animate_init_1, frames=int(ts),
                                   interval=100, repeat=False, blit=True)
