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
A1, A2, theta1, theta2, k = 1, 0.75, np.pi/2, np.pi/2.5, 0.2
# signal = k * (2 * np.random.uniform(0, 1, (1, int(ts))) - 1)
# TODO: signal = k*eegdata(1:ts)
i = np.arange(ts).reshape(1, -1)
noise1, noise2 = 1.2 * np.sin(2 * np.pi * (i - 1) / N), 0.6 * np.sin(4 * np.pi * (i - 1) / N)
noise = noise1 + noise2
filtered_noise1 = A1 * 1.20 * np.sin(2 * np.pi * (i-1) / N + theta1)
filtered_noise2 = A2 * 0.6 * np.sin(4 * np.pi * (i-1) / N + theta1)
filtered_noise = filtered_noise1 + filtered_noise2
# noisy_signal = signal + filtered_noise

w = np.array([0, -2])
time = np.arange(1, ts + 1) / ts * max_t

# P = []
# for i in range(21):
#     P.append()
# T = noisy_signal[:]


class EEGNoiseCancellation(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(EEGNoiseCancellation, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("EEG Noise Cancellation", 10, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.x_data, self.y_data = [], []
        self.ani, self.x, self.y = None, None, None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(20 * self.w_ratio, 120 * self.h_ratio, 480 * self.w_ratio, 270 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Original (blue) and Estimated (red) Signals", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(0, 0.5)
        self.axes_1.set_ylim(-2, 2)
        self.signal, = self.axes_1.plot([], linestyle='--', label="Original Signal", color="blue")
        self.signal.set_data(time, signal)
        self.signal_approx, = self.axes_1.plot([], linestyle='-', label="Approx Signal", color="red")
        self.canvas.draw()

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

        self.delays = 0
        self.label_delays = QtWidgets.QLabel(self)
        self.label_delays.setText("Delays: 10")
        self.label_delays.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_delays.setGeometry(self.x_chapter_slider_label * self.w_ratio, 350 * self.h_ratio,
                                      self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_delays = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_delays.setRange(0, 20)
        self.slider_delays.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_delays.setTickInterval(1)
        self.slider_delays.setValue(10)
        self.wid_delays = QtWidgets.QWidget(self)
        self.layout_delays = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_delays.setGeometry(self.x_chapter_usual * self.w_ratio, 380 * self.h_ratio,
                                    self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_delays.addWidget(self.slider_delays)
        self.wid_delays.setLayout(self.layout_delays)

        self.w, self.e = None, None
        self.slider_lr.valueChanged.connect(self.slide)
        self.slider_delays.valueChanged.connect(self.slide)

    def slide(self):
        self.lr = float(self.slider_lr.value() / 10)
        self.label_lr.setText("lr: " + str(self.lr))
        self.delays = int(self.slider_delays.value())
        self.label_delays.setText("Delays: " + str(self.delays))
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
            self.run_animation()

    def animate_init(self):
        self.signal_approx, = self.axes_1.plot([], linestyle='-', label="Approx Signal", color="red")
        return self.signal_approx,

    def on_animate(self, idx):
        # % NUMBER OF DELAYS
        #   R = delays + 1;
        #   P = P(1:R,:);
        # TODO
        a = np.dot(self.w, P[:, idx])
        self.e[idx] = T[0, idx] - a
        self.w = self.lr * self.e[idx] * P[:, idx].T
        self.w1_data.append(self.w[0])
        self.w2_data.append(self.w[1])
        self.signal_approx.set_data(time[:idx + 1], self.e[:idx + 1])
        return self.signal_approx,

    def run_animation(self):
        self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=int(ts),
                                 interval=100, repeat=False, blit=True)
