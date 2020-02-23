from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH

x, y = np.linspace(-2, 0+(4/31*17), 17, endpoint=False), np.linspace(-2, 0+(4/31*17), 17, endpoint=False)
X, Y = np.meshgrid(x, y)

a, b, c = np.array([[2, 1], [1, 2]]), np.array([0, 0]), 0
max_epoch = 50

F = (a[0, 0] * X ** 2 + a[0, 1] + a[1, 0] * X * Y + a[1, 1] * Y ** 2) / 2 + b[0] * X + b[1] * Y + c


class ComparisonOfMethods(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(ComparisonOfMethods, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("Comparison of Methods", 9, " Click anywhere to start an\n initial guess. The gradient\n descent path will be shown"
                                                      "\nfor both Steepest Descent\n and Conjugate Gradient",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")  # TODO: Change icons

        self.event, self.ani_1, self.ani_2 = None, None, None
        self.axes_1 = self.figure.add_subplot(2, 1, 1)
        self.axes_1.set_title("Above: Steepest Descent Path | Below: Conjugate Gradient Path", fontdict={'fontsize': 10})
        self.axes_1.contour(X, Y, F)
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.path_1, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.x_data_1, self.y_data_1 = [], []
        self.axes_2 = self.figure.add_subplot(2, 1, 2)
        # self.axes_2.set_title("Conjugate Gradient Path")
        self.axes_2.contour(X, Y, F)
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.path_2, = self.axes_2.plot([], linestyle='--', marker='o', label="Conjugate Gradient Path")
        self.x_data_2, self.y_data_2 = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.canvas.draw()

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

        self.slider_anim_speed.valueChanged.connect(self.slide)

    def slide(self):
        self.animation_speed = int(self.slider_anim_speed.value()) * 100
        self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
        if self.x_data_1:
            if self.ani_1:
                self.ani_1.event_source.stop()
                self.ani_2.event_source.stop()
            self.path_1.set_data([], [])
            self.path_2.set_data([], [])
            self.x_data_1, self.y_data_1 = [self.x_data_1[0]], [self.y_data_1[0]]
            self.x_data_2, self.y_data_2 = [self.x_data_2[0]], [self.y_data_2[0]]
            self.canvas.draw()
            self.run_animation(self.event)

    def on_mouseclick(self, event):
        self.event = event
        if self.ani_1:
            self.ani_1.event_source.stop()
        if self.ani_2:
            self.ani_2.event_source.stop()
        self.path_1.set_data([], [])
        self.path_2.set_data([], [])
        self.x_data_1, self.y_data_1 = [], []
        self.x_data_2, self.y_data_2 = [], []
        self.canvas.draw()
        self.run_animation(event)

    def animate_init_1(self):
        self.path_1, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        return self.path_1,

    def animate_init_2(self):
        self.path_2, = self.axes_2.plot([], linestyle='--', marker='o', label="Conjugate Gradient Path")
        return self.path_2,

    def on_animate_1(self, idx):
        lr = 0.07
        gradient = np.dot(a, np.array([self.x_1, self.y_1])) + b.T
        self.x_1 -= lr * gradient[0]
        self.y_1 -= lr * gradient[1]
        self.x_data_1.append(self.x_1)
        self.y_data_1.append(self.y_1)
        self.path_1.set_data(self.x_data_1, self.y_data_1)
        return self.path_1,

    def on_animate_2(self, idx):
        if self.i == 0:
            self.gradient = np.dot(a, np.array([self.x_2, self.y_2])) + b.T
            self.p = -self.gradient
            self.i += 1
        elif self.i == 1:
            gradient_old = self.gradient
            self.gradient = np.dot(a, np.array([self.x_2, self.y_2])) + b.T
            beta = np.dot(self.gradient.T, self.gradient) / np.dot(gradient_old.T, gradient_old)
            self.p = -self.gradient + np.dot(beta, self.p)
        hess = a
        lr = -np.dot(self.gradient, self.p.T) / np.dot(self.p.T, np.dot(hess, self.p))
        self.x_2 += lr * self.p[0]
        self.y_2 += lr * self.p[1]
        self.x_data_2.append(self.x_2)
        self.y_data_2.append(self.y_2)
        self.path_2.set_data(self.x_data_2, self.y_data_2)
        return self.path_2,

    def run_animation(self, event):
        if event.xdata != None and event.xdata != None:
            self.x_data_1, self.y_data_1 = [event.xdata], [event.ydata]
            self.x_data_2, self.y_data_2 = [event.xdata], [event.ydata]
            self.x_1, self.y_1 = event.xdata, event.ydata
            self.x_2, self.y_2 = event.xdata, event.ydata
            self.ani_1 = FuncAnimation(self.figure, self.on_animate_1, init_func=self.animate_init_1, frames=max_epoch,
                                       interval=self.animation_speed, repeat=False, blit=True)
            self.i = 0
            self.ani_2 = FuncAnimation(self.figure, self.on_animate_2, init_func=self.animate_init_2, frames=2,
                                       interval=self.animation_speed, repeat=False, blit=True)
