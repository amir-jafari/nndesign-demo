from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH

x, y = np.linspace(-4, 0+(0.2*22), 22, endpoint=False), np.linspace(-2, 0+(4/31*17), 17, endpoint=False)
X, Y = np.meshgrid(x, y)

a, b, c = np.array([[2, 0], [0, 50]]), np.array([0, 0]), 0
max_epoch = 50

F = (a[0, 0] * X ** 2 + a[0, 1] + a[1, 0] * X * Y + a[1, 1] * Y ** 2) / 2 + b[0] * X + b[1] * Y + c


class SteepestDescentQuadratic(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(SteepestDescentQuadratic, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("Steepest Descent for Quadratic", 9, " Click anywhere to start an\n initial guess. The gradient\n descent path will be shown\n"
                                                               " Modify the learning rate\n by moving the slide bar",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")  # TODO: Change icons

        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.contour(X, Y, F)
        self.axes.set_xlim(-4, 4)
        self.axes.set_ylim(-2, 2)
        self.path, = self.axes.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.ani, self.event = None, None

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
