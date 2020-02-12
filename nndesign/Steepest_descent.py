from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.4, -0.2, 0,
              0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
y = np.copy(x)
X, Y = np.meshgrid(x, y)

max_epoch = 50

F = (Y - X) ** 4 + 8 * X * Y - X + Y + 3
F[F < 0] = 0
F[F > 12] = 12

class SteepestDescent(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(SteepestDescent, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Steepest Descent", 9, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.x_data, self.y_data = [], []

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.contour(X, Y, F)
        self.axes_1.set_title("-", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_path, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("-", fontdict={'fontsize': 10})
        self.axes2_path, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.canvas2.draw()

        self.lr = 0.03
        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("lr: 0.001")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(self.x_chapter_slider_label * self.w_ratio, 250 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 6)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(3)
        self.wid_lr = QtWidgets.QWidget(self)
        self.layout_lr = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_lr.setGeometry(self.x_chapter_usual * self.w_ratio, 280 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout_lr.addWidget(self.slider_lr)
        self.wid_lr.setLayout(self.layout_lr)
        self.slider_lr.valueChanged.connect(self.slide)

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            # TODO: Figure out when point is outside the contour
            self.x_data, self.y_data = [], []
            self.graph()  # TODO: Figure out how to clear contour when internet
            self.x_data, self.y_data = [event.xdata], [event.ydata]
            self.axes1_path, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
            self.axes2_path, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
            self.steepest_descent(event.xdata, event.ydata)

    def graph(self):
        self.axes1_path.set_data(self.x_data, self.y_data)
        self.axes2_path.set_data(self.x_data, self.y_data)
        self.canvas.draw()
        self.canvas2.draw()

    def slide(self):
        self.lr = float(self.slider_lr.value()/100)
        self.label_lr.setText("lr: " + str(self.lr))
        if self.x_data:
            x_start, y_start = self.x_data[0], self.y_data[0]
            self.x_data, self.y_data = [], []
            self.graph()
            self.x_data, self.y_data = [x_start], [y_start]
            self.axes1_path, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
            self.steepest_descent(x_start, x_start)

    def steepest_descent(self, x_start, y_start):
        x, y = x_start, y_start

        Fo = (y - x) ** 4 + 8 * x * y - x * y + 3
        gx = -4 * (y - x) ** 3 + 8 * y - 1
        gy = 4 * (y - x) ** 3 + 8 * x + 1
        gradient = np.array([[gx], [gy]])
        dX, dY = X - x, Y - y
        Fa = gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
        self.axes_2.contour(X, Y, Fa)

        for i in range(max_epoch):
            gx = -4 * (y - x) ** 3 + 8 * y - 1
            gy = 4 * (y - x) ** 3 + 8 * x + 1
            x -= self.lr * gx
            y -= self.lr * gy
            self.x_data.append(x)
            self.y_data.append(y)

        self.graph()
