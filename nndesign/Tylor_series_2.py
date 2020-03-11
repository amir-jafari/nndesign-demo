from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


# x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0,
#               0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
x = np.linspace(-2, 2, 1000)
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

class TylorSeries2(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(TylorSeries2, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Taylor Series", 8, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.order, self.x, self.y = 0, None, None
        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.addItems(["Order 0", "Order 1", "Order 2"])
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)
        self.comboBox1.currentIndexChanged.connect(self.change_approx_order)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(20 * self.w_ratio, 120 * self.h_ratio, 230 * self.w_ratio, 230 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(270 * self.w_ratio, 120 * self.h_ratio, 230 * self.w_ratio, 230 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)

        self.figure3 = Figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.axis1 = Axes3D(self.figure3)
        self.toolbar = NavigationToolbar(self.canvas3, self)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(20 * self.w_ratio, 400 * self.h_ratio, 230 * self.w_ratio, 230 * self.h_ratio)
        self.layout3.addWidget(self.canvas3)
        self.wid3.setLayout(self.layout3)

        self.figure4 = Figure()
        self.canvas4 = FigureCanvas(self.figure4)
        self.axis2 = Axes3D(self.figure4)
        self.toolbar = NavigationToolbar(self.canvas4, self)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(270 * self.w_ratio, 400 * self.h_ratio, 230 * self.w_ratio, 230 * self.h_ratio)
        self.layout4.addWidget(self.canvas4)
        self.wid4.setLayout(self.layout4)

        self.x_data, self.y_data = [], []

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.contour(X, Y, F)
        self.axes_1.set_title("Function", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_point, = self.axes_1.plot([], "*")
        self.axes_1.set_xticks([-2, -1, 0, 1])
        self.axes_1.set_yticks([-2, -1, 0, 1])
        self.axes_1.set_xlabel("$x$")
        self.axes_1.xaxis.set_label_coords(1, -0.025)
        self.axes_1.set_ylabel("$y$")
        self.axes_1.yaxis.set_label_coords(-0.025, 1)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation", fontdict={'fontsize': 10})
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_xticks([-2, -1, 0, 1])
        self.axes_2.set_yticks([-2, -1, 0, 1])
        self.axes_2.set_xlabel("$x$")
        self.axes_2.xaxis.set_label_coords(1, -0.025)
        self.axes_2.set_ylabel("$y$")
        self.axes_2.yaxis.set_label_coords(-0.025, 1)
        self.axes2_point, = self.axes_2.plot([], "*")
        self.canvas2.draw()

        self.axis1.set_title("Function", fontdict={'fontsize': 10})
        self.axis1.plot_surface(XX, YY, FF)
        self.axis1.set_xticks([-2, -1, 0, 1])
        self.axis1.set_yticks([-2, -1, 0, 1])
        self.axis1.set_xlabel("$x$")
        self.axis1.xaxis.set_label_coords(1, -0.025)
        self.axis1.set_ylabel("$y$")
        self.axis1.yaxis.set_label_coords(-0.025, 1)
        self.axis1.view_init(30, 60)
        self.canvas3.draw()

        self.axis2.set_title("Approximation", fontdict={'fontsize': 10})
        self.axis2.set_xticks([-2, -1, 0, 1])
        self.axis2.set_yticks([-2, -1, 0, 1])
        self.axis2.set_xlabel("$x$")
        self.axis2.xaxis.set_label_coords(1, -0.025)
        self.axis2.set_ylabel("$y$")
        self.axis2.yaxis.set_label_coords(-0.025, 1)
        self.axis2.view_init(30, 60)
        self.canvas4.draw()

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:

            """# Checks whether the clicked point is in the contour
            x_event, y_event = event.xdata, event.ydata
            if round(x_event, 1) * 10 % 2 == 1:
                x_event += 0.06
                if round(x_event, 1) * 10 % 2 == 1:
                    x_event = event.xdata - 0.06
            if round(y_event, 1) * 10 % 2 == 1:
                y_event += 0.06
                if round(y_event, 1) * 10 % 2 == 1:
                    y_event = event.ydata - 0.06
            x_event = round(x_event, 1)
            y_event = round(y_event, 1)"""
            d_x, d_y = event.xdata - x, event.ydata - y
            x_event = x[np.argmin(np.abs(d_x))]
            y_event = y[np.argmin(np.abs(d_y))]
            if F[np.bitwise_and(X == x_event, Y == y_event)].item() == 12:
                return

            self.axes1_point.set_data([event.xdata], [event.ydata])
            self.axes2_point.set_data([event.xdata], [event.ydata])
            self.x, self.y = event.xdata, event.ydata
            self.draw_approx()
            self.canvas.draw()

    def change_approx_order(self, idx):
        self.order = idx
        if self.x and self.y:
            self.draw_approx()

    def draw_approx(self):
        # Removes contours from second plot
        while self.axes_2.collections:
            for collection in self.axes_2.collections:
                collection.remove()
        # Draws new contour
        Fo = (self.y - self.x) ** 4 + 8 * self.x * self.y - self.x * self.y + 3
        gx = -4 * (self.y - self.x) ** 3 + 8 * self.y - 1
        gy = 4 * (self.y - self.x) ** 3 + 8 * self.x + 1
        gradient = np.array([[gx], [gy]])
        temp = 12 * (self.y - self.x) ** 2
        hess = np.array([[temp, 8 - temp], [8 - temp, temp]])
        dX, dY = X - self.x, Y - self.y
        if self.order == 0:
            Fa = np.zeros(X.shape) + Fo
        elif self.order == 1:
            Fa = gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
        elif self.order == 2:
            Fa = (hess[0, 0] * dX ** 2 + (hess[0, 1] + hess[1, 0]) * dX * dY + hess[1, 1] * dY ** 2) / 2
            Fa += gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
        Fa[Fa < 0] = 0
        Fa[Fa > 12] = 12
        self.axes_2.contour(X, Y, Fa)
        self.canvas2.draw()

        # Removes surface from fourth plot
        while self.axis2.collections:
            for collection in self.axis2.collections:
                collection.remove()
        # Draws new surface
        dXX, dYY = XX - self.x, YY - self.y
        if self.order == 0:
            Fa = np.zeros(XX.shape) + Fo
        elif self.order == 1:
            Fa = gradient[0, 0] * dXX + gradient[1, 0] * dYY + Fo
        elif self.order == 2:
            Fa = (hess[0, 0] * dXX ** 2 + (hess[0, 1] + hess[1, 0]) * dXX * dYY + hess[1, 1] * dYY ** 2) / 2
            Fa += gradient[0, 0] * dXX + gradient[1, 0] * dYY + Fo
        Fa[Fa < 0] = 0
        Fa[Fa > 12] = 12
        self.axis2.plot_surface(XX, YY, Fa, color="blue")
        self.canvas4.draw()
