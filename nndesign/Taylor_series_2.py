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

class TaylorSeries2(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(TaylorSeries2, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Taylor Series #2", 8, "Click in the top-left graph to\ncreate a Taylor series\napproximation."
                                                 "\n\nSelect the approximation\norder below.",
                          PACKAGE_PATH + "Logo/Logo_Ch_8.svg", None)

        self.order, self.x, self.y = 2, None, None
        self.make_combobox(1, ["Order 0", "Order 1", "Order 2"], (self.x_chapter_usual, 250, self.w_chapter_slider, 100),
                           self.change_approx_order)
        self.comboBox1.setCurrentIndex(self.order)

        self.make_plot(1, (20, 120, 230, 230))
        self.make_plot(2, (270, 120, 230, 230))
        self.make_plot(3, (20, 400, 230, 230))
        self.make_plot(4, (270, 400, 230, 230))

        self.x_data, self.y_data = [], []

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.contour(X, Y, F, colors="blue")
        self.axes_1.set_title("Function", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_point, = self.axes_1.plot([], "o", fillstyle="none", color="k")
        self.axes1_point1, = self.axes_1.plot([], "o", fillstyle="none", markersize=11, color="k")
        # self.axes_1.set_xticks([-2, -1, 0, 1])
        # self.axes_1.set_yticks([-2, -1, 0, 1])
        # self.axes_1.set_xlabel("$x$")
        # self.axes_1.xaxis.set_label_coords(1, -0.025)
        # self.axes_1.set_ylabel("$y$")
        # self.axes_1.yaxis.set_label_coords(-0.025, 1)
        self.axes_1.text(-1.5, 1.65, "<CLICK ON ME>")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation", fontdict={'fontsize': 10})
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        # self.axes_2.set_xticks([-2, -1, 0, 1])
        # self.axes_2.set_yticks([-2, -1, 0, 1])
        # self.axes_2.set_xlabel("$x$")
        # self.axes_2.xaxis.set_label_coords(1, -0.025)
        # self.axes_2.set_ylabel("$y$")
        # self.axes_2.yaxis.set_label_coords(-0.025, 1)
        self.axes2_point, = self.axes_2.plot([], "o", fillstyle="none", color="k")
        self.axes2_point1, = self.axes_2.plot([], "o", fillstyle="none", markersize=11, color="k")
        self.canvas2.draw()

        self.axis1 = Axes3D(self.figure3)
        # self.axis1.dist = 20
        self.axis1.set_title("Function", fontdict={'fontsize': 10}, pad=3)
        self.axis1.plot_surface(XX, YY, FF)
        # self.axis1.set_xticks([-2, -1, 0, 1])
        # self.axis1.set_yticks([-2, -1, 0, 1])
        # self.axis1.set_xlabel("$x$")
        # self.axis1.xaxis.set_label_coords(1, -0.025)
        # self.axis1.set_ylabel("$y$")
        # self.axis1.yaxis.set_label_coords(-0.025, 1)
        self.axis1.view_init(5, -90)
        self.axis1.autoscale()
        self.canvas3.draw()

        self.axis2 = Axes3D(self.figure4)
        self.axis2.set_title("Approximation", fontdict={'fontsize': 10}, pad=3)
        # self.axis2.set_xticks([-2, -1, 0, 1])
        # self.axis2.set_yticks([-2, -1, 0, 1])
        # self.axis2.set_xlabel("$x$")
        # self.axis2.xaxis.set_label_coords(1, -0.025)
        # self.axis2.set_ylabel("$y$")
        # self.axis2.yaxis.set_label_coords(-0.025, 1)
        self.axis2.view_init(5, -90)
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
            self.axes1_point1.set_data([event.xdata], [event.ydata])
            self.axes2_point.set_data([event.xdata], [event.ydata])
            self.axes2_point1.set_data([event.xdata], [event.ydata])
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