from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0,
              0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
y = np.copy(x)
X, Y = np.meshgrid(x, y)

max_epoch = 50

F = (Y - X) ** 4 + 8 * X * Y - X + Y + 3
F[F < 0] = 0
F[F > 12] = 12

class NewtonsMethod(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(NewtonsMethod, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Newtons Method", 9, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.x_data, self.y_data = [], []
        self.ani_1, self.ani_2, self.event, self.x, self.y = None, None, None, None, None

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.contour(X, Y, F)
        self.axes_1.set_title("Function F", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.path_1, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation Fa", fontdict={'fontsize': 10})
        self.path_2, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.canvas2.draw()

    def animate_init_1(self):
        self.path_1, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        return self.path_1,

    def animate_init_2(self):
        self.path_2, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        return self.path_2,

    def on_animate_1(self, idx):
        gx = -4 * (self.y - self.x) ** 3 + 8 * self.y - 1
        gy = 4 * (self.y - self.x) ** 3 + 8 * self.x + 1
        gradient = np.array([[gx], [gy]])
        temp = 12 * (self.y - self.x) ** 2
        hess = np.array([[temp, 8 - temp], [8 - temp, temp]])
        dxy = -np.dot(np.linalg.inv(hess), gradient)
        self.x += dxy[0, 0]
        self.y += dxy[1, 0]
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.path_1.set_data(self.x_data, self.y_data)
        return self.path_1,

    def on_animate_2(self, idx):
        self.path_2.set_data(self.x_data, self.y_data)
        return self.path_2,

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:

            # Checks whether the clicked point is in the contour
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
            y_event = round(y_event, 1)
            if F[np.bitwise_and(X == x_event, Y == y_event)].item() == 12:
                return

            self.x, self.y = event.xdata, event.ydata
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
            Fa = (hess[0, 0] * dX ** 2 + (hess[0, 1] + hess[1, 0]) * dX * dY + hess[1, 1] * dY ** 2) / 2
            Fa += gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
            self.axes_2.contour(X, Y, Fa)

            self.event = event
            if self.ani_1:
                self.ani_1.event_source.stop()
            if self.ani_2:
                self.ani_2.event_source.stop()
            self.path_1.set_data([], [])
            self.path_2.set_data([], [])
            self.x_data, self.y_data = [], []
            self.canvas.draw()
            self.canvas2.draw()
            self.run_animation(event)

    def run_animation(self, event):
        if event.xdata != None and event.xdata != None:
            self.x_data, self.y_data = [self.x], [self.y]
            self.ani_1 = FuncAnimation(self.figure, self.on_animate_1, init_func=self.animate_init_1, frames=max_epoch,
                                       interval=200, repeat=False, blit=True)
            self.ani_2 = FuncAnimation(self.figure, self.on_animate_2, init_func=self.animate_init_2, frames=max_epoch,
                                       interval=200, repeat=False, blit=True)
