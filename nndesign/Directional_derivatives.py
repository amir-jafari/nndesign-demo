from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH

x1_lim = [-4, 4]
x2_lim = [-2, 2]
a = np.array([[1, 0], [0, 2]])
b = np.zeros((2, 1))
c = 0

x1 = np.arange(x1_lim[0], x1_lim[1] + 0.2, (x1_lim[1] - x1_lim[0]) / 30)
x2 = np.arange(x2_lim[0], x2_lim[1] + 0.2, (x2_lim[1] - x2_lim[0]) / 30)
X1, X2 = np.meshgrid(x1, x2)
F = (a[0, 0] * X1 ** 2 + (a[0, 1] + a[1, 0]) * X1 * X2 + a[1, 1] * X2 ** 2) / 2 + b[0, 0] * X1 + b[1, 0] * X2 + c

class DirectionalDerivatives(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(DirectionalDerivatives, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Directional Derivatives", 8, "To measure a directional derivative click on the graph and move the cursor.\n\n"
                                                        "The directional derivative is taken at the point you clicked\nin the direction of the current cursor position.\n\n"
                                                        "Click again to choose a new point where you want\nto measure the directional derivative.",
                          PACKAGE_PATH + "Logo/Logo_Ch_8.svg", None, description_coords=(30, 400, 500, 200))

        self.cid = None

        self.make_plot(1, (15, 120, 490, 310))
        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Function F")
        self.point, = self.axes_1.plot([], marker='*')
        self.line, = self.axes_1.plot([], linestyle="-", color="blue")
        self.line_data_x, self.line_data_y = [], []
        self.axes_1.contour(X1, X2, F)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.make_slider("slider_dirder", QtCore.Qt.Vertical, (-60, 60), QtWidgets.QSlider.NoTicks, 1, 0,
                         (self.x_chapter_usual + 70, 120, self.w_chapter_slider, 310), self.freeze,
                         "label_dirder", "Directional Derivative", (self.x_chapter_slider_label - 40, 400, 200, 100))
        self.dir_der = 0
        for i in range(5):
            self.make_label("scale_{}".format(i), str(6 - i * 3) + "  -", (self.x_chapter_slider_label - 15, 125 + 62 * i, 50, 50))

    def freeze(self):
        self.slider_dirder.setValue(self.dir_der * 10)

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.label_dirder.setGeometry((self.x_chapter_slider_label - 60) * self.w_ratio, 400 * self.h_ratio,
                                          200 * self.w_ratio, 100 * self.h_ratio)
            self.point.set_data([event.xdata], [event.ydata])
            self.line_data_x, self.line_data_y = [event.xdata], [event.ydata]
            self.line.set_data(self.line_data_x, self.line_data_y)
            self.canvas.draw()
            if self.cid:
                self.canvas.mpl_disconnect(self.cid)
            self.cid = self.canvas.mpl_connect("motion_notify_event", self.on_mousepressed)

    def on_mousepressed(self, event):
        if event.xdata != None and event.ydata != None:
            x1, x2 = self.line_data_x[0], self.line_data_y[0]
            y1, y2 = event.xdata, event.ydata
            angle = np.arctan2(np.array([y2 - x2]), np.array([y1 - x1])).item()
            y1 = x1 + np.cos(angle)
            y2 = x2 + np.sin(angle)
            self.line_data_x.append(y1)
            self.line_data_y.append(y2)
            self.line.set_data(self.line_data_x, self.line_data_y)
            self.canvas.draw()
            self.line_data_x.pop()
            self.line_data_y.pop()
            # xnom = [x1; x2];
            #   p = [y1-x1;y2-x2];
            #   grad = b+a*xnom;
            #   dir_der = p'*grad/norm(p);
            xnorm = np.array([[x1], [x2]])
            p = np.array([[y1 - x1], [y2 - x2]])
            grad = b + np.dot(a, xnorm)
            dir_der = np.dot(p.T, grad) / np.sqrt(np.dot(p.T, p))
            self.dir_der = dir_der
            self.slider_dirder.setValue(self.dir_der * 10)
            self.label_dirder.setText("Directional Derivative: {}".format(round(self.dir_der.item(), 2)))
