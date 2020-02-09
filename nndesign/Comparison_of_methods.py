from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

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

        self.axes_1 = self.figure.add_subplot(2, 1, 1)
        self.axes_1.set_title("Above: Steepest Descent Path | Below: Conjugate Gradient Path", fontdict={'fontsize': 10})
        self.axes_1.contour(X, Y, F)
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.path_1, = self.axes_1.plot([], 'r+', label="Gradient Descent Path")
        self.x_data_1, self.y_data_1 = [], []
        self.axes_2 = self.figure.add_subplot(2, 1, 2)
        # self.axes_2.set_title("Conjugate Gradient Path")
        self.axes_2.contour(X, Y, F)
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.path_2, = self.axes_2.plot([], 'r+', label="Conjugate Gradient Path")
        self.x_data_2, self.y_data_2 = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.graph()

        """self.lr = 0.03
        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("lr: 0.001")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(775, 250, 150, 100)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 6)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(3)

        self.wid_lr = QtWidgets.QWidget(self)
        self.layout_lr = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_lr.setGeometry(710, 280, 150, 100)
        self.layout_lr.addWidget(self.slider_lr)
        self.wid_lr.setLayout(self.layout_lr)

        self.slider_lr.valueChanged.connect(self.slide)

    def slide(self):
        self.lr = float(self.slider_lr.value()/100)
        self.label_lr.setText("lr: " + str(self.lr))

        x_start, y_start = self.x_data_1[0], self.y_data_1[0]
        self.x_data_1, self.y_data_1 = [], []
        self.x_data_2, self.y_data_2 = [], []
        self.graph()
        self.x_data_1, self.y_data_1 = [x_start], [y_start]
        self.x_data_2, self.y_data_2 = [x_start], [y_start]

        self.path_1, = self.axes_1.plot([], label="Gradient Descent Path")
        self.path_2, = self.axes_2.plot([], label="Conjugate Gradient Path")
        self.steepest_descent(x_start, x_start)
        self.conjugate_gradient(x_start, y_start)
        self.graph()"""

    def graph(self):
        self.path_1.set_data(self.x_data_1, self.y_data_1)
        self.path_2.set_data(self.x_data_2, self.y_data_2)
        self.canvas.draw()

    def steepest_descent(self, x_start, y_start):
        x, y = x_start, y_start
        for i in range(5):
            gradient = np.dot(a, np.array([x, y])) + b.T
            p = -gradient
            hess = a
            lr = -np.dot(gradient.T, p) / np.dot(p.T, np.dot(hess, p))
            x += lr * p[0]
            y += lr * p[1]
            self.x_data_1.append(x)
            self.y_data_1.append(y)

    def conjugate_gradient(self, x_start, y_start):  # TODO: Change this
        x, y = x_start, y_start
        for i in range(2):
            if i == 0:
                gradient = np.dot(a, np.array([x, y])) + b.T
                p = -gradient
            elif i == 1:
                gradient_old = gradient
                gradient = np.dot(a, np.array([x, y])) + b.T
                beta = np.dot(gradient.T, gradient) / np.dot(gradient_old.T, gradient_old)
                p = -gradient + np.dot(beta, p)
            hess = a
            lr = -np.dot(gradient, p.T) / np.dot(p.T, np.dot(hess, p))
            x += lr * p[0]
            y += lr * p[1]
            self.x_data_2.append(x)
            self.y_data_2.append(y)

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.x_data_1, self.y_data_1 = [], []
            self.x_data_2, self.y_data_2 = [], []
            self.graph()
            self.x_data_1, self.y_data_1 = [event.xdata], [event.ydata]
            self.x_data_2, self.y_data_2 = [event.xdata], [event.ydata]
            self.path_1, = self.axes_1.plot([], label="Gradient Descent Path")
            self.path_2, = self.axes_2.plot([], label="Conjugate Gradient Path")
            self.steepest_descent(event.xdata, event.ydata)
            self.conjugate_gradient(event.xdata, event.ydata)
            self.graph()
