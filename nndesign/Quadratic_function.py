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


x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0,
              0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
y = np.copy(x)
X, Y = np.meshgrid(x, y)


class QuadraticFunction(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(QuadraticFunction, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Taylor Series", 8, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

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

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Function F", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_point, = self.axes_1.plot([], "*")

        self.axes_2 = Axes3D(self.figure2)
        self.axes_2.set_title("Function F", fontdict={'fontsize': 10})
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)

        self.a_11 = QtWidgets.QLineEdit()
        self.a_11.setText("1.5")
        self.a_11.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_11)
        self.wid3.setLayout(self.layout3)

        self.a_12 = QtWidgets.QLineEdit()
        self.a_12.setText("-0.7")
        self.a_11.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_12)
        self.wid3.setLayout(self.layout3)

        self.a_21 = QtWidgets.QLineEdit()
        self.a_21.setText("-0.7")
        self.a_21.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_21)
        self.wid3.setLayout(self.layout3)

        self.a_22 = QtWidgets.QLineEdit()
        self.a_22.setText("1")
        self.a_22.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_22)
        self.wid3.setLayout(self.layout3)

        self.d_1 = QtWidgets.QLineEdit()
        self.d_1.setText("0.35")
        self.d_1.setGeometry(250 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(250 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.d_1)
        self.wid3.setLayout(self.layout3)

        self.d_2 = QtWidgets.QLineEdit()
        self.d_2.setText("0.25")
        self.d_2.setGeometry(250 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(250 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.d_2)
        self.wid3.setLayout(self.layout3)

        self.c = QtWidgets.QLineEdit()
        self.c.setText("1")
        self.c.setGeometry(350 * self.w_ratio, 530 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(350 * self.w_ratio, 530 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.c)
        self.wid3.setLayout(self.layout3)

        self.run_button = QtWidgets.QPushButton("Update", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

        self.on_run()

    def on_run(self):
        if self.a_12.text() != self.a_21.text():
            self.a_21.setText(self.a_12.text())
        A = np.array([[float(self.a_11.text()), float(self.a_12.text())], [float(self.a_21.text()), float(self.a_22.text())]])
        d = np.array([[float(self.d_1.text())], [float(self.d_2.text())]])
        c = float(self.c.text())
        self.update(A, d, c)

    def update(self, A, d, c):

        minima = -np.dot(np.linalg.inv(A), d)
        x0, y0 = minima[0, 0], minima[1, 0]
        XX = X + x0
        YY = Y + y0
        F = (A[0, 0] * XX ** 2 + (A[0, 1] + A[1, 0]) * XX * YY + A[1, 1] * YY ** 2) / 2 + d[0, 0] * XX + d[
            1, 0] * YY + c
        e, v = np.linalg.eig(A)

        # Removes stuff
        while self.axes_1.collections:
            for collection in self.axes_1.collections:
                collection.remove()
        while self.axes_2.collections:
            for collection in self.axes_2.collections:
                collection.remove()

        # Draws new stuff
        self.axes_1.contour(X, Y, F)
        self.axes_1.quiver([0], [0], [-v[0, 0]], [-v[0, 1]], units="xy", scale=1, label="Eigenvector 1")
        self.axes_1.quiver([0], [0], [v[1, 0]], [v[1, 1]], units="xy", scale=1, label="Eigenvector 2")
        self.axes_2.plot_surface(X, Y, F, color="blue")
        self.canvas.draw()
        self.canvas2.draw()
