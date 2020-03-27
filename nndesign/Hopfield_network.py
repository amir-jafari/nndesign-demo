from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


x = np.arange(-1, 1.0001, 0.05)
y = np.copy(x)
X, Y = np.meshgrid(x, y)


class HopfieldNetwork(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(HopfieldNetwork, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Hopfield Network", 21, " TODO",
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
        self.axes_1.set_title("Lyapunov Function", fontdict={'fontsize': 10})
        self.axes_1.set_xlabel("$a1$")
        self.axes_1.set_ylabel("$a2$")
        self.axes_1.set_xlim(-1, 1)
        self.axes_1.set_ylim(-1, 1)
        self.axes1_point, = self.axes_1.plot([], "*")
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.x_data, self.y_data = None, None
        self.x, self.y = None, None
        self.ani = None
        self.path, = self.axes_1.plot([], color="blue")
        self.r = None

        self.axes_2 = Axes3D(self.figure2)
        self.axes_2.set_title("Lyapunov Function", fontdict={'fontsize': 10})
        self.axes_2.set_xlabel("$a1$")
        self.axes_2.set_ylabel("$a2$")
        self.axes_2.set_zlabel("$V(a)$")
        self.axes_2.set_xlim(-1, 1)
        self.axes_2.set_ylim(-1, 1)
        # self.axes_2.set_zlim(-1, 2)

        self.a_11 = QtWidgets.QLineEdit()
        self.a_11.setText("0")
        self.a_11.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_11)
        self.wid3.setLayout(self.layout3)

        self.a_12 = QtWidgets.QLineEdit()
        self.a_12.setText("1")
        self.a_11.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_12)
        self.wid3.setLayout(self.layout3)

        self.a_21 = QtWidgets.QLineEdit()
        self.a_21.setText("1")
        self.a_21.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_21)
        self.wid3.setLayout(self.layout3)

        self.a_22 = QtWidgets.QLineEdit()
        self.a_22.setText("0")
        self.a_22.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(120 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.a_22)
        self.wid3.setLayout(self.layout3)

        self.d_1 = QtWidgets.QLineEdit()
        self.d_1.setText("0")
        self.d_1.setGeometry(250 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(250 * self.w_ratio, 500 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.d_1)
        self.wid3.setLayout(self.layout3)

        self.d_2 = QtWidgets.QLineEdit()
        self.d_2.setText("0")
        self.d_2.setGeometry(250 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(250 * self.w_ratio, 570 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.d_2)
        self.wid3.setLayout(self.layout3)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("Finite Value Gain: 1.4")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(self.x_chapter_slider_label * self.w_ratio, 470 * self.h_ratio, 150 * self.w_ratio,
                                 100 * self.h_ratio)
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(0, 20)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(14)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_b)
        self.wid4.setLayout(self.layout4)
        self.slider_b.valueChanged.connect(self.on_run)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.finite_gain = True
        self.finite_value_gain = None
        self.comboBox1.addItems(["Finite Gain", "Infinite Gain"])
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("Gain")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 550 * self.h_ratio,
                                 150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)
        self.comboBox1.currentIndexChanged.connect(self.change_gain)

        self.run_button = QtWidgets.QPushButton("Update", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

        self.on_run()

    def hop(self, t, y):
        a = 2 / np.pi * np.arctan(self.finite_value_gain * np.pi * y * 0.5)
        return -y + np.dot(self.W, a) + self.b

    def hopi(self, t, y):
        return (0.5 * np.dot(self.W, y.reshape(-1, 1)) + self.b).reshape(-1)

    def animate_init(self):
        self.path.set_data(self.x_data, self.y_data)
        self.finite_value_gain = self.slider_b.value() / 10
        self.W = np.array(
            [[float(self.a_11.text()), float(self.a_12.text())], [float(self.a_21.text()), float(self.a_22.text())]])
        self.b = np.array([[float(self.d_1.text())], [float(self.d_2.text())]])
        if self.finite_gain:
            self.r = ode(self.hop).set_integrator("zvode")
            n0 = np.array([2 * np.tan(np.pi * x / 2) / self.finite_value_gain / np.pi,
                           2 * np.tan(np.pi * y / 2) / self.finite_value_gain / np.pi])
            self.r.set_initial_value(n0, 0)
            t1 = 10
            dt = 0.1
            while self.r.successful() and self.r.t < t1:
                N = self.r.integrate(self.r.t + dt)
            a = 2 * np.arctan(self.finite_value_gain * np.pi * N / 2) / np.pi
        else:
            self.r = ode(self.hopi).set_integrator("zvode")
            n0 = np.array([[self.x], [self.y]])
            self.r.set_initial_value(n0, 0)
            t1 = 10
            dt = 0.1
            while self.r.successful() and self.r.t < t1:
                N = self.r.integrate(self.r.t + dt)
            a = N * (N < 1) * 1 + (N >= 1) * 1
        self.a = a * (a > -1) * 1 - (a <= -1) * 1
        return self.path,

    def on_animate(self, idx):
        self.path.set_data(self.a[:idx, 0], self.a[:idx, 1])
        return self.path,

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            if self.ani:
                self.ani.event_source.stop()
            self.x_data, self.y_data = [event.xdata], [event.ydata]
            self.x, self.y = event.xdata, event.ydata
            self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=100,
                                     interval=100, repeat=False, blit=True)

    def change_gain(self, idx):
        self.finite_gain = idx == 0
        self.on_run()

    def on_run(self):
        W = np.array([[float(self.a_11.text()), float(self.a_12.text())], [float(self.a_21.text()), float(self.a_22.text())]])
        b = np.array([[float(self.d_1.text())], [float(self.d_2.text())]])
        self.update(W, b)

    def update(self, W, b):

        finite_value_gain = self.slider_b.value() / 10
        self.label_b.setText("b: " + str(finite_value_gain))
        if not self.finite_gain:
            finite_value_gain = np.inf

        F = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                a = np.array([[X[i, j]], [Y[i, j]]])
                F[i, j] = -0.5 * np.dot(a.T, np.dot(W, a)) - np.dot(b.T, a)
                for k in range(2):
                    temp1 = np.cos(np.pi / 2 * a[k, 0])
                    if temp1 == 0:
                        temp2 = -np.inf
                    else:
                        temp2 = np.log(np.clip(temp1, 0.001, 100))
                    F[i, j] = F[i, j] - 4 / (finite_value_gain * np.pi ** 2) * temp2

        # Removes stuff
        while self.axes_1.collections:
            for collection in self.axes_1.collections:
                collection.remove()
        while self.axes_2.collections:
            for collection in self.axes_2.collections:
                collection.remove()

        # Draws new stuff
        self.axes_1.contour(X, Y, F, levels=[-5, -2, -1, -0.5, -0.041, -0.023, -0.003, 0.017, 0.16, 0.45, 1, 2, 4, 8, 16])
        # indxx = 3:2:(length(xx)-2);
        #   indyy = 3:2:(length(yy)-2);
        #   xx = xx(indxx);
        #   yy = yy(indyy);
        #   F = F(indxx,:);
        #   F = F(:,indyy);
        #   func_mesh = mesh(xx,yy,F)
        indxx = np.arange(2, len(X) - 2, 2)
        indyy = np.arange(2, len(Y) - 2, 2)
        xx = x[indxx]
        yy = y[indyy]
        XX, YY = np.meshgrid(xx, yy)
        F = F[indxx, :]
        F = F[:, indyy]
        self.axes_2.plot_surface(XX, YY, F, color="blue")
        self.canvas.draw()
        self.canvas2.draw()
