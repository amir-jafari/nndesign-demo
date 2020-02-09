from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class GradientDescentStochastic(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(GradientDescentStochastic, self).__init__(w_ratio, h_ratio, main_menu=2, create_plot=False)

        self.fill_chapter("Gradient Descent Stochastic", 3, "",
                          PACKAGE_PATH + "Chapters/3_D/Logo_Ch_3.svg", PACKAGE_PATH + "Chapters/2_D/poslinNet_new.svg", icon_move_left=120)

        self.data = []

        self.figure1 = Figure()
        self.canvas1 = FigureCanvas(self.figure1)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis = Axes3D(self.figure)
        self.axis.mouse_init

        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 300 * self.h_ratio, 255 * self.w_ratio, 370 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(260 * self.w_ratio, 300 * self.h_ratio, 255 * self.w_ratio, 370 * self.h_ratio)
        self.layout2.addWidget(self.canvas1)
        self.wid2.setLayout(self.layout2)

        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("Learning rate: 0.01")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(self.x_chapter_slider_label * self.w_ratio, 400, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 30)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(1)
        self.wid_lr = QtWidgets.QWidget(self)
        self.layout_lr = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_lr.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_lr.addWidget(self.slider_lr)
        self.wid_lr.setLayout(self.layout_lr)
        self.lr = float(self.slider_lr.value() / 100)

        # self.run_button = QtWidgets.QPushButton("Start", self)
        # self.run_button.setStyleSheet("font-size:13px")
        # self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 460 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)

        self.slider_lr.valueChanged.connect(self.slider)
        # self.run_button.clicked.connect(self.graph)

        self.graph()

    def slider(self):
        self.lr = float(self.slider_lr.value() / 100)
        self.label_lr.setText("Learning_rate: {}".format(self.lr))
        self.graph()

    def on_mouseclick(self, event):
        """Add an item to the plot"""
        if event.xdata != None and event.xdata != None:
            self.data.append((event.xdata, event.ydata))
            self.a1.plot(np.array([self.data[-1][0]]), np.array([self.data[-1][1]]), 'bo')
            self.graph()
            self.canvas1.draw()

    def graph(self):

        aa = self.axis
        aa.clear()  # Clear the plot

        self.a1 = self.figure1.add_subplot(111)
        self.a1.clear()  # Clear the plot

        hh = np.array([[-1, 2, 0, - 1], [2, - 1, - 1, 0]])
        t = np.array( [-1, -1, 1, 1]).reshape(-1,1)
        jj = np.dot(hh , np.transpose(hh))
        jt = np.dot(hh , t)
        a = 2 * jj
        b = -2 * jt
        c = np.dot(np.transpose(t),t)
        tt = np.arange(0.01, 1, 0.01) * 2 * np.pi
        circ_x1 = np.sin(tt) * .01 * (3 / 2)
        circ_y1 = np.cos(tt) * .01 * (3 / 2)
        circ_x2 = np.sin(tt) * .02 * (3 / 2)
        circ_y2 = np.cos(tt) * .02 * (3 / 2)
        x10 = np.array([-1])
        x20 = np.array([-2.95])
        y = np.linspace(-3,0,61)
        x = y
        X1, X2 = np.meshgrid(x, y)
        F = (a[0, 0] * np.power(X1, 2) + (a[0, 1] + a[1, 0]) * (X1 * X2) + a[1, 1] * np.power(X2, 2)) / 2 + b[0] * X1 + b[1] * X2 + c
        xc = circ_x2 + x10
        yc = circ_y2 + x20
        disp_freq = 1
        max_epoch = 80
        err_goal = -3.999

        if self.data == []:
            x1 = x10
            x2 = x20
        else:
            x1 = np.array([self.data[-1][0]])
            x2 = np.array([self.data[-1][1]])

        for i in range(max_epoch):
            Lx1 = x1
            Lx2 = x2
            select = np.random.randint(4)
            p = hh[:, select].reshape(-1,1)
            e = t[select] - np.dot(np.array([x1.flatten(), x2.flatten()]).reshape(1, 2), p).flatten()
            grad = 2 * e * p
            grad1 = np.dot(a,np.array([x1.flatten(), x2.flatten()]).reshape(2, 1))+b

            dx1 = self.lr * grad[0]
            dx2 = self.lr * grad[1]

            x1 = x1 + dx1
            x2 = x2 + dx2

            SSE = (a[0, 0] * np.power(x1, 2) + (a[0, 1] + a[1, 0]) * (x1 * x2) + a[1, 1] * np.power(x2, 2)) / 2 + b[0] * x1 + b[1] * X2 + c


            self.a1.plot(np.concatenate((Lx1, x1), 0), np.concatenate((Lx2, x2), 0), 'bo-', markersize=1)
            F1 = (a[0, 0] * np.power(x1, 2) + (a[0, 1] + a[1, 0]) * (x1 * x2) + a[1, 1] * np.power(x2, 2)) / 2 + b[0] * x1 + b[1] * x1 + c
            aa.plot(x1,x2,F1.flatten(),'ro', markersize=4)
            #aa.scatter3D(x1,x2,F1.flatten(),s=40, c='Red', marker='o')

        aa.plot_surface(X1, X2, F,color='g')
        #aa.plot_wireframe(X1, X2, F, rcount=30,ccount=30)

        self.a1.contour(X1, X2, F)
        self.a1
        self.a1.set_xlabel(r'$\mathrm{w}_{1,1}$')
        self.a1.set_ylabel(r'$\mathrm{w}_{1,2}$')
        self.a1.yaxis.tick_right()

        # Setting limits so that the point moves instead of the plot.
        #a.set_xlim(-4, 4)
        #a.set_ylim(-2, 2)

        # add grid and axes
        aa.grid(True, which='both')

        self.canvas.draw()
        self.canvas1.draw()
        self.canvas1.mpl_connect('button_press_event', self.on_mouseclick)
