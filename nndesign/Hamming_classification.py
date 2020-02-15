from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from nndesign_layout import NNDLayout
from get_package_path import PACKAGE_PATH


class HammingClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(HammingClassification, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Hamming Classification", 3, "Click GO to TODO", # TODO
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis = Axes3D(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 100 * self.h_ratio, 500 * self.w_ratio, 390 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)
        ys = np.linspace(-1, 1, 100)
        zs = np.linspace(-1, 1, 100)
        Y, Z = np.meshgrid(ys, zs)
        X = 0
        apple = np.array([-1, 1, -1])
        orange = np.array([1, 1, -1])
        self.axis.set_title("Input Space")
        self.axis.plot_surface(X, Y, Z, alpha=0.5)
        self.axis.set_xlabel("texture")
        self.axis.set_xticks([-1, 1])
        self.axis.set_ylabel("shape")
        self.axis.set_yticks([-1, 1])
        self.axis.set_zlabel("weight")
        self.axis.zaxis._axinfo['label']['space_factor'] = 0.1
        self.axis.set_zticks([-1, 1])
        self.axis.scatter(orange[0], orange[1], orange[2], color='orange')
        self.axis.scatter(apple[0], apple[1], apple[2], color='red')
        self.canvas.draw()

        self.p, self.a1, self.a2, self.fruit = None, None, None, None

        self.label_w1 = QtWidgets.QLabel(self)
        self.label_w1.setText("")
        self.label_w1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1.setGeometry(550 * self.w_ratio, 300 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(550 * self.w_ratio, 330 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_w2 = QtWidgets.QLabel(self)
        self.label_w2.setText("")
        self.label_w2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2.setGeometry(550 * self.w_ratio, 360 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_p = QtWidgets.QLabel(self)
        self.label_p.setText("")
        self.label_p.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_p.setGeometry(550 * self.w_ratio, 390 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_11 = QtWidgets.QLabel(self)
        self.label_a_11.setText("")
        self.label_a_11.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_11.setGeometry(550 * self.w_ratio, 420 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_12 = QtWidgets.QLabel(self)
        self.label_a_12.setText("")
        self.label_a_12.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_12.setGeometry(550 * self.w_ratio, 450 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_21 = QtWidgets.QLabel(self)
        self.label_a_21.setText("")
        self.label_a_21.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_21.setGeometry(550 * self.w_ratio, 480 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_22 = QtWidgets.QLabel(self)
        self.label_a_22.setText("")
        self.label_a_22.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_22.setGeometry(550 * self.w_ratio, 510 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_fruit = QtWidgets.QLabel(self)
        self.label_fruit.setText("")
        self.label_fruit.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_fruit.setGeometry(550 * self.w_ratio, 540 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.run_button = QtWidgets.QPushButton("Go", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 570 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def on_run(self):
        self.timer = QtCore.QTimer()
        self.idx = 0
        self.timer.timeout.connect(self.update_label)
        self.timer.start(1000)

    def update_label(self):
        if self.idx == 0:
            self.label_w1.setText(""); self.label_b.setText(""); self.label_w2.setText(""); self.label_p.setText("")
            self.label_a_11.setText(""); self.label_a_12.setText(""); self.label_a_21.setText(""); self.label_a_22.setText("")
            self.label_fruit.setText("")
            self.p = np.round(np.random.uniform(-1, 1, (1, 3)), 2)
            # self.p = np.array([[0.88, 0.88, -0.57]])
            w1, w2 = np.array([[1, -1, -1], [1, 1, -1]]), np.array([[1, -0.5], [-0.5, 1]])
            # TODO: b is not used?
            self.a1 = np.round(np.dot(w1, self.p.T), 2)
            self.a2 = np.round(np.dot(w2, self.a1), 2)
            self.a2 = np.round(np.array([[self.poslin(self.a2[0, 0])], [self.poslin(self.a2[1, 0])]]), 2)
            self.fruit = "Orange" if self.a2[0, 0] > 0 else "Apple"
        if self.idx == 1:
            self.label_w1.setText("W1 = [1 -1 -1; 1, 1, -1]")
        elif self.idx == 2:
            self.label_b.setText("b = [3; 3]")
        elif self.idx == 3:
            self.label_w2.setText("W2 = [1 -0.5; -0.5, 1]")
        elif self.idx == 4:
            self.label_p.setText("p = [{} {} {}]".format(self.p[0, 0], self.p[0, 1], self.p[0, 2]))
        elif self.idx == 5:
            self.label_a_11.setText("a1 = purelin(W1 * p + b)")
        elif self.idx == 6:
            self.label_a_12.setText("a1 = [{} {}]".format(self.a1[0, 0], self.a1[1, 0]))
        elif self.idx == 7:
            self.label_a_21.setText("a2 = poslin(W2 * a1)")
        elif self.idx == 8:
            self.label_a_22.setText("a2 = [{} {}]".format(self.a2[0, 0], self.a2[1, 0]))
        elif self.idx == 9:
            self.label_fruit.setText("Fruit = {}".format(self.fruit))
        else:
            pass
        self.idx += 1

    @staticmethod
    def poslin(x):
        return x if x >= 0 else 0
