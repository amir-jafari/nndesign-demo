from PyQt5 import QtWidgets, QtGui, QtCore, QtSvg
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from nndesign_layout import NNDLayout
from get_package_path import PACKAGE_PATH


class PerceptronClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(PerceptronClassification, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Perceptron Classification", 3, "Click Go to... TODO",  # TODO
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
        self.axis.scatter(apple[0], apple[1], apple[2], color='yellow')
        self.line1, self.line2, self.line3 = None, None, None
        self.canvas.draw()

        self.p, self.a, self.label = None, None, None

        self.label_w = QtWidgets.QLabel(self)
        self.label_w.setText("")
        self.label_w.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w.setGeometry(550 * self.w_ratio, 300 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(550 * self.w_ratio, 330 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_p = QtWidgets.QLabel(self)
        self.label_p.setText("")
        self.label_p.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_p.setGeometry(550 * self.w_ratio, 360 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_1 = QtWidgets.QLabel(self)
        self.label_a_1.setText("")
        self.label_a_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_1.setGeometry(550 * self.w_ratio, 390 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_2 = QtWidgets.QLabel(self)
        self.label_a_2.setText("")
        self.label_a_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_2.setGeometry(550 * self.w_ratio, 420 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_3 = QtWidgets.QLabel(self)
        self.label_a_3.setText("")
        self.label_a_3.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_3.setGeometry(550 * self.w_ratio, 450 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_fruit = QtWidgets.QLabel(self)
        self.label_fruit.setText("")
        self.label_fruit.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_fruit.setGeometry(550 * self.w_ratio, 480 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.icon3 = QtWidgets.QLabel(self)
        self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon3.setGeometry(15 * self.w_ratio, 500 * self.h_ratio, 500 * self.w_ratio, 150 * self.h_ratio)

        """from PIL import Image
        # import svgutils.compose as sc
        img = Image.open(PACKAGE_PATH + "Figures/nnd3d1.eps")
        img.load()
        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(15 * self.w_ratio, 500 * self.h_ratio, 500 * self.w_ratio, 150 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)

        self.axes2 = self.figure2.add_subplot(1, 1, 1)
        self.axes2.imshow(np.array(img))
        self.canvas2.draw()"""

        self.run_button = QtWidgets.QPushButton("Go", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 520 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def on_run(self):
        self.timer = QtCore.QTimer()
        self.idx = 0
        self.timer.timeout.connect(self.update_label)
        self.timer.start(1000)

    def update_label(self):
        if self.idx == 0:
            self.label_w.setText(""); self.label_b.setText(""); self.label_p.setText("")
            self.label_a_1.setText(""); self.label_a_2.setText(""); self.label_a_3.setText("")
            self.label_fruit.setText("")
            self.p = np.round(np.random.uniform(-1, 1, (1, 3)), 2)
            self.a = 0 * self.p[0, 0] + 1 * self.p[0, 1] + 0 * self.p[0, 2]
            self.label = 1 if self.a >= 0 else -1
            if self.line1:
                self.line1.pop().remove()
                self.line2.pop().remove()
                self.line3.pop().remove()
                self.canvas.draw()
        if self.idx == 1:
            self.label_w.setText("w = [0 1 0]")
            self.label_b.setText("b = 0")
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_2.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_7.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 2:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_3.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_8.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 3:
            self.label_p.setText("p = [{} {} {}]".format(self.p[0, 0], self.p[0, 1], self.p[0, 2]))
            self.line1 = self.axis.plot3D([self.p[0, 0]] * 10, np.linspace(-1, 1, 10), [self.p[0, 2]] * 10, color="g")
            self.line2 = self.axis.plot3D([self.p[0, 0]] * 10, [self.p[0, 1]] * 10, np.linspace(-1, 1, 10), color="g")
            self.line3 = self.axis.plot3D(np.linspace(-1, 1, 10), [self.p[0, 1]] * 10, [self.p[0, 2]] * 10, color="g")
            self.canvas.draw()
        elif self.idx == 4:
            self.label_a_1.setText("a = hardlims(W * p + b)")
        elif self.idx == 5:
            self.label_a_2.setText("a = hardlims({})".format(self.a))
        elif self.idx == 6:
            self.label_a_3.setText("a =" + str(self.label))
        elif self.idx == 7:
            self.label_fruit.setText("Fruit = {}".format("Apple" if self.label == 1 else "Orange"))
        elif self.idx == 8:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_4.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_9.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 9:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_5.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_10.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 10:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_6.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_11.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        else:
            pass
        self.idx += 1
