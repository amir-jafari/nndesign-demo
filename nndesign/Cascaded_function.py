from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class CascadedFunction(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(CascadedFunction, self).__init__(w_ratio, h_ratio, main_menu=2)

        self.fill_chapter("Cascaded Function", 2, "",
                          PACKAGE_PATH + "Chapters/2_D/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2_D/2f_1_1.svg", icon_move_left=120)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.Poslin, self.LogSig]
        self.comboBox1.addItems(["ReLU", 'LogSig'])
        self.comboBox1.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 410 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.func = self.Poslin
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 400 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(self.x_chapter_usual * self.w_ratio, 420 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout1.addWidget(self.comboBox1)
        self.wid1.setLayout(self.layout1)

        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.addItems(["Two", 'Three', "Four"])
        self.func1 = self.two
        self.comboBox2.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 480 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.label_iter = QtWidgets.QLabel(self)
        self.label_iter.setText("Number of iterations")
        self.label_iter.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_iter.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 470 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 490 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout2.addWidget(self.comboBox2)
        self.wid2.setLayout(self.layout2)

        self.sliderval = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderval.setRange(0, 50)
        self.sliderval.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderval.setTickInterval(1)
        self.sliderval.setValue(0)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 530 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.sliderval)
        self.wid3.setLayout(self.layout3)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.comboBox2.currentIndexChanged.connect(self.combo_bbox2)
        self.sliderval.valueChanged.connect(self.graph)

        self.graph()

    def combo_bbox2(self, idx):
        if idx == 0:
            self.func1 = self.two
            self.icon2.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Chapters/2_D/2f_1.svg").pixmap(500 * self.w_ratio, 200 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            self.graph()
        elif idx == 1:
            self.func1 = self.three
            self.icon2.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Chapters/2_D/3f_1.svg").pixmap(500 * self.w_ratio, 200 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            self.graph()
        if idx == 2:
            self.func1 = self.four
            self.icon2.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Chapters/2_D/4f_1.svg").pixmap(500 * self.w_ratio, 200 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            self.graph()

    def graph(self):

        a = self.figure.add_subplot(111)
        a.clear()  # Clear the plot
        p1 = np.arange(0, 1, 0.01)
        a2 = self.net1(p1)

        a.plot(p1, p1, 'g')
        a.plot(p1, a2, 'b')

        if self.func1() == 2:
            self.aaa = np.linspace(0.01, 0.99, 100)
        elif self.func1() == 3:
            self.aaa = np.linspace(0.01, 0.99, 200)
        else:
            self.aaa = np.linspace(0.01, 0.99, 300)

        self.last_idx = 0
        self.plot_1, = a.plot([], 'ko-')

        a.axhline(y=0, color='k')
        a.axvline(x=0, color='k')
        if self.sliderval.value() == 0:
            self.ani = FuncAnimation(self.figure, self.on_animate, frames=len(self.aaa), interval=50, repeat=False)
            self.canvas.draw()
        else:

            for i in range(len(self.aaa)):
                xx, yy = self.getxx(self.aaa[i], self.func1())
                a.plot(xx[-1], yy[-1], 'ro')

            xx, yy = self.getxx(float((self.sliderval.value()) / 50), self.func1())
            a.plot(xx, yy, 'ko-')

            self.canvas.draw()

    def on_animate(self, idx):  # This idx is needed, even if it's not being used explicitly!
        if self.last_idx < len(self.aaa):
            xx, yy = self.getxx(self.aaa[self.last_idx], self.func1())
            self.last_idx += 1
            self.plot_1.set_data(xx, yy)
            self.figure.add_subplot(111).plot(xx[-1], yy[-1], 'ro')
            self.canvas.draw()

    def getxx(self, p, nr):
        xx = np.array([])
        yy = np.array([])
        x = p
        x1 = p
        for i in range(int(nr)):
            out1 = self.net1(x)
            out1 = float(out1)
            xx = np.concatenate(np.array([xx, [x, x, x, out1]]))
            yy = np.concatenate(np.array([yy, [x, out1, out1, out1]]))
            x = out1
        yy[0] = np.array([0])
        xx[-1] = x1
        return xx, yy

    def net1(self, x):
        if hasattr(self, 'func'):
            if self.func.__str__().split()[2].split('.')[1] == 'LogSig':
                w1 = np.array([[15, 15]]).reshape(-1, 1)
                b1 = np.array([[-0.25*15, -(1-0.25)*15]]).reshape(-1, 1)
                w2 = np.array([[1, -1]])
                b2 = np.array([0])
                func3 = np.vectorize(self.func, otypes=[np.float])
                a1 = func3(w1 * x + b1 * np.ones((1, 1)))
                y = np.dot(w2, a1) + b2*np.ones((1, 1))
                y1 = y.flatten()
            else:
                w1 = np.array([[1, 1]]).reshape(-1, 1)
                b1 = np.array([[0, -0.5]]).reshape(-1, 1)
                w2 = np.array([[2, -4]])
                b2 = np.array([0])
                func = np.vectorize(self.func, otypes=[np.float])
                a1 = func(w1 * x + b1 * np.ones((1, 1)))
                y = np.dot(w2, a1) + b2 * np.ones((1, 1))
                y1 = y.flatten()
        else:
            w1 = np.array([[1, 1]]).reshape(-1, 1)
            b1 = np.array([[0, -0.5]]).reshape(-1, 1)
            w2 = np.array([[2, -4]])
            b2 = np.array([0])
            func = np.vectorize(self.Poslin, otypes=[np.float])
            a1 = func(w1 * x + b1 * np.ones((1, 1)))
            y = np.dot(w2, a1) + b2*np.ones((1, 1))
            y1 = y.flatten()
        return y1

    def change_transfer_function(self, idx):
        self.func = self.comboBox1_functions[idx]
        self.graph()

    @staticmethod
    def two():
        return np.array([2])

    @staticmethod
    def three():
        return np.array([3])

    @staticmethod
    def four():
        return np.array([4])

    def Poslin(self, x):
        if (x < 0):
            return 0
        else:
            return x

    def LogSig(self, x):
        return 1 / (1 + math.e ** (-x))
