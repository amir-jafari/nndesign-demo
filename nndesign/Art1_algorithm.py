from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


wid_up = 0.99
hei_up = 1.01
nrows_up = 5
ncols_up = 5
inbetween_up = 0.11
xx_up = np.arange(0, ncols_up, (wid_up + inbetween_up))
yy_up = np.arange(0, nrows_up, (hei_up + inbetween_up))


class ART1Algorithm(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(ART1Algorithm, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("ART1 Algorithm", 10, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.s2, self.s1 = 4, 25
        self.w21, self.w12 = np.ones((self.s1, self.s2)), np.zeros((self.s2, self.s1))
        for k in range(4):
            self.w12[k, :] = 2 * self.w21[:, k].T / (2 + np.sum(self.w21[:, k]) - 1)

        self.label_pattern1 = QtWidgets.QLabel(self)
        self.label_pattern1.setText("Pattern 1")
        self.label_pattern1.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern1.setGeometry(60 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure1 = Figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 130 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout1.addWidget(self.canvas1)
        self.wid1.setLayout(self.layout1)
        self.axis1 = self.figure1.add_axes([0, 0, 1, 1])
        self.pattern1 = [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
        self.pattern11 = np.flip(np.array(self.pattern1).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern11[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis1.add_patch(sq)
        self.axis1.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis1.axis("off")
        self.canvas1.draw()
        self.canvas1.mpl_connect("button_press_event", self.on_mouseclick1)

        self.button1 = QtWidgets.QPushButton("Present", self)
        self.button1.setStyleSheet("font-size:13px")
        self.button1.setGeometry(40 * self.w_ratio, 250 * self.h_ratio,
                                 100 * self.w_ratio, 25 * self.h_ratio)
        self.button1.clicked.connect(self.button1_pressed)

        # --

        self.label_pattern2 = QtWidgets.QLabel(self)
        self.label_pattern2.setText("Pattern 2")
        self.label_pattern2.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern2.setGeometry(170 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure2 = Figure(frameon=False)
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(140 * self.w_ratio, 130 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)
        self.axis2 = self.figure2.add_axes([0, 0, 1, 1])
        self.pattern2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
        self.pattern22 = np.flip(np.array(self.pattern2).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern22[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis2.add_patch(sq)
        self.axis2.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis2.axis("off")
        self.canvas2.draw()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)

        self.button2 = QtWidgets.QPushButton("Present", self)
        self.button2.setStyleSheet("font-size:13px")
        self.button2.setGeometry((30 + 125) * self.w_ratio, 250 * self.h_ratio,
                                 100 * self.w_ratio, 25 * self.h_ratio)
        self.button2.clicked.connect(self.button2_pressed)

        # --

        self.label_pattern3 = QtWidgets.QLabel(self)
        self.label_pattern3.setText("Pattern 3")
        self.label_pattern3.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern3.setGeometry(310 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure3 = Figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(260 * self.w_ratio, 130 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout3.addWidget(self.canvas3)
        self.wid3.setLayout(self.layout3)
        self.axis3 = self.figure3.add_axes([0, 0, 1, 1])
        self.pattern3 = [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.pattern33 = np.flip(np.array(self.pattern3).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern33[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis3.add_patch(sq)
        self.axis3.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis3.axis("off")
        self.canvas3.draw()
        self.canvas3.mpl_connect("button_press_event", self.on_mouseclick3)

        self.button3 = QtWidgets.QPushButton("Present", self)
        self.button3.setStyleSheet("font-size:13px")
        self.button3.setGeometry((30 + 125 * 2) * self.w_ratio, 250 * self.h_ratio,
                                 100 * self.w_ratio, 25 * self.h_ratio)
        self.button3.clicked.connect(self.button3_pressed)

        # ---

        self.label_pattern31 = QtWidgets.QLabel(self)
        self.label_pattern31.setText("Pattern 4")
        self.label_pattern31.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern31.setGeometry(450 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure31 = Figure()
        self.canvas31 = FigureCanvas(self.figure31)
        self.toolbar31 = NavigationToolbar(self.canvas31, self)
        self.wid31 = QtWidgets.QWidget(self)
        self.layout31 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid31.setGeometry(390 * self.w_ratio, 130 * self.h_ratio, 130 * self.w_ratio, 130 * self.h_ratio)
        self.layout31.addWidget(self.canvas31)
        self.wid31.setLayout(self.layout31)
        self.axis31 = self.figure31.add_axes([0, 0, 1, 1])
        self.pattern31 = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.pattern331 = np.flip(np.array(self.pattern31).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern331[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis31.add_patch(sq)
        self.axis31.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis31.axis("off")
        self.canvas31.draw()
        self.canvas31.mpl_connect("button_press_event", self.on_mouseclick31)

        self.button31 = QtWidgets.QPushButton("Present", self)
        self.button31.setStyleSheet("font-size:13px")
        self.button31.setGeometry((30 + 125 * 3) * self.w_ratio, 250 * self.h_ratio,
                                  100 * self.w_ratio, 25 * self.h_ratio)
        self.button31.clicked.connect(self.button31_pressed)

        # ---

        self.label_pattern4 = QtWidgets.QLabel(self)
        self.label_pattern4.setText("Prototype 1")
        self.label_pattern4.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern4.setGeometry(60 * self.w_ratio, 275 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure4 = Figure()
        self.canvas4 = FigureCanvas(self.figure4)
        self.toolbar4 = NavigationToolbar(self.canvas4, self)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(15 * self.w_ratio, 300 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout4.addWidget(self.canvas4)
        self.wid4.setLayout(self.layout4)
        self.axis4 = self.figure4.add_axes([0, 0, 1, 1])
        self.pattern4 = [0] * 25
        self.pattern44 = np.flip(np.array(self.pattern4).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern44[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis4.add_patch(sq)
        self.axis4.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis4.axis("off")
        self.canvas4.draw()

        # ---

        self.label_pattern5 = QtWidgets.QLabel(self)
        self.label_pattern5.setText("Prototype 2")
        self.label_pattern5.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern5.setGeometry(160 * self.w_ratio, 275 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure5 = Figure()
        self.canvas5 = FigureCanvas(self.figure5)
        self.toolbar5 = NavigationToolbar(self.canvas5, self)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(140 * self.w_ratio, 300 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout5.addWidget(self.canvas5)
        self.wid5.setLayout(self.layout5)
        self.axis5 = self.figure5.add_axes([0, 0, 1, 1])
        self.pattern5 = [0] * 25
        self.pattern55 = np.flip(np.array(self.pattern5).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern55[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis5.add_patch(sq)
        self.axis5.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis5.axis("off")
        self.canvas5.draw()

        # --

        self.label_pattern6 = QtWidgets.QLabel(self)
        self.label_pattern6.setText("Prototype 3")
        self.label_pattern6.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern6.setGeometry(290 * self.w_ratio, 275 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure6 = Figure()
        self.canvas6 = FigureCanvas(self.figure6)
        self.toolbar6 = NavigationToolbar(self.canvas6, self)
        self.wid6 = QtWidgets.QWidget(self)
        self.layout6 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid6.setGeometry(260 * self.w_ratio, 300 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout6.addWidget(self.canvas6)
        self.wid6.setLayout(self.layout6)
        self.axis6 = self.figure6.add_axes([0, 0, 1, 1])
        self.pattern6 = [0] * 25
        self.pattern66 = np.flip(np.array(self.pattern6).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern66[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis6.add_patch(sq)
        self.axis6.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis6.axis("off")
        self.canvas6.draw()

        # --

        self.label_pattern7 = QtWidgets.QLabel(self)
        self.label_pattern7.setText("Prototype 4")
        self.label_pattern7.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern7.setGeometry(420 * self.w_ratio, 275 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure7 = Figure()
        self.canvas7 = FigureCanvas(self.figure7)
        self.toolbar7 = NavigationToolbar(self.canvas7, self)
        self.wid7 = QtWidgets.QWidget(self)
        self.layout7 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid7.setGeometry(390 * self.w_ratio, 300 * self.h_ratio,
                              130 * self.w_ratio, 130 * self.h_ratio)
        self.layout7.addWidget(self.canvas7)
        self.wid7.setLayout(self.layout7)
        self.axis7 = self.figure7.add_axes([0, 0, 1, 1])
        self.pattern7 = [0] * 25
        self.pattern77 = np.flip(np.array(self.pattern7).reshape((ncols_up, nrows_up)), axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern77[yi, xi] == 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis7.add_patch(sq)
        self.axis7.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis7.axis("off")
        self.canvas7.draw()

        # --

        self.rho = 0.6
        self.label_rho = QtWidgets.QLabel(self)
        self.label_rho.setText("rho: 0.6")
        self.label_rho.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_rho.setGeometry(self.x_chapter_slider_label * self.w_ratio, 470 * self.h_ratio, 150 * self.w_ratio,
                                   100 * self.h_ratio)
        self.slider_rho = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_rho.setRange(0, 100)
        self.slider_rho.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_rho.setTickInterval(1)
        self.slider_rho.setValue(60)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_rho)
        self.wid4.setLayout(self.layout4)
        self.slider_rho.valueChanged.connect(self.slide)

    def slide(self):
        self.rho = self.slider_rho.value() / 10
        self.label_rho.setText("rho: " + str(self.rho))

    def change_squares(self, axis, canvas, k):
        while axis.patches:
            axis.patches.pop()
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if np.flip(self.w21[:, k].reshape((ncols_up, nrows_up)), axis=1)[xi, yi] > 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                axis.add_patch(sq)
        canvas.draw()

    def change_prototype(self, button_id):

        P = 1 - np.array([self.pattern1, self.pattern2, self.pattern3, self.pattern31]).T
        ind_x = []
        while True:
            if button_id == 1:
                p = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
            elif button_id == 2:
                p = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0])
            elif button_id == 3:
                p = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
            else:
                p = P[:, button_id]
            a1 = p

            n1 = np.dot(self.w12, a1)
            n1[ind_x] = -np.inf
            k = np.argmax(n1)
            a2 = np.zeros((self.s2, 1))
            a2[k, 0] = 1

            expect = self.w21[:, k]
            a1 = p * expect

            if np.sum(a1) / np.sum(p) < self.rho:
                a0 = 1
            else:
                a0 = 0

            if a0:
                ind_x.append(k)
                if len(ind_x) == self.s2:
                    if self.s2 == 4:
                        print("More than four prototypes needed")
                    else:
                        self.w21 = np.hstack((self.w21, np.ones((self.s1, 1))))
                        self.w12 = np.vstack((self.w12, 2 * np.ones((1, self.s1)) / (2 + self.s1 - 1)))
                        self.s2 += 1
                print("TODO")
            else:
                self.w12[k, :] = 2 * a1.T / (2 + np.sum(a1) - 1)
                self.w21[:, k] = a1
                break

        k = 0
        for axis, canvas in zip([self.axis4, self.axis5, self.axis6, self.axis7], [self.canvas4, self.canvas5, self.canvas6, self.canvas7]):
            self.change_squares(axis, canvas, k)
            k += 1

    def button1_pressed(self):
        self.change_prototype(0)

    def button2_pressed(self):
        self.change_prototype(1)

    def button3_pressed(self):
        self.change_prototype(2)

    def button31_pressed(self):
        self.change_prototype(3)

    def on_mouseclick1(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis1.patches:
                self.axis1.patches.pop()
            if self.pattern11[yyy, xxx] == 0:
                self.pattern11[yyy, xxx] = 1
            else:
                self.pattern11[yyy, xxx] = 0
            self.pattern1 = np.flip(self.pattern11.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern11[yi, xi] == 0:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis1.add_patch(sq)
            self.canvas1.draw()

    def on_mouseclick2(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis2.patches:
                self.axis2.patches.pop()
            if self.pattern22[yyy, xxx] == 0:
                self.pattern22[yyy, xxx] = 1
            else:
                self.pattern22[yyy, xxx] = 0
            self.pattern2 = np.flip(self.pattern22.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern22[yi, xi] == 0:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis2.add_patch(sq)
            self.canvas2.draw()

    def on_mouseclick3(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis3.patches:
                self.axis3.patches.pop()
            if self.pattern33[yyy, xxx] == 0:
                self.pattern33[yyy, xxx] = 1
            else:
                self.pattern33[yyy, xxx] = 0
            self.pattern3 = np.flip(self.pattern33.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern33[yi, xi] == 0:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis3.add_patch(sq)
            self.canvas3.draw()

    def on_mouseclick31(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis31.patches:
                self.axis31.patches.pop()
            if self.pattern331[yyy, xxx] == 0:
                self.pattern331[yyy, xxx] = 1
            else:
                self.pattern331[yyy, xxx] = 0
            self.pattern31 = np.flip(self.pattern331.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern331[yi, xi] == 0:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis31.add_patch(sq)
            self.canvas31.draw()
