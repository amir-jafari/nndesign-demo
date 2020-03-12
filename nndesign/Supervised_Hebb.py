from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.collections as coll
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


wid_up = 1
hei_up = 1
nrows_up = 6
ncols_up = 5
inbetween_up = 0.1
xx_up = np.arange(0, ncols_up, (wid_up + inbetween_up))
yy_up = np.arange(0, nrows_up, (hei_up + inbetween_up))


class SupervisedHebb(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(SupervisedHebb, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Supervised Hebb", 7, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.label_pattern1 = QtWidgets.QLabel(self)
        self.label_pattern1.setText("Pattern 1")
        self.label_pattern1.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern1.setGeometry(75 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure1 = Figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 130 * self.h_ratio,
                              170 * self.w_ratio, 170 * self.h_ratio)
        self.layout1.addWidget(self.canvas1)
        self.wid1.setLayout(self.layout1)
        self.axis1 = self.figure1.add_axes([0, 0, 1, 1])
        self.pattern1 = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]
        self.pattern11 = np.flip(np.array(self.pattern1).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern11[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis1.add_patch(sq)
        self.axis1.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis1.axis("off")
        self.canvas1.show()
        self.canvas1.mpl_connect("button_press_event", self.on_mouseclick1)

        # --

        self.label_pattern2 = QtWidgets.QLabel(self)
        self.label_pattern2.setText("Pattern 2")
        self.label_pattern2.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern2.setGeometry(235 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure2 = Figure(frameon=False)
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(175 * self.w_ratio, 130 * self.h_ratio,
                              170 * self.w_ratio, 170 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)
        self.axis2 = self.figure2.add_axes([0, 0, 1, 1])
        self.pattern2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.pattern22 = np.flip(np.array(self.pattern2).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern22[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis2.add_patch(sq)
        self.axis2.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis2.axis("off")
        self.canvas2.show()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)

        # --

        self.label_pattern3 = QtWidgets.QLabel(self)
        self.label_pattern3.setText("Pattern 3")
        self.label_pattern3.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern3.setGeometry(390 * self.w_ratio, 105 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure3 = Figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(335 * self.w_ratio, 130 * self.h_ratio,
                              170 * self.w_ratio, 170 * self.h_ratio)
        self.layout3.addWidget(self.canvas3)
        self.wid3.setLayout(self.layout3)
        self.axis3 = self.figure3.add_axes([0, 0, 1, 1])
        self.pattern3 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
        self.pattern33 = np.flip(np.array(self.pattern3).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern33[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis3.add_patch(sq)
        self.axis3.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis3.axis("off")
        self.canvas3.show()
        self.canvas3.mpl_connect("button_press_event", self.on_mouseclick3)

        # ---

        self.label_pattern4 = QtWidgets.QLabel(self)
        self.label_pattern4.setText("Test Pattern")
        self.label_pattern4.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern4.setGeometry(115 * self.w_ratio, 325 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure4 = Figure()
        self.canvas4 = FigureCanvas(self.figure4)
        self.toolbar4 = NavigationToolbar(self.canvas4, self)
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(30 * self.w_ratio, 350 * self.h_ratio,
                              240 * self.w_ratio, 240 * self.h_ratio)
        self.layout4.addWidget(self.canvas4)
        self.wid4.setLayout(self.layout4)
        self.axis4 = self.figure4.add_axes([0, 0, 1, 1])
        self.pattern4 = self.pattern1[:]
        self.pattern44 = np.copy(self.pattern11)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern44[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="gray")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis4.add_patch(sq)
        self.axis4.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis4.axis("off")
        self.canvas4.draw()
        self.canvas4.mpl_connect("button_press_event", self.on_mouseclick4)

        # ---

        self.label_pattern5 = QtWidgets.QLabel(self)
        self.label_pattern5.setText("Response Pattern")
        self.label_pattern5.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_pattern5.setGeometry(320 * self.w_ratio, 325 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.figure5 = Figure()
        self.canvas5 = FigureCanvas(self.figure5)
        self.toolbar5 = NavigationToolbar(self.canvas5, self)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(250 * self.w_ratio, 350 * self.h_ratio,
                              240 * self.w_ratio, 240 * self.h_ratio)
        self.layout5.addWidget(self.canvas5)
        self.wid5.setLayout(self.layout5)
        self.axis5 = self.figure5.add_axes([0, 0, 1, 1])
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern11[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis5.add_patch(sq)
        self.axis5.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis5.axis("off")
        self.canvas5.draw()

        # --

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.addItems(["Hebb", 'Pseudoinverse'])
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("Rule")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 14, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 550 * self.h_ratio,
                                 150 * self.w_ratio, 100 * self.h_ratio)
        self.comboBox1.currentIndexChanged.connect(self.change_rule)
        self.rule = 0
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.run_button = QtWidgets.QPushButton("Weights", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def on_run(self):

        pattern = np.array([self.pattern1, self.pattern2, self.pattern3]).T * 2 - 1
        if self.rule == 0:
            w = np.dot(pattern, pattern.T)
        elif self.rule == 1:
            w = np.dot(pattern, np.linalg.pinv(pattern))
        plt.imshow(w)
        plt.title("Network Weights")
        plt.xlabel("Input")
        plt.ylabel("Neuron")
        # TODO: Make actual squares with difference in size

    def on_mouseclick1(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis1.patches:
                self.axis1.patches.pop()
            if self.pattern11[yyy, xxx] == 1:
                self.pattern11[yyy, xxx] = 0
            else:
                self.pattern11[yyy, xxx] = 1
            self.pattern1 = np.flip(self.pattern11.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern11[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis1.add_patch(sq)
            self.canvas1.draw()
            self.response()

    def on_mouseclick2(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis2.patches:
                self.axis2.patches.pop()
            if self.pattern22[yyy, xxx] == 1:
                self.pattern22[yyy, xxx] = 0
            else:
                self.pattern22[yyy, xxx] = 1
            self.pattern2 = np.flip(self.pattern22.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern22[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis2.add_patch(sq)
            self.canvas2.draw()
            self.response()

    def on_mouseclick3(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis3.patches:
                self.axis3.patches.pop()
            if self.pattern33[yyy, xxx] == 1:
                self.pattern33[yyy, xxx] = 0
            else:
                self.pattern33[yyy, xxx] = 1
            self.pattern3 = np.flip(self.pattern33.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern33[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis3.add_patch(sq)
            self.canvas3.draw()
            self.response()

    def on_mouseclick4(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis4.patches:
                self.axis4.patches.pop()
            if self.pattern44[yyy, xxx] == 1:
                self.pattern44[yyy, xxx] = 0
            else:
                self.pattern44[yyy, xxx] = 1
            self.pattern4 = np.flip(self.pattern44.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern44[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="gray")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis4.add_patch(sq)
            self.canvas4.draw()
            self.response()

    def response(self):
        pattern = np.array([self.pattern1, self.pattern2, self.pattern3]).T * 2 - 1
        p = np.array(self.pattern4).T * 2 - 1
        if self.rule == 0:
            w = np.dot(pattern, pattern.T)
        elif self.rule == 1:
            w = np.dot(pattern, np.linalg.pinv(pattern))
        a = np.flip(np.dot(w, p).reshape(ncols_up, nrows_up).T, axis=0)
        while self.axis5.patches:
            self.axis5.patches.pop()
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if a[yi, xi] > 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis5.add_patch(sq)
        self.canvas5.draw()

    def change_rule(self, idx):
        self.rule = idx
        self.response()
