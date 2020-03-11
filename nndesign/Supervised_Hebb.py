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
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="yellow")
                self.axis1.add_patch(sq)
        self.axis1.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis1.axis("off")
        self.canvas1.show()
        self.canvas1.mpl_connect("button_press_event", self.on_mouseclick)

        # --

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
        pattern2 = np.flip(np.array(self.pattern2).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if pattern2[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="yellow")
                self.axis2.add_patch(sq)
        self.axis2.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis2.axis("off")
        self.canvas2.show()

        # --

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
        pattern3 = np.flip(np.array(self.pattern3).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if pattern3[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="yellow")
                self.axis3.add_patch(sq)
        self.axis3.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis3.axis("off")
        self.canvas3.show()

        # ---

        self.P = np.array([self.pattern1, self.pattern2, self.pattern3]).T * 2 - 1
        self.p = np.array(self.pattern1).T * 2 - 1

        w = np.dot(self.P, self.P.T)
        a = np.dot(w, self.p).reshape((len(np.dot(w, self.p)), 1))

        print(w)
        print(a)

        # ---

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
        # self.pattern4 = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]
        # pattern4 = np.flip(np.array(self.pattern1).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.p.reshape(nrows_up, ncols_up)[yi, xi] > 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="yellow")
                self.axis4.add_patch(sq)
        self.axis4.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis4.axis("off")
        self.canvas4.draw()

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
        self.pattern5 = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]
        pattern5 = np.flip(np.array(self.pattern1).reshape((ncols_up, nrows_up)).T, axis=0)
        print(a.reshape(nrows_up, ncols_up))
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if a.reshape(nrows_up, ncols_up)[yi, xi] > 0:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="yellow")
                self.axis5.add_patch(sq)
        self.axis5.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis5.axis("off")
        self.canvas5.draw()

        # --

        self.run_button = QtWidgets.QPushButton("Run", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def on_run(self):

        while self.axis3.patches:
            self.axis3.patches.pop()
        for xi in xx_up:
            for yi in yy_up:
                sq = patches.Rectangle((xi, yi), wid_up, hei_up, fill=True, color="red")
                self.axis3.add_patch(sq)

        self.canvas3.draw()

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx) for xx in xx_up]
            d_y = [abs(event.ydata - yy) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            print(xxx, yyy)
            print(event.xdata, event.ydata)
            print(xx_up)
            print(yy_up)
            while self.axis1.patches:
                self.axis1.patches.pop()
            if self.pattern11[yyy, xxx] == 1:
                self.pattern11[yyy, xxx] = 0
            else:
                self.pattern11[yyy, xxx] = 1
            # self.patter1 =  # TODO
            # self.pattern1 = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]
            # pattern1 = np.flip(np.array(self.pattern1).reshape((ncols_up, nrows_up)).T, axis=0)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern11[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="yellow")
                    self.axis1.add_patch(sq)
            self.canvas1.draw()
