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

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class SupervisedHebb(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(SupervisedHebb, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Supervised Hebb", 7, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        wid_up = 1
        hei_up = 1
        nrows_up = 6
        ncols_up = 5
        inbetween_up = 0.1
        xx = np.arange(0, ncols_up, (wid_up + inbetween_up))
        yy = np.arange(0, nrows_up, (hei_up + inbetween_up))

        wid_down = 1
        hei_down = 1
        nrows_down = 6
        ncols_down = 5
        inbetween_down = 0.1

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
        self.patches1 = []
        for xi in xx:
            for yi in yy:
                sq = patches.Rectangle((xi, yi), wid_up, hei_up, fill=True)
                self.patches1.append(sq)
                # self.axis1.add_patch(sq)
        self.p1 = coll.PatchCollection(self.patches1, cmap=cm.prism, alpha=0.4)
        self.axis1.add_collection(self.p1)
        self.axis1.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis1.axis("off")

        self.canvas1.show()

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
        for xi in xx:
            for yi in yy:
                sq = patches.Rectangle((xi, yi), wid_up, hei_up, fill=True)
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
        for xi in xx:
            for yi in yy:
                sq = patches.Rectangle((xi, yi), wid_up, hei_up, fill=True, color="yellow")
                self.axis3.add_patch(sq)
        self.axis3.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis3.axis("off")

        self.canvas3.show()

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

        self.canvas4.show()

        self.figure5 = Figure()
        self.canvas5 = FigureCanvas(self.figure5)
        self.toolbar5 = NavigationToolbar(self.canvas5, self)
        self.wid5 = QtWidgets.QWidget(self)
        self.layout5 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid5.setGeometry(250 * self.w_ratio, 350 * self.h_ratio,
                              240 * self.w_ratio, 240 * self.h_ratio)
        self.layout5.addWidget(self.canvas5)
        self.wid5.setLayout(self.layout5)

        self.canvas5.show()

        # --

        self.run_button = QtWidgets.QPushButton("Run", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def on_run(self):

        wid_up = 1
        hei_up = 1
        nrows_up = 6
        ncols_up = 5
        inbetween_up = 0.1
        xx = np.arange(0, ncols_up, (wid_up + inbetween_up))
        yy = np.arange(0, nrows_up, (hei_up + inbetween_up))

        colors = 100 * np.random.rand(len(self.patches1))
        # colors = ["yellow"] * len(self.patches1)
        # colors[0] = "red"
        self.p1.set_array(np.array(colors))

        # self.axis1.patches[0] = patches.Rectangle((xx[0], yy[0]), wid_up, hei_up, fill=True, color="red")
        self.canvas1.draw()
