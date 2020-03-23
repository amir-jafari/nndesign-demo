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
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


wid_up = 1
hei_up = 1.04
nrows_up = 4
ncols_up = 4
inbetween_up = 0.12
xx_up = np.arange(0, ncols_up, (wid_up + inbetween_up))
yy_up = np.arange(0, nrows_up, (hei_up + inbetween_up))


class LinearClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(LinearClassification, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Linear Classification", 10, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.label_pattern1 = QtWidgets.QLabel(self)
        self.label_pattern1.setText("Target = 60")
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
        self.pattern1 = [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
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
        self.canvas1.draw()
        self.canvas1.mpl_connect("button_press_event", self.on_mouseclick1)

        self.button1 = QtWidgets.QPushButton("->", self)
        self.button1.setStyleSheet("font-size:13px")
        self.button1.setGeometry(55 * self.w_ratio, 250 * self.h_ratio,
                                 50 * self.w_ratio, 50 * self.h_ratio)
        self.button1.clicked.connect(self.button1_pressed)

        # --

        self.label_pattern2 = QtWidgets.QLabel(self)
        self.label_pattern2.setText("Target = 0")
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
        self.pattern2 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
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
        self.canvas2.draw()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)

        self.button2 = QtWidgets.QPushButton("->", self)
        self.button2.setStyleSheet("font-size:13px")
        self.button2.setGeometry((55 + 125) * self.w_ratio, 250 * self.h_ratio,
                                 50 * self.w_ratio, 50 * self.h_ratio)
        self.button2.clicked.connect(self.button2_pressed)

        # --

        self.label_pattern3 = QtWidgets.QLabel(self)
        self.label_pattern3.setText("Target = -60")
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
        self.pattern3 = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
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
        self.canvas3.draw()
        self.canvas3.mpl_connect("button_press_event", self.on_mouseclick3)

        self.button3 = QtWidgets.QPushButton("->", self)
        self.button3.setStyleSheet("font-size:13px")
        self.button3.setGeometry((55 + 125 * 2) * self.w_ratio, 250 * self.h_ratio,
                                 50 * self.w_ratio, 50 * self.h_ratio)
        self.button3.clicked.connect(self.button3_pressed)

        # ---

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
        self.pattern4 = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
        self.pattern44 = np.flip(np.array(self.pattern4).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern44[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis4.add_patch(sq)
        self.axis4.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis4.axis("off")
        self.canvas4.draw()
        self.canvas4.mpl_connect("button_press_event", self.on_mouseclick4)

        self.button4 = QtWidgets.QPushButton("->", self)
        self.button4.setStyleSheet("font-size:13px")
        self.button4.setGeometry(55 * self.w_ratio, 420 * self.h_ratio,
                                 50 * self.w_ratio, 50 * self.h_ratio)
        self.button4.clicked.connect(self.button4_pressed)

        # ---

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
        self.pattern5 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        self.pattern55 = np.flip(np.array(self.pattern5).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern55[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis5.add_patch(sq)
        self.axis5.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis5.axis("off")
        self.canvas5.draw()
        self.canvas5.mpl_connect("button_press_event", self.on_mouseclick5)

        self.button5 = QtWidgets.QPushButton("->", self)
        self.button5.setStyleSheet("font-size:13px")
        self.button5.setGeometry((55 + 125) * self.w_ratio, 420 * self.h_ratio,
                                 50 * self.w_ratio, 50 * self.h_ratio)
        self.button5.clicked.connect(self.button5_pressed)

        # --

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
        self.pattern6 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
        self.pattern66 = np.flip(np.array(self.pattern6).reshape((ncols_up, nrows_up)).T, axis=0)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern66[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis6.add_patch(sq)
        self.axis6.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis6.axis("off")
        self.canvas6.draw()
        self.canvas6.mpl_connect("button_press_event", self.on_mouseclick6)

        self.button6 = QtWidgets.QPushButton("->", self)
        self.button6.setStyleSheet("font-size:13px")
        self.button6.setGeometry((55 + 125 * 2) * self.w_ratio, 420 * self.h_ratio,
                                 50 * self.w_ratio, 50 * self.h_ratio)
        self.button6.clicked.connect(self.button6_pressed)

        # --

        self.label_pattern7 = QtWidgets.QLabel(self)
        self.label_pattern7.setText("Test Input")
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
        self.pattern7 = self.pattern1[:]
        self.pattern77 = np.copy(self.pattern11)
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern77[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis7.add_patch(sq)
        self.axis7.axis([-0.1, ncols_up + 0.5, -0.1, nrows_up + 0.6])
        self.axis7.axis("off")
        self.canvas7.draw()
        self.canvas7.mpl_connect("button_press_event", self.on_mouseclick7)

        # --

        self.figure8 = Figure()
        self.canvas8 = FigureCanvas(self.figure8)
        self.toolbar8 = NavigationToolbar(self.canvas8, self)
        self.wid8 = QtWidgets.QWidget(self)
        self.layout8 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid8.setGeometry(5 * self.w_ratio, 470 * self.h_ratio,
                              270 * self.w_ratio, 200 * self.h_ratio)
        self.layout8.addWidget(self.canvas8)
        self.wid8.setLayout(self.layout8)
        self.axis8 = self.figure8.add_subplot(1, 1, 1)
        self.axis8.set_title("Errors")
        self.axis8.set_xlabel("Training Cycle")
        self.axis8.set_ylabel("Sum Squared Error")
        self.axis8.set_xlim(0, 200)
        self.axis8.set_ylim(1e-5, 1e5)
        self.axis8.set_yscale("log")
        self.error_plot, = self.axis8.plot([], color="red")

        # --

        self.figure9 = Figure()
        self.canvas9 = FigureCanvas(self.figure9)
        self.toolbar9 = NavigationToolbar(self.canvas9, self)
        self.wid9 = QtWidgets.QWidget(self)
        self.layout9 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid9.setGeometry(260 * self.w_ratio, 470 * self.h_ratio,
                              250 * self.w_ratio, 200 * self.h_ratio)
        self.layout9.addWidget(self.canvas9)
        self.wid9.setLayout(self.layout9)
        self.axis9 = self.figure9.add_subplot(1, 1, 1, polar=True)
        self.axis9.set_thetamin(-90)
        self.axis9.set_thetamax(90)
        self.axis9.set_title("Test Output")
        self.axis9.set_xlabel("Windrow-Hoff Metter")
        self.axis9.set_yticks([])
        self.axis9.set_theta_zero_location("N")
        self.axis9.set_theta_direction(-1)
        self.axis9.set_xticks(np.array([-60, -30, 0, 30, 60]) * np.pi / 180)
        # self.angle = 60
        # r = np.arange(0, 0.9, 0.01)
        # theta = self.angle * np.pi / 180
        self.meter, = self.axis9.plot([], color="red")
        self.angles = []
        self.ani = None
        self.angle, self.angle_end = None, None
        self.canvas9.draw()

        # --

        self.run_button = QtWidgets.QPushButton("Weights", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 520 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.response)

        self.response()

    def change_test_input(self, pattern):
        self.pattern7 = pattern[:]
        self.pattern77 = np.flip(np.array(self.pattern7).reshape((ncols_up, nrows_up)).T, axis=0)
        while self.axis7.patches:
            self.axis7.patches.pop()
        for xi in range(len(xx_up)):
            for yi in range(len(yy_up)):
                if self.pattern77[yi, xi] == 1:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                else:
                    sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                self.axis7.add_patch(sq)
        self.canvas7.draw()
        self.run_animation()

    def button1_pressed(self):
        self.change_test_input(self.pattern1)

    def button2_pressed(self):
        self.change_test_input(self.pattern2)

    def button3_pressed(self):
        self.change_test_input(self.pattern3)

    def button4_pressed(self):
        self.change_test_input(self.pattern4)

    def button5_pressed(self):
        self.change_test_input(self.pattern5)

    def button6_pressed(self):
        self.change_test_input(self.pattern6)

    def on_run(self):
        print("TODO")

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

            # self.response()

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
            # self.response()

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
            # self.response()

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
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis4.add_patch(sq)
            self.canvas4.draw()
            # self.response()

    def on_mouseclick5(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis5.patches:
                self.axis5.patches.pop()
            if self.pattern55[yyy, xxx] == 1:
                self.pattern55[yyy, xxx] = 0
            else:
                self.pattern55[yyy, xxx] = 1
            self.pattern5 = np.flip(self.pattern55.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern55[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis5.add_patch(sq)
            self.canvas5.draw()
            # self.response()

    def on_mouseclick6(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis6.patches:
                self.axis6.patches.pop()
            if self.pattern66[yyy, xxx] == 1:
                self.pattern66[yyy, xxx] = 0
            else:
                self.pattern66[yyy, xxx] = 1
            self.pattern6 = np.flip(self.pattern66.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern66[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="green")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis6.add_patch(sq)
            self.canvas6.draw()
            # self.response()

    def on_mouseclick7(self, event):
        if event.xdata != None and event.xdata != None:
            d_x = [abs(event.xdata - xx - 0.5) for xx in xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in yy_up]
            xxx, yyy = list(range(len(xx_up)))[np.argmin(d_x)], list(range(len(yy_up)))[np.argmin(d_y)]
            while self.axis7.patches:
                self.axis7.patches.pop()
            if self.pattern77[yyy, xxx] == 1:
                self.pattern77[yyy, xxx] = 0
            else:
                self.pattern77[yyy, xxx] = 1
            self.pattern7 = np.flip(self.pattern77.T, axis=1).reshape(-1)
            for xi in range(len(xx_up)):
                for yi in range(len(yy_up)):
                    if self.pattern77[yi, xi] == 1:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="red")
                    else:
                        sq = patches.Rectangle((xx_up[xi], yy_up[yi]), wid_up, hei_up, fill=True, color="khaki")
                    self.axis7.add_patch(sq)
            self.canvas7.draw()
            self.run_animation()

    def response(self):

        P = np.array([self.pattern1, self.pattern4, self.pattern5, self.pattern2, self.pattern3, self.pattern6]).T * 2 - 1
        T = np.array([[60, 60, 0, 0, -60, -60]])
        w, b = np.zeros((1, 16)), 0
        sse = []
        for i in range(200):
            q = int(np.fix(np.random.uniform(0, 1) * 6) + 1)
            p = P[:, q - 1]
            t = T[:, q - 1]
            a = np.dot(w, p) + b
            e = (t - a).item()
            w += 2 * 0.03 * e * p.T
            b += 2 * 0.03 * e
            sse.append(np.sum((T - np.dot(w, P) - b) ** 2))
        self.error_plot.set_data(range(len(sse)), sse)
        self.canvas8.draw()

        w_, b_ = np.array([[60, 0, -75, 0, -15, -15, -15, 0, 0, 0, 30, 0, 0, 0, 0, 0]]), 0
        self.angle = int((np.dot(w_, np.array([self.pattern7]).T * 2 - 1) + b_).item())
        self.draw_meter()
        self.canvas9.draw()

    def run_animation(self):
        w_, b_ = np.array([[60, 0, -75, 0, -15, -15, -15, 0, 0, 0, 30, 0, 0, 0, 0, 0]]), 0
        self.angle_end = (np.dot(w_, np.array([self.pattern7]).T * 2 - 1) + b_).item()
        if self.ani:
            self.ani.event_source.stop()
        self.ani = FuncAnimation(self.figure9, self.on_animate, init_func=self.animate_init,
                                 frames=abs(self.angle - self.angle_end), interval=50, repeat=False, blit=False)
        self.canvas9.draw()

    def draw_meter(self):
        r = np.arange(0, 0.9, 0.01)
        theta = self.angle * np.pi / 180
        self.meter.set_data(theta, r)

    def animate_init(self):
        self.angles = [self.angle]
        self.draw_meter()
        return self.meter,

    def on_animate(self, idx):
        if self.angle > self.angle_end:
            self.angle -= 1
        elif self.angle < self.angle_end:
            self.angle += 1
        self.draw_meter()
        return self.meter,
