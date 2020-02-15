from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class ReciprocalBasis(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(ReciprocalBasis, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Reciprocal Basis", 5, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Basis Vectors", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_points = []
        self.axes1_v1 = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1, label="v1", color="r")
        self.axes1_v2 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, label="v2", color="b")
        self.axes1_s1 = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1, label="s1", color="m")
        self.axes1_s2 = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1, label="s2", color="y")
        self.axes1_x = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1, label="x", color="g")
        self.axes1_proj = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0)
        self.axes1_proj1 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0)
        self.axes1_proj_line, = self.axes_1.plot([], "-")
        self.axes_1.legend()
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick1)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Vector Expansion", fontdict={'fontsize': 10})
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.axes2_v1 = self.axes_2.quiver([0], [0], [0], [0], units="xy", scale=1, label="v1", color="r")
        self.axes2_v2 = self.axes_2.quiver([0], [0], [0], [0], units="xy", scale=1, label="v2", color="b")
        self.axes2_x = self.axes_2.quiver([0], [0], [0], [0], units="xy", scale=1, label="x", color="g")
        self.axes2_line1, = self.axes_2.plot([], "-")
        self.axes2_line1.set_color("black")
        self.axes2_line2, = self.axes_2.plot([], "-")
        self.axes2_line2.set_color("black")
        self.axes_2.legend()
        self.canvas2.draw()

        self.button = QtWidgets.QPushButton("Expand", self)
        self.button.setStyleSheet("font-size:13px")
        self.button.setGeometry(self.x_chapter_button * self.w_ratio, 500 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.button.clicked.connect(self.expand)

        self.button = QtWidgets.QPushButton("Restart", self)
        self.button.setStyleSheet("font-size:13px")
        self.button.setGeometry(self.x_chapter_button * self.w_ratio, 530 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.button.clicked.connect(self.clear_all)

        self.label_explanation = QtWidgets.QLabel(self)
        self.label_explanation.setText("")
        self.label_explanation.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_explanation.setGeometry(530 * self.w_ratio, 300 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        self.label_warning = QtWidgets.QLabel(self)
        self.label_warning.setText("")
        self.label_warning.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_warning.setGeometry(510 * self.w_ratio, 550 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

    def on_mouseclick1(self, event):
        if event.xdata != None and event.xdata != None:
            self.axes1_points.append((event.xdata, event.ydata))
            self.draw_vector()

    def draw_vector(self):
        if len(self.axes1_points) == 1:
            self.axes1_v1.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
        elif len(self.axes1_points) == 2:
            cos_angle = (self.axes1_points[0][0] * self.axes1_points[1][0] + self.axes1_points[1][0] *
                         self.axes1_points[1][1]) / (
                                np.sqrt(self.axes1_points[0][0] ** 2 + self.axes1_points[0][1] ** 2) * np.sqrt(
                            self.axes1_points[1][0] ** 2 + self.axes1_points[1][1] ** 2)
                        )
            if cos_angle == 1:
                self.axes1_points = []
                self.axes1_v1.set_UVC(0, 0)
                self.axes1_v2.set_UVC(0, 0)
                self.label_warning.setText("WHOOPS!  You entered parallel vectors, which cannot form a basis. Please try again!")
            else:
                self.axes1_v2.set_UVC(self.axes1_points[1][0], self.axes1_points[1][1])
                self.axes1_s1.set_UVC(1, 0)
                self.axes1_s2.set_UVC(0, 1)
        elif len(self.axes1_points) == 3:
            self.axes1_x.set_UVC(self.axes1_points[2][0], self.axes1_points[2][1])
        self.canvas.draw()

    def expand(self):
        if len(self.axes1_points) < 3:
            self.label_warning.setText("WHOOPS! Please enter two basis vector and one additional vector to span")
            return
        explanation = "Your vector x is:\n x = {} * s1 + {} * s2\n".format(round(self.axes1_points[2][0], 2), round(self.axes1_points[2][1], 2))
        b = np.array([[self.axes1_points[0][0], self.axes1_points[1][0]],
                      [self.axes1_points[0][1], self.axes1_points[1][1]]])
        x = np.array([[self.axes1_points[2][0]], [self.axes1_points[2][1]]])
        xv = np.dot(np.linalg.inv(b), x)
        explanation += "The expansion for x in terms of \nv1 and v2 is:\n x = {} * v1 + {} * v2".format(round(xv[0, 0], 2), round(xv[1, 0], 2))
        self.label_explanation.setText(explanation)
        self.axes2_line1.set_data([0, xv[0, 0] * self.axes1_points[0][0]], [0, xv[0, 0] * self.axes1_points[0][1]])
        self.axes2_line2.set_data([xv[0, 0] * self.axes1_points[0][0], self.axes1_points[2][0]],
                                  [xv[0, 0] * self.axes1_points[0][1], self.axes1_points[2][1]])
        self.axes1_v1.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
        self.axes1_v2.set_UVC(self.axes1_points[1][0], self.axes1_points[1][1])
        self.axes2_v1.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
        self.axes2_v2.set_UVC(self.axes1_points[1][0], self.axes1_points[1][1])
        self.axes1_x.set_UVC(self.axes1_points[2][0], self.axes1_points[2][1])
        self.axes2_x.set_UVC(self.axes1_points[2][0], self.axes1_points[2][1])

        self.canvas.draw()
        self.canvas2.draw()

    def clear_all(self):
        self.axes1_v1.set_UVC(0, 0)
        self.axes1_v2.set_UVC(0, 0)
        self.axes1_s1.set_UVC(0, 0)
        self.axes1_s2.set_UVC(0, 0)
        self.axes2_v1.set_UVC(0, 0)
        self.axes2_v2.set_UVC(0, 0)
        self.axes1_points = []
        self.axes2_line1.set_data([], [])
        self.axes2_line2.set_data([], [])
        self.canvas.draw()
        self.canvas2.draw()
