from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class GramSchmidt(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(GramSchmidt, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Gram-Schmidt", 5, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("-", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_points = []
        self.axes1_v1 = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1, label="y1", color="r")
        self.axes1_v2 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, label="y2", color="r")
        self.axes1_proj = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0)
        self.axes1_proj1 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0)
        self.axes1_proj_line, = self.axes_1.plot([], "-")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick1)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("-", fontdict={'fontsize': 10})
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.axes2_v1 = self.axes_2.quiver([0], [0], [0], [0], units="xy", scale=1, label="y1", color="g")
        self.axes2_v2 = self.axes_2.quiver([0], [0], [0], [0], units="xy", scale=1, label="y2", color="g")
        self.canvas2.draw()

        self.button = QtWidgets.QPushButton("Compute", self)
        self.button.setStyleSheet("font-size:13px")
        self.button.setGeometry(self.x_chapter_button * self.w_ratio, 500 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.button.clicked.connect(self.gram_schmidt)

        self.button = QtWidgets.QPushButton("Restart", self)
        self.button.setStyleSheet("font-size:13px")
        self.button.setGeometry(self.x_chapter_button * self.w_ratio, 530 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.button.clicked.connect(self.clear_all)

        self.label_warning = QtWidgets.QLabel(self)
        self.label_warning.setText("")
        self.label_warning.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_warning.setGeometry(self.x_chapter_slider_label * self.w_ratio, 550 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

    def on_mouseclick1(self, event):
        if event.xdata != None and event.xdata != None:
            self.axes1_points.append((event.xdata, event.ydata))
            self.draw_vector()

    def draw_vector(self):
        if len(self.axes1_points) == 1:
            self.axes1_v1.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
        elif len(self.axes1_points) == 2:
            self.axes1_v2.set_UVC(self.axes1_points[1][0], self.axes1_points[1][1])
        self.canvas.draw()

    def gram_schmidt(self):
        if len(self.axes1_points) < 2:
            self.label_warning.setText("WHOOPS! Please enter two vectors")
            return
        cos_angle = (self.axes1_points[0][0] * self.axes1_points[1][0] + self.axes1_points[1][0] * self.axes1_points[1][1]) / (
            np.sqrt(self.axes1_points[0][0]**2 + self.axes1_points[0][1] ** 2) * np.sqrt(self.axes1_points[1][0]**2 + self.axes1_points[1][1] ** 2)
        )
        if np.abs(self.axes1_points[0]).sum() == 0 or np.abs(self.axes1_points[1]).sum() == 0:  # If a vector is at the origin
            self.axes1_points = []
            self.axes1_v1.set_UVC(0, 0)
            self.axes1_v2.set_UVC(0, 0)
            self.label_warning.setText("WHOOPS!  You entered a zero vector. Please try again!")
        elif cos_angle == 1:
            self.axes1_points = []
            self.axes1_v1.set_UVC(0, 0)
            self.axes1_v2.set_UVC(0, 0)
            self.label_warning.setText("WHOOPS!  You entered parallel vectors, which cannot be orthogonalized. Please try again!")
        elif cos_angle == 0:
            self.axes1_points = []
            self.axes1_v1.set_UVC(0, 0)
            self.axes1_v2.set_UVC(0, 0)
            self.label_warning.setText("Wooow!  You entered vectors that are already orthogonal. Please try again!")
        else:
            v1 = np.array([[self.axes1_points[0][0]], [self.axes1_points[0][1]]])
            v2 = np.array([[self.axes1_points[1][0]], [self.axes1_points[1][1]]])
            a = np.dot(v1.T, v2) / np.dot(v1.T, v1)
            proj = a * v1
            self.axes1_proj.set_UVC(proj[0, 0], proj[1, 0])
            # self.axes1_proj_line.set_data([proj[0, 0], v2[0, 0]], [proj[1, 0], v2[1, 0]])
            # self.axes1_proj1 = self.axes_1.quiver([proj[0, 0]], [proj[1, 0]], [v2[0, 0] - proj[0, 0]], [v2[1, 0] - proj[1, 0]], units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0)
            # self.axes1_proj1 = self.axes_1.quiver([v2[0, 0]], [v2[1, 0]], [proj[0, 0] - v2[0, 0]], [proj[1, 0] - v2[1, 0]], units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0)
            self.axes1_proj1 = self.axes_1.quiver([proj[0, 0] - np.sqrt(proj[0, 0] ** 2 + proj[1, 0] ** 2)], [proj[1, 0] - np.sqrt(proj[0, 0] ** 2 + proj[1, 0] ** 2)], [v2[0, 0] - np.sqrt(proj[0, 0] ** 2 + proj[1, 0] ** 2)],
                                                  [v2[1, 0] - np.sqrt(proj[0, 0] ** 2 + proj[1, 0] ** 2)], units="xy", scale=1, headlength=0,
                                                  headwidth=0, headaxislength=0)
            # self.axes1_proj_line.set_data([v2[0, 0], proj[0, 0]], [v2[1, 0], proj[1, 0]])
            # self.axes1_proj_line.set_data([0, proj[0, 0]], [0, proj[1, 0]])
            v2 = v2 - proj
            self.axes2_v1.set_UVC(v1[0, 0], v1[1, 0])
            self.axes2_v2.set_UVC(v2[0, 0], v2[1, 0])
            # self.axes1_proj1.set_UVC(v2[0, 0] - np.sqrt(proj[0, 0] ** 2 + proj[1, 0] ** 2), v2[1, 0] - np.sqrt(proj[0, 0] ** 2 + proj[1, 0] ** 2))
        self.canvas.draw()
        self.canvas2.draw()

    def clear_all(self):
        self.axes1_v1.set_UVC(0, 0)
        self.axes1_v2.set_UVC(0, 0)
        self.axes2_v1.set_UVC(0, 0)
        self.axes2_v2.set_UVC(0, 0)
        self.axes1_points = []
        self.canvas.draw()
        self.canvas2.draw()
