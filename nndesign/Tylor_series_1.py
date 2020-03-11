from PyQt5 import QtWidgets, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class TylorSeries1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(TylorSeries1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Tylor Series #1", 8, " Click ... TODO",  # TODO
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("cos(x)", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-6, 6)
        self.axes_1.set_ylim(-2, 2)
        self.axes_1.set_xticks([-6, -4, -2, 0, 2, 4])
        self.axes_1.set_yticks([-2, -1, 0, 1])
        self.axes_1.set_xlabel("$x$")
        self.axes_1.xaxis.set_label_coords(1, -0.025)
        self.axes_1.set_ylabel("$y$")
        self.axes_1.yaxis.set_label_coords(-0.025, 1)
        self.x_points = np.linspace(-6, 6)
        self.axes_1.plot(self.x_points, np.cos(self.x_points), "-")
        self.axes1_point_draw, = self.axes_1.plot([], 'mo')
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation", fontdict={'fontsize': 10})
        self.f0, self.f1, self.f2, self.f3, self.f4 = None, None, None, None, None
        self.axes2_point_draw, = self.axes_2.plot([], 'mo')
        self.axes2_function, = self.axes_2.plot([], '-')
        self.axes2_approx_0, = self.axes_2.plot([], 'r-')
        self.axes2_approx_1, = self.axes_2.plot([], 'b-')
        self.axes2_approx_2, = self.axes_2.plot([], 'g-')
        self.axes2_approx_3, = self.axes_2.plot([], 'y-')
        self.axes2_approx_4, = self.axes_2.plot([], 'c-')
        self.axes_2.set_xlim(-6, 6)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_xlim(-6, 6)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_xticks([-6, -4, -2, 0, 2, 4])
        self.axes_2.set_yticks([-2, -1, 0, 1])
        self.axes_2.set_xlabel("$x$")
        self.axes_2.xaxis.set_label_coords(1, -0.025)
        self.axes_2.set_ylabel("$y$")
        self.axes_2.yaxis.set_label_coords(-0.025, 1)
        self.canvas2.draw()

        self.function_cbx = QtWidgets.QCheckBox("Function")
        self.function_cbx.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 300 * self.h_ratio,
                                      100 * self.w_ratio, 50 * self.h_ratio)
        self.function_cbx.stateChanged.connect(self.function_checked)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 300 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout2.addWidget(self.function_cbx)
        self.wid2.setLayout(self.layout2)
        self.function_cbx.setChecked(1)

        self.order0_cbx = QtWidgets.QCheckBox("Order 0")
        self.order0_cbx.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 350 * self.h_ratio,
                                    100 * self.w_ratio, 50 * self.h_ratio)
        self.order0_cbx.stateChanged.connect(self.order0_checked)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 350 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout3.addWidget(self.order0_cbx)
        self.wid3.setLayout(self.layout3)

        self.order1_cbx = QtWidgets.QCheckBox("Order 1")
        self.order1_cbx.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 400 * self.h_ratio,
                                    100 * self.w_ratio, 50 * self.h_ratio)
        self.order1_cbx.stateChanged.connect(self.order1_checked)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 400 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout3.addWidget(self.order1_cbx)
        self.wid3.setLayout(self.layout3)
        self.order1_cbx.setChecked(1)

        self.order2_cbx = QtWidgets.QCheckBox("Order 2")
        self.order2_cbx.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 450 * self.h_ratio,
                                    100 * self.w_ratio, 50 * self.h_ratio)
        self.order2_cbx.stateChanged.connect(self.order2_checked)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 450 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout3.addWidget(self.order2_cbx)
        self.wid3.setLayout(self.layout3)

        self.order3_cbx = QtWidgets.QCheckBox("Order 3")
        self.order3_cbx.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 500 * self.h_ratio,
                                    100 * self.w_ratio, 50 * self.h_ratio)
        self.order3_cbx.stateChanged.connect(self.order3_checked)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 500 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout3.addWidget(self.order3_cbx)
        self.wid3.setLayout(self.layout3)

        self.order4_cbx = QtWidgets.QCheckBox("Order 4")
        self.order4_cbx.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 550 * self.h_ratio,
                                    100 * self.w_ratio, 50 * self.h_ratio)
        self.order4_cbx.stateChanged.connect(self.order4_checked)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry((self.x_chapter_slider_label - 20) * self.w_ratio, 550 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout3.addWidget(self.order4_cbx)
        self.wid3.setLayout(self.layout3)

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.axes1_point_draw.set_data([event.xdata], [np.cos(event.xdata)])
            self.axes2_point_draw.set_data([event.xdata], [np.cos(event.xdata)])
            self.canvas.draw()
            self.f0 = np.cos(event.xdata) + np.zeros(self.x_points.shape)
            self.f1 = self.f0 - np.sin(event.xdata) * (self.x_points - event.xdata)
            self.f2 = self.f1 - np.cos(event.xdata) * (self.x_points - event.xdata) ** 2 / 2
            self.f3 = self.f2 + np.sin(event.xdata) * (self.x_points - event.xdata) ** 3 / 6
            self.f4 = self.f3 + np.cos(event.xdata) * (self.x_points - event.xdata) ** 4 / 24
            self.draw_taylor()

    def draw_taylor(self):
        if self.function_cbx.checkState():
            self.axes2_function.set_data(self.x_points, np.cos(self.x_points))
        if self.order0_cbx.checkState():
            self.axes2_approx_0.set_data(self.x_points, self.f0)
        if self.order1_cbx.checkState():
            self.axes2_approx_1.set_data(self.x_points, self.f1)
        if self.order2_cbx.checkState():
            self.axes2_approx_2.set_data(self.x_points, self.f2)
        if self.order3_cbx.checkState():
            self.axes2_approx_3.set_data(self.x_points, self.f3)
        if self.order4_cbx.checkState():
            self.axes2_approx_4.set_data(self.x_points, self.f4)
        self.canvas2.draw()

    def function_checked(self, state):
        if state == QtCore.Qt.Checked:
            self.axes2_function.set_data(self.x_points, np.cos(self.x_points))
        else:
            self.axes2_function.set_data([], [])
        self.canvas2.draw()

    def order0_checked(self, state):
        if state == QtCore.Qt.Checked:
            if self.f0 is not None:
                self.axes2_approx_0.set_data(self.x_points, self.f0)
        else:
            self.axes2_approx_0.set_data([], [])
        self.canvas2.draw()

    def order1_checked(self, state):
        if state == QtCore.Qt.Checked:
            if self.f1 is not None:
                self.axes2_approx_1.set_data(self.x_points, self.f1)
        else:
            self.axes2_approx_1.set_data([], [])
        self.canvas2.draw()

    def order2_checked(self, state):
        if state == QtCore.Qt.Checked:
            if self.f2 is not None:
                self.axes2_approx_2.set_data(self.x_points, self.f2)
        else:
            self.axes2_approx_2.set_data([], [])
        self.canvas2.draw()

    def order3_checked(self, state):
        if state == QtCore.Qt.Checked:
            if self.f3 is not None:
                self.axes2_approx_3.set_data(self.x_points, self.f3)
        else:
            self.axes2_approx_3.set_data([], [])
        self.canvas2.draw()

    def order4_checked(self, state):
        if state == QtCore.Qt.Checked:
            if self.f4 is not None:
                self.axes2_approx_4.set_data(self.x_points, self.f4)
        else:
            self.axes2_approx_4.set_data([], [])
        self.canvas2.draw()
