from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class LinearTransformations(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(LinearTransformations, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Linear Transformations", 6, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.cid1, self.cid2 = None, None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(20 * self.w_ratio, 120 * self.h_ratio, 230 * self.w_ratio, 230 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)
        self.vectors = []
        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Original Vectors")
        self.axes_1.set_xlim(-1, 1)
        self.axes_1.set_ylim(-1, 1)
        self.line, = self.axes_1.plot([], linestyle="-", color="gray")
        self.line_data_x, self.line_data_y = [], []
        # self.line2, = self.axes_1.plot([], linestyle="-", label="line2")
        self.line2 = None
        self.line_data_x2, self.line_data_y2 = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(270 * self.w_ratio, 120 * self.h_ratio, 230 * self.w_ratio, 230 * self.h_ratio)
        self.layout2.addWidget(self.canvas2)
        self.wid2.setLayout(self.layout2)
        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_xlim(-1, 1)
        self.axes_2.set_ylim(-1, 1)
        self.axes_2.set_title("Transformed Vectors")
        self.point2, = self.axes_2.plot([], marker='*')
        self.line2, = self.axes_2.plot([], linestyle="--")
        self.line_data_x2, self.line_data_y2 = [], []
        self.canvas2.draw()
        # self.canvas2.mpl_connect('button_press_event', self.on_mouseclick)

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.vectors.append((event.xdata, event.ydata))
            n_vectors = len(self.vectors)
            if n_vectors == 1:
                self.line_data_x, self.line_data_y = [event.xdata], [event.ydata]
                self.line.set_data(self.line_data_x, self.line_data_y)
                self.axes_1.quiver([0], [0], [event.xdata], [event.ydata], units="xy", scale=1, label="Vector 1")
                self.cid1 = self.canvas.mpl_connect("motion_notify_event", self.on_mousepressed1)
            elif n_vectors == 2:
                self.axes_1.quiver([0], [0], [event.xdata], [event.ydata], units="xy", scale=1, label="Transformed 1", color="r")
                self.canvas.mpl_disconnect(self.cid1)
            elif n_vectors == 3:
                self.line_data_x2, self.line_data_y2 = [event.xdata], [event.ydata]
                self.line2.set_data(self.line_data_x2, self.line_data_y2)
                self.axes_1.quiver([0], [0], [event.xdata], [event.ydata], units="xy", scale=1, label="Vector 2")
                self.cid2 = self.canvas.mpl_connect("motion_notify_event", self.on_mousepressed2)
            elif n_vectors == 4:
                self.axes_1.quiver([0], [0], [event.xdata], [event.ydata], units="xy", scale=1, label="Transformed 2", color="r")
                self.canvas.mpl_disconnect(self.cid2)
                # TODO: Add tranformation stuff here
            else:
                return
            self.canvas.draw()

    def on_mousepressed1(self, event):
        if event.xdata != None and event.ydata != None:
            self.line_data_x.append(event.xdata)
            self.line_data_y.append(event.ydata)
            self.line.set_data(self.line_data_x, self.line_data_y)
            self.canvas.draw()
            self.line_data_x.pop()
            self.line_data_y.pop()

    def on_mousepressed2(self, event):
        if event.xdata != None and event.ydata != None:
            self.line_data_x2.append(event.xdata)
            self.line_data_y2.append(event.ydata)
            while self.axes_1.lines:
                self.axes_1.lines.pop()
            self.axes_1.lines.append(self.line)
            self.axes_1.plot(self.line_data_x2, self.line_data_y2, linestyle="-", color="gray")
            # self.line2.set_data(self.line_data_x2, self.line_data_y2)
            self.canvas.draw()
            self.line_data_x2.pop()
            self.line_data_y2.pop()
