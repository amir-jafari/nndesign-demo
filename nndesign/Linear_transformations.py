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

# v1 = v11 * e1 + v12 * e2
# v2 = v21 * e1 + v22 * e2
#
# y1 = y11 * e1 + y12 * e2
# y2 = y21 * e1 + y22 * e2
#
# T(v1) = y1 —> v11 * T(e1) + v12 * T(e2) = y11 * e1 + y12  * e2
# T(v2) = y2 —> v21 * T(e1) + v22 * T(e2) = y21 * e1 + y22 * e2
#               ----------------------------——————————————————————
# 			T(e1) = ( y11 * e1 + y12  * e2 - v12 * T(e2) ) / v11 = ( y21 * e1 + y22 * e2 - v22 * T(e2) ) / v21 —>
# 			—> - v12 * T(e2) = (v11 / v21)  * ( y21 * e1 + y22 * e2 - v22 * T(e2) ) - y11 * e1 + y12  * e2 —>
# 			—> T(e2) * (-v12 + v22 * v11 / v21) = (v11 / v21)  * ( y21 * e1 + y22 * e2) - y11 * e1 + y12  * e2 —>
# 			—> T(e2) = ( ( v11 * y21 / v21 - y11 ) * e1 + ( v11 * y22 / v21 + y12 ) * e2 ) / (-v12 + v22 * v11 / v21)
# 			—> T(e1) = ( y11 - v12 * ( v11 * y21 / v21 - y11 ) / (-v12 + v22 * v11 / v21) ) * e1 +
# 						+ ( y12 - v12 * ( v11 * y22 / v21 + y12 ) / (-v12 + v22 * v11 / v21) ) * e2
#
# A_div = -v12 + v22 * v11 / v21
#
# A = [ T(e1) T(e2) ] = [  (y11 - v12 * ( v11 * y21 / v21 - y11 ) / A_div) / v11    ( v11 * y21 / v21 - y11 ) / A_div   ]
# 				 [  (y12 - v12 * ( v11 * y22 / v21 + y12 ) /A_div) / v11     ( v11 * y22 / v21 + y12 ) / A_div ]


class LinearTransformations(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(LinearTransformations, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Linear Transformations", 6, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.cid1, self.cid2 = None, None
        self.cid3 = None
        self.A, self.eig_v1, self.eig_v2 = None, None, None

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
        self.line, = self.axes_1.plot([], linestyle="--", color="gray")
        self.line_data_x, self.line_data_y = [], []
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
        self.canvas2.mpl_connect('button_press_event', self.on_mouseclick2)

        self.run_button = QtWidgets.QPushButton("Clear", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

        self.on_run()

    def on_run(self):
        if self.cid3:
            self.canvas2.mpl_disconnect(self.cid3)
        if self.cid1:
            self.canvas.mpl_disconnect(self.cid1)
        if self.cid2:
            self.canvas.mpl_disconnect(self.cid2)
        self.vectors = []
        self.line_data_x, self.line_data_y = [], []
        while self.axes_1.lines:
            self.axes_1.lines.pop()
        while self.axes_1.collections:
            for collection in self.axes_1.collections:
                collection.remove()
        self.line, = self.axes_1.plot([], linestyle="--", color="gray")
        self.line2, = self.axes_2.plot([], linestyle="--")
        self.line_data_x2, self.line_data_y2 = [], []
        while self.axes_2.collections:
            for collection in self.axes_2.collections:
                collection.remove()
        self.canvas.draw()
        self.canvas2.draw()

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
                v1, y1, v2, y2 = self.vectors[0], self.vectors[1], self.vectors[2], self.vectors[3]
                A_div = (-v1[1] + v2[1] * v1[0] / v2[0])
                A_11 = (y1[0] - v1[1] * (v1[0] * y2[0] / v2[0] - y1[0]) / A_div) / v1[0]
                A_12 = (v1[0] * y2[0] / v2[0] - y1[0]) / A_div
                A_21 = (y1[1] - v1[1] * (v1[0] - y2[1] / v2[0] + y1[1]) / A_div) / v1[0]
                A_22 = (v1[0] + y2[1] / v2[0] + y1[1]) / A_div
                self.A = np.array([[A_11, A_12], [A_21, A_22]])
                e, v = np.linalg.eig(self.A)
                if "complex" in str(v.dtype):
                    print("Complex!!!")
                else:
                    self.axes_2.quiver([0], [0], [-v[0, 0]], [-v[0, 1]], units="xy", scale=1, label="Eigenvector 1", color="g")
                    self.axes_2.quiver([0], [0], [v[1, 0]], [v[1, 1]], units="xy", scale=1, label="Eigenvector 2", color="g")
                self.canvas2.draw()
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
            self.axes_1.plot(self.line_data_x2, self.line_data_y2, linestyle="--", color="gray")
            self.canvas.draw()
            self.line_data_x2.pop()
            self.line_data_y2.pop()

    def on_mouseclick2(self, event):
        if event.xdata != None and event.xdata != None and len(self.vectors) == 4:
            if self.axes_2.collections:
                while len(self.axes_2.collections) > 2:
                    self.axes_2.collections[-1].remove()
            if self.cid3:
                self.canvas2.mpl_disconnect(self.cid3)
            self.axes_2.quiver([0], [0], [event.xdata], [event.ydata], units="xy", scale=1)
            v_transformed = np.dot(self.A, np.array([[event.xdata], [event.ydata]]))
            self.axes_2.quiver([0], [0], [v_transformed[0, 0]], [v_transformed[1, 0]], units="xy", scale=1, color="g")
            self.cid3 = self.canvas2.mpl_connect("motion_notify_event", self.on_mousepressed3)
            self.canvas2.draw()

    def on_mousepressed3(self, event):
        if event.xdata != None and event.ydata != None:
            self.axes_2.collections[-1].remove()
            self.axes_2.collections[-1].remove()
            self.axes_2.quiver([0], [0], [event.xdata], [event.ydata], units="xy", scale=1)
            v_transformed = np.dot(self.A, np.array([[event.xdata], [event.ydata]]))
            self.axes_2.quiver([0], [0], [v_transformed[0, 0]], [v_transformed[1, 0]], units="xy", scale=1, color="g")
            self.canvas2.draw()