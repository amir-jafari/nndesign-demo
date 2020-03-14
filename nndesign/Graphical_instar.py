from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH

# Figured out why I was having so much trouble with the quiver scaling. It was because the plot was now a square...!!


class GraphicalInstar(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(GraphicalInstar, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Graphical Instar", 15, "TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Click to change weight")
        self.axes_1.set_xlim(-1.1, 1.1)
        self.axes_1.set_ylim(-1.1, 1.1)
        self.w = np.array([[1, 0.5]])
        self.axes1_w = self.axes_1.quiver([0], [0], [self.w[0, 0]], [self.w[0, 1]], units="xy", scale=1, color="green")
        self.axes_1.plot([0] * 6, np.linspace(-1.1, 1.1, 6), color="black", linestyle="--", linewidth=0.2)
        self.axes_1.plot(np.linspace(-1.1, 1.1, 6), [0] * 6, color="black", linestyle="--", linewidth=0.2)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick1)

        self.lr = 0.5
        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("Learning rate: 0.5")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(self.x_chapter_slider_label * self.w_ratio, 400 * self.h_ratio, 150 * self.w_ratio,
                                  100 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 10)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(5)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_lr)
        self.wid3.setLayout(self.layout3)
        self.slider_lr.valueChanged.connect(self.slide)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Click to change input")
        self.axes_2.set_xlim(-1.1, 1.1)
        self.axes_2.set_ylim(-1.1, 1.1)
        self.axes2_w = self.axes_2.quiver([0], [0], [self.w[0, 0]], [self.w[0, 1]], units="xy", scale=1, color="green")
        self.input = np.array([[0.5], [1]])
        self.axes2_input = self.axes_2.quiver([0], [0], [self.input[0, 0]], [self.input[1, 0]], units="xy", scale=1, color="red")
        self.axes2_v, self.v = None, None
        self.axes2_line, = self.axes_2.plot([], color="black", linewidth=0.5)
        self.axes_2.plot([0] * 6, np.linspace(-1.1, 1.1, 6), color="black", linestyle="--", linewidth=0.2)
        self.axes_2.plot(np.linspace(-1.1, 1.1, 6), [0] * 6, color="black", linestyle="--", linewidth=0.2)
        self.compute()
        self.canvas2.draw()
        self.canvas2.mpl_connect('button_press_event', self.on_mouseclick2)

        self.button = QtWidgets.QPushButton("Update", self)
        self.button.setStyleSheet("font-size:13px")
        self.button.setGeometry(self.x_chapter_button * self.w_ratio, 600 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.button.clicked.connect(self.update)

    def slide(self):
        self.lr = self.slider_lr.value() / 10
        self.label_lr.setText("Learning rate: " + str(self.lr))
        self.compute()
        self.canvas.draw()
        self.canvas2.draw()

    def on_mouseclick1(self, event):
        if event.xdata != None and event.xdata != None:
            self.w = np.array([[event.xdata, event.ydata]])
            self.axes1_w.set_UVC(event.xdata, event.ydata)
            self.axes2_w.set_UVC(event.xdata, event.ydata)
            self.canvas.draw()
            self.compute()
            self.canvas2.draw()

    def on_mouseclick2(self, event):
        if event.xdata != None and event.xdata != None:
            self.input = np.array([[event.xdata], [event.ydata]])
            self.axes2_input.set_UVC(event.xdata, event.ydata)
            self.compute()
            self.canvas2.draw()

    def compute(self):
        if len(self.axes_2.collections) == 3:
            self.axes_2.collections.pop()
        self.v = self.w + self.lr * (self.input.T - self.w)
        self.axes2_line.set_data([self.w[0, 0], self.input[0, 0]], [self.w[0, 1], self.input[1, 0]])
        self.axes2_v = self.axes_2.quiver([self.w[0, 0]], [self.w[0, 1]], [self.v[0, 0] - self.w[0, 0]], [self.v[0, 1] - self.w[0, 1]], units="xy", scale=1, color="black")

    def update(self):
        self.w = self.v
        self.axes1_w.set_UVC(self.w[0, 0], self.w[0, 1])
        self.axes2_w.set_UVC(self.w[0, 0], self.w[0, 1])
        self.compute()
        self.canvas.draw()
        self.canvas2.draw()
