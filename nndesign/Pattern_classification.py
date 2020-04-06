from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


p1, p2 = np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01)
pp1, pp2 = np.meshgrid(p1, p2)


class PatternClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(PatternClassification, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("RBF Pattern Classification", 17, "Alter the network's\nparameters by dragging\nthe slide bars.",
                          PACKAGE_PATH + "Logo/Logo_Ch_17.svg", None, description_coords=(535, 90, 450, 160))

        # self.label_eq = QtWidgets.QLabel(self)
        # self.label_eq.setText("a = purelin(w2 * tansig(w1 * p + b1) + b2))")
        # self.label_eq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_eq.setGeometry(180 * self.w_ratio, 270 * self.h_ratio, (self.w_chapter_slider + 100) * self.w_ratio, 50 * self.h_ratio)

        self.make_plot(1, (120, 120, 270, 270))
        self.make_plot(2, (120, 390, 270, 270))

        self.make_slider("slider_w1_1", QtCore.Qt.Horizontal, (-4, 4), QtWidgets.QSlider.TicksBelow, 1, -1,
                         (self.x_chapter_usual, 140, self.w_chapter_slider, 50), self.graph, "label_w1_1", "W1(1,1)")
        self.make_slider("slider_w1_2", QtCore.Qt.Horizontal, (-4, 4), QtWidgets.QSlider.TicksBelow, 1, 1,
                         (self.x_chapter_usual, 200, self.w_chapter_slider, 50), self.graph, "label_w1_2", "W1(2,1)")
        self.make_slider("slider_b1_1", QtCore.Qt.Horizontal, (-2, 2), QtWidgets.QSlider.TicksBelow, 1, 2,
                         (self.x_chapter_usual, 270, self.w_chapter_slider, 50), self.graph, "label_b1_1", "b1(1)")
        self.make_slider("slider_b1_2", QtCore.Qt.Horizontal, (-2, 2), QtWidgets.QSlider.TicksBelow, 1, 2,
                         (self.x_chapter_usual, 310, self.w_chapter_slider, 50), self.graph, "label_b1_2", "b1(2)")
        self.make_slider("slider_w2_1", QtCore.Qt.Horizontal, (-2, 2), QtWidgets.QSlider.TicksBelow, 1, 2,
                         (self.x_chapter_usual, 410, self.w_chapter_slider, 50), self.graph, "label_w2_1", "W2(1,1)")
        self.make_slider("slider_w2_2", QtCore.Qt.Horizontal, (-2, 2), QtWidgets.QSlider.TicksBelow, 1, 2,
                         (self.x_chapter_usual, 480, self.w_chapter_slider, 50), self.graph, "label_w2_2", "W2(1,2)")
        self.make_slider("slider_b2", QtCore.Qt.Horizontal, (-2, 2), QtWidgets.QSlider.TicksBelow, 1, -2,
                         (self.x_chapter_usual, 550, self.w_chapter_slider, 50), self.graph, "label_b2", "b2")

        self.labelw1_12 = QtWidgets.QLabel(self)
        self.labelw1_12.setText("w1_12")
        self.labelw1_12.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.labelw1_12.setGeometry(self.x_chapter_slider_label * self.w_ratio, 590 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.sliderw1_12 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.sliderw1_12.setRange(-40, 40)
        self.sliderw1_12.setRange(-4, 4)
        self.sliderw1_12.setTickPosition(QtWidgets.QSlider.TicksBelow)
        # self.sliderw1_12.setTickInterval(10)
        # self.sliderw1_12.setValue(10)
        self.sliderw1_12.setTickInterval(1)
        self.sliderw1_12.setValue(1)

        self.widw1_12 = QtWidgets.QWidget(self)
        self.layoutw1_12 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.widw1_12.setGeometry(self.x_chapter_usual * self.w_ratio, 620 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layoutw1_12.addWidget(self.sliderw1_12)
        self.widw1_12.setLayout(self.layoutw1_12)

        self.labelw1_22 = QtWidgets.QLabel(self)
        self.labelw1_22.setText("w1_22")
        self.labelw1_22.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.labelw1_22.setGeometry(450 * self.w_ratio, 300 * self.h_ratio,
                                    self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.sliderw1_22 = QtWidgets.QSlider(QtCore.Qt.Vertical)
        # self.sliderw1_22.setRange(-40, 40)
        self.sliderw1_22.setRange(-4, 4)
        self.sliderw1_22.setTickPosition(QtWidgets.QSlider.TicksBelow)
        # self.sliderw1_22.setTickInterval(10)
        # self.sliderw1_22.setValue(-10)
        self.sliderw1_22.setTickInterval(1)
        self.sliderw1_22.setValue(-1)

        self.widw1_22 = QtWidgets.QWidget(self)
        self.layoutw1_22 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.widw1_22.setGeometry(450 * self.w_ratio, 370 * self.h_ratio,
                                  self.w_chapter_slider * self.w_ratio, 200 * self.h_ratio)
        self.layoutw1_22.addWidget(self.sliderw1_22)
        self.widw1_22.setLayout(self.layoutw1_22)

        self.sliderw1_12.valueChanged.connect(self.graph)
        self.sliderw1_22.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a = Axes3D(self.figure)
        a.clear()
        a.set_xlim(-5, 5)
        a.set_ylim(-5, 5)
        a.set_zlim(-2, 1)
        a.set_xlabel("$p1$")
        a.set_ylabel("$p2$")
        a.set_zlabel("$a$")

        weight1_1 = self.slider_w1_1.value()  # / 10
        weight1_2 = self.slider_w1_2.value()  # / 10
        bias1_1 = self.slider_b1_1.value() / 2  # / 10
        bias1_2 = self.slider_b1_2.value() / 2  # / 10
        weight2_1 = self.slider_w2_1.value()  # / 10
        weight2_2 = self.slider_w2_2.value()  # / 10
        bias2 = self.slider_b2.value() / 2  # / 10
        weight1_12 = self.sliderw1_12.value()  # / 10
        weight1_22 = self.sliderw1_22.value()  # / 10

        self.label_w1_1.setText("W1(1,1): " + str(weight1_1))
        self.label_w1_2.setText("W1(2,1): " + str(weight1_2))
        self.label_b1_1.setText("b1(1): " + str(bias1_1))
        self.label_b1_2.setText("b1(2): " + str(bias1_2))
        self.label_w2_1.setText("W2(1,1): " + str(weight2_1))
        self.label_w2_2.setText("W2(1,2): " + str(weight2_2))
        self.label_b2.setText("b2: " + str(bias2))
        self.labelw1_12.setText("w1_12: " + str(weight1_12))
        self.labelw1_22.setText("w1_22: " + str(weight1_22))

        weight_1, bias_1 = np.array([[weight1_1, weight1_2]]), np.array([[bias1_1, bias1_2]])
        weight_2, bias_2 = np.array([[weight2_1], [weight2_2]]), np.array([[bias2]])

        # a = W2(1)*exp(-((p-W1(1)).*b1(1)).^2) + W2(2)*exp(-((p-W1(2)).*b1(2)).^2) + b2
        out = weight_2[0, 0] * np.exp(-((pp1 - weight_1[0, 0]) * bias_1[0, 0]) ** 2 - ((pp2 - weight1_12) * bias_1[0, 0]) ** 2)
        out += weight_2[1, 0] * np.exp(-((pp1 - weight_1[0, 1]) * bias_1[0, 1]) ** 2 - ((pp2 - weight1_22) * bias_1[0, 0]) ** 2) + bias_2[0, 0]

        x_0_surf, y_0_surf = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
        xx_0_surf, yy_0_surf = np.meshgrid(x_0_surf, y_0_surf)
        a.plot_surface(xx_0_surf, yy_0_surf, np.zeros((100, 100)), color="gray", alpha=0.5)
        a.plot_surface(pp1, pp2, out, color="cyan")
        a.set_xticks([-5, 0, 5])
        a.set_yticks([-5, 0, 5])
        a.set_zticks([-2, -1, 0, 1])
        self.canvas.draw()

        b = self.figure2.add_subplot(1, 1, 1)
        b.clear()
        b.scatter([1, -1], [1, -1], marker="*")
        b.scatter([1, -1], [-1, 1], marker="o")
        b.set_xticks([-5, 0, 5])
        b.set_yticks([-5, 0, 5])
        out_gray = 1 * (out >= 0)
        """row_start, row_end = 0, 0
        for row_idx in range(len(out_gray)):
            if not row_start:
                if sum(out_gray[row_idx]) > 0:
                    row_start = row_idx
            else:
                if sum(out_gray[row_idx]) == 0:
                    row_end = row_idx
                    break
        if row_start and not row_end:
            row_end = len(out_gray)
        col_start, col_end = 0, 0
        for col_idx in range(len(out_gray)):
            if not col_start:
                if sum(out_gray[:, col_idx]) > 0:
                    col_start = col_idx
            else:
                if sum(out_gray[:, col_idx]) == 0:
                    col_end = col_idx
                    break
        if col_start and not col_end:
            col_end = len(out_gray)"""
        b.contourf(pp1, pp2, out_gray, cmap=plt.cm.Paired, alpha=0.6)
        self.canvas2.draw()
