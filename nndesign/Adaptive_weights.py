from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


t = np.arange(0, 2, 0.1)


class AdaptiveWeights(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(AdaptiveWeights, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(25, 90, 450, 450))

        self.fill_chapter("Grossberg Layer 1", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(-0.1, 2.1)
        self.axis.set_ylim(-0.1, 1.1)
        for i in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
            self.axis.plot([i] * 10, np.linspace(-0.1, 1.1, 10), color="black", linestyle="dashed", linewidth=0.2)
        self.axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Weights W2")
        self.axis.set_title("Learning")
        self.lines1, self.lines2, self.lines3, self.lines4 = [], [], [], []

        self.n_11 = QtWidgets.QLineEdit()
        self.n_11.setText("0.9")
        self.n_11.setGeometry(50 * self.w_ratio, 530 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 530 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.n_11)
        self.wid3.setLayout(self.layout3)

        self.n_12 = QtWidgets.QLineEdit()
        self.n_12.setText("0.45")
        self.n_12.setGeometry(50 * self.w_ratio, 600 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(50 * self.w_ratio, 600 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.n_12)
        self.wid3.setLayout(self.layout3)

        self.n_21 = QtWidgets.QLineEdit()
        self.n_21.setText("0.45")
        self.n_21.setGeometry(300 * self.w_ratio, 530 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(300 * self.w_ratio, 530 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.n_21)
        self.wid3.setLayout(self.layout3)

        self.n_22 = QtWidgets.QLineEdit()
        self.n_22.setText("0.9")
        self.n_22.setGeometry(300 * self.w_ratio, 600 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(300 * self.w_ratio, 600 * self.h_ratio, 75 * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.n_22)
        self.wid3.setLayout(self.layout3)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions_str = ['Instar', 'Hebb']
        self.comboBox1.addItems(self.comboBox1_functions_str)
        self.rule = 1
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("Learning rule")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12))
        self.label_f.setGeometry(self.x_chapter_slider_label * self.w_ratio, 480 * self.h_ratio,
                                 150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)
        self.comboBox1.currentIndexChanged.connect(self.change_learning_rule)

        self.clear_button = QtWidgets.QPushButton("Clear", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 580 * self.h_ratio,
                                      self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.run_button = QtWidgets.QPushButton("Update", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.graph)

        self.graph()

    def change_learning_rule(self, idx):
        self.rule = idx + 1
        self.graph()

    def adapt(self, t, w):
        if np.fix(t / 0.2) % 2 == 0:
            n1 = self.n1
            n2 = np.array([[1], [0]])
        else:
            n1 = self.n2
            n2 = np.array([[0], [1]])
        w = w.reshape((2, 2))
        if self.rule == 1:
            wprime = (4 * np.dot(n2, np.ones((1, 2)))) * (np.dot(np.ones((2, 1)), n1.T) - w)
        else:
            wprime = 4 * np.dot(n2, n1.T) - 2 * w
        return wprime.reshape(-1)

    def graph(self):
        n11, n12 = float(self.n_11.text()), float(self.n_12.text())
        n21, n22 = float(self.n_21.text()), float(self.n_22.text())
        self.n1, self.n2 = np.array([[n11], [n12]]), np.array([[n21], [n22]])
        r1 = ode(self.adapt).set_integrator("zvode")
        r1.set_initial_value(np.zeros((4,)), 0)
        t1 = 2
        dt = 0.1
        out_11, out_21, out_12, out_22 = [], [], [], []
        while r1.successful() and r1.t < t1:
            out = r1.integrate(r1.t + dt)
            out_11.append(out[0])
            out_12.append(out[1])
            out_21.append(out[2])
            out_22.append(out[3])
        while len(self.lines1) > 1:
            self.lines1.pop(0).remove()
        while len(self.lines2) > 1:
            self.lines2.pop(0).remove()
        while len(self.lines3) > 1:
            self.lines3.pop(0).remove()
        while len(self.lines4) > 1:
            self.lines4.pop(0).remove()
        for line in self.lines1:
            line.set_alpha(0.2)
        for line in self.lines2:
            line.set_alpha(0.2)
        for line in self.lines3:
            line.set_alpha(0.2)
        for line in self.lines4:
            line.set_alpha(0.2)
        self.lines1.append(self.axis.plot(t, out_11, color="red")[0])
        self.lines2.append(self.axis.plot(t, out_12, color="red", linestyle="dashed")[0])
        self.lines3.append(self.axis.plot(t, out_21, color="green")[0])
        self.lines4.append(self.axis.plot(t, out_22, color="green", linestyle="dashed")[0])
        self.canvas.draw()

    def on_clear(self):
        while len(self.lines1) > 1:
            self.lines1.pop(0).remove()
        while len(self.lines2) > 1:
            self.lines2.pop(0).remove()
        while len(self.lines3) > 1:
            self.lines3.pop(0).remove()
        while len(self.lines4) > 1:
            self.lines4.pop(0).remove()
        self.canvas.draw()
