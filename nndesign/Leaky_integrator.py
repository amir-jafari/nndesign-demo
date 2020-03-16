from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


t = np.arange(0, 5.1, 0.1)


class LeakyIntegrator(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(LeakyIntegrator, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot_coords=(25, 150, 450, 450))

        self.fill_chapter("Leaky Integrator", 2, " TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 5)
        self.axis.set_ylim(0, 10)
        self.axis.plot([0] * 30, np.linspace(0, 30, 30), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot(np.linspace(0, 10, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Output n")
        self.axis.set_title("Response")
        self.lines = []

        self.label_input = QtWidgets.QLabel(self)
        self.label_input.setText("Input p: 1.00")
        self.label_input.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_input.setGeometry(self.x_chapter_slider_label * self.w_ratio, 400 * self.h_ratio,
                                     150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_input = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_input.setRange(0, 100)
        self.slider_input.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_input.setTickInterval(1)
        self.slider_input.setValue(10)

        self.label_tcte = QtWidgets.QLabel(self)
        self.label_tcte.setText("Time Constant: 1.00")
        self.label_tcte.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_tcte.setGeometry(self.x_chapter_slider_label * self.w_ratio, 470 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_tcte = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tcte.setRange(0, 50)
        self.slider_tcte.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_tcte.setTickInterval(1)
        self.slider_tcte.setValue(10)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_input)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_tcte)
        self.wid4.setLayout(self.layout4)

        self.clear_button = QtWidgets.QPushButton("Clear", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 580 * self.h_ratio,
                                      self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

        self.random_button = QtWidgets.QPushButton("Random", self)
        self.random_button.setStyleSheet("font-size:13px")
        self.random_button.setGeometry(self.x_chapter_button * self.w_ratio, 630 * self.h_ratio,
                                       self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.random_button.clicked.connect(self.on_random)

        self.slider_input.valueChanged.connect(self.graph)
        self.slider_tcte.valueChanged.connect(self.graph)
        self.do_graph = True

        self.graph()

    def graph(self):
        if self.do_graph:
            p = self.slider_input.value() / 10
            tcte = self.slider_tcte.value() / 10
            self.label_input.setText("Input p: " + str(round(p, 2)))
            self.label_tcte.setText("Tme Constant: " + str(round(tcte, 2)))
            y = p * (1 - np.exp(-t / tcte))
            while len(self.lines) > 3:
                self.lines.pop(0).remove()
            for line in self.lines:
                line.set_color("gray")
                line.set_alpha(0.5)
            self.lines.append(self.axis.plot(t, y, color="red")[0])
            self.canvas.draw()

    def on_clear(self):
        while len(self.lines) > 1:
            self.lines.pop(0).remove()
        self.canvas.draw()

    def on_random(self):
        self.do_graph = False
        self.slider_input.setValue(np.random.uniform(0, 1) * 100)
        self.do_graph = True
        self.slider_tcte.setValue(np.random.uniform(0, 1) * 50)
