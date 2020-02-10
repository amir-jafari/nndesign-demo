from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

from nndesign_layout import NNDLayout
from get_package_path import PACKAGE_PATH


class PerceptronClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(PerceptronClassification, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Perceptron Classification", 3, " Alter the weight and bias\n and input by dragging the\n triangular"
                                                 " shaped indictors.\n \n Pick the transfer function\n with the F menu.\n "
                                                 "\n Watch the change\n to the  neuron function\n and its  output.",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg")

        self.p, self.a, self.label = None, None, None

        self.label_w = QtWidgets.QLabel(self)
        self.label_w.setText("")
        self.label_w.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w.setGeometry(300 * self.w_ratio, 300 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(300 * self.w_ratio, 330 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_p = QtWidgets.QLabel(self)
        self.label_p.setText("")
        self.label_p.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_p.setGeometry(300 * self.w_ratio, 360 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_1 = QtWidgets.QLabel(self)
        self.label_a_1.setText("")
        self.label_a_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_1.setGeometry(300 * self.w_ratio, 390 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_2 = QtWidgets.QLabel(self)
        self.label_a_2.setText("")
        self.label_a_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_2.setGeometry(300 * self.w_ratio, 420 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_a_3 = QtWidgets.QLabel(self)
        self.label_a_3.setText("")
        self.label_a_3.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_a_3.setGeometry(300 * self.w_ratio, 450 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.label_fruit = QtWidgets.QLabel(self)
        self.label_fruit.setText("")
        self.label_fruit.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_fruit.setGeometry(300 * self.w_ratio, 480 * self.h_ratio, 150 * self.w_ratio, 25 * self.h_ratio)

        self.run_button = QtWidgets.QPushButton("Go", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 420 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

    def on_run(self):
        self.timer = QtCore.QTimer()
        self.idx = 0
        self.timer.timeout.connect(self.update_label)
        self.timer.start(1000)

    def update_label(self):
        if self.idx == 0:
            self.label_w.setText(""); self.label_b.setText(""); self.label_p.setText("")
            self.label_a_1.setText(""); self.label_a_2.setText(""); self.label_a_3.setText("")
            self.label_fruit.setText("")
            self.p = np.round(np.random.uniform(-1, 1, (1, 3)), 2)
            self.a = 0 * self.p[0, 0] + 1 * self.p[0, 1] + 0 * self.p[0, 2]
            self.label = 1 if self.a >= 0 else -1
        if self.idx == 1:
            self.label_w.setText("w = [0 1 0]")
        elif self.idx == 2:
            self.label_b.setText("b = 0")
        elif self.idx == 3:
            self.label_p.setText("p = [{} {} {}]".format(self.p[0, 0], self.p[0, 1], self.p[0, 2]))
        elif self.idx == 4:
            self.label_a_1.setText("a = hardlims(W * p + b)")
        elif self.idx == 5:
            self.label_a_2.setText("a = hardlims({})".format(self.a))
        elif self.idx == 6:
            self.label_a_3.setText(str(self.label))
        elif self.idx == 7:
            self.label_fruit.setText("Fruit = {}".format("Apple" if self.label == 1 else "Orange"))
        else:
            pass
        self.idx += 1
