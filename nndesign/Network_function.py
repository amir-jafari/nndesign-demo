from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class NetworkFunction(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(NetworkFunction, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("Network Function", 11, "Alter the network's\nparameters by dragging\nthe slide bars.\n\n"
                                                  "Choose the output transfer\nfunction f below.\n\n"
                                                  "Click on [Random] to\nset each parameter\nto a random value.",
                          PACKAGE_PATH + "Logo/Logo_Ch_11.svg", PACKAGE_PATH + "Figures/nnd11_1.svg",
                          icon_move_left=120, icon_coords=(130, 150, 500, 200))

        self.make_plot(1, (10, 390, 500, 290))

        self.comboBox1_functions = [self.purelin, self.logsig, self.tansig]
        self.comboBox1_functions_str = ["purelin", 'logsig', 'tansig']
        self.make_combobox(1, self.comboBox1_functions_str, (self.x_chapter_button - 8, 330, self.w_chapter_button + 16, 50),
                           self.change_transfer_function, "label_f", "f")
        self.func1 = self.purelin
        self.idx = 0

        self.make_button("random_button", "Random", (self.x_chapter_button, 380, self.w_chapter_button, self.h_chapter_button), self.on_random)

        """self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.purelin, self.logsig, self.tansig]
        self.comboBox1_functions_str = ["purelin", 'logsig', 'tansig']
        self.comboBox1.addItems(self.comboBox1_functions_str)
        self.comboBox1.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 640 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.func1 = self.purelin
        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 600 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 620 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)"""

        """self.label_eq = QtWidgets.QLabel(self)
        self.label_eq.setText("a = purelin(w2 * tansig(w1 * p + b1) + b2))")
        self.label_eq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_eq.setGeometry(180 * self.w_ratio, 270 * self.h_ratio, (self.w_chapter_slider + 100) * self.w_ratio, 50 * self.h_ratio)
        self.label_eq.setGeometry(180 * self.w_ratio, 350 * self.h_ratio, (self.w_chapter_slider + 100) * self.w_ratio, 50 * self.h_ratio)"""

        self.make_slider("slider_w1_1", QtCore.Qt.Horizontal, (-100, 100), QtWidgets.QSlider.TicksAbove, 10, 100,
                         (10, 115, 150, 50), self.graph, "label_w1_1", "W1(1,1):", (50, 115 - 25, 100, 50))

        """self.label_w1_1 = QtWidgets.QLabel(self)
        self.label_w1_1.setText("w1_1")
        self.label_w1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 120 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_1.setRange(-100, 100)
        self.slider_w1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_1.setTickInterval(10)
        self.slider_w1_1.setValue(100)

        self.wid_w1_1 = QtWidgets.QWidget(self)
        self.layout_w1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 140 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1_1.addWidget(self.slider_w1_1)
        self.wid_w1_1.setLayout(self.layout_w1_1)"""

        self.make_slider("slider_w1_2", QtCore.Qt.Horizontal, (-100, 100), QtWidgets.QSlider.TicksAbove, 10, 100,
                         (10, 360, 150, 50), self.graph, "label_w1_2", "W1(2,1):", (50, 360 - 25, 100, 50))

        """self.label_w1_2 = QtWidgets.QLabel(self)
        self.label_w1_2.setText("w1_2")
        self.label_w1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 170 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w1_2.setRange(-100, 100)
        self.slider_w1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w1_2.setTickInterval(10)
        self.slider_w1_2.setValue(100)

        self.wid_w1_2 = QtWidgets.QWidget(self)
        self.layout_w1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 200 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w1_2.addWidget(self.slider_w1_2)
        self.wid_w1_2.setLayout(self.layout_w1_2)"""

        self.make_slider("slider_b1_1", QtCore.Qt.Horizontal, (-100, 100), QtWidgets.QSlider.TicksAbove, 10, -100,
                         (170, 115, 150, 50), self.graph, "label_b1_1", "b1(1):", (210, 115 - 25, 100, 50))

        """self.label_b1_1 = QtWidgets.QLabel(self)
        self.label_b1_1.setText("b1_1")
        self.label_b1_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 240 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b1_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_1.setRange(-100, 100)
        self.slider_b1_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_1.setTickInterval(10)
        self.slider_b1_1.setValue(-100)

        self.wid_b1_1 = QtWidgets.QWidget(self)
        self.layout_b1_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_1.setGeometry(self.x_chapter_usual * self.w_ratio, 270 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b1_1.addWidget(self.slider_b1_1)
        self.wid_b1_1.setLayout(self.layout_b1_1)"""

        self.make_slider("slider_b1_2", QtCore.Qt.Horizontal, (-100, 100), QtWidgets.QSlider.TicksAbove, 10, 100,
                         (170, 360, 150, 50), self.graph, "label_b1_2", "b1(2):", (210, 360 - 25, 100, 50))

        """self.label_b1_2 = QtWidgets.QLabel(self)
        self.label_b1_2.setText("b1_2")
        self.label_b1_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b1_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 310 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b1_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b1_2.setRange(-100, 100)
        self.slider_b1_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b1_2.setTickInterval(10)
        self.slider_b1_2.setValue(100)

        self.wid_b1_2 = QtWidgets.QWidget(self)
        self.layout_b1_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b1_2.setGeometry(self.x_chapter_usual * self.w_ratio, 340 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b1_2.addWidget(self.slider_b1_2)
        self.wid_b1_2.setLayout(self.layout_b1_2)"""

        self.make_slider("slider_w2_1", QtCore.Qt.Horizontal, (-20, 20), QtWidgets.QSlider.TicksAbove, 1, 10,
                         (330, 115, 150, 50), self.graph, "label_w2_1", "W2(1,1):", (370, 115 - 25, 100, 50))

        """self.label_w2_1 = QtWidgets.QLabel(self)
        self.label_w2_1.setText("w2_1")
        self.label_w2_1.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_1.setGeometry(self.x_chapter_slider_label * self.w_ratio, 380 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w2_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_1.setRange(-20, 20)
        self.slider_w2_1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_1.setTickInterval(1)
        self.slider_w2_1.setValue(10)

        self.wid_w2_1 = QtWidgets.QWidget(self)
        self.layout_w2_1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_1.setGeometry(self.x_chapter_usual * self.w_ratio, 410 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w2_1.addWidget(self.slider_w2_1)
        self.wid_w2_1.setLayout(self.layout_w2_1)"""

        self.make_slider("slider_w2_2", QtCore.Qt.Horizontal, (-20, 20), QtWidgets.QSlider.TicksAbove, 1, 10,
                         (330, 360, 150, 50), self.graph, "label_w2_2", "W2(1,2):", (370, 360 - 25, 100, 50))

        """self.label_w2_2 = QtWidgets.QLabel(self)
        self.label_w2_2.setText("w2_2")
        self.label_w2_2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w2_2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 450 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_w2_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w2_2.setRange(-20, 20)
        self.slider_w2_2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w2_2.setTickInterval(1)
        self.slider_w2_2.setValue(10)

        self.wid_w2_2 = QtWidgets.QWidget(self)
        self.layout_w2_2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_w2_2.setGeometry(self.x_chapter_usual * self.w_ratio, 480 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_w2_2.addWidget(self.slider_w2_2)
        self.wid_w2_2.setLayout(self.layout_w2_2)"""

        self.make_slider("slider_b2", QtCore.Qt.Horizontal, (-20, 20), QtWidgets.QSlider.TicksAbove, 1, 0,
                         (360, 290, 150, 50), self.graph, "label_b2", "b2:", (400, 290 - 25, 100, 50))

        """self.label_b2 = QtWidgets.QLabel(self)
        self.label_b2.setText("b1_2")
        self.label_b2.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b2.setGeometry(self.x_chapter_slider_label * self.w_ratio, 520 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.slider_b2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b2.setRange(-20, 20)
        self.slider_b2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b2.setTickInterval(1)
        self.slider_b2.setValue(0)

        self.wid_b2 = QtWidgets.QWidget(self)
        self.layout_b2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_b2.setGeometry(self.x_chapter_usual * self.w_ratio, 550 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_b2.addWidget(self.slider_b2)
        self.wid_b2.setLayout(self.layout_b2)"""

        """self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.slider_w1_1.valueChanged.connect(self.graph)
        self.slider_w1_2.valueChanged.connect(self.graph)
        self.slider_b1_1.valueChanged.connect(self.graph)
        self.slider_b1_2.valueChanged.connect(self.graph)
        self.slider_w2_1.valueChanged.connect(self.graph)
        self.slider_w2_2.valueChanged.connect(self.graph)
        self.slider_b2.valueChanged.connect(self.graph)"""

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot
        a.set_xlim(-2, 2)
        a.set_ylim(-2, 2)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        # a.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.plot([0]*10, np.linspace(-2, 2, 10), color="black", linestyle="--", linewidth=0.2)
        a.plot(np.linspace(-2, 2, 10), [0]*10, color="black", linestyle="--", linewidth=0.2)
        a.set_title("a = {}(w2 * tansig(w1 * p + b1) + b2))".format(self.comboBox1_functions_str[self.idx]))
        # a.set_xlabel("$p$")
        # a.xaxis.set_label_coords(1, -0.025)
        # a.set_ylabel("$a$")
        # a.yaxis.set_label_coords(-0.025, 1)

        # ax.set_xticks(major_ticks)
        # ax.set_xticks(minor_ticks, minor=True)
        # ax.set_yticks(major_ticks)
        # ax.set_yticks(minor_ticks, minor=True)
        #
        # # And a corresponding grid
        # ax.grid(which='both')
        #
        # # Or if you want different settings for the grids:
        # ax.grid(which='minor', alpha=0.2)
        # ax.grid(which='major', alpha=0.5)

        weight1_1 = self.slider_w1_1.value() / 10
        weight1_2 = self.slider_w1_2.value() / 10
        bias1_1 = self.slider_b1_1.value() / 10
        bias1_2 = self.slider_b1_2.value() / 10
        weight2_1 = self.slider_w2_1.value() / 10
        weight2_2 = self.slider_w2_2.value() / 10
        bias2 = self.slider_b2.value() / 10

        self.label_w1_1.setText("W1(1,1): " + str(weight1_1))
        self.label_w1_2.setText("W1(2,1): " + str(weight1_2))
        self.label_b1_1.setText("b1(1): " + str(bias1_1))
        self.label_b1_2.setText("b1(2): " + str(bias1_2))
        self.label_w2_1.setText("W2(1,1): " + str(weight2_1))
        self.label_w2_2.setText("W2(1,2): " + str(weight2_2))
        self.label_b2.setText("b2: " + str(bias2))

        weight_1, bias_1 = np.array([[weight1_1, weight1_2]]), np.array([[bias1_1, bias1_2]])
        weight_2, bias_2 = np.array([[weight2_1], [weight2_2]]), np.array([[bias2]])

        p = np.arange(-4, 4, 0.01)
        func = np.vectorize(self.func1)
        out = func(np.dot(self.tansig(np.dot(p.reshape(-1, 1), weight_1) + bias_1), weight_2) + bias_2)

        a.plot(p, out.reshape(-1), markersize=3, color="red")
        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-2, 2)
        # a.set_ylim(-2, 2)
        # add grid and axes
        # a.grid(True, which='both')
        # a.axhline(y=0, color='k')
        # a.axvline(x=0, color='k')
        self.canvas.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions[idx]
        self.idx = idx
        # self.label_eq.setText("a = {}(w2 * tansig(w1 * p + b1) + b2))".format(self.comboBox1_functions_str[idx]))
        self.graph()

    def on_random(self):
        self.slider_w1_1.setValue(np.random.uniform(-100, 100))
        self.slider_w1_2.setValue(np.random.uniform(-100, 100))
        self.slider_b1_1.setValue(np.random.uniform(-100, 100))
        self.slider_b1_2.setValue(np.random.uniform(-100, 100))
        self.slider_w2_1.setValue(np.random.uniform(-20, 20))
        self.slider_w2_2.setValue(np.random.uniform(-20, 20))
        self.slider_b2.setValue(np.random.uniform(-20, 20))
        self.graph()
