from PyQt5 import QtWidgets, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from scipy.signal import lfilter

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class IIRNetwork(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(IIRNetwork, self).__init__(w_ratio, h_ratio, dpi, main_menu=1)

        self.fill_chapter("IIR Network", 14, "Select the input and\nfrequency to the IIR\nnetwork.\n\n"
                                             "Use the sliders to alter\nthe network weights.",  # \n\n"
                                             # "Click on [Random] to set\neach parameter to\a random value.\n\n"
                                             # "Click on [Reset] to\ninitialize the parameters",
                          PACKAGE_PATH + "Logo/Logo_Ch_14.svg", PACKAGE_PATH + "Figures/nnd14_2.svg",
                          icon_move_left=70, icon_coords=(130, 100, 500, 200), description_coords=(535, 90, 450, 200),
                          icon_rescale=True)

        self.make_plot(1, (15, 300, 500, 370))

        self.comboBox1_functions_str = ["square", 'sine']
        self.make_combobox(1, self.comboBox1_functions_str, (self.x_chapter_usual, 495 - 55, self.w_chapter_slider, 100),
                           self.change_transfer_function,
                           "label_f", "f", (self.x_chapter_slider_label + 20, 465 - 55 + 5, 150, 100))
        self.func1 = "square"

        self.comboBox2_divs = ["1/16", '1/14', '1/12', '1/10', '1/8']
        self.make_combobox(2, self.comboBox2_divs, (self.x_chapter_usual, 590 - 55, self.w_chapter_slider, 50),
                           self.change_freq,
                           "label_div", "frequency", (self.x_chapter_slider_label, 560 - 55 + 5, 150, 50))
        self.freq = 1 / 12

        self.autoscale = False
        self.make_combobox(3, ["No", "Yes"], (self.x_chapter_usual, 430 - 50, self.w_chapter_slider, 100),
                           self.change_autoscale,
                           "label_autoscale", "Autoscale", (self.x_chapter_slider_label, 400 - 50 + 5, 150, 100))

        self.make_slider("slider_w0", QtCore.Qt.Horizontal, (-20, 20), QtWidgets.QSlider.TicksBelow, 1, 5,
                         (self.x_chapter_usual, 270, self.w_chapter_slider, 50), self.graph, "label_w0", "iW(0): 0.5")

        self.make_slider("slider_w1", QtCore.Qt.Horizontal, (-20, 20), QtWidgets.QSlider.TicksBelow, 1, -5,
                         (self.x_chapter_usual, 340, self.w_chapter_slider, 50), self.graph, "label_w1", "lW(1): -0.5")

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot
        if not self.autoscale:
            a.set_xlim(0, 25)
            a.set_ylim(-6, 6)
        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        # a.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.plot(np.linspace(0, 25, 50), [0] * 50, color="red", linestyle="--", linewidth=0.2)
        # a.plot(np.linspace(-2, 2, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        # a.set_xlabel("$p$")
        # a.xaxis.set_label_coords(1, -0.025)
        # a.set_ylabel("$a$")
        # a.yaxis.set_label_coords(-0.025, 1)

        if self.func1 == "square":
            if self.freq == 1 / 16:
                p = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
            elif self.freq == 1 / 14:
                p = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
            elif self.freq == 1 / 12:
                p = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
            elif self.freq == 1 / 10:
                p = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
            elif self.freq == 1 / 8:
                p = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
        else:
            p = np.sin(np.arange(0, 24, 1) * 2 * np.pi * self.freq)

        weight_0 = self.slider_w0.value() / 10
        weight_1 = self.slider_w1.value() / 10
        self.label_w0.setText("iW(0): " + str(weight_0))
        self.label_w1.setText("lW(1): " + str(weight_1))

        a0, a_1, t, t1 = 0, 0, list(range(1, len(p) + 1)), list(range(len(p) + 1))
        num = np.array([weight_0])
        den = np.array([1, weight_1])
        zi = np.array([a0])
        A = lfilter(num, den, p, zi=zi)

        a.scatter(t, p, color="white", marker="o", edgecolor="red")
        a.scatter(t1, [a0] + list(A[0]), color="blue", marker=".", s=[6]*len(t1))
        # Setting limits so that the point moves instead of the plot.
        # a.set_xlim(-4, 4)
        # a.set_ylim(-2, 2)
        # add grid and axes
        # a.grid(True, which='both')
        # a.axhline(y=0, color='k')
        # a.axvline(x=0, color='k')
        self.canvas.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions_str[idx]
        self.graph()

    def change_freq(self, idx):
        self.freq = eval(self.comboBox2_divs[idx])
        self.graph()

    def change_autoscale(self, idx):
        self.autoscale = True if idx == 1 else False
        self.graph()