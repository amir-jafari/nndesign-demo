import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from mpl_toolkits.mplot3d import Axes3D

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class NormAndInitEffects(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(NormAndInitEffects, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Normalization & Initialization Effects", 3,
                          "\nChoose a initialization\nscheme and whether to\nuse BatchNorm or not"
                          "\n\nThe distribution of input,\nnet input and output\nis shown on the left.\n\n",
                          PACKAGE_PATH + "Chapters/4_D/Logo_Ch_4.svg", None,
                          icon_move_left=120, description_coords=(535, 105, 450, 250))

        self.make_plot(1, (0, 140, 500, 500))
        # self.make_plot(2, (100, 390, 290, 290))
        self.figure.set_tight_layout(True)
        # self.figure2.set_tight_layout(True)

        self.a = self.figure.add_subplot(111)

        self.make_slider("slider_n_examples", QtCore.Qt.Horizontal, (1, 100), QtWidgets.QSlider.TicksBelow, 1, 10,
                         (self.x_chapter_usual, 360, self.w_chapter_slider, 50), self.graph, "label_n_examples",
                         "Number of examples: 1000", (self.x_chapter_usual + 20, 360 - 25, self.w_chapter_slider, 50))
        self.n_examples = int(self.slider_n_examples.value())

        self.make_slider("slider_n_neurons", QtCore.Qt.Horizontal, (1, 100), QtWidgets.QSlider.TicksBelow, 1, 4,
                         (self.x_chapter_usual, 420, self.w_chapter_slider, 50), self.graph, "label_n_neurons",
                         "Number of neurons: 1000", (self.x_chapter_usual + 20, 420 - 25, self.w_chapter_slider, 50))
        self.n_neurons = int(self.slider_n_neurons.value())

        # TODO: Make combobox ['uniform', 'normal']
        self.input_distrib = 'normal'

        # TODO: Make combobox ['xav', 'nw', 'kai', 'smr'] (Xavier, Nguyen - Widrow, Kaiming, small)
        self.weight_init = 'smr'

        # TODO: Make combobox ['tansig', 'poslin']
        self.act = self.tansig

        # TODO: Make checkbox
        self.batch_norm = True

        # TODO: Combox to choose what to display ['input', 'normalized input', 'net input', 'output hist', 'output']
        self.combobox_displayed_vars = ["Input", "Norm Input", "Net Input", "Output (hist)", "Output"]
        self.make_combobox(1, self.combobox_displayed_vars, (self.x_chapter_usual, 300, self.w_chapter_slider, 50),
                           self.change_graph, "label_displayed_var", "Displayed variable",
                           (self.x_chapter_usual + 30, 270, self.w_chapter_slider, 50))
        self.displayed_var = 'Norm Input'
        # TODO: Secondary input box to choose the specific one to show (input1, input2, net input1, etc.)
        self.make_input_box("dimension", "1", (self.x_chapter_usual + 50, 480, 55, 55), self.update_dim)
        # TODO: Make label for this input box
        self.make_label('label_input_box', 'Dimension', (self.x_chapter_usual + 50, 450, 100, 55))

        self.p_size = 2  # Number of elements of the input vector
        self.p_mean = np.array([[0], [0]])
        self.p_std = np.array([[4], [1]])
        # TODO: Set random seed?

        # TODO: Add option to open the graphs on a new tab

        self.graph()

    def graph(self):

        self.n_examples = int(self.slider_n_examples.value() * 100)
        self.label_n_examples.setText("Number of examples: {}".format(self.n_examples))
        self.n_neurons = int(self.slider_n_neurons.value())
        self.label_n_neurons.setText("Number of neurons: {}".format(self.n_neurons))

        dim = int(self.dimension.text()) - 1
        if dim <= -1:
            print("Please select a dimension greater than 0")
            return
        if self.displayed_var in ['Input', "Norm Input"] and dim >= self.p_size:
            print('Please select a dimension less or equal than the number of inputs')
            return
        if self.displayed_var in ["Net Input", "Output (hist)", "Output"] and dim >= self.n_neurons:
            print('Please select a dimension less or equal than the number of neurons')
            return

        self.a.clear()  # Clear the plot

        if self.input_distrib == 'uniform':
            p = np.diag(self.p_std) * (np.random.uniform(size=(self.p_size, self.n_examples)) - 0.5) * np.sqrt(12) +\
                self.p_mean * np.ones((1, self.n_examples))
        elif self.input_distrib == 'normal':
            p = np.diag(self.p_std) * np.random.normal(size=(self.p_size, self.n_examples)) +\
                self.p_mean * np.ones((1, self.n_examples))

        if self.displayed_var == 'Input':
            self.a.hist(p[dim, :], bins=25)
            self.canvas.draw()
            return

        if self.batch_norm:
            p_mean, p_std = np.mean(p, axis=1)[:, None], np.std(p, axis=1)[:, None]
            p = np.divide(p - p_mean.dot(np.ones((1, self.n_examples))), p_std.dot(np.ones((1, self.n_examples))))

        if self.displayed_var == 'Norm Input':
            self.a.hist(p[dim, :], bins=25)
            self.canvas.draw()
            return

        p_range = np.array([np.min(p, axis=1).tolist(), np.max(p, axis=1).tolist()]).T
        if self.weight_init == 'nw':
            n = np.array([-1, 1])
            w, b = self.nw_init(p_range, n)
        else:
            if self.weight_init == 'xav':
                w_factor = np.sqrt(2 / (self.p_size + self.n_neurons))
            elif self.weight_init == 'kai':
                w_factor = np.sqrt(2 / self.p_size)
            elif self.weight_init == 'smr':
                w_factor = 0.1
            w, b = w_factor * np.random.normal((self.n_neurons, self.p_size)), np.zeros((self.n_neurons, 1))

        net_input = w.dot(p) + b * np.ones((1, self.n_examples))
        if self.displayed_var == 'Net Input':
            self.a.hist(net_input[dim, :], bins=25)
            self.canvas.draw()
            return
        output = self.act(net_input)
        if self.displayed_var == 'Output (hist)':
            self.a.hist(output[dim, :], bins=25)
            self.canvas.draw()
            return

        if self.displayed_var == 'Output':
            raise Exception("TODO, 3D plot, plotfcn3d in MATLAB")
            return

    def change_graph(self, idx):
        self.displayed_var = self.combobox_displayed_vars[idx]
        self.graph()

    def update_dim(self):
        if self.dimension.text() == '':
            return
        try:
            int(self.dimension.text())
        except:
            print('Please enter a integer')
            return
        self.graph()

    def nw_init(self, p_range, n):
        r = p_range.shape[0]

        wMag = 0.7 * self.n_neurons ** (1 / r)
        raise Exception("Check wDir, could not find what function it's actually doing...")
        wDir = np.random.normal((self.n_neurons, r))
        w = wMag * wDir

        raise Exception("Check all of this math!!")

        if self.n_neurons == 1:
            b = 0
        else:
            b = wMag * np.linspace(1, -1, self.n_neurons).T * np.sign(w[:, 0])

        x, y = 0.5 * (n[1] - n[0]), 0.5 * (n[1] + n[0])
        w *= x
        b = x * b + y

        x = 2 / (p_range[:, 1] - p_range[:, 0])
        y = 1 - p_range[:, 1] * x
        xp = x.T
        b = w * y + b
        w = w * xp[np.ones((1, self.n_neurons)), :]

        return w, b
