from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class NormAndInitEffect(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(NormAndInitEffect, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Normalization & Initialization Effects", 3,
                          "\nChoose a initialization\nscheme and whether to\nuse BatchNorm or not."
                          "\n\nThe distribution of input,\nnet input and output\nis shown on the left.\n\n",
                          PACKAGE_PATH + "Chapters/4_D/Logo_Ch_4.svg", None,
                          icon_move_left=120, description_coords=(535, 105, 450, 250))

        self.p_size = 2  # Number of elements of the input vector
        self.p_mean = np.array([[0], [0]])
        self.p_std = np.array([[4], [1]])

        self.make_plot(1, (10, 180, 500, 500))
        self.figure.set_tight_layout(True)

        self.make_slider("slider_n_examples", QtCore.Qt.Orientation.Horizontal, (1, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
                         (self.x_chapter_usual, 310, self.w_chapter_slider, 50), self.graph, "label_n_examples",
                         "Number of examples: 1000", (self.x_chapter_usual + 20, 310 - 25, self.w_chapter_slider, 50))
        self.n_examples = int(self.slider_n_examples.value() * 100)

        self.make_slider("slider_n_neurons", QtCore.Qt.Orientation.Horizontal, (1, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 4,
                         (self.x_chapter_usual, 360, self.w_chapter_slider, 50), self.graph, "label_n_neurons",
                         "Number of neurons: 1000", (self.x_chapter_usual + 20, 360 - 25, self.w_chapter_slider, 50))
        self.n_neurons = int(self.slider_n_neurons.value())

        self.combobox_input_distrib = ["Normal", "Uniform"]
        self.make_combobox(2, self.combobox_input_distrib, (10, 150, self.w_chapter_slider - 20, 50),
                           self.change_input_distrib, "label_input_distrib", "Input distribution",
                           (30, 130, self.w_chapter_slider, 50))
        self.input_distrib = 'Normal'
        self.random_seed = np.random.randint(100000000)
        np.random.seed(self.random_seed)
        self.p = np.diag(self.p_std) * np.random.normal(size=(self.p_size, self.n_examples)) + \
                 self.p_mean * np.ones((1, self.n_examples))

        self.weight_init = "Small random"
        self.combobox_weight_inits = ["Small random", "Xavier", "Nguyen - Widrow", "Kaiming"]
        self.make_combobox(3, self.combobox_weight_inits, (300, 110, self.w_chapter_slider, 50),
                           self.change_weight_init, "label_weight_init", "Weights init",
                           (320, 85, self.w_chapter_slider, 50))

        self.act = self.tansig
        self.combobox_act_functions = [self.tansig, self.poslin]
        self.make_combobox(3, ["tansig", "poslin"], (self.x_chapter_usual, 540, self.w_chapter_slider, 50),
                           self.change_act_function, "label_act_function", "Activation function",
                           (self.x_chapter_usual + 30, 515, self.w_chapter_slider, 50))

        self.batch_norm = True
        self.make_checkbox('checkbox_batch_norm', 'Use Batch Norm', (310, 150, self.w_chapter_slider - 20, 50),
                           self.select_bn, True)

        self.combobox_displayed_vars = ["Output (hist)", "Output"]
        self.make_combobox(1, self.combobox_displayed_vars, (self.x_chapter_usual, 420, self.w_chapter_slider, 50),
                           self.change_displayed_var, "label_displayed_var", "Displayed variable",
                           (self.x_chapter_usual + 30, 395, self.w_chapter_slider, 50))
        self.displayed_var = 'Output (hist)'

        self.make_slider("slider_n_dimensions", QtCore.Qt.Orientation.Horizontal, (1, self.n_neurons), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 4,
                         (self.x_chapter_usual, 480, self.w_chapter_slider, 50), self.graph, "label_n_dimensions",
                         "Number of dimensions: 1000", (self.x_chapter_usual + 20, 480 - 25, self.w_chapter_slider, 50))

        self.make_button('button_random_seed', "Random", (self.x_chapter_usual + 40, 590, 100, 55), self.change_random_seed)

        self.graph()

    def graph(self, pop_up=False):

        self.n_examples = int(self.slider_n_examples.value() * 100)
        self.label_n_examples.setText("Number of examples: {}".format(self.n_examples))
        self.n_neurons = int(self.slider_n_neurons.value())
        self.label_n_neurons.setText("Number of neurons: {}".format(self.n_neurons))

        self.change_input_distrib(self.combobox_input_distrib.index(self.input_distrib), graph=False)

        self.slider_n_dimensions.setMaximum(self.n_neurons)
        dim = int(self.slider_n_dimensions.value()) - 1
        self.label_n_dimensions.setText("Number of dimensions: {}".format(dim + 1))

        self.figure.clf()  # Clear the plot
        if self.displayed_var == 'Output':
            self.a = self.figure.add_subplot(projection='3d')
        else:
            self.a = self.figure.add_subplot(111)

        if self.displayed_var == 'Input':
            self.a.hist(self.p[dim, :], bins=25)
            self.a.set_title(f'Input histogram (dimension {dim + 1})')
            self.canvas.draw()
            return

        p = self.p.copy()
        if self.batch_norm:
            p_mean, p_std = np.mean(p, axis=1)[:, None], np.std(p, axis=1)[:, None]
            p = np.divide(p - p_mean.dot(np.ones((1, self.n_examples))), p_std.dot(np.ones((1, self.n_examples))))

        if self.displayed_var == 'Norm Input':
            self.a.hist(p[dim, :], bins=25)
            self.a.set_title(f'Normalized Input histogram (dimension {dim + 1})')
            self.canvas.draw()
            return

        p_range = np.array([np.min(p, axis=1).tolist(), np.max(p, axis=1).tolist()]).T
        if self.weight_init == "Nguyen - Widrow":
            n = np.array([-1, 1])
            # print('p_range, n', p_range, n, self.weight_init)
            w, b = self.nw_init(p_range, n)
        else:
            if self.weight_init == 'Xavier':
                w_factor = np.sqrt(2 / (self.p_size + self.n_neurons))
            elif self.weight_init == "Kaiming":
                w_factor = np.sqrt(2 / self.p_size)
            elif self.weight_init == "Small random":
                w_factor = 0.1
            w, b = w_factor * np.random.normal(size=(self.n_neurons, self.p_size)), np.zeros((self.n_neurons, 1))

        net_input = w.dot(p) + b * np.ones((1, self.n_examples))
        if self.displayed_var == 'Net Input':
            self.a.hist(net_input[dim, :], bins=25)
            self.a.set_title(f'Net Input histogram (dimension {dim + 1})')
            self.canvas.draw()
            return
        output = self.act(net_input)
        if self.displayed_var == 'Output (hist)':
            self.a.hist(output[dim, :], bins=25)
            self.a.set_title(f'Output histogram (dimension {dim + 1})')
            self.canvas.draw()
            return

        if self.displayed_var == 'Output':
            p1 = np.linspace(p_range[0, 0], p_range[0, 1], 20)
            p2 = np.linspace(p_range[1, 0], p_range[1, 1], 20)
            XX, YY = np.meshgrid(p1, p2)
            PP = np.array([XX.reshape(-1).tolist(), YY.reshape(-1).tolist()])
            AA = self.act(w.dot(PP) + b * np.ones((1, 400)))

            self.a.plot_surface(XX, YY, AA[dim].reshape(20, 20))
            self.a.set_title(f'Output (dimension {dim + 1}) wrt. input')
            self.canvas.draw()
            return

    def change_displayed_var(self, idx):
        self.displayed_var = self.combobox_displayed_vars[idx]
        self.graph()

    def change_weight_init(self, idx):
        self.weight_init = self.combobox_weight_inits[idx]
        self.graph()

    def change_act_function(self, idx):
        self.act = self.combobox_act_functions[idx]
        self.graph()

    def change_input_distrib(self, idx, graph=True):
        self.input_distrib = self.combobox_input_distrib[idx]
        np.random.seed(self.random_seed)
        if self.input_distrib == 'Uniform':
            self.p = np.diag(self.p_std) * (np.random.uniform(size=(self.p_size, self.n_examples)) - 0.5) * np.sqrt(12) +\
                self.p_mean * np.ones((1, self.n_examples))
        elif self.input_distrib == 'Normal':
            self.p = np.diag(self.p_std) * np.random.normal(size=(self.p_size, self.n_examples)) +\
                self.p_mean * np.ones((1, self.n_examples))
        if graph:
            self.graph()

    def nw_init(self, p_range, n):
        r = p_range.shape[0]

        wMag = 0.7 * self.n_neurons ** (1 / r)
        np.random.seed(self.random_seed)
        wDir = np.random.normal(size=(self.n_neurons, r))
        w = wMag * wDir

        if self.n_neurons == 1:
            b = 0
        else:
            b = wMag * np.multiply(np.linspace(1, -1, self.n_neurons), np.sign(w[:, 0]))

        x, y = 0.5 * (n[1] - n[0]), 0.5 * (n[1] + n[0])
        w *= x
        b = x * b + y

        x = np.divide(2, p_range[:, 1] - p_range[:, 0])
        y = 1 - np.multiply(p_range[:, 1], x)
        xp = x.T
        b = w.dot(y) + b
        w = np.multiply(w, np.repeat(xp.reshape(1, -1), len(w), axis=0))

        return w, b.reshape(-1, 1)

    def change_random_seed(self):
        self.random_seed = np.random.randint(100000000)
        self.change_input_distrib(self.combobox_input_distrib.index(self.input_distrib))

    def select_bn(self):
        if self.checkbox_batch_norm.checkState().value == 2 and not self.batch_norm:
            self.batch_norm = True
            self.graph()
        if self.checkbox_batch_norm.checkState().value == 0 and self.batch_norm:
            self.batch_norm = False
            self.graph()
