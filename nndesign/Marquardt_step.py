from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from scipy.io import loadmat
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


def logsigmoid(n):
    return 1 / (1 + np.exp(-n))


def logsigmoid_der(n):
    return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))


def purelin(n):
    return n


def purelin_der(n):
    return np.array([1]).reshape(n.shape)

W1 = np.array([[10], [10]])
b1 = np.array([[-5], [5]])
W2 = np.array([[1, 1]])
b2 = np.array([-1])
P = np.arange(-2, 2.1, 0.1).reshape(1, -1)
A1 = logsigmoid(np.dot(W1, P) + b1)
T = logsigmoid(np.dot(W2, A1) + b2)


class MarquardtStep(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(MarquardtStep, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Marquardt Step", 9, " Click anywhere to start an\n initial guess. The gradient\n descent path will be shown\n"
                                                               " Modify the learning rate\n by moving the slide bar",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)  # TODO: Change icons

        self.W1, self.b1 = np.array([[10], 10]), np.array([[-5], [5]])
        self.W2, self.b2 = np.array([[1, 1]]), np.array([[-1]])

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(50 * self.w_ratio, 200 * self.h_ratio, 450 * self.w_ratio, 450 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)

        self.axes = self.figure.add_subplot(1, 1, 1)
        self.path, = self.axes.plot([], label="Gradient Descent Path", color="blue")
        self.x_data, self.y_data = [], []
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.v1 = self.axes.quiver([0], [0], [0], [0], units="xy", scale=1, color="r")
        self.v2 = self.axes.quiver([0], [0], [0], [0], units="xy", scale=1, color="k")
        self.pair_of_params = 1
        self.pair_params = [["W1(1, 1)", "W2(1, 1)"], ["W1(1, 1)", "b1(1)"], ["b1(1)", "b1(2)"]]
        self.plot_data()

        self.x, self.y = None, None

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.addItems(["W1(1, 1), W2(1, 1)", 'W1(1, 1), b1(1)', 'b1(1), b1(2)'])
        self.label_combo = QtWidgets.QLabel(self)
        self.label_combo.setText("Pair of parameters")
        self.label_combo.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_combo.setGeometry((self.x_chapter_slider_label + 10) * self.w_ratio, 550 * self.h_ratio,
                                     150 * self.w_ratio, 100 * self.h_ratio)
        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(self.x_chapter_usual * self.w_ratio, 580 * self.h_ratio,
                              self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)
        self.comboBox1.currentIndexChanged.connect(self.change_pair_of_params)

        self.mu = 0.0012
        self.nu = 1.2
        self.canvas.draw()

    def change_pair_of_params(self, idx):
        self.pair_of_params = idx + 1
        self.init_params()
        self.plot_data()
        self.canvas.draw()

    def plot_data(self):
        self.x_data = []
        self.y_data = []
        self.path.set_data(self.x_data, self.y_data)
        while self.axes.collections:
            for collection in self.axes.collections:
                collection.remove()
        f_data = loadmat("nndbp{}.mat".format(self.pair_of_params))
        x1, y1 = np.meshgrid(f_data["x1"], f_data["y1"])
        self.axes.contour(x1, y1, f_data["E1"], list(f_data["levels"].reshape(-1)))
        if self.pair_of_params == 1:
            self.axes.set_xlim(-5, 15)
            self.axes.set_ylim(-5, 15)
            self.axes.set_xticks([-5, 0, 5, 10])
            self.axes.set_yticks([-5, 0, 5, 10])
        elif self.pair_of_params == 2:
            self.axes.set_xlim(-10, 30)
            self.axes.set_ylim(-20, 10)
            self.axes.set_xticks([-10, 0, 10, 20])
            self.axes.set_yticks([-20, -15, -10, -5, 0, 5])
        elif self.pair_of_params == 3:
            self.axes.set_xlim(-10, 10)
            self.axes.set_ylim(-10, 10)
            self.axes.set_xticks([-10, -5, 0, 5])
            self.axes.set_xticks([-10, -5, 0, 5])
        self.axes.set_xlabel(self.pair_params[self.pair_of_params - 1][0], fontsize=8)
        self.axes.xaxis.set_label_coords(0.95, -0.025)
        self.axes.set_ylabel(self.pair_params[self.pair_of_params - 1][1], fontsize=8)
        self.axes.yaxis.set_label_coords(-0.025, 0.95)

    def train_step(self):

        self.mu /= self.nu
        self.a1 = np.kron(self.a1, np.ones((1, 1)))
        d2 = self.log_delta(self.a2)
        d1 = self.log_delta(self.a1, d2, self.W2)
        jac1 = self.marq(np.kron(P, np.ones((1, 1))), d1)
        jac2 = self.marq(self.a1, d2)
        jac = np.hstack((jac1, d1.T))
        jac = np.hstack((jac, jac2))
        jac = np.hstack((jac, d2.T))
        if self.pair_of_params == 1:
            jac = np.array([list(jac[:, 0]), list(jac[:, 4])]).T
        elif self.pair_of_params == 2:
            jac = np.array([list(jac[:, 0]), list(jac[:, 2])]).T
        elif self.pair_of_params == 3:
            jac = np.array([list(jac[:, 2]), list(jac[:, 3])]).T

        je = np.dot(jac.T, self.e.T)
        grad = np.sqrt(np.dot(je.T, je)).item()
        if grad < 0:
            return

        jj = np.dot(jac.T, jac)
        dw = -np.dot(np.linalg.inv(jj + self.mu * self.ii), je)
        gx, gy = dw[0].item(), dw[1].item()
        dist = np.sqrt(gx ** 2 + gy ** 2)
        self.v2 = self.axes.quiver([self.Lx], [self.Ly], [gx], [gy], units="xy", scale=1, color="black")
        Lx1, Ly1 = self.Lx + gx, self.Ly + gy

        dw = -je
        gx, gy = dw[0], dw[1]
        gx1 = dist * gx / np.sqrt(gx ** 2 + gy ** 2)
        gy1 = dist * gy / np.sqrt(gx ** 2 + gy ** 2)
        self.v1 = self.axes.quiver([self.Lx], [self.Ly], [gx1], [gy1], units="xy", scale=1, color="r")

        W1, b1, W2, b2 = np.copy(self.W1), np.copy(self.b1), np.copy(self.W2), np.copy(self.b2)
        if self.pair_of_params == 1:
            self.x, self.y = self.W1[0, 0] + dw[0], self.W2[0, 0] + dw[1]
            W1[0, 0], W2[0, 0] = self.x, self.y
        elif self.pair_of_params == 2:
            self.x, self.y = self.W1[0, 0] + dw[0], self.b1[0] + dw[1]
            W1[0, 0], b1[0] = self.x, self.y
        elif self.pair_of_params == 3:
            self.x, self.y = self.b1[0] + dw[0], self.b1[1] + dw[1]
            b1[0], b1[1] = self.x, self.y

        self.a1 = self.logsigmoid_stable(np.dot(W1, P) + b1)
        self.a2 = self.logsigmoid_stable(np.dot(W2, self.a1) + b2)
        self.e = T - self.a2
        error = np.dot(self.e, self.e.T).item()

        while abs(error - self.error_prev) > 0.001 * self.error_prev:

            try:

                self.mu *= self.nu
                if self.mu > 1e10:
                    break

                dw = -np.dot(np.linalg.inv(jj + self.mu * self.ii), je)
                W1, b1, W2, b2 = np.copy(self.W1), np.copy(self.b1), np.copy(self.W2), np.copy(self.b2)
                if self.pair_of_params == 1:
                    self.x, self.y = self.W1[0, 0] + dw[0], self.W2[0, 0] + dw[1]
                    W1[0, 0], W2[0, 0] = self.x, self.y
                elif self.pair_of_params == 2:
                    self.x, self.y = self.W1[0, 0] + dw[0], self.b1[0] + dw[1]
                    W1[0, 0], b1[0] = self.x, self.y
                elif self.pair_of_params == 3:
                    self.x, self.y = self.b1[0] + dw[0], self.b1[1] + dw[1]
                    b1[0], b1[1] = self.x, self.y

                Lx1, Ly1 = self.x, self.y
                self.x_data.append(Lx1.item())
                self.y_data.append(Ly1.item())
                self.a1 = self.logsigmoid_stable(np.dot(W1, P) + b1)
                self.a2 = self.logsigmoid_stable(np.dot(W2, self.a1) + b2)
                self.e = T - self.a2
                error = np.dot(self.e, self.e.T).item()

            except Exception as e:
                if str(e) == "Singular matrix":
                    print("The matrix was singular... Increasing mu 10-fold")
                    self.mu *= self.nu
                else:
                    raise e

        if error < self.error_prev:
            self.W1, self.b1, self.W2, self.b2 = np.copy(W1), np.copy(b1), np.copy(W2), np.copy(b2)
            self.error_prev = error

        if self.error_prev <= 0.0000001:
            return

        self.x_data.append(Lx1.item())
        self.y_data.append(Ly1.item())
        return

    def on_mouseclick(self, event):

        self.init_params()
        self.plot_data()
        self.x_data, self.y_data = [event.xdata], [event.ydata]
        self.x, self.y = event.xdata, event.ydata
        if self.pair_of_params == 1:
            self.W1[0, 0], self.W2[0, 0] = self.x, self.y
        elif self.pair_of_params == 2:
            self.W1[0, 0], self.b1[0] = self.x, self.y
        elif self.pair_of_params == 3:
            self.b1[0], self.b1[1] = self.x, self.y

        self.path.set_data(self.x_data, self.y_data)
        self.a1 = self.logsigmoid_stable(np.dot(self.W1, P) + self.b1)
        self.a2 = self.logsigmoid_stable(np.dot(self.W2, self.a1) + self.b2)
        self.e = T - self.a2
        self.error_prev = np.dot(self.e, self.e.T).item()
        self.ii = np.eye(2)
        self.Lx, self.Ly = self.x, self.y

        self.train_step()
        self.path.set_data(self.x_data[1:][::-1], self.y_data[1:][::-1])
        self.canvas.draw()

    def init_params(self):
        self.W1, self.b1 = np.array([[10.], [10.]]), np.array([[-5.], [5.]])
        self.W2, self.b2 = np.array([[1., 1.]]), np.array([[-1.]])
        self.mu = 0.0012
        self.nu = 1.2
