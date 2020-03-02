from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class CompetitiveLearning(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(CompetitiveLearning, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=False)

        self.fill_chapter("Gram-Schmidt", 5, "",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.alpha = 0.4

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.wid1 = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid1.setGeometry(15 * self.w_ratio, 100 * self.h_ratio, 500 * self.w_ratio, 500 * self.h_ratio)
        self.layout1.addWidget(self.canvas)
        self.wid1.setLayout(self.layout1)
        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        # self.axes_1.add_patch(plt.Circle((0, 0), 1.7, color='r'))
        self.axes_1.set_yticks([0])
        self.axes_1.set_xticks([0])
        self.axes_1.set_xlim(-1.2, 1.2)
        self.axes_1.set_ylim(-1.2, 1.2)
        self.axes_1.grid(True, linestyle='--')
        self.ani = None

        angles = np.array([[105, 110, 115, 120, -10, -5, 0, 5, -120, -115, -110, -105]]) * np.pi / 180
        self.p_x = list(np.cos(angles).reshape(-1))
        self.p_y = list(np.sin(angles).reshape(-1))
        self.p_points_1, = self.axes_1.plot([], ".", color="r")
        self.p_points_2, = self.axes_1.plot([], ".", color="g")
        self.p_points_3, = self.axes_1.plot([], ".", color="black")
        self.p_point_higlight, = self.axes_1.plot([], "*", color="blue")
        self.P = None

        self.W_1 = [np.sqrt(0.5), np.sqrt(0.5)]
        self.W_2 = [np.sqrt(0.5), -np.sqrt(0.5)]
        self.W_3 = [-1, 0]
        self.W = None
        self.axes1_points = []
        self.axes1_w1 = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1, color="r")
        self.axes1_w2 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, color="g")
        self.axes1_w3 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, color="black")
        self.axes1_proj_line, = self.axes_1.plot([], "-")
        self.update_plot()
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        self.cid, self.w_change = None, None

        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("Learning rate: 0.4")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry(self.x_chapter_slider_label * self.w_ratio, 400, self.w_chapter_slider * self.w_ratio,
                                  50 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 10)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(1)
        self.slider_lr.setValue(4)
        self.wid_lr = QtWidgets.QWidget(self)
        self.layout_lr = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid_lr.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio,
                                self.w_chapter_slider * self.w_ratio, 50 * self.h_ratio)
        self.layout_lr.addWidget(self.slider_lr)
        self.wid_lr.setLayout(self.layout_lr)
        self.alpha = float(self.slider_lr.value() / 10)
        self.slider_lr.valueChanged.connect(self.slide)

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 500 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)

        self.run_button = QtWidgets.QPushButton("Random", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 530 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.init_weights)

    def slide(self):
        if self.ani:
            self.ani.event_source.stop()
        self.alpha = float(self.slider_lr.value() / 10)
        self.label_lr.setText("Learning_rate: {}".format(self.alpha))
        self.update_plot()
        self.canvas.draw()
        # self.on_run()

    def init_weights(self):
        if self.ani:
            self.ani.event_source.stop()
        self.W_1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        self.W_2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        self.W_3 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        self.update_plot()
        self.canvas.draw()

    def update_plot(self):
        self.axes1_w1.set_UVC(self.W_1[0], self.W_1[1])
        self.axes1_w2.set_UVC(self.W_2[0], self.W_2[1])
        self.axes1_w3.set_UVC(self.W_3[0], self.W_3[1])
        self.W = np.array([self.W_1, self.W_2, self.W_3])
        self.P = np.array([self.p_x, self.p_y])
        a = self.compet(np.dot(self.W, self.P), axis=0)
        x_1_data, y_1_data, x_2_data, y_2_data, x_3_data, y_3_data = [], [], [], [], [], []
        for i in range(a.shape[1]):
            if np.argmax(a[:, i]) == 0:
                x_1_data.append(self.p_x[i])
                y_1_data.append(self.p_y[i])
            elif np.argmax(a[:, i]) == 1:
                x_2_data.append(self.p_x[i])
                y_2_data.append(self.p_y[i])
            else:
                x_3_data.append(self.p_x[i])
                y_3_data.append(self.p_y[i])
        self.p_points_1.set_data(x_1_data[:], y_1_data[:])
        self.p_points_2.set_data(x_2_data[:], y_2_data[:])
        self.p_points_3.set_data(x_3_data[:], y_3_data[:])

    def animate_init_train(self):
        np.random.shuffle(self.P)
        return self.axes1_w1, self.axes1_w2, self.axes1_w3, self.p_points_1, self.p_points_2, self.p_points_3, self.p_point_higlight, self.axes1_proj_line

    def on_animate_train(self, idx):
        if idx % 2 != 0:
            self.p_point_higlight.set_data([], [])
            self.axes1_proj_line.set_data([], [])
            self.update_plot()
        else:
            idx = int(idx / 2)
            p = self.P[:, idx]
            a = self.compet(np.dot(self.W, p[..., None]), axis=0)
            self.p_point_higlight.set_data([self.P[0, idx]], [self.P[1, idx]])
            if np.argmax(a) == 0:
                W_1 = self.W_1[:]
                self.W_1 = list((1 - self.alpha) * np.array(self.W_1) + self.alpha * p)
                self.axes1_proj_line.set_data([W_1[0], self.W_1[0]], [W_1[1], self.W_1[1]])
            elif np.argmax(a) == 1:
                W_2 = self.W_2[:]
                self.W_2 = list((1 - self.alpha) * np.array(self.W_2) + self.alpha * p)
                self.axes1_proj_line.set_data([W_2[0], self.W_2[0]], [W_2[1], self.W_2[1]])
            else:
                W_3 = self.W_3[:]
                self.W_3 = list((1 - self.alpha) * np.array(self.W_3) + self.alpha * p)
                self.axes1_proj_line.set_data([W_3[0], self.W_3[0]], [W_3[1], self.W_3[1]])
        return self.axes1_w1, self.axes1_w2, self.axes1_w3, self.p_points_1, self.p_points_2, self.p_points_3, self.p_point_higlight, self.axes1_proj_line

    def on_run(self):
        if self.ani:
            self.ani.event_source.stop()
        self.ani = FuncAnimation(self.figure, self.on_animate_train, init_func=self.animate_init_train, frames=2 * self.P.shape[1],
                                 interval=500, repeat=False, blit=False)
        self.update_plot()
        self.canvas.draw()

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:

            if self.ani:
                self.ani.event_source.stop()

            # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points/39840218
            d_w_1 = np.linalg.norm(np.cross(
                np.array([self.W_1[0], self.W_1[1]]) - np.array([0, 0]),
                np.array([0, 0]) - np.array([event.xdata, event.ydata])
            )) / np.linalg.norm(np.array([self.W_1[0], self.W_1[1]]) - np.array([0, 0]))
            d_w_2 = np.linalg.norm(np.cross(
                np.array([self.W_2[0], self.W_2[1]]) - np.array([0, 0]),
                np.array([0, 0]) - np.array([event.xdata, event.ydata])
            )) / np.linalg.norm(np.array([self.W_2[0], self.W_2[1]]) - np.array([0, 0]))
            d_w_3 = np.linalg.norm(np.cross(
                np.array([self.W_3[0], self.W_3[1]]) - np.array([0, 0]),
                np.array([0, 0]) - np.array([event.xdata, event.ydata])
            )) / np.linalg.norm(np.array([self.W_3[0], self.W_3[1]]) - np.array([0, 0]))
            min_d_idx, min_d = np.argmin([d_w_1, d_w_2, d_w_3]), np.min([d_w_1, d_w_2, d_w_3])

            if min_d < 0.03:
                self.w_change = min_d_idx + 1
                if self.w_change == 1:
                    self.axes1_w1.set_UVC(0, 0)
                elif self.w_change == 2:
                    self.axes1_w2.set_UVC(0, 0)
                elif self.w_change == 3:
                    self.axes1_w3.set_UVC(0, 0)
                self.canvas.draw()
                self.cid = self.canvas.mpl_connect("button_release_event", self.on_mousepressed)
            else:
                if self.cid:
                    self.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.p_x.append(event.xdata)
                self.p_y.append(event.ydata)
                self.update_plot()
                self.canvas.draw()

    def on_mousepressed(self, event):
        if self.w_change == 1:
            self.W_1 = [event.xdata, event.ydata]
        elif self.w_change == 2:
            self.W_2 = [event.xdata, event.ydata]
        elif self.w_change == 3:
            self.W_3 = [event.xdata, event.ydata]
        self.update_plot()
        self.canvas.draw()
