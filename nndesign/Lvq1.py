from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance


from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


W2 = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])


class LVQ1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(LVQ1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=False)

        self.fill_chapter("LVQ1", 5, "",
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
        self.axes_1.set_yticks([])
        self.axes_1.set_xticks([])
        self.axes_1.set_xlim(-3, 3)
        self.axes_1.set_ylim(-3, 3)
        self.axes_1.grid(True, linestyle='--')
        self.ani = None

        self.p_point_higlight, = self.axes_1.plot([], "*")
        self.P = np.array([[-1.5, -2.0, 2.0, 1.5, 2.0, 2.0, -2.0, -1.5],
                           [2.0, 1.5, 2.0, 2.0, -2.0, -1.5, -2.0, -2.0]])
        self.T = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1]])
        self.pos_line, = self.axes_1.plot([], 'mo', label="Positive Class")
        self.neg_line, = self.axes_1.plot([], 'cs', label="Negative Class")
        self.miss_line_pos, = self.axes_1.plot([], 'ro')
        self.miss_line_neg, = self.axes_1.plot([], 'rs')
        self.W = None
        self.pos_weight, = self.axes_1.plot([], 'm+', label="Positive Weight", markersize=16)
        self.neg_weight, = self.axes_1.plot([], 'c+', label="Negative Weight", markersize=16)
        # self.axes_1.legend(loc='lower center', fontsize=8, framealpha=0.9, numpoints=2, ncol=2,
        #                    bbox_to_anchor=(0, -.28, 1, -.280), mode='expand')
        # self.axes_1.legend(loc="lower center", ncol=4, bbox_to_anchor=(-3, -.28, 6, -.280))
        self.axes_1.legend()
        self.init_weights()
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

        self.warning_label = QtWidgets.QLabel(self)
        self.warning_label.setText("")
        self.warning_label.setGeometry((self.x_chapter_usual + 10) * self.w_ratio, 610 * self.h_ratio,
                                       300 * self.w_ratio, 100 * self.h_ratio)

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

        self.undo_click_button = QtWidgets.QPushButton("Undo Last Mouse Click", self)
        self.undo_click_button.setStyleSheet("font-size:13px")
        self.undo_click_button.setGeometry(self.x_chapter_button * self.w_ratio, 560 * self.h_ratio,
                                           self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.undo_click_button.clicked.connect(self.on_undo_mouseclick)

        self.clear_button = QtWidgets.QPushButton("Clear Data", self)
        self.clear_button.setStyleSheet("font-size:13px")
        self.clear_button.setGeometry(self.x_chapter_button * self.w_ratio, 590 * self.h_ratio,
                                      self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.clear_button.clicked.connect(self.on_clear)

    def update_plot(self, omit_idx=None):

        if len(self.P) == 0:
            self.pos_line.set_data([], [])
            self.neg_line.set_data([], [])
            self.miss_line_pos.set_data([], [])
            self.miss_line_neg.set_data([], [])
            return

        pos_points_x, neg_points_x, pos_points_y, neg_points_y = [], [], [], []
        for i in range(self.P.shape[1]):
            if omit_idx is not None and i == omit_idx:
                continue
            if self.T[0, i] == 0:
                pos_points_x.append(self.P[0, i])
                pos_points_y.append(self.P[1, i])
            else:
                neg_points_x.append(self.P[0, i])
                neg_points_y.append(self.P[1, i])
        self.pos_line.set_data(pos_points_x, pos_points_y)
        self.neg_line.set_data(neg_points_x, neg_points_y)

        pos_weights_x, neg_weights_x, pos_weights_y, neg_weights_y = [], [], [], []
        for i in range(len(self.W)):
            if i == 0 or i == 1:
                pos_weights_x.append(self.W[i, 0])
                pos_weights_y.append(self.W[i, 1])
            else:
                neg_weights_x.append(self.W[i, 0])
                neg_weights_y.append(self.W[i, 1])
        self.pos_weight.set_data(pos_weights_x, pos_weights_y)
        self.neg_weight.set_data(neg_weights_x, neg_weights_y)

        dist_v = np.zeros((len(self.W), self.P.shape[1]))
        for i in range(len(self.W)):
            for j in range(self.P.shape[1]):
                dist_v[i, j] = distance.euclidean(self.W[i].reshape(-1), self.P[:, j].reshape(-1))
        a1 = self.compet(-dist_v, axis=0)
        a2 = np.dot(W2, a1)
        e = self.T - a2
        miss_pos_points_x, miss_pos_points_y, miss_neg_points_x, miss_neg_points_y = [], [], [], []
        for i in range(self.P.shape[1]):
            if omit_idx is not None and i == omit_idx:
                continue
            if e[0, i] == 0:
                if self.T[0, i] == 0:
                    miss_pos_points_x.append(self.P[0, i])
                    miss_pos_points_y.append(self.P[1, i])
                else:
                    miss_neg_points_x.append(self.P[0, i])
                    miss_neg_points_y.append(self.P[1, i])
        self.miss_line_pos.set_data(miss_pos_points_x, miss_pos_points_y)
        self.miss_line_neg.set_data(miss_neg_points_x, miss_neg_points_y)

    def slide(self):
        if self.ani:
            self.ani.event_source.stop()
        if self.axes_1.collections:
            self.axes_1.collections.pop()
        self.alpha = float(self.slider_lr.value() / 10)
        self.label_lr.setText("Learning_rate: {}".format(self.alpha))
        self.update_plot()
        self.canvas.draw()
        # self.on_run()

    def init_weights(self):
        if self.ani:
            self.ani.event_source.stop()
        self.W = 2 * np.random.uniform(0, 1, (4, 2)) - 1
        self.update_plot()
        self.canvas.draw()

    def animate_init_train(self):
        if self.axes_1.collections:
            self.axes_1.collections.pop()
        return self.pos_line, self.neg_line, self.pos_weight, self.neg_weight, self.miss_line_pos, self.miss_line_neg, self.p_point_higlight

    def on_animate_train(self, idx):

        if idx in list(np.arange(0, 10000, 3)):
            i = np.random.randint(0, self.P.shape[1])
            self.p = self.P[:, int(idx / 3)]
            self.t = self.T[:, int(idx / 3)]
            dist_v = np.zeros((len(self.W),))
            for i in range(len(self.W)):
                dist_v[i] = distance.euclidean(self.W[i].reshape(-1), self.p.reshape(-1))
            self.a1 = self.compet(-dist_v)
            a2 = np.dot(W2, self.a1)
            e = self.t - a2
            if sum(e == 0) > 0:
                self.color = "red"
            else:
                self.color = "green"
        elif idx in list(np.arange(1, 10000, 3)):
            e = np.dot(W2.T, self.t)
            i = np.argmax(self.a1)
            w1_new = self.W[i, :] - (e[i] * 2 - 1) * (self.alpha * self.a1[i] * (self.p.T - self.W[i, :]))
        elif idx in list(np.arange(2, 10000, 3)):
            self.p_point_higlight.set_data([], [])

        if idx in list(np.arange(0, 10000, 3)):
            self.update_plot(omit_idx=idx / 3)
            self.p_point_higlight.set_data([self.p[0]], [self.p[1]])
            self.p_point_higlight.set_color(self.color)
        elif idx in list(np.arange(1, 1000, 3)):
            self.update_plot(omit_idx=idx - 1 / 3)
            self.update_v = self.axes_1.quiver([self.W[i, :][0]], [self.W[i, :][1]], [w1_new[0] - self.W[i, :][0]], [w1_new[1] - self.W[i, :][1]], units="xy", scale=1, color=self.color)
            self.W[i, :] = np.copy(w1_new)
        elif idx in list(np.arange(2, 1000, 3)):
            self.axes_1.collections.pop()
            self.update_plot()

        return self.pos_line, self.neg_line, self.pos_weight, self.neg_weight, self.miss_line_pos, self.miss_line_neg, self.p_point_higlight

    def on_run(self):
        if self.ani:
            self.ani.event_source.stop()
        if len(self.P) > 0:
            self.ani = FuncAnimation(self.figure, self.on_animate_train, init_func=self.animate_init_train, frames=3 * self.P.shape[1],
                                     interval=1000, repeat=False, blit=False)
        else:
            self.warning_label.setText("  Please draw at least\n  one point")
        self.update_plot()
        self.canvas.draw()

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            if self.ani:
                self.ani.event_source.stop()
            if len(self.P) > 0:
                self.P = np.hstack((self.P, np.array([[event.xdata], [event.ydata]])))
                if event.button == 1:
                    self.T = np.hstack((self.T, np.array([[1], [0]])))
                else:
                    self.T = np.hstack((self.T, np.array([[0], [1]])))
            else:
                self.P = np.array([[event.xdata], [event.ydata]])
                if event.button == 1:
                    self.T = np.array([[1], [0]])
                else:
                    self.T = np.array([[0], [1]])
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

    def on_undo_mouseclick(self):
        if self.ani:
            self.ani.event_source.stop()
        if len(self.P) > 0:
            self.P = self.P[:, :-1]
            self.T = self.T[:, :-1]
            self.update_plot()
            self.canvas.draw()

    def on_clear(self):
        if self.ani:
            self.ani.event_source.stop()
        if len(self.P) > 0:
            self.P = []
            self.T = []
            self.update_plot()
            self.canvas.draw()
