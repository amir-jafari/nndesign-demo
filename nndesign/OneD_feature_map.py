from PyQt5 import QtWidgets, QtGui, QtCore
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.animation import FuncAnimation

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


Sx, Sy = 1, 20
S = Sx * Sy
max_dist = np.ceil(np.sqrt(np.sum(np.array([Sx, Sy]) ** 2)))
NDEC = 0.998

W = np.zeros((S, 3))
W[:, -1] = 1
Y, X = np.meshgrid(np.arange(1, Sy + 1), np.arange(1, Sx + 1))
Ind2Pos = np.array([X.reshape(-1), Y.reshape(-1)]).T
N = np.zeros((S, S))
for i in range(S):
    for j in range(i):
        N[i, j] = np.sqrt(np.sum((Ind2Pos[i, :] - Ind2Pos[j, :]) ** 2))

Nfrom, Nto = list(range(2, 21)), list(range(1, 20))
NN = len(Nfrom)
NV = np.zeros((1, NN))
for i in range(NN):
    from_ = Nfrom[i]
    to_ = Nto[i]

N = N + N.T

P = np.ones((3, 1000))
# np.random.seed(0)  # This is only for testing - comment out for production
P[:2, :] = np.random.random((1000, 2)).T - 0.5  # The transpose is done so we get the same random numbers as in MATLAB
P = np.divide(P, (np.ones((3, 1)) * np.sqrt(np.sum(P ** 2, axis=0))))

up = np.arange(-0.5, 0.5, 0.1)
down = -np.copy(up)
flat = np.zeros((1, len(up))) + 0.5
xx = np.array(list(up) + list(flat.reshape(-1)) + list(down) + list(-flat.reshape(-1)) + [up[0]])
yy = np.array(list(-flat.reshape(-1)) + list(up) + list(flat.reshape(-1)) + list(down) + [-flat[0, 0]])
zz = np.array([list(xx), list(yy)])
zz = zz / (np.ones((2, 1)) * np.sqrt(np.sum(zz ** 2, axis=0) + 1))


#
#   [Nfrom,Nto] = find(N == 1);
#   NN = length(Nfrom);
#   NV = zeros(1,NN);
#   for i=1:NN
#     from = Nfrom(i);
#     to = Nto(i);
#     NV(i) = plot([W(from,1) W(to,1)],[W(from,2) W(to,2)],...
#       'color',nnred,...
#       'CreateFcn','');
#   end
#   N = N + N';
#
#   % INPUT VECTORS
#   P = [rand(2,1000)-0.5; ones(1,1000)];
#   P = P ./ (ones(3,1)*sqrt(sum(P.^2)));


class OneDFeatureMap(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(OneDFeatureMap, self).__init__(w_ratio, h_ratio, main_menu=1)

        self.fill_chapter("One input neuron", 2, "",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axis1 = self.figure.add_subplot(1, 1, 1)
        self.axis1.set_xlim(-1, 1)
        self.axis1.set_ylim(-1, 1)
        self.axis1.plot(zz[0, :], zz[1, :])
        self.lines = []
        self.lines_anim = []
        self.canvas.draw()

        self.W = W
        self.ani = None
        self.n_runs = 0

        # self.label_eq = QtWidgets.QLabel(self)
        # self.label_eq.setText("a = purelin(w * p + b)")
        # self.label_eq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_eq.setGeometry((self.x_chapter_slider_label - 30) * self.w_ratio, 350 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)

        self.make_label("label_presentations", "Presentations: 0", (self.x_chapter_slider_label - 40, 300, 150, 100), )

        self.label_lr = QtWidgets.QLabel(self)
        self.label_lr.setText("Learning rate: 1")
        self.label_lr.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_lr.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 400 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lr.setRange(0, 100)
        self.slider_lr.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_lr.setTickInterval(10)
        self.slider_lr.setValue(100)
        self.lr = 1

        self.label_nei = QtWidgets.QLabel(self)
        self.label_nei.setText("Neighborhood: 21")
        self.label_nei.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_nei.setGeometry((self.x_chapter_slider_label - 40) * self.w_ratio, 470 * self.h_ratio, 150 * self.w_ratio, 100 * self.h_ratio)
        self.slider_nei = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_nei.setRange(0, 210)
        self.slider_nei.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_nei.setTickInterval(10)
        self.slider_nei.setValue(210)
        self.nei = 21

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(self.x_chapter_usual * self.w_ratio, 430 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout3.addWidget(self.slider_lr)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(self.x_chapter_usual * self.w_ratio, 500 * self.h_ratio, self.w_chapter_slider * self.w_ratio, 100 * self.h_ratio)
        self.layout4.addWidget(self.slider_nei)
        self.wid4.setLayout(self.layout4)

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 600 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run_2)

        self.make_button("reset_button", "Reset", (self.x_chapter_button, 630, self.w_chapter_button, self.h_chapter_button), self.on_reset)

        self.do_slide = True
        self.slider_lr.valueChanged.connect(self.slide)
        self.slider_nei.valueChanged.connect(self.slide)

    def on_reset(self):
        self.W = W
        while self.lines_anim:
            self.lines_anim.pop().remove()
        self.canvas.draw()
        self.do_slide = False
        self.lr = 1
        self.nei = 21
        self.label_lr.setText("Learning rate: " + str(self.lr))
        self.label_nei.setText("Neighborhood: " + str(self.nei))
        self.slider_lr.setValue(self.lr * 100)
        self.slider_nei.setValue(self.nei * 10)
        self.do_slide = True
        self.n_runs = 0
        self.label_presentations.setText("Presentations: 0")

    def slide(self):
        if self.do_slide:
            self.lr = self.slider_lr.value() / 100
            self.nei = self.slider_nei.value() / 10
            self.label_lr.setText("Learning rate: " + str(self.lr))
            self.label_nei.setText("Neighborhood: " + str(self.nei))

    def on_run(self):

        if self.lines:
            for line in self.lines:
                line.pop(0).remove()
            self.lines = []

        s, r = self.W.shape
        Q = P.shape[1]

        for z in range(500):

            q = int(np.fix(np.random.random() * Q))
            p = P[:, q].reshape(-1, 1)

            a = self.compet_(np.dot(self.W, p))
            i = np.argmax(a)
            N_c = np.copy(N)[:, i]
            N_c[N_c <= self.nei] = 1
            N_c[N_c != 1] = 0
            a = 0.5 * (a + N_c.reshape(-1, 1))

            self.W = self.W + self.lr * np.dot(a, np.ones((1, r))) * (np.dot(np.ones((s, 1)), p.T) - self.W)
            self.lr = (self.lr - 0.01) * 0.998 + 0.01
            self.nei = (self.nei - 1) * NDEC + 1

        for i in range(NN):
            from_ = Nfrom[i] - 1
            to_ = Nto[i] - 1
            print(self.W[from_, 0], self.W[to_, 0], "---", self.W[from_, 1], self.W[to_, 1])
            self.lines.append(self.axis1.plot([self.W[from_, 0], self.W[to_, 0]], [self.W[from_, 1], self.W[to_, 1]], color="red"))

        nei_temp = self.nei
        self.slider_lr.setValue(self.lr * 100)
        self.nei = nei_temp
        self.slider_nei.setValue(self.nei * 10)
        self.label_lr.setText("Learning rate: " + str(self.lr))
        self.label_nei.setText("Neighborhood: " + str(self.nei))

        self.canvas.draw()

    def animate_init(self):
        while self.lines_anim:
            self.lines_anim.pop().remove()
        for _ in range(NN - 1):
            self.lines_anim.append(self.axis1.plot([], color="red")[0])

    def on_animate(self, idx):

        s, r = self.W.shape
        Q = P.shape[1]

        for z in range(100):
            q = int(np.fix(np.random.random() * Q))
            p = P[:, q].reshape(-1, 1)

            a = self.compet_(np.dot(self.W, p))
            i = np.argmax(a)
            N_c = np.copy(N)[:, i]
            N_c[N_c <= self.nei] = 1
            N_c[N_c != 1] = 0
            a = 0.5 * (a + N_c.reshape(-1, 1))

            self.W = self.W + self.lr * np.dot(a, np.ones((1, r))) * (np.dot(np.ones((s, 1)), p.T) - self.W)
            self.lr = (self.lr - 0.01) * 0.998 + 0.01
            self.nei = (self.nei - 1) * NDEC + 1
            self.do_slide = False
            self.slider_lr.setValue(self.lr * 100)
            self.slider_nei.setValue(self.nei * 10)
            self.label_lr.setText("Learning rate: " + str(round(self.lr, 2)))
            self.label_nei.setText("Neighborhood: " + str(round(self.nei, 2)))
            self.do_slide = True
            self.label_presentations.setText("Presentations: " + str((self.n_runs - 1) * 500 + idx * 100 + z + 1))

        for i in range(NN - 1):
            from_ = Nfrom[i] - 1
            to_ = Nto[i] - 1
            self.lines_anim[i].set_data([self.W[from_, 0], self.W[to_, 0]], [self.W[from_, 1], self.W[to_, 1]])

    def on_run_2(self):
        if self.ani:
            self.ani.event_source.stop()
        self.n_runs += 1
        self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init,
                                 frames=5, interval=0, repeat=False, blit=False)
        self.canvas.draw()

    @staticmethod
    def compet_(n):
        max_idx = np.argmax(n)
        out = np.zeros(n.shape)
        out[max_idx] = 1
        return out
