from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from nndesign_layout import NNDLayout
from get_package_path import PACKAGE_PATH


class HammingClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(HammingClassification, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Hamming Classification", 3, "Click [Go] to send a fruit\ndown the belt to be\nclassified"
                          " by a Hamming\nnetwork.\n\nThe calculations for the\nHamming network will\nappear below.",
                          PACKAGE_PATH + "Logo/Logo_Ch_3.svg", None)

        self.make_plot(1, (15, 100, 500, 390))
        self.axis = Axes3D(self.figure)
        ys = np.linspace(-1, 1, 100)
        zs = np.linspace(-1, 1, 100)
        Y, Z = np.meshgrid(ys, zs)
        X = 0
        apple = np.array([-1, 1, -1])
        orange = np.array([1, 1, -1])
        self.axis.set_title("Input Space")
        self.axis.plot_surface(X, Y, Z, alpha=0.5)
        self.axis.set_xlabel("texture")
        self.axis.set_xticks([-1, 1])
        self.axis.set_ylabel("shape")
        self.axis.set_yticks([-1, 1])
        self.axis.set_zlabel("weight")
        self.axis.zaxis._axinfo['label']['space_factor'] = 0.1
        self.axis.set_zticks([-1, 1])
        self.axis.scatter(orange[0], orange[1], orange[2], color='orange')
        self.axis.scatter(apple[0], apple[1], apple[2], color='yellow')
        self.line1, self.line2, self.line3 = None, None, None
        self.canvas.draw()

        self.p, self.a1, self.a2, self.fruit, self.label = None, None, None, None, None

        self.make_label("label_w1", "", (550, 300, 150, 25))
        self.make_label("label_b", "", (550, 330, 150, 25))
        self.make_label("label_w2", "", (550, 360, 150, 25))
        self.make_label("label_p", "", (550, 390, 150, 25))
        self.make_label("label_a_11", "", (550, 420, 150, 25))
        self.make_label("label_a_12", "", (550, 450, 150, 25))
        self.make_label("label_a_21", "", (550, 480, 150, 25))
        self.make_label("label_a_22", "", (550, 510, 150, 25))
        self.make_label("label_fruit", "", (550, 540, 150, 25))

        self.icon3 = QtWidgets.QLabel(self)
        self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon3.setGeometry(15 * self.w_ratio, 500 * self.h_ratio, 500 * self.w_ratio, 150 * self.h_ratio)

        self.make_button("run_button", "Go", (self.x_chapter_button, 570, self.w_chapter_button, self.h_chapter_button), self.on_run)

    def on_run(self):
        self.timer = QtCore.QTimer()
        self.idx = 0
        self.timer.timeout.connect(self.update_label)
        self.timer.start(1000)

    def update_label(self):
        if self.idx == 0:
            self.label_w1.setText(""); self.label_b.setText(""); self.label_w2.setText(""); self.label_p.setText("")
            self.label_a_11.setText(""); self.label_a_12.setText(""); self.label_a_21.setText(""); self.label_a_22.setText("")
            self.label_fruit.setText("")
            self.p = np.round(np.random.uniform(-1, 1, (1, 3)), 2)
            # self.p = np.array([[0.88, 0.88, -0.57]])
            w1, w2 = np.array([[1, -1, -1], [1, 1, -1]]), np.array([[1, -0.5], [-0.5, 1]])
            # TODO: b is not used?
            self.a1 = np.round(np.dot(w1, self.p.T), 2)
            self.a2 = np.round(np.dot(w2, self.a1), 2)
            self.a2 = np.round(np.array([[self.poslin(self.a2[0, 0])], [self.poslin(self.a2[1, 0])]]), 2)
            self.fruit = "Orange" if self.a2[0, 0] > 0 else "Apple"
            self.label = 1 if self.fruit == "Apple" else 0
            if self.line1:
                self.line1.pop().remove()
                self.line2.pop().remove()
                self.line3.pop().remove()
                self.canvas.draw()
        if self.idx == 1:
            self.label_w1.setText("W1 = [1 -1 -1; 1, 1, -1]")
            self.label_b.setText("b = [3; 3]")
            self.label_w2.setText("W2 = [1 -0.5; -0.5, 1]")
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_2.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_7.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 2:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_3.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_8.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 3:
            self.label_p.setText("p = [{} {} {}]".format(self.p[0, 0], self.p[0, 1], self.p[0, 2]))
            self.line1 = self.axis.plot3D([self.p[0, 0]] * 10, np.linspace(-1, 1, 10), [self.p[0, 2]] * 10, color="g")
            self.line2 = self.axis.plot3D([self.p[0, 0]] * 10, [self.p[0, 1]] * 10, np.linspace(-1, 1, 10), color="g")
            self.line3 = self.axis.plot3D(np.linspace(-1, 1, 10), [self.p[0, 1]] * 10, [self.p[0, 2]] * 10, color="g")
            self.canvas.draw()
        elif self.idx == 4:
            self.label_a_11.setText("a1 = purelin(W1 * p + b)")
        elif self.idx == 5:
            self.label_a_12.setText("a1 = [{} {}]".format(self.a1[0, 0], self.a1[1, 0]))
        elif self.idx == 6:
            self.label_a_21.setText("a2 = poslin(W2 * a1)")
        elif self.idx == 7:
            self.label_a_22.setText("a2 = [{} {}]".format(self.a2[0, 0], self.a2[1, 0]))
        elif self.idx == 8:
            self.label_fruit.setText("Fruit = {}".format(self.fruit))
        elif self.idx == 9:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_4.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_9.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 10:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_5.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_10.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        elif self.idx == 11:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_6.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_11.svg").pixmap(500 * self.w_ratio, 150 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        else:
            pass
        self.idx += 1

    @staticmethod
    def poslin(x):
        return x if x >= 0 else 0
