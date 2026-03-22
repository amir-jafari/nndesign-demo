from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np
from time import sleep

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class HopfieldClassification(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(HopfieldClassification, self).__init__(w_ratio, h_ratio, dpi, main_menu=1)

        self.fill_chapter("Hopfield Classification", 3, "Click [Go] to send a fruit\ndown the belt to be\nclassified"
                          " by a Hopfield\nnetwork.\n\nThe calculations for the\nHopfield network will\nappear below.",
                          PACKAGE_PATH + "Logo/Logo_Ch_3.svg", None)

        if self.play_sound:
            self.initial_sound('start_sound1', "Sound/blip.wav")
            self.initial_sound('start_sound2', "Sound/bloop.wav")
            self.initial_sound('wind_sound', "Sound/wind.wav")
            self.initial_sound('knock_sound', "Sound/knock.wav")
            self.initial_sound('scan_sound', "Sound/buzz.wav")
            self.initial_sound('classify_sound', "Sound/classify.wav")

        self.make_plot(1, (15, 100, 500, 390))
        self.axis = self.figure.add_subplot(projection='3d')
        X, Z = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        self.axis.set_title("Input Space")
        self.axis.plot_surface(X, 0, Z, alpha=0.5)
        self.axis.set_xlabel("shape")
        self.axis.set_xlim(-1, 1)
        self.axis.set_xticks([-1, 1])
        self.axis.set_ylabel("texture")
        self.axis.set_ylim(-1, 1)
        self.axis.set_yticks([-1, 1])
        self.axis.set_zlabel("weight")
        self.axis.zaxis._axinfo['label']['space_factor'] = 0.1
        self.axis.set_zlim(-1, 1)
        self.axis.set_zticks([-1, 1])
        self.axis.scatter(1, -1, -1, color='orange')
        self.axis.scatter(1, 1, -1, color='green')
        self.line1, self.line2, self.line3 = None, None, None
        self.axis.view_init(10, 20)
        self.canvas.draw()

        self.p, self.a1, self.a2, self.fruit, self.label = None, None, None, None, None

        self.make_label("label_w", "W = [.2 0 0; 0 1.2 0; 0 0 .2]", (532, 320 - 5, 170, 25))
        self.make_label("label_b", "b = [0.9; 0; -0.9]", (532, 350 - 5, 170, 25))
        self.make_label("label_p", "", (532, 380 - 5, 170, 25))
        self.make_label("label_a_11", "", (532, 410 - 5, 170, 25))
        self.make_label("label_a_12", "", (532, 440 - 5, 170, 25))
        self.make_label("label_fruit", "", (532, 470 - 5, 170, 25))

        if self.dpi > 113.5:
            self.figure_w, self.figure_h = round(575 / (self.dpi / 113.5)), round(190 / (self.dpi / 113.5))
        else:
            self.figure_w, self.figure_h = 575, 190
        self.icon3 = QtWidgets.QLabel(self)
        if self.running_on_windows:
            if self.dpi > 113.5:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_1.svg").pixmap(self.figure_w * self.h_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
                self.icon3.setGeometry(round(28 * self.h_ratio * (self.dpi / 113.5)), 485 * self.h_ratio, self.figure_w * self.h_ratio, self.figure_h * self.h_ratio)
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_1.svg").pixmap(self.figure_w * self.h_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
                self.icon3.setGeometry(28 * self.h_ratio, 485 * self.h_ratio, self.figure_w * self.h_ratio, self.figure_h * self.h_ratio)
        else:
            if self.dpi > 113.5:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_1.svg").pixmap(round(self.figure_w * self.w_ratio / (self.dpi / 113.5)), round(self.figure_h * self.h_ratio / (self.dpi / 113.5)), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
                self.icon3.setGeometry(round(28 * self.w_ratio * (self.dpi / 113.5)), 485 * self.h_ratio, round(self.figure_w * self.w_ratio / (self.dpi / 113.5)), round(self.figure_h * self.h_ratio / (self.dpi / 113.5)))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_1.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
                self.icon3.setGeometry(28 * self.w_ratio, 485 * self.h_ratio, self.figure_w * self.w_ratio, self.figure_h * self.h_ratio)
        self.text_shape, self.text_texture, self.text_weight = "?", "?", "?"

        self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_1.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
        self.make_button("run_button", "Go", (self.x_chapter_button, 500, self.w_chapter_button, self.h_chapter_button), self.on_run)
        self.make_button("btn_pause", "Pause", (self.x_chapter_button, 500 + self.h_chapter_button + 5, self.w_chapter_button, self.h_chapter_button), self.toggle_pause)

    def paintEvent(self, event):
        super(HopfieldClassification, self).paintEvent(event)
        pixmap = self.icon3.pixmap()
        painter = QtGui.QPainter(pixmap)
        if self.running_on_windows:
            w_ratio, h_ratio = self.h_ratio, self.h_ratio
        else:
            w_ratio, h_ratio = self.w_ratio, self.h_ratio
        if self.dpi > 113.5:
            w_ratio = round(w_ratio / (self.dpi / 113.5))
            h_ratio = round(h_ratio / (self.dpi / 113.5))
        painter.setFont(QtGui.QFont("times", 12 * (w_ratio + h_ratio) // 2))
        painter.drawText(QtCore.QPoint(100 * w_ratio, 26 * h_ratio), self.text_shape)
        painter.drawText(QtCore.QPoint(230 * w_ratio, 26 * h_ratio), self.text_texture)
        painter.drawText(QtCore.QPoint(360 * w_ratio, 26 * h_ratio), self.text_weight)
        painter.end()
        self.icon3.setPixmap(pixmap)

    def on_run(self):
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        self.btn_pause.setText("Pause")
        self.timer = QtCore.QTimer()
        self.idx = 0
        self.converge_step = 0
        self.text_shape, self.text_texture, self.text_weight = "?", "?", "?"
        self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_1.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
        self.label_p.setText("")
        self.label_a_11.setText("")
        self.label_a_12.setText("")
        self.label_fruit.setText("")
        self.p = np.round(np.random.uniform(-1, 1, (1, 3)), 2)
        # self.p = np.array([[-.74, .01, -.14]]) # A boundary Test

        w = np.array([[0.2, 0, 0], [0, 1.2, 0], [0, 0, 0.2]])
        b = np.array([[0.9], [0], [-0.9]])
        # Iterate until convergence: while any(a ~= orange) & any(a ~= apple)
        self.a_history = []
        a_curr = self.p.T.copy()
        orange_proto = np.array([[1.0], [-1.0], [-1.0]])
        apple_proto = np.array([[1.0], [1.0], [-1.0]])
        for _ in range(50):
            if np.all(a_curr == orange_proto) or np.all(a_curr == apple_proto):
                break
            a_next = np.round(self.satlins(np.dot(w, a_curr) + b), 2)
            self.a_history.append(a_next)
            if np.array_equal(a_next, a_curr):
                break
            a_curr = a_next
        self.a1 = self.a_history[0] if self.a_history else self.p.T.copy()
        self.a2 = self.a_history[-1] if self.a_history else self.p.T.copy()
        self.converged = np.all(self.a2 == orange_proto) or np.all(self.a2 == apple_proto)
        self.fruit = "Orange" if self.a2[1, 0] < 0 else "Apple"
        self.label = 1 if self.fruit == "Apple" else 0
        if self.line1:
            self.line1.pop().remove()
            self.line2.pop().remove()
            self.line3.pop().remove()
            self.canvas.draw()
        self.timer.timeout.connect(self.update_label)
        self.timer.start(900)

    def toggle_pause(self):
        if not hasattr(self, 'timer') or not self.timer:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("Play")
        else:
            self.timer.start()
            self.btn_pause.setText("Pause")

    def update_label(self):
        if self.idx == 0:
            if self.play_sound:
                self.start_sound1.play()
                sleep(0.5)
                self.start_sound2.play()
        if self.idx == 1:
            if self.play_sound:
                self.start_sound1.play()
                sleep(0.5)
                self.start_sound2.play()
        if self.idx == 2:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_2.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_7.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            if self.play_sound:
                self.wind_sound.play()
        elif self.idx == 3:
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_3.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_8.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
        elif self.idx == 4:
            self.text_shape, self.text_texture, self.text_weight = str(self.p[0, 0]), str(self.p[0, 1]), str(self.p[0, 2])
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_3.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_8.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            if self.play_sound:
                self.scan_sound.play()
        elif self.idx == 5:
            self.line1 = self.axis.plot3D([self.p[0, 0]] * 10, np.linspace(-1, 1, 10), [self.p[0, 2]] * 10, color="g")
            self.line2 = self.axis.plot3D([self.p[0, 0]] * 10, [self.p[0, 1]] * 10, np.linspace(-1, 1, 10), color="g")
            self.line3 = self.axis.plot3D(np.linspace(-1, 1, 10), [self.p[0, 1]] * 10, [self.p[0, 2]] * 10, color="g")
            self.canvas.draw()
            if self.play_sound:
                self.classify_sound.play()
        elif self.idx == 6:
            self.label_p.setText("a(0) = p = [{} {} {}]".format(self.p[0, 0], self.p[0, 1], self.p[0, 2]))
        elif self.idx == 7:
            self.label_a_11.setText("a(t+1) = satlins(W*a(t)+b)")
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_4.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_9.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            if self.play_sound:
                self.start_sound1.play()
                sleep(0.5)
                self.start_sound2.play()
        elif self.idx == 8:
            if self.converge_step < len(self.a_history):
                a = self.a_history[self.converge_step]
                self.label_a_12.setText("a({}) = [{} {} {}]".format(self.converge_step + 1, a[0, 0], a[1, 0], a[2, 0]))
                self.converge_step += 1
                return
            if not self.converged:
                self.timer.stop()
                QtWidgets.QMessageBox.warning(self, "No Convergence",
                    "The network could not classify this input.\n"
                    "The input may be too ambiguous (e.g. texture \u2248 0).\n\n"
                    "Please click Go again to try a new input.")
                return
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_5.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_10.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            if self.play_sound:
                self.wind_sound.play()
        elif self.idx == 9:
            self.label_fruit.setText("Fruit = {}".format(self.fruit))
            if self.label == 1:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_6.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            else:
                self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Figures/nnd3d1_11.svg").pixmap(self.figure_w * self.w_ratio, self.figure_h * self.h_ratio, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On))
            if self.play_sound:
                self.knock_sound.play()
                sleep(0.5)
                self.knock_sound.play()
            self.timer.stop()
        else:
            pass
        self.idx += 1

    @staticmethod
    @np.vectorize
    def satlins(x):
        return max([-1.0, min([x, 1.0])])
