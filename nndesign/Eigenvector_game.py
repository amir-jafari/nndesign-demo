from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH

# Figured out why I was having so much trouble with the quiver scaling. It was because the plot was now a square...!!


class EigenvectorGame(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(EigenvectorGame, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=True)

        self.fill_chapter("Eigenvector Game", 6, "TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_6.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Original Vectors", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_points = []
        self.axes1_v1 = self.axes_1.quiver([0], [0], [0], [0], units="xy", scale=1)
        self.axes1_v2 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1)
        self.axes1_eig1 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, color="green")
        self.eig1_found, self.slope1 = False, None
        self.axes1_eig2 = self.axes_1.quiver([0], [0], [0], [0],  units="xy", scale=1, color="green")
        self.eig2_found = False
        self.axes_1.set_xticks([-2, -1, 0, 1])
        self.axes_1.set_yticks([-2, -1, 0, 1])
        self.axes_1.set_xlabel("$x$")
        self.axes_1.xaxis.set_label_coords(1, -0.025)
        self.axes_1.set_ylabel("$y$")
        self.axes_1.yaxis.set_label_coords(-0.025, 1)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick1)

        self.label_n_tries = QtWidgets.QLabel(self)
        self.label_n_tries.setText("Number of Tries Left: 10")
        self.label_n_tries.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_n_tries.setGeometry(530 * self.w_ratio, 300 * self.h_ratio, 150 * self.w_ratio, 50 * self.h_ratio)
        self.slider_n_tries = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider_n_tries.setRange(0, 10)
        self.slider_n_tries.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_n_tries.setTickInterval(1)
        self.slider_n_tries.setValue(10)
        self.n_tries = 10
        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(530 * self.w_ratio, 330 * self.h_ratio, 100 * self.w_ratio, 200 * self.h_ratio)
        self.layout4.addWidget(self.slider_n_tries)
        self.wid4.setLayout(self.layout4)
        self.slider_n_tries.valueChanged.connect(self.freeze)

        self.label_message = QtWidgets.QLabel(self)
        self.label_message.setText("")
        self.label_message.setFont(QtGui.QFont("Times New Roman", 18))
        self.label_message.setGeometry(150 * self.w_ratio, 100 * self.h_ratio, 400 * self.w_ratio, 150 * self.h_ratio)

        self.label_message1 = QtWidgets.QLabel(self)
        self.label_message1.setText("")
        self.label_message1.setFont(QtGui.QFont("Times New Roman", 14))
        self.label_message1.setGeometry(150 * self.w_ratio, 200 * self.h_ratio, 400 * self.w_ratio, 150 * self.h_ratio)

        self.button = QtWidgets.QPushButton("Restart", self)
        self.button.setStyleSheet("font-size:13px")
        self.button.setGeometry(self.x_chapter_button * self.w_ratio, 530 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.button.clicked.connect(self.random_transform)
        self.A, self.eig_v1, self.eig_v2 = None, None, None
        self.random_transform()

    def freeze(self):
        self.slider_n_tries.setValue(self.n_tries)

    def random_transform(self):
        self.n_tries = 10
        self.label_n_tries.setText("Number of Tries Left: 10")
        self.slider_n_tries.setValue(10)
        self.clear_all()
        self.eig1_found, self.slope1, self.eig2_found = False, None, False
        self.axes1_eig1.set_UVC(0, 0)
        self.axes1_eig2.set_UVC(0, 0)
        self.canvas.draw()
        self.label_message.setText("")
        self.A = np.eye(2)
        while np.any(np.abs(np.array([self.A[0, 1], self.A[1, 0]])) < 0.15):
            angle1 = np.random.normal() * 360 - 180
            if angle1 <= -180:
                angle1 = 180
            elif angle1 >= 180:
                angle1 = 180
            else:
                angle1 = round(angle1, 1)
                while angle1 % 10 == 0:
                    angle1 += 1
            v1 = 0.8 * np.array([[np.cos(angle1 * 0.0175)], [np.sin(angle1 * 0.0175)]])
            angle2 = angle1
            while np.sum(angle2 == (angle1 + np.array([0, 170, 180, 190, 350, -170, -180, -190, -350]))) > 0:
                angle2 = np.random.normal() * 360 - 180
                if angle2 <= -180:
                    angle2 = 180
                elif angle2 >= 180:
                    angle2 = 180
                else:
                    angle2 = round(angle2, 1)
                    while angle2 % 10 == 0:
                        angle2 += 1
            v2 = 0.8 * np.array([[np.cos(angle2 * 0.0175)], [np.sin(angle2 * 0.0175)]])
            m = np.array([[v1[0, 0], v2[0, 0]], [v1[1, 0], v2[1, 0]]])
            e = (0.4 * np.random.uniform(0, 1, (2, 1)) + 0.3) * np.sign(np.random.uniform(0, 1, (2, 1)) - 0.5)
            self.A = np.dot(m, np.dot(np.diag(e.reshape(-1)), np.linalg.inv(m)))
            e, v = np.linalg.eig(self.A)
        self.eig_v1, self.eig_v2 = np.array([[v[0, 0]], [v[1, 0]]]), np.array([[v[0, 1]], [v[1, 1]]])

    def on_mouseclick1(self, event):
        if event.xdata != None and event.xdata != None and not self.eig2_found:
            if self.n_tries > 0:
                self.n_tries -= 1
                self.label_n_tries.setText("Number of Tries Left: {}".format(self.n_tries))
                self.slider_n_tries.setValue(self.n_tries)
                self.clear_all()
                self.axes1_points = [(round(event.xdata, 2), round(event.ydata, 2))]
                self.draw_vector()

    def draw_vector(self):
        self.axes1_v1.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
        slope_v1 = self.axes1_points[0][1] / self.axes1_points[0][0]
        v_transformed = np.dot(self.A, np.array([[self.axes1_points[0][0]], [self.axes1_points[0][1]]]))
        slope_v1_t = v_transformed[1, 0] / v_transformed[0, 0]
        self.axes1_v2.set_UVC(v_transformed[0, 0], v_transformed[1, 0])
        if not self.slope1:
            if abs(slope_v1 - slope_v1_t) < 0.1:
                self.axes1_eig1.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
                self.eig1_found, self.slope1 = True, slope_v1
                self.label_message1.setText("First eigenvector found!")
        else:
            if abs(slope_v1 - self.slope1) >= 0.1 and abs(slope_v1 - slope_v1_t) < 0.1:
                self.axes1_eig2.set_UVC(self.axes1_points[0][0], self.axes1_points[0][1])
                self.eig2_found = True
                self.clear_all()
                self.label_message1.setText("Second eigenvector found!")
                self.label_message.setText("You won :D. Click Restart to play again")
        if not self.eig2_found and self.n_tries == 0:
            self.label_message.setText("You lost :(. Click Restart to try again")
        self.canvas.draw()

    def clear_all(self):
        self.axes1_v1.set_UVC(0, 0)
        self.axes1_v2.set_UVC(0, 0)
        self.canvas.draw()
