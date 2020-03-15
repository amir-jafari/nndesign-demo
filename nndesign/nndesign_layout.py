from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow

import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import math
# from matplotlib import rc
# rc('text', usetex=True)


WM_MAC_MAIN, HM_MAC_MAIN = 1280 - 750, 800 - 120  # For my Mac
WM_MAC_CHAPTER, HM_MAC_CHAPTER = 1280 - 580, 800 - 120  # For my Mac

# -------------------------------------------------------------------------------------------------------------
xlabel, ylabel, wlabel, hlabel, add = 30, 5, 500, 100, 20
xtabel, ytlabel = 120, 25
xautor, yautor = 100, 580

x_info, y_info, w_info, h_info = 530, 100, 450, 250

wp_pic2_1 = 100; hp_pic2_1 = 80; x_pic2_1 = 550; y_pic2_1= 50; w_pic2_1= wp_pic2_1; h_pic2_1=hp_pic2_1;
wp_pic2_2 = 500; hp_pic2_2 = 200; x_pic2_2 = 130; y_pic2_2= 100; w_pic2_2= 500; h_pic2_2=200;

# Lines
# Starting line point for my MAC. The ending point is determined by the w, h and ratio of screen compared to mine
xl1, yl1 = 10, 90
xl2 = 520
# -------------------------------------------------------------------------------------------------------------


class NNDLayout(QMainWindow):
    def __init__(self, w_ratio, h_ratio, chapter_window=True, main_menu=False, draw_vertical=True,
                 create_plot=True, create_plot_coords=(90, 300, 370, 370),
                 create_two_plots=False, print_mouse_coords=False):

        super(NNDLayout, self).__init__()

        self.print_mouse_coords = print_mouse_coords
        self.setMouseTracking(print_mouse_coords)

        self.w_ratio, self.h_ratio = w_ratio, h_ratio
        if chapter_window:
            self.wm, self.hm = WM_MAC_CHAPTER * w_ratio, HM_MAC_CHAPTER * h_ratio
        else:
            self.wm, self.hm = WM_MAC_MAIN * w_ratio, HM_MAC_MAIN * h_ratio
        self.setFixedSize(self.wm, self.hm)
        self.center()

        self.x_chapter_usual, self.w_chapter_button, self.h_chapter_button = 520, 170, 30
        self.x_chapter_button = 525
        self.x_chapter_slider_label = 590
        self.w_chapter_slider = 180

        self.label3, self.label4, self.label5, self.icon1, self.icon2 = None, None, None, None, None

        self.draw_vertical = draw_vertical
        if main_menu == 1:
            self.setWindowTitle("Neural Network Design")
            self.label1 = QtWidgets.QLabel(self)
            self.label1.setText("Neural Network")
            self.label1.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
            self.label1.setGeometry(xlabel * self.w_ratio, ylabel * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)
            self.label2 = QtWidgets.QLabel(self)
            self.label2.setText("DESIGN")
            self.label2.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
            self.label2.setGeometry(xlabel * self.w_ratio, (ylabel + add) * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        if main_menu == 2:
            self.setWindowTitle("Neural Network Design")
            self.label1 = QtWidgets.QLabel(self)
            self.label1.setText("Neural Network Design")
            self.label1.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
            self.label1.setGeometry(xlabel * self.w_ratio, ylabel * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)
            self.label2 = QtWidgets.QLabel(self)
            self.label2.setText("DEEP LEARNING")
            self.label2.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
            self.label2.setGeometry(xlabel * self.w_ratio, (ylabel + add) * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        if create_plot:

            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.wid1 = QtWidgets.QWidget(self)
            self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
            self.wid1.setGeometry(create_plot_coords[0] * self.w_ratio, create_plot_coords[1] * self.h_ratio,
                                  create_plot_coords[2] * self.w_ratio, create_plot_coords[3] * self.h_ratio)
            self.layout1.addWidget(self.canvas)
            self.wid1.setLayout(self.layout1)

        elif create_two_plots:

            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.wid1 = QtWidgets.QWidget(self)
            self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
            self.wid1.setGeometry(120 * self.w_ratio, 120 * self.h_ratio, 270 * self.w_ratio, 270 * self.h_ratio)
            self.layout1.addWidget(self.canvas)
            self.wid1.setLayout(self.layout1)

            self.figure2 = Figure()
            self.canvas2 = FigureCanvas(self.figure2)
            self.toolbar2 = NavigationToolbar(self.canvas2, self)
            self.wid2 = QtWidgets.QWidget(self)
            self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
            self.wid2.setGeometry(120 * self.w_ratio, 390 * self.h_ratio, 270 * self.w_ratio, 270 * self.h_ratio)
            self.layout2.addWidget(self.canvas2)
            self.wid2.setLayout(self.layout2)

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        color = QtGui.QColor(0, 0, 0)
        color.setNamedColor('#d4d4d4')
        qp.begin(self)
        self.draw_lines(qp)
        qp.end()

    def draw_lines(self, qp):
        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        # qp.drawLine(xl1 * self.w_ratio, yl1 * self.h_ratio, self.wm - xl1 * self.w_ratio, yl1 * self.h_ratio)
        qp.drawLine(xl1 * self.w_ratio, yl1 * self.h_ratio, xl2 * self.w_ratio, yl1 * self.h_ratio)
        if self.draw_vertical:
            pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            # qp.drawLine(self.wm - xl1 * self.w_ratio, yl1 * self.h_ratio + 35, self.wm - xl1 * self.w_ratio, 750 * self.h_ratio)
            qp.drawLine(xl2 * self.w_ratio, yl1 * self.h_ratio + 35, xl2 * self.w_ratio, 670 * self.h_ratio)

    def fill_chapter(self, title, number, description, logo_path, icon_path, show_info=True, icon_move_left=0, show_pic=True):

        # TODO: Use len of title to modify position of text, or actually, to set the line breaks on the right place in order to also scale according to resolution

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText(title)
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
        self.label3.setGeometry((xl2 - 120) * self.w_ratio, (ylabel + add) * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("Chapter {}".format(number))
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry((xl2 - 120) * self.w_ratio, ylabel * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        if show_info:
            self.label5 = QtWidgets.QLabel(self)
            self.label5.setText(description)
            self.label5.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
            self.label5.setGeometry(x_info * self.w_ratio, y_info * self.h_ratio, w_info * self.w_ratio, h_info * self.h_ratio)

        self.icon1 = QtWidgets.QLabel(self)
        self.icon1.setPixmap(QtGui.QIcon(logo_path).pixmap(wp_pic2_1 * self.w_ratio, hp_pic2_1 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon1.setGeometry(x_pic2_1 * self.w_ratio, y_pic2_1 * self.h_ratio, w_pic2_1 * self.w_ratio, h_pic2_1 * self.h_ratio)

        if show_pic:
            self.icon2 = QtWidgets.QLabel(self)
            self.icon2.setPixmap(QtGui.QIcon(icon_path).pixmap(wp_pic2_2 * self.w_ratio, hp_pic2_2 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            self.icon2.setGeometry((x_pic2_2 - icon_move_left) * self.w_ratio, y_pic2_2 * self.h_ratio, w_pic2_2 * self.w_ratio, h_pic2_2 * self.h_ratio)

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mouseMoveEvent(self, event):
        if self.print_mouse_coords:
            print('Mouse coords: ( %d : %d )' % (event.x(), event.y()))

    # https://stackoverflow.com/questions/32035251/displaying-latex-in-pyqt-pyside-qtablewidget
    def mathTex_to_QPixmap(self, mathTex, fs):

        # ---- set up a mpl figure instance ----

        fig = Figure()
        fig.patch.set_facecolor('none')
        fig.set_canvas(FigureCanvas(fig))
        renderer = fig.canvas.get_renderer()

        # ---- plot the mathTex expression ----

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')
        t = ax.text(0, 0, mathTex, ha='left', va='bottom', fontsize=fs)

        # ---- fit figure size to text artist ----

        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)

        text_bbox = t.get_window_extent(renderer)

        tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        tight_fheight = text_bbox.height * fheight / fig_bbox.height

        fig.set_size_inches(tight_fwidth, tight_fheight)

        # ---- convert mpl figure to QPixmap ----

        buf, size = fig.canvas.print_to_buffer()
        qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                      QtGui.QImage.Format_ARGB32))
        qpixmap = QtGui.QPixmap(qimage)

        return qpixmap

    @staticmethod
    def logsigmoid(n):
        return 1 / (1 + np.exp(-n))

    @staticmethod
    def logsigmoid_stable(n):
        n = np.clip(n, -100, 100)
        return 1 / (1 + np.exp(-n))

    @staticmethod
    def logsigmoid_der(n):
        return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))

    @staticmethod
    def purelin(n):
        return n

    @staticmethod
    def purelin_der(n):
        return np.array([1]).reshape(n.shape)

    @staticmethod
    def lin_delta(a, d=None, w=None):
        na, ma = a.shape
        if d is None and w is None:
            return -np.kron(np.ones((1, ma)), np.eye(na))
        else:
            return np.dot(w.T, d)

    @staticmethod
    def log_delta(a, d=None, w=None):
        s1, _ = a.shape
        if d is None and w is None:
            return -np.kron((1 - a) * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
        else:
            return (1 - a) * a * np.dot(w.T, d)

    @staticmethod
    def tan_delta(a, d=None, w=None):
        s1, _ = a.shape
        if d is None and w is None:
            return -np.kron(1 - a * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
        else:
            return (1 - a * a) * np.dot(w.T, d)

    @staticmethod
    def marq(p, d):
        s, _ = d.shape
        r, _ = p.shape
        return np.kron(p.T, np.ones((1, s))) * np.kron(np.ones((1, r)), d.T)

    @staticmethod
    def compet(n, axis=None):
        if axis is not None:
            max_idx = np.argmax(n, axis=axis)
            out = np.zeros(n.shape)
            for i in range(out.shape[1]):
                out[max_idx[i], i] = 1
            return out
        else:
            max_idx = np.argmax(n)
            out = np.zeros(n.shape)
            out[max_idx] = 1
            return out

    @staticmethod
    def hardlim(x):
        if x < 0:
            return 0
        else:
            return 1

    @staticmethod
    def hardlims(x):
        if x < 0:
            return -1
        else:
            return 1

    @staticmethod
    def satlin(x):
        if x < 0:
            return 0
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    def satlins(x):
        if x < -1:
            return 0
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    def logsig(x):
        return 1 / (1 + math.e ** (-x))

    @staticmethod
    def tansig(x):
        return 2 / (1 + math.e ** (-2 * x)) - 1
