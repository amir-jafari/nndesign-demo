from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
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
    def __init__(self, w_ratio, h_ratio, chapter_window=True, main_menu=False, draw_vertical=True, create_plot=True):
        super(NNDLayout, self).__init__()

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

        # TODO: elif main_menu == 2 (for second book)

        if create_plot:

            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)

            self.wid1 = QtWidgets.QWidget(self)
            self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
            self.wid1.setGeometry(15 * self.w_ratio, 300 * self.h_ratio, 490 * self.w_ratio, 370 * self.h_ratio)
            self.layout1.addWidget(self.canvas)
            self.wid1.setLayout(self.layout1)

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

    def fill_chapter(self, title, number, description, logo_path, icon_path, show_info=True):

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

        self.icon2 = QtWidgets.QLabel(self)
        self.icon2.setPixmap(QtGui.QIcon(icon_path).pixmap(wp_pic2_2 * self.w_ratio, hp_pic2_2 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon2.setGeometry(x_pic2_2 * self.w_ratio, y_pic2_2 * self.h_ratio, w_pic2_2 * self.w_ratio, h_pic2_2 * self.h_ratio)

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

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
