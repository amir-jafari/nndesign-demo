from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


xm =500; ym= 150; wm = 900; hm =800;
xlabel =80; ylabel= 5; wlabel = 500; hlabel =100; add =20;

xl1 =10; yl1= 90; wl1 = 700; hl1 =90;
xl2 =700; yl2= 780; wl2 = 900; hl2 =780;
xl3 = wl1;yl3 = hl1+35;wl3 = wl1;hl3 = 750;


class NNDLayout(QMainWindow):
    def __init__(self, main_menu=False, draw_vertical=True, create_plot=True):
        super(NNDLayout, self).__init__()

        self.draw_vertical = draw_vertical
        if main_menu == 1:

            self.setGeometry(xm, ym, wm, hm)
            self.setWindowTitle("Neural Network Design")

            self.label1 = QtWidgets.QLabel(self)
            self.label1.setText("Neural Network")
            self.label1.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
            self.label1.setGeometry(xlabel, ylabel, wlabel, hlabel)

            self.label2 = QtWidgets.QLabel(self)
            self.label2.setText("DESIGN")
            self.label2.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
            self.label2.setGeometry(xlabel, ylabel + add, wlabel, hlabel)

        # TODO: elif main_menu == 2 (for second book)

        if create_plot:

            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)

            self.wid1 = QtWidgets.QWidget(self)
            self.layout1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
            self.wid1.setGeometry(10, 300, 680, 500)
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
        qp.drawLine(xl1, yl1, wl1, hl1)

        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(xl2, yl2, wl2, hl2)

        if self.draw_vertical:
            pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(xl3, yl3, wl3, hl3)
