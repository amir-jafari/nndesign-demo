from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow

from nndesign_layout import NNDLayout
from Window import MainWindowNN

from get_package_path import PACKAGE_PATH


# -------------------------------------------------------------------------------------------------------------
xm =500; ym= 150; wm = 900; hm =800;

xlabel =80; ylabel= 5; wlabel = 500; hlabel =100; add =20;
xtabel =560; ytabel=25 ; wtabel =500 ; htabel =100;
xautor = 700; yautor= 715; wautor = 500; hautor=100;

xcm1 =250; ycm1= 120; wcm1 = 250; hcm1 =20; add1 = 80; subt=20;
xbtn1 =150; ybtn1= 430; wbtn1 = 60; hbtn1=20; add2 = 80; add2_1 = 30;

w_Logo1 = 100;h_Logo1 = 80; xL_g1 = 150; yL_g1= 90; wL_g1= w_Logo1; hL_g1=h_Logo1; add_l = 80;

xl1 =10; yl1= 90; wl1 = 700; hl1 =90;
xl2 =700; yl2= 780; wl2 = 900; hl2 =780;

w_Logom = 200; h_Logom = 100; xL_gm = 50; yL_gm= 140; wL_gm= 3*w_Logo1; hL_gm = h_Logom;
w_Logom1 = 200; h_Logom1 = 100; xL_gm1 = 20; yL_gm1= 450; wL_gm1= 3*w_Logo1; hL_gm1=h_Logom1;

xbtnm =300; ybtnm= 160; wbtnm = 300; hbtnm=50;
xbtnm1 =300; ybtnm1= 470; wbtnm1 = 300; hbtnm1=50;
# -------------------------------------------------------------------------------------------------------------


class MainWindow(NNDLayout):
    def __init__(self):
        """ Window that shows the main menu, to choose between the two books """
        super(MainWindow, self).__init__(draw_vertical=False, create_plot=False)

        self.setGeometry(xm, ym, wm, hm)
        self.setWindowTitle("Neural Network Design")

        self.label1 = QtWidgets.QLabel(self)
        self.label1.setText("Neural Network Design:")
        self.label1.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
        self.label1.setGeometry(xlabel, ylabel, wlabel, hlabel)

        self.label2 = QtWidgets.QLabel(self)
        self.label2.setText("Deep Learning")
        self.label2.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
        self.label2.setGeometry(xlabel, ylabel + add, wlabel, hlabel)

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText("Table of Contents")
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.StyleItalic))
        self.label3.setGeometry(xtabel, ytabel, wtabel, htabel)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("By Hagan, Jafari")
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry(xautor, yautor, wautor, hautor)

        self.statusBar()
        self.main_menu = self.menuBar()

        self.icon1 = QtWidgets.QLabel(self)
        self.icon1.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/NN.svg").pixmap(w_Logom, h_Logom, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon1.setGeometry(xL_gm, yL_gm, wL_gm, hL_gm)

        self.icon2 = QtWidgets.QLabel(self)
        self.icon2.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/DL.svg").pixmap(w_Logom1, h_Logom1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon2.setGeometry(xL_gm1, yL_gm1, wL_gm1, hL_gm1)

        self.button1 = QtWidgets.QPushButton("Neural Network Design", self)
        self.button1.setGeometry(xbtnm, ybtnm, wbtnm, hbtnm)
        self.button1.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.button1.clicked.connect(self.new_window1)
        self.button1.setStyleSheet("background-color: rgb(125, 150, 255);\nborder:3px solid rgb(100, 170, 255);")
        self.button1_win = None

        self.button2 = QtWidgets.QPushButton("Neural Network Design : Deep Learning", self)
        self.button2.setGeometry(xbtnm1, ybtnm1, wbtnm1, hbtnm1)
        self.button2.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.button2.clicked.connect(self.new_window2)
        self.button2.setStyleSheet("background-color: rgb(125, 150, 255);\nborder:3px solid rgb(100, 170, 255);")
        self.button2_win = None

    def new_window1(self):
        self.button1_win = MainWindowNN()
        self.button1_win.show()

    @staticmethod
    def new_window2():
        print("TODO")  # TODO
        # self.button2_win = MainWindowDL()
        # self.button2_win.show()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        color = QtGui.QColor(0, 0, 0)
        color.setNamedColor('#d4d4d4')
        qp.begin(self)
        self.draw_lines(qp)
        qp.end()

    @staticmethod
    def draw_lines(qp):
        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(xl1, yl1, wl1, hl1)

        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(xl2, yl2, wl2, hl2)


def nndtoc():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

nndtoc()
