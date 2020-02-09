from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication

from nndesign_layout import NNDLayout
from Window import MainWindowNN

from get_package_path import PACKAGE_PATH


# -------------------------------------------------------------------------------------------------------------
xlabel, ylabel, wlabel, hlabel, add = 30, 5, 500, 100, 20
xtabel, ytlabel = 120, 25
xautor, yautor = 100, 580

w_Logom = 200; h_Logom = 100; xL_gm = 50; yL_gm= 140; wL_gm= 300; hL_gm = h_Logom;
w_Logom1 = 200; h_Logom1 = 100; xL_gm1 = 20; yL_gm1= 450; wL_gm1= 300; hL_gm1=h_Logom1;

xbtnm, ybtnm, wbtnm, hbtnm = 250, 160, 230, 50
ybtnm1 = 470
# -------------------------------------------------------------------------------------------------------------


class MainWindow(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        """ Window that shows the main menu, to choose between the two books """
        super(MainWindow, self).__init__(w_ratio, h_ratio, chapter_window=False, draw_vertical=False, create_plot=False)

        self.setWindowTitle("Neural Network Design")

        self.label1 = QtWidgets.QLabel(self)
        self.label1.setText("Neural Network Design")
        self.label1.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
        self.label1.setGeometry(xlabel * self.w_ratio, ylabel * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        self.label2 = QtWidgets.QLabel(self)
        self.label2.setText("Deep Learning")
        self.label2.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
        self.label2.setGeometry(xlabel * self.w_ratio, (ylabel + add) * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText("Table of Contents")
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.StyleItalic))
        self.label3.setGeometry(self.wm - xtabel * self.w_ratio, (ylabel + add) * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("By Hagan, Jafari")
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry(self.wm - xautor * self.w_ratio, yautor * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        self.statusBar()
        self.main_menu = self.menuBar()

        self.icon1 = QtWidgets.QLabel(self)
        self.icon1.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/NN.svg").pixmap(w_Logom, h_Logom, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon1.setGeometry(xL_gm * self.w_ratio, yL_gm * self.h_ratio, wL_gm * self.w_ratio, hL_gm * self.h_ratio)

        self.icon2 = QtWidgets.QLabel(self)
        self.icon2.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/DL.svg").pixmap(w_Logom1, h_Logom1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        # self.icon2 = self.icon2.scaled(32, 32, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.icon2.setGeometry(xL_gm1 * self.w_ratio, yL_gm1 * self.h_ratio, wL_gm1 * self.w_ratio, hL_gm1 * self.h_ratio)

        self.button1 = QtWidgets.QPushButton("Neural Network Design", self)
        self.button1.setGeometry(self.wm - xbtnm * self.w_ratio, ybtnm * self.h_ratio, wbtnm * self.w_ratio, hbtnm * self.h_ratio)
        self.button1.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.button1.clicked.connect(self.new_window1)
        self.button1.setStyleSheet("background-color: rgb(125, 150, 255);\nborder:3px solid rgb(100, 170, 255);")
        self.button1_win = None

        self.button2 = QtWidgets.QPushButton("Neural Network Design : Deep Learning", self)
        self.button2.setGeometry(self.wm - xbtnm * self.w_ratio, ybtnm1 * self.h_ratio, wbtnm * self.w_ratio, hbtnm * self.h_ratio)
        self.button2.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.button2.clicked.connect(self.new_window2)
        self.button2.setStyleSheet("background-color: rgb(125, 150, 255);\nborder:3px solid rgb(100, 170, 255);")
        self.button2_win = None

    def new_window1(self):
        self.button1_win = MainWindowNN(self.w_ratio, self.h_ratio)
        self.button1_win.show()

    @staticmethod
    def new_window2():
        print("TODO")  # TODO
        # self.button2_win = MainWindowDL()
        # self.button2_win.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


import sys
app = QApplication(sys.argv)
dimensions = QtWidgets.QDesktopWidget().screenGeometry(-1)
W_SCREEN, H_SCREEN = dimensions.width(), dimensions.height()
# W_SCREEN, H_SCREEN = 1900, 850  # To check how it would look on a bigger screen
W_RATIO, H_RATIO = W_SCREEN / 1280, H_SCREEN / 800
win = MainWindow(W_RATIO, H_RATIO)
win.show()
sys.exit(app.exec_())
