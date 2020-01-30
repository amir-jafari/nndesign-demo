from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow

from One_input_neuron import OneInputNeuron

from get_package_path import PACKAGE_PATH


# -------------------------------------------------------------------------------------------------------------
xm =500; ym= 150; wm = 900; hm =800;

xlabel =80; ylabel= 5; wlabel = 500; hlabel =100; add =20;
xtabel =560; ytabel=25 ; wtabel =500 ; htabel =100;
xautor = 650; yautor= 715; wautor = 500; hautor=100;

xcm1 =250; ycm1= 140; wcm1 = 350; hcm1 =20; add1 = 140; subt=20;
xbtn1 =150; ybtn1= 740; wbtn1 = 60; hbtn1=20; add2 = 80; add2_1 = 30;

xl1 =10; yl1= 90; wl1 = 700; hl1 =90;
xl2 =650; yl2= 780; wl2 = 900; hl2 =780;

w_Logo1 = 100;h_Logo1 = 80; xL_g1 = 150; yL_g1= 110; wL_g1= w_Logo1; hL_g1=h_Logo1; add_l = 140;


w_Logom = 200; h_Logom = 100; xL_gm = 80; yL_gm= 140; wL_gm= 3*w_Logo1; hL_gm = h_Logom;
w_Logom1 = 200; h_Logom1 = 100; xL_gm1 = 80; yL_gm1= 450; wL_gm1= 3*w_Logo1; hL_gm1=h_Logom1;

xbtnm =300; ybtnm= 140; wbtnm = 300; hbtnm=50;
xbtnm1 =300; ybtnm1= 470; wbtnm1 = 300; hbtnm1=50;
# -------------------------------------------------------------------------------------------------------------


class MainWindowNN(QMainWindow):
    def __init__(self):
        super(MainWindowNN, self).__init__()

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

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText("Table of Contents")
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.StyleItalic))
        self.label3.setGeometry(xtabel, ytabel, wtabel, htabel)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("By Hagan, Demuth, Beale, Jafari")
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry(xautor, yautor, wautor, hautor)

        self.icon1 = QtWidgets.QLabel(self)
        self.icon1.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_2.svg").pixmap(w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon1.setGeometry(xL_g1, yL_g1, wL_g1, hL_g1)

        self.icon2 = QtWidgets.QLabel(self)
        self.icon2.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_3.svg").pixmap(w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon2.setGeometry(xL_g1, yL_g1 + add_l, wL_g1, hL_g1)

        self.icon3 = QtWidgets.QLabel(self)
        self.icon3.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_4.svg").pixmap(w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon3.setGeometry(xL_g1, yL_g1 + 2 * add_l, wL_g1, hL_g1)

        self.icon4 = QtWidgets.QLabel(self)
        self.icon4.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_5.svg").pixmap(w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon4.setGeometry(xL_g1, yL_g1 + 3 * add_l, wL_g1, hL_g1)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.label_box1 = QtWidgets.QLabel(self)
        self.label_box1.setText("Neuron Model & Network Architecture:")
        self.label_box1.setGeometry(xcm1, ycm1 - subt, wcm1, hcm1)
        self.comboBox1.addItems(["Chapter2 demos", "One-input Neuron", "Two-input Neuron"])
        self.comboBox1.currentIndexChanged.connect(self.chapter2)
        self.comboBox1.setGeometry(xcm1, ycm1, wcm1, hcm1)
        self.one_input_neuron = None

        self.comboBox2 = QtWidgets.QComboBox(self)
        self.label_box2 = QtWidgets.QLabel(self)
        self.label_box2.setText("An Illustrative Example:")
        self.label_box2.setGeometry(xcm1, ycm1 + add1 - subt, wcm1, hcm1)
        self.comboBox2.addItems(["Chapter3 demos", "Perceptron classification", "Hamming classification", "Hopfield classification"])
        self.comboBox2.currentIndexChanged.connect(self.chapter3)
        self.comboBox2.setGeometry(xcm1, ycm1 + add1, wcm1, hcm1)

        self.comboBox3 = QtWidgets.QComboBox(self)
        self.label_box3 = QtWidgets.QLabel(self)
        self.label_box3.setText("Perceptron Learning Rule:")
        self.label_box3.setGeometry(xcm1, ycm1 + 2 * add1 - subt, wcm1, hcm1)
        self.comboBox3.addItems(["Chapter4 demos", "Decision boundaries", "Perceptron rule"])
        self.comboBox3.currentIndexChanged.connect(self.chapter3)
        self.comboBox3.setGeometry(xcm1, ycm1 + 2 * add1, wcm1, hcm1)

        self.comboBox4 = QtWidgets.QComboBox(self)
        self.label_box4 = QtWidgets.QLabel(self)
        self.label_box4.setText("Signal & weight Vector Spaces:")
        self.label_box4.setGeometry(xcm1, ycm1 + 3 * add1 - subt, wcm1, hcm1)
        self.comboBox4.addItems(["Chapter5 demos", "Gram schmidt", "Reciprocal basis"])
        self.comboBox4.currentIndexChanged.connect(self.chapter3)
        self.comboBox4.setGeometry(xcm1, ycm1 + 3 * add1, wcm1, hcm1)

        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setText("2-5")
        self.button1.setGeometry(xbtn1, ybtn1, wbtn1, hbtn1)
        self.button1.clicked.connect(self.new_window1)

        self.button2 = QtWidgets.QPushButton(self)
        self.button2.setText("6-9")
        self.button2.setGeometry(xbtn1 + add2, ybtn1, wbtn1, hbtn1)
        self.button2.clicked.connect(self.new_window2)

        self.button3 = QtWidgets.QPushButton(self)
        self.button3.setText("10-13")
        self.button3.setGeometry(xbtn1 + 2 * add2, ybtn1, wbtn1, hbtn1)
        self.button3.clicked.connect(self.new_window3)

        self.button4 = QtWidgets.QPushButton(self)
        self.button4.setText("14-17")
        self.button4.setGeometry(xbtn1 + 3* add2, ybtn1, wbtn1, hbtn1)
        self.button4.clicked.connect(self.new_window4)

        self.button5 = QtWidgets.QPushButton(self)
        self.button5.setText("18-21")
        self.button5.setGeometry(xbtn1, ybtn1 + add2_1, wbtn1, hbtn1)
        self.button5.clicked.connect(self.new_window5)

        self.button6 = QtWidgets.QPushButton(self)
        self.button6.setText("Textbook Info")
        self.button6.setGeometry(xbtn1 + add2, ybtn1 + add2_1, wbtn1, hbtn1)
        self.button6.clicked.connect(self.new_window6)

        self.button7 = QtWidgets.QPushButton(self)
        self.button7.setText("Close")
        self.button7.setGeometry(xbtn1 + 3 * add2, ybtn1 + add2_1, wbtn1, hbtn1)
        self.button7.clicked.connect(self.new_window7)

    def chapter2(self, idx):
        if idx == 1:
            self.one_input_neuron = OneInputNeuron()
            self.one_input_neuron.show()
        elif idx == 2:
            self.myOtherWindow1 = two_input_neuron()

    def chapter3(self, idx):
        if idx == 1:
            self.myOtherWindow = perceptron_classification()
        if idx == 2:
            self.myOtherWindow1 = hamming_classification()
        elif idx == 3:
            self.myOtherWindow1 = hopfield_classification()

    def chapter4(self, idx):
        if idx == 1:
            self.myOtherWindow = decision_boundaries()
        elif idx == 2:
            self.myOtherWindow1 = perceptron_rule()

    def chapter5(self, idx):
        if idx == 1:
            self.myOtherWindow = gram_schmidt()
        elif idx == 2:
            self.myOtherWindow1 = reciprocal_basis()

    @staticmethod
    def new_window1():
        print("TODO")

    @staticmethod
    def new_window2():
        print("TODO")

    @staticmethod
    def new_window3():
        print("TODO")

    @staticmethod
    def new_window4():
        print("TODO")

    @staticmethod
    def new_window5():
        print("TODO")

    @staticmethod
    def new_window6():
        print("TODO")

    @staticmethod
    def new_window7():
        print("TODO")

    """def paintEvent(self, e):
        qp = QtGui.QPainter()
        color = QtGui.QColor(0, 0, 0)
        color.setNamedColor('#d4d4d4')
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):
        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(xl1, yl1, wl1, hl1)

        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(xl2, yl2, wl2, hl2)

    def close_application(self):
        choice = QtGui.QMessageBox.question(self, 'Extract!', "Get into the chopper?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)"""