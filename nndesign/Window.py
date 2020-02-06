from PyQt5 import QtWidgets, QtGui
from functools import partial

from nndesign_layout import NNDLayout

from One_input_neuron import OneInputNeuron
from Perceptron_rule import PerceptronRule
from Decision_boundary import DecisionBoundaries
from Function_approximation import FunctionApproximation
from Network_function import NetworkFunction
from Steepest_descent_quadratic import SteepestDescentQuadratic
from Comparison_of_methods import ComparisonOfMethods

from get_package_path import PACKAGE_PATH


# -------------------------------------------------------------------------------------------------------------
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

BOOK1_CHAPTERS_DEMOS = {
    2: ["Neuron Model & Network Architecture:", "Chapter2 demos", "One-input Neuron", "Two-input Neuron"],
    3: ["An Illustrative Example", "Chapter3 demos", "Perceptron classification", "Hamming classification", "Hopfield classification"],
    4: ["Perceptron Learning Rule", "Chapter4 demos", "Decision boundaries", "Perceptron rule"],
    5: ["Signal & weight Vector Spaces", "Chapter5 demos", "Gram schmidt", "Reciprocal basis"],
    6: ["Linear Transformations For N. Networks", "Chapter6 demos", "Linear transformations ", "Eigenvector game"],
    7: ["Supervised Hebb", "Chapter7 demos", "Supervised Hebb"],
    8: ["Performance Surfaces & Optimum Points", "Chapter8 demos", "Tylor series #1", "Tylor series #1", "Directional derivatives ", "Quadratic function"],
    9: ["Performance Optimization", "Chapter9 demos", "Steepest descent for Quadratic", "Metod comparison", "Newton's method", "Steepest descent"],
    10: ["Widrow - Hoff Learning", "Chapter10 demos", "Adaptive noise cancellation", "EEG noise cancellation", "Linear classification"],
    11: ["Backpropagation", "Chapter11 demos", "Network Function", "Backpropagation Calculation", "Function Approximation", "Generalization"],
    12: ["Variations on Backpropagation", "Chapter12 demos", "Steepest Descent #1", "Steepest Descent #2", "Momentum", "Variable Learning Rate", "CG Line Search", "Conjugate Gradient", "Marquardt Step", "Marquardt" ],
    13: ["Generalization", "Chapter13 demos", "Early Stopping", "Regularization", "Bayesian Regularization", "Early Stopping-Regularization"],
    14: ["Dynamic Networks", "Chapter14 demos", "FIR Network", "IIR Network", "Dynamic Derivatives", "Recurrent Network Training"],
    15: ["Associative Learning", "Chapter15 demos", "Unsupervised Hebb", "Effects of Decay Rate", "Hebb with Decay", "Graphical Instar", "Outstar"],
    16: ["Competitive Networks", "Chapter16 demos", "Competitive Classification", "Competitive Learning", "1-D Feature Map", "2-D Feature Map", "LVQ 1", "LVQ 2"],
    17: ["Competitive Networks", "Chapter17 demos", "Network Function Radial", "Pattern Classification", "Linear Least Squares", "Orthogonal Least Squares", "Non Linear Optimization"],
    18: ["Grossberg Network", "Chapter18 demos", "Leaky Integrator", "Shunting Network", "Grossberg Layer 1", "Grossberg Layer 2", "Adaptive Weights"],
    19: ["Adaptive Resonance Theory", "Chapter19 demos", "ART1 Layer 1", "ART1 Layer 2", "Orienting Subsystem", "ART1 Algorithm"],
    20: ["Stability", "Chapter20 demos", "Dynamical System"],
    21: ["Hopfield Network", "Chapter21 demos", "Hopfield Network"]
}
# -------------------------------------------------------------------------------------------------------------


class MainWindowNN(NNDLayout):
    def __init__(self):
        """ Main Window for the Neural Network Design Book. Inherits basic layout from NNDLayout """
        super(MainWindowNN, self).__init__(main_menu=1, draw_vertical=False, create_plot=False)

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText("Table of Contents")
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.StyleItalic))
        self.label3.setGeometry(xtabel, ytabel, wtabel, htabel)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("By Hagan, Demuth, Beale, Jafari")
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry(xautor, yautor, wautor, hautor)

        # ---- Chapter icons and dropdown menus ----

        self.chapter_window1, self.chapter_window2, self.chapter_window3, self.chapter_window4 = None, None, None, None

        self.icon1 = QtWidgets.QLabel(self)
        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.connected = False  # Need to create this attribute so that we don't have more than one connected function
        self.comboBox1.setGeometry(xcm1, ycm1, wcm1, hcm1)
        self.label_box1 = QtWidgets.QLabel(self)
        self.label_box1.setGeometry(xcm1, ycm1 - subt, wcm1, hcm1)

        self.icon2 = QtWidgets.QLabel(self)
        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.connected = False
        self.comboBox2.setGeometry(xcm1, ycm1 + add1, wcm1, hcm1)
        self.label_box2 = QtWidgets.QLabel(self)
        self.label_box2.setGeometry(xcm1, ycm1 + add1 - subt, wcm1, hcm1)

        self.icon3 = QtWidgets.QLabel(self)
        self.comboBox3 = QtWidgets.QComboBox(self)
        self.comboBox3.connected = False
        self.comboBox3.setGeometry(xcm1, ycm1 + 2 * add1, wcm1, hcm1)
        self.label_box3 = QtWidgets.QLabel(self)
        self.label_box3.setGeometry(xcm1, ycm1 + 2 * add1 - subt, wcm1, hcm1)

        self.icon4 = QtWidgets.QLabel(self)
        self.comboBox4 = QtWidgets.QComboBox(self)
        self.comboBox4.connected = False
        self.comboBox4.setGeometry(xcm1, ycm1 + 3 * add1, wcm1, hcm1)
        self.label_box4 = QtWidgets.QLabel(self)
        self.label_box4.setGeometry(xcm1, ycm1 + 3 * add1 - subt, wcm1, hcm1)

        self.show_chapters()

        # ---- Buttons at the bottom to switch between blocks of chapters ----

        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setText("2-5")
        self.button1.setGeometry(xbtn1, ybtn1, wbtn1, hbtn1)
        self.button1.clicked.connect(partial(self.show_chapters, "2-5"))

        self.button2 = QtWidgets.QPushButton(self)
        self.button2.setText("6-9")
        self.button2.setGeometry(xbtn1 + add2, ybtn1, wbtn1, hbtn1)
        self.button2.clicked.connect(partial(self.show_chapters, "6-9"))

        self.button3 = QtWidgets.QPushButton(self)
        self.button3.setText("10-13")
        self.button3.setGeometry(xbtn1 + 2 * add2, ybtn1, wbtn1, hbtn1)
        self.button3.clicked.connect(partial(self.show_chapters, "10-13"))

        self.button4 = QtWidgets.QPushButton(self)
        self.button4.setText("14-17")
        self.button4.setGeometry(xbtn1 + 3 * add2, ybtn1, wbtn1, hbtn1)
        self.button4.clicked.connect(partial(self.show_chapters, "14-17"))

        self.button5 = QtWidgets.QPushButton(self)
        self.button5.setText("18-21")
        self.button5.setGeometry(xbtn1, ybtn1 + add2_1, wbtn1, hbtn1)
        self.button5.clicked.connect(partial(self.show_chapters, "18-21"))

        self.button6 = QtWidgets.QPushButton(self)
        self.button6.setText("Textbook Info")
        self.button6.setGeometry(xbtn1 + add2, ybtn1 + add2_1, wbtn1, hbtn1)
        self.button6.clicked.connect(self.new_window6)

        self.button7 = QtWidgets.QPushButton(self)
        self.button7.setText("Close")
        self.button7.setGeometry(xbtn1 + 3 * add2, ybtn1 + add2_1, wbtn1, hbtn1)
        self.button7.clicked.connect(self.new_window7)

    def show_chapters(self, chapters="2-5"):
        """ Updates the icons and dropdown menus based on a block of chapters (chapters) """

        chapters = chapters.split("-")
        chapter_numbers = list(range(int(chapters[0]), int(chapters[1]) + 1))
        chapter_functions = [self.chapter2, self.chapter3, self.chapter4, self.chapter5, self.chapter6, self.chapter7,
                             self.chapter8, self.chapter9, self.chapter10, self.chapter11, self.chapter12, self.chapter13,
                             self.chapter14, self.chapter15, self.chapter16, self.chapter17, self.chapter18, self.chapter19,
                             self.chapter20, self.chapter21]

        idx = 0
        for icon in [self.icon1, self.icon2, self.icon3, self.icon4]:
            icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_{}.svg".format(chapter_numbers[idx])).pixmap(
                w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
            icon.setGeometry(xL_g1, yL_g1 + idx * add_l, wL_g1, hL_g1)
            idx += 1

        idx = 0
        for label_box, comboBox in zip([self.label_box1, self.label_box2, self.label_box3, self.label_box4],
                                       [self.comboBox1, self.comboBox2, self.comboBox3, self.comboBox4]):
            label_box.setText(BOOK1_CHAPTERS_DEMOS[chapter_numbers[idx]][0])
            if comboBox.connected:
                comboBox.currentIndexChanged.disconnect()
            comboBox.clear()
            comboBox.addItems(BOOK1_CHAPTERS_DEMOS[chapter_numbers[idx]][1:])
            comboBox.currentIndexChanged.connect(chapter_functions[chapter_numbers[idx] - 2])
            comboBox.connected = True
            idx += 1

    def chapter2(self, idx):
        if idx == 1:
            self.chapter_window1 = OneInputNeuron()
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = two_input_neuron()

    def chapter3(self, idx):
        if idx == 1:
            self.myOtherWindow = perceptron_classification()
        if idx == 2:
            self.myOtherWindow1 = hamming_classification()
        elif idx == 3:
            self.myOtherWindow1 = hopfield_classification()

    def chapter4(self, idx):
        if idx == 1:
            self.chapter_window1 = DecisionBoundaries()
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = PerceptronRule()
            self.chapter_window2.show()

    def chapter5(self, idx):
        if idx == 1:
            self.myOtherWindow = gram_schmidt()
        elif idx == 2:
            self.myOtherWindow1 = reciprocal_basis()

    def chapter6(self, idx):
        print("TODO")

    def chapter7(self, idx):
        print("TODO")

    def chapter8(self, idx):
        print("TODO")

    def chapter9(self, idx):
        if idx == 1:
            self.chapter_window1 = SteepestDescentQuadratic()
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = ComparisonOfMethods()
            self.chapter_window2.show()
        elif idx == 3:
            print("TODO")
        elif idx == 4:
            print("TODO")

    def chapter10(self, idx):
        print("TODO")

    def chapter11(self, idx):
        if idx == 1:
            self.chapter_window1 = NetworkFunction()
            self.chapter_window1.show()
        elif idx == 2:
            print("TODO")
        elif idx == 3:
            self.chapter_window4 = FunctionApproximation()
            self.chapter_window4.show()
        elif idx == 4:
            print("TODO")

    def chapter12(self, idx):
        print("TODO")

    def chapter13(self, idx):
        print("TODO")

    def chapter14(self, idx):
        print("TODO")

    def chapter15(self, idx):
        print("TODO")

    def chapter16(self, idx):
        print("TODO")

    def chapter17(self, idx):
        print("TODO")

    def chapter18(self, idx):
        print("TODO")

    def chapter19(self, idx):
        print("TODO")

    def chapter20(self, idx):
        print("TODO")

    def chapter21(self, idx):
        print("TODO")

    @staticmethod
    def new_window6():
        print("TODO")

    @staticmethod
    def new_window7():
        print("TODO")
