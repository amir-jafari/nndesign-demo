from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import math
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from get_package_path import PACKAGE_PATH

# -------------------------------------------------------------------------------------------------------------
xm =500; ym= 150; wm = 900; hm =800;

xlabel =80; ylabel= 5; wlabel = 500; hlabel =100; add =20;
xtabel =560; ytabel=25 ; wtabel =500 ; htabel =100;
xautor = 700; yautor= 715; wautor = 500; hautor=100;

xcm1 =250; ycm1= 140; wcm1 = 350; hcm1 =20; add1 = 140; subt=20;
xbtn1 =150; ybtn1= 740; wbtn1 = 60; hbtn1=20; add2 = 80; add2_1 = 30;

xl1 =10; yl1= 90; wl1 = 700; hl1 =90;
xl2 =700; yl2= 780; wl2 = 900; hl2 =780;

w_Logo1 = 100;h_Logo1 = 80; xL_g1 = 150; yL_g1= 110; wL_g1= w_Logo1; hL_g1=h_Logo1; add_l = 140;


w_Logom = 200; h_Logom = 100; xL_gm = 80; yL_gm= 140; wL_gm= 3*w_Logo1; hL_gm = h_Logom;
w_Logom1 = 200; h_Logom1 = 100; xL_gm1 = 80; yL_gm1= 450; wL_gm1= 3*w_Logo1; hL_gm1=h_Logom1;

xbtnm =300; ybtnm= 140; wbtnm = 300; hbtnm=50;
xbtnm1 =300; ybtnm1= 470; wbtnm1 = 300; hbtnm1=50;

# -------------------------------------------------------------------------------------------------------------
wp_pic2_1 = 100; hp_pic2_1 = 80; x_pic2_1 = 750; y_pic2_1= 50; w_pic2_1= wp_pic2_1; h_pic2_1=hp_pic2_1;
wp_pic2_2 = 500; hp_pic2_2 = 200; x_pic2_2 = 250; y_pic2_2= 100; w_pic2_2= 500; h_pic2_2=200;

x_info = 710; y_info= 150; w_info= 450; h_info=250;

xl3 = wl1;yl3 = hl1+35;wl3 = wl1;hl3 = 750;
# -------------------------------------------------------------------------------------------------------------


class OneInputNeuron(QMainWindow):
    def __init__(self):
        super(OneInputNeuron, self).__init__()

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
        self.label3.setText("One input neuron")
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.Bold))
        self.label3.setGeometry(xtabel, ytabel, wtabel, htabel)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("Chapter 2")
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry(xautor, yautor, wautor, hautor)

        self.label5 = QtWidgets.QLabel(self)
        self.label5.setText("Alter the weight and bias\n and input by dragging the\n triangular shaped indictors.\n"
                            " \n Pick the transfer function\n with the F menu.\n "
                            "\n Watch the change\n to the  neuron function\n and its  output.")
        self.label5.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label5.setGeometry(x_info, y_info, w_info, h_info)

        self.icon1 = QtWidgets.QLabel(self)
        self.icon1.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg").pixmap(wp_pic2_1, hp_pic2_1, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon1.setGeometry(x_pic2_1, y_pic2_1, w_pic2_1, h_pic2_1)

        self.icon1 = QtWidgets.QLabel(self)
        self.icon1.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Chapters/2/nn2d1.svg").pixmap(wp_pic2_2, hp_pic2_2, QtGui.QIcon.Normal, QtGui.QIcon.On))
        self.icon1.setGeometry(x_pic2_2, y_pic2_2, w_pic2_2, h_pic2_2)

        # ----------------------

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1_functions = [self.purelin, self.hardlim, self.hardlims, self.satlin, self.satlins, self.logsig, self.tansig]
        self.comboBox1.addItems(["Purelin", 'Hardlim', 'Hardlims', 'Satlin', 'Satlins', 'LogSig', 'TanSig'])
        self.func1 = self.purelin

        self.label_f = QtWidgets.QLabel(self)
        self.label_f.setText("f")
        self.label_f.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_f.setGeometry(775, 550, 150, 100)

        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("b")
        self.label_b.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_b.setGeometry(775, 470, 150, 100)

        self.label_w = QtWidgets.QLabel(self)
        self.label_w.setText("w")
        self.label_w.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        self.label_w.setGeometry(775, 400, 150, 100)

        self.slider_w = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_w.setRange(-3, 3)
        self.slider_w.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_w.setTickInterval(1)
        self.slider_w.setValue(1)

        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setRange(-3, 3)
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(1)
        self.slider_b.setValue(0)

        self.wid = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid.setGeometry(10, 300, 680, 500)
        self.layout.addWidget(self.canvas)
        self.wid.setLayout(self.layout)

        self.wid2 = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid2.setGeometry(710, 580, 150, 100)
        self.layout2.addWidget(self.comboBox1)
        self.wid2.setLayout(self.layout2)

        self.wid3 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid3.setGeometry(710, 430, 150, 100)
        self.layout3.addWidget(self.slider_w)
        self.wid3.setLayout(self.layout3)

        self.wid4 = QtWidgets.QWidget(self)
        self.layout4 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid4.setGeometry(710, 500, 150, 100)
        self.layout4.addWidget(self.slider_b)
        self.wid4.setLayout(self.layout4)

        self.comboBox1.currentIndexChanged.connect(self.change_transfer_function)
        self.slider_w.valueChanged.connect(self.graph)
        self.slider_b.valueChanged.connect(self.graph)

        self.graph()

    def graph(self):

        a = self.figure.add_subplot(1, 1, 1)
        a.clear()  # Clear the plot

        weight = self.slider_w.value()
        bias = self.slider_b.value()
        self.label_w.setText("w: " + str(weight))
        self.label_b.setText("b: " + str(bias))
        p = np.arange(-4, 4, 0.1)
        func = np.vectorize(self.func1)
        out = func(np.dot(weight, p) + bias)

        a.plot(p, out, markersize=3, color="red")
        # Setting limits so that the point moves instead of the plot.
        a.set_xlim(-4, 4)
        a.set_ylim(-2, 2)
        # add grid and axes
        a.grid(True, which='both')
        a.axhline(y=0, color='k')
        a.axvline(x=0, color='k')
        self.canvas.draw()

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions[idx]
        self.graph()

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
    def purelin(x):
        return x

    @staticmethod
    def satlin(x):
        if x < 0:
            return 0;
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
