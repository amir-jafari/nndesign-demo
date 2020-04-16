from PyQt5 import QtWidgets, QtGui
from functools import partial

# -------- Global imports ----------------
from nndesign_layout import NNDLayout
from get_package_path import PACKAGE_PATH

# ----------------------------------------------------- Book 1 ---------------------------------------------------------
# ------ Chapter 2 --------
from One_input_neuron import OneInputNeuron
# from One_input_neuron_new import OneInputNeuron
from Two_input_neuron import TwoInputNeuron
# ------ Chapter 3 --------
from Perceptron_classification import PerceptronClassification
from Hamming_classification import HammingClassification
from Hopfield_classification import HopfieldClassification
# ------ Chapter 4 --------
from Perceptron_rule import PerceptronRule
from Decision_boundary import DecisionBoundaries
# ------ Chapter 5 --------
from Gram_Schmidt import GramSchmidt
from Reciprocal_basis import ReciprocalBasis
# ------ Chapter 6 --------
from Linear_transformations import LinearTransformations
from Eigenvector_game import EigenvectorGame
# ------ Chapter 7 --------
from Supervised_Hebb import SupervisedHebb
# ------ Chapter 8 --------
from Taylor_series_1 import TaylorSeries1
from Taylor_series_2 import TaylorSeries2
from Directional_derivatives import DirectionalDerivatives
from Quadratic_function import QuadraticFunction
# ------ Chapter 9 --------
from Steepest_descent_quadratic import SteepestDescentQuadratic
from Comparison_of_methods import ComparisonOfMethods
from Newtons_method import NewtonsMethod
from Steepest_descent import SteepestDescent
# ------ Chapter 10 --------
from Adaptive_noise_cancellation import AdaptiveNoiseCancellation
from EEG_noise_cancellation import EEGNoiseCancellation
from Linear_classification import LinearClassification
# ------ Chapter 11 -------
from Function_approximation import FunctionApproximation
# from Backpropagation_calculation import BackpropagationCalculation
from Network_function import NetworkFunction
from Generalization import Generalization
# ------ Chapter 12 -------
from Steepest_descent_backprop_1 import SteepestDescentBackprop1
from Steepest_descent_backprop_2 import SteepestDescentBackprop2
from Momentum import Momentum
from Variable_learning_rate import VariableLearningRate
from Conjugate_gradient_line_search import ConjugateGradientLineSearch
from Conjugate_gradient import ConjugateGradient
from Marquardt_step import MarquardtStep
from Marquardt import Marquardt
# ------ Chapter 13 -------
from Early_stopping import EarlyStopping
from Regularization import Regularization
from Bayesian_regularization import BayesianRegularization
from Early_stoppping_regularization import EarlyStoppingRegularization
# ------ Chapter 14 -------
from FIR_network import FIRNetwork
from IIR_network import IIRNetwork
from Dynamic_derivatives import DynamicDerivatives
from Recurrent_network_training import RecurrentNetworkTraining
# ------ Chapter 15 -------
from Effects_of_decay_rate import EffectsOfDecayRate
from Graphical_instar import GraphicalInstar
# ------ Chapter 16 -------
from Competitive_classification import CompetitiveClassification
from Competitive_learning import CompetitiveLearning
from OneD_feature_map import OneDFeatureMap
from TwoD_feature_map import TwoDFeatureMap
from Lvq1 import LVQ1
from Lvq2 import LVQ2
# ------ Chapter 17 -------
from Network_function_radial import NetworkFunctionRadial
from Pattern_classification import PatternClassification
from Linear_least_squares import LinearLeastSquares
from Orthogonal_least_squares import OrthogonalLeastSquares
# ------ Chapter 18 -------
from Leaky_integrator import LeakyIntegrator
from Shunting_network import ShuntingNetwork
from Grossberg_layer_1 import GrossbergLayer1
from Grossberg_layer_2 import GrossbergLayer2
from Adaptive_weights import AdaptiveWeights
# ------ Chapter 19 -------
from Art1_layer1 import ART1Layer1
from Art1_layer2 import ART1Layer2
from Orienting_subsystem import OrientingSubsystem
from Art1_algorithm import ART1Algorithm
# ------ Chapter 20 -------
from Dynamical_system import DynamicalSystem
# ------ Chapter 21 -------
from Hopfield_network import HopfieldNetwork

# ----------------------------------------------------- Book 2 ---------------------------------------------------------
# ------ Chapter 2 --------
from Poslin_network_function import PoslinNetworkFunction
from Poslin_decision_regions import PoslinDecisionRegions
from Cascaded_function import CascadedFunction
# ------ Chapter 3 --------
from Gradient_descent import GradientDescent
from Gradient_descent_stochastic import GradientDescentStochastic


# -------------------------------------------------------------------------------------------------------------
xlabel, ylabel, wlabel, hlabel, add = 120, 5, 500, 100, 20
xautor, yautor = 180, 600

xcm1, ycm1, wcm1, hcm1, add1, subt = 340, 140, 250, 20, 140, 20
xcm2 = 333
xbtn1, ybtn1, wbtn1, hbtn1, add2 = 10, 635, 60, 30, 65

w_Logo1, h_Logo1, xL_g1, yL_g1, add_l = 100, 80, 100, 110, 140

BOOK1_CHAPTERS_DEMOS = {
    2: ["Neuron Model & Network Architecture:", "Chapter 2 demos", "One-input Neuron", "Two-input Neuron"],
    3: ["An Illustrative Example", "Chapter 3 demos", "Perceptron classification", "Hamming classification", "Hopfield classification"],
    4: ["Perceptron Learning Rule", "Chapter 4 demos", "Decision boundaries", "Perceptron rule"],
    5: ["Signal & weight Vector Spaces", "Chapter 5 demos", "Gram schmidt", "Reciprocal basis"],
    6: ["Linear Transformations For N. Networks", "Chapter 6 demos", "Linear transformations", "Eigenvector game"],
    7: ["Supervised Hebb", "Chapter 7 demos", "Supervised Hebb"],
    8: ["Performance Surfaces & Optimum Points", "Chapter 8 demos", "Taylor series #1", "Taylor series #2", "Directional derivatives", "Quadratic function"],
    9: ["Performance Optimization", "Chapter 9 demos", "Steepest descent for Quadratic", "Method comparison", "Newton's method", "Steepest descent"],
    10: ["Widrow - Hoff Learning", "Chapter 10 demos", "Adaptive noise cancellation", "EEG noise cancellation", "Linear classification"],
    # 11: ["Backpropagation", "Chapter 11 demos", "Network Function", "Backpropagation Calculation", "Function Approximation", "Generalization"],
    11: ["Backpropagation", "Chapter 11 demos", "Network Function", "Function Approximation", "Generalization"],
    12: ["Variations on Backpropagation", "Chapter 12 demos", "Steepest Descent #1", "Steepest Descent #2", "Momentum", "Variable Learning Rate", "CG Line Search", "Conjugate Gradient", "Marquardt Step", "Marquardt"],
    13: ["Generalization", "Chapter 13 demos", "Early Stopping", "Regularization", "Bayesian Regularization", "Early Stopping-Regularization"],
    14: ["Dynamic Networks", "Chapter 14 demos", "FIR Network", "IIR Network", "Dynamic Derivatives", "Recurrent Network Training"],
    15: ["Associative Learning", "Chapter 15 demos", "Unsupervised Hebb", "Effects of Decay Rate", "Hebb with Decay", "Graphical Instar", "Outstar"],
    16: ["Competitive Networks", "Chapter 16 demos", "Competitive Classification", "Competitive Learning", "1-D Feature Map", "2-D Feature Map", "LVQ 1", "LVQ 2"],
    17: ["Radial Basis Function", "Chapter 17 demos", "Network Function Radial", "Pattern Classification", "Linear Least Squares", "Orthogonal Least Squares", "Non Linear Optimization"],
    18: ["Grossberg Network", "Chapter 18 demos", "Leaky Integrator", "Shunting Network", "Grossberg Layer 1", "Grossberg Layer 2", "Adaptive Weights"],
    19: ["Adaptive Resonance Theory", "Chapter 19 demos", "ART1 Layer 1", "ART1 Layer 2", "Orienting Subsystem", "ART1 Algorithm"],
    20: ["Stability", "Chapter 20 demos", "Dynamical System"],
    21: ["Hopfield Network", "Chapter 21 demos", "Hopfield Network"]
}

BOOK2_CHAPTERS_DEMOS = {
    2: ["Multilayer Networks", "Chapter 2 demos", "Poslin Network Function", "Poslin Decision Regions", "Poslin Decision Regions 2D", "Poslin Decision Regions 3D", "Cascaded Function"],
    3: ["Multilayer Network Train", "Chapter 3 demos", "Gradient Descent", "Gradient Descent Stochastic"]
}
# -------------------------------------------------------------------------------------------------------------


class MainWindowNN(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        """ Main Window for the Neural Network Design Book. Inherits basic layout from NNDLayout """
        super(MainWindowNN, self).__init__(w_ratio, h_ratio, chapter_window=False, main_menu=1, draw_vertical=False, create_plot=False)

        self.make_label("label_3", "Table of Contents", (380, ylabel + add, wlabel, hlabel), font_size=18)
        self.make_label("label4", "By Hagan, Demuth, Beale, Jafari", (self.wm - xautor, yautor, wlabel, hlabel))

        # ---- Chapter icons and dropdown menus ----

        self.chapter_window1, self.chapter_window2, self.chapter_window3, self.chapter_window4 = None, None, None, None
        self.chapter_window5, self.chapter_window6, self.chapter_window7, self.chapter_window8 = None, None, None, None

        self.icon1 = QtWidgets.QLabel(self)
        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.connected = False  # Need to create this attribute so that we don't have more than one connected function
        self.comboBox1.setGeometry(self.wm - xcm1 * self.w_ratio, ycm1 * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box1 = QtWidgets.QLabel(self)
        self.label_box1.setGeometry(self.wm - xcm2 * self.w_ratio, (ycm1 - subt) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box1.setFont(QtGui.QFont("Times New Roman", 14))  # , 14 * (self.w_ratio + self.h_ratio) / 2))

        self.icon2 = QtWidgets.QLabel(self)
        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.connected = False
        self.comboBox2.setGeometry(self.wm - xcm1 * self.w_ratio, (ycm1 + add1) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box2 = QtWidgets.QLabel(self)
        self.label_box2.setGeometry(self.wm - xcm2 * self.w_ratio, (ycm1 + add1 - subt) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box2.setFont(QtGui.QFont("Times New Roman", 14))

        self.icon3 = QtWidgets.QLabel(self)
        self.comboBox3 = QtWidgets.QComboBox(self)
        self.comboBox3.connected = False
        self.comboBox3.setGeometry(self.wm - xcm1 * self.w_ratio, (ycm1 + 2 * add1) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box3 = QtWidgets.QLabel(self)
        self.label_box3.setGeometry(self.wm - xcm2 * self.w_ratio, (ycm1 + 2 * add1 - subt) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box3.setFont(QtGui.QFont("Times New Roman", 14))

        self.icon4 = QtWidgets.QLabel(self)
        self.comboBox4 = QtWidgets.QComboBox(self)
        self.comboBox4.connected = False
        self.comboBox4.setGeometry(self.wm - xcm1 * self.w_ratio, (ycm1 + 3 * add1) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box4 = QtWidgets.QLabel(self)
        self.label_box4.setGeometry(self.wm - xcm2 * self.w_ratio, (ycm1 + 3 * add1 - subt) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box4.setFont(QtGui.QFont("Times New Roman", 14))

        self.show_chapters()

        # ---- Buttons at the bottom to switch between blocks of chapters ----

        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setText("2-5")
        self.button1.setGeometry(xbtn1 * self.w_ratio, ybtn1 * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button1.clicked.connect(partial(self.show_chapters, "2-5"))

        self.button2 = QtWidgets.QPushButton(self)
        self.button2.setText("6-9")
        self.button2.setGeometry((xbtn1 + add2) * self.w_ratio, ybtn1 * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button2.clicked.connect(partial(self.show_chapters, "6-9"))

        self.button3 = QtWidgets.QPushButton(self)
        self.button3.setText("10-13")
        self.button3.setGeometry((xbtn1 + 2 * add2) * self.w_ratio, ybtn1 * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button3.clicked.connect(partial(self.show_chapters, "10-13"))

        self.button4 = QtWidgets.QPushButton(self)
        self.button4.setText("14-17")
        self.button4.setGeometry((xbtn1 + 3 * add2) * self.w_ratio, ybtn1 * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button4.clicked.connect(partial(self.show_chapters, "14-17"))

        self.button5 = QtWidgets.QPushButton(self)
        self.button5.setText("18-21")
        self.button5.setGeometry((xbtn1 + 4 * add2) * self.w_ratio, ybtn1 * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button5.clicked.connect(partial(self.show_chapters, "18-21"))

        """self.button6 = QtWidgets.QPushButton(self)
        self.button6.setText("Textbook Info")
        self.button6.setGeometry((xbtn1 + add2) * self.w_ratio, (ybtn1 + add2_1) * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button6.clicked.connect(self.new_window6)

        self.button7 = QtWidgets.QPushButton(self)
        self.button7.setText("Close")
        self.button7.setGeometry((xbtn1 + 3 * add2) * self.w_ratio, (ybtn1 + add2_1) * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button7.clicked.connect(self.new_window7)"""

        self.center()

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
            """if "2" in chapters and idx == 0:
                print(chapters, idx)
                icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/2_1.svg".format(chapter_numbers[idx])).pixmap(
                    w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
            elif "13" in chapters and idx == 3:
                # icon.setPixmap(QtGui.QPixmap(PACKAGE_PATH + "Logo/13.svg").scaled(w_Logo1 * self.w_ratio, h_Logo1 * self.h_ratio))
                icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/13.svg".format(chapter_numbers[idx])).pixmap(
                    w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
            elif "14" in chapters and idx == 0:
                icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/14.svg".format(chapter_numbers[idx])).pixmap(
                    w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
                # icon.setPixmap(QtGui.QPixmap(PACKAGE_PATH + "Logo/14.svg").scaled(w_Logo1 * self.w_ratio, h_Logo1 * self.h_ratio))
            else:
                icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_{}.svg".format(chapter_numbers[idx])).pixmap(
                    w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
                # icon.setGeometry(xL_g1, yL_g1 + idx * add_l, w_Logo1, h_Logo1)"""
            icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/book_logos/{}.svg".format(chapter_numbers[idx])).pixmap(
                w_Logo1 * self.w_ratio, h_Logo1 * self.h_ratio, QtGui.QIcon.Normal, QtGui.QIcon.On))
            icon.setGeometry(xL_g1 * self.w_ratio, (yL_g1 + idx * add_l) * self.h_ratio, w_Logo1 * self.w_ratio, h_Logo1 * self.h_ratio)
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
        self.comboBox1.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = OneInputNeuron(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = TwoInputNeuron(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()

    def chapter3(self, idx):
        self.comboBox2.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = PerceptronClassification(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        if idx == 2:
            self.chapter_window2 = HammingClassification(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = HopfieldClassification(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()

    def chapter4(self, idx):
        self.comboBox3.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = DecisionBoundaries(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = PerceptronRule(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()

    def chapter5(self, idx):
        self.comboBox4.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = GramSchmidt(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = ReciprocalBasis(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()

    def chapter6(self, idx):
        self.comboBox1.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = LinearTransformations(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = EigenvectorGame(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()

    def chapter7(self, idx):
        self.comboBox2.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = SupervisedHebb(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()

    def chapter8(self, idx):
        self.comboBox3.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = TaylorSeries1(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = TaylorSeries2(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = DirectionalDerivatives(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = QuadraticFunction(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()

    def chapter9(self, idx):
        self.comboBox4.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = SteepestDescentQuadratic(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = ComparisonOfMethods(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = NewtonsMethod(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = SteepestDescent(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()

    def chapter10(self, idx):
        self.comboBox1.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = AdaptiveNoiseCancellation(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = EEGNoiseCancellation(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = LinearClassification(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()

    def chapter11(self, idx):
        self.comboBox2.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = NetworkFunction(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        # elif idx == 2:
        #     self.chapter_window2 = BackpropagationCalculation(self.w_ratio, self.h_ratio)
        #     self.chapter_window2.show()
        elif idx == 2:
            self.chapter_window3 = FunctionApproximation(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 3:
            self.chapter_window4 = Generalization(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()

    def chapter12(self, idx):
        self.comboBox3.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = SteepestDescentBackprop1(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = SteepestDescentBackprop2(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = Momentum(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = VariableLearningRate(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()
        elif idx == 5:
            self.chapter_window5 = ConjugateGradientLineSearch(self.w_ratio, self.h_ratio)
            self.chapter_window5.show()
        elif idx == 6:
            self.chapter_window6 = ConjugateGradient(self.w_ratio, self.h_ratio)
            self.chapter_window6.show()
        elif idx == 7:
            self.chapter_window7 = MarquardtStep(self.w_ratio, self.h_ratio)
            self.chapter_window7.show()
        elif idx == 8:
            self.chapter_window8 = Marquardt(self.w_ratio, self.h_ratio)
            self.chapter_window8.show()

    def chapter13(self, idx):
        self.comboBox4.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = EarlyStopping(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = Regularization(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = BayesianRegularization(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = EarlyStoppingRegularization(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()

    def chapter14(self, idx):
        if idx == 1:
            self.chapter_window1 = FIRNetwork(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = IIRNetwork(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = DynamicDerivatives(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = RecurrentNetworkTraining(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()
        self.comboBox1.setCurrentIndex(0)

    def chapter15(self, idx):
        self.comboBox2.setCurrentIndex(0)
        if idx == 2:
            self.chapter_window2 = EffectsOfDecayRate(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 4:
            self.chapter_window4 = GraphicalInstar(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()
        else:
            print("TODO")

    def chapter16(self, idx):
        self.comboBox3.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = CompetitiveClassification(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = CompetitiveLearning(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = OneDFeatureMap(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = TwoDFeatureMap(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()
        elif idx == 5:
            self.chapter_window5 = LVQ1(self.w_ratio, self.h_ratio)
            self.chapter_window5.show()
        elif idx == 6:
            self.chapter_window6 = LVQ2(self.w_ratio, self.h_ratio)
            self.chapter_window6.show()

    def chapter17(self, idx):
        self.comboBox4.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = NetworkFunctionRadial(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = PatternClassification(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = LinearLeastSquares(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = OrthogonalLeastSquares(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()
        else:
            print("TODO")

    def chapter18(self, idx):
        self.comboBox1.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = LeakyIntegrator(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = ShuntingNetwork(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = GrossbergLayer1(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = GrossbergLayer2(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()
        elif idx == 5:
            self.chapter_window5 = AdaptiveWeights(self.w_ratio, self.h_ratio)
            self.chapter_window5.show()

    def chapter19(self, idx):
        self.comboBox2.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = ART1Layer1(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = ART1Layer2(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
        elif idx == 3:
            self.chapter_window3 = OrientingSubsystem(self.w_ratio, self.h_ratio)
            self.chapter_window3.show()
        elif idx == 4:
            self.chapter_window4 = ART1Algorithm(self.w_ratio, self.h_ratio)
            self.chapter_window4.show()

    def chapter20(self, idx):
        self.comboBox3.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = DynamicalSystem(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()

    def chapter21(self, idx):
        self.comboBox4.setCurrentIndex(0)
        if idx == 1:
            self.chapter_window1 = HopfieldNetwork(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()

    @staticmethod
    def new_window6():
        print("TODO")

    @staticmethod
    def new_window7():
        print("TODO")


class MainWindowDL(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        """ Main Window for the Neural Network Design - Deep Learning Book. Inherits basic layout from NNDLayout """
        super(MainWindowDL, self).__init__(w_ratio, h_ratio, chapter_window=False, main_menu=2, draw_vertical=False, create_plot=False)

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText("Table of Contents")
        self.label3.setFont(QtGui.QFont("Times New Roman", 14, QtGui.QFont.StyleItalic))
        self.label3.setGeometry(self.wm - xlabel * self.w_ratio, (ylabel + add) * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText("By Hagan, Jafari")
        self.label4.setFont(QtGui.QFont("Times New Roman", 12, QtGui.QFont.Bold))
        self.label4.setGeometry(self.wm - 100 * self.w_ratio, 580 * self.h_ratio, wlabel * self.w_ratio, hlabel * self.h_ratio)

        # ---- Chapter icons and dropdown menus ----

        self.chapter_window1, self.chapter_window2, self.chapter_window3, self.chapter_window4, self.chapter_window5 = None, None, None, None, None

        self.icon1 = QtWidgets.QLabel(self)
        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.connected = False  # Need to create this attribute so that we don't have more than one connected function
        self.comboBox1.setGeometry(self.wm - xcm1 * self.w_ratio, ycm1 * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box1 = QtWidgets.QLabel(self)
        self.label_box1.setGeometry(self.wm - xcm2 * self.w_ratio, (ycm1 - subt) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)

        self.icon2 = QtWidgets.QLabel(self)
        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.connected = False
        self.comboBox2.setGeometry(self.wm - xcm1 * self.w_ratio, (ycm1 + add1) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)
        self.label_box2 = QtWidgets.QLabel(self)
        self.label_box2.setGeometry(self.wm - xcm2 * self.w_ratio, (ycm1 + add1 - subt) * self.h_ratio, wcm1 * self.w_ratio, hcm1 * self.h_ratio)

        self.show_chapters()

        # ---- Buttons at the bottom to switch between blocks of chapters ----

        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setText("2-3")
        self.button1.setGeometry(xbtn1 * self.w_ratio, ybtn1 * self.h_ratio, wbtn1 * self.w_ratio, hbtn1 * self.h_ratio)
        self.button1.clicked.connect(partial(self.show_chapters, "2-3"))

        self.center()

    def show_chapters(self, chapters="2-3"):
        """ Updates the icons and dropdown menus based on a block of chapters (chapters) """

        chapters = chapters.split("-")
        chapter_numbers = list(range(int(chapters[0]), int(chapters[1]) + 1))
        chapter_functions = [self.chapter2, self.chapter3]

        idx = 0
        for icon in [self.icon1, self.icon2]:  # TODO: Change logo path when we have them
            icon.setPixmap(QtGui.QIcon(PACKAGE_PATH + "Logo/Logo_Ch_{}.svg".format(chapter_numbers[idx])).pixmap(
                w_Logo1, h_Logo1, QtGui.QIcon.Normal, QtGui.QIcon.On))
            # icon.setGeometry(xL_g1, yL_g1 + idx * add_l, w_Logo1, h_Logo1)
            icon.setGeometry(xL_g1 * self.w_ratio, (yL_g1 + idx * add_l) * self.h_ratio, w_Logo1 * self.w_ratio, h_Logo1 * self.h_ratio)
            idx += 1

        idx = 0
        for label_box, comboBox in zip([self.label_box1, self.label_box2],
                                       [self.comboBox1, self.comboBox2]):
            label_box.setText(BOOK2_CHAPTERS_DEMOS[chapter_numbers[idx]][0])
            if comboBox.connected:
                comboBox.currentIndexChanged.disconnect()
            comboBox.clear()
            comboBox.addItems(BOOK2_CHAPTERS_DEMOS[chapter_numbers[idx]][1:])
            comboBox.currentIndexChanged.connect(chapter_functions[chapter_numbers[idx] - 2])
            comboBox.connected = True
            idx += 1

    def chapter2(self, idx):
        if idx == 1:
            self.chapter_window1 = PoslinNetworkFunction(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window2 = PoslinDecisionRegions(self.w_ratio, self.h_ratio)
            self.chapter_window2.show()
            print("TODO")
        elif idx == 3:
            print("TODO")
        elif idx == 4:
            print("TODO")
        elif idx == 5:
            self.chapter_window5 = CascadedFunction(self.w_ratio, self.h_ratio)
            self.chapter_window5.show()

    def chapter3(self, idx):
        if idx == 1:
            self.chapter_window1 = GradientDescent(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
        elif idx == 2:
            self.chapter_window1 = GradientDescentStochastic(self.w_ratio, self.h_ratio)
            self.chapter_window1.show()
