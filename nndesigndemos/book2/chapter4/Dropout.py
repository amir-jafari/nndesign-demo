from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter4.deephist import deephist


class Dropout(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(Dropout, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)
