import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout, open_link
from nndesigndemos.get_package_path import PACKAGE_PATH


class Cheatsheet5(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(Cheatsheet5, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Early Stopping", 5, "Use the slider to change the\nNoise Standard Deviation of\nthe training points.\n\n"
                                                "Click [Train] to train\non the training points.",
                          PACKAGE_PATH + "Logo/Logo_Ch_13.svg", None, description_coords=(535, 40, 450, 300))

        root_path = PACKAGE_PATH + 'book2/chapter5/cheatsheets/'
        all_files = [
            'Scipy-Linalgebra.pdf',
            'Functions.pdf',
            'Classes.pdf',
            'Pands2.pdf',
            'Bokeh.pdf',
            'Pands1.pdf',
            'Matplotlib.pdf',
            'List.pdf',
            'Code Debug.pdf',
            'Django.pdf',
            'Jupyter Notebook.pdf',
            'Numpy.pdf',
            'If While.pdf',
            'Data Structure 3.pdf',
            'Data Structure 2.pdf',
            'Matplotlib2.pdf',
            'Data Structure 1.pdf',
            'Base Python.pdf',
            'Open-WriteFiles .pdf',
            'Dictionaries.pdf'
        ]

        for i, file_name in enumerate(all_files):
            label_str = f'<a href="file://{root_path}{file_name}">{i}. {file_name}</a>'
            label_attr_name = f"book{i}_link"
            self.make_label(label_attr_name, label_str, (40, 100 + 25 * i, 200, 50))
            label = getattr(self, label_attr_name)
            label.linkActivated.connect(open_link)
