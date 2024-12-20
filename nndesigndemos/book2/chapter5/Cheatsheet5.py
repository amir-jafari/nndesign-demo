from pathlib import Path
from urllib.parse import urljoin
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

        relative_dir = "./nndesigndemos/book2/chapter5/cheatsheets/"

        all_files = [
            'Jupyter Notebook.pdf',
            'Base Python.pdf',
            'If While.pdf',
            'List.pdf',
            'Dictionaries.pdf',
            'Open-WriteFiles .pdf',
            'Functions.pdf',
            'Classes.pdf',
            'Code Debug.pdf',
            'Scipy-Linalgebra.pdf',
            'Numpy.pdf',
            'Pands1.pdf',
            'Pands2.pdf',
            'Matplotlib.pdf',
            'Matplotlib2.pdf',
            'Bokeh.pdf',
            'Django.pdf',
            'Data Structure 1.pdf',
            'Data Structure 2.pdf',
            'Data Structure 3.pdf',
        ]

        for i, file_name in enumerate(all_files):
            absolute_path = Path(relative_dir + file_name).resolve()
            file_uri = urljoin("file:///", str(absolute_path).replace("\\", "/"))

            label_str = f'<a href="{file_uri}">{i}. {file_name}</a>'
            label_attr_name = f"book{i}_link"
            self.make_label(label_attr_name, label_str, (40, 100 + 25 * i, 200, 50))
            label = getattr(self, label_attr_name)
            label.linkActivated.connect(open_link)
