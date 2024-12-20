import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout, open_link
from nndesigndemos.get_package_path import PACKAGE_PATH


class Cheatsheet6(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(Cheatsheet6, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Early Stopping", 6, "Use the slider to change the\nNoise Standard Deviation of\nthe training points.\n\n"
                                                "Click [Train] to train\non the training points.",
                          PACKAGE_PATH + "Logo/Logo_Ch_13.svg", None, description_coords=(535, 40, 450, 300))

        relative_path = "./nndesigndemos/book2/chapter6/TensorFlow2Cheatsheet.pdf"
        absolute_path = Path(relative_path).resolve()

        if os.name == "nt":
            absolute_path = str(absolute_path).replace("\\", "/")
            file_uri = f"file:///{absolute_path}"
        else:
            file_uri = f"file://{absolute_path}"

        label_str = f'<a href="{file_uri}">Open TensorFlow Cheatsheet</a>'

        self.make_label("book1_link", label_str, (40, 550, 200, 50))
        self.book1_link.linkActivated.connect(open_link)
