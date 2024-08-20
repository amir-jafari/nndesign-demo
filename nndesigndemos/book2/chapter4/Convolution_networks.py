import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.patches as patches

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH

color_dic = {
    2: ['khaki', 'green', 'olive', 'orange', 'red'],
    3: ['#c3b091', '#cba36d', '#d39649', '#dc8926', '#e37c12', '#e96409', '#f04b06', '#f63104', '#fb1602', '#ff0000'],
}


class Convol(NNDLayout):
    class PatternPlot:
        def __init__(self, axis, nrows_up, pattern=None, response=False):
            self.wid_up = 1
            self.nrows_up = nrows_up
            inbetween_up = 0.1
            self.xx_up = np.arange(0, self.nrows_up, (self.wid_up + inbetween_up))
            self.yy_up = np.arange(0, self.nrows_up, (self.wid_up + inbetween_up))

            self.axis1 = axis
            self.pattern11 = pattern if pattern is not None else np.random.randint(0, 2, size=(self.nrows_up, self.nrows_up))
            self.response = response
            self.color_lst = color_dic[3] if response else ['khaki', 'green']

            self.texts = []
            self.aaaabbbb(self.pattern11)
            self.axis1.axis([-0.1, self.nrows_up + 0.1 * self.nrows_up, -0.1, self.nrows_up + 0.1 * self.nrows_up])
            self.axis1.axis("off")

        def aaaabbbb(self, pattern11):
            if self.response:
                for text in self.texts:
                    text.remove()
                self.texts = []

            for xi in range(len(self.xx_up)):
                for yi in range(len(self.yy_up)):
                    # color = "green" if pattern11[yi, xi] == 1 else "khaki"
                    color = self.color_lst[pattern11[yi, xi]]
                    sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.wid_up, fill=True,
                                           color=color)
                    self.axis1.add_patch(sq)

                    if self.response:
                        text = self.axis1.text(self.xx_up[xi] + self.wid_up / 2, self.yy_up[yi] + self.wid_up / 2,
                                    str(pattern11[yi, xi]), color="black", ha='center', va='center', fontsize=12)
                        self.texts.append(text)

        def remove_patch(self):
            for patch in self.axis1.patches:
                patch.remove()

        def remove_text(self):
            for text in self.axis1.texts:
                text.remove()


    def __init__(self, w_ratio, h_ratio, dpi):
        super(Convol, self).__init__(w_ratio, h_ratio, dpi, main_menu=1)

        self.fill_chapter("Convolution", 4, "Choose the kernel shape \n"
                                            "from the list.\n\n"
                                                "Choose the input pattern"
                                            "\nsize from the list.\n\n",
                          PACKAGE_PATH + "Logo/Logo_Ch_7.svg", None)

        self.make_label("label_pattern1", "Input Pattern", (75, 105, 150, 50))
        self.make_plot(1, (15, 130, 170, 170))
        self.axis1 = self.figure.add_axes([0, 0, 1, 1])
        self.size1 = 6
        self.pattern1 = self.PatternPlot(self.axis1, self.size1)
        self.canvas.show()
        self.canvas.mpl_connect("button_press_event", self.on_mouseclick1)

        self.make_label("label_pattern2", "Kernel", (235, 105, 150, 50))
        self.make_plot(2, (175, 130, 170, 170))
        self.axis2 = self.figure2.add_axes([0, 0, 1, 1])
        self.size2 = 2
        self.pattern2 = self.PatternPlot(self.axis2, self.size2)
        self.canvas2.show()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)

        self.make_label("label_pattern3", "Response Pattern", (320, 305, 150, 50))
        self.make_plot(3, (250, 330, 240, 240))
        self.axis3 = self.figure3.add_axes([0, 0, 1, 1])
        self.output = self.size1 - self.size2 + 1
        self.pattern3 = self.PatternPlot(self.axis3, self.output, self.calculate(), True)
        self.canvas3.show()
        print('self.axis2', self.axis2)

        # Shape of the feature map
        # size of input
        # size of kernel
        # Stripe
        # Padding
        # animation run train blur or focus
        # No number and numbers options
        # train option. Focus on changes

        self.make_combobox(1, ['Diamond', 'Slash', 'Square'], (self.x_chapter_usual, 300, self.w_chapter_slider, 100),
                           self.change_shape, "label_combobox", "Learning Rule", (self.x_chapter_usual + 50, 300, 100, 50))

        # x, y, width, height
        self.size1_lst = ['6', '7', '8']
        self.make_combobox(2, self.size1_lst, (self.x_chapter_usual, 380, self.w_chapter_slider, 100),
                           self.change_input_size, "label_combobox", "Learning Rule2", (self.x_chapter_usual + 50, 380, 100, 50))

        self.size2_lst = ['2', '3', '4']
        self.make_combobox(3, self.size2_lst, (self.x_chapter_usual, 460, self.w_chapter_slider, 100),
                           self.change_kernel_size, "label_combobox", "Learning Rule3", (self.x_chapter_usual + 50, 460, 100, 50))

        self.batch_norm = True
        self.make_checkbox('checkbox_batch_norm', 'Padding', (self.x_chapter_usual, 500, self.w_chapter_slider, 100),
                           self.use_pad, True)

        self.batch_norm = True
        self.make_checkbox('checkbox_batch_norm', 'Stride', (self.x_chapter_usual, 500, self.w_chapter_slider, 100),
                           self.use_stride, True)


    def calculate(self):
        P = self.pattern1.pattern11
        kernel = self.pattern2.pattern11

        output = np.zeros((self.output, self.output), dtype=int)

        # Perform the convolution
        for i in range(self.output):
            for j in range(self.output):
                output[i, j] = np.sum(P[i:i + self.size2, j:j + self.size2] * kernel)

        return output

    def on_mouseclick_base(self, event, pattern, canvas, axis1):
        if event.xdata != None and event.xdata != None:
            print('event', event, 'event.xdata', event.xdata)
            d_x = [abs(event.xdata - xx - 0.5) for xx in pattern.xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in pattern.yy_up]
            xxx, yyy = list(range(len(pattern.xx_up)))[np.argmin(d_x)], list(range(len(pattern.yy_up)))[np.argmin(d_y)]

            pattern.pattern11[yyy, xxx] = 1 - pattern.pattern11[yyy, xxx]
            position = xxx * pattern.nrows_up + yyy

            new_color = "green" if pattern.pattern11[yyy, xxx] == 1 else "khaki"
            axis1.patches[position].set_facecolor(new_color)

            canvas.draw()

            self.response()

    def on_mouseclick1(self, event):
        self.on_mouseclick_base(event, self.pattern1, self.canvas, self.axis1)

    def on_mouseclick2(self, event):
        self.on_mouseclick_base(event, self.pattern2, self.canvas2, self.axis2)

    def response(self):
        for patch in self.axis3.patches:
            patch.remove()
        number = self.axis3.patches
        print('self.axis3.patches', number)
        new_result = self.calculate()

        # print(pattern.pattern11)
        # print('len(self.axis1.patches)', len(axis1.patches))

        self.pattern3.aaaabbbb(new_result)
        self.canvas3.draw()
        # self.pattern3 = self.PatternPlot(self.axis3, 7, self.calculate())
        # self.canvas3.show()

    def change_shape(self, idx):
        if idx == 0:
            pattern = generate_diamond(self.size1)
        elif idx == 1:
            pattern = generate_slash(self.size1)
        else:
            pattern = generate_square(self.size1, self.size1-2)

        self.pattern1.remove_text()
        self.pattern1.remove_patch()

        self.pattern1 = self.PatternPlot(self.axis1, self.size1, pattern)
        self.canvas.draw()

        self.pattern3.remove_text()
        self.pattern3.remove_patch()
        self.output = self.size1 - self.size2 + 1
        self.pattern3 = self.PatternPlot(self.axis3, self.output, self.calculate(), True)
        self.canvas3.draw()

    # def change_size_base(self, idx, size_lst):
    #     new_size = int(size_lst[idx])

    def change_input_size(self, idx):
        self.size1 = int(self.size1_lst[idx])
        self.pattern1.remove_text()
        self.pattern1.remove_patch()

        self.pattern1 = self.PatternPlot(self.axis1, self.size1)
        self.canvas.draw()

        self.pattern3.remove_text()
        self.pattern3.remove_patch()
        self.output = self.size1 - self.size2 + 1
        self.pattern3 = self.PatternPlot(self.axis3, self.output, self.calculate(), True)
        self.canvas3.draw()

    def change_kernel_size(self, idx):
        self.size2 = int(self.size2_lst[idx])
        self.pattern2.remove_text()
        self.pattern2.remove_patch()

        self.pattern2 = self.PatternPlot(self.axis2, self.size2)
        self.canvas2.draw()

        self.pattern3.remove_text()
        self.pattern3.remove_patch()
        self.output = self.size1 - self.size2 + 1
        self.pattern3 = self.PatternPlot(self.axis3, self.output, self.calculate(), True)
        self.canvas3.draw()

    def use_pad(self):
        if self.checkbox_batch_norm.checkState().value == 2 and not self.batch_norm:
            self.batch_norm = True
            self.graph()
        if self.checkbox_batch_norm.checkState().value == 0 and self.batch_norm:
            self.batch_norm = False
            self.graph()

    def use_stride(self):
        pass


def generate_diamond(n):
    diamond = np.zeros((n, n), dtype=int)
    center = n // 2
    for i in range(n):
        for j in range(n):
            if abs(i - center) + abs(j - center) == center:
                diamond[i, j] = 1
    return diamond


def generate_square(n, size):
    square = np.zeros((n, n), dtype=int)
    start = (n - size) // 2
    end = start + size

    # Create the boundary
    square[start:end, start] = 1  # Left side
    square[start:end, end - 1] = 1  # Right side
    square[start, start:end] = 1  # Top side
    square[end - 1, start:end] = 1  # Bottom side

    return square

def generate_slash(n):
    slash = np.zeros((n, n), dtype=int)
    for i in range(n):
        slash[i, n - i - 1] = 1
    return slash