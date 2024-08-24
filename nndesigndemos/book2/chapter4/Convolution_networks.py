import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.patches as patches

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH

# animation run train blur or focus
# train option. Focus on changes

color_dic = {
    'input': ['khaki', 'green'],
    # 'output': ['#c3b091', '#cba36d', '#d39649', '#dc8926', '#e37c12', '#e96409', '#f04b06', '#f63104', '#fb1602', '#ff0000'],
    'output': ['#c3b091', '#c89f7f', '#cd8e6e', '#d27d5c', '#d76c4b', '#dc5b39', '#e14a28', '#e63916', '#eb2805', '#f01700', '#f30f00', '#f60800', '#f90000', '#fc0000', '#ff0000', '#ff1010', '#ff2020'],

}


def generate_diamond(n):
    diamond = np.zeros((n, n), dtype=int)
    center = n // 2
    if n % 2 == 0:
        for i in range(center):
            left = center - i - 1
            right = center + i
            bottom = n - i - 1

            diamond[left, i] = 1
            diamond[right, i] = 1
            diamond[left, bottom] = 1
            diamond[right, bottom] = 1
    else:
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


def gen_random_matrix(size):
    return np.random.randint(0, 2, size=(size, size))


def gen_shape_matrix(size, idx):
    if idx == 0:
        matrix = generate_diamond(size)
    elif idx == 1:
        matrix = generate_slash(size)
    else:
        matrix = generate_square(size, size-2)

    return matrix


class PatternPlot:
    def __init__(self, axis, matrix, label_on, response_pattern=False):

        self.axis = axis
        self.matrix = matrix
        self.size = len(matrix)
        self.color_lst = color_dic[('output' if response_pattern else 'input')]

        self.wid_up = 1
        inbetween_up = 0.1
        self.xx_up = np.arange(0, self.size+0.1, (self.wid_up + inbetween_up))
        self.yy_up = np.arange(0, self.size+0.1, (self.wid_up + inbetween_up))

        self.label_on = label_on
        self.texts = []
        self.plot(self.matrix)
        self.axis.axis([-0.1, self.size + 0.1 * self.size, -0.1, self.size + 0.1 * self.size])
        self.axis.axis("off")

    def get_size(self):
        return self.size

    def remove_text(self):
        for text in self.texts:
            text.remove()
        self.texts = []

    def add_text(self):
        for xi in range(len(self.xx_up)):
            for yi in range(len(self.yy_up)):
                text = self.axis.text(self.xx_up[xi] + self.wid_up / 2, self.yy_up[yi] + self.wid_up / 2,
                                      str(self.matrix[yi, xi]), color="black", ha='center', va='center', fontsize=12)
                self.texts.append(text)

    def plot(self, matrix):
        self.matrix = matrix

        for xi in range(len(self.xx_up)):
            for yi in range(len(self.yy_up)):
                color = self.color_lst[matrix[yi, xi]]
                sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.wid_up, fill=True,
                                       color=color)
                self.axis.add_patch(sq)

        if self.label_on:
            self.remove_text() # remove old and add new
            self.add_text()

    def label_display(self, label_on):
        self.label_on = label_on
        if self.label_on:
            self.add_text()
        else:
            self.remove_text()

    def remove_patch(self):
        for patch in self.axis.patches:
            patch.remove()


def matrix_size_down(old_matrix, padding_bottom_right, padding_top_left):
    old_len = len(old_matrix)
    matrix = old_matrix[
        padding_bottom_right:old_len - padding_top_left,
        padding_top_left:old_len - padding_bottom_right
    ]
    return matrix


def matrix_size_up(old_matrix, padding_bottom_right, padding_top_left):
    matrix = np.pad(
        old_matrix,
        pad_width=(
            (padding_bottom_right, padding_top_left), # reverse the order because the display is upside down
            (padding_top_left, padding_bottom_right)
        ),
        mode='constant',
        constant_values=0,
    )
    return matrix


class Convol(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(Convol, self).__init__(w_ratio, h_ratio, dpi, main_menu=1)

        self.fill_chapter("Convolution", 4, "Change the input shape, \ninput size and kernel size\nfrom the lists.\n\nUse checkboxs to change\npadding, stride and label\nstatus.\n\n",
                          PACKAGE_PATH + "Logo/Logo_Ch_7.svg", None)

        self.stride = 1
        self.pad_on = False
        self.padding_top_left = 0
        self.padding_bottom_right = 0
        self.label_on = False
        self.shape_idx = 0

        self.make_label("label_pattern1", "Input Pattern", (115, 105, 150, 50))
        self.make_plot(1, (15, 130, 270, 270))
        self.axis1 = self.figure.add_axes([0, 0, 1, 1])
        self.pattern1 = PatternPlot(self.axis1, gen_shape_matrix(6, self.shape_idx), self.label_on)
        self.canvas.show()
        self.canvas.mpl_connect("button_press_event", self.on_mouseclick1)

        self.make_label("label_pattern2", "Kernel", (380, 180, 150, 50))
        self.make_plot(2, (340, 205, 120, 120))
        self.axis2 = self.figure2.add_axes([0, 0, 1, 1])
        self.pattern2 = PatternPlot(self.axis2, gen_random_matrix(2), self.label_on)
        self.canvas2.show()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)

        self.make_label("label_pattern3", "Response Pattern", (210, 405, 150, 50))
        self.make_plot(3, (150, 430, 220, 220))
        self.axis3 = self.figure3.add_axes([0, 0, 1, 1])
        self.pattern3 = PatternPlot(self.axis3, self.get_response_matrix(), self.label_on, True)
        self.canvas3.show()

        # coords meaning: x, y, width, height
        self.make_combobox(1, ['Diamond', 'Slash', 'Square'], (self.x_chapter_usual, 270, self.w_chapter_slider, 100),
                           self.change_input_shape, "label_combobox", "Input Shape", (self.x_chapter_usual + 50, 270, 100, 50))

        self.size1_lst = ['6', '7', '8']
        self.make_combobox(2, self.size1_lst, (self.x_chapter_usual, 335, self.w_chapter_slider, 100),
                           self.change_input_size, "label_combobox", "Input Size", (self.x_chapter_usual + 50, 335, 100, 50))

        self.size2_lst = ['2', '3', '4']
        self.make_combobox(3, self.size2_lst, (self.x_chapter_usual, 400, self.w_chapter_slider, 100),
                           self.change_kernel_size, "label_combobox", "Kernel Size", (self.x_chapter_usual + 50, 400, 100, 50))

        self.make_checkbox('checkbox_pad', 'Padding', (self.x_chapter_usual, 480, self.w_chapter_slider, 40),
                           self.use_pad, False)

        self.make_checkbox('checkbox_stride', 'Stride', (self.x_chapter_usual, 530, self.w_chapter_slider, 40),
                           self.use_stride, False)

        self.make_checkbox('checkbox_label', 'Show Label', (self.x_chapter_usual, 580, self.w_chapter_slider, 40),
                           self.use_label, False)

    def get_response_matrix(self):
        stride = self.stride
        pattern1 = self.pattern1
        pattern2 = self.pattern2
        size1 = pattern1.get_size()
        size2 = pattern2.get_size()

        size3 = (size1 - size2) // stride + 1
        output = np.zeros((size3, size3), dtype=int)

        for i in range(0, size3 * stride, stride):
            for j in range(0, size3 * stride, stride):
                output[i // stride, j // stride] = np.sum(
                    pattern1.matrix[i:i + size2, j:j + size2] * pattern2.matrix
                )

        return output

    def on_mouseclick_base(self, event, pattern, canvas, axis, pattern_idx):
        if event.xdata is not None and event.ydata is not None:
            # print('event', event, 'event.xdata', event.xdata)
            d_x = [abs(event.xdata - xx - 0.5) for xx in pattern.xx_up]
            d_y = [abs(event.ydata - yy - 0.5) for yy in pattern.yy_up]
            xxx, yyy = list(range(len(pattern.xx_up)))[np.argmin(d_x)], list(range(len(pattern.yy_up)))[np.argmin(d_y)]

            pattern.matrix[yyy, xxx] = 1 - pattern.matrix[yyy, xxx]

            new_pattern = self.draw_pattern12(pattern, axis, pattern.matrix, canvas)
            if pattern_idx == 1:
                self.pattern1 = new_pattern
            else:
                self.pattern2 = new_pattern

            self.draw_pattern3()

    def on_mouseclick1(self, event):
        self.on_mouseclick_base(event, self.pattern1, self.canvas, self.axis1, 1)

    def on_mouseclick2(self, event):
        self.on_mouseclick_base(event, self.pattern2, self.canvas2, self.axis2, 2)

    def draw_pattern12(self, pattern, axis, matrix, canvas):
        pattern.remove_text()
        pattern.remove_patch()
        pattern = PatternPlot(axis, matrix, self.label_on)
        canvas.draw()
        return pattern

    def draw_pattern3(self):
        self.pattern3.remove_text()
        self.pattern3.remove_patch()
        self.pattern3 = PatternPlot(self.axis3, self.get_response_matrix(), self.label_on, True)
        self.canvas3.draw()

    def change_input(self, size, shape_idx):
        matrix1 = gen_shape_matrix(size, shape_idx)
        if self.pad_on:
            matrix1 = self.gen_padding_matrix(matrix1, self.pattern2.get_size())

        self.pattern1 = self.draw_pattern12(self.pattern1, self.axis1, matrix1, self.canvas)
        self.draw_pattern3()

    def change_input_shape(self, idx):
        self.shape_idx = idx
        self.change_input(self.pattern1.get_size(), self.shape_idx)

    def change_input_size(self, idx):
        new_size = int(self.size1_lst[idx])
        self.change_input(new_size, self.shape_idx)

    def change_kernel_size(self, idx):
        new_kernel_size = int(self.size2_lst[idx])

        # Changing kernel size causes response pattern size changes. To keep response size
        # as the same as input, we need to recalculate the input if it's in the padding status.
        # Steps: 1.reverse input to no padding status; 2.use new kernel size to pad input again.
        if self.pad_on:
            self.pad_on = False
            matrix1_reverse = self.gen_padding_matrix(self.pattern1.matrix, self.pattern2.get_size())
            self.pad_on = True
            matrix1_new = self.gen_padding_matrix(matrix1_reverse, new_kernel_size)
            self.pattern1 = self.draw_pattern12(self.pattern1, self.axis1, matrix1_new, self.canvas)

        # The following is just boring activity to make the change more elegant.
        # The simplest method to get matrix2 is just one line:
        # matrix2 = gen_random_matrix(new_kernel_size)
        old_matrix = self.pattern2.matrix
        old_kernel_size = self.pattern2.get_size()
        if abs(new_kernel_size-old_kernel_size) == 2:
            padding_top_left = 1
            padding_bottom_right = 1
        elif old_kernel_size == 2 or new_kernel_size == 2:
            padding_top_left = 0
            padding_bottom_right = 1
        else:
            padding_top_left = 1
            padding_bottom_right = 0
        if new_kernel_size > old_kernel_size:
            matrix2 = matrix_size_up(old_matrix, padding_bottom_right, padding_top_left)
        else:
            matrix2 = matrix_size_down(old_matrix, padding_bottom_right, padding_top_left)

        self.pattern2 = self.draw_pattern12(self.pattern2, self.axis2, matrix2, self.canvas2)

        self.draw_pattern3()

    def gen_padding_matrix(self, old_matrix, kernel_size):
        if self.pad_on:
            self.padding_top_left = (kernel_size - 1) // 2
            self.padding_bottom_right = kernel_size // 2
            matrix = matrix_size_up(old_matrix, self.padding_bottom_right, self.padding_top_left)
        else:
            matrix = matrix_size_down(old_matrix, self.padding_bottom_right, self.padding_top_left)
            self.padding_top_left = 0
            self.padding_bottom_right = 0

        return matrix

    def use_pad(self):
        self.pad_on = True if self.checkbox_pad.checkState().value == 2 else False
        matrix = self.gen_padding_matrix(self.pattern1.matrix, self.pattern2.get_size())

        self.pattern1 = self.draw_pattern12(self.pattern1, self.axis1, matrix, self.canvas)
        self.draw_pattern3()

    def use_stride(self):
        self.stride = self.pattern2.get_size() if self.checkbox_stride.checkState().value == 2 else 1
        self.draw_pattern3()

    def use_label(self):
        self.label_on = True if self.checkbox_label.checkState().value == 2 else False

        self.pattern1.label_display(self.label_on)
        self.pattern2.label_display(self.label_on)
        self.pattern3.label_display(self.label_on)

        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
