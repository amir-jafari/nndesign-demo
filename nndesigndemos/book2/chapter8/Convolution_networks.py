import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.patches as patches

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from PyQt6 import QtWidgets, QtCore

# To do:
# try draw a box. apply the changes to the image
# https://www.color-hex.com

KERNEL_SIZE_MAX = 6


# Dynamically generate color ranges for the response pattern.
def interpolate_colors(start_hex, end_hex, steps):
    # Convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Convert RGB to hex
    def rgb_to_hex(rgb_color):
        return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)

    # Calculate the difference
    r_diff = (end_rgb[0] - start_rgb[0]) / steps
    g_diff = (end_rgb[1] - start_rgb[1]) / steps
    b_diff = (end_rgb[2] - start_rgb[2]) / steps

    # Generate the colors
    colors = []
    for i in range(steps + 1):
        r = int(start_rgb[0] + (r_diff * i))
        g = int(start_rgb[1] + (g_diff * i))
        b = int(start_rgb[2] + (b_diff * i))
        colors.append(rgb_to_hex((r, g, b)))

    return colors


def pick_items_with_intervals(lst, num_items):
    indices = np.linspace(0, len(lst) - 1, num_items, dtype=int)
    return [lst[i] for i in indices]


color_dic = {
    'input': ['khaki', 'green'],
    'output': interpolate_colors('#f8f7e2', '#2c2d2a', KERNEL_SIZE_MAX * KERNEL_SIZE_MAX),  # Generate color ranges
}


def generate_diamond(n):
    diamond = gen_zero_matrix(n)
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
    square = gen_zero_matrix(n)
    start = (n - size) // 2
    end = start + size

    # Create the boundary
    square[start:end, start] = 1  # Left side
    square[start:end, end - 1] = 1  # Right side
    square[start, start:end] = 1  # Top side
    square[end - 1, start:end] = 1  # Bottom side

    return square


def generate_slash(n):
    slash = gen_zero_matrix(n)
    for i in range(n):
        slash[i, i] = 1
    return slash


def gen_random_matrix(size):
    return np.random.randint(0, 2, size=(size, size))


def gen_zero_matrix(size):
    return np.zeros((size, size), dtype=int)


def gen_shape_matrix(size, idx):
    if idx == 0:
        matrix = generate_diamond(size)
    elif idx == 1:
        matrix = generate_square(size, size - 2)
    elif idx == 2:
        matrix = gen_random_matrix(size)
    elif idx == 3:
        matrix = gen_zero_matrix(size)
    else:
        raise Exception('Not possible')

    return matrix


class PatternPlot:
    def __init__(self, axis, matrix, label_on, response_pattern=False, kernel_size=None):

        self.axis = axis
        self.matrix = matrix
        self.size = len(matrix)
        if response_pattern:
            response_color_range = kernel_size * kernel_size + 1
            self.color_lst = pick_items_with_intervals(color_dic['output'], response_color_range)
        else:
            self.color_lst = color_dic['input']

        self.wid_up = 1
        inbetween_up = 0.1
        self.xx_up = np.arange(0, self.size * 1.1, (self.wid_up + inbetween_up))
        self.yy_up = np.arange(0, self.size * 1.1, (self.wid_up + inbetween_up))

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
            self.remove_text()  # remove old and add new
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
            (padding_bottom_right, padding_top_left),  # reverse the order because the display is upside down
            (padding_top_left, padding_bottom_right)
        ),
        mode='constant',
        constant_values=0,
    )
    return matrix


class Convol(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(Convol, self).__init__(w_ratio, h_ratio, dpi, main_menu=1)

        self.fill_chapter("Convolution", 4,
                          "Change the input shape, \ninput size, kernel size,\nand stride below.\n\nUse checkboxs to change\npadding and value status.\n\nClick input or kernel images\nto change the input pattern\nor kernel pattern.",
                          PACKAGE_PATH + "Logo/Logo_Ch_7.svg", None)

        self.stride = 1
        self.pad_on = False
        self.padding_top_left = 0
        self.padding_bottom_right = 0
        self.label_on = False
        self.shape_idx = 0

        self.kernel_size = 2

        self.make_label("label_pattern1", "Input Pattern", (115, 105, 150, 50))
        self.make_plot(1, (15, 130, 270, 270))
        self.axis1 = self.figure.add_axes([0, 0, 1, 1])
        self.pattern1 = PatternPlot(self.axis1, gen_shape_matrix(6, self.shape_idx), self.label_on)
        self.canvas.show()
        self.canvas.mpl_connect("button_press_event", self.on_mouseclick1)

        self.make_label("label_pattern2", "Kernel", (380, 180, 150, 50))
        self.make_plot(2, (340, 205, 120, 120))
        self.axis2 = self.figure2.add_axes([0, 0, 1, 1])
        self.pattern2 = PatternPlot(self.axis2, generate_slash(self.kernel_size), self.label_on)
        self.canvas2.show()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)

        self.make_label("label_pattern3", "Response Pattern", (210, 405, 150, 50))
        self.make_plot(3, (150, 430, 220, 220))
        self.axis3 = self.figure3.add_axes([0, 0, 1, 1])
        self.pattern3 = PatternPlot(self.axis3, self.get_response_matrix(), self.label_on, True, self.kernel_size)
        self.canvas3.show()

        # coords meaning: x, y, width, height
        self.make_combobox(1, ['Diamond', 'Square', 'Random', 'Custom'],
                           (self.x_chapter_usual, 310, self.w_chapter_slider, 100),
                           self.change_input_shape, "label_combobox", "Input shape",
                           (self.x_chapter_usual + 50, 310, 100, 50))

        self.make_slider("slider_n_input", QtCore.Qt.Orientation.Horizontal, (6, 20), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 6,
                         (self.x_chapter_usual, 400, self.w_chapter_slider, 50), self.change_input_size, "label_n_input",
                         "Input size: 6", (self.x_chapter_usual + 20, 400 - 25, self.w_chapter_slider, 50))

        self.make_slider("slider_n_kernel", QtCore.Qt.Orientation.Horizontal, (2, KERNEL_SIZE_MAX), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 2,
                         (self.x_chapter_usual, 460, self.w_chapter_slider, 50), self.change_kernel_size, "label_n_kernel",
                         "Kernel size: 2", (self.x_chapter_usual + 20, 460 - 25, self.w_chapter_slider, 50))

        self.make_slider("slider_n_strides", QtCore.Qt.Orientation.Horizontal, (1, 3), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 1,
                         (self.x_chapter_usual, 520, self.w_chapter_slider, 50), self.use_stride, "label_n_strides",
                         "Stride(s): 1", (self.x_chapter_usual + 20, 520 - 25, self.w_chapter_slider, 50))

        self.make_checkbox('checkbox_pad', 'Padding', (self.x_chapter_usual, 560, self.w_chapter_slider, 40),
                           self.use_pad, False)

        self.make_checkbox('checkbox_label', 'Show values', (self.x_chapter_usual, 595, self.w_chapter_slider, 40),
                           self.use_label, False)

    def get_response_matrix(self):
        stride = self.stride
        pattern1 = self.pattern1
        pattern2 = self.pattern2
        size1 = pattern1.get_size()
        size2 = pattern2.get_size()

        size3 = (size1 - size2) // stride + 1
        output = gen_zero_matrix(size3)

        matrix1 = pattern1.matrix[::-1]
        matrix2 = pattern2.matrix[::-1]

        for i in range(0, size3 * stride, stride):
            for j in range(0, size3 * stride, stride):
                output[i // stride, j // stride] = np.sum(
                    matrix1[i:i + size2, j:j + size2] * matrix2
                )

        return output[::-1]

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
        self.pattern3 = PatternPlot(self.axis3, self.get_response_matrix(), self.label_on, True, self.kernel_size)
        self.canvas3.draw()

    def change_input(self, size):
        # if self.shape_idx == 3:
        #     matrix2 = gen_zero_matrix(self.pattern2.get_size())
        #     self.pattern2 = self.draw_pattern12(self.pattern2, self.axis2, matrix2, self.canvas2)

        matrix1 = gen_shape_matrix(size, self.shape_idx)
        if self.pad_on:
            matrix1 = self.gen_padding_matrix(matrix1, self.pattern2.get_size())

        self.pattern1 = self.draw_pattern12(self.pattern1, self.axis1, matrix1, self.canvas)
        self.draw_pattern3()

    def change_input_shape(self, idx):
        self.shape_idx = idx
        self.change_input(self.pattern1.get_size())

    def change_input_size(self):
        new_size = self.slider_n_input.value()
        self.label_n_input.setText(f"Input size: {new_size}")
        self.change_input(new_size)

    def change_kernel_size(self):
        self.kernel_size = self.slider_n_kernel.value()
        self.label_n_kernel.setText(f"Kernel size: {self.kernel_size}")

        # Changing kernel size causes response pattern size changes. To keep response size
        # as the same as input, we need to recalculate the input if it's in the padding status.
        # Steps: 1.reverse input to no padding status; 2.use new kernel size to pad input again.
        if self.pad_on:
            self.pad_on = False
            matrix1_reverse = self.gen_padding_matrix(self.pattern1.matrix, self.pattern2.get_size())
            self.pad_on = True
            matrix1_new = self.gen_padding_matrix(matrix1_reverse, self.kernel_size)
            self.pattern1 = self.draw_pattern12(self.pattern1, self.axis1, matrix1_new, self.canvas)

        matrix2 = generate_slash(self.kernel_size)

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
        self.stride = self.slider_n_strides.value()
        self.label_n_strides.setText(f"Stride(s): {self.stride}")
        self.draw_pattern3()

    def use_label(self):
        self.label_on = True if self.checkbox_label.checkState().value == 2 else False

        self.pattern1.label_display(self.label_on)
        self.pattern2.label_display(self.label_on)
        self.pattern3.label_display(self.label_on)

        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
