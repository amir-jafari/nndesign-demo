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
        def __init__(self, axis1, nrows_up, pattern=None):
            self.wid_up = 1
            self.nrows_up = nrows_up
            inbetween_up = 0.1
            self.xx_up = np.arange(0, self.nrows_up, (self.wid_up + inbetween_up))
            self.yy_up = np.arange(0, self.nrows_up, (self.wid_up + inbetween_up))

            # make_label("label_pattern1", "Input Pattern", (75, 105, 150, 50))
            # make_plot(1, (15, 130, 170, 170))
            self.axis1 = axis1

            if pattern is not None:
                self.pattern11 = pattern
                self.response = True
                self.color_lst = color_dic[3]

            else:
                self.pattern11 = np.random.randint(0, 2, size=(self.nrows_up, self.nrows_up))
                self.response = False
                self.color_lst = ['khaki', 'green']

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
        self.n = 8
        self.pattern = self.PatternPlot(self.axis1, self.n)

        # self.wid_up = 1
        # self.nrows_up = 8
        # inbetween_up = 0.1
        # self.xx_up = np.arange(0, self.nrows_up, (self.wid_up + inbetween_up))
        # self.yy_up = np.arange(0, self.nrows_up, (self.wid_up + inbetween_up))
        #

        #
        # self.pattern11 = np.random.randint(0, 2, size=(self.nrows_up, self.nrows_up))
        #
        # for xi in range(len(self.xx_up)):
        #     for yi in range(len(self.yy_up)):
        #         color = "green" if self.pattern11[yi, xi] == 1 else "khaki"
        #         sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.wid_up, fill=True, color=color)
        #         self.axis1.add_patch(sq)
        # self.axis1.axis([-0.1, self.nrows_up + 0.1 * self.nrows_up, -0.1, self.nrows_up + 0.1 * self.nrows_up])
        # self.axis1.axis("off")
        self.canvas.show()
        self.canvas.mpl_connect("button_press_event", self.on_mouseclick1)


        self.make_label("label_pattern2", "Kernel", (235, 105, 150, 50))
        self.make_plot(2, (175, 130, 170, 170))
        self.axis2 = self.figure2.add_axes([0, 0, 1, 1])
        self.m = 3
        self.pattern2 = self.PatternPlot(self.axis2, self.m)
        self.canvas2.show()
        self.canvas2.mpl_connect("button_press_event", self.on_mouseclick2)


        self.make_label("label_pattern5", "Response Pattern", (320, 305, 150, 50))
        self.make_plot(5, (250, 330, 240, 240))
        self.axis5 = self.figure5.add_axes([0, 0, 1, 1])
        self.output = self.n - self.m + 1
        self.pattern5 = self.PatternPlot(self.axis5, self.output, self.calculate())
        self.canvas5.show()

        self.calculate()

        # for xi in range(len(self.xx_up)):
        #     for yi in range(len(self.yy_up)):
        #         if self.pattern11[yi, xi] == 1:
        #             sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.hei_up, fill=True, color="red")
        #         else:
        #             sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.hei_up, fill=True, color="khaki")
        #         self.axis5.add_patch(sq)
        # self.axis5.axis([-0.1, self.ncols_up + 0.5, -0.1, self.nrows_up + 0.6])
        # self.axis5.axis("off")
        # self.canvas5.draw()

    def calculate(self):
        P = self.pattern.pattern11
        kernel = self.pattern2.pattern11

        output = np.zeros((self.output, self.output), dtype=int)

        # Perform the convolution
        for i in range(self.output):
            for j in range(self.output):
                output[i, j] = np.sum(P[i:i + self.m, j:j + self.m] * kernel)

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
        self.on_mouseclick_base(event, self.pattern, self.canvas, self.axis1)

    def on_mouseclick2(self, event):
        self.on_mouseclick_base(event, self.pattern2, self.canvas2, self.axis2)

    def response(self):
        for patch in self.axis5.patches:
            patch.remove()
        number = self.axis5.patches
        print('self.axis5.patches', number)
        new_result = self.calculate()

        # print(pattern.pattern11)
        # print('len(self.axis1.patches)', len(axis1.patches))

        self.pattern5.aaaabbbb(new_result)
        self.canvas5.draw()
        # self.pattern5 = self.PatternPlot(self.axis5, 7, self.calculate())
        # self.canvas5.show()

        # pattern = np.array([self.pattern1, self.pattern2, self.pattern3]).T * 2 - 1
        # p = np.array(self.pattern4).T * 2 - 1
        # if self.rule == 0:
        #     w = np.dot(pattern, pattern.T)
        # elif self.rule == 1:
        #     w = np.dot(pattern, np.linalg.pinv(pattern))
        # a = np.flip(np.dot(w, p).reshape(self.ncols_up, self.nrows_up).T, axis=0)
        # for patch in self.axis5.patches:
        #     patch.remove()
        # for xi in range(len(self.xx_up)):
        #     for yi in range(len(self.yy_up)):
        #         if a[yi, xi] > 0:
        #             sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.hei_up, fill=True, color="red")
        #         else:
        #             sq = patches.Rectangle((self.xx_up[xi], self.yy_up[yi]), self.wid_up, self.hei_up, fill=True, color="khaki")
        #         self.axis5.add_patch(sq)
        # self.canvas5.draw()

# import numpy as np
#
#
# def convolution(P, kernel):
#     # Get the dimensions of the input image and the kernel
#     n = P.shape[0]  # Assuming P is square with shape (n, n)
#     m = kernel.shape[0]  # Assuming kernel is square with shape (m, m)
#
#     # Calculate the size of the output
#     output_size = n - m + 1
#     output = np.zeros((output_size, output_size), dtype=int)
#
#     # Perform the convolution
#     for i in range(output_size):
#         for j in range(output_size):
#             output[i, j] = np.sum(P[i:i + m, j:j + m] * kernel)
#
#     return output
#
#
# # Example usage
# n = 8  # Size of the input image P
# m = 3  # Size of the kernel
#
# # Generate random input image and kernel
# P = np.random.randint(0, 2, size=(n, n))
# kernel = np.random.randint(0, 2, size=(m, m))
#
# # Perform the convolution
# output = convolution(P, kernel)
#
# # Display the result
# print("Input Image (P):")
# print(P)
# print("\nKernel:")
# print(kernel)
# print("\nOutput after Convolution:")
# print(output)
