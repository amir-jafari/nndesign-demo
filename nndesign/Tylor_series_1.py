import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


class TylorSeries1(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(TylorSeries1, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False, create_two_plots=True)

        self.fill_chapter("Tylor Series #1", 8, " TODO",
                          PACKAGE_PATH + "Logo/Logo_Ch_5.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("-", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-6, 6)
        self.axes_1.set_ylim(-2, 2)
        self.x_points = np.linspace(-6, 6)
        self.axes_1.plot(self.x_points, np.cos(self.x_points), "-")
        self.axes1_point_draw, = self.axes_1.plot([], 'mo')
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("-", fontdict={'fontsize': 10})
        self.f0, self.f1, self.f2, self.f3, self.f4 = None, None, None, None, None
        self.axes2_point_draw, = self.axes_2.plot([], 'mo')
        self.axes2_function, = self.axes_2.plot([], '-')
        self.axes2_approx_0, = self.axes_2.plot([], 'r-')
        self.axes2_approx_1, = self.axes_2.plot([], 'b-')
        self.axes2_approx_2, = self.axes_2.plot([], 'g-')
        self.axes2_approx_3, = self.axes_2.plot([], 'y-')
        self.axes2_approx_4, = self.axes_2.plot([], 'c-')
        self.axes_2.set_xlim(-6, 6)
        self.axes_2.set_ylim(-2, 2)
        self.canvas2.draw()

        # TODO: Add checkboxes

    def on_mouseclick(self, event):
        if event.xdata != None and event.xdata != None:
            self.axes1_point_draw.set_data([event.xdata], [np.cos(event.xdata)])
            self.axes2_point_draw.set_data([event.xdata], [np.cos(event.xdata)])
            self.canvas.draw()
            self.f0 = np.cos(event.xdata) + np.zeros(self.x_points.shape)
            self.f1 = self.f0 - np.sin(event.xdata) * (self.x_points - event.xdata)
            self.f2 = self.f1 - np.cos(event.xdata) * (self.x_points - event.xdata) ** 2 / 2
            self.f3 = self.f2 + np.sin(event.xdata) * (self.x_points - event.xdata) ** 3 / 6
            self.f4 = self.f3 + np.cos(event.xdata) * (self.x_points - event.xdata) ** 4 / 24
            self.draw_taylor()

    def draw_taylor(self):
        # TODO: Add conditionals on each checkbox
        self.axes2_function.set_data(self.x_points, np.cos(self.x_points))
        self.axes2_approx_0.set_data(self.x_points, self.f0)
        self.axes2_approx_1.set_data(self.x_points, self.f1)
        self.axes2_approx_2.set_data(self.x_points, self.f2)
        self.axes2_approx_3.set_data(self.x_points, self.f3)
        self.axes2_approx_4.set_data(self.x_points, self.f4)
        self.canvas2.draw()
