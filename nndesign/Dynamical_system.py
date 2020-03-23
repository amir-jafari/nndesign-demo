from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.integrate import ode

from nndesign_layout import NNDLayout

from get_package_path import PACKAGE_PATH


wid_up = 1
hei_up = 1.04
nrows_up = 4
ncols_up = 4
inbetween_up = 0.12
xx_up = np.arange(0, ncols_up, (wid_up + inbetween_up))
yy_up = np.arange(0, nrows_up, (hei_up + inbetween_up))


class DynamicalSystem(NNDLayout):
    def __init__(self, w_ratio, h_ratio):
        super(DynamicalSystem, self).__init__(w_ratio, h_ratio, main_menu=1, create_plot=False)

        self.fill_chapter("Dynamical System", 20, "TODO",
                          PACKAGE_PATH + "Chapters/2/Logo_Ch_2.svg", PACKAGE_PATH + "Chapters/2/nn2d1.svg", show_pic=False)

        self.figure8 = Figure()
        self.canvas8 = FigureCanvas(self.figure8)
        self.toolbar8 = NavigationToolbar(self.canvas8, self)
        self.wid8 = QtWidgets.QWidget(self)
        self.layout8 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid8.setGeometry(250 * self.w_ratio, 130 * self.h_ratio,
                              270 * self.w_ratio, 200 * self.h_ratio)
        self.layout8.addWidget(self.canvas8)
        self.wid8.setLayout(self.layout8)
        self.axis8 = self.figure8.add_subplot(1, 1, 1)
        self.axis8.set_title("Pendulum Energy")
        self.axis8.set_xlabel("Time (sec)")
        self.axis8.set_ylabel("Energy")
        self.axis8.set_xlim(0, 800)
        self.axis8.set_xticks([0, 200, 400, 600, 800], [0, 10, 20, 30, 40])
        self.axis8.set_ylim(0, 500)
        # self.axis8.set_yscale("log")
        self.energy, self.energy_ = [], []
        self.energy_plot, = self.axis8.plot([], color="red")

        # --

        self.figure9 = Figure()
        self.canvas9 = FigureCanvas(self.figure9)
        self.toolbar9 = NavigationToolbar(self.canvas9, self)
        self.wid9 = QtWidgets.QWidget(self)
        self.layout9 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid9.setGeometry(5 * self.w_ratio, 130 * self.h_ratio,
                              250 * self.w_ratio, 200 * self.h_ratio)
        self.layout9.addWidget(self.canvas9)
        self.wid9.setLayout(self.layout9)
        self.axis9 = self.figure9.add_subplot(1, 1, 1, polar=True)
        self.axis9.set_rmax(2)
        # self.axis9.grid(False)
        self.axis9.set_rticks([])
        self.axis9.set_theta_zero_location("S")
        self.pendulum_line, = self.axis9.plot([], color="black", linewidth=2)
        self.pendulum_point, = self.axis9.plot([], color="red", marker="*", markersize=10)
        self.angle, self.velocity = 2.1, -1.45
        self.draw_pendulum()
        # line = self.axis9.plot(theta, r)[0]
        # tick = [self.axis9.get_rmax(), self.axis9.get_rmax() * 0.97]
        # for t in np.deg2rad(np.arange(0, 360, 10)):
        #    self.axis9.plot([t, t], tick, lw=0.72, color="k")
        # line.remove()
        self.ani, self.ani2, self.ani3, self.r, self.dt, self.t_end = None, None, None, None, 0.05, 40
        self.canvas9.mpl_connect('button_press_event', self.on_mouseclick)
        self.canvas9.draw()

        # --

        self.figure10 = Figure()
        self.canvas10 = FigureCanvas(self.figure10)
        self.toolbar10 = NavigationToolbar(self.canvas10, self)
        self.wid10 = QtWidgets.QWidget(self)
        self.layout10 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        self.wid10.setGeometry(10 * self.w_ratio, 350 * self.h_ratio, 500 * self.w_ratio, 250 * self.h_ratio)
        self.layout10.addWidget(self.canvas10)
        self.wid10.setLayout(self.layout10)
        self.axis10 = self.figure10.add_subplot(1, 1, 1)
        self.axis10.set_title("Energy Contour")
        self.axis10.set_xlabel("Angle (rad)")
        self.axis10.set_ylabel("Velocity (rad/s)")
        self.axis10.set_xlim(-15, 15)
        self.axis10.set_ylim(-3, 3)
        xx = np.arange(-15, 15, 0.01)
        yy = np.arange(-3, 3, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        EE = 0.5 * 9.8 ** 2 * YY ** 2 + 9.81 * 9.8 * (1 - np.cos(XX))
        self.axis10.contour(XX, YY, EE)
        self.energy_initial, = self.axis10.plot([], marker="*", color="red")
        self.energy_path, = self.axis10.plot([], color="red")
        self.energy_path_data_x, self.energy_path_data_y = [], []
        self.draw_energy_path()
        self.canvas10.draw()

        # --

        self.run_button = QtWidgets.QPushButton("Go", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 520 * self.h_ratio,
                                    self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.run_animation)

    def on_mouseclick(self, event):
        # d_angle_click_pendulum_point = abs(self.angle - (event.xdata + 90) * np.pi / 180)
        # d_r_click_pendulum_point = abs(1.51 - event.ydata)
        # if (d_angle_click_pendulum_point + d_r_click_pendulum_point) / 2 < 0.05:
        #     print("!")
        if self.ani:
            self.ani.event_source.stop()
        if self.ani2:
            self.ani2.event_source.stop()
        if self.ani3:
            self.ani3.event_source.stop()
        self.angle, self.velocity = event.xdata, 0
        self.draw_pendulum()
        self.energy_plot.set_data([], [])
        self.energy_initial.set_data([self.angle], [self.velocity])
        self.energy_path.set_data([], [])
        self.canvas8.draw()
        self.canvas9.draw()
        self.canvas10.draw()

    def run_animation(self):
        if self.ani:
            self.ani.event_source.stop()
        if self.ani2:
            self.ani2.event_source.stop()
        if self.ani3:
            self.ani3.event_source.stop()
        self.ani = FuncAnimation(self.figure9, self.on_animate, init_func=self.animate_init,
                                 frames=800, interval=0, repeat=False, blit=False)
        self.ani2 = FuncAnimation(self.figure8, self.on_animate_2, init_func=self.animate_init_2,
                                  frames=800, interval=0, repeat=False, blit=True)
        self.ani3 = FuncAnimation(self.figure10, self.on_animate_3, init_func=self.animate_init_3,
                                  frames=800, interval=0, repeat=False, blit=True)
        self.canvas9.draw()
        self.canvas8.draw()
        self.canvas10.draw()

    def draw_pendulum(self):
        r = np.arange(0, 1.5, 0.01)
        theta = [self.angle] * len(r)
        self.pendulum_line.set_data(theta, r)
        self.pendulum_point.set_data(theta[0], r[-1] + 0.01)

    def draw_energy(self):
        self.energy.append(0.5 * 9.8 ** 2 * self.velocity ** 2 + 9.81 * 9.8 * (1 - np.cos(self.angle)))
        self.energy_plot.set_data(range(len(self.energy)), self.energy)

    def draw_energy_path(self):
        self.energy_path_data_x.append(self.angle)
        self.energy_path_data_y.append(self.velocity)
        self.energy_initial.set_data(self.energy_path_data_x[0], self.energy_path_data_y[0])
        self.energy_path.set_data(self.energy_path_data_x, self.energy_path_data_y)

    def animate_init(self):
        self.draw_pendulum()
        self.r = ode(self.pendulum_pos_vel).set_integrator("zvode")
        self.r.set_initial_value([self.angle, self.velocity], 0)
        return self.pendulum_line, self.pendulum_point

    def animate_init_2(self):
        self.energy, self.energy_ = [], []
        self.draw_energy()
        return self.energy_plot,

    def animate_init_3(self):
        self.energy_path_data_x, self.energy_path_data_y = [], []
        self.draw_energy_path()
        return self.energy_path, self.energy_initial

    def on_animate(self, idx):
        if self.r.successful() and self.r.t < self.t_end:
            self.angle, self.velocity = self.r.integrate(self.r.t + self.dt)
        self.draw_pendulum()
        return self.pendulum_line, self.pendulum_point

    def on_animate_2(self, idx):
        self.draw_energy()
        return self.energy_plot,

    def on_animate_3(self, idx):
        self.draw_energy_path()
        return self.energy_path, self.energy_initial

    def pendulum_pos_vel(self, t, y):
        return [y[1], -np.sin(y[0]) - 0.2 * y[1]]
