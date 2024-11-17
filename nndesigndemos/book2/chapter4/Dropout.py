import random

from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH

from nndesigndemos.book2.chapter4.DropoutDir.newmultilay import newmultilay
from nndesigndemos.book2.chapter4.DropoutDir.trainscg0 import trainscg0
from nndesigndemos.book2.chapter4.DropoutDir.simnet import simnet
from nndesigndemos.book2.chapter4.DropoutDir.softmax0 import softmax0
from nndesigndemos.book2.chapter4.DropoutDir.crossentr import crossentr
from nndesigndemos.book2.chapter4.DropoutDir.tansig0 import tansig0

import math
import numpy as np
import time
from nndesigndemos.book2.chapter4.DropoutDir.getx import getx
from nndesigndemos.book2.chapter4.DropoutDir.newtr import newtr
from nndesigndemos.book2.chapter4.DropoutDir.get_do_mask import get_do_mask
from nndesigndemos.book2.chapter4.DropoutDir.calcperf0 import calcperf0
from nndesigndemos.book2.chapter4.DropoutDir.setx import setx
from nndesigndemos.book2.chapter4.DropoutDir.calcgx0 import calcgx0
from nndesigndemos.book2.chapter4.DropoutDir.cliptr import cliptr

class Dropout(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(Dropout, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Early Stopping", 13, "Use the slider to change the\nNoise Standard Deviation of\nthe training points.\n\n"
                                                "Click [Train] to train\non the training points.\n\nThe training and validation\n"
                                                "performance indexes will be\npresented on the right.\n\nYou will notice that\n"
                                                "without early stopping\nthe validation error\nwill increase.",
                          PACKAGE_PATH + "Logo/Logo_Ch_13.svg", None, description_coords=(535, 120, 450, 300))

        self.max_epoch = 10
        self.T = 2
        self.pp0 = np.linspace(-1, 1, 201)
        self.tt0 = np.sin(2 * np.pi * self.pp0 / self.T)

        self.pp = np.linspace(-0.95, 0.95, 20)
        self.p = np.linspace(-1, 1, 21)

        self.make_plot(1, (100, 90, 300, 300))
        self.make_plot(2, (100, 380, 300, 300))

        self.train_error, self.error_train = [], None
        self.ani_1 = None
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.S1, self.random_state = 20, 42
        # np.random.seed(self.random_state)
        self.tt, self.t = None, None

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Function", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-1, 1)
        self.axes_1.set_ylim(-1.5, 1.5)
        self.axes_1.plot(self.pp0, np.sin(2 * np.pi * self.pp0 / self.T))
        self.net_approx, = self.axes_1.plot([], linestyle="--")
        self.train_points, = self.axes_1.plot([], marker='*', label="Train", linestyle="")
        self.test_points, = self.axes_1.plot([], marker='.', label="Test", linestyle="")
        self.axes_1.legend()
        self.canvas.draw()

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Performance Indexes", fontdict={'fontsize': 10})
        self.train_e, = self.axes_2.plot([], [], linestyle='-', color="b", label="train error")
        self.axes_2.legend()
        # self.axes_2.plot([1, 1])
        self.axes_2.plot(1, 100, marker="*")
        self.axes_2.plot(300, 100, marker="*")
        self.axes_2.plot(1, 0.01, marker="*")
        self.axes_2.plot(300, 0.01, marker="*")
        # self.axes_2.set_xscale("log")
        # self.axes_2.set_yscale("log")
        # self.axes_2.set_xlim(1, 100)
        # self.axes_2.set_ylim(0.1, 1000)
        # self.axes_2.set_xticks([1, 10, 100])
        # self.axes_2.set_yticks([0.1, 0, 10, 100, 1000])
        # for line in self.axes_2.lines:
        #     line.remove()
        self.figure2.set_tight_layout(True)
        self.canvas2.draw()

        self.nsd = 1
        self.make_slider("slider_nsd", QtCore.Qt.Orientation.Horizontal, (0, 30), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
                         (self.x_chapter_usual, 410, self.w_chapter_slider, 100), self.slide,
                         "label_nsd", "Noise standard deviation: 1.0", (self.x_chapter_usual + 10, 380, self.w_chapter_slider, 100))

        self.animation_speed = 100

        self.plot_train_test_data()

        self.run_button = QtWidgets.QPushButton("Train", self)
        self.run_button.setStyleSheet("font-size:13px")
        self.run_button.setGeometry(self.x_chapter_button * self.w_ratio, 490 * self.h_ratio, self.w_chapter_button * self.w_ratio, self.h_chapter_button * self.h_ratio)
        self.run_button.clicked.connect(self.on_run)


        self.make_button("pause_button", "Pause", (self.x_chapter_button, 520, self.w_chapter_button, self.h_chapter_button), self.on_stop)
        self.pause = True

        self.init_params()
        self.full_batch = False

    def testTrainSCG(self):
        net = newmultilay({
            'f': [tansig0, softmax0],
            'R': 2,
            'S': [300, 2],
            'Init': 'xav',
            'perf': crossentr,
            'do': [0.95, 1],
            'doflag': 0
        })

        # Set standard deviation for noise
        stdv = 0.3

        # Training data (inputs Pd and targets Tl)
        Pd = np.array([
            [0.2, 0.2, 0, 0, -0.35, -0.35, -0.5, 0, 0.25, 0, -0.25, 0, 0.25, -0.15, -0.15, 0.1, 0.1],
            [-0.75, 0.75, 0.65, -0.65, -0.45, 0.45, 0, -0.5, 0.5, 0.25, 0, -0.25, -0.5, 0.2, -0.2, 0.3, -0.3]
        ])
        Tl = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        # Adding noise to input data
        Pd = np.hstack([Pd] + [Pd + stdv * (np.random.rand(*Pd.shape) - 0.5) for _ in range(6)])
        Tl = np.hstack([Tl for _ in range(7)])

        # Train the network using the SCG algorithm (placeholder)
        net['trainParam'] = {
            'epochs': 300,
            'show': 25,
            'goal': 0,
            'max_time': float('inf'),
            'min_grad': 1.0e-6,
            'max_fail': 5,
            'sigma': 5.0e-5,
            'lambda': 5.0e-7
        }

        # Below is the original trainscg0 function
        """
        Scaled conjugate gradient backpropagation algorithm for neural network training.

        Parameters:
        net : object
            Neural network object with trainable parameters and properties.
        Pd : array-like
            Delayed input vectors.
        Tl : array-like
            Target layer vectors.
        VV : object, optional
            Validation vectors, used to stop training early if validation performance degrades.
        TV : object, optional
            Test vectors for evaluating generalization performance.

        Returns:
        net : object
            Trained neural network.
        tr : dict
            Training record over epochs (perf, vperf, tperf, alphak, deltak).
        """
        this = 'TRAINSCG'
        epochs = net['trainParam']['epochs']
        show = net['trainParam']['show']
        goal = net['trainParam']['goal']
        max_time = net['trainParam']['max_time']
        min_grad = net['trainParam']['min_grad']
        sigma = net['trainParam']['sigma']
        lambda_param = net['trainParam']['lambda']

        # Initialize
        flag_stop = 0
        stop = ''
        startTime = time.time()
        X = getx(net)
        num_X = len(X)
        dropout = 0
        for i in net['do']:
            if i != 1:
                dropout = 1

        tr = newtr(epochs, 'perf', 'vperf', 'tperf', 'alphak', 'deltak')  # Initialize training record
        success = 1
        lambdab = 0
        lambdak = lambda_param

        for epoch in range(0, epochs + 1):
            if epoch == 0:
                if dropout:
                    net['doflag'] = 1
                    net = get_do_mask(net, Pd)

                perf, Ac = calcperf0(net, Pd, Tl)

                gX = calcgx0(net, Pd, Ac, Tl)

                normgX = np.sqrt(np.dot(gX, gX))
                dX = -gX  # Initial search direction
                nrmsqr_dX = np.dot(dX, dX)
                norm_dX = np.sqrt(nrmsqr_dX)

            # Training record and stopping criteria
            currentTime = time.time() - startTime
            tr['perf'][epoch] = perf

            # Stopping Criteria
            if (perf <= goal):
                stop = 'Performance goal met.'
            elif (epoch == epochs):
                stop = 'Maximum epoch reached, performance goal was not met.'
            elif (currentTime > max_time):
                stop = 'Maximum time elapsed, performance goal was not met.'
            elif (normgX < min_grad):
                stop = 'Minimum gradient reached, performance goal was not met.'
            elif flag_stop:
                stop = 'User stop.'

            if math.isfinite(show) and (epoch % show == 0 or stop != ''):
                # Print general information
                print(this, end='')
                if math.isfinite(epochs):
                    print(f', Epoch {epoch}/{epochs}', end='')
                if math.isfinite(max_time):
                    print(f', Time {currentTime / max_time * 100:.2f}%', end='')
                if math.isfinite(goal):
                    func_name = net['perf'].__name__
                    print(f', {func_name} {perf}/{goal}', end='')
                if math.isfinite(min_grad):
                    print(f', Gradient {normgX}/{min_grad}', end='')
                print()  # Print a newline
                flag_stop = False
                # print(tr['epoch'])
                # print("tr['perf']:", tr['perf'])
                if stop:
                    print(f'{this}, {stop}\n\n')
            # flag_stop = plotperf0(tr, goal, this, epoch)  # Assuming plotperf0 is defined

            if stop:
                print('last epoch and break', epoch)
                # self.plot_result(net, Pd, Tl)
                break

            if success == 1:
                sigmak = sigma / norm_dX
                X_temp = X + sigmak * dX
                net_temp = setx(net, X_temp)  # Assuming function to set weights
                _, Ac = calcperf0(net_temp, Pd, Tl)
                gX_temp = calcgx0(net_temp, Pd, Ac, Tl)
                sk = (gX_temp - gX) / sigmak
                deltak = np.dot(dX, sk)

            # Scale deltak and calculate step size
            deltak += (lambdak - lambdab) * nrmsqr_dX

            if deltak <= 0:
                lambdab = 2 * (lambdak - deltak / nrmsqr_dX)
                deltak = -deltak + lambdak * nrmsqr_dX
                lambdak = lambdab

            muk = -np.dot(dX, gX)
            alphak = muk / deltak

            # Parameter update
            X_temp = X + alphak * dX
            net_temp = setx(net, X_temp)
            perf_temp, _ = calcperf0(net_temp, Pd, Tl)
            difk = 2 * deltak * (perf - perf_temp) / (muk ** 2)

            # Update success condition and gradient
            if difk >= 0:
                if dropout:
                    net = get_do_mask(net, Pd)
                    _, Ac = calcperf0(net, Pd, Tl)
                    gX_old = calcgx0(net, Pd, Ac, Tl)
                else:
                    gX_old = gX

                X = X_temp
                net = net_temp

                perf, Ac = calcperf0(net, Pd, Tl)

                gX = calcgx0(net, Pd, Ac, Tl)
                normgX = np.sqrt(np.dot(gX, gX))
                lambdab = 0
                success = 1

                # Update direction for next epoch
                if epoch % num_X == 0:
                    dX = -gX
                else:
                    betak = (np.dot(gX, gX) - np.dot(gX, gX_old)) / muk
                    dX = -gX + betak * dX
                    if dropout:
                        ind1 = np.where(gX == 0)  # Find indices where gX is zero
                        dX[ind1] = 0

                nrmsqr_dX = np.dot(gX, gX)
                norm_dX = np.sqrt(nrmsqr_dX)

                if difk >= 0.75:
                    lambdak = 0.25 * lambdak
            else:
                lambdab = lambdak
                success = 0

            if difk < 0.25:
                lambdak = lambdak + deltak * (1 - difk) / nrmsqr_dX

            # Training records
            tr['alphak'][epoch] = alphak
            tr['deltak'][epoch] = deltak

            print('epoch', epoch, 'perf', tr['perf'][epoch])

            yield tr['perf'][epoch]

    def plot_result(self, net1, P, T):
        # Plot decision boundary
        mx = [1.02, 1.02]
        mn = [-1, -1]
        xlim = [mn[0], mx[0]]
        ylim = [mn[1], mx[1]]

        dx = (mx[0] - mn[0]) / 101
        dy = (mx[1] - mn[1]) / 101
        xpts = np.arange(xlim[0], xlim[1], dx)
        ypts = np.arange(ylim[0], ylim[1], dy)
        X, Y = np.meshgrid(xpts, ypts)

        testInput = np.vstack([X.ravel(), Y.ravel()])
        net1['doflag'] = 0
        testOutputs = simnet(net1, testInput)
        testOutputs = testOutputs[1][0, :] - testOutputs[1][1, :]

        F = testOutputs.reshape(X.shape)

        # Create a contour plot
        plt.figure()
        # plt.contourf(xpts, ypts, F, levels=[0.0, 0.0], colors=['lightblue'])
        plt.contourf(xpts, ypts, F, levels=[-0.1, 0.0, 0.1], colors=['lightblue', 'lightgreen', 'lightyellow'])

        plt.colorbar()

        # Plot points from P
        plt.plot(P[0, :], P[1, :], 'x', label='All P points')

        # Identify indices where T(1,:) is non-zero
        ind = np.nonzero(T[0, :])[0]

        # Plot points with condition T(1, :)
        plt.plot(P[0, ind], P[1, ind], 'or', label='T(1,:) non-zero points')

        # Add reference lines and set axis properties
        plt.plot([-1, 1], [0, 0], 'k')  # Horizontal line
        plt.plot([0, 0], [-1, 1], 'k')  # Vertical line
        plt.axis('square')
        plt.xlabel("xpts")
        plt.ylabel("ypts")

        # Show the plot
        plt.legend()
        plt.show()

    def ani_stop(self):
        if self.ani_1 and self.ani_1.event_source:
            self.ani_1.event_source.stop()

    def ani_start(self):
        if self.ani_1 and self.ani_1.event_source:
            self.ani_1.event_source.start()

    def on_stop(self):
        if self.pause:
            self.ani_stop()
            self.pause_button.setText("Unpause")
            self.pause = False
        else:
            self.ani_start()
            self.pause_button.setText("Pause")
            self.pause = True

    def on_animate_2(self, perf):
        self.error_train = perf
        self.train_error.append(self.error_train)
        self.train_e.set_data(list(range(len(self.train_error))), self.train_error)

        return self.train_e,

    def on_run(self):
        self.pause_button.setText("Pause")
        self.pause = True
        self.init_params()
        self.ani_stop()
        self.net_approx.set_data([], [])
        self.train_error = []
        self.train_e.set_data([], [])
        self.canvas2.draw()

        print('self.max_epoch', self.max_epoch, self.animation_speed)

        self.ani_1 = FuncAnimation(self.figure2, self.on_animate_2, frames=self.testTrainSCG,
                                   interval=self.animation_speed, repeat=False, blit=True)

    def slide(self):
        # if list(self.ani_1.frame_seq):  # If the animation is running
        #     self.slider_nsd.setValue(self.nsd * 10)
            # self.ani_1.event_source.start()
        # else:
        print('slideslideslideslide')
        self.init_params()
        # np.random.seed(self.random_state)
        self.nsd = float(self.slider_nsd.value() / 10)
        self.label_nsd.setText("Noise standard deviation: " + str(self.nsd))
        self.plot_train_test_data()
        # self.animation_speed = int(self.slider_anim_speed.value()) * 100
        # self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
        self.ani_stop()
        self.train_error = []
        self.net_approx.set_data([], [])
        self.canvas.draw()
        self.canvas2.draw()

    def plot_train_test_data(self):
        self.tt = np.sin(2 * np.pi * self.pp / self.T) + np.random.uniform(-2, 2, self.pp.shape) * 0.2 * self.nsd
        self.train_points.set_data(self.pp, self.tt)
        self.t = np.sin(2 * np.pi * self.p / self.T) + np.random.uniform(-2, 2, self.p.shape) * 0.2 * self.nsd
        self.test_points.set_data(self.p, self.t)

    def init_params(self):
        # np.random.seed(self.random_state)
        self.W1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.b1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (1, self.S1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))
