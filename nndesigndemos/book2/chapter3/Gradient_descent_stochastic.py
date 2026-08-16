import numpy as np

from nndesigndemos.book2.chapter3.Gradient_descent_base import GradientDescentBase


class GradientDescentStochastic(GradientDescentBase):
    """Stochastic gradient descent: each step uses one randomly chosen pattern."""

    title = "Stochastic Gradient Descent"
    max_epoch = 80

    def gradient_step(self, x):
        select = np.random.randint(self.hh.shape[1])
        p = self.hh[:, select]
        e = self.t[select] - x @ p
        return 2 * e * p

    def setup_contour_axes(self):
        self.ax2d.yaxis.tick_right()
