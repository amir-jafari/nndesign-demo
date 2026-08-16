import numpy as np

from nndesigndemos.book2.chapter3.Gradient_descent_base import GradientDescentBase


class GradientDescent(GradientDescentBase):
    """Batch (steepest descent) gradient descent on the ADALINE error surface."""

    title = "Gradient Descent"
    max_epoch = 100

    def gradient_step(self, x):
        # Negative gradient of the sum squared error over all four patterns
        grad = np.zeros(2)
        for select in range(self.hh.shape[1]):
            p = self.hh[:, select]
            e = self.t[select] - x @ p
            grad += 2 * e * p
        return grad
