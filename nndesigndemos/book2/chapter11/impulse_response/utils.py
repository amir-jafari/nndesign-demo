import numpy as np
import matplotlib.pyplot as plt


class state_space:
    """
    State space network
    """
    def __init__(self, iw11, lw11, b1, lw21, b2, a_0=[]):
        """
        Initialize network with input and layer weights and biases.

        Args:
            iw11: Input weights (list or array)
            lw11: Feedback layer weights (list or array)
            b1:   First layer bias (list or array)
            lw21: Feedforward layer weights (list or array)
            b1:   Second layer bias (list or array)
            a_0:  Initial state (list or array)
        """
        self.iw11 = np.expand_dims(iw11,axis=1)
        self.lw11 = np.array(lw11)
        self.lw21 = np.array(lw21)
        self.b1 = np.array(b1)
        self.b2 = np.array(b2)


        # Initialize state
        if len(a_0) > 0:
            self.a_0 = np.array(a_0)
        else:
            if len(self.lw11) > 0:
                self.a_0 = np.zeros(self.lw11.shape[0])
            else:
                self.a_0 = np.zeros(0)

    def step(self, p):
        """
        Perform one step

        Args:
            p: Input sample

        Returns:
            a: Output sample
        """
        # Compute first layer output
        p = np.atleast_1d(p)
        a1 =  np.matmul(self.iw11, p) + np.matmul(self.lw11, self.a_0) + self.b1

        # Compute the second layer output
        a2 = np.matmul(self.lw21, a1) + self.b2

        # Update state
        if len(self.a_0) > 0:
            self.a_0 = a1

        return a2

    def process(self, input_sequence):
        """
        Process an entire sequence.

        Args:
            input_sequence: Input sequence (list or array)

        Returns:
            output_sequence: Network output
        """
        output = np.zeros(len(input_sequence)+1)
        # Compute the output at t=0 from the initial condition
        output[0] = np.matmul(self.lw21, self.a_0) + self.b2
        # Compute the remaining outputs. There will be one more output than input.
        for i in range(len(input_sequence)):
            output[i+1] = self.step(input_sequence[i])
        return output


def plot_response(a,t):
    # Args:
    #       a: sequence to be plotted
    #       t: time
    #
    # Make a step plot of the sequence
    markerline, stemlines, baseline = plt.stem(t, a,
        linefmt='b-',             # blue solid line for stems
        markerfmt='bo',           # blue circle markers
        basefmt='r-')             # red baseline

    # Customize stem line thickness
    plt.setp(stemlines, linewidth=2.0)    # make stems thicker
    plt.setp(baseline, linewidth=2.0)     # make baseline thicker

    # Customize marker size
    plt.setp(markerline, markersize=10)   # make markers bigger
    plt.tick_params(axis='both', labelsize=16)
    plt.show()


if __name__ == "__main__":
    lw11 = np.array([[1, -0.24],[1, 0]]).transpose()
    iw11 = np.array([1, 0])
    lw21 = np.array([1, 0]).transpose()
    b1 = np.zeros(2)
    b2 = 0
    a0 = [0, 0]
    p = [1, 0, 0, 0, 0]
    net = state_space(iw11, lw11, b1, lw21, b2, a0)
    a = net.process(p)

    t = np.arange(len(p)+1)
    # Plot the response
    plot_response(a,t)