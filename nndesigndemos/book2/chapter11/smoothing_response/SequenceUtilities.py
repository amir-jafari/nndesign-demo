import numpy as np
import matplotlib.pyplot as plt   # MATLAB plotting functions


def plot_poles(poles):
    # Plot poles in the complex plane
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.gca().set_aspect('equal', 'box')

    # Plot unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')

    # Plot poles
    plt.plot(poles.real, poles.imag, 'rx', markersize=10, label='Poles')

    plt.title("Poles in the Complex Plane")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True)


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


class input_output:
    """
    Simple input/output network
    """

    def __init__(self, iw, lw, b, p_tdl=[], a_tdl=[]):
        """
        Initialize network with input and layer weights.

        Args:
            iw: input weights (list or array)
            lw: Layer weights (list or array)
            b: bias
            p_tdl: Previous inputs in TDL (list or array)
            a_tdl: Previous outputs in TDL (list or array)
        """
        self.iw = np.array(iw)
        self.lw = np.array(lw)
        self.b = b

        # Initialize tapped delay lines (previous inputs)
        if len(p_tdl) > 0:
            self.p_tdl = np.array(p_tdl)
        else:
            if len(self.iw) > 0:
                self.p_tdl = np.zeros(len(self.iw) - 1)
            else:
                self.p_tdl = np.zeros(0)
        # Initialize tapped delay lines (previous outputs)
        if len(a_tdl) > 0:
            self.a_tdl = np.array(a_tdl)
        else:
            if len(self.lw) > 0:
                self.a_tdl = np.zeros(len(self.lw))
            else:
                self.a_tdl = np.zeros(0)


    def step(self, p):
        """
        Perform one step

        Args:
            p: Input sample

        Returns:
            a: Output sample
        """
        # Multiply input weight times current input
        a = self.iw[0] * p

        # Multiply input weights times previous inputs in TDL
        for i in range(len(self.p_tdl)):
            a += self.iw[i + 1] * self.p_tdl[i]

        # Multiply layer weights times previous outputs in TDL
        for i in range(len(self.a_tdl)):
            a += self.lw[i] * self.a_tdl[i]

        # Add the bias
        a += self.b

        # Update TDLs
        if len(self.p_tdl) > 0:
            self.p_tdl = np.roll(self.p_tdl, 1)
            self.p_tdl[0] = p

        if len(self.a_tdl) > 0:
            self.a_tdl = np.roll(self.a_tdl, 1)
            self.a_tdl[0] = a

        return a

    def process(self, input_sequence):
        """
        Process an entire sequence.

        Args:
            input_sequence: Input sequence (list or array)

        Returns:
            output_sequence: Network output
        """
        output = np.zeros(len(input_sequence))
        for i in range(len(input_sequence)):
            output[i] = self.step(input_sequence[i])
        return output

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
