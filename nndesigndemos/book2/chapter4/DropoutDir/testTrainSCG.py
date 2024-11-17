import numpy as np
import matplotlib.pyplot as plt

from nndesigndemos.book2.chapter4.DropoutDir.newmultilay import newmultilay
from nndesigndemos.book2.chapter4.DropoutDir.trainscg0 import trainscg0
from nndesigndemos.book2.chapter4.DropoutDir.simnet import simnet
from nndesigndemos.book2.chapter4.DropoutDir.softmax0 import softmax0
from nndesigndemos.book2.chapter4.DropoutDir.crossentr import crossentr
from nndesigndemos.book2.chapter4.DropoutDir.tansig0 import tansig0


def testTrainSCG():
    # Create the network
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

    # Training data (inputs P and targets T)
    P = np.array([
        [0.2, 0.2, 0, 0, -0.35, -0.35, -0.5, 0, 0.25, 0, -0.25, 0, 0.25, -0.15, -0.15, 0.1, 0.1],
        [-0.75, 0.75, 0.65, -0.65, -0.45, 0.45, 0, -0.5, 0.5, 0.25, 0, -0.25, -0.5, 0.2, -0.2, 0.3, -0.3]
    ])
    T = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Adding noise to input data
    # P = np.hstack([P for _ in range(7)]) # For test
    P = np.hstack([P] + [P + stdv * (np.random.rand(*P.shape) - 0.5) for _ in range(6)])
    T = np.hstack([T for _ in range(7)])

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

    net1, tr = trainscg0(net, P, T, [], [])

    plt.plot(tr['perf'])  # Automatically uses index as x-axis if only y-data is provided
    plt.xlabel("Epoch")  # Label for the x-axis
    plt.ylabel("Value")  # Label for the y-axis
    plt.title("Line Plot of Data")  # Title of the plot
    plt.grid(True)  # Optional: adds a grid to the plot for readability

    # Display the plot
    plt.show()
    # exit()

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


if __name__ == '__main__':
    # Run the function
    testTrainSCG()
