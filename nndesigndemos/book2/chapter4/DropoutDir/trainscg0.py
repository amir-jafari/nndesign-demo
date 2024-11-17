import math
import numpy as np
import time
import copy
from nndesigndemos.book2.chapter4.DropoutDir.getx import getx
from nndesigndemos.book2.chapter4.DropoutDir.newtr import newtr
from nndesigndemos.book2.chapter4.DropoutDir.get_do_mask import get_do_mask
from nndesigndemos.book2.chapter4.DropoutDir.calcperf0 import calcperf0
from nndesigndemos.book2.chapter4.DropoutDir.setx import setx
from nndesigndemos.book2.chapter4.DropoutDir.calcgx0 import calcgx0
from nndesigndemos.book2.chapter4.DropoutDir.cliptr import cliptr


def trainscg0(net, Pd=None, Tl=None, VV=[], TV=[]):
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

    for epoch in range(0, epochs+1):
        if epoch == 0:
            if dropout:
                net['doflag'] = 1
                net = get_do_mask(net, Pd)

            perf, Ac = calcperf0(net, Pd, Tl)

            gX = calcgx0(net, Pd, Ac, Tl)


            normgX = np.sqrt(np.dot(gX.T, gX))
            dX = -gX  # Initial search direction
            nrmsqr_dX = np.dot(dX.T, dX)
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
            break

        if success == 1:
            sigmak = sigma / norm_dX
            X_temp = X + sigmak * dX
            net_temp = setx(net, X_temp)  # Assuming function to set weights
            _, Ac = calcperf0(net_temp, Pd, Tl)
            gX_temp = calcgx0(net_temp, Pd, Ac, Tl)
            sk = (gX_temp - gX) / sigmak
            deltak = np.dot(dX.T, sk)

        # Scale deltak and calculate step size
        deltak += (lambdak - lambdab) * nrmsqr_dX

        if deltak <= 0:
            lambdab = 2 * (lambdak - deltak / nrmsqr_dX)
            deltak = -deltak + lambdak * nrmsqr_dX
            lambdak = lambdab

        muk = np.dot(-dX.T, gX)
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
            net = copy.deepcopy(net_temp)

            perf, Ac = calcperf0(net, Pd, Tl)

            gX = calcgx0(net, Pd, Ac, Tl)
            normgX = np.sqrt(np.dot(gX.T, gX))
            lambdab = 0
            success = 1

            # Update direction for next epoch
            if epoch % num_X == 0:
                dX = -gX
            else:
                betak = (np.dot(gX.T, gX) - np.dot(gX.T, gX_old)) / muk
                dX = -gX + betak * dX
                if dropout:
                    ind1 = np.where(gX == 0)  # Find indices where gX is zero
                    dX[ind1] = 0

            nrmsqr_dX = np.dot(dX.T, dX)
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

    tr = cliptr(tr, epoch)

    return net, tr
