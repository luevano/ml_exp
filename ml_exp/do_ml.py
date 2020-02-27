"""MIT License

Copyright (c) 2019 David Luevano Alvarado

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import time
import numpy as np
from ml_exp.misc import printc
from ml_exp.kernels import gaussian_kernel
from ml_exp.math import cholesky_solve
# from ml_exp.qm7db import qm7db


def simple_ml(descriptors,
              energies,
              training_size,
              identifier=None,
              test_size=None,
              sigma=1000.0,
              show_msgs=True):
    """
    Basic ML methodology for a single descriptor type.
    descriptors: array of descriptors.
    energies: array of energies.
    training_size: size of the training set to use.
    identifier: string with the name of the descriptor used.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    sigma: depth of the kernel.
    show_msgs: if debu messages should be shown.
    NOTE: identifier is just a string and is only for identification purposes.
    Also, training is done with the first part of the data and
        testing with the ending part of the data.
    """
    tic = time.perf_counter()
    # Initial calculations for later use.
    data_size = descriptors.shape[0]

    if not identifier:
        identifier = 'NOT SPECIFIED'

    if not data_size == energies.shape[0]:
        raise ValueError('Energies size is different than descriptors size.')

    if training_size >= data_size:
        raise ValueError('Training size is greater or equal to the data size.')

    # If test_size is not set, it is set to a maximum size of 1500.
    if not test_size:
        test_size = data_size - training_size
        if test_size > 1500:
            test_size = 1500

    if show_msgs:
        printc(f'{identifier} ML started.', 'GREEN')
        printc(f'\tTraining size: {training_size}', 'CYAN')
        printc(f'\tTest size: {test_size}', 'CYAN')
        printc(f'\tSigma: {test_size}', 'CYAN')

    X_training = descriptors[:training_size]
    Y_training = energies[:training_size]
    K_training = gaussian_kernel(X_training, X_training, sigma)
    alpha = cholesky_solve(K_training, Y_training)

    X_test = descriptors[-test_size:]
    Y_test = energies[-test_size:]
    K_test = gaussian_kernel(X_test, X_training, sigma)
    Y_predicted = np.dot(K_test, alpha)

    mae = np.mean(np.abs(Y_predicted - Y_test))
    if show_msgs:
        printc('\tMAE for {identifier}: {mae:.4f}', 'GREEN')

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'\t{identifier} ML took {tictoc:.4f} seconds.', 'GREEN')
        printc(f'\t\tTraining size: {training_size}', 'CYAN')
        printc(f'\t\tTest size: {test_size}', 'CYAN')
        printc(f'\t\tSigma: {sigma}', 'CYAN')

    return mae, tictoc
