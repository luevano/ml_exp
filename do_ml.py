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
from colorama import Fore, Style
import numpy as np
from gauss_kernel import gauss_kernel
from cholesky_solve import cholesky_solve


def printc(text, color):
    """
    Prints texts normaly, but in color. Using colorama.
    text: string with the text to print.
    """
    print(color + text + Style.RESET_ALL)


def do_ml(desc_data,
          energy_data,
          training_size,
          test_size,
          sigma=1000.0,
          desc_type=None,
          show_msgs=True):
    """
    Does the ML methodology.
    desc_data: descriptor (or representation) data.
    energy_data: energy data associated with desc_data.
    training_size: size of the training set to use.
    test_size: size of the test set to use.
    sigma: depth of the kernel.
    desc_type: string with the name of the descriptor used.
    show_msgs: Show debug messages or not.
    NOTE: desc_type is just a string and is only for identification purposes.
    Also, training is done with the first part of the data and
    testing with the ending part of the data.
    """
    if not desc_type:
        desc_type = 'NOT SPECIFIED'

    d_len = len(desc_data)
    e_len = len(energy_data)
    if d_len != e_len:
        printc(''.join(['ERROR. Descriptor data size different ',
                        'than energy data size.']), Fore.RED)
        return None

    if training_size > d_len or test_size > d_len:
        printc('ERROR. Training or test size greater than data size.',
               Fore.RED)
        return None

    tic = time.perf_counter()
    if show_msgs:
        printc('{} ML started, with parameters:'.format(desc_type), Fore.CYAN)
        printc('\tTraining size: {}'.format(training_size), Fore.BLUE)
        printc('\tTest size: {}'.format(test_size), Fore.BLUE)
        printc('\tSigma: {}'.format(sigma), Fore.BLUE)

    Xcm_training = desc_data[:training_size]
    Ycm_training = energy_data[:training_size]
    Kcm_training = gauss_kernel(Xcm_training, Xcm_training, sigma)
    alpha_cm = cholesky_solve(Kcm_training, Ycm_training)

    Xcm_test = desc_data[-test_size:]
    Ycm_test = energy_data[-test_size:]
    Kcm_test = gauss_kernel(Xcm_test, Xcm_training, sigma)
    Ycm_predicted = np.dot(Kcm_test, alpha_cm)

    mae = np.mean(np.abs(Ycm_predicted - Ycm_test))
    if show_msgs:
        print('\tMAE for {}: {}'.format(desc_type, mae))

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc('\t{} ML took {:.4f} seconds.'.format(desc_type, tictoc),
               Fore.GREEN)

    return mae, tictoc
