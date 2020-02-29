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
from ml_exp.qm7db import qm7db


def simple_ml(descriptors,
              energies,
              training_size,
              test_size=None,
              sigma=1000.0,
              opt=True,
              identifier=None,
              show_msgs=True):
    """
    Basic ML methodology for a single descriptor type.
    descriptors: array of descriptors.
    energies: array of energies.
    training_size: size of the training set to use.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    sigma: depth of the kernel.
    opt: if the optimized algorithm should be used. For benchmarking purposes.
    identifier: string with the name of the descriptor used.
    show_msgs: if debug messages should be shown.
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
    K_training = gaussian_kernel(X_training,
                                 X_training,
                                 sigma,
                                 opt=opt)
    alpha = cholesky_solve(K_training,
                           Y_training)

    X_test = descriptors[-test_size:]
    Y_test = energies[-test_size:]
    K_test = gaussian_kernel(X_test,
                             X_training,
                             sigma,
                             opt=opt)
    Y_predicted = np.dot(K_test,
                         alpha)

    mae = np.mean(np.abs(Y_predicted - Y_test))
    if show_msgs:
        printc(f'\tMAE for {identifier}: {mae:.4f}', 'GREEN')

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'\t{identifier} ML took {tictoc:.4f} seconds.', 'GREEN')
        printc(f'\t\tTraining size: {training_size}', 'CYAN')
        printc(f'\t\tTest size: {test_size}', 'CYAN')
        printc(f'\t\tSigma: {sigma}', 'CYAN')

    return mae, tictoc


def do_ml(db_path='data',
          is_shuffled=True,
          r_seed=111,
          diag_value=None,
          lj_sigma=1.0,
          lj_epsilon=1.0,
          use_forces=False,
          stuff='bonds',
          size=23,
          as_eig=True,
          bohr_ru=False,
          training_size=1500,
          test_size=None,
          sigma=1000.0,
          opt=True,
          identifiers=["CM"],
          show_msgs=True):
    """
    Main function that does the whole ML process.
    db_path: path to the database directory.
    is_shuffled: if the resulting list of compounds should be shuffled.
    r_seed: random seed to use for the shuffling.
    diag_value: if special diagonal value is to be used.
    lj_sigma: sigma value.
    lj_epsilon: epsilon value.
    use_forces: if the use of forces instead of k_cx should be used.
    stuff: elements of the bag, by default the known bag of bonds.
    size: compound size.
    as_eig: if the representation should be as the eigenvalues.
    bohr_ru: if radius units should be in bohr's radius units.
    training_size: size of the training set to use.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    sigma: depth of the kernel.
    opt: if the optimized algorithm should be used. For benchmarking purposes.
    identifiers: list of names (strings) of descriptors to use.
    show_msgs: if debug messages should be shown.
    """
    if type(identifiers) != list:
        raise TypeError('\'identifiers\' is not a list.')

    init_time = time.perf_counter()

    # Data reading.
    tic = time.perf_counter()
    compounds, energy_pbe0, energy_delta = qm7db(db_path=db_path,
                                                 is_shuffled=is_shuffled,
                                                 r_seed=r_seed)
    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'Data reading took {tictoc:.4f} seconds.', 'CYAN')

    # Matrices calculation.
    tic = time.perf_counter()
    for compound in compounds:
        if 'CM' in identifiers:
            compound.gen_cm(size=size,
                            as_eig=as_eig,
                            bohr_ru=bohr_ru)
        if 'LJM' in identifiers:
            compound.gen_ljm(diag_value=diag_value,
                             sigma=lj_sigma,
                             epsilon=lj_epsilon,
                             size=size,
                             as_eig=as_eig,
                             bohr_ru=bohr_ru)
        if 'AM' in identifiers:
            compound.gen_hd(size=size,
                            bohr_ru=bohr_ru)
            compound.gen_am(use_forces=use_forces,
                            size=size)
        if 'BOB' in identifiers:
            compound.gen_bob(size=size)

    # Create a numpy array for the descriptors.
    if 'CM' in identifiers:
        cm_data = np.array([comp.cm for comp in compounds], dtype=float)
    if 'LJM' in identifiers:
        ljm_data = np.array([comp.ljm for comp in compounds], dtype=float)
    if 'AM' in identifiers:
        am_data = np.array([comp.cm for comp in compounds], dtype=float)
    if 'BOB' in identifiers:
        bob_data = np.array([comp.bob for comp in compounds], dtype=float)

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'Matrices calculation took {tictoc:.4f} seconds.', 'CYAN')

    # ML calculation.
    if 'CM' in identifiers:
        cm_mae, cm_tictoc = simple_ml(cm_data,
                                      energy_pbe0,
                                      training_size=training_size,
                                      test_size=test_size,
                                      sigma=sigma,
                                      opt=opt,
                                      identifier='CM',
                                      show_msgs=show_msgs)
    if 'LJM' in identifiers:
        ljm_mae, ljm_tictoc = simple_ml(ljm_data,
                                        energy_pbe0,
                                        training_size=training_size,
                                        test_size=test_size,
                                        sigma=sigma,
                                        opt=opt,
                                        identifier='LJM',
                                        show_msgs=show_msgs)
    if 'AM' in identifiers:
        am_mae, am_tictoc = simple_ml(am_data,
                                      energy_pbe0,
                                      training_size=training_size,
                                      test_size=test_size,
                                      sigma=sigma,
                                      opt=opt,
                                      identifier='CM',
                                      show_msgs=show_msgs)
    if 'BOB' in identifiers:
        bob_mae, bob_tictoc = simple_ml(bob_data,
                                        energy_pbe0,
                                        training_size=training_size,
                                        test_size=test_size,
                                        sigma=sigma,
                                        opt=opt,
                                        identifier='CM',
                                        show_msgs=show_msgs)

    # End of program
    end_time = time.perf_counter()
    totaltime = end_time - init_time
    printc(f'Program took {totaltime:.4f} seconds.', 'CYAN')
