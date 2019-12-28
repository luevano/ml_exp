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
from multiprocessing import Process, Pipe
from lj_matrix.misc import printc
from lj_matrix.gauss_kernel import gauss_kernel
from lj_matrix.cholesky_solve import cholesky_solve
from lj_matrix.read_qm7_data import read_qm7_data
from lj_matrix.parallel_create_matrices import parallel_create_matrices


def ml(desc_data,
       energy_data,
       training_size,
       desc_type=None,
       pipe=None,
       test_size=None,
       sigma=1000.0,
       show_msgs=True):
    """
    Does the ML methodology.
    desc_data: descriptor (or representation) data.
    energy_data: energy data associated with desc_data.
    training_size: size of the training set to use.
    desc_type: string with the name of the descriptor used.
    pipe: for multiprocessing purposes. Sends the data calculated
        through a pipe.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    sigma: depth of the kernel.
    show_msgs: Show debug messages or not.
    NOTE: desc_type is just a string and is only for identification purposes.
    Also, training is done with the first part of the data and
    testing with the ending part of the data.
    """
    tic = time.perf_counter()
    # Initial calculations for later use.
    d_len = len(desc_data)
    e_len = len(energy_data)

    if not desc_type:
        desc_type = 'NOT SPECIFIED'

    if d_len != e_len:
        printc(''.join(['ERROR. Descriptor data size different ',
                        'than energy data size.']), 'RED')
        return None

    if training_size >= d_len:
        printc('ERROR. Training size greater or equal than data size.', 'RED')
        return None

    if not test_size:
        test_size = d_len - training_size
        if test_size > 1500:
            test_size = 1500

    if show_msgs:
        printc('{} ML started.'.format(desc_type), 'GREEN')
        printc('\tTraining size: {}'.format(training_size), 'CYAN')
        printc('\tTest size: {}'.format(test_size), 'CYAN')
        printc('\tSigma: {}'.format(sigma), 'CYAN')

    X_training = desc_data[:training_size]
    Y_training = energy_data[:training_size]
    K_training = gauss_kernel(X_training, X_training, sigma)
    alpha_ = cholesky_solve(K_training, Y_training)

    X_test = desc_data[-test_size:]
    Y_test = energy_data[-test_size:]
    K_test = gauss_kernel(X_test, X_training, sigma)
    Y_predicted = np.dot(K_test, alpha_)

    mae = np.mean(np.abs(Y_predicted - Y_test))
    if show_msgs:
        printc('\tMAE for {}: {:.4f}'.format(desc_type, mae), 'GREEN')

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc('\t{} ML took {:.4f} seconds.'.format(desc_type, tictoc),
               'GREEN')
        printc('\t\tTraining size: {}'.format(training_size), 'CYAN')
        printc('\t\tTest size: {}'.format(test_size), 'CYAN')
        printc('\t\tSigma: {}'.format(sigma), 'CYAN')

    if pipe:
        pipe.send([desc_type, training_size, test_size, sigma, mae, tictoc])

    return mae, tictoc


def do_ml(min_training_size,
          max_training_size=None,
          training_increment_size=500,
          test_size=None,
          ljm_diag_value=None,
          ljm_sigma=1.0,
          ljm_epsilon=1.0,
          save_benchmarks=False,
          max_len=25,
          as_eig=True,
          bohr_radius_units=False,
          sigma=1000.0,
          show_msgs=True):
    """
    Main function that does the whole ML process.
    min_training_size: minimum training size.
    max_training_size: maximum training size.
    training_increment_size: training increment size.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    ljm_diag_value: if a special diagonal value should be used in lj matrix.
    ljm_sigma: sigma value for lj matrix.
    ljm_epsilon: epsilon value for lj matrix.
    save_benchmarks: if benchmarks should be saved.
    max_len: maximum amount of atoms in molecule.
    as_eig: if data should be returned as matrix or array of eigenvalues.
    bohr_radius_units: if units should be in bohr's radius units.
    sigma: depth of the kernel.
    show_msgs: Show debug messages or not.
    """
    # Initialization time.
    init_time = time.perf_counter()
    if not max_training_size:
        max_training_size = min_training_size + training_increment_size

    # Data reading.
    molecules, nuclear_charge, energy_pbe0, energy_delta = read_qm7_data()

    # Matrices calculation.
    cm_data, ljm_data = parallel_create_matrices(molecules,
                                                 nuclear_charge,
                                                 ljm_diag_value,
                                                 ljm_sigma,
                                                 ljm_epsilon,
                                                 max_len,
                                                 as_eig,
                                                 bohr_radius_units)

    # ML calculation.
    procs = []
    cm_pipes = []
    ljm_pipes = []
    for i in range(min_training_size,
                   max_training_size + 1,
                   training_increment_size):
        cm_recv, cm_send = Pipe(False)
        p1 = Process(target=ml,
                     args=(cm_data,
                           energy_pbe0,
                           i,
                           'CM',
                           cm_send,
                           test_size,
                           sigma,
                           show_msgs))
        procs.append(p1)
        cm_pipes.append(cm_recv)
        p1.start()

        ljm_recv, ljm_send = Pipe(False)
        p2 = Process(target=ml,
                     args=(ljm_data,
                           energy_pbe0,
                           i,
                           'L-JM',
                           ljm_send,
                           test_size,
                           sigma,
                           show_msgs))
        procs.append(p2)
        ljm_pipes.append(ljm_recv)
        p2.start()

    cm_bench_results = []
    ljm_bench_results = []
    for cd_pipe, ljd_pipe in zip(cm_pipes, ljm_pipes):
        cm_bench_results.append(cd_pipe.recv())
        ljm_bench_results.append(ljd_pipe.recv())

    for proc in procs:
        proc.join()

    if save_benchmarks:
        with open('data\\benchmarks.csv', 'a') as save_file:
            # save_file.write(''.join(['ml_type,tr_size,te_size,kernel_s,',
            #                          'mae,time,lj_s,lj_e,date_ran\n']))
            ltime = time.localtime()[:3][::-1]
            ljm_se = ',' + str(ljm_sigma) + ',' + str(ljm_epsilon) + ','
            date = '/'.join([str(field) for field in ltime])
            for cm, ljm, in zip(cm_bench_results, ljm_bench_results):
                cm_text = ','.join([str(field) for field in cm])\
                    + ',' + date + '\n'
                ljm_text = ','.join([str(field) for field in ljm])\
                    + ljm_se + date + '\n'
                save_file.write(cm_text)
                save_file.write(ljm_text)

    # End of program
    end_time = time.perf_counter()
    printc('Program took {:.4f} seconds.'.format(end_time - init_time),
           'CYAN')
