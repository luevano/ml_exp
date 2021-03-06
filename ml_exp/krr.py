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
from scipy import linalg as LA
try:
    import tensorflow as tf
    TF_AV = True
except ImportError:
    print('Tensorflow couldn\'t be imported. Maybe it is not installed.')
    TF_AV = False
from ml_exp.misc import printc
from ml_exp.kernels import gaussian_kernel, laplacian_kernel,\
    wasserstein_kernel
from ml_exp.readdb import qm7db


def krr(descriptors,
        labels,
        training_size=1500,
        test_size=None,
        sigma=1000.0,
        opt=True,
        identifier=None,
        kernel='gaussian',
        use_tf=True,
        show_msgs=True):
    """
    Basic krr methodology for a single descriptor type.
    descriptors: array of descriptors.
    labels: array of labels.
    training_size: size of the training set to use.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    sigma: depth of the kernel.
    opt: if the optimized algorithm should be used. For benchmarking purposes.
    identifier: string with the name of the descriptor used.
    kernel: which kernel to use.
    use_tf: if tensorflow should be used.
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

    if not data_size == labels.shape[0]:
        raise ValueError('Labels size is different than descriptors size.')

    if training_size >= data_size:
        raise ValueError('Training size is greater or equal to the data size.')

    # If tf is to be used but couldn't be imported, don't try to use it.
    if use_tf and not TF_AV:
        use_tf = False

    # If test_size is not set, it is set to a maximum size of 1500.
    # Also, no overlapping with training data is achieved.
    if not test_size:
        test_size = data_size - training_size
        if test_size > 1500:
            test_size = 1500

    if show_msgs:
        printc(f'{identifier} ML started.', 'GREEN')
        printc(f'\tTraining size: {training_size}', 'CYAN')
        printc(f'\tTest size: {test_size}', 'CYAN')
        printc(f'\tSigma: {sigma}', 'CYAN')
        printc(f'\tKernel: {kernel}', 'CYAN')
        printc(f'\tUse tf: {use_tf}', 'CYAN')

    if use_tf:
        if tf.config.experimental.list_physical_devices('GPU'):
            with tf.device('GPU:0'):
                X_tr = descriptors[:training_size]
                Y_tr = labels[:training_size]
                if kernel == 'gaussian':
                    K_tr = gaussian_kernel(X_tr,
                                           X_tr,
                                           sigma,
                                           use_tf=use_tf)

                elif kernel == 'laplacian':
                    K_tr = laplacian_kernel(X_tr,
                                            X_tr,
                                            sigma,
                                            use_tf=use_tf)

                elif kernel == 'wasserstein':
                    K_tr = wasserstein_kernel(X_tr,
                                              X_tr,
                                              sigma,
                                              use_tf=use_tf)

                else:
                    raise TypeError(f'{kernel} kernel not found.')

                # Adding a small value on the diagonal for cho_solve.
                dv = tf.linalg.tensor_diag(tf.constant(1e-8,
                                                       shape=(training_size),
                                                       dtype=tf.float64))
                K_tr += dv
                Y_tr = tf.expand_dims(Y_tr, 1)
                alpha = tf.linalg.cholesky_solve(tf.linalg.cholesky(K_tr),
                                                 Y_tr)

                X_te = descriptors[-test_size:]
                Y_te = labels[-test_size:]
                if kernel == 'gaussian':
                    K_te = gaussian_kernel(X_te,
                                           X_tr,
                                           sigma,
                                           use_tf=use_tf)

                elif kernel == 'laplacian':
                    K_te = laplacian_kernel(X_te,
                                            X_tr,
                                            sigma,
                                            use_tf=use_tf)

                elif kernel == 'wasserstein':
                    K_te = wasserstein_kernel(X_te,
                                              X_tr,
                                              sigma,
                                              use_tf=use_tf)

                else:
                    raise TypeError(f'{kernel} kernel not found.')

                Y_te = tf.expand_dims(Y_te, 1)
                Y_pr = tf.tensordot(K_te, alpha, 1)

                mae = tf.reduce_mean(tf.abs(Y_pr - Y_te))
        else:
            raise TypeError('No GPU found, could not create Tensor objects.')
    else:
        X_tr = descriptors[:training_size]
        Y_tr = labels[:training_size]
        if kernel == 'gaussian':
            K_tr = gaussian_kernel(X_tr,
                                   X_tr,
                                   sigma,
                                   use_tf=use_tf)

        elif kernel == 'laplacian':
            K_tr = laplacian_kernel(X_tr,
                                    X_tr,
                                    sigma,
                                    use_tf=use_tf)

        elif kernel == 'wasserstein':
            K_tr = wasserstein_kernel(X_tr,
                                      X_tr,
                                      sigma,
                                      use_tf=use_tf)

        else:
            raise TypeError(f'{kernel} kernel not found.')

        # Adding a small value on the diagonal for cho_solve.
        K_tr[np.diag_indices_from(K_tr)] += 1e-8
        alpha = LA.cho_solve(LA.cho_factor(K_tr),
                             Y_tr)

        X_te = descriptors[-test_size:]
        Y_te = labels[-test_size:]
        if kernel == 'gaussian':
            K_te = gaussian_kernel(X_te,
                                   X_tr,
                                   sigma,
                                   use_tf=use_tf)

        elif kernel == 'laplacian':
            K_te = laplacian_kernel(X_te,
                                    X_tr,
                                    sigma,
                                    use_tf=use_tf)

        elif kernel == 'wasserstein':
            K_te = wasserstein_kernel(X_te,
                                      X_tr,
                                      sigma,
                                      use_tf=use_tf)

        else:
            raise TypeError(f'{kernel} kernel not found.')
        Y_pr = np.dot(K_te, alpha)

        mae = np.mean(np.abs(Y_pr - Y_te))

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'\tMAE for {identifier}: {mae:.4f}', 'GREEN')
        printc(f'\t{identifier} ML took {tictoc:.4f} seconds.', 'GREEN')

    return mae, tictoc


def multi_krr(db_path='data',
              is_shuffled=True,
              r_seed=111,
              diag_value=None,
              lj_sigma=1.0,
              lj_epsilon=1.0,
              use_forces=False,
              acount={'C':7, 'H':16, 'N':3, 'O':3, 'S':1},
              size=23,
              sort=False,
              flatten=True,
              as_eig=True,
              bohr_ru=False,
              training_size=1500,
              test_size=None,
              sigma=1000.0,
              identifiers=['CM'],
              use_tf=True,
              show_msgs=True):
    """
    Does multiple KRR for several descriptors.
    db_path: path to the database directory.
    is_shuffled: if the resulting list of compounds should be shuffled.
    r_seed: random seed to use for the shuffling.
    diag_value: if special diagonal value is to be used.
    lj_sigma: sigma value.
    lj_epsilon: epsilon value.
    use_forces: if the use of forces instead of k_cx should be used.
    acount: atom count for the compound, defaults to qm7 sizes.
    size: compound size.
    sort: if the representation should be sorted row-norm or bag-wise.
    flatten: if the representation should be 1D.
    as_eig: if the representation should be as the eigenvalues.
    bohr_ru: if radius units should be in bohr's radius units.
    training_size: size of the training set to use.
    test_size: size of the test set to use. If no size is given,
        the last remaining molecules are used.
    sigma: depth of the kernel.
    identifiers: list of names (strings) of descriptors to use.
    use_tf: if tensorflow should be used.
    show_msgs: if debug messages should be shown.
    """
    if type(identifiers) != list:
        raise TypeError('\'identifiers\' is not a list.')

    # If tf is to be used but couldn't be imported, don't try to use it.
    if use_tf and not TF_AV:
        use_tf = False

    init_time = time.perf_counter()

    # Data reading.
    tic = time.perf_counter()
    compounds, energy_pbe0, energy_delta = qm7db(db_path=db_path,
                                                 is_shuffled=is_shuffled,
                                                 r_seed=r_seed,
                                                 use_tf=use_tf)
    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'Data reading took {tictoc:.4f} seconds.', 'CYAN')

    # Matrices calculation.
    tic = time.perf_counter()
    for compound in compounds:
        if 'CM' in identifiers:
            compound.gen_cm(size=size,
                            sort=sort,
                            flatten=flatten,
                            as_eig=as_eig,
                            bohr_ru=bohr_ru)
        if 'LJM' in identifiers:
            compound.gen_ljm(diag_value=diag_value,
                             sigma=lj_sigma,
                             epsilon=lj_epsilon,
                             size=size,
                             sort=sort,
                             flatten=flatten,
                             as_eig=as_eig,
                             bohr_ru=bohr_ru)
        if 'AM' in identifiers:
            compound.gen_hd(size=size,
                            bohr_ru=bohr_ru)
            compound.gen_am(use_forces=use_forces,
                            size=size,
                            sort=sort,
                            flatten=flatten)
        if 'BOB' in identifiers:
            compound.gen_bob(sort=sort,
                             acount=acount)

    # Create a numpy array (or tensorflow tensor) for the descriptors.
    if 'CM' in identifiers:
        cm_data = np.array([comp.cm for comp in compounds], dtype=np.float64)
    if 'LJM' in identifiers:
        ljm_data = np.array([comp.ljm for comp in compounds], dtype=np.float64)
    if 'AM' in identifiers:
        am_data = np.array([comp.am for comp in compounds], dtype=np.float64)
    if 'BOB' in identifiers:
        bob_data = np.array([comp.bob for comp in compounds], dtype=np.float64)

    if use_tf:
        if tf.config.experimental.list_physical_devices('GPU'):
            with tf.device('GPU:0'):
                if 'CM' in identifiers:
                    cm_data = tf.convert_to_tensor(cm_data)
                if 'LJM' in identifiers:
                    ljm_data = tf.convert_to_tensor(ljm_data)
                if 'AM' in identifiers:
                    am_data = tf.convert_to_tensor(am_data)
                if 'BOB' in identifiers:
                    bob_data = tf.convert_to_tensor(bob_data)
        else:
            raise TypeError('No GPU found, could not create Tensor objects.')

    toc = time.perf_counter()
    tictoc = toc - tic
    if show_msgs:
        printc(f'Matrices calculation took {tictoc:.4f} seconds.', 'CYAN')

    # ML calculation.
    if 'CM' in identifiers:
        cm_mae, cm_tictoc = krr(cm_data,
                                energy_pbe0,
                                training_size=training_size,
                                test_size=test_size,
                                sigma=sigma,
                                identifier='CM',
                                kernel='gaussian',
                                use_tf=use_tf,
                                show_msgs=show_msgs)
    if 'LJM' in identifiers:
        ljm_mae, ljm_tictoc = krr(ljm_data,
                                  energy_pbe0,
                                  training_size=training_size,
                                  test_size=test_size,
                                  sigma=sigma,
                                  identifier='LJM',
                                  kernel='gaussian',
                                  use_tf=use_tf,
                                  show_msgs=show_msgs)
    if 'AM' in identifiers:
        am_mae, am_tictoc = krr(am_data,
                                energy_pbe0,
                                training_size=training_size,
                                test_size=test_size,
                                sigma=sigma,
                                identifier='AM',
                                kernel='gaussian',
                                use_tf=use_tf,
                                show_msgs=show_msgs)
    if 'BOB' in identifiers:
        bob_mae, bob_tictoc = krr(bob_data,
                                  energy_pbe0,
                                  training_size=training_size,
                                  test_size=test_size,
                                  sigma=sigma,
                                  identifier='BOB',
                                  kernel='laplacian',
                                  use_tf=use_tf,
                                  show_msgs=show_msgs)

    # End of program
    end_time = time.perf_counter()
    totaltime = end_time - init_time
    printc(f'Program took {totaltime:.4f} seconds.', 'CYAN')
