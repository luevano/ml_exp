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
import os
import time
from colorama import init, Fore, Style
# import math
# import random
import numpy as np
# from numpy.linalg import cholesky, eig
# import matplotlib.pyplot as plt
from read_nc_data import read_nc_data
from read_db_edata import read_db_edata
from c_matrix import c_matrix
from lj_matrix import lj_matrix
# from frob_norm import frob_norm
from gauss_kernel import gauss_kernel
from cholesky_solve import cholesky_solve

# Initialization.
init_time = time.perf_counter()
init()

# Move to data folder.
init_path = os.getcwd()
os.chdir('../../data')
data_path = os.getcwd()

print(Fore.CYAN
      + 'Data reading started.'
      + Style.RESET_ALL)
tic = time.perf_counter()

# Read nuclear charge data.
zi_data = read_nc_data(data_path)

# Read molecule data.
molecules, nuclear_charge, energy_pbe0, energy_delta = \
    read_db_edata(zi_data, data_path)

toc = time.perf_counter()
print(Fore.GREEN
      + '\tData reading took {:.4f} seconds.'.format(toc-tic)
      + Style.RESET_ALL)

# Go back to main folder.
os.chdir(init_path)

print(Fore.CYAN
      + 'Coulomb Matrices calculation started.'
      + Style.RESET_ALL)
tic = time.perf_counter()

cm_data = np.array([c_matrix(mol, nc, as_eig=True)
                    for mol, nc in zip(molecules, nuclear_charge)])

toc = time.perf_counter()
print(Fore.GREEN
      + '\tCoulomb matrices calculation took {:.4f} seconds.'.format(toc-tic)
      + Style.RESET_ALL)

print(Fore.CYAN
      + 'L-J Matrices calculation started.'
      + Style.RESET_ALL)
tic = time.perf_counter()

ljm_data = np.array([lj_matrix(mol, nc, as_eig=True)
                     for mol, nc in zip(molecules, nuclear_charge)])

toc = time.perf_counter()
print(Fore.GREEN
      + '\tL-J matrices calculation took {:.4f} seconds.'.format(toc-tic)
      + Style.RESET_ALL)

#
# Problem solving with Coulomb Matrix.
#
print(Fore.CYAN
      + 'CM ML started.'
      + Style.RESET_ALL)
tic = time.perf_counter()

sigma = 1000.0

Xcm_training = cm_data[:6000]
Ycm_training = energy_pbe0[:6000]
Kcm_training = gauss_kernel(Xcm_training, Xcm_training, sigma)
alpha_cm = cholesky_solve(Kcm_training, Ycm_training)

Xcm_test = cm_data[-1000:]
Ycm_test = energy_pbe0[-1000:]
Kcm_test = gauss_kernel(Xcm_test, Xcm_training, sigma)
Ycm_predicted = np.dot(Kcm_test, alpha_cm)

print('\tMean absolute error for CM: {}'.format(np.mean(np.abs(Ycm_predicted
                                                               - Ycm_test))))

toc = time.perf_counter()
print(Fore.GREEN
      + '\tCM ML took {:.4f} seconds.'.format(toc-tic)
      + Style.RESET_ALL)


#
# Problem solving with L-J Matrix.
#
print(Fore.CYAN
      + 'L-JM ML started.'
      + Style.RESET_ALL)
tic = time.perf_counter()

sigma = 1000.0

Xljm_training = ljm_data[:6000]
Yljm_training = energy_pbe0[:6000]
Kljm_training = gauss_kernel(Xljm_training, Xljm_training, sigma)
alpha_ljm = cholesky_solve(Kljm_training, Yljm_training)

Xljm_test = ljm_data[-1000:]
Yljm_test = energy_pbe0[-1000:]
Kljm_test = gauss_kernel(Xljm_test, Xljm_training, sigma)
Yljm_predicted = np.dot(Kljm_test, alpha_ljm)

print('\tMean absolute error for LJM: {}'.format(np.mean(np.abs(Yljm_predicted
                                                                - Yljm_test))))

toc = time.perf_counter()
print(Fore.GREEN
      + '\tL-JM ML took {:.4f} seconds.'.format(toc-tic)
      + Style.RESET_ALL)


# End of program
end_time = time.perf_counter()
print(Fore.CYAN
      + 'The program took {:.4f} seconds of runtime.'.format(end_time
                                                             - init_time)
      + Style.RESET_ALL)
