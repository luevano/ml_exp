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
# from gauss_kernel import gauss_kernel
# from cholesky_solve import cholesky_solve
from do_ml import do_ml


def printc(text, color):
    """
    Prints texts normaly, but in color. Using colorama.
    text: string with the text to print.
    """
    print(color + text + Style.RESET_ALL)


# Initialization.
init_time = time.perf_counter()
init()

# Move to data folder.
init_path = os.getcwd()
os.chdir('data')
data_path = os.getcwd()

printc('Data reading started.', Fore.CYAN)
tic = time.perf_counter()

# Read nuclear charge data.
zi_data = read_nc_data(data_path)

# Read molecule data.
molecules, nuclear_charge, energy_pbe0, energy_delta = \
    read_db_edata(zi_data, data_path)

toc = time.perf_counter()
printc('\tData reading took {:.4f} seconds.'.format(toc-tic), Fore.GREEN)

# Go back to main folder.
os.chdir(init_path)

printc('Coulomb Matrices calculation started.', Fore.CYAN)
tic = time.perf_counter()

cm_data = np.array([c_matrix(mol, nc, as_eig=True)
                    for mol, nc in zip(molecules, nuclear_charge)])

toc = time.perf_counter()
printc('\tCoulomb matrices calculation took {:.4f} seconds.'.format(toc-tic),
       Fore.GREEN)

printc('L-J Matrices calculation started.', Fore.CYAN)
tic = time.perf_counter()

ljm_data = np.array([lj_matrix(mol, nc, as_eig=True)
                     for mol, nc in zip(molecules, nuclear_charge)])

toc = time.perf_counter()
printc('\tL-J matrices calculation took {:.4f} seconds.'.format(toc-tic),
       Fore.GREEN)

# Coulomb Matrix ML
do_ml(cm_data, 'CM', energy_pbe0, 1000, 100, 1000.0)
# Lennard-Jones Matrix ML
do_ml(ljm_data, 'L-JM', energy_pbe0, 1000, 100, 1000.0)

# End of program
end_time = time.perf_counter()
printc('Program took {:.4f} seconds of runtime.'.format(end_time - init_time),
       Fore.CYAN)
