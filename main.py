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
from misc import printc
# import matplotlib.pyplot as plt
from read_nc_data import read_nc_data
from read_db_edata import read_db_edata
from c_matrix import c_matrix_multiple
from lj_matrix import lj_matrix_multiple
from do_ml import do_ml


# Initialization time.
init_time = time.perf_counter()

# Data reading.
tic = time.perf_counter()
printc('Data reading started.', 'CYAN')

init_path = os.getcwd()
os.chdir('data')
data_path = os.getcwd()

zi_data = read_nc_data(data_path)
molecules, nuclear_charge, energy_pbe0, energy_delta = \
    read_db_edata(zi_data, data_path)

os.chdir(init_path)
toc = time.perf_counter()
printc('\tData reading took {:.4f} seconds.'.format(toc-tic), 'GREEN')

# Matrices calculation.
cm_data = c_matrix_multiple(molecules, nuclear_charge, as_eig=True)
ljm_data = lj_matrix_multiple(molecules, nuclear_charge, as_eig=True)

# ML calculation.
do_ml(cm_data, energy_pbe0, 1000, 100, sigma=1000.0, desc_type='CM')
do_ml(ljm_data, energy_pbe0, 1000, 100, sigma=1000.0, desc_type='L-JM')

# End of program
end_time = time.perf_counter()
printc('Program took {:.4f} seconds of runtime.'.format(end_time - init_time),
       'CYAN')
