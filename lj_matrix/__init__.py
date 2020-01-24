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
from lj_matrix.read_qm7_data import read_nc_data, read_db_data, read_qm7_data
from lj_matrix.c_matrix import c_matrix, c_matrix_multiple
from lj_matrix.lj_matrix import lj_matrix, lj_matrix_multiple
from lj_matrix.frob_norm import frob_norm
from lj_matrix.gauss_kernel import gauss_kernel
from lj_matrix.cholesky_solve import cholesky_solve
from lj_matrix.do_ml import do_ml
from lj_matrix.parallel_create_matrices import parallel_create_matrices
from lj_matrix.misc import plot_benchmarks


# If somebody does "from package import *", this is what they will
# be able to access:
__all__ = ['read_nc_data',
           'read_db_data',
           'read_qm7_data',
           'c_matrix',
           'c_matrix_multiple',
           'lj_matrix',
           'lj_matrix_multiple',
           'frob_norm',
           'gauss_kernel',
           'cholesky_solve',
           'do_ml',
           'parallel_create_matrices',
           'plot_benchmarks']
