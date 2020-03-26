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
from ml_exp.compound import Compound
from ml_exp.representations import coulomb_matrix, lennard_jones_matrix,\
        get_helping_data, adjacency_matrix, epsilon_index, check_bond, bag_of_bonds
from ml_exp.readdb import qm7db, qm9db
from ml_exp.data import NUCLEAR_CHARGE, POSSIBLE_BONDS
from ml_exp.kernels import gaussian_kernel, laplacian_kernel, wasserstein_kernel
from ml_exp.krr import krr, multi_krr

__all__ = ['Compound',
           'coulomb_matrix',
           'lennard_jones_matrix',
           'get_helping_data',
           'adjacency_matrix',
           'epsilon_index',
           'check_bond',
           'bag_of_bonds',
           'qm7db',
           'qm9db',
           'gaussian_kernel',
           'laplacian_kernel',
           'wasserstein_kernel',
           'krr',
           'multi_krr',
           'NUCLEAR_CHARGE',
           'POSSIBLE_BONDS']
