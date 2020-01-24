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
from multiprocessing import Process, Pipe
from ml_exp.c_matrix import c_matrix_multiple
from ml_exp.lj_matrix import lj_matrix_multiple


def parallel_create_matrices(mol_data,
                             nc_data,
                             ljm_diag_value=None,
                             ljm_sigma=1.0,
                             ljm_epsilon=1.0,
                             max_len=25,
                             as_eig=True,
                             bohr_radius_units=False):
    """
    Creates the Coulomb and L-J matrices in parallel.
    mol_data: molecule data, matrix of atom coordinates.
    nc_data: nuclear charge data, array of atom data.
    ljm_diag_value: if special diagonal value is to be used for lj matrix.
    ljm_sigma: sigma value for lj matrix.
    ljm_epsilon: psilon value for lj matrix.
    max_len: maximum amount of atoms in molecule.
    as_eig: if data should be returned as matrix or array of eigenvalues.
    bohr_radius_units: if units should be in bohr's radius units.
    """

    # Matrices calculation.
    procs = []
    pipes = []

    cm_recv, cm_send = Pipe(False)
    p1 = Process(target=c_matrix_multiple,
                 args=(mol_data,
                       nc_data,
                       cm_send,
                       max_len,
                       as_eig,
                       bohr_radius_units))
    procs.append(p1)
    pipes.append(cm_recv)
    p1.start()

    ljm_recv, ljm_send = Pipe(False)
    p2 = Process(target=lj_matrix_multiple,
                 args=(mol_data,
                       nc_data,
                       ljm_send,
                       ljm_diag_value,
                       ljm_sigma,
                       ljm_epsilon,
                       max_len,
                       as_eig,
                       bohr_radius_units))
    procs.append(p2)
    pipes.append(ljm_recv)
    p2.start()

    cm_data = pipes[0].recv()
    ljm_data = pipes[1].recv()

    for proc in procs:
        proc.join()

    return cm_data, ljm_data
