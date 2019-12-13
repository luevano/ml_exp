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
from multiprocessing import Process, Pipe
# import matplotlib.pyplot as plt
from misc import printc
from read_qm7_data import read_qm7_data
from c_matrix import c_matrix_multiple
from lj_matrix import lj_matrix_multiple
# from do_ml import do_ml


def main():
    # Initialization time.
    init_time = time.perf_counter()
    procs = []
    pipes = []

    # Data reading.
    zi_data, molecules, nuclear_charge, energy_pbe0, energy_delta =\
        read_qm7_data()

    # Matrices calculation.
    cm_recv, cm_send = Pipe()
    pipes.append(cm_send)
    p1 = Process(target=c_matrix_multiple,
                 args=(molecules, nuclear_charge, cm_send))
    procs.append(p1)
    p1.start()

    ljm_recv, ljm_send = Pipe()
    pipes.append(ljm_send)
    p2 = Process(target=lj_matrix_multiple,
                 args=(molecules, nuclear_charge, ljm_send))
    procs.append(p2)
    p2.start()

    cm_data = cm_recv.recv()
    ljm_data = ljm_recv.recv()

    for pipe, proc in zip(pipes, procs):
        pipe.close()
        proc.join()

    print(type(cm_data), cm_data[0])
    print(type(ljm_data), ljm_data[0])

    """
    # ML calculation.
    do_ml(cm_data,
        energy_pbe0,
        1000,
        test_size=100,
        sigma=1000.0,
        desc_type='CM')
    do_ml(ljm_data,
        energy_pbe0,
        1000,
        test_size=100,
        sigma=1000.0,
        desc_type='L-JM')
    """

    # End of program
    end_time = time.perf_counter()
    printc('Program took {:.4f} seconds.'.format(end_time - init_time),
           'CYAN')


if __name__ == '__main__':
    main()
