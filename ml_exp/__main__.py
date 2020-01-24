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
# from ml_exp.do_ml import do_ml
# from ml_exp.misc import plot_benchmarks
from ml_exp.read_qm7_data import read_qm7_data
from ml_exp.adj_matrix import fneig_matrix, adj_matrix

if __name__ == '__main__':
    """
    do_ml(min_training_size=1500,
          max_training_size=2000,
          training_increment_size=500,
          test_size=None,
          ljm_diag_value=None,
          ljm_sigma=1.0,
          ljm_epsilon=1.0,
          r_seed=111,
          save_benchmarks=False,
          show_msgs=True)
    """
    # plot_benchmarks()
    xyz, nc, pbe0, delta, atoms = read_qm7_data(return_atoms=True)
    for i in range(1):
        fnm, bonds, forces = fneig_matrix(atoms[i], xyz[i], nc[i], True)
        am = adj_matrix(bonds, forces)

        print(f'{i} first neighbor matrix\n{fnm}')
        print(f'{i} bond list\n{bonds}')
        print(f'{i} force list\n{forces}')
        print(f'{i} adjacency matrix\n{am}')
        print('-'*30)
    print('OK!')
