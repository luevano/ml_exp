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
import numpy as np
import random


# 'hof_qm7.txt.txt' retrieved from
# https://github.com/qmlcode/tutorial
def qm7db(nc,
          db_path='data',
          is_shuffled=True,
          r_seed=111):
    """
    Creates a list of compounds with the qm7 database.
    nc: dictionary containing nuclear charge data.
    db_path: path to the database directory.
    is_shuffled: if the resulting list of compounds should be shuffled.
    r_seed: random seed to use for the shuffling.
    """
    fname = f'{db_path}/hof_qm7.txt'
    with open(fname, 'r') as f:
        lines = f.readlines()

    compounds = []
    for i, line in enumerate(lines):
        line = line.split()
        compounds.append(Compound(f'{db_path}/{line[0]}'))
        compounds[i].pbe0 = float(line[1])
        compounds[i].delta = float(line[1]) - float(line[2])

    # Shuffle the compounds list
    random.seed(r_seed)
    random.shuffle(compounds)

    e_pbe0 = np.array([compound.pbe0 for compound in compounds], dtype=float)
    e_delta = np.array([compound.delta for compound in compounds], dtype=float)

    return compounds, e_pbe0, e_delta
