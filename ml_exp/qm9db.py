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
try:
    import tensorflow as tf
    TF_AV = True
except ImportError:
    print('Tensorflow couldn\'t be imported. Maybe it is not installed.')
    TF_AV = False
import random


def qm9db(db_path='data',
          is_shuffled=True,
          r_seed=111,
          use_tf=True):
    """
    Creates a list of compounds with the qm9 database.
    db_path: path to the database directory.
    is_shuffled: if the resulting list of compounds should be shuffled.
    r_seed: random seed to use for the shuffling.
    use_tf: if tensorflow should be used.
    """
    # If tf is to be used but couldn't be imported, don't try to use it.
    if use_tf and not TF_AV:
        use_tf = False

    fname = f'{db_path}/xyz_qm9.txt'
    with open(fname, 'r') as f:
        lines = f.readlines()

    compounds = []
    for i, line in enumerate(lines):
        line = line.strip()
        compounds.append(Compound(f'{db_path}/{line}', db='qm9'))

    if is_shuffled:
        random.seed(r_seed)
        random.shuffle(compounds)

    return compounds
