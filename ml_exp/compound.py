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
import numpy as np


class Compound:
    def __init__(self, xyz=None):
        self.n = None
        self.atoms = None
        self.coordinates = None

        if xyz is not None:
            self.read_xyz(xyz)

    def read_xyz(self, filename):
        """
        Reads an xyz file and adds the corresponding data to the Compound.
        filename: (path to) the xyz file.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.n = int(lines[0])
        self.atoms = []
        self.coordinates = np.empty((self.n, 3), dtype=float)

        for i, atom in enumerate(lines[2:self.n + 2]):
            atom_d = atom.split()

            self.atoms.append(atom_d[0])
            self.coordinates[i] = np.asarray(atom_d[1:4], dtype=float)
