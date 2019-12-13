# lj_matrix

This is an exploration of a representation (or descriptor), based on the Lennard-Jones potential, for use in prediction of atomization energies (and possibly other properties) of molecules using Machine Learning (ML).

An implementation of existing representations (for now only Coulomb matrix), this new representation and most of the ML routine is created from the, mostly, ground up.

## Data used

* The *QM7* dataset obtained from the [QML tutorial repository](https://github.com/qmlcode/tutorial). This can also be retrieved from the [quantum-machine webpage](http://www.quantum-machine.org/datasets/), but for its use with python, the dataset given by QML is more useful.
* The *QM9* dataset is obtained from the [quantum-machine webpage](http://www.quantum-machine.org/datasets/), but it's slightly modified for its use with python.
* On the other hand, the *periodic table of elements* data was retrieved from [this handy Gist](https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee).

*NOTE*: This is not supposed to be a python package (for now).
