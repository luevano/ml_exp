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
import math


def frob_norm(array):
    """
    Calculates the frobenius norm of a given array or matrix.
    array: array of data.
    """

    arr_sh_len = len(array.shape)
    arr_range = range(len(array))
    fn = 0.0

    # If it is a 'vector'.
    if arr_sh_len == 1:
        for i in arr_range:
            fn += array[i]*array[i]

        return math.sqrt(fn)

    # If it is a matrix.
    elif arr_sh_len == 2:
        for i in arr_range:
            for j in arr_range:
                fn += array[i, j]*array[i, j]

        return math.sqrt(fn)
    else:
        print('Error. Array size greater than 2 ({}).'.format(arr_sh_len))
