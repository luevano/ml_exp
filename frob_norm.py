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
