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
import pandas as pd
from misc import printc
from read_qm7_data import read_qm7_data
from c_matrix import c_matrix_multiple
from lj_matrix import lj_matrix_multiple
from do_ml import do_ml


# Test
def ml():
    """
    Main function that does the whole ML process.
    """
    # Initialization time.
    init_time = time.perf_counter()

    # Data reading.
    zi_data, molecules, nuclear_charge, energy_pbe0, energy_delta =\
        read_qm7_data()

    # Matrices calculation.
    procs = []
    pipes = []

    # cm_recv, cm_send = Pipe(False)
    # p1 = Process(target=c_matrix_multiple,
    #              args=(molecules, nuclear_charge, cm_send))
    # procs.append(p1)
    # pipes.append(cm_recv)
    # p1.start()

    ljm_recv, ljm_send = Pipe(False)
    p2 = Process(target=lj_matrix_multiple,
                 args=(molecules, nuclear_charge, ljm_send, 1, 0.25))
    procs.append(p2)
    pipes.append(ljm_recv)
    p2.start()

    # cm_data = pipes[0].recv()
    ljm_data = pipes[0].recv()

    for proc in procs:
        proc.join()

    # ML calculation.
    procs = []
    # cm_pipes = []
    ljm_pipes = []
    for i in range(1500, 6500 + 1, 500):
        # cm_recv, cm_send = Pipe(False)
        # p1 = Process(target=do_ml,
        #              args=(cm_data, energy_pbe0, i, 'CM', cm_send))
        # procs.append(p1)
        # cm_pipes.append(cm_recv)
        # p1.start()

        ljm_recv, ljm_send = Pipe(False)
        p2 = Process(target=do_ml,
                     args=(ljm_data, energy_pbe0, i, 'L-JM', ljm_send))
        procs.append(p2)
        ljm_pipes.append(ljm_recv)
        p2.start()

    # cm_bench_results = []
    ljm_bench_results = []
    for ljd_pipe in ljm_pipes:  # cd_pipe, ljd_pipe in zip(cm_pipes, ljm_pipes):
        # cm_bench_results.append(cd_pipe.recv())
        ljm_bench_results.append(ljd_pipe.recv())

    for proc in procs:
        proc.join()

    with open('benchmarks.csv', 'a') as save_file:
        # save_file.write(''.join(['ml_type,tr_size,te_size,kernel_s,',
        #                          'mae,time,lj_s,lj_e,date_ran\n']))
        date = '/'.join([str(field) for field in time.localtime()[:3][::-1]])
        for ljm in ljm_bench_results:  # cm, ljm, in zip(cm_bench_results, ljm_bench_results):
            # cm_text = ','.join([str(field) for field in cm])\
            #     + ',' + date + '\n'
            ljm_text = ','.join([str(field) for field in ljm])\
                + ',1,0.25,' + date + '\n'
            # save_file.write(cm_text)
            save_file.write(ljm_text)

    # End of program
    end_time = time.perf_counter()
    printc('Program took {:.4f} seconds.'.format(end_time - init_time),
           'CYAN')


def pl():
    """
    Function for plotting the benchmarks.
    """
    or_cols = ['ml_type',
               'tr_size',
               'te_size',
               'kernel_s',
               'mae',
               'time',
               'lj_s',
               'lj_e',
               'date_ran']
    dor_cols = ['te_size',
                'kernel_s',
                'time',
                'date_ran']

    data_temp = pd.read_csv('benchmarks.csv',)
    data = pd.DataFrame(data_temp, columns=or_cols)
    data = data.drop(columns=dor_cols)
    # print(data)

    first_data = pd.DataFrame(data, index=range(0, 22))
    first_data = first_data.drop(columns=['lj_s', 'lj_e'])

    fd_columns = ['ml_type', 'tr_size', 'mae']
    first_data_cm = pd.DataFrame(columns=fd_columns)
    first_data_ljm = pd.DataFrame(columns=fd_columns)
    for i in range(first_data.shape[0]):
        temp_df = first_data.iloc[[i]]
        if first_data.at[i, 'ml_type'] == 'CM':
            first_data_cm = first_data_cm.append(temp_df)
        else:
            first_data_ljm = first_data_ljm.append(temp_df)
    first_data_cm = first_data_cm.drop(columns=['ml_type']).rename(columns={'mae': 'cm_mae'})
    first_data_ljm = first_data_ljm.drop(columns=['ml_type']).rename(columns={'mae': 'ljm_mae'})
    print(first_data_cm)
    print(first_data_ljm)

    cm_axis = first_data_cm.plot(x='tr_size',
                                 y='cm_mae',
                                 kind='line')
    plot_axis = first_data_ljm.plot(ax=cm_axis,
                                    x='tr_size',
                                    y='ljm_mae',
                                    kind='line')
    plot_axis.set_xlabel('tr_size')
    plot_axis.set_ylabel('mae')
    plot_axis.set_title('mae for different tr_sizes')
    plot_axis.get_figure().savefig('data\\figs\\mae_diff_tr_sizes.pdf')

    new_data = data.drop(index=range(0, 22))
    new_data = new_data.drop(columns=['ml_type'])
    # print(new_data)


if __name__ == '__main__':
    # ml()
    pl()
