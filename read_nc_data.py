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
# 'periodic_table_of_elements.txt' retrieved from
# https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee


def read_nc_data(data_path):
    """
    Reads nuclear charge data from file and returns a dictionary.
    data_path: path to the data directory.
    """
    fname = 'periodic_table_of_elements.txt'
    with open(''.join([data_path, '\\', fname]), 'r') as infile:
        temp_lines = infile.readlines()

    del temp_lines[0]

    lines = []
    for temp_line in temp_lines:
        new_line = temp_line.split(sep=',')
        lines.append(new_line)

    # Dictionary of nuclear charge.
    return {line[2]: int(line[0]) for line in lines}
