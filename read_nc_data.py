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
