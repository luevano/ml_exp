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
from colorama import init, Fore, Style

init()


def printc(text, color):
    """
    Prints texts normaly, but in color. Using colorama.
    text: string with the text to print.
    color: color to be used, same as available in colorama.
    """
    color_dic = {'BLACK': Fore.BLACK,
                 'RED': Fore.RED,
                 'GREEN': Fore.GREEN,
                 'YELLOW': Fore.YELLOW,
                 'BLUE': Fore.BLUE,
                 'MAGENTA': Fore.MAGENTA,
                 'CYAN': Fore.CYAN,
                 'WHITE': Fore.WHITE,
                 'RESET': Fore.RESET}

    color_dic_keys = color_dic.keys()
    if color not in color_dic_keys:
        print(Fore.RED
              + '\'{}\' not found, using default color.'.format(color)
              + Style.RESET_ALL)
        actual_color = Fore.RESET
    else:
        actual_color = color_dic[color]

    print(actual_color + text + Style.RESET_ALL)
