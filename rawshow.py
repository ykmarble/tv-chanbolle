#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctutils
import sys

def main():
    if (len(sys.argv) != 2):
        print("Usage: {} rawfile".format(sys.argv[0]))
        return
    path = sys.argv[1]
    img = ctutils.load_rawimage(path)
    img.show()

if __name__ == '__main__':
    main()
