#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import ctutils
import sys

def main():
    if (len(sys.argv) < 2):
        print("Usage: {} image-file...".format(sys.argv[0]))
        return
    paths = sys.argv[1:]
    for p in paths:
        img = Image.open(p)
        basename = ".".join(p.split(".")[:-1])
        if basename == "":
            basename = p
        ctutils.save_rawimage(img.convert("F"), "{}_{}x{}_f.dat".format(basename, img.width, img.height))


if __name__ == '__main__':
    main()
