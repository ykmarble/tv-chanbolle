#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctutils
from PIL import Image
import numpy
import sys
import os.path

def main():
    if (len(sys.argv) != 2):
        print("Usage: {} rawfile".format(sys.argv[0]))
        return
    path = sys.argv[1]
    img = ctutils.load_rawimage(path)
    img_array = numpy.array(img.getdata())
    img_array += numpy.random.normal(0, 20, img.width * img.height)
    img_array = numpy.maximum(img_array, 0)
    img_array = numpy.minimum(img_array, 255)
    noise_img = Image.new("F", (img.width, img.height))
    noise_img.putdata(img_array)
    basename = ".".join(path.split(".")[:-1])
    if basename == "":
        basename = path
    ctutils.save_rawimage(noise_img, "{}_noised.dat".format(basename))

if __name__ == '__main__':
    main()
