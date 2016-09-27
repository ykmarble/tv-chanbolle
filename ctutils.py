#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import sys
import struct
import os.path
import struct

def load_rawimage(path):
    with open(path, "rb") as f:
        header = struct.unpack("ccxxII", f.read(12))
        if not (header[0] == b"P" and header[1] == b"0"):
            print("Invalied file.")
            sys.exit(1)
        width = header[2]
        height = header[3]
        print(width, height)
        img_seq = struct.unpack("{}f".format(width*height), f.read())
    img = Image.new("F", (width, height))
    img.putdata(img_seq)
    return img

def save_rawimage(img, outpath):
    img_seq = img.getdata()
    header = struct.pack("ccxxII", b"P", b"0", img.width, img.height)
    payload = struct.pack("{}f".format(len(img_seq)), *img_seq)
    with open(outpath, "wb") as f:
        f.write(header)
        f.write(payload)
