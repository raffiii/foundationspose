import cv2
import numpy
import os
import argparse

def mask_object(path, file):
    rgb_path = f"{path}/rgb/{file}"
    mask_path = f"{path}/mask/{file}"
    rgb = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path)
    if rgb is not None and mask is not None:
        masked = rgb & mask
        cv2.imshow(rgb_path, masked)
        cv2.imshow("unmasked", rgb)
        cv2.waitKey(0)
    else:
        print(f"---------- WARN --------------")
        print(f"path {path} is not loading")

def show_all(path, file='0000001.png'):
    dirs = os.listdir(path)
    for d in dirs:
        p = f"{path}/{d}"
        mask_object(p,file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-f", "--file", type=str, default = "0000001.png")

    opts = parser.parse_args()
    show_all(opts.path)