import cv2
import numpy as np
import collections
import os
from tqdm import tqdm
import argparse


def getColorList():
    dict = collections.defaultdict(list)
    #yellow
    # lower_yellow = np.array([26, 43, 46])
    lower_yellow = np.array([23, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list
    #green
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list
    return dict

def fill_contours(arr):
    return np.maximum.accumulate(arr, 1) &\
           np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1] &\
           np.maximum.accumulate(arr[::-1, :], 0)[::-1, :] &\
           np.maximum.accumulate(arr, 0)

def mask_generate(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # find green contour
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    bool_mask = np.where((green_mask == 255), 1, 0).astype(int)
    
    # fill green contour
    fill_mask = fill_contours(bool_mask) * 255
    
    # remove noise
    kernel = np.ones((7, 7), np.uint8)
    fill_mask = fill_mask.astype(np.uint8)
    output_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel)

    return output_mask

def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    for file_name in tqdm(LISTDIR(args.input_path)):
        full_path = os.path.join(args.input_path, file_name)
        out_file_path = os.path.join(args.output_path, file_name)
        file_name = file_name.split(".")[0]

        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            mask = mask_generate(img)
            cv2.imwrite(out_file_path, mask)
            