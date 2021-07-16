import cv2
import numpy as np
import collections
import os
from tqdm import tqdm
import argparse


def fill_contours(arr):
    return np.maximum.accumulate(arr, 1) &\
           np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1] &\
           np.maximum.accumulate(arr[::-1, :], 0)[::-1, :] &\
           np.maximum.accumulate(arr, 0)


def fill_contours_2(arr):
    output_horizontal = np.zeros(arr.shape)
    output_vertical = np.zeros(arr.shape)
    h, w = arr.shape
    
    # horizontal scanning
    for y in range(h):
        start_point_forward = -1
        stop_point_forward = -1
        start_point_backward = -1
        stop_point_backward = -1

        for x in range(w):
            # from left to right
            if arr[y, x- 1] > 0 and arr[y, x] == 0:
                if start_point_forward < 0:
                    start_point_forward = x - 1
                else:
                    stop_point_forward = x - 1
            if start_point_forward > 0 and stop_point_forward > 0:
                output_horizontal[y, start_point_forward:stop_point_forward] = 1
                start_point_forward = stop_point_forward = -1

            # from right to left
            x_backward = w - x - 1
            if arr[y, x_backward] == 0 and arr[y, x_backward - 1] > 0:
                if stop_point_backward < 0:
                    stop_point_backward = x_backward
                else:
                    start_point_backward = x_backward
            if start_point_backward > 0 and stop_point_backward > 0:
                output_horizontal[y, start_point_backward:stop_point_backward] = 1
                start_point_backward = stop_point_backward = -1

    # vertical scanning
    for x in range(w):
        start_point_forward = -1
        stop_point_forward = -1
        start_point_backward = -1
        stop_point_backward = -1
        
        for y in range(h):
            # from top to bottom
            if arr[y - 1, x] > 0 and arr[y, x] == 0:
                if start_point_forward < 0:
                    start_point_forward = y - 1
                else:
                    stop_point_forward = y - 1
            if start_point_forward > 0 and stop_point_forward > 0:
                output_vertical[start_point_forward:stop_point_forward, x] = 1
                start_point_forward = stop_point_forward = -1

            # from bottom to top
            y_backward = h - y - 1
            if arr[y_backward - 1, x] == 0 and arr[y_backward, x] > 0:
                if stop_point_backward < 0:
                    stop_point_backward = y_backward
                else:
                    start_point_backward = y_backward

            if start_point_backward > 0 and stop_point_backward > 0:
                output_vertical[start_point_backward:stop_point_backward, x] = 1
                start_point_backward = stop_point_backward = -1
                
    return np.logical_and(output_horizontal, output_vertical)

def mask_generate(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # find green contour
    lower_green = np.array([38, 43, 46])
    upper_green = np.array([75, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    bool_mask = np.where((green_mask == 255), 1, 0).astype(int)
    
    # fill green contour
    fill_mask = fill_contours_2(bool_mask) * 255
    # remove noise
    kernel = np.ones((9, 9), np.uint8)
    fill_mask = fill_mask.astype(np.uint8)
    output_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel)
    output_mask = cv2.dilate(output_mask, kernel)

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
            