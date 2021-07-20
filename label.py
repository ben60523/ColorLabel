import cv2
import json
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

def generate_mask_by_color(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # find green contour
    lower_green = np.array([38, 43, 46])
    upper_green = np.array([75, 255, 255])

    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([37, 255, 255])

    # yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # bool_mask = np.where((yellow_mask == 255), 1, 0).astype(int)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bool_mask = np.where((green_mask == 255), 1, 0).astype(int)
    
    # fill green contour
    fill_mask = fill_contours_2(bool_mask) * 255
    # remove noise
    kernel = np.ones((9, 9), np.uint8)
    fill_mask = fill_mask.astype(np.uint8)
    output_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel)
    output_mask = cv2.dilate(output_mask, kernel)

    return output_mask

def generate_mask_by_trajectory(tag_list, nb_tags, mask_shape):
    mask = np.zeros(mask_shape)
    fill_mask = np.zeros(mask_shape,np.uint8)

    for i in range(nb_tags):
        point_list = tag_list[i]['points']
        for point in point_list:
            mask[point['top'], point['left']] = 1
        fill_mask = fill_contours(mask.astype(np.uint8)) * 255
    kernel = np.ones((2, 2), np.uint8)
    output_mask = cv2.dilate(fill_mask, kernel)
    return output_mask

def LISTDIR(path):
    out = os.listdir(path)
    out.sort()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument('-c', '--color', help='Generate mask with green color in the Image', action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    if args.color:
        print('Generate mask with green color in the image')
        for file_name in tqdm(LISTDIR(args.input_path)):
            full_path = os.path.join(args.input_path, file_name)
            out_file_path = os.path.join(args.output_path, file_name)
            file_name = file_name.split(".")[0]

            if os.path.isfile(full_path):
                img = cv2.imread(full_path)
                mask = generate_mask_by_color(img)
                cv2.imwrite(out_file_path, mask)
    else:
        print('Generate mask with trajectory')
        try:
            files = LISTDIR(args.input_path)
            index = files.index('pages.json')
            if index < 0:
                Exception('pages.json is missing')
            index = files.index('labels.json')
            if index < 0:
                Exception('label.json is missing')
        except Exception as e:
            print(e)
            exit(-1)

        pages = json.load(open(os.path.join(args.input_path, 'pages.json')))
        for img in tqdm(pages):
            tag_list = img['tags']
            nb_tags = len(tag_list)
            if nb_tags > 0:
                name = img['name']
                try:
                    img = cv2.imread(os.path.join(args.input_path, name))
                    h, w, ch = img.shape
                except:
                    print('\nThe image file is missing (%s)\n'%name)
                    exit(-1)
                
                labels_list = json.load(open(os.path.join(args.input_path, 'labels.json')))
                mask_shape = (h, w)
                congestion_mask = np.zeros(mask_shape,np.uint8)
                lesion_mask = np.zeros(mask_shape,np.uint8)
                normal_mask = np.zeros(mask_shape,np.uint8)
                for i in range(nb_tags):
                    point_list = tag_list[i]['points']
                    tag_id = tag_list[i]['labelID']
                    tag_name = ''
                    for label in labels_list:
                        if label['key'] == tag_id:
                            tag_name = label['title']
                            break
                    
                    dilate_matrix = np.arange(-9, 9)
                    points = np.empty((1,2))

                    for point in point_list:
                        points = np.append(points, [[point['left'], point['top']]], 0)

                    if tag_name == 'congestion':
                        cv2.polylines(congestion_mask, np.int32([points]), False, (255, 255, 255))
                    elif tag_name == 'lesion':
                        cv2.polylines(lesion_mask, np.int32([points]), True, (255, 255, 255))
                    elif tag_name == 'normal':
                        cv2.polylines(normal_mask, np.int32([points]), True, (255, 255, 255))
                
                kernel = np.ones((3, 3), np.uint8)
                if congestion_mask.max() > 0:
                    fill_mask = fill_contours(congestion_mask)
                    output_mask = cv2.dilate(fill_mask, kernel)
                    cv2.imwrite(os.path.join(args.output_path, name[:(len(name)-4)] + '_congestion' + name[(len(name)-4):]), output_mask)
                    
                if lesion_mask.max() > 0:
                    fill_mask = fill_contours(lesion_mask)
                    output_mask = cv2.dilate(fill_mask, kernel)
                    cv2.imwrite(os.path.join(args.output_path, name[:(len(name)-4)] + '_lesion' + name[(len(name)-4):]), output_mask)
                    
                if normal_mask.max() > 0:
                    fill_mask = fill_contours(normal_mask)
                    output_mask = cv2.dilate(fill_mask, kernel)
                    cv2.imwrite(os.path.join(args.output_path, name[:(len(name)-4)] + '_normal' + name[(len(name)-4):]), output_mask)
