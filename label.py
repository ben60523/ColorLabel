import cv2
import numpy as np
import colorList
import os
from tqdm import tqdm
import argparse


check_ll = [-2, -1, 0, 1, 2]
def get_color(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    color_dict = colorList.getColorList()
    yellow_point_num = 0
    green_point_num = 0
    yellow_point_list = []
    green_point_list = []
    for color_which, d in enumerate(color_dict):
        if color_which == 0:
            yellow_mask = cv2.inRange(hsv,color_dict[d][0], color_dict[d][1])
            for i in range(yellow_mask.shape[0]):
                for j in range(yellow_mask.shape[1]):
                    if yellow_mask[i][j] == 255:
                        check_num = 0
                        for x in check_ll:
                            for y in check_ll:
                                if yellow_mask[i+x][j+y] == 255:
                                    check_num += 1
                        if check_num >= 10:        
                            yellow_point_num += 1
                            yellow_point_list.append([j,i])
                        else:
                            yellow_mask[i][j] = 0
        if color_which == 1:
            green_mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            for i in range(green_mask.shape[0]):
                for j in range(green_mask.shape[1]):
                    if green_mask[i][j] == 255:
                        
                        check_num = 0
                        for x in check_ll:
                            for y in check_ll:
                                if green_mask[i + x][j + y] == 255:
                                    check_num += 1
                        if check_num >= 10:
                            green_point_num += 1
                            green_point_list.append([j, i])
                        else:
                            green_mask[i][j] = 0
    if yellow_point_num >= green_point_num:
        return yellow_mask, np.array(yellow_point_list)
    else:
        return green_mask, np.array(green_point_list)

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
    for num_files in tqdm(LISTDIR(args.input_path)):
        full_path = os.path.join(args.input_path, num_files)
        out_file_path = os.path.join(args.output_path, num_files)

        if os.path.isfile(full_path):
            frame = cv2.imread(full_path)
            gs_img, point_list = get_color(frame)
            if point_list.size == 0:
                cv2.imwrite(out_file_path, frame)
            else:
                cv2.fillPoly(gs_img, [point_list], 255) 
                cv2.imwrite(out_file_path, gs_img)                  