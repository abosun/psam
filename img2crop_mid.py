import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import linalg as la      
import os
import tensorflow as tf


data_set_dir = 'ucf_101_top5_sum'
IMG_DIR = data_set_dir+'_crop_mid'
img_n = 0
img_nums =20
#def frame_pool(frame):
# Get the total number of video framms , the input is the path of video, the output is the total number. 
def load_dataset(data_set_dir):
    dataset_dict = {}
    import glob
    sub_dirs = [x[0] for x in os.walk(data_set_dir)][1:]
    sub_names = [os.path.basename(x) for x in sub_dirs]
    class_n = len(sub_dirs)
    for class_i in range(class_n):
        glob_path = os.path.join(sub_dirs[class_i],'*.jpg')
        avi_list = glob.glob(glob_path)
        dataset_dict[sub_names[class_i]] = avi_list
    return dataset_dict

def img_crop(img, size=240):
    w,h,dim=img.shape
    return img[(w//2-size//2):(w//2+size//2),:,:]

def main():
    dataset_dict = load_dataset(data_set_dir)
    for action in dataset_dict:
        img_out_dir = os.path.join(IMG_DIR, action)
        if not os.path.exists(img_out_dir): os.makedirs(img_out_dir)
        for img_path in dataset_dict[action]:
            img = cv.imread(img_path)
            img_name = os.path.basename(img_path)
            new_path = os.path.join(img_out_dir, img_name)
            print(new_path)
            cv.imwrite(new_path, img_crop(img,240))

if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()
