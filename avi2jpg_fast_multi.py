from __future__ import division
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import linalg as la      
import os
import tensorflow as tf
import multiprocessing
from multiprocessing import Process, current_process

nums = 10
data_set_dir = 'HMDB51'
IMG_DIR = data_set_dir+"_img_mul"+str(nums)
img_n = 0
jc = 20
#def frame_pool(frame):
# Get the total number of video framms , the input is the path of video, the output is the total number. 
def load_dataset(data_set_dir):
    dataset_dict = {}
    import glob
    sub_dirs = [x[0] for x in os.walk(data_set_dir)][1:]
    sub_names = [os.path.basename(x) for x in sub_dirs]
    class_n = len(sub_dirs)
    for class_i in range(class_n):
        glob_path = os.path.join(sub_dirs[class_i],'*.avi')
        avi_list = glob.glob(glob_path)
        dataset_dict[sub_names[class_i]] = avi_list
    return dataset_dict
class Video():
    def __init__(self, video_path):
        self.video_path = video_path
        self.nums = -1
        self.get_shape()
    def get_shape(self):
        video = cv.VideoCapture(self.video_path)
        ret, frame = video.read()
        if not ret : raise ValueError
        self.shape = frame.shape
    def frame_count(self):
        video = cv.VideoCapture(self.video_path)
        self.nums = 0
        while 1:
            ret, frame = video.read()
            if not ret : break
            self.nums += 1
        return self.nums
    def get_frame(self):
        if self.nums<0: self.frame_count()
        video = cv.VideoCapture(self.video_path)
        for i in range(self.nums//2):
            ret, frame = video.read()
        self.center_frame = frame
        return frame
    def get_frame_multi(self, num):
        if self.nums<0: self.frame_count()
        video = cv.VideoCapture(self.video_path)
        stride = self.nums // num
        frame_n = -1
        frame_list = []
        for i in range(num):
            frame_n += 1
            ret, frame = video.read()
            while frame_n%stride != 0:
                frame_n += 1
                ret, frame = video.read()
            frame_list.append(frame)
        return frame_list
class Video2img(Video):
    def __init__(self, video_path,sub_dir):
        self.frame_n=0
        Video.__init__(self,video_path)
        self.video_path = video_path
        self.sub_dir = sub_dir
    def img_write(self):
        self.frame_path = os.path.join(IMG_DIR, self.sub_dir, os.path.basename(self.video_path)+'.jpg')
        frame = Video.get_frame(self)
        frame_resize = cv.resize(frame, (299,299))
        cv.imwrite(self.frame_path,frame_resize) 
    def multi_img_write(self,num):
        frame_list = Video.get_frame_multi(self,num)
        for frame in frame_list:
            self.frame_path = os.path.join(IMG_DIR, self.sub_dir, os.path.basename(self.video_path)+"%02d"%(self.frame_n)+'.jpg')
            frame_resize = cv.resize(frame, (299,299))
            cv.imwrite(self.frame_path,frame_resize) 
            self.frame_n += 1
def split_i(action,dataset_dict):
    img_dir = os.path.join(IMG_DIR, action)
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    for video_path in dataset_dict[action]:
        try:
            Video2img(video_path, action).multi_img_write(nums) 
        except:
            print(video_path)
def main():
    dataset_dict = load_dataset(data_set_dir)
    pool = multiprocessing.Pool(processes=jc)
    for action in dataset_dict:
        pool.apply_async(split_i, (action, dataset_dict))
    pool.close()
    pool.join()

if __name__ == '__main__' :
    main()
