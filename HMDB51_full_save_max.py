import cv2 as cv
import numpy as np
import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import random
import glob
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile 
import shutil

SPLIT_PATH = 'hmdbTrainTestlist/testlist03.txt'
LABEL_PATH = 'hmdbTrainTestlist/classInd.txt'
TOP_DIR = 'HMDB51_img_mul12_top8'
MID_DIR = 'HMDB51_img_mul12_mid35'
NECK_DIR = 'HMDB51_img_mul12_NECK'
OUT_DIR = 'HMDB51_img_mul12_NECK_OUT'
# pb_file_path = 'full_ucfs1.pb'
test_dir = {'neck':NECK_DIR, 'top':TOP_DIR, 'mid':MID_DIR}
target_dir = {'neck':NECK_DIR+'_max', 'top':TOP_DIR+'_max', 'pre_out':OUT_DIR+'_max', 'mid':MID_DIR+'_max'}
# 'neck', 'pre_out', 'top', 'if_train'
shape = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288),'pre_out':(51)}
max_file = 'HMDB51_neck_max_s3.txt'


class convert_max():
    def __init__(self, test_dir, target_dir, max_file):
        self.sourse_dir = test_dir
        self.target_dir = target_dir
        self.max_file = max_file
        self.read_max_list()
        print("has read")
        for kk in test_dir.keys():
            self.copy2target(self.sourse_dir[kk], self.target_dir[kk])
    def initial(self,dic):
        keys = dic.keys()
        for key in keys:
            dic[key] = []
    def copy2target(self, sourse_dir, target_dir):
        for path, index, class_name in self.max_info:
            path_2 = '*'.join(path.split('['))
            glob_path = os.path.join(sourse_dir,class_name,path_2)+"*"
            glob_list = sorted(glob.glob(glob_path))
            target_path = os.path.join(target_dir,class_name,path)+'avi.txt'
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
            print(os.path.dirname(target_path))
            shutil.copyfile(glob_list[index], target_path)
    def read_max_list(self):
        self.max_info = []
        with open(self.max_file) as f:
            line = f.readline()
            while len(line) > 0:
                lines = line.split(',')
                index = int(lines[1])
                path = os.path.basename(lines[0])+'.'
                class_name = os.path.basename(os.path.dirname(lines[0]))
                self.max_info.append((path, index, class_name))
                line = f.readline()

convert_max(test_dir, target_dir, max_file)
