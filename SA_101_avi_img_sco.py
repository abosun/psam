import cv2 as cv
import numpy as np
import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import random
import glob
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile 

import multiprocessing
from multiprocessing import Process, current_process
#blog: xiaorui.cc
import copy_reg
import types

data_test_dir   = 'ucf_101_img_mul12_NECK'
SPLIT_PATH = 'ucfTrainTestlist/testlist03.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
MODEL_DIR = 'models/'
SA_MODEL_FILE = 'SAH_rtr_JT_s3_10001.pb'
V3_MODEL_FILE = 'classify_image_graph_def.pb'
GPU = '3'
IMG_DTR = 'ucf_101_img_mulx20'
TOP_DIR = 'ucf_101_img_test10_top8'
MID_DIR = 'ucf_101_img_test10_mid35'
NECK_DIR = 'ucf_101_img_test10_NECK'
OUT_DIR = 'ucf_101_img_test10_NECK_OUT'
dir_list = [NECK_DIR, TOP_DIR, MID_DIR]
shape = [2048, 8*8*1280, 35*35*288]
max_n = 8
sum_frame = 20

# img_list, video_n, class_n = avi2list(avi, label_dict)
def avi2list(avi, label_dict):
    class_n = os.path.basename(os.path.dirname(avi))
    glob_path = os.path.join(IMG_DTR, avi) + '*.jpg'
    img_list = sorted(glob.glob(glob_path))
    # print(img_list[0])
    return [cv.imread(x) for x in img_list], os.path.basename(avi), class_n

def read_split():
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_list = sorted([os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x).split(',')[0]) for x in split_list])
    # split_set = set([os.path.basename(x).split('.')[0] for x in split_list])
    return split_list

def get_class_dict():
    label_dict = {}
    with open(LABEL_PATH , 'r') as label_file:
        label_string = label_file.read()
        label_list = label_string.split('\r\n')[:-1]
    for x in label_list:
        index = int(x.split(' ')[0])-1
        name = x.split(' ')[1]
        label_dict[index] = name
        label_dict[name] = index
    return label_dict

def read_data(path, shape):
    with open(path) as f:
        line = f.readline().split(',')
        data = [float(x) for x in line]
    return np.reshape(np.array(data), shape)

class get_test_data():
    def __init__(self):
        jc = 20
        self.get_class_dict()
        self.test_data = []
        self.test_lable = []
        self.test_list = []
        self.test_read_i = -1
        self.test_read_len = 0
        threads = []
        with open(SPLIT_PATH , 'r') as split_file:
            split_string = split_file.read()
            self.split_list = split_string.split('\r\n')
            self.test_read_len = len(self.split_list)
        pool = multiprocessing.Pool(processes=jc)
        result = [0]*self.test_read_len
        for i in range(self.test_read_len):
            result[i] = pool.apply_async(self.read_i, (i, ))
        pool.close()
        pool.join()
        for i in range(len(result)):
            result[i] = result[i].get()
        self.result = result
    def read_i(self,i):
        line = self.split_list[i]
        # print(i)
        data_one = {'neck':[], 'top':[], 'pre_out':[]}
        action = os.path.dirname(line)
        for kk in test_dir.keys():
            glob_path = os.path.join(test_dir[kk], line.split('.')[0]+'*.txt')
            glob_list = sorted(glob.glob(glob_path))
            data_one[kk] = [read_data(x, shape[kk]) for x in glob_list]
        return (data_one,glob_list,self.label_dict[action])
    def data(self):
        return self.test_data, self.test_list, self.test_lable
    def get_class_dict(self):
        self.label_dict = {}
        with open(LABEL_PATH , 'r') as label_file:
            label_string = label_file.read()
            label_list = label_string.split('\r\n')[:-1]
        for x in label_list:
            index = int(x.split(' ')[0])-1
            name = x.split(' ')[1]
            self.label_dict[index] = name
            self.label_dict[name] = index
        return self.label_dict

def lsit_tranpose(a):
    return [[row[i] for row in a] for i in range(len(a[0]))]

def cuda_set(gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.allow_growth = True
    return config

def main():
    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, SA_MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    # with graph.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, V3_MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    # test_data = get_test_data().result
    label_dict = get_class_dict()
    # print(label_dict)
    avilist = read_split()
    # print(avilist)
    with tf.Session(config=cuda_set(gpu=GPU), graph=graph) as sess:
        names = [op.name for op in sess.graph.get_operations()]
        # print(names)
        result = []
        v3_mid = sess.graph.get_tensor_by_name("mixed_2/join:0")
        v3_neck = sess.graph.get_tensor_by_name("pool_3/_reshape:0")
        v3_top = sess.graph.get_tensor_by_name("mixed_8/join:0")
        v3_img = sess.graph.get_tensor_by_name("DecodeJpeg:0")

        sa_neck = sess.graph.get_tensor_by_name("neck:0")
        sa_top = sess.graph.get_tensor_by_name("top:0")
        if_train = sess.graph.get_tensor_by_name("if_train:0")
        soft_tensor = sess.graph.get_tensor_by_name("softmax:0")
        def get_out(input_img):
            neck, top, mid =  sess.run([v3_neck ,v3_top, v3_mid],feed_dict = {v3_img:input_img})
            out = sess.run([soft_tensor],feed_dict = {sa_neck:neck, sa_top:top, if_train:False})
            return out[0][0], neck, top, mid
        for avi in avilist:
            print(avi)
            img_list, video_n, class_n = avi2list(avi, label_dict)
            class_id = label_dict[class_n]
            # print(img_list)
            # print(len(img_list))
            if len(img_list) != sum_frame :raise ValueError
            img_datas = [0]*sum_frame
            outs = []
            out_index = []
            list_30 = list(range(sum_frame))
            for i in range(sum_frame):
                out_i = 0
                img_datas[i] = get_out(img_list[i])
                outs.append(list(img_datas[i][0]))
            outs = lsit_tranpose(outs)
            inde = zip(outs[class_id], list_30)
            inde = sorted(inde)
            # raise ValueError
            for dir_i in range(len(dir_list)):
                file_dir = dir_list[dir_i]
                target_dir = os.path.join(file_dir, class_n)
                if not os.path.exists(target_dir):os.makedirs(target_dir)
                target_base_name = os.path.join(file_dir, class_n, video_n)
                for i in range(max_n):
                    big_i = inde[sum_frame-i-1][1]
                    target_name = target_base_name + '.%02d'%(i) +'.txt'
                    line_str = ','.join([str(x) for x in list(np.reshape(img_datas[big_i][dir_i+1], (shape[dir_i])))])
                    with open(target_name,'w') as f:
                        f.write(line_str)

if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()