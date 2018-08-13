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
def _pickle_method(m):
    if m.im_self is None: return getattr, (m.im_class, m.im_func.func_name)
    else:return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _pickle_method)

data_test_dir   = 'ucf_101_img_mul12_NECK'
data_train_dirs  = ['ucf_101_img_centre_NECK','ucf_101_img_mul12_NECK','ucf_101_img_centre_top8_F_NECK']
CLASS_NUM = 101
BATCH_SIZE = 50
learning_rate = 0.0001
ITEMS = 3000
drop_rate = 0.9
SPLIT_PATH = 'ucf_all_list.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
MODEL_DIR = 'models/'
MODEL_FILE = 'SA101_chuanmax_ss_s1_sigmoid10901.pb'
GPU = '0'
# TOP_DIR = 'ucf_101_img_mul12_top8'
# MID_DIR = 'ucf_101_img_mul12_mid35'
# NECK_DIR = 'ucf_101_img_mul12_NECK'
# OUT_DIR = 'ucf_101_img_mul12_NECK_OUT'

TOP_DIR = 'ucf_101_img_test10_top8'
MID_DIR = 'ucf_101_img_test10_mid35'
#TOP_DIR = 'ucf_101_img_centre_top8'
#MID_DIR = 'ucf_101_img_centre_mid35'

NECK_DIR = 'ucf_101_img_test10_NECK'
OUT_DIR = 'ucf_101_img_test10_NECK_OUT'

target_dir = 'ucf_101_feat_maxgai_ss_sigmod/'

# pb_file_path = 'full_ucfs1.pb'
test_dir = {'top':TOP_DIR, 'mid':MID_DIR}
shape = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288),'pre_out':(101)}

def read_split():
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_set = set([ os.path.basename(x).split('.')[0] for x in split_list])
    return split_set

def read_data(path, shape):
    with open(path) as f:
        line = f.readline().split(',')
        data = [float(x) for x in line]
    return np.reshape(np.array(data), shape)

class get_test_data():
    def __init__(self):
        self.jc = 24
        self.global_c = 0
        self.get_class_dict()
        self.test_data = []
        self.test_lable = []
        self.test_list = []
        self.test_read_i = -1
        self.test_read_len = 0
        threads = []
        with open(SPLIT_PATH , 'r') as split_file:
            split_string = split_file.read()
            self.split_list = [x.split(' ')[0] for x in split_string.split('\r\n')]
            self.test_read_len = len(self.split_list)
    def get_data(self):
        pool = multiprocessing.Pool(processes=self.jc)
        result = [0]*(self.test_read_len//10)
        for i in range(self.test_read_len//10*self.global_c, self.test_read_len//10*(self.global_c+1)):
            result[i-self.test_read_len//10*self.global_c] = pool.apply_async(read_i, ((i, self.split_list[i], self.label_dict)))
        pool.close()
        pool.join()
        for i in range(len(result)):
            result[i] = result[i].get()
        self.result = result
        self.global_c += 1
        return result
    def read_i(self,i):
        line = self.split_list[i]
        #print(i)
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

def read_i(i, line, label_dict):
    #print(i)
    data_one = {'neck':[], 'top':[], 'pre_out':[]}
    action = os.path.dirname(line)
    for kk in test_dir.keys():
        glob_path = os.path.join(test_dir[kk], line.split('.')[0]+'*.txt')
        glob_list = sorted(glob.glob(glob_path))
        data_one[kk] = [read_data(x, shape[kk]) for x in glob_list]
    return (data_one,glob_list,label_dict[action])

def cuda_set(gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.allow_growth = True
    return config

def featext_i(test_data, gpu):
    print(len(test_data))
    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    with tf.Session(config=cuda_set(gpu=gpu), graph=graph) as sess:
        names = [op.name for op in sess.graph.get_operations()]
        # print(names)
        result = []
        right = 0.0
        right_2 = 0.0
        right_h5 = 0.0
        right_fusion = 0.0
        total = 0.0
        jishu = [0]*12
        names = [op.name for op in sess.graph.get_operations()]
        mid = sess.graph.get_tensor_by_name("mid:0")
        top = sess.graph.get_tensor_by_name("top:0")
        soft_tensor = sess.graph.get_tensor_by_name("softmax:0")
        if_train = sess.graph.get_tensor_by_name("is_training:0")
        Z_total = sess.graph.get_tensor_by_name("concat:0")
        
        for xx in test_data:
            data_list = xx[0]
            path_list = xx[1]
            label = xx[2]
            total += 1
            #
            feature = sess.run([Z_total], feed_dict={top:data_list['top'], mid:data_list['mid'], 
                if_train:False})
            soft = sess.run([soft_tensor], feed_dict={Z_total:feature[0], if_train:False})

            lv = np.transpose(soft[0])[label]
            index = np.argmax(lv)

            base_name = os.path.basename(path_list[index]).split('.')[0]
            class_name = os.path.basename(os.path.dirname(path_list[index]))
            feat_data = feature[0][index]
            path = os.path.join(target_dir,class_name,base_name)+'.cnnfeat.txt'
            feat_str = ' '.join([str(x) for x in feat_data])
            if not os.path.exists(os.path.join(target_dir,class_name)):
                os.makedirs(os.path.join(target_dir,class_name))
            with open(path, 'w') as f:
                f.write(feat_str)

def main():
    mul_jc = 18
    data_base = get_test_data()
    for k in range(10):
        test_data = data_base.get_data()
        data_len = len(test_data)
        stride = data_len//mul_jc
        print(stride)
        test_data_list = []
        for i in range(mul_jc-1):
            test_data_list.append(test_data[i*stride:(i+1)*stride])
        test_data_list.append(test_data[(mul_jc-1)*stride:])
        print(len(test_data_list))
        print(len(test_data_list[0]))
        print(len(test_data_list[-1]))
        pool = multiprocessing.Pool(processes=mul_jc)
        gpu_id = 0
#        featext_i(test_data_list[0],'0')
        for i in range(mul_jc):
            # print(i)
            gpu = str(gpu_id%4)
            pool.apply_async(featext_i, (test_data_list[i], gpu))
            gpu_id=gpu_id+1;
        pool.close()
        pool.join()


if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()
