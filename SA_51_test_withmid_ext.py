import cv2 as cv
import numpy as np
import os
import random
import glob
import multiprocessing
from multiprocessing import Process, current_process
#blog: xiaorui.cc
data_test_dir   = 'ucf_101_img_mul12_NECK'
data_train_dirs  = ['ucf_101_img_centre_NECK','ucf_101_img_mul12_NECK','ucf_101_img_centre_top8_F_NECK']
CLASS_NUM = 101
BATCH_SIZE = 50
learning_rate = 0.0001
ITEMS = 3000
drop_rate = 0.9
FILE_PATH = 'hmdbTraintestsplits/hmdball_path_list.txt'
SPLIT_PATH = 'hmdbTrainTestlist/testlist01.txt'
LABEL_PATH = 'hmdbTraintestsplits/classInd.txt'
MODEL_DIR = 'models/'
MODEL_FILE = 'SA51_rtr_JT_s1_10751.pb'
GPU = '0'
# DIR = 'ucf_101_img_mul12'
# DIR = 'ucf_101_img_test10'
DIR = 'HMDB51_img_mul1'

TOP_DIR = DIR + '_top8'
MID_DIR = DIR + '_mid35'
NECK_DIR = DIR + '_NECK'

target_dir = 'HMDB51_feat_s1/'

# pb_file_path = 'full_ucfs1.pb'
test_dir = {'neck':NECK_DIR, 'top':TOP_DIR, 'mid':MID_DIR}
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
        self.jc = 20
        self.items = 10
        self.global_c = 0
        self.get_class_dict()
        self.split_set = read_split()
        with open(FILE_PATH , 'r') as split_file:
            split_string = split_file.read()
            self.split_list = [x.split(' ')[0] for x in split_string.split('\r\n')]
            self.lenth = len(self.split_list)
        self.multi_list = split_list(self.split_list, self.items)
    def get_data(self):
        pool = multiprocessing.Pool(processes=self.jc)
        result = [0]*len(self.multi_list[self.global_c])
        idex = 0
        for i in range(len(self.multi_list[self.global_c])):
            if not os.path.basename(self.multi_list[self.global_c][i]).split('.')[0] in self.split_set:
                result[idex] = pool.apply_async(read_i, ((i, self.multi_list[self.global_c][i], self.label_dict)))
                idex += 1
        pool.close()
        pool.join()
        data = []
        for i in range(idex):
            data.append(result[i].get())
        self.global_c += 1
        return data
    def get_class_dict(self):
        self.label_dict = {}
        with open(LABEL_PATH , 'r') as label_file:
            label_string = label_file.read()
            label_list = label_string.split('\r\n')
        if label_list[-1] == '':
            label_list = label_list[:-1]
        for x in label_list:
            index = int(x.split(' ')[0])-1
            name = x.split(' ')[1]
            self.label_dict[index] = name
            self.label_dict[name] = index
        return self.label_dict
def split_list(list_0, n):
    lenth = len(list_0)
    result = []
    stride = lenth // n
    for i in range(n-1):
        result.append(list_0[i*stride:(i+1)*stride])
    result.append(list_0[(n-1)*stride:])
    return result
def read_i(i, line, label_dict):
    try:
        data_one = {'neck':[], 'top':[], 'pre_out':[]}
        action = os.path.dirname(line)
        for kk in test_dir.keys():
            glob_path = os.path.join(test_dir[kk],action, os.path.basename(line).split('.')[0]+'.*.txt')
            glob_list = sorted(glob.glob('*'.join(glob_path.split('['))))
            data_one[kk] = [read_data(x, shape[kk]) for x in glob_list]
            if len(data_one[kk])!=1:
                print('----------------')
                print(len(data_one[kk]),glob_path)
        #print(glob_list[-1],label_dict[action],action)
    except:
        print(line)
    return (data_one,glob_list,label_dict[action])

def cuda_set(tf,gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.allow_growth = True
    return config

def featext_i(test_data, gpu):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    def cuda_set(gpu='3'):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.allow_growth = True
        return config
    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    with tf.Session(config=cuda_set(gpu=gpu), graph=graph) as sess:
        neck = sess.graph.get_tensor_by_name("neck:0")
        # mid = sess.graph.get_tensor_by_name("mid:0")
        top = sess.graph.get_tensor_by_name("top:0")
        soft_tensor = sess.graph.get_tensor_by_name("softmax:0")
        if_train = sess.graph.get_tensor_by_name("if_train:0")
        Z_total = sess.graph.get_tensor_by_name("ZN_comb:0")
        for xx in test_data:
            df = xx[1][0]
            data_list = xx[0]
            path_list = xx[1]
            label = xx[2]
            #
            try:
                feature = sess.run([Z_total], feed_dict={neck:data_list['neck'], top:data_list['top'],# mid:data_list['mid'], 
                    if_train:False})
            except:
                print('*****', path_list)
            base_name = os.path.basename(path_list[0]).split('.')[0]
            class_name = os.path.basename(os.path.dirname(path_list[0]))
            feat_data = feature[0].mean(axis=0)
            path = os.path.join(target_dir,class_name,base_name)+'.cnnfeat.txt'
            if os.path.exists(path) and len(open(path).read())>3000:
                continue
            feat_str = ' '.join([str(x) for x in feat_data])
            if not os.path.exists(os.path.join(target_dir,class_name)):
                os.makedirs(os.path.join(target_dir,class_name))
            with open(path, 'w') as f:
                f.write(feat_str)

def main():
    mul_jc = 16
    gpu = 0
    data_base = get_test_data()
    for _ in range(data_base.items) :
        test_data = data_base.get_data()
        data_len = len(test_data)
        print(data_len)
        stride = data_len//mul_jc
        test_data_list = []
        for i in range(mul_jc-1):
            test_data_list.append(test_data[i*stride:(i+1)*stride])
        test_data_list.append(test_data[(mul_jc-1)*stride:])
        #featext_i(test_data_list[0],'0')
        # print(len(test_data_list))
        pool = multiprocessing.Pool(processes=mul_jc)
        for i in range(mul_jc):
            pool.apply_async(featext_i, (test_data_list[i], str(gpu%4)))
            gpu += 1
        pool.close()
        pool.join()


if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()
