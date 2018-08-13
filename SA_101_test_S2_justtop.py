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
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _pickle_method)

data_test_dir   = 'ucf_101_img_mul12_NECK'
data_train_dirs  = ['ucf_101_img_centre_NECK','ucf_101_img_mul12_NECK','ucf_101_img_centre_top8_F_NECK']
CLASS_NUM = 101
BATCH_SIZE = 50
learning_rate = 0.0001
ITEMS = 3000
drop_rate = 0.9
SPLIT_PATH = 'ucfTrainTestlist/testlist02.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
MODEL_DIR = 'models/'
MODEL_FILE = 'SA_v2_0.747.pb'
GPU = '3'
TOP_DIR = 'ucf_101_img_mul12_top8'
MID_DIR = 'ucf_101_img_mul12_mid35'
NECK_DIR = 'ucf_101_img_mul12_NECK'
OUT_DIR = 'ucf_101_img_mul12_NECK_OUT'
# pb_file_path = 'full_ucfs1.pb'
test_dir = {'neck':NECK_DIR, 'top':TOP_DIR, 'pre_out':OUT_DIR}
#'neck', 'pre_out', 'top', 'if_train'
shape = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288),'pre_out':(101)}

def read_split():
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_set = set([ os.path.basename(x).split('.')[0] for x in split_list])
    return split_set

def load_dataset_to_input(data_set_dir, label_dict):
    path_list = []
    label_list = []
    label_dict_name = {}
    import glob
    sub_dirs = [x[0] for x in os.walk(data_set_dir)][1:]
    sub_names = [os.path.basename(x) for x in sub_dirs]
    class_n = len(sub_dirs)
    for class_i in range(class_n):
        action = sub_names[class_i]
        glob_path = os.path.join(sub_dirs[class_i], '*.txt')
        file_list = glob.glob(glob_path)
        path_list.extend(file_list)
        label_list.extend([label_dict[action]]*len(file_list))
    return path_list, label_list

def cuda_set(gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.allow_growth = True
    return config

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

def get_test_batch(dict_name):
    test_data = []
    test_lable = []
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
        split_list = split_string.split('\r\n')
    for i in split_list:
        path = os.path.join(data_test_dir, i) + '.jpg.txt'
        action = os.path.dirname(i)
        test_data.append(read_data(path))
        test_lable.append(dict_name[action])
    return test_data, test_lable

class get_test_data():
    def __init__(self,label_dict):
        jc = 25
        self.label_dict = label_dict
        self.test_data = []
        self.test_lable = []
        self.test_list = []
        self.test_read_i = -1
        self.test_read_len = 0
        threads = []
        result = []
        with open(SPLIT_PATH , 'r') as split_file:
            split_string = split_file.read()
            self.split_list = split_string.split('\r\n')
            self.test_read_len = len(self.split_list)
        pool = multiprocessing.Pool(processes=jc)
        print(self.test_read_len)
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
        print(i)
        data_one = {'neck':[], 'top':[], 'pre_out':[]}
        action = os.path.dirname(line)
        for kk in test_dir.keys():
            glob_path = os.path.join(test_dir[kk], line.split('.')[0]+'*.txt')
            glob_list = sorted(glob.glob(glob_path))
            data_one[kk] = [read_data(x, shape[kk]) for x in glob_list]
        return (data_one,glob_list,self.label_dict[action])
    def data(self):
        return self.test_data, self.test_list, self.test_lable

def main():
    label_dict = get_class_dict()
    test_data = get_test_data(label_dict).result
    #test_data, test_list, test_lable = get_test_data(label_dict)
    # print('read_test')
    # test_set = read_split()
    # train_path_list = []
    # train_label_list = []
    # for data_train_dir in data_train_dirs:
    #     train_path_list2, train_label_list2 = load_dataset_to_input(data_train_dir, label_dict)
    #     train_path_list.extend(train_path_list2)
    #     train_label_list.extend(train_label_list2)
    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    with tf.Session(config=cuda_set(gpu=GPU), graph=graph) as sess:
        names = [op.name for op in sess.graph.get_operations()]
        print(names)
        result = []
        right = 0.0
        total = 0.0
        jishu = [0]*12
        neck = sess.graph.get_tensor_by_name("neck:0")
        pre_out = sess.graph.get_tensor_by_name("pre_out:0")
        top = sess.graph.get_tensor_by_name("top:0")
        soft_tensor = sess.graph.get_tensor_by_name("output:0")
        if_train = sess.graph.get_tensor_by_name("if_train:0")
        attention = sess.graph.get_tensor_by_name("MatMul:0")
        for xx in test_data:
            data_list = xx[0]
            path_list = xx[1]
            label = xx[2]
            # for (data,path) in zip(data_list,path_list):
            soft = sess.run([soft_tensor], feed_dict={neck:data_list['neck'], top:data_list['top'], pre_out:data_list['pre_out'], if_train:False})
            # reduce_score = soft[0].sum(axis=0)
            # predict_label = np.argmax(reduce_score)
            # if predict_label==label:right+=1
            # total += 1
            lv = np.transpose(soft[0])[label]
            index = np.argmax(lv)
            name = path_list[index].split('.')[0]
            line = name + "," + str(index) + "," + str(lv[index])
            soft_line = soft[0][index]
            index_label = np.argmax(soft_line)
            jishu[index] += 1
            if index_label==label:right+=1
            total += 1
            result.append(line)
            print(line)
        with open("UCF_101_s2_SA_max.txt",'w') as f:
            f.write('\n'.join(result))
        print(right/total)


    # acc_list = []
    # index_list = []

    # input_tensor = tf.placeholder(dtype=tf.float32, shape=[None,192], name='input_tensor')
    # label_index = tf.placeholder(dtype=tf.int64, shape=[None], name='label_index')
    # label = tf.one_hot(label_index, CLASS_NUM)
    # logits = graph_def(input_tensor)
    # cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))
    # slim.losses.add_loss(cross_loss)
    # total_loss = slim.losses.get_total_loss()
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    # pre_out = tf.nn.softmax(logits, name='output')
    # predict_label = tf.argmax(logits,1)
    # correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(logits,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # with tf.Session(config=cuda_set(gpu='1')) as sess:
    #     sess.run( tf.global_variables_initializer())
    #     print(data_test_dir)
    #     for i in range(ITEMS):
    #         train_batch, label_batch = get_train_batch(train_path_list, train_label_list, test_set)
    #         if (i+1)%100 == 0 :
    #             _, loss, acc = sess.run([train_op, total_loss, accuracy],
    #                 feed_dict={input_tensor:train_batch, label_index:label_batch})
    #             test_acc, test_loss = sess.run([accuracy, total_loss], feed_dict={input_tensor:test_batch, label_index:test_label_batch})
    #             print(test_acc)
    #             print("step %6d  loss=%f  acc=%f  TEST_acc : %f Test_loss=%f"%(i+1, loss, acc, test_acc, test_loss))
    #             with open('ucf-101-img-combine-record.txt','a') as f:
    #                 acc_str = 'Step %05d TEST_acc : %f  '%(i, test_acc)+'\n'
    #                 f.write(acc_str)
    #         else:
    #             sess.run([train_op], feed_dict={input_tensor:train_batch, label_index:label_batch})
    #     # _, test_acc[0] = sess.run([train_op, accuracy], feed_dict={input_tensor:test_batch, label_index:test_label_batch})
    #     # print('TEST_acc : %f  '%(test_acc[0]) )
    #     # with open('ucf-101-img-combine-record.txt','a') as f:
    #     #     acc_str = 'TEST_acc : %f  '%(test_acc[0])
    #     #     f.write(acc_str)
    #     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    #     with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    #         f.write(constant_graph.SerializeToString())


if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()

