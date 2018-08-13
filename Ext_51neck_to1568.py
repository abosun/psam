import cv2 as cv
import numpy as np
import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import random
import glob
from tensorflow.python.framework import graph_util
from utils import pb_writer
from tensorflow.python.platform import gfile


data_train_dirs  = ['HMDB51_img_mul1_NECK']
dim = 2048
CLASS_NUM = 51
BATCH_SIZE = 50
learning_rate = 0.0001
ITEMS = 10000
drop_rate = 0.8
SPLIT_DIR = 'hmdbTrainTestlist/'
SPLIT_PATH = SPLIT_DIR+'testlist01.txt'
LABEL_PATH = SPLIT_DIR+'classInd.txt'
pb_file_path = 'full_hmdbs1.pb'
GPU = '1'
pb_base_name = 'models/hmdb51_all'

def read_split():
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_set = set([ os.path.basename(x).split('.avi')[0] for x in split_list])
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
        glob_path = os.path.join(sub_dirs[class_i], '*.npy')
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
        label_list = label_string.split('\r\n')
        if len(label_list[-1])<2:
            label_list = label_list
    for x in label_list:
        index = int(x.split(' ')[0])
        name = x.split(' ')[1]
        label_dict[index] = name
        label_dict[name] = index
    print(label_dict)
    return label_dict

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

def get_test_data(label_dict):
    test_data = []
    test_lable = []
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
        split_list = split_string.split('\r\n')
    for i in split_list:
        if len(i)==0:continue
        if len(i.split('['))>1:
            # print(i)
            i = '*'.join(i.split('['))
        glob_path = os.path.join(data_test_dir, i.split('.avi')[0]) +'.*txt'
        # print(glob_path)  
        action = os.path.dirname(i)
        glob_list = glob.glob(glob_path)
        if len(glob_list)!=12:
            print(i)
            print(glob_path)
            print(glob_list[0])
            print(glob_list[-1])
            raise ValueError
        # print(glob_path)
        test_data.append(read_data(glob_list[6]))
        test_lable.append(label_dict[action])
    return test_data, test_lable

# def get_train_batch(path_list, label_list, test_set):
#     data_num = len(path_list)
#     train_batch = []
#     label_batch = []
#     batch_n = 0
#     # print(list(test_set)[:10])
#     while batch_n < BATCH_SIZE :
#         rand_n = random.randint(0,data_num-1)
#         train_name = os.path.basename(path_list[rand_n]).split('.avi')[0]
#         if train_name in test_set :
#             continue
#         else:
#             train_batch.append(read_data(path_list[rand_n]))
#             label_batch.append(label_list[rand_n])
#             batch_n += 1
#             # print('train_'+str(batch_n))
#     return train_batch, label_batch

def get_train_batch_all(path_list, label_list):
    data_num = len(path_list)
    train_batch = []
    label_batch = []
    batch_n = 0
    while batch_n < BATCH_SIZE :
        rand_n = random.randint(0,data_num-1)
        train_name = os.path.basename(path_list[rand_n]).split('.avi')[0]
        train_batch.append(read_data_npy(path_list[rand_n]))
        label_batch.append(label_list[rand_n])
        batch_n += 1
    return train_batch, label_batch

def read_data_npy(path):
    return np.load(path)

def read_data(path):
    with open(path) as f:
        line = f.read()
        data = [float(x) for x in line.split(',')]
        if len(data)!=2048:
            for i in data:
                print(i)
            raise ValueError
    return np.array(data)

def graph_def(net,is_train):
    with slim.arg_scope([slim.fully_connected], #activation_fn=tf.nn.tanh,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.02),
                      weights_regularizer=slim.l2_regularizer(0.005)):
        net = slim.fully_connected(net, 1568, scope='fc0')
        net = tf.nn.relu(net, name='feature')
    with slim.arg_scope([slim.fully_connected], #activation_fn=tf.nn.tanh,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.02),
                      weights_regularizer=slim.l2_regularizer(0.05)):
        net = slim.dropout(net, drop_rate, is_training=is_train, scope='fc1_drop')
        net = slim.fully_connected(net, CLASS_NUM, activation_fn=None, scope='fc1')
        # net = slim.dropout(input, drop_rate, scope='fc2_drop')
        # net = slim.fully_connected(net, CLASS_NUM, activation_fn=None, scope='fc2')
    return net

def main():
    label_dict = get_class_dict()
    # test_batch, test_label_batch = get_test_data(label_dict)
    # print('read_test')
    # test_set = read_split()
    # print(len(test_set))
    train_path_list = []
    train_label_list = []
    for data_train_dir in data_train_dirs:
        train_path_list2, train_label_list2 = load_dataset_to_input(data_train_dir, label_dict)
        train_path_list.extend(train_path_list2)
        train_label_list.extend(train_label_list2)
    sorted(train_path_list)
    # train_batch, label_batch = get_train_batch(train_path_list, train_label_list, test_set)
    acc_list = []
    index_list = []

    # input_tensor = tf.placeholder(dtype=tf.float32, shape=[None,dim], name='input_tensor')
    # label_index = tf.placeholder(dtype=tf.int64, shape=[None], name='label_index')
    # is_train = tf.placeholder(tf.bool, name='if_train')
    # label = tf.one_hot(label_index, CLASS_NUM)
    # logits = graph_def(input_tensor, is_train)
    # cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))
    # slim.losses.add_loss(cross_loss)
    # total_loss = slim.losses.get_total_loss()
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    # pre_out = tf.nn.softmax(logits, name='softmax')
    # predict_label = tf.argmax(logits,1)
    # correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(logits,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # max_acc = 0.

    with gfile.FastGFile(os.path.join('models/hmdb51_all49999.000.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    NECK_TENSOR_NAME = 'input_tensor:0'
    FRAT_TENSOR_NAME = 'feature:0'
    target = "HMDB51_feat_1568_test"
    neck_tensor, out_tensor = tf.import_graph_def(graph_def, return_elements=[NECK_TENSOR_NAME, FRAT_TENSOR_NAME])
    with tf.Session(config=cuda_set(gpu=GPU)) as sess:
        for path in train_path_list:
            data = [np.load(path)]
            feat = sess.run([out_tensor],feed_dict={neck_tensor:data})
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            dirname = os.path.join(target,os.path.basename(dirname))
            if not os.path.exists(dirname):os.makedirs(dirname)
            to_path = os.path.join(dirname, basename+'.cnnfeat')
            line = [str(x) for x in feat[0][0]]
            # print(len(feat[0][0]))
            with open(to_path, 'w') as f:
                f.write(' '.join(line))
            # break


if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()
