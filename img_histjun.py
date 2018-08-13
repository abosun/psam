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

data_test_dir   = 'ucf_101_img_centre_NECK'
data_train_dirs  = ['ucf_101_img_centre_NECK','ucf_101_img_mul12_NECK','ucf_101_img_centre_top8_F_NECK']
CLASS_NUM = 101
BATCH_SIZE = 50
learning_rate = 0.0001
ITEMS = 3000
drop_rate = 0.9
SPLIT_PATH = 'ucfTrainTestlist/testlist01.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
MODEL_DIR = 'models/'
MODEL_FILE = 'SA_v1_0.759.pb'
GPU = '0'
TOP_DIR = 'ucf_101_img_mul12_top8'
MID_DIR = 'ucf_101_img_mul12_mid35'
NECK_DIR = 'ucf_101_img_mul12_NECK'
OUT_DIR = 'ucf_101_img_mul12_NECK_OUT'
# pb_file_path = 'full_ucfs1.pb'
test_dir = {'img':'ucf_101_img_centre'}
target_dir = {'img':'ucf_101_img_centre_jun'}
# 'neck', 'pre_out', 'top', 'if_train'
shape = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288),'pre_out':(101)}
max_file = 'UCF_101_s1_SA_max.txt'

def image(sourse_path):#, target_path):
    img = cv.imread(sourse_path)
    img[:,:,0] = cv.equalizeHist(img[:,:,0])
    img[:,:,1] = cv.equalizeHist(img[:,:,1])
    img[:,:,2] = cv.equalizeHist(img[:,:,2])
    return img

class convert_max():
    def __init__(self, test_dir, target_dir):
        self.sourse_dir = test_dir
        self.target_dir = target_dir
        self.read_dir()
        print("has read")
        for kk in test_dir.keys():
            self.copy2target(self.sourse_dir[kk], self.target_dir[kk])
    def initial(self,dic):
        keys = dic.keys()
        for key in keys:
            dic[key] = []
    def jun_image(self,sourse_path, target_path):#, target_path):
        img = cv.imread(sourse_path)
        img[:,:,0] = cv.equalizeHist(img[:,:,0])
        img[:,:,1] = cv.equalizeHist(img[:,:,1])
        img[:,:,2] = cv.equalizeHist(img[:,:,2])
        cv.imwrite(target_path, img)
    def copy2target(self, sourse_dir, target_dir):
        for sourse_path in self.path_list:
            class_name = os.path.basename(os.path.dirname(sourse_path))
            path = os.path.basename(sourse_path)
            target_path = os.path.join(target_dir,class_name,path)
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
                print(os.path.dirname(target_path))
            self.jun_image(sourse_path, target_path)
    def read_dir(self):
        self.path_list = []
        # dir_list = os.walk(self.sourse_dir)[1:]
        glob_path = os.path.join(self.sourse_dir['img'], "*", '*.jpg')
        print(glob_path)
        self.path_list = glob.glob(glob_path)


        # with open(self.max_file) as f:
        #     line = f.readline()
        #     while len(line) > 0:
        #         lines = line.split(',')
        #         index = int(lines[1])
        #         path = os.path.basename(lines[0])
        #         class_name = os.path.dirname(os.path.basename(lines[0]))
        #         self.max_info.append((path, index, class_name))
        #         line = f.readline()




convert_max(test_dir, target_dir)


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


# if __name__ == '__main__' :
#     # get_center_frame(data_path)
#     main()
