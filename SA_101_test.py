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
SPLIT_PATH = 'ucfTrainTestlist/testlist03.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
MODEL_DIR = 'models/'
MODEL_FILE = 'SAH_rtr_JT_s3_10001.pb'
GPU = '1'
TOP_DIR = 'ucf_101_img_test10_top8'
MID_DIR = 'ucf_101_img_test10_mid35'
NECK_DIR = 'ucf_101_img_test10_NECK'
OUT_DIR = 'ucf_101_img_mul12_NECK_OUT'
# TOP_DIR = 'ucf_101_img_mul12_top8'
# MID_DIR = 'ucf_101_img_mul12_mid35'
# NECK_DIR = 'ucf_101_img_mul12_NECK'
# OUT_DIR = 'ucf_101_img_mul12_NECK_OUT'
# TOP_DIR = 'ucf_101_img_mul12_top8_max2'
# MID_DIR = 'ucf_101_img_mul12_mid35_max2'
# NECK_DIR = 'ucf_101_img_mul12_NECK_max2'
# OUT_DIR = 'ucf_101_img_mul12_NECK_OUT_max2'
# TOP_DIR = 'ucf_101_img_centre_top8'
# MID_DIR = 'ucf_101_img_centre_mid35'
# NECK_DIR = 'ucf_101_img_centre_NECK'
# OUT_DIR = 'ucf_101_img_centre_NECK_OUT'
# pb_file_path = 'full_ucfs1.pb'
test_dir = {'neck':NECK_DIR, 'top':TOP_DIR, 'pre_out':OUT_DIR}
'neck', 'pre_out', 'top', 'if_train'
shape = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288),'pre_out':(101)}

def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('0')
    plt.figure(figsize=(10, 8), dpi=120,)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0
    print('1')
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    # if(intFlag):
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # else:
    #     plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    print('2')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    # plt.colorbar()
    print('3')
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    # plt.show()

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

def cuda_set(gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.allow_growth = True
    return config

def main():
    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    test_data = get_test_data().result
    with tf.Session(config=cuda_set(gpu=GPU), graph=graph) as sess:
        names = [op.name for op in sess.graph.get_operations()]
        print(names)
        real_list = []
        pred_list = []
        result = []
        right = 0.0
        right_2 = 0.0
        right_h5 = 0.0
        right_fusion = 0.0
        total = 0.0
        jishu = [0]*12
        neck = sess.graph.get_tensor_by_name("neck:0")
        # pre_out = sess.graph.get_tensor_by_name("pre_out:0")
        top = sess.graph.get_tensor_by_name("top:0")
        soft_tensor = sess.graph.get_tensor_by_name("softmax:0")
        if_train = sess.graph.get_tensor_by_name("if_train:0")
        Z_total = sess.graph.get_tensor_by_name("ZN_comb:0")
        for xx in test_data:
            data_list = xx[0]
            path_list = xx[1]
            label = xx[2]
            total += 1
            #
            feature = sess.run([Z_total], feed_dict={neck:data_list['neck'], top:data_list['top'],# pre_out:data_list['pre_out'], 
                if_train:False})
            soft = sess.run([soft_tensor], feed_dict={Z_total:[feature[0].mean(axis=0)], if_train:False})
            fusion_label = np.argmax(soft[0])
            if fusion_label == label: right_fusion+=1

            real_list.append(label)
            pred_list.append(fusion_label)

            soft = sess.run([soft_tensor], feed_dict={neck:data_list['neck'], top:data_list['top'],# pre_out:data_list['pre_out'], 
                if_train:False})
            lv = np.transpose(soft[0])[label]
            index = np.argmax(lv)
            name = path_list[index].split('.')[0]
            line = name + "," + str(index) + "," + str(lv[index])
            soft_line = soft[0][index]
            index_label = np.argmax(soft_line)
            # jishu[index] += 1
            if index_label==label:right+=1
            result.append(line)

            # print(len(soft[0]))
            for i in range(8):
                if np.argmax(soft[0][i])==label:
                    jishu[i] += 1
            # print(line)
        # with open("UCF_101_SA_rtr_max_s2.txt",'w') as f:
        #     f.write('\n'.join(result))
        
        print(total)
        print(right_fusion/total)
        print(right/total)
        jishu = [x/total for x in jishu]
        print(jishu)

        # plot_confusion_matrix(real_list, pred_list, list(range(101)))
        from sklearn.metrics import confusion_matrix  
        labels = list(set(real_list))  
        conf_mat = confusion_matrix(real_list, pred_list, labels = labels) 
        mid = conf_mat.max()/2
        stride = 21
        hx_mat = np.zeros((101*stride+1, 101*stride+1))
        for i in range(101):
            for j in range(101):
                # print(hx_mat[i*stride:(i+1)*stride, j*stride:(j+1)*stride])
                # print(conf_mat[i,j])
                # raise ValueError
                hx_mat[i*stride:(i+1)*stride, j*stride:(j+1)*stride] = conf_mat[i,j]
                hx_mat[(i+1)*stride-1, j*stride:(j+1)*stride] = mid
                hx_mat[i*stride:(i+1)*stride,(j+1)*stride-1] = mid
        cv.imwrite('hunxiao.jpg',255-(hx_mat/hx_mat.max())*255)

if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()