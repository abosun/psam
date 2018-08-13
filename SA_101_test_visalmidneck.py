import cv2 as cv
import numpy as np
import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import random
import glob
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile 

CLASS_NUM = 101
SPLIT_PATH = 'ucfTrainTestlist/testlist01.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
MODEL_DIR = 'models/'
IN3_FILE = 'classify_image_graph_def.pb'
MODEL_FILE = 'SA101_neckmid1_s1_06251.pb'
GPU = '0'
IMG_DIR = 'ucf_101_img_mul10'
TOP_DIR = 'ucf_101_img_test10_top8'
MID_DIR = 'ucf_101_img_test10_mid35'
NECK_DIR = 'ucf_101_img_test10_NECK'
#OUT_DIR = 'ucf_101_img_test10_35'
# pb_file_path = 'full_ucfs1.pb'
#test_dir = {'neck':NECK_DIR, 'top':TOP_DIR, '_out':OUT_DIR}
shape = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288)}
file = 'v_BlowingCandles_g05_c01'

def read_split():
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_set = set([ os.path.basename(x).split('.')[0] for x in split_list])
    return split_set

def resize299(img8,size):
    big = np.max(img8)
    img = img8 / big
    img299 = np.zeros((315,315))
    s = 299 // size+1
    for i in range(size):
        for j in range(size):
            a = int(img[i,j])
            img299[i*s:(i+1)*s+1,j*s:(j+1)*s+1] = img[i,j]
    return cv.resize(img299,(299,299))    
#return img299[1:300,1:300]
def pinghua(img, s=11):
    w,h = img.shape
    img0 = img.copy()
    for i in range(w):
        for j in range(h):
            img[i,j] = img0[max(0,i-s):min(h-1,i+s),max(0,j-s):min(h-1,j+s)].mean()
    return img
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

def get_test_data(label_dict):
    test_data = []
    test_lable = []
    test_list = []
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
        split_list = split_string.split('\r\n')
    for i in sorted(split_list):
        data_one = {'neck':[], 'top':[], 'pre_out':[]}
        action = os.path.dirname(i)
        for kk in test_dir.keys():
            glob_path = os.path.join(test_dir[kk], i.split('.')[0]+'*.txt')
            glob_list = sorted(glob.glob(glob_path))
            data_one[kk] = [read_data(x, shape[kk]) for x in glob_list]
        test_list.append(glob_list)
        test_lable.append(label_dict[action])
        test_data.append(data_one)
    return test_data, test_list, test_lable

def get_one_data(file_name):
    test_data = []
    test_lable = []
    test_list = []
    data_one = {'neck':[], 'top':[], 'mid':[]}
    for kk in test_dir.keys():
        glob_path = os.path.join(test_dir[kk], '*', file_name+'*.txt')
        glob_list = sorted(glob.glob(glob_path))
        data_one[kk] = [read_data(x, shape[kk]) for x in glob_list]
    return data_one
def get_one_img(file_name):
    test_data = []
    test_lable = []
    test_list = []
    glob_path = os.path.join(IMG_DIR, '*', file_name+'*.jpg')
    glob_list = sorted(glob.glob(glob_path))
    img = [cv.imread(x) for x in glob_list]
    return img

def get_train_batch(path_list, label_list, test_set):
    data_num = len(path_list)
    train_batch = []
    label_batch = []
    batch_n = 0
    while batch_n < BATCH_SIZE :
        rand_n = random.randint(0,data_num-1)
        train_name = os.path.basename(path_list[rand_n]).split('.')[0]
        if train_name in test_set :
            continue
        else:
            train_batch.append(read_data(path_list[rand_n]))
            label_batch.append(label_list[rand_n])
            batch_n += 1
    return train_batch, label_batch

# def read_data(path):
#     with open(path) as f:
#         line = f.read()
#         data = [float(x) for x in line.split(',')]
#     return np.array(data)

def graph_def(input):
    with slim.arg_scope([slim.fully_connected], #activation_fn=tf.nn.tanh,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.02),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.dropout(input, drop_rate, scope='fc1_drop')
        net = slim.fully_connected(net, 101, activation_fn=None, scope='fc1')
        # net = slim.dropout(input, drop_rate, scope='fc2_drop')
        # net = slim.fully_connected(net, CLASS_NUM, activation_fn=None, scope='fc2')
    return net
def get_atte0(sess, file):
    top = sess.graph.get_tensor_by_name("top:0")
    soft_tensor = sess.graph.get_tensor_by_name("softmax:0")
    if_train = sess.graph.get_tensor_by_name("is_training:0")
    attention = sess.graph.get_tensor_by_name("Relu:0")
    attention = tf.reshape(attention,[-1,8,8])
    data_list = get_one_data(file)
    attention_map = sess.run([attention], feed_dict={top:data_list['top'], if_train:False})
    frame = get_one_img(file)[0]
    img = resize299(attention_map[0][0])
    img = pinghua(img)
    frame_w = pay_atte(frame, img, 0.2)
    return frame_w
def get_atte1(sess, file):
    neck = sess.graph.get_tensor_by_name("neck:0")
    pre_out = sess.graph.get_tensor_by_name("pre_out:0")
    top = sess.graph.get_tensor_by_name("top:0")
    soft_tensor = sess.graph.get_tensor_by_name("softmax:0")
    if_train = sess.graph.get_tensor_by_name("if_train:0")
    attention = sess.graph.get_tensor_by_name("MatMul:0")
    attention = tf.reshape(attention,[-1,8,8])
    data_list = get_one_data(file)
    attention_map = sess.run([attention], feed_dict={neck:data_list['neck'], top:data_list['top'], pre_out:data_list['pre_out'], if_train:False})
    frame = get_one_img(file)[0]
    img = resize299(attention_map[0][0])
    img = pinghua(img)
    frame_w = pay_atte(frame, img, 0.2)
    return frame_w
def main():
    graph_0 = tf.Graph()
    with graph_0.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, IN3_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    sess_in3 = tf.Session(config=cuda_set(gpu=GPU), graph=graph_0)
    graph_1 = tf.Graph()
    with graph_1.as_default():
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    sess_att = tf.Session(config=cuda_set(gpu=GPU), graph=graph_1)
    names = [op.name for op in sess_att.graph.get_operations()]
    #for name in names:
    #    print(sess_att.graph.get_tensor_by_name(name+":0"))
    v3_mid = sess_in3.graph.get_tensor_by_name("mixed_2/join:0")
    v3_neck = sess_in3.graph.get_tensor_by_name("pool_3/_reshape:0")
    v3_top = sess_in3.graph.get_tensor_by_name("mixed_8/join:0")
    v3_img = sess_in3.graph.get_tensor_by_name("DecodeJpeg:0")
    v3_out = sess_in3.graph.get_tensor_by_name("softmax:0")

    neck = sess_att.graph.get_tensor_by_name("neck:0")
    #top = sess_att.graph.get_tensor_by_name("top:0")
    mid = sess_att.graph.get_tensor_by_name("mid:0")
#    if_train = sess_att.graph.get_tensor_by_name("is_training:0")
    mid_atte = sess_att.graph.get_tensor_by_name("Relu:0")
    
    img = get_one_img(file)[4]
    Mid, Neck = sess_in3.run([v3_mid, v3_neck], feed_dict={v3_img:img})

    Atte_mid = sess_att.run([mid_atte], feed_dict={mid:Mid, neck:Neck})#, if_train:False})
    frame = get_one_img(file)[4]
    print(Atte_mid[0][0].shape)
    img = resize299(np.reshape(Atte_mid[0],(35,35)),35)
    img = pinghua(img)
    f_max = frame.max()
    frame[:,:,0] = frame[:,:,0] * 0.2
    frame[:,:,1] = frame[:,:,1] * 0.2
    frame[:,:,2] = frame[:,:,2] * np.maximum(img*1.5,0.2)
    #frame = frame * (255.0/frame.max())
    cv.imwrite('keshihua.jpg',frame)


if __name__ == '__main__' :
    # get_center_frame(data_path)
    main()

