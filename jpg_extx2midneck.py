# -*- coding: utf-8 -*-
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import cv2 as cv
# from avi2jpg import read_data 
def read_data(file_path):
    time0=time.time()
    global img_n
    sub_dirs = [x[0] for x in os.walk(file_path)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        sub_dir_path = os.path.join(IMG_DTR,os.path.basename(sub_dir))
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)
        else:
            continue
        img_n = 0
        extensions = ['avi']
        dir_name = os.path.basename(sub_dir)
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(file_path, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        for file_name in file_list:
            avi2img(file_name, dir_name)
    time1=time.time()
    print(time1-time0)
# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048
tensor_shape = (8,8,1280)
# Inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提出的Inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
IMG_TENSOR_NAME = 'DecodeJpeg:0'
# softmax层输出张量对应的名称
SOFTMAX_TENSOR_NAME = 'softmax/logits:0'
MID_TENSOR_NAME = 'mixed_8/join:0'#'mixed_8/join:0'#'mixed_3/join:0'#'mixed_10/join:0'#'pool_1:0'
# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = 'models/'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'classify_image_graph_def.pb'

# 视频数据
IMG_DTR = 'ucf_101Extend'
OUT_DIR_mid = IMG_DTR+'_081280'
OUT_DIR_neck = IMG_DTR+'_2048'
# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中，免去重复的计算。
# 下面的变量定义了这些文件的存放地址。
gpu_id = '1'
# 图片数据文件夹。
# 在这个文件夹中每一个子文件夹代表一个需要区分的类别，每个子文件夹中存放了对应类别的图片。
# INPUT_DATA = 'data/img/'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 5000
BATCH = 100

def load_dataset(data_set_dir):
    dataset_dict = {}
    import glob
    sub_dirs = [x[0] for x in os.walk(data_set_dir)][1:]
    sub_names = [os.path.basename(x) for x in sub_dirs]
    class_n = len(sub_dirs)
    for class_i in range(class_n):
        glob_path = os.path.join(sub_dirs[class_i],'*.jpg')
        avi_list = glob.glob(glob_path)
        dataset_dict[sub_names[class_i]] = avi_list
    return dataset_dict

def cuda_set(gpu='1'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu   #指定第一块GPU可用  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1  # 程序最多只能占用指定gpu50%的显存  
    config.gpu_options.allow_growth = True      #程序按需申请内存  
    return config

def img_crop(img, size=240):
    w,h,dim=img.shape
    return img[(w//2-size//2):(w//2+size//2),:,:]

def video_count(data_path):
    video = cv.VideoCapture(data_path)
    nums = 0
    while 1:
        ret, frame = video.read()
        if not ret : break
        nums += 1
    return nums

def main():
    time0=time.time()
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, mid_tensor, img_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, MID_TENSOR_NAME, IMG_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])
    category='training'
    k=8
    sess = tf.Session(config=cuda_set(gpu=gpu_id))
    dataset_dict = load_dataset(IMG_DTR)
    for action in dataset_dict:
        out_dir_neck = os.path.join(OUT_DIR_neck, action)
        out_dir_mid = os.path.join(OUT_DIR_mid, action)
        if not os.path.exists(out_dir_neck): 
            os.makedirs(out_dir_neck)
        if not os.path.exists(out_dir_mid): 
            os.makedirs(out_dir_mid)       
        for file_path in dataset_dict[action]:
            out_path_mid = os.path.join(out_dir_mid, os.path.basename(file_path))+'.txt'
            if os.path.exists(out_path_mid):continue
            img = cv.imread(file_path)
            mid_values = sess.run([mid_tensor], {img_tensor: img})
            botten_values = sess.run([bottleneck_tensor], {mid_tensor: mid_values[0]})

            out_path_mid = os.path.join(out_dir_mid, os.path.basename(file_path))+'.txt'
            mid_string = ','.join(str(x) for x in np.reshape(mid_values[0],(8*8*1280)))
            with open(out_path_mid, 'w') as mid_file:
                mid_file.write(mid_string)
            print(out_path_mid)
            out_path_neck = os.path.join(out_dir_neck, os.path.basename(file_path))+'.txt'
            bottleneck_string = ','.join(str(x) for x in botten_values[0][0])
            with open(out_path_neck, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)

if __name__ == '__main__':
    # read_data(VIDEO_DTR)
#    tf.app.run()
    main()
