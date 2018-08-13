# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np  
import tensorflow as tf  
import os  
import sys  
import cv2 as cv
model_dir = 'Autocoder_tmp'
name_path = model_dir+'names.txt'
if len(sys.argv) > 1:  model_dir = sys.argv[1]

checkpoint = tf.train.get_checkpoint_state(model_dir) 
input_checkpoint = checkpoint.model_checkpoint_path
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver.restore(sess, input_checkpoint)

# saver = tf.train.import_meta_graph('tmp/model.ckpt.meta', clear_devices=True)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# saver.restore(sess, 'tmp/model.ckpt.data-00000-of-00001')

names = [op.name for op in sess.graph.get_operations()]
print(names)
input_tensor = sess.graph.get_tensor_by_name('Image:0')
mid_tensor = sess.graph.get_tensor_by_name('conv1/Relu:0')
out_img = sess.graph.get_tensor_by_name('deconv4/Tanh:0')

# out_mid = sess.run(mid_tensor, feed_dict={input_tensor: [cv.resize(cv.imread('test.jpg'),(320,320))]})
# print(out_mid)


# with open(name_path , 'a') as name_file:
#     tensor_list = [str(sess.graph.get_tensor_by_name(tensor+":0")) for tensor in names]
#     tensor_string = ('\r\n').join(tensor_list)
#     print(tensor_string)
#     # name_file.write(tensor_string)

# #print values