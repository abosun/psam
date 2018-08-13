import os
from tensorflow.python.framework import graph_util
import tensorflow as tf
class pb_writer():
    def __init__(self,sess,end_point,path):
        self.path_base = path
        self.max_acc = 0.0
        self.sess = sess
        self.point = end_point
        self.lastpath = ""
        self.lastitem = ""
    def save_max(self, acc):
        if acc < self.max_acc: return 0
        else : self.max_acc = acc
        print('save pb with acc=%f'%(acc))
        if len(self.lastpath) > 0 and os.path.exists(self.lastpath) :os.remove(self.lastpath)
        self.lastpath = self.path_base + "%.3f"%(acc) + ".pb"
        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, [self.point])
        with tf.gfile.FastGFile(self.lastpath, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
    def save_item(self, item):
        print('save pb with item=%f'%(item))
        if len(self.lastitem) > 0 and os.path.exists(self.lastitem) :os.remove(self.lastitem)
        self.lastitem = self.path_base + "%05d"%(item) + ".pb"
        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, [self.point])
        with tf.gfile.FastGFile(self.lastitem, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
