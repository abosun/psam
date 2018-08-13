from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np 
import os
import glob
import time
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util  

GPU = '3'
LEARNING_RATE = 0.0001
SPLIT_PATH_TEST = 'ucfTrainTestlist/testlist03.txt'
SPLIT_PATH_TRAIN = 'ucfTrainTestlist/trainlist03.txt'
LABEL_PATH = 'ucfTrainTestlist/classInd.txt'
pb_file_path = 'full_ucfs2.pb'
TOP_DIR = 'ucf_101_img_centre_top8'
MID_DIR = 'ucf_101_img_centre_mid35'
NECK_DIR = 'ucf_101_img_centre_NECK'
OUT_DIR = 'ucf_101_img_centre_NECK_OUT'
DATA_BASE_PATH = 'DATA_BASE/DATA_ucf101_centre_top_out.pkl'
batch_size = 100
tensor_shape = {'neck':[2048], 'top':[8,8,1280], 'mid':[35,35,288], 'pre_out':[101]}
c = 101
ITEMS = 50000
print_item = 50
tight_a = 0.1
large_batch = 1000
large_item = 30000
print([tight_a,large_batch,large_item])

def read_split():
    with open(SPLIT_PATH , 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_set = set([ os.path.basename(x).split('.')[0] for x in split_list])
    return split_set

def load_class(label_path):
    label_dict = {}
    with open(label_path) as f:
        line = f.readline()
        while(1):
            if line=='' :break
            data = line.split('\r\n')[0].split(' ')
            label_dict[data[1]] = int(data[0])-1
            label_dict[int(data[0])-1] = data[1]
            line = f.readline()
    return label_dict

def hang_matric(classes):
    matric = []
    line = [0]*classes
    line[-1] = 1
    matric.append(line)
    for i in range(classes-1):
        line = [0.0]*classes
        line[i] = 1.0
        matric.append(line)
    result = np.array([matric])
    return result.astype(np.float32)


class Data_base():
    def __init__(self, neck_dir, top_dir, mid_dir, out_dir):
        print("Reading data ...")
        self.features = ['neck', 'top', 'out']
        self.file_dir = {'neck':neck_dir, 'top':top_dir, 'out':out_dir, 'mid':mid_dir}
        self.tensor_size = {'neck':(2048),'top':(8,8,1280),'mid':(35,35,288),'out':(101)}
        self.path_list = {'neck':[], 'top':[], 'mid':[], 'out':[],'label':[]}
        self.data_train = {'neck':[], 'top':[], 'mid':[], 'out':[],'label':[]}
        self.data_test = {'neck':[], 'top':[], 'mid':[], 'out':[],'label':[]}
        self.data_batch = {'neck':[], 'top':[], 'mid':[], 'out':[],'label':[]}
        self.data_test_batch = {'neck':[], 'top':[], 'mid':[], 'out':[],'label':[]}
        self.data_lenth = {'neck_train':[], 'top_train':[], 'out_train':[], 'mid_train':[], 
            'neck_test':[], 'top_test':[], 'out_test':[], 'mid_test':[]}
        self.read_split(SPLIT_PATH_TEST)
        self.load_class(LABEL_PATH)
        top_sub_dirs = [x[0] for x in os.walk(top_dir)][1:]
        sub_names = sorted([os.path.basename(x) for x in top_sub_dirs])
        for class_name in sub_names:
            class_index = self.label_dict[class_name]
            for kk in self.features:
                self.read_dir(class_name, kk)
            if self.data_lenth['top_train'][-1] != self.data_lenth['out_train'][-1] or \
                self.data_lenth['top_test'][-1] != self.data_lenth['out_test'][-1] :
                print(self.data_lenth)
                raise ValueError
            self.data_train['label'].extend([class_index]*self.data_lenth['top_train'][-1])
            self.data_test['label'].extend([class_index]*self.data_lenth['top_test'][-1])
        self.lenth = len(self.data_train['label'])
        print("Reading done ...")
    def load_class(self, label_path):
        self.label_dict = {}
        with open(label_path) as f:
            line = f.readline()
            while(1):
                if line=='' :break
                data = line.split('\r\n')[0].split(' ')
                self.label_dict[data[1]] = int(data[0])-1
                self.label_dict[int(data[0])-1] = data[1]
                line = f.readline()
    def next_batch(self, batch_size):
        import random
        rand_list = random.sample(range(self.lenth), batch_size)
        # self.data_train[kk].extend([self.read_data(x, self.tensor_size[kk]) for x in train_list])
        for kk in self.features:
            # self.data_batch[kk] = [self.read_data(self.path_list[kk][i], self.tensor_size[kk]) for i in rand_list]
            self.data_batch[kk] = [self.data_train[kk][i] for i in rand_list]
        self.data_batch['label'] = [self.data_train['label'][i] for i in rand_list]
        return self.data_batch
    def get_test(self, test_batch_size):
        test_data_list = []
        self.test_lenth = len(self.data_test['out'])
        list_n = self.test_lenth // test_batch_size
        if self.test_lenth % test_batch_size == 0 : list_n -= 1
        self.test_batch_n = list_n

        # self.test_batch_last = self.test_lenth % test_batch_size
        # for kk in self.data_test.keys():
        #     for i in range(list_n):
        #         self.data_test_batch[kk].append(self.data_test[kk][test_batch_size*i: test_batch_size*(i+1)])
        #     self.data_test_batch[kk].append(self.data_test[kk][test_batch_size*(i+1):])
        # return self.data_test_batch

        self.test_lenth_list = []
        test_batch_last = self.test_lenth % test_batch_size
        mark = True
        for kk in self.data_test.keys():
            for i in range(list_n):
                self.data_test_batch[kk].append(self.data_test[kk][test_batch_size*i: test_batch_size*(i+1)])
                if mark: self.test_lenth_list.append(test_batch_size)
            self.data_test_batch[kk].append(self.data_test[kk][test_batch_size*(list_n):])
            if mark:
            	self.test_lenth_list.append(test_batch_last)
            	mark = False
        return self.data_test_batch
    def read_data(self, path, shape):
        with open(path) as f:
            line = f.readline().split(',')
            data = [float(x) for x in line]
        return np.reshape(np.array(data), shape)
    def read_dir(self, class_name, kk):
        class_index = self.label_dict[class_name]
        top_glob = os.path.join(self.file_dir[kk], class_name, "*.txt")
        top_path_list = sorted(glob.glob(top_glob))
        train_list, test_list = self.split_list(top_path_list)
        self.path_list[kk].extend(train_list)
        self.data_train[kk].extend([self.read_data(x, self.tensor_size[kk]) for x in train_list])
        self.data_test[kk].extend([self.read_data(x, self.tensor_size[kk]) for x in test_list])
        # self.data_train[kk].extend([x for x in train_list])
        # self.data_test[kk].extend([ x for x in test_list])
        self.data_lenth[kk+'_train'].append(len(train_list))
        self.data_lenth[kk+'_test'].append(len(test_list))
    def split_list(self, path_list):
        train_list = []
        test_list = []
        for path in path_list:
            name = os.path.basename(path).split('.')[0]
            if name in self.split_set:
                test_list.append(path)
            else:
                train_list.append(path)
        return train_list, test_list
    def read_split(self,split_path):
        with open(split_path , 'r') as split_file:
            split_string = split_file.read()
        split_list = split_string.split('\r\n')
        self.split_set = set([ os.path.basename(x).split('.')[0] for x in split_list])
def cuda_set(gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.allow_growth = True
    return config

def attention_net(S, X, C, batch_size):
    n = tensor_shape['top'][0]
    f = tensor_shape['top'][-1]

    L = tf.squeeze(tf.one_hot(C,c,1.0,0.0,1,tf.float32)) + 0.00000000001
    print(L)
    C = tf.expand_dims(C, -1) 
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.sigmoid,
              weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
              weights_regularizer=slim.l2_regularizer(0.005)):
        B = slim.conv2d(X, c, [1, 1], scope='conv1_1')
    slim.losses.add_loss(tf.nn.l2_loss(B)*0.0001)
    B_r = tf.reshape(B, [-1, n*n, c])
    Z_r = tf.matmul(B_r, tf.expand_dims(S,-1))
    print(Z_r)
    Z = tf.matmul(tf.transpose(tf.reshape(X,[-1,n*n,f]), perm=[0,2,1]),Z_r)
    Z = tf.reshape(Z, [-1,f], name='features')
    # with slim.arg_scope([slim.fully_connected], #activation_fn=None,
    #               weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
    #               weights_regularizer=slim.l2_regularizer(0.005)):
    #     Z = slim.fully_connected(Z, f, activation_fn=tf.nn.relu, scope='fc_Z_w')
    # Z = tf.nn.tanh(Z)
    Z_o = Z
    Z_t = tf.transpose(Z, [1,0])
    Z_mean,Z_var = tf.nn.moments(Z_t,0)
    Z_t = tf.nn.batch_normalization(Z_t,Z_mean,Z_var,0,1,0.001)
    Z = tf.transpose(Z_t, [1,0])
    L_yi = tf.Variable(hang_matric(c),name='L_yi')
    L_sum = tf.expand_dims(tf.reduce_sum(L, 0),-1, name='L_sum')
    Z_c = tf.matmul( tf.expand_dims(tf.transpose(L),0), tf.expand_dims(Z,0), name='class_center')
    Z_c = tf.expand_dims(tf.squeeze(tf.transpose(Z_c, perm=[1,2,0]))/L_sum, 0,name='class_c')
    Z_c_3 = Z_c
    Z_n = tf.matmul( tf.expand_dims(L,0), Z_c, name='class_n')
    Z_c = tf.squeeze(tf.transpose(Z_c, [1,2,0]))

    Z_n_i = tf.square(Z - tf.squeeze(tf.transpose(Z_n, [1,2,0])))
    Z_n_i_s =  tf.expand_dims(tf.reduce_sum(Z_n_i,1),0)
    L_yi_k = L_yi
    for i in range(c-1):
    	if i>0:L_yi_k = tf.matmul(L_yi_k, L_yi)
        Z_nj = tf.matmul( tf.matmul(tf.expand_dims(L,0),L_yi), Z_c_3)
        Z_n_j = tf.square(Z - tf.squeeze(tf.transpose(Z_nj, [1,2,0])))
        Z_n_j_s =  tf.expand_dims(tf.reduce_sum(Z_n_j,1),0)
        # if i==0:loss_tight = tf.reduce_mean(tf.div(Z_n_i_s, Z_n_j_s),1)
        # else:  loss_tight += tf.reduce_mean(tf.div(Z_n_i_s, Z_n_j_s),1)
        if i==0:loss_fuse = tf.reduce_mean(tf.nn.relu(Z_n_i_s-Z_n_j_s),1)
        else:loss_fuse += tf.reduce_mean(tf.nn.relu(Z_n_i_s-Z_n_j_s),1)
    # loss_tight = tf.stop_gradient(loss_tight)
    loss_fuse = tf.stop_gradient(loss_fuse)
    return loss_fuse, Z_o

def attention_moduel(X, S, n, f, name):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.sigmoid,
          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
          weights_regularizer=slim.l2_regularizer(0.005)):
        B_x = slim.conv2d(X, c, [1, 1], scope='conv'+name)
    slim.losses.add_loss(tf.nn.l2_loss(B_x)*0.0001)
    B_x_r = tf.reshape(B_x, [-1, n*n, c])
    Z_x_r = tf.squeeze(tf.matmul(B_x_r, tf.expand_dims(S,-1)))
    Z_x = tf.matmul(tf.transpose(tf.expand_dims(Z_x_r,-1), perm=[0,2,1]),tf.reshape(X,[-1,n*n,f]))
    Z_x = tf.reshape(Z_x, [-1,f], name='features'+name)
    return Z_x

def attention_net2(S, X, Y, C, batch_size):
    n_x = tensor_shape['top'][0]
    f_x = tensor_shape['top'][-1]
    n_y = tensor_shape['mid'][0]
    f_y = tensor_shape['mid'][-1]
    L = tf.squeeze(tf.one_hot(C,c,1.0,0.0,1,tf.float32)) + 0.000000001
    print(L)
    C = tf.expand_dims(C, -1) 

    Z_x = attention_moduel(X, S, n_x, f_x, 'x')
    Z_y = attention_moduel(Y, S, n_y, f_y, 'y')

    Z = tf.concat([Z_x, Z_y], axis=1, name='XY_comb')

    L_yi = tf.Variable(hang_matric(c),name='L_yi')
    L_sum = tf.expand_dims(tf.reduce_sum(L, 0),-1, name='L_sum')
    Z_c = tf.matmul( tf.expand_dims(tf.transpose(L),0), tf.expand_dims(Z,0), name='class_center')
    Z_c = tf.expand_dims(tf.squeeze(tf.transpose(Z_c, perm=[1,2,0]))/L_sum, 0,name='class_c')
    Z_c_3 = Z_c
    Z_n = tf.matmul( tf.expand_dims(L,0), Z_c, name='class_n')
    Z_c = tf.squeeze(tf.transpose(Z_c, [1,2,0]))
    Z_list = tf.unstack(Z, num=batch_size,axis=0)

    loss_tight = tf.Variable(0.0, tf.float32)
    print(loss_tight)
    Z_n_i = tf.square(Z - tf.squeeze(tf.transpose(Z_n, [1,2,0])))
    Z_n_i_s =  tf.expand_dims(tf.reduce_sum(Z_n_i,1),-1)
    for i in range(c-1):
        Z_nj = tf.matmul( tf.matmul(tf.expand_dims(L,0),L_yi), Z_c_3)
        Z_n_j = tf.square(Z - tf.squeeze(tf.transpose(Z_nj, [1,2,0])))
        Z_n_j_s =  tf.expand_dims(tf.reduce_sum(Z_n_j,1),-1)
        loss_tight += tf.reduce_sum(tf.div(Z_n_i_s, Z_n_j_s),0)  
        loss_tight += tf.reduce_sum(Z_n_i_s)  
    return tf.reshape(loss_tight,shape=[1]), Z

class pb_writer():
    def __init__(self,sess,end_point,path):
        from tensorflow.python.framework import graph_util  
        self.path_base = path
        self.max_acc = 0.0
        self.sess = sess
        self.point = end_point
        self.lastpath = ""
    def save_max(self, acc):
        if acc < self.max_acc: return 0
        else : self.max_acc = acc
        print('save pb with acc=%f'%(acc))
        if len(self.lastpath) > 0 and os.path.exists(self.lastpath) :os.remove(self.lastpath)
        self.lastpath = self.path_base + "%.3f"%(acc) + ".pb"
        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, [self.point])
        with tf.gfile.FastGFile(self.lastpath, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
    def save(self):
        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, [self.point])
        with tf.gfile.FastGFile(self.lastpath, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

def load_database(pick_path, TOP_DIR, MID_DIR, OUT_DIR):
    import cPickle as pickle
    if os.path.exists(pick_path) :
        print("Loading data ...")
        with open(pick_path, 'rb') as f:
            data_base = pickle.load(f)
        print("Loading done !")
    else:
        data_base = Data_base(TOP_DIR, MID_DIR, OUT_DIR)
        print("Writing data ...")
        with open(pick_path, 'wb') as f:
            pickle.dump(data_base, f)
        print("Writing done !")
    return data_base

# data_base = load_database(DATA_BASE_PATH, TOP_DIR, MID_DIR, OUT_DIR)
data_base = Data_base(NECK_DIR, TOP_DIR, MID_DIR, OUT_DIR)
# print(data_base.data_test_batch)
# print(data_base)
with tf.Session(config=cuda_set(GPU)) as sess:
    N = tf.placeholder(tf.float32, [None]+tensor_shape['neck'] ,name='neck')
    S = tf.placeholder(tf.float32, [None]+tensor_shape['pre_out'] ,name='pre_out')
    X = tf.placeholder(tf.float32, [None]+tensor_shape['top'], name='top')
    print(X)
    Y = tf.placeholder(tf.float32, [None]+tensor_shape['mid'], name='mid')
    C = tf.placeholder(tf.int32, [None], name='C')
    is_train = tf.placeholder(tf.bool, name='if_train')
    label = tf.squeeze(tf.one_hot(C,c,1.0,0.0,1,tf.float32),name='label')

    r = tf.placeholder(tf.float32)  
    if len(data_base.features) == 4:
        loss_tight, Z = attention_net2(S, X, Y, C, batch_size)
    else:
        loss_tight, Z = attention_net(S, X, C, batch_size)
    Z = tf.concat([Z, N], axis=1, name='ZN_comb')
    Z = slim.dropout(Z, 0.8, is_training=is_train,scope='fc_drop')
    print(Z)
    with slim.arg_scope([slim.fully_connected], #activation_fn=None,
                  weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                  weights_regularizer=slim.l2_regularizer(0.01)):
        logits_Z = slim.fully_connected(Z, c, activation_fn=None, scope='fc_Z')

    loss_cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits_Z))
    slim.losses.add_loss(loss_cross)
    # slim.losses.add_loss(loss_tight*0.05)
    total_loss = slim.losses.get_total_loss() #+ loss_tight * tight_a
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)
    correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(logits_Z,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    output = tf.squeeze(logits_Z, name='output')
    # loss_cross_mul = -tf.reduce_mean(tf.reduce_sum(label*tf.log((tf.nn.softmax(logits_Z)+S)/2)))
    # train_op_2 = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_cross_mul)
    # mul_correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(tf.nn.softmax(logits_Z)+S,1))
    # mul_acc = tf.reduce_mean(tf.cast(mul_correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    # data_base = Data_base(TOP_DIR, OUT_DIR)
    # next_batch = data_base.next_batch(batch_size)
    # print(next_batch)
    # for i in range(10000):
    #     _, L_yi_1 = sess.run([train_op, L_yi],
    #         feed_dict = {S:next_batch['out'], X:next_batch['top'], C:next_batch['label']})
    #     print(L_yi_1[0][0])
    test_batch_list = data_base.get_test(50)
    # print(test_batch_list)
    # print(sum(data_base.test_lenth_list))
    # print(data_base.test_lenth)
    to_pb = pb_writer(sess,'output','models/SA_v3_')
    for i_ in range(ITEMS):
        import random
        rand_k = random.gauss(1,0.1)
    	if i_>large_item: batch_size=large_batch
        next_batch = data_base.next_batch(batch_size)
        # test_batch_list = data_base.get_test(100)
        # _, loss,test_cross, acc2, loss_tight_v = sess.run([train_op, total_loss, loss_cross, accuracy, loss_tight],
        #     feed_dict = {S:next_batch['out'], X:next_batch['top'], Y:next_batch['mid'], C:next_batch['label']})
        _, loss,test_cross, acc2, loss_tight_v = sess.run([train_op, total_loss, loss_cross, accuracy, loss_tight],
            feed_dict = {N:next_batch['neck'], S:next_batch['out'],# Y:next_batch['mid'],
                X:next_batch['top'], C:next_batch['label'], r:rand_k, is_train:True})
        # print("step %d trai_acc=%f train_loss=%f cross=%f tight=%f"%(i,acc2,loss,test_cross,loss_tight_v))
        if i_%print_item ==1 :
            print("step %d trai_acc=%f train_loss=%f cross=%f tight=%f"%(i_,acc2,loss,test_cross,loss_tight_v))
            # test_batch_list = data_base.get_test(100)
            test_acc = 0.0
            for i in range(data_base.test_batch_n+1):
                acc_test,test_loss, test_cross, loss_tight_v = sess.run([accuracy ,total_loss, loss_cross, loss_tight],
                    feed_dict={N:test_batch_list['neck'][i], S:test_batch_list['out'][i],# Y:test_batch_list['mid'][i],
                        X:test_batch_list['top'][i], C:test_batch_list['label'][i], r:rand_k, is_train:False})            
                test_acc += acc_test * data_base.test_lenth_list[i]
            print("step %d test_acc=%f test_loss=%f cross=%f tight=%f"%(i_,test_acc/data_base.test_lenth, test_loss, test_cross, loss_tight_v))
            to_pb.save_max(test_acc/data_base.test_lenth)
