import tensorflow as tf 
import numpy as np 

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
    print(result.shape)
    return result.astype(np.float32)


NN = 1000
n = 8
f = 1024
c = 101
S = tf.placeholder(tf.float32, [NN, c, 1])
X = tf.placeholder(tf.float32, [NN, n, n, f], name='X')
M = tf.Variable(tf.random_normal([1,1,f,c]))  
# filter = tf.Variable(tf.random_normal([1,1,5,1]))  
B = tf.nn.conv2d(X, M, strides=[1, 1, 1, 1], padding='VALID')
B = tf.nn.relu(B)
B_r = tf.reshape(B, [-1, n*n, c])
Z_r = tf.nn.softmax(tf.matmul(B_r, S))
Z = tf.matmul(tf.transpose(Z_r, perm=[0,2,1]),tf.reshape(X,[-1,n*n,f]))
Z = tf.reshape(Z, [-1,f], name='features')
# Z = tf.reshape(Z_r, [-1, n, n, 1])
print(Z)

L = tf.placeholder(tf.float32, [NN, c], name='L')
L_yi = tf.Variable(hang_matric(c),name='L_yi')
print(L_yi)
L_sum = tf.expand_dims(tf.reduce_sum(L, 0),-1, name='L_sum')
print(L_sum)
C = tf.placeholder(tf.int32, [NN, 1], name='C')
print(L)
Z_c = tf.matmul( tf.expand_dims(tf.transpose(L),0), tf.expand_dims(Z,0), name='class_center')
Z_c = tf.expand_dims(tf.squeeze(tf.transpose(Z_c, perm=[1,2,0]))/L_sum, 0,name='class_c')
print(Z_c)
Z_c_3 = Z_c
Z_n = tf.matmul( tf.expand_dims(L,0), Z_c, name='class_n')
print(Z_n)
Z_c = tf.squeeze(tf.transpose(Z_c, [1,2,0]))
Z_list = tf.unstack(Z, num=NN,axis=0)
print(Z_list[0])
print(tf.reduce_sum(tf.square(Z_list[0])))
loss_tight = tf.Variable(0.0, tf.float32)

Z_n_i = Z - tf.squeeze(tf.transpose(Z_n, [1,2,0]))
Z_n_i_s =  tf.expand_dims(tf.reduce_sum(Z_n_i,1),-1)
print(Z_n_i)
for i in range(c-1):
    Z_nj = tf.matmul( tf.matmul(tf.expand_dims(L,0),L_yi), Z_c_3)
    Z_n_j = Z - tf.squeeze(tf.transpose(Z_nj, [1,2,0]))
    Z_n_j_s =  tf.expand_dims(tf.reduce_sum(Z_n_j,1),-1)
    loss_tight += tf.reduce_sum(tf.div(Z_n_i_s, Z_n_j_s),0)

print(Z_nj)
print(Z_n_j)
print(Z_n_i_s)
print(Z_n_j_s)
print(loss_tight)
print(tf.div(Z_n_i_s, Z_n_j_s))



# for i in range(NN):
#     z_n = Z_list[i]
#     z_nc = C[i]
#     z_i = Z_c[z_nc[0]]
#     for j in range(c):
#         if j == z_nc[0] : continue
#         z_j = Z_c[j]
#         loss_tight += tf.reduce_sum(tf.square(z_n-z_i)) / tf.reduce_sum(tf.square(z_n-z_j))
# print(loss_tight)