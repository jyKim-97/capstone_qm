import tensorflow as tf
import numpy as np
import csv
import pandas as pd

# generating matrix with loadtxt()
data = np.genfromtxt('Dataset3.csv')
wavfn = data[:,0:-1]    # every row, every column except the last
label = data[:,[-1]]    # every row, last column

tfph_BNbool = tf.placeholder(tf.bool)

# shape examination
#print(wavfn.shape, '\n', wavfn)
#print(label.shape, '\n', label)

Xdim_1 = 50
Xdim_2 = 50
deep_total_iteration = 100000
deep_print_iteration = 1000

initializer = tf.contrib.layers.xavier_initializer()

# placeholder
X = tf.placeholder(tf.float32, shape=[None,200])
#Y = tf.placeholder(tf.float32, shape=[None,1])

W1 = tf.Variable(initializer([200,Xdim_1]),dtype=tf.float32)
b1 = tf.Variable(tf.random_normal([Xdim_1]),dtype=tf.float32)
L1 = tf.matmul(X, W1) + b1
L1 = tf.layers.batch_normalization(L1, training=tfph_BNbool)
L1 = tf.nn.softplus(L1)

W2 = tf.Variable(initializer([Xdim_1,Xdim_2]),dtype=tf.float32)
b2 = tf.Variable(initializer([Xdim_2]),dtype=tf.float32)
L2 = tf.matmul(L1, W2) + b2
L2 = tf.layers.batch_normalization(L2, training=tfph_BNbool)
L2 = tf.nn.softplus(L2)

W3 = tf.Variable(initializer([Xdim_2,1]),dtype=tf.float32)
b3 = tf.Variable(initializer([1]),dtype=tf.float32)
Y = tf.nn.softplus(tf.matmul(L2,W3) + b3)
Y_ = tf.placeholder(tf.float32, shape=[None,1])

deep_cost = tf.reduce_mean(tf.square(Y - Y_))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    deep_train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(deep_cost)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(init)

#predict = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
#accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

for i in range(deep_total_iteration):
    sess.run(deep_train, feed_dict={X: wavfn, Y_: label, tfph_BNbool: True})
    if i%deep_print_iteration == 0:
        deep_cost_v = sess.run(deep_cost, feed_dict={X: wavfn, Y_: label, tfph_BNbool: True})
        print(str((100.*i)/deep_total_iteration) + '% complete')
print('Process Complete')

# Hypothesis
#hypothesis = tf.matmul(X,W) + b

# cost/loss function
#cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
#train = optimizer.minimize(cost)

# launch the graph in a session
#sess = tf.Session()

# initialize global variables in the graph
#sess.run(tf.global_variables_initializer())

#iteration = 1000001

#for step in range(iteration):
#    cost_v, hypo_v, misc = sess.run([cost, hypothesis, train], feed_dict={X: wavfn, Y: label})
#    if step%10000 == 0:
#        print(step, 'cost: ', cost_v)

test = np.genfromtxt('Dataset.csv')
number = int(input("Test Number: "))
wav_t = test[number, 0:-1]
a = [wav_t]
ans_t = test[number, [-1]]
b = [ans_t]

examination = sess.run(Y, feed_dict={X: a, tfph_BNbool: False})
print(examination)
print(b)
#print(wav_t)
#print(ans_t)
#print("Test: ", sess.run(hypothesis, feed_dict={X: a}))
#print("Answer: ", b)
