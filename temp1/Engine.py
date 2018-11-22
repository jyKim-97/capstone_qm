import tensorflow as tf
import matplotlib.pyplot as plt
import random

batch_size = 100
input_scale = 30
bins = 100         # for convenience
iteration = 10000

# input data
x = tf.random_normal([batch_size, input_scale])
potential = tf.zeros([batch_size, bins])

# weight and bias
W = tf.Variable(tf.random_normal([input_scale, bins]), dtype = tf.float32)
b = tf.Variable(tf.random_normal([batch_size,bins]), dtype = tf.float32)

psi = tf.matmul(x, W) + b   # y = Wx + b
psi = tf.divide(psi, tf.sqrt(tf.reduce_mean(tf.square(psi))))

location = 40
bumpsize = 5

zeroten = tf.zeros([batch_size, 1], tf.float32)
bump = tf.Variable(zeroten)
bump
zeroten[39:44,1] = 10
left = tf.concat([psi[:,1:], zeroten], 1)           # left boundary condition
right = tf.concat([zeroten, psi[:,:-1]], 1)         # right boundary condition

# E = rm[((V + bins^2) * psi^2) - ((l + r) * psi) * 0.5 * bins^2]
E = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi), tf.add(potential, 1.0*bins*bins)), tf.multiply(tf.multiply(tf.add(left, right), psi), 0.5*bins*bins)), axis = -1)
cost = tf.reduce_mean(E)    # regard energy as a cost!

# train to minimize 'cost = energy' using Adam Optimizer
train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

# using CPU
config = tf.ConfigProto(device_count = {'GPU': 0})

sess = tf.Session(config = config)          # create a session
init = tf.global_variables_initializer()    # initializing global variables used in learning
sess.run(init)                              # run the session

for i in range(iteration):
    psi_tmp = sess.run(psi)
    prob_tmp = psi_tmp**2
    cost_tmp = sess.run(cost)
    sess.run(train)
    if i % 1000 == 0:
        print("iteration = %d, cost = %f" % (i, cost_tmp))
    elif i>8750:
        plt.plot(prob_tmp[0]/max(prob_tmp[0]))
        plt.axis('off')
        plt.savefig('./Dataset/'+str(i)+'.png')
        plt.clf()
