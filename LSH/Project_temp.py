import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

# variables
batch_size = 200
input_scale = 30
bins = 200
iteration = 50001

initializer = tf.contrib.layers.xavier_initializer()

# random periodic potential generator
def genpot(j,k):
    gap = random.randint(20,60)
    width = int(gap/4) + j
    height = 80. + 1.0*k   # This should be float
    interval = gap - width
    bump = tf.constant(height,shape=[batch_size,width])
    plain = tf.zeros([batch_size,2*bins])
    pot1 = tf.concat([plain[:,:interval],bump],-1)
    pot2 = tf.concat([pot1,plain[:,interval+width:]],-1)
    pot3 = tf.concat([pot2[:,:2*interval+width],bump],-1)
    pot4 = tf.concat([pot3,plain[:,2*interval+2*width:]],-1)
    pot5 = tf.concat([pot4[:,:3*interval+2*width],bump],-1)
    pot6 = tf.concat([pot5,plain[:,3*interval+3*width:]],-1)
    pot7 = tf.concat([pot6[:,:4*interval+3*width],bump],-1)
    pot8 = tf.concat([pot7,plain[:,4*interval+4*width:]],-1)
    pot9 = tf.concat([pot8[:,:5*interval+4*width],bump],-1)
    pot10 = tf.concat([pot9,plain[:,5*interval+5*width:]],-1)
    pot11 = tf.concat([pot10[:,:6*interval+5*width],bump],-1)
    pot12 = tf.concat([pot11,plain[:,6*interval+6*width:]],-1)
    pot13 = tf.concat([pot12[:,:7*interval+6*width],bump],-1)
    pot14 = tf.concat([pot13,plain[:,7*interval+7*width:]],-1)
    pot15 = tf.concat([pot14[:,:8*interval+7*width],bump],-1)
    pot16 = tf.concat([pot15,plain[:,8*interval+8*width:]],-1)
    pot17 = tf.concat([pot16[:,:9*interval+8*width],bump],-1)
    pot18 = tf.concat([pot17,plain[:,9*interval+9*width:]],-1)
    pot19 = tf.concat([pot18[:,:10*interval+9*width],bump],-1)
    pot20 = tf.concat([pot19,plain[:,10*interval+10*width:]],-1)
    pot21 = tf.concat([pot20[:,:11*interval+10*width],bump],-1)
    pot22 = tf.concat([pot21,plain[:,11*interval+11*width:]],-1)
#    pot23 = tf.concat([pot22[:,:12*interval+11*width],bump],-1)
#    pot24 = tf.concat([pot23,plain[:,12*interval+12*width:]],-1)
    # slicing
    m = random.randint(1,gap)
    pot = pot22[:,m:m+200]
#    pot = tf.concat([pot20[:,m:m+200],plain[:,0:0]],-1)
    return pot,gap

x = tf.random_normal([batch_size,input_scale])
#potential = genpot(0,0)
W = tf.Variable(initializer([input_scale,bins]),dtype=tf.float32)
b = tf.Variable(initializer([bins]),dtype=tf.float32)

# loss
W_loss,b_loss = tf.nn.l2_loss(W),tf.nn.l2_loss(b)

# psi
binsnorm = tf.sqrt(tf.cast(bins,tf.float32))
psi = tf.matmul(x,W) + b
psi = tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi),axis=-1,keepdims=True)))/binsnorm

zeroten = tf.zeros([batch_size,1],tf.float32)
psil = tf.concat([psi[:,1:],zeroten],1)
psir = tf.concat([zeroten,psi[:,:-1]],1)

# using CPU
config = tf.ConfigProto(device_count={'GPU':0})
sess = tf.Session(config=config)

# trainer
def fabricator(train):
    for i in range(iteration):
        sess.run(train)
        if i+1 == iteration:
            psi_now,cost_now,W_loss_v,b_loss_v = sess.run([psi,cost,W_loss,b_loss])
            probability = psi_now**2
    return probability

f = open('./Data2/Dataset2.csv','w',newline='')
writer = csv.writer(f)
for j in range(10):
    for k in range(40):
        # generate a random periodic potential
        potential,gap = genpot(j,k)

        # energy and cost
        E = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(potential,1.*bins*bins)),tf.multiply(tf.multiply((tf.add(psil,psir)),psi),0.5*bins*bins)),axis=-1,keepdims=True)*bins
        cost = tf.reduce_mean(E) + 0.01*W_loss

        # train
        train = tf.train.AdamOptimizer(learning_rate=0.008).minimize(cost)
        init = tf.global_variables_initializer()
        sess.run(init)
        resultsum = 0

        preprocess = 50000*40*j + 50000*k
        # Let's start!
        for i in range(iteration):
            sess.run(train)
            psi_now,cost_now,W_loss_v,b_loss_v = sess.run([psi,cost,W_loss,b_loss])
            probability = psi_now**2
            if i%10000 == 0:
                progress = i + preprocess
                percentage = round(progress*100/(10*40*iteration),2)
                print(str(percentage)+'% complete')
                #print('iteration = %d, cost = %f' % (i, cost_now))
            elif i == iteration - 2:
                prob = probability[0]
                label = [float(gap)]
                data = np.zeros((1,201))
                data[:,:-1] = prob
                data[:,-1] = label
                #print(data)
                #print(gap)
                writer.writerow(data)
            else:
                pass
f.close()
