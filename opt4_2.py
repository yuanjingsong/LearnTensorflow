import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455
COST = 1
PROFIT = 9
rdm = np.random.RandomState(seed)
X = rdm.rand(32, 2)
Y_ = [[x1+x2+(rdm.rand()/ 10.0-0.05)] for (x1, x2) in X]
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)
#w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#a = tf.matmul(x, w1)
#y = tf.matmul(a, w2)

loss_mse = tf.reduce_mean(tf.square(y_-y))
loss_mse = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y)* PROFIT))
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 3500
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict = {x: X[start:end],y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print ("After %d training step(s) loss on ad data is" % (i))
            print sess.run(w1), "\n"
    print "w1: \n" ,sess.run(w1)
