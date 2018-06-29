from __future__ import print_function

import tensorflow as tf
import numpy as np
from models import *
import h5py

# -----------------------------

LOSS = 'VGG54' # 'MSE', 'WGAN'

# *** Please update the path!!!
f = h5py.File('/YOUR/PATH/TO/TRAINING.hdf5', 'r')
data = f.get('input')
label = f.get('label')
f.close()

# *** Please update the path!!!
f = h5py.File('/YOUR/PATH/TO/TESTING.hdf5', 'r')
test_data = f.get('input')
test_label = f.get('label')
f.close()

input_width = data.shape[1]
input_height = data.shape[2]

output_width = label.shape[1]
output_height = label.shape[2]

batch_size = 128

# generator networks
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_width, input_height, 1])
with tf.variable_scope('cnn_model') as scope:
    Y_ = cnn_model(X, padding='valid')

# discriminator networks
real_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_width, output_height, 1])

alpha = tf.random_uniform(
    shape=[batch_size,1], 
    minval=0.,
    maxval=1.
)

with tf.variable_scope('discriminator_model') as scope:
    disc_real = discriminator_model(real_data)
    scope.reuse_variables()
    disc_fake = discriminator_model(Y_)
    interpolates = alpha*tf.reshape(real_data, [batch_size, -1]) + (1-alpha)*tf.reshape(Y_, [batch_size, -1])
    interpolates = tf.reshape(interpolates, [batch_size, output_width, output_height, 1])
    gradients = tf.gradients(discriminator_model(interpolates), [interpolates])[0]

gen_cost = -1.0 * tf.reduce_mean(disc_fake)    # generator loss
w_distance = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)  # discriminator loss

slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost = w_distance + gradient_penalty   # add gradient constraint to discriminator loss

gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn_model')
disc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_model')

# vgg network
with tf.variable_scope('vgg19') as scope:
    vgg_real = vgg_model(real_data)
    scope.reuse_variables()
    vgg_fake = vgg_model(Y_)

vgg_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19')
mse_cost = tf.reduce_sum(tf.squared_difference(Y_, real_data)) / (2.0 * batch_size)
vgg_cost = tf.reduce_sum(tf.squared_difference(vgg_real, vgg_fake)) / (2.0 * batch_size)

# generator loss
if LOSS == 'VGG54': 
    gen_cost = gen_cost + 1e-1 * vgg_cost
elif LOSS == 'MSE':
    gen_cost = gen_cost + 1e-1 * mse_cost

# optimizer
gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(gen_cost, var_list=gen_params)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5, 
    beta2=0.9
).minimize(disc_cost, var_list=disc_params)


# training
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# load vgg weights
print("Initialize VGG network ... ")

# *** Please update the path!!!
weights = np.load('/YOUR/PATH/TO/vgg19.npy', encoding='latin1').item()
keys = sorted(weights.keys())
layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
        'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']

for i, k in enumerate(layers):
    print(i, k, weights[k][0].shape, weights[k][1].shape)
    sess.run(vgg_params[2*i].assign(weights[k][0]))
    sess.run(vgg_params[2*i+1].assign(weights[k][1]))

num_epoch = 100
disc_iters = 4 
num_batches = data.shape[0] // batch_size
saver = tf.train.Saver()


print("Start training ... ")
for iteration in range(num_epoch):

    i = 0

    while i < num_batches:

        # discriminator
        for j in range(disc_iters):
            index = np.random.randint(0, data.shape[0]-batch_size, size=1)[0]
            batch_data = np.array(data[index:(index+batch_size)])
            batch_label = np.array(label[index:(index+batch_size)])
            _disc_cost, _w_distance, _ = sess.run([disc_cost, w_distance, disc_train_op], feed_dict={real_data: batch_label, X: batch_data})


        # generator
        batch_data = np.array(data[i*batch_size:(i+1)*batch_size])
        batch_label = np.array(label[i*batch_size:(i+1)*batch_size])

        _gen_cost, _ = sess.run([gen_cost, gen_train_op], feed_dict={X: batch_data, real_data: batch_label})
        print('Epoch: %d - num_batch: %d - gen_loss: %.6f - disc_loss: %.6f - w_distance: %.6f' % (iteration, i+1, _gen_cost, _disc_cost, _w_distance))

        i = i + 1
    if LOSS == 'VGG54':
        saver.save(sess, './Network/wgan-vgg54/wgan_vgg54_'+repr(iteration)+'.ckpt')
    elif LOSS == 'MSE':
        saver.save(sess, './Network/wgan-mse/wgan_mse_'+repr(iteration)+'.ckpt')
    else:
        saver.save(sess, './Network/wgan/wgan_'+repr(iteration)+'.ckpt')

sess.close()

