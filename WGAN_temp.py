from dataloader import DataGenerator
import tensorflow as tf
import numpy as np
import logging
import time
import os
import sys

src_path = '/home/jason/github/dataset/cocostuff/dataset/annotations'

logging.basicConfig(level=logging.DEBUG)

class Net:
    def __init__(self, shape):
        self.name = 'net'
        self.layer_index = 0
        self.x_shape = shape
        logging.debug('Net init')

    @property
    def next_layer_name(self):
        self.layer_index += 1
        return "{}_layer_{}".format(self.name, self.layer_index)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator(Net):
    def __init__(self, shape):
        super().__init__(shape)
        self.name = 'd_net'
        logging.debug('D init')

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            layer = tf.reshape(tensor=x, shape=self.x_shape)
            layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu, kernel_regularizer=None, name=self.next_layer_name)
            layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=[2, 2], padding='same', name=self.next_layer_name)
            layer = tf.layers.conv2d(inputs=layer, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu, kernel_regularizer=None, name=self.next_layer_name)
            layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=[2, 2], padding='same', name=self.next_layer_name)
            layer = tf.layers.flatten(inputs=layer, name=self.next_layer_name)
            layer = tf.layers.dense(inputs=layer, activation=tf.nn.relu, units=10, name=self.next_layer_name)
        return layer


class Generator(Net):
    def __init__(self, shape, z_shape):
        super().__init__(shape)
        self.z_shape = z_shape
        self.name = 'g_net'
        logging.debug('G init')

    def __call__(self, z, batch_size):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            #z_shape=[batch_size, self.z_shape[1]]
            #x_shape=[batch_size, self.x_shape[1], self.x_shape[2], self.x_shape[3]]
            layer = tf.reshape(tensor=z, shape=self.z_shape)
            layer = tf.layers.dense(inputs=layer, activation=tf.nn.sigmoid, units=32 * 32 * 3, name=self.next_layer_name)
            layer = tf.reshape(tensor=layer, shape=[-1, 32, 32, 3])
            layer = tf.image.resize_nearest_neighbor(images=layer, size=[layer.shape[1] * 2, layer.shape[2] * 2])
            layer = tf.layers.conv2d_transpose(inputs=layer, filters=32, kernel_size=[3, 3], strides=[1, 1],
                                               padding='same', activation=tf.nn.sigmoid, name=self.next_layer_name)
            layer = tf.image.resize_nearest_neighbor(images=layer, size=[layer.shape[1] * 2, layer.shape[2] * 2])
            layer = tf.layers.conv2d_transpose(inputs=layer, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                               padding='same', activation=tf.nn.sigmoid, name=self.next_layer_name)
            layer = tf.image.resize_nearest_neighbor(images=layer, size=[layer.shape[1] * 2, layer.shape[2] * 2])
            layer = tf.layers.conv2d_transpose(inputs=layer, filters=3, kernel_size=[3, 3], strides=[1, 1],
                                               padding='same', activation=tf.nn.sigmoid, name=self.next_layer_name)
            #             layer = tf.layers.flatten(inputs=layer, name='flatten_d')
            #             layer = tf.layers.dense(inputs=layer, activation=tf.nn.sigmoid, units=784, name='fc_d2')
            #layer = tf.reshape(tensor=layer, shape=self.x_shape)
            layer = tf.reshape(tensor=layer, shape=self.x_shape)
        return layer

##################################################################################
tf.reset_default_graph()

X_SHAPE = [-1, 256, 256, 3]
Z_SHAPE = [-1, 100]
BATCH_SIZE = 16

netD = Discriminator(X_SHAPE)
netG = Generator(X_SHAPE, Z_SHAPE)
global_step = tf.train.get_or_create_global_step()

z = tf.placeholder(tf.float32, [None, Z_SHAPE[1]], name='z')

real_images = tf.placeholder(tf.float32, [None, X_SHAPE[1], X_SHAPE[2], X_SHAPE[3]], name='x')

# generated images
gen_images = netG(z, BATCH_SIZE)

# get the output from D on the real and fake data
errD_real = netD(real_images)
errD_fake = netD(gen_images)

# cost functions
errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
errG = tf.reduce_mean(errD_fake)

# gradient penalty
epsilon = tf.random_uniform([], 0.0, 1.0)
#sampling x_hat
x_hat = real_images*epsilon + (1-epsilon) * gen_images
d_hat = netD(x_hat)
gradients = tf.gradients(d_hat, x_hat)[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
lambda_weight = 10 # suggested in paper
gradient_penalty = lambda_weight * tf.reduce_mean((slopes - 1.0) ** 2)
errD += gradient_penalty

# tensorboard summaries
tf.summary.scalar('d_loss', errD, collections=['train_loss'])
tf.summary.scalar('g_loss', errG, collections=['train_loss'])
merged_summary_op = tf.summary.merge_all(key='train_loss')

with tf.variable_scope('Optimizer', reuse=tf.AUTO_REUSE):
    # optimize G
    G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=netG.vars, global_step=global_step)

    # optimize D
    D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=netD.vars)

#################################################################################

train_dir = '/tmp/wgan/model/'
summary_dir = '/tmp/wgan/summary/'
restore_dir = '/tmp/wgan/restore/'
max_steps = 1
test_frequency = 1

n_critic_per_step = 1
best_test_G_loss = sys.maxsize

datagenerator = DataGenerator(src_path, BATCH_SIZE, X_SHAPE)

next_batch = datagenerator('train2017')
next_test_batch = datagenerator('val2017')

def add_value_summary(name, value, step, summary_writer):
    summary_time_epoch = tf.Summary()
    summary_time_epoch.value.add(tag=name, simple_value=value)
    summary_writer.add_summary(summary_time_epoch, global_step=step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_test_accuracy = 0

    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)

    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # restore path not empty
    if os.path.exists(os.path.dirname(restore_dir)):
        # gcloud_load()
        tf.train.Saver().restore(sess, restore_dir)

    for _ in range(max_steps):

        start = time.time()

        # train the discriminator for n_critic_per_step runs
        for critic_itr in range(n_critic_per_step):
            train_images = sess.run(next_batch)
            batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
            sess.run(D_train_op, feed_dict={real_images: train_images, z: batch_z})

        # now train the generator once! use normal distribution, not uniform!!
        train_images = sess.run(next_batch)
        batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
        _, D_loss, G_loss, summary, step = sess.run([G_train_op, errD, errG, merged_summary_op, global_step],
                                                    feed_dict={real_images: train_images, z: batch_z})

        summary_writer.add_summary(summary, step)

        time_epoch = time.time() - start

        add_value_summary('time_epoch', time_epoch, step, summary_writer)

        logging.debug("step: {}, D_loss: {}, G_loss: {}, time: {}".format(step, D_loss, G_loss, time_epoch))

        if step % test_frequency == 0 or step + 1 == max_steps:
            test_images = sess.run(next_test_batch)
            batch_test_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
            D_test_loss, G_test_loss = sess.run([errD, errG], feed_dict={real_images: test_images, z: batch_test_z})

            add_value_summary('test_G_loss', G_test_loss, step, summary_writer)
            add_value_summary('test_D_loss', D_test_loss, step, summary_writer)

            logging.debug("test step: {}, D_loss: {}, G_loss: {}".format(step, D_test_loss, G_test_loss))

            if best_test_G_loss > G_test_loss:
                best_test_G_loss = G_test_loss
                best_test_step = step
                tf.train.Saver().save(sess, save_path=train_dir, global_step=step)

                batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)

                gen_imgs = sess.run(gen_images, feed_dict={z: batch_z})

                tf.summary.image('original', test_images, collections=['image'])
                tf.summary.image('reconstruct', gen_imgs, collections=['image'])
                merge_image = tf.summary.merge_all(key='image')
                merge_image_summary = sess.run(merge_image)

                summary_writer.add_summary(merge_image_summary)

    summary_writer.flush()