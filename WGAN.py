import tensorflow as tf
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.DEBUG)

"""
"""

import zipfile
import sys
from six.moves import urllib

URL = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip'
# src_path = '/tmp/dataset/annotations'
src_path = '/home/jason/github/dataset/cocostuff/dataset/annotations'

def list_files(src_path):
    name = []
    for filename in os.listdir(src_path):
        path = os.path.join(src_path, filename)
        name.append(path)
    print(name)
    return name


def tfconvert(image):
    #return tf.divide(tf.subtract(image, 127.5), 255.0)
    return tf.subtract(tf.divide(image, 127.5), 1)


def tfrevert(image):
    #     return tf.add(tf.multiply(image, 255.0), 127.5)
    return tf.clip_by_value(tf.multiply(tf.add(image, 1), 127.5), 0, 255)


class CocoAnnotationData:
    def __init__(self, src_path):
        self.src_path = src_path
        # self.maybe_download_and_extract(src_path)
        pass

    @staticmethod
    def maybe_download_and_extract(data_dir=src_path, DATA_URL=URL):
        """Download and extract the tarball from Alex's website."""
        dest_directory = data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            zipfile.ZipFile(filepath).extractall(dest_directory)
        return extracted_dir_path

    def __call__(self, sub_set_path, batch_size=32, buffle_size=1000, shape=[-1, 256, 256, 3]):
        self.shape = shape
        src_path = os.path.join(self.src_path, sub_set_path)
        filenames = list_files(src_path)
        dataset = self._get_dataset(filenames, batch_size, shape)
        # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffle_size))
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset

    def _get_dataset(self, filenames, batch_size, shape, augmentation=False):
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # input format.
        height = shape[1]
        width = shape[2]
        depth = shape[3]

        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        def transform(filename):
            image_string = tf.read_file(filename)
            image = tf.image.decode_png(image_string, channels=3)
            #image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            image = tf.cast(image, dtype=tf.float32)

            image = tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            image = tf.reshape(image, [height, width, depth])
            # if augmentation:
            #
            #     # Image processing for training the network. Note the many random
            #     # distortions applied to the image.
            #
            #     # Randomly crop a [height, width] section of the image.
            #     distorted_image = tf.random_crop(image, [height, width, 3])
            #
            #     # Randomly flip the image horizontally.
            #     distorted_image = tf.image.random_flip_left_right(distorted_image)
            #
            #     # Because these operations are not commutative, consider randomizing
            #     # the order their operation.
            #     # NOTE: since per_image_standardization zeros the mean and makes
            #     # the stddev unit, this likely has no effect see tensorflow#1458.
            #     distorted_image = tf.image.random_brightness(distorted_image,
            #                                                  max_delta=63)
            #     distorted_image = tf.image.random_contrast(distorted_image,
            #                                                lower=0.2, upper=1.8)
            #     image = distorted_image

            #             image = tf.image.per_image_standardization(image)
            image = tfconvert(image)
            return image

        #         dataset = dataset.map(map_func=transform, num_parallel_calls=8)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=transform, batch_size=batch_size, num_parallel_batches=8))

        return dataset
"""
"""

class DataGenerator:
    def __init__(self, dir, batch_size, shape):
        self.coco = CocoAnnotationData(dir)
        self.batch_size = batch_size
        self.shape = shape

    def __call__(self, sub_set_path):
        dataset = self.coco(sub_set_path, self.batch_size, self.shape)
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        return next_batch

"""
"""

class Net:
    def __init__(self, shape):
        self.name = 'net'
        self.layer_index = 0
        self.x_shape = shape

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

"""
"""

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
    G_train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(errG, var_list=netG.vars, global_step=global_step)

    # optimize D
    D_train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(errD, var_list=netD.vars)

"""
"""


TENSORBOARD_PATH = '/tmp/wgan/summary'
restore_dir = '/tempssd/wgan/restore/'
max_steps = 1
test_frequency = 1

n_critic_per_step = 1
best_test_G_loss = sys.maxsize

datagenerator = DataGenerator(src_path, BATCH_SIZE, X_SHAPE)

next_batch = datagenerator('train2017')
next_test_batch = datagenerator('val2017')

from time import gmtime, strftime
SAVE_PATH = '/tempssd/wgan/save/'

timestring = strftime("%Y_%b_%d_%H_%M_%S", gmtime())
default_dir = os.path.join(SAVE_PATH, timestring)
summmary_path = os.path.join(default_dir, 'summary')
if not (os.path.isdir(summmary_path)):
    os.makedirs(summmary_path)

model_path = os.path.join(default_dir, 'model')
if not (os.path.isdir(model_path)):
    os.makedirs(model_path)

best_model_name = os.path.join(model_path, "step")

def add_value_summary(name, value, step, summary_writer):
    summary_time_epoch = tf.Summary()
    summary_time_epoch.value.add(tag=name, simple_value=value)
    summary_writer.add_summary(summary_time_epoch, global_step=step)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_test_accuracy = 0

    summary_writer = tf.summary.FileWriter(summmary_path, sess.graph)

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
                tf.train.Saver().save(sess, save_path=best_model_name, global_step=step)

                batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
                gen_imgs = sess.run(gen_images, feed_dict={z: batch_z})

                tf.summary.image('original', tfrevert(test_images), collections=['image'])
                tf.summary.image('reconstruct', tfrevert(gen_imgs), collections=['image'])
                merge_image = tf.summary.merge_all(key='image')
                merge_image_summary = sess.run(merge_image)

                summary_writer.add_summary(merge_image_summary)

    summary_writer.flush()

    if not (os.path.isdir(os.path.dirname(TENSORBOARD_PATH))):
        os.makedirs(os.path.dirname(TENSORBOARD_PATH))

    if os.path.exists(TENSORBOARD_PATH):
        os.unlink(TENSORBOARD_PATH)

    os.symlink(summmary_path, TENSORBOARD_PATH)

print("model: {} \nsummary: {}".format(best_model_name, summmary_path))

"""
"""