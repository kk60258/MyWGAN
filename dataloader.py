import os
import tensorflow as tf
import zipfile
import sys
from six.moves import urllib

URL = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip'
src_path = '/tmp/dataset/annotations'


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
        #self.maybe_download_and_extract(src_path)
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


if __name__ == '__main__':

    datagenerator = DataGenerator('/home/jason/github/dataset/cocostuff/dataset/annotations/', 32, [-1, 256, 256, 3])

    # next_batch = datagenerator('train2017')
    next_batch = datagenerator('val2017')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./summary', sess.graph)

        image = sess.run(next_batch)

        tf.summary.image('image', image)
        merged = tf.summary.merge_all()

        summary = sess.run(merged)
        summary_writer.add_summary(summary)
        summary_writer.flush()


