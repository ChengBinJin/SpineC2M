# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by vanhuyz
# ---------------------------------------------------------
import tensorflow as tf


class Reader(object):
    def __init__(self, tfrecords_file, image_size=(300, 400, 1), min_queue_examples=100, batch_size=1,
                 num_threads=8, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.channel = self.image_size[2]
        self.name = name

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example, features={
                'image/file_name': tf.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.FixedLenFeature([], tf.string)})

            image_buffer = features['image/encoded_image']
            image = tf.image.decode_jpeg(image_buffer, channels=self.channel)
            image = self._preprocess(image)
            images = tf.train.shuffle_batch([image], batch_size=self.batch_size, num_threads=self.num_threads,
                                            capacity=self.min_queue_examples + 3 * self.batch_size,
                                            min_after_dequeue=self.min_queue_examples)

        return images

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size[0], self.image_size[1]))
        image = (tf.image.convert_image_dtype(image, dtype=tf.float32) / 127.5) - 1.0
        image.set_shape(self.image_size)

        return image
