# ---------------------------------------------------------
# Tensorflow SpineC2M-mrgan Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by vanhuyz
# ---------------------------------------------------------
import time
import math
import tensorflow as tf


class Reader(object):
    def __init__(self, tfrecords_file, image_size=(300, 200, 1), min_queue_examples=100, batch_size=1,
                 num_threads=8, is_train=True, name=''):
        self.tfrecords_file = tfrecords_file
        self.ori_img_size = image_size
        self.resize_factor = 1.05
        self.rotate_angle = 5.
        self.image_size = (image_size[0], 2 * image_size[1] ,image_size[2])  # H, 2W, C

        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.is_train = is_train
        self.name = name

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example, features={
                'image/file_name': tf.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.FixedLenFeature([], tf.string)})

            image_buffer = features['image/encoded_image']
            img_name_buffer = features['image/file_name']
            image = tf.image.decode_jpeg(image_buffer, channels=self.image_size[2])
            x_img, y_img, x_img_ori, y_img_ori = self._preprocess(image, is_train=self.is_train)

            if self.is_train:
                x_imgs, y_imgs, x_imgs_ori, y_imgs_ori, img_name = tf.train.shuffle_batch(
                    [x_img, y_img, x_img_ori, y_img_ori, img_name_buffer], batch_size=self.batch_size,
                    num_threads=self.num_threads, capacity=self.min_queue_examples + 3 * self.batch_size,
                    min_after_dequeue=self.min_queue_examples)
            else:
                x_imgs, y_imgs, x_imgs_ori, y_imgs_ori, img_name = tf.train.batch(
                    [x_img, y_img, x_img_ori, y_img_ori, img_name_buffer], batch_size=self.batch_size,
                    num_threads=1, capacity=self.min_queue_examples + 3 * self.batch_size,
                    allow_smaller_final_batch=True)

        return x_imgs, y_imgs, x_imgs_ori, y_imgs_ori, img_name

    def _preprocess(self, img, is_train):
        # Resize to 2D and split to left and right image
        img = tf.image.resize_images(img, size=(self.image_size[0], self.image_size[1]))
        x_img_ori, y_img_ori = tf.split(img, [self.ori_img_size[1], self.ori_img_size[1]], axis=1)

        x_img, y_img = x_img_ori, y_img_ori
        # Data augmentation
        if is_train:
            random_seed = int(round(time.time()))

            # Make image bigger
            x_img = tf.image.resize_images(x_img_ori, size=(int(self.resize_factor * self.ori_img_size[0]),
                                                          int(self.resize_factor * self.ori_img_size[1])))
            y_img = tf.image.resize_images(y_img_ori, size=(int(self.resize_factor * self.ori_img_size[0]),
                                                          int(self.resize_factor * self.ori_img_size[1])))

            # Random crop
            x_img = tf.random_crop(x_img, size=self.ori_img_size, seed=random_seed)
            y_img = tf.random_crop(y_img, size=self.ori_img_size, seed=random_seed)

            # Random flip
            x_img = tf.image.random_flip_left_right(x_img, seed=random_seed)
            y_img = tf.image.random_flip_left_right(y_img, seed=random_seed)

            # Random rotate
            radian_min = -self.rotate_angle * math.pi / 180.
            radian_max = self.rotate_angle * math.pi / 180.
            random_angle = tf.random_uniform(shape=[1], minval=radian_min, maxval=radian_max, seed=random_seed)
            x_img = tf.contrib.image.rotate(x_img, angles=random_angle, interpolation='NEAREST')
            y_img = tf.contrib.image.rotate(y_img, angles=random_angle, interpolation='NEAREST')

        # Do resize and zero-centering
        x_img = self.basic_preprocess(x_img)
        y_img = self.basic_preprocess(y_img)
        x_img_ori = self.basic_preprocess(x_img_ori)
        y_img_ori = self.basic_preprocess(y_img_ori)

        return x_img, y_img, x_img_ori, y_img_ori

    def basic_preprocess(self, img):
        img = tf.image.resize_images(img, size=(self.ori_img_size[0], self.ori_img_size[1]))
        img = (tf.image.convert_image_dtype(img, dtype=tf.float32) / 127.5) - 1.0
        img.set_shape(self.ori_img_size)

        return img
