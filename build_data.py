# ---------------------------------------------------------
# Tensorflow build_data Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by vanhuyz
# ---------------------------------------------------------
import os
import random
import numpy as np
from datetime import datetime
from os import scandir
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dataA', '../train', 'data A input directory, default: ../train')
tf.flags.DEFINE_string('input_dataB', '../test', 'data B input directory, default: ../test')
tf.flags.DEFINE_string('output_dataA', 'train', 'data A output directory, default: train')
tf.flags.DEFINE_string('output_dataB', 'test', 'data B output directory, default: test')
tf.flags.DEFINE_string('extension', '.jpg', 'image extension, default: .jpg')


def data_writer(input_dir, output_name, extension='.jpg'):
    file_paths = data_reader(input_dir, extension=extension)
    num_imgs = len(file_paths)

    # create tfrecords dir if not exists
    output_file = '../tfrecords/{}.tfrecords'.format(output_name)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for idx in range(num_imgs):
        img_path = file_paths[idx]

        with tf.gfile.FastGFile(img_path, 'rb') as f:
            img_data = f.read()

        example = _convert_to_example(img_path, img_data)
        writer.write(example.SerializeToString())

        if np.mod(idx, 100) == 0:
            print('Processed {}/{}...'.format(idx, num_imgs))

    print('Finished!')
    writer.close()


def data_reader(input_dir, extension='.jpg', is_shuffle=True):
    file_paths = []
    print('input_dir: {}'.format(input_dir))

    for img_file in scandir(input_dir):
        if img_file.name.endswith(extension) and img_file.is_file():
            file_paths.append(img_file.path)

    # shuffle the ordering of all iamge files in order to guarantee random ordering of the images with
    # respect to label in the saved TFRecord files. Make the randomization repeatable.
    if is_shuffle:
        shuffled_index = list(range(len(file_paths)))
        random.seed(datetime.now())
        random.shuffle(shuffled_index)

        file_paths = [file_paths[idx] for idx in shuffled_index]

    return file_paths


def _convert_to_example(img_path, img_buffer):
    # build an example proto
    img_name = os.path.basename(img_path)

    example = tf.train.Example(features=tf.train.Features(
        feature={'image/file_name': _bytes_feature(tf.compat.as_bytes(img_name)),
                 'image/encoded_image': _bytes_feature(img_buffer)}))

    return example


def _bytes_feature(value):
    # wrapper for inserting bytes features into example proto
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    print("Convert {} data to tfrecords...".format(FLAGS.input_dataA))
    data_writer(FLAGS.input_dataA, FLAGS.output_dataA, extension=FLAGS.extension)

    print("Convert {} data to tfrecords...".format(FLAGS.input_dataB))
    data_writer(FLAGS.input_dataB, FLAGS.output_dataB, extension=FLAGS.extension)


if __name__ == '__main__':
    tf.app.run()
