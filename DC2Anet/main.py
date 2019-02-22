# ---------------------------------------------------------
# Tensorflow SpineC2M-mrgan++ Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_bool('is_cycle_consistent', True, 'cycle-consistent loss for the generator, default: True')
tf.flags.DEFINE_float('cycle_consistent_weight', 10., 'weight for the cycle-consistent loss term, default: 10.')
tf.flags.DEFINE_bool('is_voxel', True, 'voxel-wise loss for the generator, default: True')
tf.flags.DEFINE_float('L1_lambda', 100., 'L1 lambda for conditional voxel-wise loss, default: 100.')
tf.flags.DEFINE_bool('is_gdl', True, 'gradient difference loss (GDL) for the generator, default: True')
tf.flags.DEFINE_float('gdl_weight', 100., 'weight (hyper-parameter) for gradient difference loss term, default: 100.')
tf.flags.DEFINE_bool('is_perceptual', True, 'perceptual loss for for the generator, default: True')
tf.flags.DEFINE_float('perceptual_weight', 1., 'weight (hyper-parameter) for perceputal loss term, default: 1.')
tf.flags.DEFINE_integer('perceptual_mode', 5, 'feature layers [1|2|3|4|5], default: 5')
tf.flags.DEFINE_bool('is_ssim', True, 'SSIM loss for the generator, default: True')
tf.flags.DEFINE_float('ssim_weight', 0.05, 'weight (hyper-parameter) for ssim loss term, default: 0.05')
tf.flags.DEFINE_string('dis_model', 'a', 'discriminator model, select from [a|b|c|d|e|f|g], default: a')
tf.flags.DEFINE_string('learning_mode', 'super', 'learning mode, select from [super, unsuper, semi], default, semi')
tf.flags.DEFINE_bool('is_alterative_optim', True, 'optimizing by alterative or integrated optimziation. default: True')
tf.flags.DEFINE_bool('is_lsgan', False, 'use LSGAN loss, default: False')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'spine06', 'dataset name, default: spine06')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial leraning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20181127-2116), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
