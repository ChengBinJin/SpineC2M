# ---------------------------------------------------------
# Tensorflow SpineC2M-pix2pix Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from vanhuyz
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import logging
import collections
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

# noinspection PyPep8Naming
import tensorflow_utils as tf_utils
import utils as utils
from reader import Reader

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
class Pix2Pix(object):
    def __init__(self, sess, flags, img_size, data_path, log_path=None):
        self.sess = sess
        self.flags = flags
        self.img_size = img_size
        self.data_path = data_path
        self.log_path = log_path

        self.L1_lamba = 100.
        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [64, 128, 256, 512, 512, 512, 512, 512,
                      512, 512, 512, 512, 256, 128, 64, self.img_size[2]]
        self.dis_c = [64, 128, 256, 512, 1]
        self.start_decay_step = int(self.flags.iters / 2)
        self.decay_steps = self.flags.iters - self.start_decay_step
        self.eps = 1e-12

        self._init_logger()     # init logger
        self._build_net()       # init graph
        self._tensorboard()     # init tensorboard

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        # tfph: TensorFlow PlaceHolder
        self.x_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='x_test_tfph')
        self.generator = Generator(name='gen', gen_c=self.gen_c, image_size=self.img_size, _ops=self._gen_train_ops)
        self.discriminator = Discriminator(name='dis', dis_c=self.dis_c, _ops=self._dis_train_ops)

        data_reader = Reader(self.data_path, name='data', image_size=self.img_size, batch_size=self.flags.batch_size,
                             is_train=self.flags.is_train)
        # self.x_imgs_ori and self.y_imgs_ori are the images before data augmentation
        self.x_imgs, self.y_imgs, self.x_imgs_ori, self.y_imgs_ori, self.img_name = data_reader.feed()

        self.g_samples = self.generator(self.x_imgs)
        self.real_pair = tf.concat([self.x_imgs, self.y_imgs], axis=3)
        self.fake_pair = tf.concat([self.x_imgs, self.g_samples], axis=3)

        d_logit_real = self.discriminator(self.real_pair)
        d_logit_fake = self.discriminator(self.fake_pair)

        # discriminator loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # generator loss
        self.gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        self.cond_loss = self.L1_lamba * tf.reduce_mean(tf.abs(self.y_imgs - self.g_samples))
        self.g_loss = self.gan_loss + self.cond_loss

        gen_op = self.optimizer(loss=self.g_loss, variables=self.generator.variables, name='Adam_gen')
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

        dis_op = self.optimizer(loss=self.d_loss, variables=self.discriminator.variables, name='Adam_dis')
        dis_ops = [dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)

        # for sampling function
        self.fake_y_sample = self.generator(self.x_test_tfph)

    def optimizer(self, loss, variables, name='Adam'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.flags.learning_rate
        end_learning_rate = 0.
        start_decay_step = self.start_decay_step
        decay_steps = self.decay_steps

        learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                  tf.train.polynomial_decay(starter_learning_rate,
                                                            global_step - start_decay_step,
                                                            decay_steps, end_learning_rate, power=1.0),
                                  starter_learning_rate))
        tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learn_step = tf.train.AdamOptimizer(learning_rate, beta1=self.flags.beta1, name=name).\
            minimize(loss, global_step=global_step, var_list=variables)

        return learn_step

    def _tensorboard(self):
        tf.summary.scalar('loss/gen_total_loss', self.g_loss)
        tf.summary.scalar('loss/gen_gan_loss', self.gan_loss)
        tf.summary.scalar('loss/gen_cond_loss', self.cond_loss)
        tf.summary.scalar('loss/dis_total_loss', self.d_loss)
        self.summary_op = tf.summary.merge_all()

    def train_step(self):
        dis_ops = [self.dis_optim, self.d_loss]
        gen_ops = [self.gen_optim, self.gan_loss, self.cond_loss, self.g_loss, self.summary_op]

        _, d_loss = self.sess.run(dis_ops)
        _, gan_loss, cond_loss, g_loss, summary = self.sess.run(gen_ops)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, gan_loss, cond_loss, g_loss, summary = self.sess.run(gen_ops)

        return [gan_loss, cond_loss, g_loss, d_loss], summary

    def test_step(self):
        x_vals, y_vals, img_name = self.sess.run([self.x_imgs, self.y_imgs, self.img_name])
        fakes_y = self.sess.run(self.fake_y_sample, feed_dict={self.x_test_tfph: x_vals})

        return [x_vals, fakes_y, y_vals], img_name

    def sample_imgs(self):
        x_vals, y_vals = self.sess.run([self.x_imgs, self.y_imgs])
        fakes_y = self.sess.run(self.fake_y_sample, feed_dict={self.x_test_tfph: x_vals})

        return [x_vals, fakes_y, y_vals]

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('gen_gan_loss', loss[0]), ('gen_cond_loss', loss[1]),
                                                  ('gen_total_loss', loss[2]), ('dis_total_loss', loss[3]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    @staticmethod
    def plots(imgs, iter_time, image_size, save_file):
        # parameters for plot size
        scale, margin = 0.02, 0.02
        n_cols, n_rows = len(imgs), imgs[0].shape[0]
        cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow((imgs[col_index][row_index]).reshape(image_size[0], image_size[1]), cmap='Greys_r')

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time).zfill(5)), bbox_inches='tight')
        plt.close(fig)

    def plots_test(self, imgs, img_name, save_file, eval_file, gt_file):
        num_imgs = len(imgs)

        canvas = np.zeros((self.img_size[0], num_imgs * self.img_size[1]), np.uint8)
        for idx in range(num_imgs):
            canvas[:, idx * self.img_size[1]: (idx+1) * self.img_size[1]] = \
                np.squeeze(255. * utils.inverse_transform(imgs[idx]))

        img_name_ = img_name.astype('U26')[0]
        # save imgs on test folder
        cv2.imwrite(os.path.join(save_file, img_name_), canvas)
        # save imgs on eval folder
        cv2.imwrite(os.path.join(eval_file, img_name_), canvas[:,self.img_size[1]:2*self.img_size[1]])
        # save imgs on gt folder
        cv2.imwrite(os.path.join(gt_file, img_name_), canvas[:, 2*self.img_size[1]:3*self.img_size[1]])

class Generator(object):
    def __init__(self, name=None, gen_c=None, image_size=(300, 200, 1), _ops=None):
        self.name = name
        self.gen_c = gen_c
        self.image_size = image_size
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (300, 200) -> (150, 100)
            e0_conv2d = tf_utils.conv2d(x, self.gen_c[0], name='e0_conv2d')
            e0_lrelu = tf_utils.lrelu(e0_conv2d, name='e0_lrelu')

            # (150, 100) -> (75, 50)
            e1_conv2d = tf_utils.conv2d(e0_lrelu, self.gen_c[1], name='e1_conv2d')
            e1_batchnorm = tf_utils.batch_norm(e1_conv2d, name='e1_batchnorm', _ops=self._ops)
            e1_lrelu = tf_utils.lrelu(e1_batchnorm, name='e1_lrelu')

            # (75, 50) -> (38, 25)
            e2_conv2d = tf_utils.conv2d(e1_lrelu, self.gen_c[2], name='e2_conv2d')
            e2_batchnorm = tf_utils.batch_norm(e2_conv2d, name='e2_batchnorm', _ops=self._ops)
            e2_lrelu = tf_utils.lrelu(e2_batchnorm, name='e2_lrelu')

            # (38, 25) -> (19, 13)
            e3_conv2d = tf_utils.conv2d(e2_lrelu, self.gen_c[3], name='e3_conv2d')
            e3_batchnorm = tf_utils.batch_norm(e3_conv2d, name='e3_batchnorm', _ops=self._ops)
            e3_lrelu = tf_utils.lrelu(e3_batchnorm, name='e3_lrelu')

            # (19, 13) -> (10, 7)
            e4_conv2d = tf_utils.conv2d(e3_lrelu, self.gen_c[4], name='e4_conv2d')
            e4_batchnorm = tf_utils.batch_norm(e4_conv2d, name='e4_batchnorm', _ops=self._ops)
            e4_lrelu = tf_utils.lrelu(e4_batchnorm, name='e4_lrelu')

            # (10, 7) -> (5, 4)
            e5_conv2d = tf_utils.conv2d(e4_lrelu, self.gen_c[5], name='e5_conv2d')
            e5_batchnorm = tf_utils.batch_norm(e5_conv2d, name='e5_batchnorm', _ops=self._ops)
            e5_lrelu = tf_utils.lrelu(e5_batchnorm, name='e5_lrelu')

            # (5, 4) -> (3, 2)
            e6_conv2d = tf_utils.conv2d(e5_lrelu, self.gen_c[6], name='e6_conv2d')
            e6_batchnorm = tf_utils.batch_norm(e6_conv2d, name='e6_batchnorm', _ops=self._ops)
            e6_lrelu = tf_utils.lrelu(e6_batchnorm, name='e6_lrelu')

            # (3, 2) -> (2, 1)
            e7_conv2d = tf_utils.conv2d(e6_lrelu, self.gen_c[7], name='e7_conv2d')
            e7_batchnorm = tf_utils.batch_norm(e7_conv2d, name='e7_batchnorm', _ops=self._ops)
            e7_relu = tf_utils.relu(e7_batchnorm, name='e7_relu')

            # (2, 1) -> (4, 2)
            d0_deconv = tf_utils.deconv2d(e7_relu, self.gen_c[8], name='d0_deconv2d')
            shapeA = e6_conv2d.get_shape().as_list()[1]
            shapeB = d0_deconv.get_shape().as_list()[1] - e6_conv2d.get_shape().as_list()[1]
            # (4, 2) -> (3, 2)
            d0_split, _ = tf.split(d0_deconv, [shapeA, shapeB], axis=1, name='d0_split')
            tf_utils.print_activations(d0_split)
            d0_batchnorm = tf_utils.batch_norm(d0_split, name='d0_batchnorm', _ops=self._ops)
            d0_drop = tf.nn.dropout(d0_batchnorm, keep_prob=0.5, name='d0_dropout')
            d0_concat = tf.concat([d0_drop, e6_batchnorm], axis=3, name='d0_concat')
            d0_relu = tf_utils.relu(d0_concat, name='d0_relu')

            # (3, 2) -> (6, 4)
            d1_deconv = tf_utils.deconv2d(d0_relu, self.gen_c[9], name='d1_deconv2d')
            # (6, 4) -> (5, 4)
            shapeA = e5_batchnorm.get_shape().as_list()[1]
            shapeB = d1_deconv.get_shape().as_list()[1] - e5_batchnorm.get_shape().as_list()[1]
            d1_split, _ = tf.split(d1_deconv, [shapeA, shapeB], axis=1, name='d1_split')
            tf_utils.print_activations(d1_split)
            d1_batchnorm = tf_utils.batch_norm(d1_split, name='d1_batchnorm', _ops=self._ops)
            d1_drop = tf.nn.dropout(d1_batchnorm, keep_prob=0.5, name='d1_dropout')
            d1_concat = tf.concat([d1_drop, e5_batchnorm], axis=3, name='d1_concat')
            d1_relu = tf_utils.relu(d1_concat, name='d1_relu')

            # (5, 4) -> (10, 8)
            d2_deconv = tf_utils.deconv2d(d1_relu, self.gen_c[10], name='d2_deconv2d')
            # (10, 8) -> (10, 7)
            shapeA = e4_batchnorm.get_shape().as_list()[2]
            shapeB = d2_deconv.get_shape().as_list()[2] - e4_batchnorm.get_shape().as_list()[2]
            d2_split, _ = tf.split(d2_deconv, [shapeA, shapeB], axis=2, name='d2_split')
            tf_utils.print_activations(d2_split)
            d2_batchnorm = tf_utils.batch_norm(d2_split, name='d2_batchnorm', _ops=self._ops)
            d2_drop = tf.nn.dropout(d2_batchnorm, keep_prob=0.5, name='d2_dropout')
            d2_concat = tf.concat([d2_drop, e4_batchnorm], axis=3, name='d2_concat')
            d2_relu = tf_utils.relu(d2_concat, name='d2_relu')

            # (10, 7) -> (20, 14)
            d3_deconv = tf_utils.deconv2d(d2_relu, self.gen_c[11], name='d3_deconv2d')
            # (20, 14) -> (19, 14)
            shapeA = e3_batchnorm.get_shape().as_list()[1]
            shapeB = d3_deconv.get_shape().as_list()[1] - e3_batchnorm.get_shape().as_list()[1]
            d3_split_1, _ = tf.split(d3_deconv, [shapeA, shapeB], axis=1, name='d3_split_1')
            tf_utils.print_activations(d3_split_1)
            # (19, 14) -> (19, 13)
            shapeA = e3_batchnorm.get_shape().as_list()[2]
            shapeB = d3_split_1.get_shape().as_list()[2] - e3_batchnorm.get_shape().as_list()[2]
            d3_split_2, _ = tf.split(d3_split_1, [shapeA, shapeB], axis=2, name='d3_split_2')
            tf_utils.print_activations(d3_split_2)
            d3_batchnorm = tf_utils.batch_norm(d3_split_2, name='d3_batchnorm', _ops=self._ops)
            d3_concat = tf.concat([d3_batchnorm, e3_batchnorm], axis=3, name='d3_concat')
            d3_relu = tf_utils.relu(d3_concat, name='d3_relu')

            # (19, 13) -> (38, 26)
            d4_deconv = tf_utils.deconv2d(d3_relu, self.gen_c[12], name='d4_deconv2d')
            # (38, 26) -> (38, 25)
            shapeA = e2_batchnorm.get_shape().as_list()[2]
            shapeB = d4_deconv.get_shape().as_list()[2] - e2_batchnorm.get_shape().as_list()[2]
            d4_split, _ = tf.split(d4_deconv, [shapeA, shapeB], axis=2, name='d4_split')
            tf_utils.print_activations(d4_split)
            d4_batchnorm = tf_utils.batch_norm(d4_split, name='d4_batchnorm', _ops=self._ops)
            d4_concat = tf.concat([d4_batchnorm, e2_batchnorm], axis=3, name='d4_concat')
            d4_relu = tf_utils.relu(d4_concat, name='d4_relu')

            # (38, 25) -> (76, 50)
            d5_deconv = tf_utils.deconv2d(d4_relu, self.gen_c[13], name='d5_deconv2d')
            # (76, 50) -> (75, 50)
            shapeA = e1_batchnorm.get_shape().as_list()[1]
            shapeB = d5_deconv.get_shape().as_list()[1] - e1_batchnorm.get_shape().as_list()[1]
            d5_split, _ = tf.split(d5_deconv, [shapeA, shapeB], axis=1, name='d5_split')
            tf_utils.print_activations(d5_split)
            d5_batchnorm = tf_utils.batch_norm(d5_split, name='d5_batchnorm', _ops=self._ops)
            d5_concat = tf.concat([d5_batchnorm, e1_batchnorm], axis=3, name='d5_concat')
            d5_relu = tf_utils.relu(d5_concat, name='d5_relu')

            # (75, 50) -> (150, 100)
            d6_deconv = tf_utils.deconv2d(d5_relu, self.gen_c[14], name='d6_deconv2d')
            d6_batchnorm = tf_utils.batch_norm(d6_deconv, name='d6_batchnorm', _ops=self._ops)
            d6_concat = tf.concat([d6_batchnorm, e0_conv2d], axis=3, name='d6_concat')
            d6_relu = tf_utils.relu(d6_concat, name='d6_relu')

            # (150, 100) -> (300, 200)
            d7_deconv = tf_utils.deconv2d(d6_relu, self.gen_c[15], name='d7_deconv2d')
            output = tf_utils.tanh(d7_deconv, name='output_tanh')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Discriminator(object):
    def __init__(self, name=None, dis_c=None, _ops=None):
        self.name = name
        self.dis_c = dis_c
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # 200 -> 100
            h0_conv2d = tf_utils.conv2d(x, self.dis_c[0], name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv2d, name='h0_lrelu')

            # 100 -> 50
            h1_conv2d = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
            h1_batchnorm = tf_utils.batch_norm(h1_conv2d, name='h1_batchnorm', _ops=self._ops)
            h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

            # 50 -> 25
            h2_conv2d = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
            h2_batchnorm = tf_utils.batch_norm(h2_conv2d, name='h2_batchnorm', _ops=self._ops)
            h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

            # 25 -> 13
            h3_conv2d = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv2d, name='h3_batchnorm', _ops=self._ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            # Patch GAN: 13 -> 13
            output = tf_utils.conv2d(h3_lrelu, self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1, name='output_conv2d')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
