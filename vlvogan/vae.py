# ---------------------------------------------------------
# TensorFlow SpineC2M-VLVOGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import logging
import collections
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

import tensorflow_utils as tf_utils
import utils as utils
from reader import Reader

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
class VAE(object):
    def __init__(self, sess, flags, img_size, data_path, log_path=None):
        self.sess = sess
        self.flags = flags
        self.img_size = img_size
        self.data_path = data_path
        self.log_path = log_path

        self.nef, self.ndf = 64, 64
        self.enc_c = [64, 128, 256, 512, 512, 512]
        self.dec_c = [5*4*512, 512, 512, 256, 128, 64, self.img_size[2]]
        self.start_decay_step = int(self.flags.iters / 2)
        self.decay_steps = self.flags.iters - self.start_decay_step
        self.enc_train_ops, self.dec_train_ops = [], []

        self._init_logger()     # init logger
        self._build_net()       # init graph
        self._tensorboard()       # init tensorboard
        logger.info('Initialized VAE SUCCESS!')

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        self.input_tfph = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='input_tfph')
        self.z_in_tfph = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_variable_tfph')
        self.keep_prob_tfph = tf.placeholder(tf.float32, name='keep_prob_tfph')

        # Data reader
        data_reader = Reader(self.data_path, name='data', image_size=self.img_size, batch_size=self.flags.batch_size,
                             is_train=self.flags.is_train)
        self.x_imgs, self.y_imgs, self.x_imgs_ori, self.y_imgs_ori = data_reader.feed()

        # Encoder and decoder objects
        self.encoder = Encoder(name='encoder', enc_c=self.enc_c, _ops=self.enc_train_ops)
        self.decoder = Decoder(name='decoder', dec_c=self.dec_c, _ops=self.dec_train_ops)

        # Encoding
        mu_x, sigma_x = self.encoder(self.x_imgs, keep_prob=self.keep_prob_tfph)
        mu_y, sigma_y = self.encoder(self.y_imgs, keep_prob=self.keep_prob_tfph)

        # Sampling by re-parameterization technique
        self.z_x = mu_x + sigma_x * tf.random_normal(tf.shape(mu_x), mean=0., stddev=1., dtype=tf.float32)
        self.z_y = mu_y + sigma_y * tf.random_normal(tf.shape(mu_y), mean=0., stddev=1., dtype=tf.float32)

        # Decoding
        x_imgs_recon = self.decoder(self.z_x, keep_prob=self.keep_prob_tfph)
        self.x_imgs_recon = tf.clip_by_value(x_imgs_recon, 1e-8, 1 - 1e-8)
        y_imgs_recon = self.decoder(self.z_y, keep_prob=self.keep_prob_tfph)
        self.y_imgs_recon = tf.clip_by_value(y_imgs_recon, 1e-8, 1 - 1e-8)

        # Loss
        marginal_likelihood_x = self.marginal_likelihood_loss(self.x_imgs, self.x_imgs_recon)
        marginal_likelihood_y = self.marginal_likelihood_loss(self.y_imgs, self.y_imgs_recon)
        KL_divergence_x = self.kl_divergence_loss(mu_x, sigma_x)
        KL_divergence_y = self.kl_divergence_loss(mu_y, sigma_y)

        self.marginal_likelihood = tf.reduce_mean(marginal_likelihood_x) + tf.reduce_mean(marginal_likelihood_y)
        self.KL_divergence = tf.reduce_mean(KL_divergence_x) + tf.reduce_mean(KL_divergence_y)
        self.ELBO = self.marginal_likelihood - self.KL_divergence
        self.LVC = self.latent_vector_constrain(self.z_x, self.z_y)
        self.loss = - self.ELBO + self.LVC

        self.train_op = tf.train.AdamOptimizer(self.flags.learning_rate).minimize(self.loss)

        # Sampling
        sample_img = self.decoder(self.z_in_tfph, keep_prob=self.keep_prob_tfph)
        self.sample_img = tf.clip_by_value(sample_img, 1e-8, 1 - 1e-8)

        mu, sigma = self.encoder(self.input_tfph, keep_prob=self.keep_prob_tfph)
        # Sampling by re-parameterization technique
        self.sample_z = mu + sigma * tf.random_normal(tf.shape(mu), mean=0., stddev=1., dtype=tf.float32)

    def _tensorboard(self):
        tf.summary.scalar('loss/marginal_likelihood', -self.marginal_likelihood)
        tf.summary.scalar('loss/KL_divergence', self.KL_divergence)
        tf.summary.scalar('loss/latent_vector_constrain', self.LVC)
        tf.summary.scalar('loss/total_loss', self.loss)

        self.summary_op = tf.summary.merge_all()

    @staticmethod
    def marginal_likelihood_loss(imgs, imgs_recon):
        loss = tf.reduce_sum(imgs * tf.log(imgs_recon) + (1 - imgs) * tf.log(1 - imgs_recon), axis=[1, 2, 3])
        return loss

    @staticmethod
    def kl_divergence_loss(mu, sigma):
        loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
        return loss

    def latent_vector_constrain(self, vector_x, vector_y):
        loss = self.flags.beta * tf.maximum(tf.reduce_mean(tf.square(vector_x - vector_y)) - self.flags.k, 0.)
        return loss

    def train_step(self):
        train_ops = [self.train_op, self.loss, self.KL_divergence, self.marginal_likelihood, self.LVC, self.summary_op]
        train_feed = {self.keep_prob_tfph: 0.9}
        _, loss, KL_divergence, marginal_likelihood, latent_vector_constrain, summary = \
            self.sess.run(train_ops, feed_dict=train_feed)

        return [loss, KL_divergence, -marginal_likelihood, latent_vector_constrain], summary

    def test_step(self):
        test_ops = [self.x_imgs, self.x_imgs_recon, self.y_imgs, self.y_imgs_recon, self.z_x, self.z_y]
        test_feed = {self.keep_prob_tfph: 1.0}
        x_imgs, x_imgs_recon, y_imgs, y_imgs_recon, z_x, z_y = self.sess.run(test_ops, feed_dict=test_feed)

        return [x_imgs, x_imgs_recon, y_imgs, y_imgs_recon], [z_x, z_y]

    def sample_imgs(self):
        test_ops = [self.x_imgs, self.x_imgs_recon, self.y_imgs, self.y_imgs_recon]
        test_feed = {self.keep_prob_tfph: 1.0}
        x_imgs, x_imgs_recon, y_imgs, y_imgs_recon = self.sess.run(test_ops, feed_dict=test_feed)

        x_imgs = x_imgs[:self.flags.sample_batch]
        x_imgs_recon = x_imgs_recon[:self.flags.sample_batch]
        y_imgs = y_imgs[:self.flags.sample_batch]
        y_imgs_recon = y_imgs_recon[:self.flags.sample_batch]

        return [x_imgs, x_imgs_recon, y_imgs, y_imgs_recon]

    def decoder_y(self, z):
        y_imgs = self.sess.run(self.sample_img, feed_dict={self.z_in_tfph: z, self.keep_prob_tfph: 1.0})
        return y_imgs

    def decoder_z(self, imgs):
        z_vectors = self.sess.run(self.sample_z, feed_dict={self.input_tfph: imgs, self.keep_prob_tfph: 1.0})
        return z_vectors

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('total_loss', loss[0]), ('KL divergence', loss[1]),
                                                  ('marginal likelihood', loss[2]), ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    @staticmethod
    def plots(imgs, iter_time, img_size, save_file):
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
                plt.imshow((imgs[col_index][row_index]).reshape(img_size[0], img_size[1]), cmap='Greys_r')

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time).zfill(6)), bbox_inches='tight')
        plt.close(fig)


class Encoder(object):
    def __init__(self, name=None, enc_c=None, z_dim=128, _ops=None):
        self.name = name
        self.enc_c = enc_c
        self.z_dim = z_dim
        self._ops = _ops
        self.reuse =False

    def __call__(self, x, keep_prob=0.9):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (300, 200) -> (150, 100)
            e0_conv2d = tf_utils.conv2d(x, self.enc_c[0], name='e0_conv2d')
            e0_elu = tf_utils.elu(e0_conv2d, name='e0_elu')
            e0_drop = tf.nn.dropout(e0_elu, keep_prob=keep_prob, name='e0_drop')
            tf_utils.print_activations(e0_drop)

            # (150, 100) -> (75, 50)
            e1_conv2d = tf_utils.conv2d(e0_drop, self.enc_c[1], name='e1_conv2d')
            e1_batchnorm = tf_utils.batch_norm(e1_conv2d, name='e1_batchnorm', _ops=self._ops)
            e1_elu = tf_utils.elu(e1_batchnorm, name='e1_elu')
            e1_drop = tf.nn.dropout(e1_elu, keep_prob=keep_prob, name='e1_drop')
            tf_utils.print_activations(e1_drop)

            # (75, 50) -> (38, 25)
            e2_conv2d = tf_utils.conv2d(e1_drop, self.enc_c[2], name='e2_conv2d')
            e2_batchnorm = tf_utils.batch_norm(e2_conv2d, name='e2_batchnorm', _ops=self._ops)
            e2_elu = tf_utils.elu(e2_batchnorm, name='e2_elu')
            e2_drop = tf.nn.dropout(e2_elu, keep_prob=keep_prob, name='e2_drop')
            tf_utils.print_activations(e2_drop)

            # (38, 25) -> (19, 13)
            e3_conv2d = tf_utils.conv2d(e2_drop, self.enc_c[3], name='e3_conv2d')
            e3_batchnorm = tf_utils.batch_norm(e3_conv2d, name='e3_batchnorm', _ops=self._ops)
            e3_elu = tf_utils.lrelu(e3_batchnorm, name='e3_elu')
            e3_drop = tf.nn.dropout(e3_elu, keep_prob=keep_prob, name='e3_drop')
            tf_utils.print_activations(e3_drop)

            # (19, 13) -> (10, 7)
            e4_conv2d = tf_utils.conv2d(e3_drop, self.enc_c[4], name='e4_conv2d')
            e4_batchnorm = tf_utils.batch_norm(e4_conv2d, name='e4_batchnorm', _ops=self._ops)
            e4_lrelu = tf_utils.elu(e4_batchnorm, name='e4_elu')
            e4_drop = tf.nn.dropout(e4_lrelu, keep_prob=keep_prob, name='e4_drop')
            tf_utils.print_activations(e4_drop)

            # (10, 7) -> (5, 4)
            e5_conv2d = tf_utils.conv2d(e4_drop, self.enc_c[5], name='e5_conv2d')
            e5_batchnorm = tf_utils.batch_norm(e5_conv2d, name='e5_batchnorm', _ops=self._ops)
            e5_tanh = tf_utils.tanh(e5_batchnorm, name='e5_tanh')
            e5_drop = tf.nn.dropout(e5_tanh, keep_prob=keep_prob, name='e5_drop')
            tf_utils.print_activations(e5_drop)

            e5_flatten = tf.contrib.layers.flatten(e5_drop)
            e6_linear = tf_utils.linear(e5_flatten, 2*self.z_dim, name='e6_linear')
            tf_utils.print_activations(e6_linear)

            # The mean parameter is unconstrained
            mean = e6_linear[:, :self.z_dim]
            # The standard deviation must be positive.
            # Parameterize with a sofplus and add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(e6_linear[:, self.z_dim:])

            tf_utils.print_activations(mean)
            tf_utils.print_activations(stddev)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return mean, stddev

class Decoder(object):
    def __init__(self, name=None, dec_c=None, _ops=None):
        self.name = name
        self.dec_c = dec_c
        self._ops = _ops
        self.reuse = False

    def __call__(self, x, keep_prob=0.9):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (128) -> (5, 4)
            d0_linear = tf_utils.linear(x, self.dec_c[0], name='d0_linear')
            d0_tanh = tf_utils.tanh(d0_linear, name='d0_tanh')
            d0_drop = tf.nn.dropout(d0_tanh, keep_prob=keep_prob, name='d0_drop')
            d0_reshape = tf.reshape(d0_drop, shape=[5, 4, -1], name='d0_reshape')

            # (5, 4) -> (10, 8)
            d1_deconv = tf_utils.deconv2d(d0_reshape, self.dec_c[1], name='d1_deconv2d')
            # (10, 8) -> (10, 7)
            d1_split, _ = tf.split(d1_deconv, [-1, 1], axis=2, name='d1_split')
            tf_utils.print_activations(d1_split)
            d1_batchnorm = tf_utils.batch_norm(d1_split, name='d1_batchnorm', _ops=self._ops)
            d1_elu = tf_utils.elu(d1_batchnorm, name='d1_elu')
            d1_drop = tf.nn.dropout(d1_elu, keep_prob=keep_prob, name='d1_dropout')

            # (10, 7) -> (20, 14)
            d2_deconv = tf_utils.deconv2d(d1_drop, self.dec_c[2], name='d2_deconv2d')
            # (20, 14) -> (19, 14)
            d2_split_1, _ = tf.split(d2_deconv, [-1, 1], axis=1, name='d2_split_1')
            tf_utils.print_activations(d2_split_1)
            # (19, 14) -> (19, 13)
            d2_split_2, _ = tf.split(d2_split_1, [-1, 1], axis=2, name='d2_split_2')
            tf_utils.print_activations(d2_split_2)
            d2_batchnorm = tf_utils.batch_norm(d2_split_2, name='d2_batchnorm', _ops=self._ops)
            d2_elu = tf_utils.relu(d2_batchnorm, name='d2_elu')
            d2_drop = tf.nn.dropout(d2_elu, keep_prob=keep_prob, name='d2_dropout')

            # (19, 13) -> (38, 26)
            d3_deconv = tf_utils.deconv2d(d2_drop, self.dec_c[3], name='d3_deconv2d')
            # (38, 26) -> (38, 25)
            d3_split, _ = tf.split(d3_deconv, [-1, 1], axis=2, name='d3_split')
            tf_utils.print_activations(d3_split)
            d3_batchnorm = tf_utils.batch_norm(d3_split, name='d3_batchnorm', _ops=self._ops)
            d3_elu = tf_utils.elu(d3_batchnorm, name='d3_elu')
            d3_drop = tf.nn.dropout(d3_elu, keep_prob=keep_prob, name='d3_dropout')

            # (38, 25) -> (76, 50)
            d4_deconv = tf_utils.deconv2d(d3_drop, self.dec_c[4], name='d4_deconv2d')
            # (76, 50) -> (75, 50)
            d4_split, _ = tf.split(d4_deconv, [-1, 1], axis=1, name='d4_split')
            tf_utils.print_activations(d4_split)
            d4_batchnorm = tf_utils.batch_norm(d4_split, name='d4_batchnorm', _ops=self._ops)
            d4_elu = tf_utils.elu(d4_batchnorm, name='d4_elu')
            d4_drop = tf.nn.dropout(d4_elu, keep_prob=keep_prob, name='d4_dropout')

            # (75, 50) -> (150, 100)
            d5_deconv = tf_utils.deconv2d(d4_drop, self.dec_c[5], name='d5_deconv2d')
            d5_batchnorm = tf_utils.batch_norm(d5_deconv, name='d5_batchnorm', _ops=self._ops)
            d5_elu = tf_utils.elu(d5_batchnorm, name='d5_elu')
            d5_drop = tf.nn.dropout(d5_elu, keep_prob=keep_prob, name='d5_dropout')

            # (150, 100) -> (300, 200)
            d6_deconv = tf_utils.deconv2d(d5_drop, self.dec_c[6], name='d6_deconv2d')
            output = tf_utils.tanh(d6_deconv, name='output_tanh')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output



