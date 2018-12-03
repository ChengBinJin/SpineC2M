# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from vanhuyz
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

# noinspection PyPep8Naming
import tensorflow_utils as tf_utils
import utils as utils
from reader import Reader

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
class cycleGAN(object):
    def __init__(self, sess, flags, image_size, data_path, log_path=None):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.data_path = data_path
        self.log_path = log_path

        # True: use lsgan (mean squared error)
        # False: use cross entropy loss
        self.use_lsgan = True
        self.use_sigmoid = not self.use_lsgan
        # [instance|batch] use instance norm or batch norm, default: instance
        self.norm = 'instane'
        self.lambda1, self.lambda2 = 10.0, 10.0
        self.ngf, self.ndf = 64, 64
        self.real_label = 0.9
        self.start_decay_step = int(self.flags.iters / 2)
        self.decay_steps = self.flags.iters - self.start_decay_step
        self.eps = 1e-12

        self._G_gen_train_ops, self._F_gen_train_ops = [], []
        self._Dy_dis_train_ops, self._Dx_dis_train_ops = [], []

        self._init_logger()     # init logger
        self._build_net()       # init graph
        self._tensorboard()     # init tensorboard

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        # tfph: TensorFlow PlaceHolder
        self.x_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='x_test_tfph')
        self.y_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='y_test_tfph')
        self.fake_x_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='fake_x_tfph')
        self.fake_y_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='fake_y_tfph')

        self.G_gen = Generator(name='G', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._G_gen_train_ops)
        self.Dy_dis = Discriminator(name='Dy', ndf=self.ndf, norm=self.norm, _ops=self._Dy_dis_train_ops,
                                    use_sigmoid=self.use_sigmoid)
        self.F_gen = Generator(name='F', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._F_gen_train_ops)
        self.Dx_dis = Discriminator(name='Dx', ndf=self.ndf, norm=self.norm, _ops=self._Dx_dis_train_ops,
                                    use_sigmoid=self.use_sigmoid)

        data_reader = Reader(self.data_path, name='data', image_size=self.image_size, batch_size=self.flags.batch_size)
        img_pairs = data_reader.feed()
        self.x_imgs, self.y_imgs = tf.split(img_pairs, [self.image_size[1], self.image_size[1]], axis=2)

        self.fake_x_pool_obj = utils.ImagePool(pool_size=50)
        self.fake_y_pool_obj = utils.ImagePool(pool_size=50)

        # cycle consistency loss
        cycle_loss = self.cycle_consistency_loss(self.x_imgs, self.y_imgs)

        # X -> Y
        self.fake_y_imgs = self.G_gen(self.x_imgs)
        self.G_gen_loss = self.generator_loss(self.Dy_dis, self.fake_y_imgs, use_lsgan=self.use_lsgan)
        self.G_loss = self.G_gen_loss + cycle_loss
        self.Dy_dis_loss = self.discriminator_loss(self.Dy_dis, self.y_imgs, self.fake_y_tfph,
                                                   use_lsgan=self.use_lsgan)

        # Y -> X
        self.fake_x_imgs = self.F_gen(self.y_imgs)
        self.F_gen_loss = self.generator_loss(self.Dx_dis, self.fake_x_imgs, use_lsgan=self.use_lsgan)
        self.F_loss = self.F_gen_loss + cycle_loss
        self.Dx_dis_loss = self.discriminator_loss(self.Dx_dis, self.x_imgs, self.fake_x_tfph,
                                                   use_lsgan=self.use_lsgan)

        G_optim = self.optimizer(loss=self.G_loss, variables=self.G_gen.variables, name='Adam_G')
        Dy_optim = self.optimizer(loss=self.Dy_dis_loss, variables=self.Dy_dis.variables, name='Adam_Dy')
        F_optim = self.optimizer(loss=self.F_loss, variables=self.F_gen.variables, name='Adam_F')
        Dx_optim = self.optimizer(loss=self.Dx_dis_loss, variables=self.Dx_dis.variables, name='Adam_Dx')
        self.optims = tf.group([G_optim, Dy_optim, F_optim, Dx_optim])
        # with tf.control_dependencies([G_optim, Dy_optim, F_optim, Dx_optim]):
        #     self.optims = tf.no_op(name='optimizers')

        # for sampling function
        self.fake_y_sample = self.G_gen(self.x_test_tfph)
        self.fake_x_sample = self.F_gen(self.y_test_tfph)

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

    def cycle_consistency_loss(self, x_imgs, y_imgs):
        forward_loss = tf.reduce_mean(tf.abs(self.F_gen(self.G_gen(x_imgs)) - x_imgs))
        backward_loss = tf.reduce_mean(tf.abs(self.G_gen(self.F_gen(y_imgs)) - y_imgs))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def generator_loss(self, dis_obj, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            loss = 0.5 * tf.reduce_mean(tf.squared_difference(dis_obj(fake_img), self.real_label))
        else:
            # heuristic, non-saturating loss (I don't understand here!)
            # loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps)) / 2.  (???)
            loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps))
        return loss

    def discriminator_loss(self, dis_obj, real_img, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(dis_obj(real_img), self.real_label))
            error_fake = tf.reduce_mean(tf.square(dis_obj(fake_img)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(tf.log(dis_obj(real_img) + self.eps))
            error_fake = -tf.reduce_mean(tf.log(1. - dis_obj(fake_img) + self.eps))

        loss = 0.5 * (error_real + error_fake)
        return loss

    def _tensorboard(self):
        tf.summary.scalar('loss/G_gen', self.G_gen_loss)
        tf.summary.scalar('loss/Dy_dis', self.Dy_dis_loss)
        tf.summary.scalar('loss/F_gen', self.F_gen_loss)
        tf.summary.scalar('loss/Dx_dis', self.Dx_dis_loss)
        self.summary_op = tf.summary.merge_all()

    def train_step(self):
        fake_y_val, fake_x_val = self.sess.run([self.fake_y_imgs, self.fake_x_imgs])
        ops = [self.optims, self.G_loss, self.Dy_dis_loss, self.F_loss, self.Dx_dis_loss, self.summary_op]
        feed_dict = {self.fake_x_tfph: self.fake_x_pool_obj.query(fake_x_val),
                     self.fake_y_tfph: self.fake_y_pool_obj.query(fake_y_val)}

        _, G_loss, Dy_loss, F_loss, Dx_loss, summary = self.sess.run(ops, feed_dict=feed_dict)

        return [G_loss, Dy_loss, F_loss, Dx_loss], summary

    def sample_imgs(self):
        x_val, y_val = self.sess.run([self.x_imgs, self.y_imgs])
        fake_y, fake_x = self.sess.run([self.fake_y_sample, self.fake_x_sample],
                                       feed_dict={self.x_test_tfph: x_val, self.y_test_tfph: y_val})

        return [x_val, fake_y, y_val, fake_x]

    # def test_step(self, img, mode='XtoY'):
    #     if mode == 'XtoY':
    #         fake_y = self.sess.run(self.fake_y_sample, feed_dict={self.x_test_tfph: img})
    #         return [img, fake_y]
    #     elif mode == 'YtoX':
    #         fake_x = self.sess.run(self.fake_x_sample, feed_dict={self.y_test_tfph: img})
    #         return [img, fake_x]
    #     else:
    #         raise NotImplementedError

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('G_loss', loss[0]), ('Dy_loss', loss[1]),
                                                  ('F_loss', loss[2]), ('Dx_loss', loss[3]),
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


class Generator(object):
    def __init__(self, name=None, ngf=64, norm='instance', image_size=(128, 256, 3), _ops=None):
        self.name = name
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H, W, 64)
            conv1 = tf_utils.padding2d(x, p_h=3, p_w=3, pad_type='REFLECT', name='conv1_padding')
            conv1 = tf_utils.conv2d(conv1, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID',
                                    name='conv1_conv')
            conv1 = tf_utils.norm(conv1, _type='instance', _ops=self._ops, name='conv1_norm')
            conv1 = tf_utils.relu(conv1, name='conv1_relu', is_print=True)

            # (N, H, W, 64)  -> (N, H/2, W/2, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm',)
            conv2 = tf_utils.relu(conv2, name='conv2_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H/4, W/4, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm',)
            conv3 = tf_utils.relu(conv3, name='conv3_relu', is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/4, W/4, 256)
            if (self.image_size[0] <= 128) and (self.image_size[1] <= 128):
                # use 6 residual blocks for 128x128 images
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=6, is_print=True)
            else:
                # use 9 blocks for higher resolution
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=9, is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/2, W/2, 128)
            conv4 = tf_utils.deconv2d(res_out, 2*self.ngf, name='conv4_deconv2d')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.relu(conv4, name='conv4_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H, W, 64)
            conv5 = tf_utils.deconv2d(conv4, self.ngf, name='conv5_deconv2d')
            conv5 = tf_utils.norm(conv5, _type='instance', _ops=self._ops, name='conv5_norm')
            conv5 = tf_utils.relu(conv5, name='conv5_relu', is_print=True)

            # (N, H, W, 64) -> (N, H, W, 3)
            conv6 = tf_utils.padding2d(conv5, p_h=3, p_w=3, pad_type='REFLECT', name='output_padding')
            conv6 = tf_utils.conv2d(conv6, self.image_size[2], k_h=7, k_w=7, d_h=1, d_w=1,
                                    padding='VALID', name='output_conv')
            output = tf_utils.tanh(conv6, name='output_tanh', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Discriminator(object):
    def __init__(self, name='', ndf=64, norm='instance', _ops=None, use_sigmoid=False):
        self.name = name
        self.ndf = ndf
        self.norm = norm
        self._ops = _ops
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            if self.use_sigmoid:
                output = tf_utils.sigmoid(conv5, name='output_sigmoid', is_print=True)
            else:
                output = tf.identity(conv5, name='output_without_sigmoid')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
