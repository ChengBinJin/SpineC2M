# ---------------------------------------------------------
# Tensorflow SpineC2M-mrgan+ Implementation
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
from vgg16 import VGG16
from reader import Reader

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


# noinspection PyPep8Naming
class DC2Anet(object):
    def __init__(self, sess, flags, img_size, data_path, log_path=None):
        self.sess = sess
        self.flags = flags
        self.img_size = img_size
        self.data_path = data_path
        self.log_path = log_path

        # cycle-consistent loss term
        self.is_cycle_consistent = self.flags.is_cycle_consistent
        self.lambda1, self.lambda2 = self.flags.cycle_consistent_weight, self.flags.cycle_consistent_weight
        # voxel-wise loss term
        self.is_voxel = self.flags.is_voxel
        self.L1_lambda = self.flags.L1_lambda
        # Gradient difference loss (GDL) term
        self.is_gdl = self.flags.is_gdl  # True: use gradient difference loss
        self.gdl_weight = self.flags.gdl_weight
        # Percepual loss term
        self.is_perceptual = self.flags.is_perceptual
        self.perceptual_weight = self.flags.perceptual_weight
        # SSIM loss term
        self.is_ssim = self.flags.is_ssim
        self.ssim_weight = self.flags.ssim_weight

        # [instance|batch] use instance norm or batch norm, default: instance
        self.norm = 'instane'
        self.ngf, self.ndf = 64, 64
        self.real_label = 0.9
        self.start_decay_step = int(self.flags.iters / 2)
        self.decay_steps = self.flags.iters - self.start_decay_step
        # self.eps = 1e-12

        self._G_gen_train_ops, self._F_gen_train_ops = [], []
        self._Dy_dis_train_ops, self._Dx_dis_train_ops = [], []

        self._init_logger()     # init logger
        self._build_net()       # init graph
        self._tensorboard()     # init tensorboard

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils.init_logger(self.log_path)

    def _build_net(self):
        # tfph: TensorFlow PlaceHolder
        self.x_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='x_test_tfph')
        self.y_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='y_test_tfph')

        # Supervised learning placeholders for Image Pool Tech.
        self.xy_fake_pairs_tfph = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 2],
                                                 name='xy_fake_pairs_tfph')
        self.yx_fake_pairs_tfph = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 2],
                                                 name='yx_fake_pairs_tfph')

        # Unsupervised learning placeholders for Image Pool Tech.
        self.xy_fake_unpairs_tfph = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 1],
                                                   name='xy_fake_unpairs_tfph')
        self.yx_fake_unpairs_tfph = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 1],
                                                   name='yx_fake_unpairs_tfph')

        self.G_gen = Generator(name='G', ngf=self.ngf, norm=self.norm, image_size=self.img_size,
                               _ops=self._G_gen_train_ops)
        # Implementing: G model
        self.Dy_dis_sup = Discriminator(
            name='Dy_sup', ndf=self.ndf, norm=self.norm, model=self.flags.dis_model, shared_reuse=False,
            _ops=self._Dy_dis_train_ops)
        self.Dy_dis_unsup = Discriminator(
            name='Dy_unsup', ndf=self.ndf, norm=self.norm, model=self.flags.dis_model, shared_reuse=True,
            _ops=self._Dy_dis_train_ops)
        self.F_gen = Generator(
            name='F', ngf=self.ngf, norm=self.norm, image_size=self.img_size, _ops=self._F_gen_train_ops)
        self.Dx_dis_sup = Discriminator(
            name='Dx_sup', ndf=self.ndf, norm=self.norm, model=self.flags.dis_model, shared_reuse=False,
            _ops=self._Dx_dis_train_ops)
        self.Dx_dis_unsup = Discriminator(
            name='Dx_unsup', ndf=self.ndf, norm=self.norm, model=self.flags.dis_model, shared_reuse=True,
            _ops=self._Dx_dis_train_ops)
        self.vggModel = VGG16(name='VGG16_Pretrained')

        data_reader = Reader(self.data_path, name='data', image_size=self.img_size, batch_size=self.flags.batch_size,
                             is_train=self.flags.is_train)
        # self.x_imgs_ori and self.y_imgs_ori are the images before data augmentation
        self.x_imgs, self.y_imgs, self.x_imgs_ori, self.y_imgs_ori, self.img_name = data_reader.feed()

        self.fake_xy_pool_obj_sup = utils.ImagePool(pool_size=50)
        self.fake_yx_pool_obj_sup = utils.ImagePool(pool_size=50)
        self.fake_xy_pool_obj_unsup = utils.ImagePool(pool_size=50)
        self.fake_yx_pool_obj_unsup = utils.ImagePool(pool_size=50)

        # cycle consistency loss
        self.cycle_loss = self.cycle_consistency_loss(self.x_imgs, self.y_imgs)

        # concatenation
        self.fake_y_imgs = self.G_gen(self.x_imgs)
        self.xy_real_pairs = tf.concat([self.x_imgs, self.y_imgs], axis=3)
        self.xy_fake_pairs = tf.concat([self.x_imgs, self.fake_y_imgs], axis=3)

        self.fake_x_imgs = self.F_gen(self.y_imgs)
        self.yx_real_pairs = tf.concat([self.y_imgs, self.x_imgs], axis=3)
        self.yx_fake_pairs = tf.concat([self.y_imgs, self.fake_x_imgs], axis=3)

        # X -> Y
        # Supervised learning
        self.G_gen_loss_sup = self.generator_loss(self.Dy_dis_sup, self.xy_fake_pairs)
        self.G_cond_loss = self.voxel_loss(preds=self.fake_y_imgs, gts=self.y_imgs)
        self.G_gdl_loss = self.gradient_difference_loss(preds=self.fake_y_imgs, gts=self.y_imgs)
        self.G_perceptual_loss = self.perceptual_loss_fn(preds=self.fake_y_imgs, gts=self.y_imgs)
        self.G_ssim_loss = self.ssim_loss_fn(preds=self.fake_y_imgs, gts=self.y_imgs)
        self.G_loss_sup = self.G_gen_loss_sup + self.G_cond_loss + self.cycle_loss + self.G_gdl_loss + \
                      self.G_perceptual_loss + self.G_ssim_loss
        self.Dy_dis_loss_sup = self.discriminator_loss(self.Dy_dis_sup, self.xy_real_pairs, self.xy_fake_pairs_tfph)

        # Unsupervised learning
        self.G_gen_loss_unsup = self.generator_loss(self.Dy_dis_unsup, self.fake_y_imgs)
        self.G_loss_unsup = self.G_gen_loss_unsup + self.cycle_loss
        self.Dy_dis_loss_unsup = self.discriminator_loss(self.Dy_dis_unsup, self.y_imgs, self.xy_fake_unpairs_tfph)

        # Integrated optimization
        self.G_gen_loss_integrated = self.G_loss_sup + self.G_loss_unsup
        self.Dy_dis_loss_integrated = self.Dy_dis_loss_sup + self.Dy_dis_loss_unsup

        # Y -> X
        # Supervised learning
        self.F_gen_loss_sup = self.generator_loss(self.Dx_dis_sup, self.yx_fake_pairs)
        self.F_cond_loss = self.voxel_loss(preds=self.fake_x_imgs, gts=self.x_imgs)
        self.F_gdl_loss = self.gradient_difference_loss(preds=self.fake_x_imgs, gts=self.x_imgs)
        self.F_perceputal_loss = self.perceptual_loss_fn(preds=self.fake_x_imgs, gts=self.x_imgs)
        self.F_ssim_loss = self.ssim_loss_fn(preds=self.fake_x_imgs, gts=self.x_imgs)
        self.F_loss_sup = self.F_gen_loss_sup + self.F_cond_loss + self.cycle_loss + self.F_gdl_loss + \
                      self.F_perceputal_loss + self.F_ssim_loss
        self.Dx_dis_loss_sup = self.discriminator_loss(self.Dx_dis_sup, self.yx_real_pairs, self.yx_fake_pairs_tfph)

        # Unsupervised Learning
        self.F_gen_loss_unsup = self.generator_loss(self.Dx_dis_unsup, self.fake_x_imgs)
        self.F_loss_unsup = self.F_gen_loss_unsup + self.cycle_loss
        self.Dx_dis_loss_unsup = self.discriminator_loss(self.Dx_dis_unsup, self.x_imgs, self.yx_fake_unpairs_tfph)

        # Integrated optimization
        self.F_gen_loss_integrated = self.F_loss_sup + self.F_loss_unsup
        self.Dx_dis_loss_integrated = self.Dx_dis_loss_sup + self.Dx_dis_loss_unsup

        # Supervised learning
        G_optim_sup = self.optimizer(
            loss=self.G_loss_sup, variables=self.G_gen.variables, name='Adam_G_sup')
        Dy_optim_sup = self.optimizer(
            loss=self.Dy_dis_loss_sup, variables=self.Dy_dis_sup.variables, name='Adam_Dy_sup')
        F_optim_sup = self.optimizer(
            loss=self.F_loss_sup, variables=self.F_gen.variables, name='Adam_F_sup')
        Dx_optim_sup = self.optimizer(
            loss=self.Dx_dis_loss_sup, variables=self.Dx_dis_sup.variables, name='Adam_Dx_sup')
        self.optims_sup = tf.group([G_optim_sup, Dy_optim_sup, F_optim_sup, Dx_optim_sup])

        # Unsupervised learning
        G_optim_unsup = self.optimizer(
            loss=self.G_loss_unsup, variables=self.G_gen.variables, name='Adam_G_unsup')
        Dy_optim_unsup = self.optimizer(
            loss=self.Dy_dis_loss_unsup, variables=self.Dy_dis_unsup.variables, name='Adam_Dy_unsup')
        F_optim_unsup = self.optimizer(
            loss=self.F_loss_unsup, variables=self.F_gen.variables, name='Adam_F_unsup')
        Dx_optim_unsup = self.optimizer(
            loss=self.Dx_dis_loss_unsup, variables=self.Dx_dis_unsup.variables, name='Adam_Dx_unsup')
        self.optims_unsup = tf.group([G_optim_unsup, Dy_optim_unsup, F_optim_unsup, Dx_optim_unsup])

        # Integrated optimization
        G_optim_integrated = self.optimizer(
            loss=self.G_gen_loss_integrated, variables=self.G_gen.variables, name='Adam_G_integrated')
        Dy_optim_integrated = self.optimizer(
            loss=self.Dy_dis_loss_integrated, variables=[self.Dy_dis_sup.variables, self.Dy_dis_unsup.variables],
            name='Adam_Dy_integrated')
        F_optim_integrated = self.optimizer(
            loss=self.F_gen_loss_integrated, variables=self.F_gen.variables, name='Adam_F_integrated')
        Dx_optim_integrated = self.optimizer(
            loss=self.Dx_dis_loss_integrated, variables=[self.Dx_dis_sup.variables, self.Dx_dis_unsup.variables],
            name='Adam_Dx_integrated')
        self.optims_integrated = tf.group(
            [G_optim_integrated, Dy_optim_integrated, F_optim_integrated, Dx_optim_integrated])


        # for sampling function
        self.fake_y_sample = self.G_gen(self.x_test_tfph)
        self.fake_x_sample = self.F_gen(self.y_test_tfph)

        self.print_network_vars(is_print=True)

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
        loss = tf.constant(0., dtype=tf.float32)
        if self.is_cycle_consistent:
            forward_loss = tf.reduce_mean(tf.abs(self.F_gen(self.G_gen(x_imgs)) - x_imgs))
            backward_loss = tf.reduce_mean(tf.abs(self.G_gen(self.F_gen(y_imgs)) - y_imgs))
            loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def voxel_loss(self, preds, gts):
        loss = tf.constant(0., dtype=tf.float32)
        if self.is_voxel:
            cond_loss = tf.reduce_mean(tf.abs(preds - gts))
            loss = self.L1_lambda * cond_loss

        return loss

    def gradient_difference_loss(self, preds, gts):
        gdl_loss = tf.constant(0., dtype=tf.float32)
        if self.is_gdl:
            preds_dy, preds_dx = tf.image.image_gradients(preds)
            gts_dy, gts_dx = tf.image.image_gradients(gts)

            gdl_loss = self.gdl_weight * tf.reduce_mean(tf.abs(tf.abs(preds_dy) - tf.abs(gts_dy)) +
                                                        tf.abs(tf.abs(preds_dx) - tf.abs(gts_dx)))

        return gdl_loss

    def perceptual_loss_fn(self, preds, gts):
        perceptual_loss = tf.constant(0., dtype=tf.float32)
        if self.is_perceptual:
            preds_features = self.vggModel(preds, mode=self.flags.perceptual_mode)
            gts_features = self.vggModel(gts, mode=self.flags.perceptual_mode)

            # There are several feature layers
            for preds_feature, gts_feature in zip(preds_features, gts_features):
                # print('perceputal mode: {}'.format(self.flags.perceptual_mode))
                perceptual_loss += tf.reduce_mean(tf.abs(preds_feature - gts_feature))

            perceptual_loss = self.perceptual_weight * perceptual_loss / len(preds_features)

        return perceptual_loss

    def ssim_loss_fn(self, preds, gts):
        ssim_loss = tf.constant(0., dtype=tf.float32)
        if self.is_ssim:
            # inputs of the ssim should be non-negative
            preds = (preds + 1.) / 2.
            gts = (gts + 1.) / 2.
            ssim_positive = tf.math.maximum(0., tf.image.ssim(preds, gts, max_val=1.0))
            ssim_loss = -tf.log(ssim_positive)
            ssim_loss = self.ssim_weight * tf.reduce_mean(ssim_loss)

        return ssim_loss


    @staticmethod
    def generator_loss(dis_obj, fake_img):
        d_logit_fake = dis_obj(fake_img)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

        return loss

    @staticmethod
    def discriminator_loss(dis_obj, real_img, fake_img):
        # use cross entropy
        d_logit_real = dis_obj(real_img)
        d_logit_fake = dis_obj(fake_img)
        error_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        error_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))

        loss = 0.5 * (error_real + error_fake)
        return loss

    def _tensorboard(self):
        tf.summary.scalar('loss/G_loss', self.G_loss_sup)
        tf.summary.scalar('loss/G_gen', self.G_gen_loss_sup)
        tf.summary.scalar('loss/Dy_dis', self.Dy_dis_loss_sup)
        tf.summary.scalar('loss/F_loss', self.F_loss_sup)
        tf.summary.scalar('loss/F_gen', self.F_gen_loss_sup)
        tf.summary.scalar('loss/Dx_dis', self.Dx_dis_loss_sup)

        if self.is_cycle_consistent:
            tf.summary.scalar('loss/cycle', self.cycle_loss)
        if self.is_voxel:
            tf.summary.scalar('loss/G_cond', self.G_cond_loss)
            tf.summary.scalar('loss/F_cond', self.F_cond_loss)
        if self.is_gdl:
            tf.summary.scalar('loss/G_gdl', self.G_gdl_loss)
            tf.summary.scalar('loss/F_gdl', self.F_gdl_loss)
        if self.is_perceptual:
            tf.summary.scalar('loss/G_perceptual', self.G_perceptual_loss)
            tf.summary.scalar('loss/F_perceptual', self.F_perceputal_loss)
        if self.is_ssim:
            tf.summary.scalar('loss/G_ssim', self.G_ssim_loss)
            tf.summary.scalar('loss/F_ssim', self.F_ssim_loss)

        self.summary_op = tf.summary.merge_all()

    def print_network_vars(self, is_print=False):
        if is_print:
            print('\nDy_dis_sup variables:')
            for var in self.Dy_dis_sup.variables:
                print(var.name)

            print('\nDy_dis_unsup variables:')
            for var in self.Dy_dis_unsup.variables:
                print(var.name)

            print('\nDx_dis_sup variables:')
            for var in self.Dx_dis_sup.variables:
                print(var.name)

            print('\nDx_dis_unsup variables:')
            for var in self.Dx_dis_unsup.variables:
                print(var.name)

    def train_step_integrated(self):
        # Supervised learning
        xy_fake_pairs, yx_fake_pairs, fake_y_imgs, fake_x_imgs = self.sess.run(
            [self.xy_fake_pairs, self.yx_fake_pairs, self.fake_y_imgs, self.fake_x_imgs])

        sup_ops = [self.optims_integrated,
                   self.G_gen_loss_integrated, self.Dy_dis_loss_integrated,
                   self.F_gen_loss_integrated, self.Dx_dis_loss_integrated]

        feed_dict_sup = {self.xy_fake_pairs_tfph: self.fake_xy_pool_obj_sup.query(xy_fake_pairs),
                         self.yx_fake_pairs_tfph: self.fake_yx_pool_obj_sup.query(yx_fake_pairs),
                         self.xy_fake_unpairs_tfph: self.fake_xy_pool_obj_unsup.query(fake_y_imgs),
                         self.yx_fake_unpairs_tfph: self.fake_yx_pool_obj_unsup.query(fake_x_imgs)}

        _, G_gen_loss_integrated, Dy_dis_loss_integrated, F_gen_loss_integrated, Dx_dis_loss_integrated = \
            self.sess.run(sup_ops, feed_dict=feed_dict_sup)

        return [G_gen_loss_integrated, Dy_dis_loss_integrated, F_gen_loss_integrated, Dx_dis_loss_integrated]


    def train_step_sup(self):
        # Supervised learning
        xy_fake_pairs, yx_fake_pairs = self.sess.run([self.xy_fake_pairs, self.yx_fake_pairs])
        sup_ops = [self.optims_sup,
                   self.G_loss_sup, self.G_gen_loss_sup, self.G_cond_loss,
                   self.G_gdl_loss, self.G_perceptual_loss, self.G_ssim_loss, self.cycle_loss, self.Dy_dis_loss_sup,
                   self.F_loss_sup, self.F_gen_loss_sup, self.F_cond_loss,
                   self.F_gdl_loss, self.F_perceputal_loss, self.F_ssim_loss, self.Dx_dis_loss_sup,
                   self.summary_op]
        feed_dict_sup = {self.xy_fake_pairs_tfph: self.fake_xy_pool_obj_sup.query(xy_fake_pairs),
                     self.yx_fake_pairs_tfph: self.fake_yx_pool_obj_sup.query(yx_fake_pairs)}

        _, G_loss_sup, G_gen_loss_sup, G_cond_loss, G_gdl_loss, G_perceptual_loss, G_ssim_loss, cycle_loss, \
        Dy_loss_sup, F_loss_sup, F_gen_loss_sup, F_cond_loss, F_gdl_loss, F_perceptual_loss, F_ssim_loss, Dx_loss_sup, \
        summary = self.sess.run(sup_ops, feed_dict=feed_dict_sup)

        return [G_loss_sup, G_gen_loss_sup, G_cond_loss, G_gdl_loss, G_perceptual_loss, G_ssim_loss, cycle_loss,
                Dy_loss_sup, F_loss_sup, F_gen_loss_sup, F_cond_loss, F_gdl_loss, F_perceptual_loss, F_ssim_loss,
                Dx_loss_sup], summary

    def train_step_unsup(self):
        # Unsupervised learning
        fake_y_imgs, fake_x_imgs = self.sess.run([self.fake_y_imgs, self.fake_x_imgs])
        unsup_ops = [self.optims_unsup,
                     self.G_loss_unsup, self.G_gen_loss_unsup, self.cycle_loss, self.Dy_dis_loss_unsup,
                     self.F_loss_unsup, self.F_gen_loss_unsup, self.Dx_dis_loss_unsup]
        feed_dict_unsup = {self.xy_fake_unpairs_tfph: self.fake_xy_pool_obj_unsup.query(fake_y_imgs),
                           self.yx_fake_unpairs_tfph: self.fake_yx_pool_obj_unsup.query(fake_x_imgs)}

        _, G_loss_unsup, G_gen_loss_unsup, cycle_loss, Dy_loss_unsup, F_loss_unsup, F_gen_loss_unsup, Dy_loss_unsup = \
            self.sess.run(unsup_ops, feed_dict=feed_dict_unsup)

        return [G_loss_unsup, G_gen_loss_unsup, cycle_loss, Dy_loss_unsup,
                F_loss_unsup, F_gen_loss_unsup, Dy_loss_unsup]

    def test_step(self):
        x_val, y_val, img_name = self.sess.run([self.x_imgs, self.y_imgs, self.img_name])
        fake_y, fake_x = self.sess.run([self.fake_y_sample, self.fake_x_sample],
                                       feed_dict={self.x_test_tfph: x_val, self.y_test_tfph: y_val})

        return [x_val, fake_y, y_val, fake_x], img_name

    def sample_imgs(self):
        x_val, y_val = self.sess.run([self.x_imgs, self.y_imgs])
        fake_y, fake_x = self.sess.run([self.fake_y_sample, self.fake_x_sample],
                                       feed_dict={self.x_test_tfph: x_val, self.y_test_tfph: y_val})

        return [x_val, fake_y, y_val, fake_x]

    def print_info(self, loss, iter_time, is_sup=True):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            if is_sup:
                ord_output = collections.OrderedDict([('tar_iters', self.flags.iters),
                                                      ('G_loss_sup', loss[0]), ('G_gen_loss_sup', loss[1]),
                                                      ('G_cond_loss', loss[2]), ('G_gdl_loss', loss[3]),
                                                      ('G_perceptual_loss', loss[4]), ('G_ssim_loss', loss[5]),
                                                      ('G_cycle_loss_sup', loss[6]), ('Dy_loss_sup', loss[7]),
                                                      ('F_loss_sup', loss[8]), ('F_gen_loss_sup', loss[9]),
                                                      ('F_cond_loss', loss[10]), ('F_gdl_loss', loss[11]),
                                                      ('F_perceptual_loss', loss[12]), ('F_ssim_loss', loss[13]),
                                                      ('F_cycle_loss_sup', loss[6]), ('Dx_loss_sup', loss[14]),
                                                      ('gpu_index', self.flags.gpu_index)])
            else:
                ord_output = collections.OrderedDict([('tar_iters', self.flags.iters),
                                                      ('G_loss_unsup', loss[0]), ('G_gen_loss_unsup', loss[1]),
                                                      ('G_cycle_loss_unsup', loss[2]), ('Dy_loss_unsup', loss[3]),
                                                      ('F_loss_unsup', loss[4]), ('F_gen_loss_unsup', loss[5]),
                                                      ('F_cycle_loss_unsup', loss[2]), ('Dx_loss_unsup', loss[6]),
                                                      ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def print_info_integrated(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('tar_iters', self.flags.iters),
                                                  ('G_gen_loss', loss[0]), ('Dy_dis_loss', loss[1]),
                                                  ('F_gen_loss', loss[2]), ('Dx_dis_loss', loss[3]),
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
        # cv2.imwrite(os.path.join(gt_file, img_name_), canvas[:, 2*self.img_size[1]:3*self.img_size[1]])


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
    def __init__(self, name='', ndf=64, norm='instance', model='a', shared_reuse=False, _ops=None):
        self.name = name
        self.shared_name = self.name[0:3] + 'shared'
        self.ndf = ndf
        self.norm = norm
        self._ops = _ops
        self.reuse = False
        self.shared_reuse = shared_reuse
        self.model = model
        self.variables = None

    def __call__(self, x):
        if self.model.lower() == 'a':
            output = self.model_a(x)
        elif self.model.lower() == 'b':
            output = self.model_b(x)
        elif self.model.lower() == 'c':
            output = self.model_c(x)
        elif self.model.lower() == 'd':
            output = self.model_d(x)
        elif self.model.lower() == 'e':
            output = self.model_e(x)
        elif self.model.lower() == 'f':
            output = self.model_f(x)
        elif self.model.lower() == 'g':
            print("Hello model G!")
            output = self.model_g(x)
        else:
            raise NotImplementedError

        return output

    def model_a(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

        with tf.variable_scope(self.shared_name, reuse=self.shared_reuse):
            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            output = tf.identity(conv5, name='output_without_sigmoid')

        # set reuse=True for next call
        self.reuse = True
        self.shared_reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.shared_name)

        return output

    def model_b(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

        with tf.variable_scope(self.shared_name, reuse=self.shared_reuse):
            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            output = tf.identity(conv5, name='output_without_sigmoid')

        # set reuse=True for next call
        self.reuse = True
        self.shared_reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.shared_name)

        return output

    def model_c(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

        with tf.variable_scope(self.shared_name, reuse=self.shared_reuse):
            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            output = tf.identity(conv5, name='output_without_sigmoid')

        # set reuse=True for next call
        self.reuse = True
        self.shared_reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.shared_name)

        return output

    def model_d(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

        with tf.variable_scope(self.shared_name, reuse=self.shared_reuse):
            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            output = tf.identity(conv5, name='output_without_sigmoid')

        # set reuse=True for next call
        self.reuse = True
        self.shared_reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.shared_name)

        return output

    def model_e(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

        with tf.variable_scope(self.shared_name, reuse=self.shared_reuse):
            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            output = tf.identity(conv5, name='output_without_sigmoid')

        # set reuse=True for next call
        self.reuse = True
        self.shared_reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.shared_name)

        return output

    def model_f(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

        with tf.variable_scope(self.shared_name, reuse=self.shared_reuse):
            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8 * self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            output = tf.identity(conv5, name='output_without_sigmoid')

        # set reuse=True for next call
        self.reuse = True
        self.shared_reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.shared_name)

        return output

    def model_g(self, x):
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

            output = tf.identity(conv5, name='output_without_sigmoid')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output

