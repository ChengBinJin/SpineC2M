import _pickle as cpickle
import tensorflow as tf

import tensorflow_utils as tf_utils

# noinspection PyPep8Naming
class VGG16(object):
    def __init__(self, name='vgg16'):
        self.name = name
        self.reuse = False

        weight_file_path = '../../Models_zoo/caffe_layers_value.pickle'
        with open(weight_file_path, 'rb') as f:
            self.pretrained_weights = cpickle.load(f, encoding='latin1')

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            x = tf.concat([x, x, x], axis=-1, name='concat')
            tf_utils.print_activations(x)

            # conv1
            relu1_1 = self.conv_layer(x, 'conv1_1', trainable=False)
            relu1_2 = self.conv_layer(relu1_1, 'conv1_2', trainable=False)
            pool_1 = tf_utils.max_pool_2x2(relu1_2, name='max_pool_1')
            tf_utils.print_activations(pool_1)

            # conv2
            relu2_1 = self.conv_layer(pool_1, 'conv2_1', trainable=False)
            relu2_2 = self.conv_layer(relu2_1, 'conv2_2', trainable=False)
            pool_2 = tf_utils.max_pool_2x2(relu2_2, name='max_pool_2')
            tf_utils.print_activations(pool_2)

            # conv3
            relu3_1 = self.conv_layer(pool_2, 'conv3_1', trainable=False)
            relu3_2 = self.conv_layer(relu3_1, 'conv3_2', trainable=False)
            relu3_3 = self.conv_layer(relu3_2, 'conv3_3', trainable=False)
            pool_3 = tf_utils.max_pool_2x2(relu3_3, name='max_pool_3')
            tf_utils.print_activations(pool_3)

            # conv4
            relu4_1 = self.conv_layer(pool_3, 'conv4_1', trainable=False)
            relu4_2 = self.conv_layer(relu4_1, 'conv4_2', trainable=False)
            relu4_3 = self.conv_layer(relu4_2, 'conv4_3', trainable=False)
            pool_4 = tf_utils.max_pool_2x2(relu4_3, name='max_pool_4')
            tf_utils.print_activations(pool_4)

            # conv5
            relu5_1 = self.conv_layer(pool_4, 'conv5_1', trainable=False)
            relu5_2 = self.conv_layer(relu5_1, 'conv5_2', trainable=False)
            relu5_3 = self.conv_layer(relu5_2, 'conv5_3', trainable=False)

            # set reuse=True for next call
            self.reuse = True

            return  relu5_3



    def conv_layer(self, bottom, name, trainable=False):
        with tf.variable_scope(name):
            w = self.get_conv_weight(name)
            b = self.get_bias(name)
            conv_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w),
                                           trainable=trainable)
            conv_biases = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b),
                                          trainable=trainable)

            conv = tf.nn.conv2d(bottom, conv_weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            tf_utils.print_activations(relu)

        return relu

    def get_conv_weight(self, name):
        f = self.get_weight(name)
        return f.transpose((2, 3, 1, 0))

    def get_weight(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[1]
