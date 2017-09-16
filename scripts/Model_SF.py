from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorlayer as tl
from Model import Model
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag
import numpy as np
import shutil
import utility

class Model_SF(Model):
    """ Prediction model for single future frame """
    def __init__(self, args):
        super(Model_SF, self).__init__(args)
        self.ckpt_dir = os.path.join(self.args.train_dir, 'sf_ckpt')
        if not args.test and not args.restore_training and not args.train_condition == 'real_time_play':
            if os.path.exists(self.ckpt_dir):
                print("Checkpoint file path :%s already exist..."%self.ckpt_dir)
                print("Do you want to delete this folder and recreate one? ( \'y\' or \'n\')")
                while True:
                    keyboard_input = raw_input("Enter your choice:\n")
                    if keyboard_input == 'y':
                        shutil.rmtree(self.ckpt_dir)
                        break
                    elif keyboard_input == 'n':
                        break
                    else:
                        print("Unrecognized response, please enter \'y\' or \'n\'")
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.args.seq_length, self.args.features_dim],
                                    name='input_data')
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.args.gaussian_dim], name='target')
        self.build_model()
        self.initialize()


    def build_model(self):
        net = tl.layers.InputLayer(self.input, name='input_layer')
        with tf.variable_scope('fc1'):
            net = tl.layers.TimeDistributedLayer(net,
                                                 layer_class=tl.layers.DenseLayer,
                                                 args={'n_units': 64,
                                                       'act': tf.nn.elu,
                                                       'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                                       'W_init_args': {'regularizer':
                                                           tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                                       'name': 'fc1_'},
                                                 name='time_dense_fc1')
            # net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='fc1_drop')
        with tf.variable_scope('highway'):
            num_highway = 3
            for idx in xrange(num_highway):
                highway_args = {'n_units': 64,
                                'act': tf.nn.elu,
                                'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                'b_init': tf.constant_initializer(value=0.0),
                                'W_init_args': {'regularizer':
                                                    tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                'name': 'highway_%03d_' % idx}
                net = tl.layers.TimeDistributedLayer(net,
                                                     layer_class=utility.Highway,
                                                     args=highway_args,
                                                     name='time_dense_highway_%d' % idx)
                # if idx % 8 == 0:
                #     net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='highway_drop_%d' % idx)
        # net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='highway_drop')
        with tf.variable_scope('fc2'):
            net = tl.layers.TimeDistributedLayer(net,
                                                 layer_class=tl.layers.DenseLayer,
                                                 args={'n_units': 64,
                                                       'act': tf.nn.elu,
                                                       'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                                       'W_init_args': {'regularizer':
                                                                           tf.contrib.layers.l2_regularizer(
                                                                               self.args.weight_decay)},
                                                       'name': 'highway_to_fc_'},
                                                 name='time_dense_highway_to_fc')
            net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='hw_to_fc_drop')
        with tf.variable_scope('RNN'):
            if self.args.rnn_cell == 'lstm':
                rnn_cell_fn = tf.contrib.rnn.BasicLSTMCell
            elif self.args.rnn_cell == 'gru':
                rnn_cell_fn = tf.contrib.rnn.GRUCell
            else:
                raise ValueError('Unimplemented RNN Cell, should be \'lstm\' or \'gru\'')
            self.rnn_keep_prob = tf.placeholder(tf.float32)
            rnn_layer_name = 'DRNN_layer'
            net = tl.layers.DynamicRNNLayer(layer=net,
                                            cell_fn=rnn_cell_fn,
                                            n_hidden=128,
                                            dropout=(1.0, self.rnn_keep_prob),
                                            n_layer=self.args.num_cells,
                                            return_last=True,
                                            name=rnn_layer_name)
            rnn_weights_params = [var for var in net.all_params
                                  if rnn_layer_name in var.name
                                  and 'weights' in var.name]
            self.add_regularization_loss(rnn_weights_params)
        # net = tl.layers.DenseLayer(net,
        #                            n_units=50,
        #                            act=tf.nn.elu,
        #                            W_init=tf.contrib.layers.variance_scaling_initializer(),
        #                            name='fc1')

        # with tf.variable_scope('Highway'):
        #     num_highway = 15
        #     for idx in xrange(num_highway - 1):
        #         net = utility.Highway(net,
        #                               n_units=64,
        #                               act=tf.nn.elu,
        #                               W_init=tf.contrib.layers.variance_scaling_initializer(),
        #                               b_init=tf.constant_initializer(value=0.0),
        #                               W_init_args={'regularizer': tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
        #                               reuse=False,
        #                               name='highway_%d'%idx)
        net = tl.layers.DenseLayer(net,
                                   n_units=64,
                                   act=tf.nn.elu,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   W_init_args={'regularizer':
                                                    tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                   name='fc_3')
        mus_num = self.args.num_mixtures * self.args.gaussian_dim
        sigmas_num = self.args.num_mixtures * self.args.gaussian_dim
        weights_num = self.args.num_mixtures
        num_output = mus_num + sigmas_num + weights_num
        self.net = tl.layers.DenseLayer(net,
                                        n_units=num_output,
                                        act=tf.identity,
                                        W_init=tf.contrib.layers.variance_scaling_initializer(),
                                        W_init_args={'regularizer':
                                                         tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                        name='nn_output')
        output = self.net.outputs
        with tf.variable_scope('MDN'):
            mus =output[:, :mus_num]
            sigmas = tf.exp(output[:, mus_num: mus_num + sigmas_num])
            self.weight_logits = output[:, mus_num + sigmas_num:]
            self.mus = tf.reshape(mus, (-1, self.args.num_mixtures, self.args.gaussian_dim))
            self.sigmas = tf.reshape(sigmas, (-1, self.args.num_mixtures, self.args.gaussian_dim))
            self.weights = tf.nn.softmax(self.weight_logits)
            cat = Categorical(logits=self.weight_logits)
            components = [MultivariateNormalDiag(mu=mu, diag_stdev=sigma) for mu, sigma
                          in zip(tf.unstack(tf.transpose(self.mus, (1, 0, 2))),
                                 tf.unstack(tf.transpose(self.sigmas, (1, 0, 2))))]
            self.y_mix = Mixture(cat=cat, components=components)
        self.loss = self.get_loss()


    def get_loss(self):
        with tf.variable_scope('Loss'):
            loss = -self.y_mix.log_prob(self.target)
            loss = tf.reduce_mean(loss) + tf.losses.get_total_loss()
        return loss


    def print_stats(self, distances, title=None, draw=True, save_to_file=False):
        if len(distances.shape) == 2:
            distances = np.average(distances, axis=1)
        from scipy import stats
        n, min_max, mean, var, skew, kurt = stats.describe(distances)
        median = np.median(distances)
        first_quartile = np.percentile(distances, 25)
        third_quartile = np.percentile(distances, 75)
        print('\nDistances statistics:')
        print("Minimum: {0:9.4f} Maximum: {1:9.4f}".format(min_max[0], min_max[1]))
        print("Mean: {0:9.4f}".format(mean))
        print("Variance: {0:9.4f}".format(var))
        print("Median: {0:9.4f}".format(median))
        print("First quartile: {0:9.4f}".format(first_quartile))
        print("Third quartile: {0:9.4f}".format(third_quartile))
        threshold = 0.01
        percentage_thr = (distances <= threshold).sum() / float(distances.size) * 100.0
        percentage_double_thr = (distances <= 2 * threshold).sum() / float(distances.size) * 100.0
        percentage_triple_thr = (distances <= 3 * threshold).sum() / float(distances.size) * 100.0
        print("Percentage of testing with distance less than {0:.3f}m is: {1:4.2f} %".format(threshold,
                                                                                             percentage_thr))
        print("Percentage of testing with distance less than {0:.3f}m is: {1:4.2f} %".format(2 * threshold,
                                                                                             percentage_double_thr))
        print("Percentage of testing with distance less than {0:.3f}m is: {1:4.2f} %".format(3 * threshold,
                                                                                             percentage_triple_thr))
        if draw:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.set_style("whitegrid")
                plt.figure()
                vio_ax = sns.violinplot(x=distances, cut=0)
                vio_ax.set_xlabel('distances_error')
                if title is not None:
                    plt.title(title)
                plt.figure()
                strip_ax = sns.stripplot(x=distances)
                strip_ax.set_xlabel('distances_error')
                if title is not None:
                    plt.title(title)
            except ImportError:
                pass

        if save_to_file:
            import csv
            filename = os.path.join(self.ckpt_dir, 'error_stats.csv')
            with open(filename, 'a+') as f:
                csv_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
                data = [percentage_thr, percentage_double_thr, percentage_triple_thr]
                csv_writer.writerow(data)

