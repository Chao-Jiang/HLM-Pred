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

class Model_MF(Model):
    """ Prediction model for multiple future frames """
    def __init__(self, args):
        super(Model_MF, self).__init__(args)
        self.ckpt_dir = os.path.join(self.args.train_dir, 'mf_ckpt')
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
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.args.pred_frames_num, self.args.gaussian_dim],
                                     name='target')
        self.build_model()
        self.initialize()


    def build_model(self):
        net = tl.layers.InputLayer(self.input, name='input_layer')
        with tf.variable_scope('fc1'):
            net = tl.layers.TimeDistributedLayer(net,
                                                 layer_class=tl.layers.DenseLayer,
                                                 args={'n_units': 32,
                                                       'act': tf.nn.elu,
                                                       'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                                       'W_init_args': {'regularizer':
                                                                           tf.contrib.layers.l2_regularizer(
                                                                               self.args.weight_decay)},
                                                       'name': 'fc1_'},
                                                 name='time_dense_fc1')
            # net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='fc1_drop')
        with tf.variable_scope('highway'):
            num_highway = 3
            for idx in xrange(num_highway):
                highway_args = {'n_units': 32,
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
        net = tl.layers.DenseLayer(net,
                                   n_units=256,
                                   act=tf.nn.elu,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   W_init_args={'regularizer':
                                                    tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                   name='fc_3')
        net = tl.layers.DenseLayer(net,
                                   n_units=128,
                                   act=tf.nn.elu,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   W_init_args={'regularizer':
                                                    tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                   name='fc_4')
        mus_num = self.args.num_mixtures * self.args.gaussian_dim
        sigmas_num = self.args.num_mixtures * self.args.gaussian_dim
        weights_num = self.args.num_mixtures
        num_output = mus_num + sigmas_num + weights_num
        net = tl.layers.DenseLayer(net,
                                   n_units=num_output * self.args.pred_frames_num,
                                   act=tf.identity,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   W_init_args={'regularizer':
                                                    tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                   name='nn_output')
        self.net = tl.layers.ReshapeLayer(net, shape=[-1, self.args.pred_frames_num, num_output], name='reshape')

        output = self.net.outputs
        with tf.variable_scope('MDN'):
            mus = output[:, :, :mus_num]
            sigmas = tf.exp(output[:, :, mus_num: mus_num + sigmas_num])
            self.weight_logits = output[:, :, mus_num + sigmas_num:]
            self.mus = tf.reshape(mus, (-1, self.args.pred_frames_num,
                                        self.args.num_mixtures, self.args.gaussian_dim))
            self.sigmas = tf.reshape(sigmas, (-1, self.args.pred_frames_num,
                                              self.args.num_mixtures, self.args.gaussian_dim))
            self.weights = tf.nn.softmax(self.weight_logits)
            self.y_mix = []
            for time_step in xrange(self.args.pred_frames_num):
                cat = Categorical(logits=self.weight_logits[:, time_step, :])
                components = [MultivariateNormalDiag(mu=mu, diag_stdev=sigma) for mu, sigma
                              in zip(tf.unstack(tf.transpose(self.mus[:, time_step, :, :], (1, 0, 2))),
                                     tf.unstack(tf.transpose(self.sigmas[:, time_step, :, :], (1, 0, 2))))]
                self.y_mix.append(Mixture(cat=cat, components=components))
        self.loss = self.get_loss()

    def get_loss(self):
        with tf.variable_scope('Loss'):
            for time_step in xrange(self.args.pred_frames_num):
                if time_step == 0:
                    loss = -self.y_mix[time_step].log_prob(self.target[:, time_step, :])
                else:
                    loss += -self.y_mix[time_step].log_prob(self.target[:, time_step, :])
            loss = tf.reduce_mean(loss) + tf.losses.get_total_loss()
        return loss

    def print_stats(self, distances, title=None, draw=True, save_to_file=False):
        for idx in xrange(distances.shape[1]):
            time_distance = distances[:, idx]
            print('\nTime step %d Distances statistics:' % idx)
            threshold = 0.015
            percentage_thr = (time_distance <= threshold).sum() / float(time_distance.size) * 100.0
            percentage_double_thr = (time_distance <= 2 * threshold).sum() / float(time_distance.size) * 100.0
            percentage_triple_thr = (time_distance <= 3 * threshold).sum() / float(time_distance.size) * 100.0
            print("Percentage with distance less than {0:.3f}m is: {1:4.2f} %".format(threshold,
                                                                                                 percentage_thr))
            print("Percentage with distance less than {0:.3f}m is: {1:4.2f} %".format(2 * threshold,
                                                                                                 percentage_double_thr))
            print("Percentage with distance less than {0:.3f}m is: {1:4.2f} %".format(3 * threshold,
                                                                                                 percentage_triple_thr))
        if draw:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.set_style("whitegrid")
                for idx in xrange(distances.shape[1]):
                    plt.figure()
                    vio_ax = sns.violinplot(x=distances[:, idx], cut=0)
                    vio_ax.set_xlabel('distances_error')
                    if title is not None:
                        plt.title(title + '-Timestep %d' % idx)
                    plt.figure()
                    strip_ax = sns.stripplot(x=distances[:, idx])
                    strip_ax.set_xlabel('distances_error')
                    if title is not None:
                        plt.title(title + '-Timestep %d' % idx)
            except ImportError:
                pass

        if save_to_file:
            import csv
            filename = os.path.join(self.ckpt_dir, 'error_stats.csv')
            with open(filename, 'a+') as f:
                csv_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
                data = [percentage_thr, percentage_double_thr, percentage_triple_thr]
                csv_writer.writerow(data)


# Arguments: --train_condition=multiframe_pred -fd=4 --num_cells=2 -wd=0.00013 -sl=7 --keep_prob=0.75
# Training Result:
# Testing data testing
# ---------------------
# Start testing the network ...
# Euc distance of testing data without outliers:  0.0124441407791
#
# Time step 0 Distances statistics:
# Percentage with distance less than 0.015m is: 71.17 %
# Percentage with distance less than 0.030m is: 92.83 %
# Percentage with distance less than 0.045m is: 95.83 %
#
# Time step 1 Distances statistics:
# Percentage with distance less than 0.015m is: 69.83 %
# Percentage with distance less than 0.030m is: 90.67 %
# Percentage with distance less than 0.045m is: 95.17 %
#
# Time step 2 Distances statistics:
# Percentage with distance less than 0.015m is: 66.33 %
# Percentage with distance less than 0.030m is: 89.33 %
# Percentage with distance less than 0.045m is: 94.83 %
#
# Time step 3 Distances statistics:
# Percentage with distance less than 0.015m is: 60.67 %
# Percentage with distance less than 0.030m is: 88.17 %
# Percentage with distance less than 0.045m is: 94.83 %
#
# Time step 4 Distances statistics:
# Percentage with distance less than 0.015m is: 51.00 %
# Percentage with distance less than 0.030m is: 84.17 %
# Percentage with distance less than 0.045m is: 94.00 %
