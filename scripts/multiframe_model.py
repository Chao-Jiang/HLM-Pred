from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
from datetime import datetime
from datetime import timedelta
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag, MultivariateNormalFull
import numpy as np
import utility
import shutil



class MultiFrameModel(object):
    def __init__(self, args):
        self.args = args
        self.ckpt_dir = os.path.join(self.args.train_dir, 'ckpt')
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

        tf.set_random_seed(self.args.random_seed)
        if self.args.test:
            self.args.keep_prob = 1.0
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.args.seq_length, self.args.features_dim], name='input_data')
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.args.pred_frames_num, self.args.gaussian_dim], name='target')
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
                                                           tf.contrib.layers.l2_regularizer(self.args.weight_decay)},
                                                       'name': 'fc1_'},
                                                 name='time_dense_fc1')
            # net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='fc1_drop')
        with tf.variable_scope('highway'):
            num_highway = 4
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
                # if idx % 8 == 0:
                #     net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='highway_drop_%d' % idx)
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
                                            return_last=False,
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
        net = utility.LastXSeq(net, seq_length=self.args.pred_frames_num, name='get_%02d_seq'%self.args.pred_frames_num)
        net = tl.layers.TimeDistributedLayer(net,
                                             layer_class=tl.layers.DenseLayer,
                                             args={'n_units': 32,
                                                   'act': tf.nn.elu,
                                                   'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                                   'W_init_args': {'regularizer':
                                                                       tf.contrib.layers.l2_regularizer(
                                                                           self.args.weight_decay)},
                                                   'name': 'fc3_'},
                                             name='time_dense_fc3')
        net = tl.layers.TimeDistributedLayer(net,
                                             layer_class=tl.layers.DenseLayer,
                                             args={'n_units': 32,
                                                   'act': tf.nn.elu,
                                                   'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                                   'W_init_args': {'regularizer':
                                                                       tf.contrib.layers.l2_regularizer(
                                                                           self.args.weight_decay)},
                                                   'name': 'fc4_'},
                                             name='time_dense_fc4')
        mus_num = self.args.num_mixtures * self.args.gaussian_dim
        sigmas_num = self.args.num_mixtures * self.args.gaussian_dim
        weights_num = self.args.num_mixtures
        num_output = mus_num + sigmas_num + weights_num
        self.net = tl.layers.TimeDistributedLayer(net,
                                             layer_class=tl.layers.DenseLayer,
                                             args={'n_units': num_output,
                                                   'act': tf.identity,
                                                   'W_init': tf.contrib.layers.variance_scaling_initializer(),
                                                   'W_init_args': {'regularizer':
                                                       tf.contrib.layers.l2_regularizer(
                                                           self.args.weight_decay)},
                                                   'name': 'nn_output'},
                                             name='time_dense_nn_output')

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


    def add_regularization_loss(self, params):
        with tf.variable_scope('Regularization'):
            for param in params:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     tf.contrib.layers.l2_regularizer(self.args.weight_decay)(param))


    def get_loss(self):
        with tf.variable_scope('Loss'):
            for time_step in xrange(self.args.pred_frames_num):
                if time_step == 0:
                    loss = -self.y_mix[time_step].log_prob(self.target[:, time_step, :])
                else:
                    loss += -self.y_mix[time_step].log_prob(self.target[:, time_step, :])
            loss = tf.reduce_mean(loss) + tf.losses.get_total_loss()
        return loss


    def initialize(self, weight_histograms=True):
        self.featurewise_min = tf.Variable(tf.zeros([self.args.seq_length, self.args.features_dim]),
                                           dtype=tf.float32,
                                           trainable=False,
                                           name='featurewise_min')
        self.featurewise_max = tf.Variable(tf.zeros([self.args.seq_length, self.args.features_dim]),
                                           dtype=tf.float32,
                                           trainable=False,
                                           name='featurewise_max')
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(self.args.learning_rate,
                                                            self.global_step,
                                                            decay_steps=3000,
                                                            decay_rate=0.9,
                                                            staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            var_list = tf.trainable_variables()
            grads = tf.gradients(self.loss, [v._ref() for v in var_list])
            grads_and_vars = list(zip(grads, var_list))
            grads_and_vars = [(tf.clip_by_norm(grad, self.args.grad_clip), var)
                                   for grad, var in grads_and_vars if grad is not None]
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        tf.summary.scalar('learning_rate', self.learning_rate)
        if weight_histograms:
            for param in self.net.all_params:
                tf.summary.histogram(param.name, param)
        tf.summary.scalar('train_loss', self.loss)

        self.eval_loss = tf.placeholder(dtype=tf.float32, shape=(), name='eval_loss')
        tf.summary.scalar('evaluation_loss', self.eval_loss)
        self.train_euc = tf.placeholder(dtype=tf.float32, shape=(), name='train_euc')
        tf.summary.scalar('train_euclidean_dist', self.train_euc)
        self.eval_euc = tf.placeholder(dtype=tf.float32, shape=(), name='eval_euc')
        tf.summary.scalar('evaluation_euclidean_dist', self.eval_euc)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tf_config)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.ckpt_dir, self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=0)

        tf.global_variables_initializer().run()



    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        X_train = X_train[:, : self.args.seq_length, :]
        if self.args.restore_training:
            self.restore_model()
            X_train, train_min, train_max = utility.featurewise_minmax(X_train,
                                                                       self.sess.run(self.featurewise_min),
                                                                       self.sess.run(self.featurewise_max))
        else:
            X_train, train_min, train_max = utility.featurewise_minmax(X_train)
            self.featurewise_min = self.featurewise_min + train_min#tf.assign(self.featurewise_min, train_min)
            self.featurewise_max = self.featurewise_max + train_max#tf.assign(self.featurewise_max, train_max)
            min_max_file = os.path.join(self.ckpt_dir, 'min_max.json')
            with open(min_max_file, 'w') as f:
                data = {}
                data['featurewise_min'] = self.sess.run(self.featurewise_min).tolist()
                data['featurewise_max'] = self.sess.run(self.featurewise_max).tolist()
                json.dump(data, f, indent=4)

        if X_eval is None or y_eval is None:
            if X_eval is None and y_eval is None:
                from sklearn.model_selection import train_test_split
                X_train, X_eval, y_train, y_eval = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.02,
                                                                    random_state=self.args.random_seed)
            else:
                raise Exception('X_eval and y_eval should be both provided or neither of them should be provided')
        print('Start training the network ...')
        start_time_begin = time.time()
        for epoch in xrange(self.args.num_epochs):
            epoch_start_time = time.time()
            loss_ep = 0
            n_step = 0
            for X_train_batch, y_train_batch in tl.iterate.minibatches(
                    X_train, y_train, self.args.batch_size, shuffle=True):
                feed_dict = {self.input: X_train_batch,
                             self.target: y_train_batch,
                             self.rnn_keep_prob: self.args.keep_prob}
                feed_dict.update(self.net.all_drop)
                loss_value, _, global_step = self.sess.run([self.loss, self.train_op, self.global_step], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                loss_ep += loss_value
                n_step += 1

                if global_step % self.args.ckpt_interval == 0:
                    _, train_euc = self.test(X_train_batch, y_train_batch, scale=False, verbose=False)
                    _, eval_euc = self.test(X_eval, y_eval, scale=False, verbose=False)
                    feed_dict = {self.input: X_eval,
                                 self.target: y_eval,
                                 self.rnn_keep_prob: 1}
                    dp_dict = tl.utils.dict_to_one(self.net.all_drop)
                    feed_dict.update(dp_dict)
                    eval_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    format_str = ('\n%s: step %d, loss = \033[93m %.2f\033[00m\n')
                    print(format_str % (datetime.now(), global_step, loss_value))
                    feed_dict = {self.input: X_train_batch,
                                 self.target: y_train_batch,
                                 self.rnn_keep_prob: self.args.keep_prob,
                                 self.eval_loss: eval_loss,
                                 self.train_euc: train_euc,
                                 self.eval_euc: eval_euc}
                    feed_dict.update(self.net.all_drop)
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.train_writer.add_summary(summary_str, global_step)
                    self.save_model(global_step)
            loss_ep = loss_ep / n_step
            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            remaining_time = epoch_elapsed_time * (self.args.num_epochs - epoch - 1)
            est_finished_time = datetime.now() + timedelta(seconds=remaining_time)
            print("Epoch %d / %d took %.2f s, epoch avg loss \033[94m %.2f \033[00m, "
                  "estimated finished time: %s" % (epoch + 1, self.args.num_epochs,
                                                  epoch_elapsed_time, loss_ep,
                                                  est_finished_time.strftime("%Y-%m-%d %H:%M:%S")))
        print("Total training time: %fs" % (time.time() - start_time_begin))



    def test(self, X_test, y_test, scale=True, verbose=True, name=None, draw=True, save_to_file=False):
        if verbose:
            print('Start testing the network ...')
        with tf.variable_scope('test'):
            y_preds = self.predict(X_test, scale=scale)
            y_true = y_test
            euc_dis = np.linalg.norm(y_preds - y_true, axis=2)
            # Remove potential outliers
            q25 = np.percentile(euc_dis, 25)
            q75 = np.percentile(euc_dis, 75)
            iqr = q75 - q25 # interquartile range
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            euc_remove_low = euc_dis[euc_dis > lower_bound]
            euc_filted = euc_remove_low[euc_remove_low < upper_bound]
            avg_euc_dist = np.average(euc_filted)
            if verbose:
                print('Euc distance of testing data without outliers: ', avg_euc_dist)
                self.print_stats(euc_dis, name, draw, save_to_file)
        return y_preds, avg_euc_dist


    def predict(self, X, scale=True):
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        X = X[:, : self.args.seq_length, :]
        if scale:
            X, _, _ = utility.featurewise_minmax(X,
                                                 self.sess.run(self.featurewise_min),
                                                 self.sess.run(self.featurewise_max))
        y_preds = None
        if X.shape[0] > 64:
            for X_batch, _ in utility.minibatches(
                    X, np.ones((X.shape[0], 1)), 32, shuffle=False):
                feed_dict = {self.input: X_batch,
                             self.rnn_keep_prob: 1}
                dp_dict = tl.utils.dict_to_one(self.net.all_drop)
                feed_dict.update(dp_dict)
                pred_weights, pred_means, pred_std = self.sess.run(
                    [self.weights, self.mus, self.sigmas], feed_dict=feed_dict)
                for idx in xrange(pred_weights.shape[0]):
                    mixture_index = np.argmax(pred_weights[idx], axis=1)
                    y_pred = None
                    for time_step in xrange(mixture_index.shape[0]):
                        if y_pred is None:
                            y_pred = pred_means[idx, time_step, mixture_index[time_step]].reshape(1, self.args.gaussian_dim)
                        else:
                            y_pred = tf.concat([y_pred,
                                                pred_means[idx, time_step,
                                                           mixture_index[time_step]].reshape(1, self.args.gaussian_dim)],
                                               axis=0)
                    if y_preds is None:
                        y_preds = tf.expand_dims(y_pred, axis=0)
                    else:
                        y_preds = tf.concat([y_preds, tf.expand_dims(y_pred, axis=0)], axis=0)
        else:
            feed_dict = {self.input: X,
                         self.rnn_keep_prob: 1}
            dp_dict = tl.utils.dict_to_one(self.net.all_drop)
            feed_dict.update(dp_dict)
            pred_weights, pred_means, pred_std = self.sess.run(
                [self.weights, self.mus, self.sigmas], feed_dict=feed_dict)
            for idx in xrange(pred_weights.shape[0]):
                mixture_index = np.argmax(pred_weights[idx], axis=1)
                y_pred = None
                for time_step in xrange(mixture_index.shape[0]):
                    if y_pred is None:
                        y_pred = pred_means[idx, time_step, mixture_index[time_step]].reshape(1, self.args.gaussian_dim)
                    else:
                        y_pred = tf.concat([y_pred,
                                            pred_means[idx, time_step,
                                                       mixture_index[time_step]].reshape(1, self.args.gaussian_dim)],
                                           axis=0)
                if y_preds is None:
                    y_preds = tf.expand_dims(y_pred, axis=0)
                else:
                    y_preds = tf.concat([y_preds, tf.expand_dims(y_pred, axis=0)], axis=0)
        if isinstance(y_preds, tf.Tensor):
            y_preds = self.sess.run(y_preds)
        return y_preds


    def save_model(self, step):
        checkpoint_path = os.path.join(self.ckpt_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=step)


    def restore_model(self, step=None):
        if step is None:
            step = self.args.restore_step
        if step is None:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                restore_path = ckpt.model_checkpoint_path
        else:
            ckpt = os.path.join(self.ckpt_dir, 'model.ckpt-%d'%step)
            restore_path = ckpt
            assert os.path.exists(ckpt+'.meta'), 'Checkpoint file %s not exist'%ckpt
        # Restores from checkpoint
        print('\nRestoring model from: ', os.path.abspath(restore_path))
        self.saver.restore(self.sess, restore_path)
        min_max_file = os.path.join(self.ckpt_dir, 'min_max.json')
        with open(min_max_file, 'r') as f:
            data = json.load(f)
            self.featurewise_min = tf.assign(self.featurewise_min, data['featurewise_min'])
            self.featurewise_max = tf.assign(self.featurewise_max, data['featurewise_max'])


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
        threshold = 0.02
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








