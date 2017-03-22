from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag, MultivariateNormalFull
import numpy as np
import utility



class Model(object):
    def __init__(self, args):
        self.args = args
        self.ckpt_dir = os.path.join(self.args.train_dir, 'ckpt')
        tf.set_random_seed(self.args.random_seed)
        if self.args.test:
            self.args.keep_prob = 1.0
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.args.seq_length, self.args.features_dim], name='input_data')
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.args.gaussian_dim], name='target')
        self.build_model()
        self.initialize()



    def build_model(self):
        input = tf.reshape(self.input, shape=tf.stack([-1, self.args.seq_length * self.args.features_dim]))
        net = tl.layers.InputLayer(input, name='input_layer')
        net = tl.layers.DenseLayer(net,
                                   n_units=128,
                                   act=tf.nn.elu,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   name='fc1')
        net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='drop1')
        net = tl.layers.DenseLayer(net,
                                   n_units=64,
                                   act=tf.nn.elu,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   name='fc2')
        net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='drop2')
        net = tl.layers.DenseLayer(net,
                                   n_units=64,
                                   act=tf.nn.elu,
                                   W_init=tf.contrib.layers.variance_scaling_initializer(),
                                   name='fc3')
        net = tl.layers.DropoutLayer(net, keep=self.args.keep_prob, name='drop3')
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
        # net = tl.layers.DenseLayer(net,
        #                            n_units=32,
        #                            act=tf.nn.elu,
        #                            W_init=tf.contrib.layers.variance_scaling_initializer(),
        #                            name='fc2')
        mus_num = self.args.num_mixtures * self.args.gaussian_dim
        sigmas_num = self.args.num_mixtures * self.args.gaussian_dim
        weights_num = self.args.num_mixtures
        num_output = mus_num + sigmas_num + weights_num
        self.net = tl.layers.DenseLayer(net,
                                        n_units=num_output,
                                        W_init=tf.contrib.layers.variance_scaling_initializer(),
                                        name='nn_output')
        output = self.net.outputs
        with tf.variable_scope('MDN'):
            mus =output[:, :mus_num]
            sigmas = tf.exp(output[:, mus_num: mus_num + sigmas_num])
            self.weight_logits = output[:, mus_num + sigmas_num:]
            self.mus = tf.reshape(mus, (-1, self.args.num_mixtures, self.args.gaussian_dim))
            self.sigmas = tf.reshape(sigmas, (-1, self.args.num_mixtures, self.args.gaussian_dim))
            self.weights = tf.nn.softmax(self.weight_logits)

        # if self.args.normal_cov == 'diag':
        #     with tf.variable_scope('sigmas'):
        #         sigmas = tl.layers.DenseLayer(net,
        #                                       n_units=self.args.num_mixtures * self.args.gaussian_dim,
        #                                       act=tf.exp,
        #                                       W_init=tf.contrib.layers.variance_scaling_initializer())
        #         self.sigmas = tf.reshape(sigmas, (-1, self.args.num_mixtures, self.args.gaussian_dim))
        #     components = [MultivariateNormalDiag(mu=mu, diag_stdev=sigma) for mu, sigma
        #                   in zip(tf.unstack(tf.transpose(self.mus, (1, 0, 2))),
        #                          tf.unstack(tf.transpose(self.sigmas, (1, 0, 2))))]
        # elif self.args.normal_cov == 'full':
        #     with tf.variable_scope('sigmas'):
        #         sigmas = tl.layers.DenseLayer(net,
        #                                       n_units=self.args.num_mixtures * self.args.gaussian_dim ** 2,
        #                                       act=tf.exp,
        #                                       W_init=tf.contrib.layers.variance_scaling_initializer())
        #         self.sigmas = tf.reshape(sigmas, (-1, self.args.num_mixtures, self.args.gaussian_dim, self.args.gaussian_dim))
        #     components = [MultivariateNormalFull(mu=mu, sigma=sigma) for mu, sigma
        #                   in zip(tf.unstack(tf.transpose(self.mus, (1, 0, 2, 3))),
        #                          tf.unstack(tf.transpose(self.sigmas, (1, 0, 2, 3))))]
        # else:
        #     raise ValueError('Unimplemented covariance matrix type, should be \'diag\' or \'full\'')
            cat = Categorical(logits=self.weight_logits)
            components = [MultivariateNormalDiag(mu=mu, diag_stdev=sigma) for mu, sigma
                          in zip(tf.unstack(tf.transpose(self.mus, (1, 0, 2))),
                                 tf.unstack(tf.transpose(self.sigmas, (1, 0, 2))))]
            self.y_mix = Mixture(cat=cat, components=components)
        with tf.variable_scope('Regularization'):
            for param in self.net.all_params:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.contrib.layers.l2_regularizer(self.args.weight_decay)(param))
        self.loss = self.get_loss()


    def get_loss(self):
        with tf.variable_scope('Loss'):
            loss = -self.y_mix.log_prob(self.target)
            loss = tf.reduce_sum(loss) + tf.losses.get_total_loss()
        return loss


    def initialize(self, weight_histograms=True):
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
        tf.summary.scalar('loss', self.loss)

        self.eval_loss = tf.placeholder(dtype=tf.float32, shape=(), name='eval_loss')
        tf.summary.scalar('evaluation_mse', self.eval_loss)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tf_config)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.ckpt_dir, self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=0)

        tf.global_variables_initializer().run()



    def fit(self, X_train, y_train, X_eval=None, y_eval=None, restore=False):
        if restore:
            self.restore_model()
        if X_eval is None or y_eval is None:
            if X_eval is None and y_eval is None:
                from sklearn.model_selection import train_test_split
                X_train, X_eval, y_train, y_eval = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.1,
                                                                    random_state=0)
                # if X_eval.shape[0] >
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
                             self.target: y_train_batch}
                feed_dict.update(self.net.all_drop)
                loss_value, _, global_step = self.sess.run([self.loss, self.train_op, self.global_step], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                loss_ep += loss_value
                n_step += 1

                if global_step % self.args.ckpt_interval == 0:
                    feed_dict = {self.input: X_eval,
                                 self.target: y_eval}
                    dp_dict = tl.utils.dict_to_one(self.net.all_drop)
                    feed_dict.update(dp_dict)
                    eval_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    format_str = ('\n%s: step %d, loss = \033[93m %.2f\033[00m\n')
                    print(format_str % (datetime.now(), global_step, loss_value))
                    feed_dict = {self.input: X_train_batch,
                                 self.target: y_train_batch,
                                 self.eval_loss: eval_loss}
                    feed_dict.update(self.net.all_drop)
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.train_writer.add_summary(summary_str, global_step)
                    self.save_model(global_step)
            loss_ep = loss_ep / n_step
            print("Epoch %d of %d took %fs, epoch avg loss \033[94m %.2f \033[00m" % (epoch + 1,
                                                        self.args.num_epochs,
                                                        time.time() - epoch_start_time, loss_ep))
        print("Total training time: %fs" % (time.time() - start_time_begin))


    def resume_training(self, X_train, y_train):
        self.restore_model()
        self.fit(X_train, y_train)


    def test(self, X_test, y_test, verbose=True, name=None, draw=True):
        if verbose:
            print('Start testing the network ...')
        with tf.variable_scope('test'):
            y_preds = self.predict(X_test)
            y_true = y_test
            from sklearn.metrics import mean_squared_error
            sk_mse = mean_squared_error(y_preds, y_true)
            if verbose:
                print('Mean squared error of testing data by sklearn:', sk_mse)
                euc_dis = np.linalg.norm(y_preds - y_true, axis=1)
                self.print_stats(euc_dis, name, draw)
        return y_preds, sk_mse


    def predict(self, X):
        y_preds = tf.Variable(np.empty((0, self.args.gaussian_dim)), dtype=tf.float32)
        if X.shape[0] > 64:
            for X_batch, _ in utility.minibatches(
                    X, np.ones((X.shape[0], 1)), 32, shuffle=False):
                feed_dict = {self.input: X_batch}
                dp_dict = tl.utils.dict_to_one(self.net.all_drop)
                feed_dict.update(dp_dict)
                pred_weights, pred_means, pred_std = self.sess.run(
                    [self.weights, self.mus, self.sigmas], feed_dict=feed_dict)
                for idx in xrange(pred_weights.shape[0]):
                    mixture_index = np.argmax(pred_weights[idx])
                    y_pred = pred_means[idx, mixture_index].reshape(1, self.args.gaussian_dim)
                    y_preds = tf.concat([y_preds, y_pred], axis=0)
        else:
            feed_dict = {self.input: X}
            dp_dict = tl.utils.dict_to_one(self.net.all_drop)
            feed_dict.update(dp_dict)
            pred_weights, pred_means, pred_std = self.sess.run(
                [self.weights, self.mus, self.sigmas], feed_dict=feed_dict)
            for idx in xrange(pred_weights.shape[0]):
                mixture_index = np.argmax(pred_weights[idx])
                y_pred = pred_means[idx, mixture_index].reshape(1, self.args.gaussian_dim)
                y_preds = tf.concat([y_preds, y_pred], axis=0)
        y_preds = self.sess.run(y_preds)
        return y_preds


    def save_model(self, step):
        checkpoint_path = os.path.join(self.ckpt_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=step)


    def restore_model(self, step=None):
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
        tf.assign(self.global_step, int(restore_path.split('/')[-1].split('-')[-1]))


    def print_stats(self, distances, title=None, draw=True):
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
        percentage = (distances <= threshold).sum() / float(distances.size) * 100.0
        percentage_double_thr = (distances <= 2 * threshold).sum() / float(distances.size) * 100.0
        percentage_triple_thr = (distances <= 3 * threshold).sum() / float(distances.size) * 100.0
        print("Percentage of testing with distance less than {0:.3f}m is: {1:4.2f} %".format(threshold,
                                                                                             percentage))
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




