from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
from datetime import timedelta
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import utility



class Model(object):
    def __init__(self, args):
        self.args = args
        tf.set_random_seed(self.args.random_seed)
        if self.args.test:
            self.args.keep_prob = 1.0
        self.input = None
        self.target = None
        self.loss = None
        self.rnn_keep_prob = tf.placeholder(tf.float32)
        self.net = None



    def build_model(self):
        raise NotImplementedError()


    def get_loss(self):
        raise NotImplementedError()

    def add_regularization_loss(self, params):
        with tf.variable_scope('Regularization'):
            for param in params:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     tf.contrib.layers.l2_regularizer(self.args.weight_decay)(param))


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
        self.summary_op = tf.summary.merge_all()

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tf_config)

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
            self.sess.run(tf.assign(self.featurewise_min, train_min))
            self.sess.run(tf.assign(self.featurewise_max, train_max))

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
            if len(y_preds.shape) > 2:
                euc_dis = np.linalg.norm(y_preds - y_true, axis=2)
            else:
                euc_dis = np.linalg.norm(y_preds - y_true, axis=1)
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
                y_preds_batch = self.predict_batch(X_batch)
                if y_preds is None:
                    y_preds = y_preds_batch
                else:
                    y_preds = np.concatenate((y_preds, y_preds_batch), axis=0)
        else:
            y_preds = self.predict_batch(X)

        return y_preds


    def predict_batch(self, X_batch):
        feed_dict = {self.input: X_batch,
                     self.rnn_keep_prob: 1}
        dp_dict = tl.utils.dict_to_one(self.net.all_drop)
        feed_dict.update(dp_dict)
        pred_weights, pred_means, pred_std = self.sess.run(
            [self.weights, self.mus, self.sigmas], feed_dict=feed_dict)

        if len(pred_weights.shape) > 2:
            b_idx = np.repeat(np.arange(pred_weights.shape[0]), pred_weights.shape[1])
            s_idx = np.tile(np.arange(pred_weights.shape[1]), pred_weights.shape[0])
            g_idx = np.argmax(pred_weights, axis=-1).reshape(-1)
            y_preds = pred_means[b_idx, s_idx, g_idx].reshape(pred_weights.shape)
        else:
            y_preds = pred_means[np.arange(pred_means.shape[0]), np.argmax(pred_weights, axis=-1)]
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


    def print_stats(self, distances, title=None, draw=True, save_to_file=False):
        raise NotImplementedError()