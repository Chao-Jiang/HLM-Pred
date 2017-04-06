from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import time
import cv2
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import tensorlayer as tl

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def build_toy_dataset(speed, alpha, beta, origin, time_steps, dt = 0.05):
    """
    create points on a parabola trajectory
    :param speed: starting speed
    :param alpha: polar angle
    :param beta: azimuthal angle
    :param origin: starting point coordinates
    :param time_steps: time steps of the trajectory to record
    :param dt: time step interval
    :return: time_steps x 3 points on the parabola trajectory
    """
    vel_z = speed * np.sin(alpha)
    vel_x = speed * np.cos(alpha) * np.cos(beta)
    vel_y = speed * np.cos(alpha) * np.sin(beta)
    coord_seq =[]
    gravity = 9.8
    delta_time_begin = np.random.rand() * dt  # add noise
    vel_z = vel_z - gravity * delta_time_begin  # add noise to at which time we start recording since the point starts moving
    origin[0] = origin[0] + vel_x * delta_time_begin
    origin[1] = origin[1] + vel_y * delta_time_begin
    origin[2] = origin[2] + vel_z * delta_time_begin * dt - 0.5 * gravity * (delta_time_begin ** 2)

    time_fall = (vel_z + np.sqrt(vel_z ** 2 + 2 * gravity * origin[2]))/gravity
    time_fall_step = np.ceil(time_fall/dt).astype(int)
    for time_step in xrange(time_fall_step):
        coord_x = origin[0] + vel_x * time_step * dt
        coord_y = origin[1] + vel_y * time_step * dt
        coord_z = origin[2] + vel_z * time_step * dt - 0.5 * gravity * ((time_step * dt) ** 2)
        coord_now = [coord_x, coord_y, np.fabs(coord_z)]
        coord_seq.append(coord_now)
    vel_z_reachground = (gravity*time_fall - vel_z) * 0.8
    delta_t = 0.05 - time_fall % dt
    coord_z = vel_z_reachground * delta_t - 1.0/2*gravity*(delta_t**2)
    coord_x = origin[0] + vel_x * time_fall_step * dt
    coord_y = origin[1] + vel_y * time_fall_step * dt
    newcoord = [coord_x, coord_y, coord_z]
    # coord_seq.append(newcoord)
    vel_z_new = vel_z_reachground-gravity * delta_t
    left_step = time_steps - time_fall_step
    for new_time_step in range(left_step):
        coord_x = newcoord[0] + vel_x * new_time_step * dt
        coord_y = newcoord[1] + vel_y * new_time_step * dt
        coord_z = newcoord[2] + vel_z_new * new_time_step * dt - 1.0 / 2 * gravity * ((new_time_step * dt) ** 2)
        coord_now = [coord_x, coord_y, np.fabs(coord_z)]
        coord_seq.append(coord_now)

    coord_seq = np.asarray(coord_seq)
    return coord_seq


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def plot_3d(points_list, title=None, draw_now=True, seq_length=None, start=0):
    assert len(points_list) <= 2, 'Length of points list should not be greater than two'
    fig = plt.figure()
    plt.rcParams['axes.facecolor'] = 'white'
    ax = fig.add_subplot(111, projection='3d')
    for idx in xrange(points_list[0].shape[0]):
        if seq_length is None:
            ax.plot(points_list[0][idx, :, 0],
                    points_list[0][idx, :, 1],
                    points_list[0][idx, :, 2],
                    'b-', label='input')
        else:
            import numbers
            assert isinstance(seq_length, numbers.Integral), 'Sequence length must be integer if provided'
            # if start > 0:
            #     ax.plot(points_list[0][idx, 0: start+1, 0],
            #             points_list[0][idx, 0: start+1, 1],
            #             points_list[0][idx, 0: start+1, 2],
            #             'm-', linewidth=1, label='true_trajectory')
            ax.plot(points_list[0][idx, :, 0],
                    points_list[0][idx, :, 1],
                    points_list[0][idx, :, 2],
                    'm-', linewidth=1, label='true_trajectory')
            ax.plot(points_list[0][idx, start: start+seq_length, 0],
                    points_list[0][idx, start: start+seq_length, 1],
                    points_list[0][idx, start: start+seq_length, 2],
                    'b-', linewidth=1, label='input')
            # ax.plot(points_list[0][idx, start+seq_length-1:, 0],
            #         points_list[0][idx, start+seq_length-1:, 1],
            #         points_list[0][idx, start+seq_length-1:, 2],
            #         'm-', linewidth=1, label='true_trajectory')
        ax.scatter(points_list[0][idx, :, 0],
                   points_list[0][idx, :, 1],
                   points_list[0][idx, :, 2],
                   marker='o', c='r', s=6)
        # ax.scatter(points_list[0][idx, 0, 0],
        #            points_list[0][idx, 0, 1],
        #            points_list[0][idx, 0, 2],
        #            marker='s', c='y', s=6)
        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if len(points_list) == 2:
            if len(points_list[1].shape) == 2:
                ax.scatter(points_list[1][idx, 0],
                           points_list[1][idx, 1],
                           points_list[1][idx, 2],
                           c='k', marker='x', s=10,
                           label='prediction')
            elif len(points_list[1].shape) == 3:
                ax.scatter(points_list[1][idx, :, 0],
                           points_list[1][idx, :, 1],
                           points_list[1][idx, :, 2],
                           c='k', marker='x', s=10,
                           label='prediction')
                ax.plot(points_list[1][idx, :, 0],
                        points_list[1][idx, :, 1],
                        points_list[1][idx, :, 2],
                        'kx-', linewidth=1)
    ax.legend(ncol=1, prop={'size': 12})
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    set_aspect_equal_3d(ax)
    if title is not None:
        plt.title(title)
    if draw_now:
        plt.show()


def remove_outliers(data, sup_data, cutoff_frame=50, sor_threshold=0.8, var_threshold=0.001):
    from sklearn import linear_model
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    valid_idx = []
    invalid_idx = []
    for idx in xrange(data.shape[0]):
        x = data[idx, :cutoff_frame, 1].reshape(-1, 1)
        y = data[idx, :cutoff_frame, 0].reshape(-1)
        regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
        regr.fit(x, y)
        res = regr.residues_
        if isinstance(res, list) or (not isinstance(res, list) and res < sor_threshold):
            valid_idx.append(idx)
        else:
            invalid_idx.append(idx)
            print('Invalid data idx: ', idx, ' residue: ', res)
    print('\nValid data length: ', len(valid_idx))
    return data[valid_idx, :cutoff_frame, :], sup_data[valid_idx, :cutoff_frame, :], invalid_idx


def featurewise_norm(X, featurewise_mean=None, featurewise_std=None, epsilon=1e-7, copy=True):
    """
    X shape: [batch_size, time_steps, input_dim]
    """
    if copy:
        import copy
        X = copy.deepcopy(X)
    if featurewise_mean is None:
        featurewise_mean = np.mean(X, axis=0)
    if featurewise_std is None:
        featurewise_std = np.std(X, axis=0, ddof=1)
    X = X - featurewise_mean
    X = X / (featurewise_std + epsilon)
    return X, featurewise_mean, featurewise_std


def featurewise_minmax(X, featurewise_min=None, featurewise_max=None, epsilon=1e-7, copy=True):
    """
    X shape: [batch_size, time_steps, input_dim]
    """
    if copy:
        import copy
        X = copy.deepcopy(X)
    if featurewise_min is None:
        featurewise_min = np.min(X, axis=0)
    if featurewise_max is None:
        featurewise_max = np.max(X, axis=0)
    X = (X - featurewise_min) / (featurewise_max - featurewise_min)
    X = X * 2 - 1 # scale to [-1, 1]
    return X, featurewise_min, featurewise_max


def samplewise_norm(X, epsilon=1e-7, copy=True):
    """
    X shape: [batch_size, time_steps, input_dim]
    """
    if copy:
        import copy
        X = copy.deepcopy(X)
    for i in xrange(X.shape[0]):
        X[i] = X[i] - np.mean(X[i], axis=0, keepdims=True)
        X[i] = X[i] / (np.std(X[i], axis=0, keepdims=True) + epsilon)
    return X


def get_weights(shape, W_init, regularizer_func, weight_decay):
    regularizer = regularizer_func(weight_decay)
    W = tf.get_variable('w', shape=shape, dtype=tf.float32,
                        initializer=W_init,
                        regularizer=regularizer,
                        trainable=True)
    return W


def get_biases(shape, b_init):
    b = tf.get_variable('b', shape=shape, dtype=tf.float32,
                        initializer=b_init,
                        trainable=True)
    return b

class BatchNormLayer(tl.layers.Layer):
    def __init__(self,
                 layer=None,
                 decay=0.9,
                 epsilon=0.00001,
                 act=tf.identity,
                 is_train=False,
                 name='batchnorm_layer',
                 ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  [TL] BatchNormLayer  %s: %s" % (self.name, act.__name__))
        with tf.variable_scope(name) as scope:
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            batch_norm_params = {'is_training': is_train,
                                 'center': True,
                                 'scale': True,
                                 'decay': decay,
                                 'activation_fn': act,
                                 'epsilon': epsilon}
            # vars = tf.trainable_variables()
            self.outputs = tf.contrib.slim.batch_norm(self.inputs, **batch_norm_params)
            new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            bn_vars = list(set(new_vars) - set(vars))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(bn_vars)


class LastXSeq(tl.layers.Layer):
    def __init__(self,
                 layer=None,
                 seq_length=1,
                 name='last_x_seq',
                 ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = self.inputs[:, -seq_length:, :]
        dim1, dim2, dim3 = self.inputs.get_shape().as_list()
        self.outputs = tf.reshape(self.outputs, (-1, seq_length, dim3))
        # dim1 = tf.shape(self.inputs)[0]
        # dim2 = tf.shape(self.inputs)[1]
        # dim3 = tf.shape(self.inputs)[2]
        # self.outputs = tf.slice(self.inputs, [0, dim2-seq_length, 0], [dim1, seq_length, dim3])

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class Highway(tl.layers.Layer):
    def __init__(self,
                 layer=None,
                 n_units=100,
                 act=tf.nn.elu,
                 W_init=tf.truncated_normal_initializer(stddev=0.1),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_args={},
                 b_init_args={},
                 name='highway',
                 ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")
        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  [TL] Highway  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as scope:
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args)
            b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
            W_T = tf.get_variable(name='W_T', shape=(n_in, n_units), initializer=W_init, **W_init_args)
            b_T = tf.get_variable(name='b_T', shape=(n_units), initializer=tf.constant_initializer(-2), **b_init_args)
            H = act(tf.matmul(self.inputs, W) + b)
            T = tf.sigmoid(tf.matmul(self.inputs, W_T) + b_T, name='transform_gate')
            C = tf.subtract(1.0, T, name='carry_gate')
            self.outputs = tf.add(tf.multiply(H, T), tf.multiply(self.inputs, C))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend([W, b, W_T, b_T])



def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    """Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batchsize.
    **The last batch contains all the inputs and targets remained**

    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=2, shuffle=False):
    >>>     print(batch)
    ... (array([['a', 'a'],
    ...        ['b', 'b']],
    ...         dtype='<U1'), array([0, 1]))
    ... (array([['c', 'c'],
    ...        ['d', 'd']],
    ...         dtype='<U1'), array([2, 3]))
    ... (array([['e', 'e'],
    ...        ['f', 'f']],
    ...         dtype='<U1'), array([4, 5]))
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    batch_indices = range(0, len(inputs) - batch_size + 1, batch_size)
    for start_idx in batch_indices:
        if start_idx == batch_indices[-1]:
            end = len(inputs)
        else:
            end = start_idx + batch_size
        if shuffle:
            excerpt = indices[start_idx:end]
        else:
            excerpt = slice(start_idx, end)
        yield inputs[excerpt], targets[excerpt]


def get_coordinates(args):
    data_dir = os.path.join(os.path.join(args.train_dir, 'data'), 'raw_data')
    force_folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, folder))]
    num_frames = 25
    xyzs = np.empty((0, num_frames, 3))
    labels = np.empty((0, 3))
    for force_folder in force_folders:
        trial_folders = [os.path.join(force_folder, folder)
                         for folder in os.listdir(force_folder)
                         if os.path.isdir(os.path.join(force_folder, folder))]
        for trial_folder in trial_folders:
            tmp_xyz = np.zeros((num_frames, 3))
            tmp_label = np.zeros((1, 3))
            xyz_dir = os.path.join(trial_folder, 'xyz')
            for idx in xrange(num_frames):
                xyz_file = os.path.join(xyz_dir, 'xyz-%d.txt'%idx)
                with open(xyz_file, 'r') as f:
                    data = json.load(f)
                    tmp_xyz[idx, 0] = data['xyz'][0]['x']
                    tmp_xyz[idx, 1] = data['xyz'][0]['y']
                    tmp_xyz[idx, 2] = data['xyz'][0]['z']
            label_dir = os.path.join(trial_folder, 'label')
            label_file = os.path.join(label_dir, os.listdir(label_dir)[0])
            with open(label_file, 'r') as f:
                data = json.load(f)
                tmp_label[0, 0] = data['xyz'][0]['x']
                tmp_label[0, 1] = data['xyz'][0]['y']
                tmp_label[0, 2] = data['xyz'][0]['z']
            xyzs = np.concatenate((xyzs, np.expand_dims(tmp_xyz, axis=0)), axis=0)
            labels = np.concatenate((labels, tmp_label), axis=0)
    return xyzs, labels


def get_left_right_center_pixel(args, restore=False, save=True):
    restore_dir = os.path.join(os.path.join(args.train_dir, 'data'), 'processed_data')
    center_pixel_dir = os.path.join(restore_dir, 'c_pixel')
    if not os.path.exists(center_pixel_dir):
        os.makedirs(center_pixel_dir)
    if restore:
        filename = os.path.join(center_pixel_dir, 'c_pixel_train.json')
        with open(filename, 'r') as f:
            tmp_dict = json.load(f)
            xys_train = np.asarray(tmp_dict['xys'])
            xyzs_train = np.asarray(tmp_dict['xyzs'])
        filename = os.path.join(center_pixel_dir, 'c_pixel_test.json')
        with open(filename, 'r') as f:
            tmp_dict = json.load(f)
            xys_test = np.asarray(tmp_dict['xys'])
            xyzs_test = np.asarray(tmp_dict['xyzs'])
    else:
        fetch_start_time = time.time()
        data_dir = os.path.join(os.path.join(args.train_dir, 'data'), 'raw_data')
        force_folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, folder))]
        num_frames = 58
        xys = np.empty((0, num_frames, 4)) # center pixel in left and right image
        xyzs = np.empty((0, num_frames, 3)) # cartesian coordinate of table tennis ball in the world

        example_total_num = 0
        for force_folder in force_folders:
            h_folder_list = os.listdir(force_folder)
            h_folder_list = [folder for folder in h_folder_list
                             if os.path.isdir(os.path.join(force_folder, folder))]
            example_total_num += len(h_folder_list)
        example_processed = 0

        for force_folder in force_folders:
            trial_folders = [os.path.join(force_folder, folder)
                             for folder in os.listdir(force_folder)
                             if os.path.isdir(os.path.join(force_folder, folder))]
            for trial_folder in trial_folders:
                start = time.time()
                tmp_center_pixels, tmp_coord = get_center_pixel(trial_folder, num_frames)
                tmp_xy = np.asarray(tmp_center_pixels)
                tmp_xyz = np.asarray(tmp_coord)
                xys = np.concatenate((xys, np.expand_dims(tmp_xy, axis=0)), axis=0)
                xyzs = np.concatenate((xyzs, np.expand_dims(tmp_xyz, axis=0)), axis=0)

                example_processed += 1
                end = time.time()
                time_elapsed = end - start
                total_time_elapsed = end - fetch_start_time
                finished_percentage = example_processed / float(example_total_num)
                print('Examples finished: %.2f %%, example processed time: %.2f s,'
                      ' total time elapsed %.2f s' %
                      (finished_percentage * 100.0, time_elapsed, total_time_elapsed))
        if save:
            from sklearn.model_selection import train_test_split
            xys_train, xys_test, xyzs_train, xyzs_test = train_test_split(xys,
                                                                          xyzs,
                                                                          test_size=0.1,
                                                                          random_state=args.random_seed)
            filename = os.path.join(center_pixel_dir, 'c_pixel_train.json')
            with open(filename, 'w+') as f:
                tmp_dict = {}
                tmp_dict['xys'] = xys_train.tolist()
                tmp_dict['xyzs'] = xyzs_train.tolist()
                json.dump(tmp_dict, f, indent=4)
            filename = os.path.join(center_pixel_dir, 'c_pixel_test.json')
            with open(filename, 'w+') as f:
                tmp_dict = {}
                tmp_dict['xys'] = xys_test.tolist()
                tmp_dict['xyzs'] = xyzs_test.tolist()
                json.dump(tmp_dict, f, indent=4)
        fetch_end_time = time.time()
        fetch_time_elapsed = fetch_end_time - fetch_start_time
        print("\nSummary: fetching center pixels take %.2f s" % fetch_time_elapsed)
    print("Loading center pixels done...")
    return xys_train, xys_test, xyzs_train, xyzs_test


def get_center_pixel(trial_folder, num_images):
    left_image_folder = os.path.join(trial_folder, 'left')
    right_image_folder = os.path.join(trial_folder, 'right')
    xyz_folder = os.path.join(trial_folder, 'xyz')
    center_pixels = []
    xyzs = []
    left_fgbg = cv2.BackgroundSubtractorMOG()
    right_fgbg = cv2.BackgroundSubtractorMOG()
    left_images_list = os.listdir(left_image_folder)
    left_images_list = [image for image in left_images_list if image.endswith('.jpg')]
    assert len(left_images_list) > num_images, 'Cannot fetch %d images in %s' % (num_images, left_images_list)
    idx = 0
    left_bkg_image = '%s-%d.jpg' % ('left', len(left_images_list) - 1)
    right_bkg_image = '%s-%d.jpg' % ('right', len(left_images_list) - 1)
    left_image = cv2.imread(os.path.join(left_image_folder, left_bkg_image))
    right_image = cv2.imread(os.path.join(right_image_folder, right_bkg_image))
    left_fgmask = left_fgbg.apply(left_image)
    right_fgmask = right_fgbg.apply(right_image)
    while True:
        left_image_name = '%s-%d.jpg' % ('left', idx)
        right_image_name = '%s-%d.jpg' % ('right', idx)
        xyz_filename = '%s-%d.txt' % ('xyz', idx)
        left_image = cv2.imread(os.path.join(left_image_folder, left_image_name))
        right_image = cv2.imread(os.path.join(right_image_folder, right_image_name))
        xyz_file = os.path.join(xyz_folder, xyz_filename)
        left_fgmask = left_fgbg.apply(left_image)
        right_fgmask = right_fgbg.apply(right_image)
        left_contours, left_hierarchy = cv2.findContours(left_fgmask,
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)
        right_contours, right_hierarchy = cv2.findContours(right_fgmask,
                                                           cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)

        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        left_xy_radius = []
        right_xy_radius = []
        for contour in left_contours:
            if contour.shape[0] < 3:
                continue
            # M = cv2.moments(contour)
            # cx = int(M['m10'] / M['m00'])
            # cy = int(M['m01'] / M['m00'])
            (x, y), radius = cv2.minEnclosingCircle(contour)
            left_xy_radius.append((int(x), int(y), radius))
            # xy_radius.append((cx, cy, radius))
        for contour in right_contours:
            if contour.shape[0] < 3:
                continue
            # M = cv2.moments(contour)
            # cx = int(M['m10'] / M['m00'])
            # cy = int(M['m01'] / M['m00'])
            (x, y), radius = cv2.minEnclosingCircle(contour)
            right_xy_radius.append((int(x), int(y), radius))
            # xy_radius.append((cx, cy, radius))
        if not left_xy_radius:
            left_x = 0
            left_y = 0
            left_radius = 0
        else:
            left_xy_radius = sorted(left_xy_radius, key=lambda tup: tup[0])
            left_x, left_y, left_radius = left_xy_radius[-1]
        if not right_xy_radius:
            right_x = 0
            right_y = 0
            right_radius = 0
        else:
            right_xy_radius = sorted(right_xy_radius, key=lambda tup: tup[0])
            right_x, right_y, right_radius = right_xy_radius[-1]
        # cv2.circle(left_image, (left_x, left_y), int(left_radius), (0, 255, 0), 5)
        # cv2.circle(right_image, (right_x, right_y), int(right_radius), (0, 255, 0), 5)
        # cv2.imshow('left_ori', left_image)
        # cv2.imshow('right_ori', right_image)
        # cv2.waitKey(6000)

        center_pixels.append([left_x, left_y, right_x, right_y])
        tmp_xyz = []
        with open(xyz_file, 'r') as f:
            data = json.load(f)
            tmp_xyz.append(data['xyz'][0]['x'])
            tmp_xyz.append(data['xyz'][0]['y'])
            tmp_xyz.append(data['xyz'][0]['z'])
        xyzs.append(tmp_xyz)
        idx += 1
        if idx == num_images:
            break
        assert len(left_images_list) > idx, 'Cannot fetch %d images in %s' % (num_images, left_image_folder)
        # cv2.destroyAllWindows()

    return center_pixels, xyzs


def read_from_tfrecords(args, test=False):
    data_dir = os.path.join(os.path.join(args.train_dir, 'data'), 'raw_data')
    tfrecords_dir = os.path.join(data_dir, 'tfrecords')
    if not test:
        record_dir = os.path.join(tfrecords_dir, 'train')
    else:
        record_dir = os.path.join(tfrecords_dir, 'test')
    record_files_op = tf.train.match_filenames_once('%s/*.tfrecords'%(record_dir))
    filename_queue = tf.train.string_input_producer(record_files_op)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'time_steps': tf.FixedLenFeature([], tf.int64),
                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'channels': tf.FixedLenFeature([], tf.int64),
                                                 'label': tf.FixedLenFeature([], tf.string),
                                                 'xyzs': tf.FixedLenFeature([], tf.string),
                                                 'images_left': tf.FixedLenFeature([], tf.string),
                                                 'images_right': tf.FixedLenFeature([], tf.string),
                                                 })

    time_steps = tf.cast(features['time_steps'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)
    images_left = tf.decode_raw(features['images_left'], tf.uint8)
    images_right = tf.decode_raw(features['images_right'], tf.uint8)
    images_left = tf.reshape(images_left, [time_steps, height, width, channels])
    images_right = tf.reshape(images_right, [time_steps, height, width, channels])
    xyzs = tf.decode_raw(features['xyzs', tf.float32])
    xyzs = tf.reshape(xyzs, [time_steps, -1])
    label = tf.cast(features['label'], tf.float32)
    min_after_dequeue = 600
    capacity = min_after_dequeue + 3 * args.batch_size
    images_left_batch, images_left_batch, xyzs_batch, label_batch \
        = tf.train.shuffle_batch([images_left, images_right, xyzs, label],
                                 batch_size=args.batch_size,
                                 capacity=capacity,
                                 min_after_dequeue=min_after_dequeue)
    return images_left_batch, images_left_batch, xyzs_batch, label_batch

def find_best_ckpt(args, model, X_train, y_train, X_test, y_test, restore=False):
    ckpt = tf.train.get_checkpoint_state(model.ckpt_dir)
    max_ckpt = ckpt.model_checkpoint_path
    max_ckpt_idx = int(max_ckpt.split('-')[-1])
    error_filename = os.path.join(model.ckpt_dir, 'error_stats.csv')
    files_lst = os.listdir(model.ckpt_dir)
    min_index = max_ckpt_idx
    for filename in files_lst:
        if 'tfevents' in filename or 'ckpt' not in filename:
            continue
        fileindex_list = re.findall(r'\d+', filename)
        if not fileindex_list:
            continue
        fileindex = int(fileindex_list[0])
        if fileindex <= min_index:
            min_index = fileindex
    start = 16500#min_index
    if not restore:
        if os.path.exists(error_filename):
            os.remove(error_filename)
        for step in xrange(start, max_ckpt_idx + 1, args.ckpt_interval):
            model.restore_model(step=step)
            print("\nTraining data testing, Ckpt %d\n---------------------" % step)
            train_y_preds, _ = model.test(X_train,
                                          y_train,
                                          name='Train%d' % step,
                                          draw=False,
                                          save_to_file=True)
            print("\nTesting data testing, Ckpt %d\n---------------------" % step)
            test_y_preds, _ = model.test(X_test,
                                         y_test,
                                         name='Test%d' % step,
                                         draw=False,
                                         save_to_file=True)

    df = pd.read_csv(error_filename, header=None)
    train_errors = df.values[::2]
    test_errors = df.values[1::2]
    x_ticks = range(start,
                    start + args.ckpt_interval * train_errors.shape[0],
                    args.ckpt_interval)
    max_idx = np.argmax(test_errors, axis=0)
    print('\n---------------------\nStatistics:')
    for idx in xrange(max_idx.size):
        index = max_idx[idx]
        print(' %d Threshold max: step = %d' % (idx + 1, start + index * args.ckpt_interval))
        print('       %.2f %%    %.2f %%    %.2f %%' % (test_errors[index, 0], test_errors[index, 1], test_errors[index, 2]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_ticks, train_errors[:, 2], 'r-', label='train')
    ax.plot(x_ticks, test_errors[:, 2], 'g-', label='test')
    ax.legend(ncol=1, prop={'size': 12})
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    plt.title('Percentage of cases with error less than 3cm')








