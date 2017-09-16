#!/usr/bin/env python
# -----------------------------------
# separate the training and testing data and convert images to tfrecords
# Author: Tao Chen
# Date: 2016.10.16
# -----------------------------------

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import collections
import tensorflow as tf
import shutil
import numpy as np
import time
import threading
import json
from PIL import Image
from multiprocessing import Pool
Dataset = collections.namedtuple('Dataset', ['left_images_path', 'right_images_path', 'labels_path'])
FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def chunks(l, n):
    """Separate elements in list l into n chunks"""
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def convert_data_sets(directory):
    f_folders = os.listdir(directory)
    f_folders = [folder for folder in f_folders if os.path.isdir(os.path.join(directory, folder))]
    f_folders_parallel = list(chunks(f_folders, int(np.ceil(len(f_folders) / float(FLAGS.num_parallel)))))

    tfrecords_folder = os.path.join(FLAGS.directory, 'tfrecords')
    tfrecords_train_folder = os.path.join(tfrecords_folder, 'train')
    tfrecords_test_folder = os.path.join(tfrecords_folder, 'test')
    if os.path.exists(tfrecords_train_folder):
        shutil.rmtree(tfrecords_train_folder)
    os.makedirs(tfrecords_train_folder)
    if os.path.exists(tfrecords_test_folder):
        shutil.rmtree(tfrecords_test_folder)
    os.makedirs(tfrecords_test_folder)
    if FLAGS.multiprocessing:
        try:
            args = [(thread_index,
                     f_folders_parallel[thread_index],
                     directory,
                     tfrecords_train_folder,
                     tfrecords_test_folder) for thread_index in xrange(FLAGS.num_parallel)]
            pool = Pool(processes=FLAGS.num_parallel)
            pool.map_async(process_image_files_batch_wrapper, args).get(0xFFFF)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
    else:
        coord = tf.train.Coordinator()
        threads = []
        for thread_index in xrange(FLAGS.num_parallel):
            args = (thread_index,
                    f_folders_parallel[thread_index],
                    directory,
                    tfrecords_train_folder,
                    tfrecords_test_folder)
            t = threading.Thread(target=process_image_files_batch, args=args)
            t.start()
            threads.append(t)
        # Wait for all the threads to terminate.
        coord.join(threads)

def process_image_files_batch_wrapper(args):
    return process_image_files_batch(*args)

def process_image_files_batch(thread_index, f_folder_list, root_directory, tfrecords_train_folder, tfrecords_test_folder):
    train_output_filename = '%s-%.5d%s' % ('train', thread_index, '.tfrecords')
    test_output_filename = '%s-%.5d%s' % ('test', thread_index, '.tfrecords')
    if FLAGS.compression_option == 'gzip':
        cmp_type = tf.python_io.TFRecordCompressionType.GZIP
    elif FLAGS.compression_option == 'zlib':
        cmp_type = tf.python_io.TFRecordCompressionType.ZLIB
    elif FLAGS.compression_option == 'None':
        cmp_type = tf.python_io.TFRecordCompressionType.NONE
    else:
        raise ValueError("Unrecognized compression type %s"%FLAGS.compression_option)
    tfrecord_option = tf.python_io.TFRecordOptions(cmp_type)
    train_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_train_folder, train_output_filename),
                                               options=tfrecord_option)
    test_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_test_folder, test_output_filename),
                                              options=tfrecord_option)

    example_total_num = 0
    for f_folder in f_folder_list:
        f_folder_path = os.path.join(root_directory, f_folder)
        h_folder_list = os.listdir(f_folder_path)
        h_folder_list = [folder for folder in h_folder_list
                         if os.path.isdir(os.path.join(f_folder_path, folder))]
        example_total_num += len(h_folder_list)
    example_processed = 0
    for f_folder in f_folder_list:
        f_folder_path = os.path.join(root_directory, f_folder)
        h_folder_list = os.listdir(f_folder_path)
        h_folder_list = [folder for folder in h_folder_list
                         if os.path.isdir(os.path.join(f_folder_path, folder))]
        num_train = int(len(h_folder_list) * FLAGS.train_ratio)
        for index, h_folder in enumerate(h_folder_list):
            start = time.time()
            h_folder_path = os.path.join(f_folder_path, h_folder)
            left_folder = os.path.join(h_folder_path, 'left')
            right_folder = os.path.join(h_folder_path, 'right')
            xyzs_folder = os.path.join(h_folder_path, 'xyz')
            left_image_list = os.listdir(left_folder)
            left_images = np.empty((0, 480, 640, 3))
            right_images = np.empty((0, 480, 640, 3))
            xyzs = np.empty((0, 3))
            for left_image_file in left_image_list:
                if not left_image_file.endswith('.jpg'):
                    continue
                right_image_file = left_image_file.replace('left', 'right')
                xyz_file = left_image_file.replace('left', 'xyz')
                xyz_file = os.path.join(xyzs_folder, xyz_file.replace('jpg', 'txt'))
                left_image = np.asarray(Image.open(os.path.join(left_folder, left_image_file)))
                right_image = np.asarray(Image.open(os.path.join(right_folder, right_image_file)))
                xyz = np.zeros((1, 3))
                with open(xyz_file, 'r') as f:
                    data = json.load(f)
                    xyz[0, 0] = data['xyz'][0]['x']
                    xyz[0, 1] = data['xyz'][0]['y']
                    xyz[0, 2] = data['xyz'][0]['z']
                left_images = np.concatenate((left_images, np.expand_dims(left_image, axis=0)), axis=0)
                right_images = np.concatenate((right_images, np.expand_dims(right_image, axis=0)), axis=0)
                xyzs = np.concatenate((xyzs, xyz), axis=0)
            label_folder = os.path.join(h_folder_path, 'label')
            label_folder_files = [file for file in os.listdir(label_folder) if file.endswith('.txt')]
            label_file = os.path.join(label_folder, label_folder_files[0])
            label = np.zeros(3)
            with open(label_file, 'r') as f:
                data = json.load(f)
                label[0] = data['xyz'][0]['x']
                label[1] = data['xyz'][0]['y']
                label[2] = data['xyz'][0]['z']
            time_steps = left_images.shape[0]
            height = left_images.shape[1]
            width = left_images.shape[2]
            channels = left_images.shape[3]
            left_images_string = left_images.tostring()
            right_images_string = right_images.tostring()
            label_string = label.tostring()
            xyzs_string = xyzs.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'time_steps': _int64_feature(time_steps),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'label': _bytes_feature(label_string),
                'xyzs': _bytes_feature(xyzs_string),
                'images_left': _bytes_feature(left_images_string),
                'images_right': _bytes_feature(right_images_string)}))
            if index < num_train:
                train_writer.write(example.SerializeToString())
            else:
                test_writer.write(example.SerializeToString())
            example_processed += 1
            end = time.time()
            time_elapsed = end - start
            finished_percentage = example_processed / float(example_total_num)
            print('\nPID %s Processing %s/%s, examples finished: %.2f %%, example processed time: %.2f s' %
                  (os.getpid(), f_folder, h_folder, finished_percentage * 100.0, time_elapsed))
    train_writer.close()
    test_writer.close()
    print('\nPID %s finished...'%os.getpid())


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return float(total_size) / 1024.0 ** 3

def main(unused_argv):
    start_time = time.time()
    convert_data_sets(FLAGS.directory)
    end_time = time.time()
    print('elapsed time:', end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='../train/data/raw_data',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--num_parallel',
        type=int,
        default=4,
        help='number of parallel process to convert data into tfrecords'
    )
    parser.add_argument(
        '--multiprocessing',
        '-mp',
        type=bool,
        default='True',
        help='use multiprocessing by default, use multithreading otherwise'
    )
    parser.add_argument(
        '--compression_option',
        type=str,
        choices=['None', 'gzip', 'zlib'],
        default='gzip',
        help='type of compression used in tfrecord writer: \'None\', \'gzip\', or \'zlib\''
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default='0.8',
        help='percentage of data that is used for training'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
