import os
from scripts import utility
import numpy as np
import json
import cv2
import os
import time
import cPickle
import argparse
from sklearn.model_selection import train_test_split

def create_train_test(args):
    restore_dir = os.path.join(args.train_dir, 'processed_data')
    center_pixel_dir = os.path.join(restore_dir, 'c_pixel')
    train_test_dir = os.path.join(center_pixel_dir, 'train_test')
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)
    c_pixel_files = os.listdir(center_pixel_dir)
    c_pixel_files = [c_file for c_file in c_pixel_files if c_file.endswith('.json')]
    num_frames = 58
    traj_repeat_num = 50
    xys = np.empty((0, num_frames, 4))  # center pixel in left and right image
    xyzs = np.empty((0, num_frames, 3))
    for filename in c_pixel_files:
        filename = os.path.join(center_pixel_dir, filename)
        with open(filename, 'r') as f:
            tmp_dict = json.load(f)
            xys = np.concatenate((xys, np.asarray(tmp_dict['xys'])), axis=0)
            xyzs = np.concatenate((xyzs, np.asarray(tmp_dict['xyzs'])), axis=0)

    filename = os.path.join(train_test_dir, 'c_pixel_with_outliers.cpickle')
    with open(filename, 'w+') as f:
        tmp_dict = {}
        tmp_dict['xys'] = xys.tolist()
        tmp_dict['xyzs'] = xyzs.tolist()
        cPickle.dump(tmp_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)

    filted_xys = np.empty((0, num_frames, 4))  # center pixel in left and right image
    filted_xyzs = np.empty((0, num_frames, 3))
    invalid_idx = []
    for idx in xrange(int(xyzs.shape[0] / traj_repeat_num)):
        indices = range(idx * traj_repeat_num, idx * traj_repeat_num + traj_repeat_num)
        tmp_xyzs = xyzs[indices]
        tmp_xys = xys[indices]
        xyzs_median = np.median(tmp_xyzs, axis=0)
        xyzs_dist = np.linalg.norm(tmp_xyzs[:, num_frames - 6: num_frames, :]
                                   - xyzs_median[num_frames - 6: num_frames, :],
                                   axis=2)
        xyzs_dist = np.mean(xyzs_dist, axis=1)
        threshold = 0.09
        valid_indices = xyzs_dist < threshold
        invalid_idx.extend((np.where(xyzs_dist >= threshold)[0] + idx * traj_repeat_num).tolist())
        filted_xys = np.concatenate((filted_xys, tmp_xys[valid_indices]), axis=0)
        filted_xyzs = np.concatenate((filted_xyzs, tmp_xyzs[valid_indices]), axis=0)

    filted_xyzs, filted_xys, invalid_idx_sup = utility.remove_outliers(data=filted_xyzs,
                                                                       sup_data=filted_xys,
                                                                       cutoff_frame=50,
                                                                       sor_threshold=0.005)
    xys_train, xys_test, xyzs_train, xyzs_test = train_test_split(filted_xys,
                                                                  filted_xyzs,
                                                                  test_size=0.1,
                                                                  random_state=args.random_seed)
    invalid_idx.extend(invalid_idx_sup)
    # invalid_idx = invalid_idx_sup
    filename = os.path.join(train_test_dir, 'invalid_idx.json')
    with open(filename, 'w+') as f:
        json.dump(invalid_idx, f, indent=4)
    filename = os.path.join(train_test_dir, 'c_pixel_train.cpickle')
    with open(filename, 'w+') as f:
        tmp_dict = {}
        tmp_dict['xys'] = xys_train.tolist()
        tmp_dict['xyzs'] = xyzs_train.tolist()
        cPickle.dump(tmp_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
    filename = os.path.join(train_test_dir, 'c_pixel_test.cpickle')
    with open(filename, 'w+') as f:
        tmp_dict = {}
        tmp_dict['xys'] = xys_test.tolist()
        tmp_dict['xyzs'] = xyzs_test.tolist()
        cPickle.dump(tmp_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print('Training data: ', xys_train.shape[0])
    print('Testing data: ', xys_test.shape[0])
    print('Invalid data: ', len(invalid_idx))
    print('Seperating train and test data done...')

def draw_invalid_traj(args):
    restore_dir = os.path.join(args.train_dir, 'processed_data')
    center_pixel_dir = os.path.join(restore_dir, 'c_pixel')
    train_test_dir = os.path.join(center_pixel_dir, 'train_test')
    filename = os.path.join(train_test_dir, 'invalid_idx.json')
    with open(filename, 'r') as f:
        invalid_idx = json.load(f)
    print('Number of invalid traj: ', len(invalid_idx))
    filename = os.path.join(train_test_dir, 'c_pixel_with_outliers.cpickle')
    with open(filename, 'r') as f:
        tmp_dict = cPickle.load(f)
        xys = np.asarray(tmp_dict['xys'])
        xyzs = np.asarray(tmp_dict['xyzs'])
    for idx in invalid_idx:
        # utility.plot_3d([xyzs[idx:idx + 1]], draw_now=False)
        utility.plot_3d([xyzs[idx:idx + 5, :50, :]], draw_now=True)

def draw_valid_traj(args):
    traj_repeat_num = 50
    restore_dir = os.path.join(args.train_dir, 'processed_data')
    center_pixel_dir = os.path.join(restore_dir, 'c_pixel')
    train_test_dir = os.path.join(center_pixel_dir, 'train_test')
    filename = os.path.join(train_test_dir, 'invalid_idx.json')
    with open(filename, 'r') as f:
        invalid_idx = json.load(f)
    print('Number of invalid traj: ', len(invalid_idx))
    filename = os.path.join(train_test_dir, 'c_pixel_with_outliers.cpickle')
    with open(filename, 'r') as f:
        tmp_dict = cPickle.load(f)
        xys = np.asarray(tmp_dict['xys'])
        xyzs = np.asarray(tmp_dict['xyzs'])
    for idx in xrange(6, int(xyzs.shape[0] / traj_repeat_num)):
        if idx * traj_repeat_num in invalid_idx:
            continue
        # utility.plot_3d([xyzs[idx:idx + 1]], draw_now=False)
        # a = xyzs[idx * 100]
        # z = xyzs[idx * 100 + 2, 50 - 5: 50, 2]
        # print('===y:')
        # for i in xrange(a.shape[0]):
        #     print a[i, 1] , ','
        # print('===z:')
        # for i in xrange(a.shape[0]):
        #     print a[i, 2] , ','
        # var = np.var(z)
        # print('var: ',var)
        utility.plot_3d([xyzs[idx * traj_repeat_num: idx * traj_repeat_num + traj_repeat_num, :50, :]], draw_now=False, title='%d'%idx)
        indices = range(idx * traj_repeat_num, idx * traj_repeat_num + traj_repeat_num)
        tmp_xyzs = xyzs[indices]
        tmp_xys = xys[indices]
        xyzs_median = np.median(tmp_xyzs, axis=0)
        xyzs_dist = np.linalg.norm(tmp_xyzs[:, 44: 50, :] - xyzs_median[44: 50, :], axis=2)
        xyzs_dist = np.mean(xyzs_dist, axis=1)
        np.set_printoptions(precision=2, suppress=True)
        print xyzs_dist
        valid_indices = xyzs_dist < 0.09
        filted_xys =tmp_xys[valid_indices]
        filted_xyzs = tmp_xyzs[valid_indices]
        print('filter: ', filted_xys.shape[0])
        utility.plot_3d([filted_xyzs[:, :50, :]], draw_now=True, title='filted%d' % idx)

        # while True:
        #     key = raw_input('Do you think it\' valid: y/n ?')
        #     if key == 'y':
        #         break
        #     elif key == 'n':
        #         invalid_idx.extend(range(idx * 100, idx * 100 + 100))
        #         break
        #     else:
        #         continue
    filename = os.path.join(train_test_dir, 'invalid_idx_more.json')
    with open(filename, 'w+') as f:
        json.dump(invalid_idx, f, indent=4)
    print('Number of invalid traj: ', len(invalid_idx))



if __name__ == "__main__":
    process_index = 0
    parser = argparse.ArgumentParser(description='Run the Table Tennis Ball Prediction algorithm.')
    parser.add_argument('--train_dir', type=str, default='./../train/data',
                        help='path to the training directory.')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    create_train_test(args)
    # draw_invalid_traj(args)
    # draw_valid_traj(args)