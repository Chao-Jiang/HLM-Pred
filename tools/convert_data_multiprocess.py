import numpy as np
import json
import cv2
import os
import time
import argparse
from multiprocessing import Pool

def chunks(l, n):
    """Separate elements in list l into n chunks"""
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def convert_data_multiprocessing(args, process_index=0):
    data_dir = os.path.join(args.train_dir, 'raw_data')
    restore_dir = os.path.join(os.path.dirname(data_dir), 'processed_data')
    center_pixel_dir = os.path.join(restore_dir, 'c_pixel')
    if not os.path.exists(center_pixel_dir):
        os.makedirs(center_pixel_dir)
    f_folders = os.listdir(data_dir)
    f_folders = [folder for folder in f_folders if os.path.isdir(os.path.join(data_dir, folder))]
    f_folders_parallel = list(chunks(f_folders, int(np.ceil(len(f_folders) / float(args.num_parallel)))))
    try:
        p_start = time.time()
        pass_args = [(thread_index,
                 f_folders_parallel[thread_index],
                 data_dir,
                 process_index) for thread_index in xrange(args.num_parallel)]
        pool = Pool(processes=args.num_parallel)
        pool.map_async(get_left_right_center_pixel_wrapper, pass_args).get(0xFFFF)
        pool.close()
        pool.join()
        p_end = time.time()
        print(' Program finished, total elapsed time: %.2f s' % (p_end - p_start))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


def get_left_right_center_pixel_wrapper(args):
    xys, xyzs = get_left_right_center_pixel(*args)
    return xys, xyzs


def get_left_right_center_pixel(thread_index, f_folder_list, data_dir, process_index):
    restore_dir = os.path.join(os.path.dirname(data_dir), 'processed_data')
    center_pixel_dir = os.path.join(restore_dir, 'c_pixel')
    fetch_start_time = time.time()
    force_folders = [os.path.join(data_dir, folder) for folder in f_folder_list
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
            print('\nPID %s Examples finished: %.2f %%, example processed time: %.2f s,'
                  ' total time elapsed %.2f s' %
                  (os.getpid(), finished_percentage * 100.0, time_elapsed, total_time_elapsed))

    filename = os.path.join(center_pixel_dir, 'c_pixel_%d_%d.json'%(process_index, thread_index))
    with open(filename, 'w') as f:
        tmp_dict = {}
        tmp_dict['xys'] = xys.tolist()
        tmp_dict['xyzs'] = xyzs.tolist()
        json.dump(tmp_dict, f, indent=4)
    fetch_end_time = time.time()
    fetch_time_elapsed = fetch_end_time - fetch_start_time
    print("\nPID %s Summary: fetching center pixels take %.2f s" % (os.getpid(), fetch_time_elapsed))
    print("\nPID %s Loading center pixels done..." % os.getpid())
    return xys, xyzs


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
        xyz_filename = '%s-%d.json' % ('xyz', idx)
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
            (x, y), radius = cv2.minEnclosingCircle(contour)
            left_xy_radius.append((int(x), int(y), radius))
        for contour in right_contours:
            if contour.shape[0] < 3:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            right_xy_radius.append((int(x), int(y), radius))
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

    return center_pixels, xyzs

if __name__ == "__main__":
    process_index = 0
    parser = argparse.ArgumentParser(description='Run the Table Tennis Ball Prediction algorithm.')
    parser.add_argument('--train_dir', type=str, default='./',
                        help='path to the training directory.')
    parser.add_argument(
        '--num_parallel',
        type=int,
        default=2,
        help='number of parallel process to convert data into tfrecords'
    )
    args = parser.parse_args()
    convert_data_multiprocessing(args, process_index=process_index)