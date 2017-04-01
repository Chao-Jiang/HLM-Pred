import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from datetime import datetime
from datetime import timedelta
import Queue
import os
import time
import json
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, \
    SetModelState, SetModelStateRequest, ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import Wrench, Pose
import cv2
import collections
from cv_bridge import CvBridge, CvBridgeError
import cv_bridge
import threading
from sensor_msgs.msg import Image
import message_filters
from scripts import utility
# import utility

FrameData = collections.namedtuple('FrameData', 'left, right, ball_pos')
PredictionData = collections.namedtuple('PredictionData', 'position, frame')

class RepeatedTimer(object):
  def __init__(self, interval, function, *args, **kwargs):
    self._timer = None
    self.interval = interval
    self.function = function
    self.args = args
    self.kwargs = kwargs
    self.is_running = False
    self.next_call = time.time()
    self.start()

  def _run(self):
    self.is_running = False
    self.start()
    self.function(*self.args, **self.kwargs)

  def start(self):
    if not self.is_running:
      self.next_call += self.interval
      self._timer = threading.Timer(self.next_call - time.time(), self._run)
      self._timer.start()
      self.is_running = True

  def stop(self):
    self._timer.cancel()
    self.is_running = False


class SaveFrame:
    def __init__(self, camera):
        self.camera = camera
        self.image_frequency = 30.0
        self.image_period = 1 / float(self.image_frequency)
        self.image_ROSRate = rospy.Rate(self.image_frequency)
        self.px = 0.7
        self.py = -0.5
        self.fx = 0.0
        self.fy = 0.04
        self.fz = 0.10
        self.frames_data = Queue.Queue()
        self.data_dir='./raw_data/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.offline_action_filename = 'plan_point.json'

    def subscribe_frames_data(self):
        meta_data_one_frame = self.camera.get_camera_data()
        invalid_num = 0
        if meta_data_one_frame is None:
            self.frames_data = Queue.Queue()
            print('!!!!!Warning: Received None from camera, reseting the queue...')
            invalid_num += 1
            if invalid_num > 0:
                self.relaunch_ball = True
        else:
            self.frames_data.put(meta_data_one_frame)
        # self.end = time.time()
        # print '\ntime: ', self.end-self.start, ' hz:', 1/float(self.end - self.start)
        # self.start = self.end

    def acquire_frames_data(self, num_images=14):
        center_pixels = []
        # ball_pos = []
        # self.start = time.time()
        discard_num = 1
        self.relaunch_ball = False
        self.acquire_repeated_timer = RepeatedTimer(self.image_period, self.subscribe_frames_data)
        # self.acquire_repeated_timer.start()
        while self.frames_data.qsize() < num_images + discard_num:
            if self.relaunch_ball:
                self.relaunch_ball = False
                self.acquire_repeated_timer.stop()
                print('Relaunching ball...')
                # self.reset()
                time.sleep(1)
                self.set_ball_position(px=self.px, py=self.py)
                self.hit_ball(fx=self.fx, fy=self.fy, fz=self.fz, dt=0.15)
                self.acquire_repeated_timer = RepeatedTimer(self.image_period, self.subscribe_frames_data)
            pass
        self.acquire_repeated_timer.stop()
        meta_data = []
        while not self.frames_data.empty():
            meta_data_one_frame = self.frames_data.get()
            meta_data.append(meta_data_one_frame)
        return meta_data

    def save_data(self, fx, fy, fz, meta_data, f_idx, h_idx):
        ball_poses = []
        f_folder = os.path.join(self.data_dir, 'f%d'%f_idx)
        h_folder = os.path.join(f_folder, 'h%d'%h_idx)
        left_folder = os.path.join(h_folder, 'left')
        right_folder = os.path.join(h_folder, 'right')
        xyz_folder = os.path.join(h_folder, 'xyz')
        for folder in [left_folder, right_folder, xyz_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        f_file = os.path.join(f_folder, 'force.txt')
        f_data = {'fx': fx, 'fy': fy, 'fz': fz}
        with open(f_file, 'w+') as f:
            json.dump(f_data, f, indent=4)

        for idx, meta_data_one_frame in enumerate(meta_data):
            left_image = meta_data_one_frame.left
            right_image = meta_data_one_frame.right
            ball_pos = meta_data_one_frame.ball_pos
            ball_poses.append(ball_pos)
            left_image_filename = os.path.join(left_folder, 'left-%d.jpg'%idx)
            right_image_filename = os.path.join(right_folder, 'right-%d.jpg' % idx)
            xyz_filename = os.path.join(xyz_folder, 'xyz-%d.json'%idx)
            cv2.imwrite(left_image_filename, left_image)
            cv2.imwrite(right_image_filename, right_image)
            xyz_data = {}
            xyz_data['xyz'] = []
            xyz_data['xyz'].append({
                'x': ball_pos[0],
                'y': ball_pos[1],
                'z': ball_pos[2],
            })
            with open(xyz_filename, 'w+') as f:
                json.dump(xyz_data, f, indent=4)
        return ball_poses

    def set_ball_position(self, px=0.7, py=-0.5, pz=0.85):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.px = px
            self.py = py
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_position_re = SetModelStateRequest()
            pose = Pose()
            pose.position.x = px
            pose.position.y = py
            pose.position.z = pz
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = np.sin(np.pi / 2)
            pose.orientation.z = np.sin(np.pi / 2)
            pose.orientation.z = np.sin(np.pi / 2)
            pose.orientation.w = np.cos(np.pi / 2)
            set_position_re.model_state.model_name = 'Ping Pong Ball'
            set_position_re.model_state.pose = pose
            set_state(set_position_re)
            time.sleep(3)
        except rospy.ServiceException, e:
            raise "Service call failed: %s" % e


    def hit_ball(self, fx=0.0, fy=0.15, fz=0.0, dt=0.1):
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        try:
            self.fx = fx
            self.fy = fy
            self.fz = fz
            apply_force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
            apply_force_re = ApplyBodyWrenchRequest()
            apply_force_re.body_name = 'Ping Pong Ball::ball'
            wrench = Wrench()
            wrench.force.x = fx
            wrench.force.y = fy
            wrench.force.z = fz
            wrench.torque.x = 0
            wrench.torque.y = 0
            wrench.torque.z = 0
            apply_force_re.wrench = wrench
            apply_force_re.start_time = rospy.get_rostime()
            apply_force_re.duration = rospy.Duration(dt)
            apply_force(apply_force_re)
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def generate_force(self, bounds, num):
        yz = np.mgrid[bounds[0][0]: bounds[0][1]: complex(num[0]),
              bounds[1][0]: bounds[1][1]: complex(num[1])].reshape(2, -1).T
        for i in range(yz.shape[1]):
            yz[:, i] = yz[:, i] if num[i] > 1 else bounds.mean(axis=1)[i]
        return yz


class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self._br = cv_bridge.CvBridge()
        self._cameraRight_img = None
        self._cameraLeft_img = None
        self._ball_pos = None
        self._meta_data = None
        self._observation_stale = True
        self.camera_l_sub = message_filters.Subscriber("/multi/camera/basler/left/image_raw",
                                                       Image)
        self.camera_r_sub = message_filters.Subscriber("/multi/camera/basler/right/image_raw",
                                                       Image)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.camera_l_sub, self.camera_r_sub],
                                                                queue_size=1, slop=0.3)

        self.sync.registerCallback(self.sync_callback)
        time.sleep(0.1)


    def sync_callback(self, leftImage, rightImage):
        self.lock.acquire()
        self.get_position_callback()
        self.cameraLeft_callback(leftImage)
        self.cameraRight_callback(rightImage)
        self._meta_data = FrameData(left=self._cameraLeft_img, right=self._cameraRight_img, ball_pos=self._ball_pos)
        self._observation_stale = False
        self.lock.release()
        self.end = time.time()


    def cameraLeft_callback(self, data):
        try:
            self._cameraLeft_img = self._br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def cameraRight_callback(self, data):
        try:
            self._cameraRight_img = self._br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def get_position_callback(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            get_position_msg = GetModelStateRequest()
            get_position_msg.model_name = 'Ping Pong Ball'
            state = get_state(get_position_msg)
            self._ball_pos = (state.pose.position.x, state.pose.position.y, state.pose.position.z)
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def get_camera_data(self):
        self.lock.acquire()
        if self._observation_stale:
            return_value = None
        else:
            self._observation_stale = True
            return_value = self._meta_data
        self.lock.release()
        return return_value

if __name__ == "__main__":
    process_index = 0
    rospy.init_node('acquire_frame_data')
    camera = Camera()
    SF = SaveFrame(camera=camera)
    repeat_num = 50
    ball_poses_all = []
    bounds = np.array([[0.050, 0.057],
                       [0.068, 0.076]])
    num = np.array([8, 5])
    x_offset = np.arange(-0.6, 0.6, 0.1).reshape(-1, 1)
    yz = SF.generate_force(bounds=bounds, num=num)
    x_offset_num = x_offset.shape[0]
    yz_num = yz.shape[0]
    x_offset = np.repeat(x_offset, yz_num, axis=0)
    yz = np.tile(yz, (x_offset_num, 1))
    # x_offset = np.tile(x_offset, (yz_num, 1))
    # yz = np.repeat(yz, x_offset_num, axis=0)
    print 'force condition size:', yz.shape[0]
    process_indices = np.array_split(np.arange(yz.shape[0]), 3)
    f_start_idx = process_indices[process_index][0]
    f_end_idx = process_indices[process_index][-1]
    assert f_start_idx < yz.shape[0], 'f_start_idx should be less than yz.shape[0]'
    conditions = (f_end_idx + 1 - f_start_idx) * repeat_num
    for idx in xrange(f_start_idx, f_end_idx+1):
        for r_idx in xrange(repeat_num):
            start_time = time.time()
            px = x_offset[idx] + 0.7
            fx = 0.0
            fy, fz = yz[idx]
            print('Collecting data for f%d-h%d, x=%.4f, fx=%.4f, fy=%.4f, fz=%.4f ...' %
                  (idx, r_idx, px, fx, fy, fz))
            SF.set_ball_position(px=px, py=-0.5, pz=0.85)
            SF.hit_ball(fx=fx, fy=fy, fz=fz, dt=0.15)
            meta_data = SF.acquire_frames_data(num_images=60)
            ball_poses = SF.save_data(fx=fx, fy=fy, fz=fz,
                                      meta_data=meta_data,
                                      f_idx=idx,
                                      h_idx=r_idx)
            end_time = time.time()
            elapsed_time = end_time - start_time
            loop_num = (idx - f_start_idx) * repeat_num + r_idx + 1
            remaining_time = elapsed_time * (conditions - loop_num)
            est_finished_time = datetime.now() + timedelta(seconds=remaining_time)
            print("    Loop %d / %d took %.2f s, "
                  "estimated finished time: \033[94m %s \033[00m" % (loop_num, conditions,
                                                   elapsed_time,
                                                   est_finished_time.strftime("%Y-%m-%d %H:%M:%S")))
            # ball_poses_all.append(ball_poses)
    # ball_poses_all = np.asarray(ball_poses_all)
    # utility.plot_3d([ball_poses_all, ball_poses_all[:, 47, :]], draw_now=True)









