from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import rospy
import os
import moveit_commander
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
import Queue
import copy
import time
import json
from pingpang_control.srv import *
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, \
    SetModelState, SetModelStateRequest, ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import Wrench, Pose
from actionlib import SimpleActionClient
import geometry_msgs.msg
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import cv2
import collections
from cv_bridge import CvBridge, CvBridgeError
import cv_bridge
import threading
from sensor_msgs.msg import Image
import message_filters
from scipy.interpolate import spline, interp1d
# import utility

FrameData = collections.namedtuple('FrameData', 'left, right, timestamp')
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


class RobotControl:
    def __init__(self, camera, single_frame_pred=True, moveit_execute=False, make_plan=False):
        self.single_frame_pred = single_frame_pred
        self.camera = camera
        self.moveit_execute = moveit_execute
        self.make_plan = make_plan
        if self.moveit_execute or self.make_plan:
            moveit_commander.roscpp_initialize(['abc'])
            self._robot = moveit_commander.RobotCommander()
            self._scene = moveit_commander.PlanningSceneInterface()
            self._group = moveit_commander.MoveGroupCommander("manipulator")
            self._group.set_planner_id("RRTkConfigDefault")
            self._group.set_num_planning_attempts(1)
            self._group.set_goal_position_tolerance(0.005)
            self._group.set_goal_orientation_tolerance(0.005)
            self._group.set_max_velocity_scaling_factor(1)
            self._group.set_planning_time(0.5)
        self._control_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        self._action_msg = JointTrajectory()
        self.joint_orders = ['shoulder_pan_joint',
                             'shoulder_lift_joint',
                             'elbow_joint',
                             'wrist_1_joint',
                             'wrist_2_joint',
                             'wrist_3_joint']
        self._action_msg.joint_names = self.joint_orders

        self.adapt_hit_direction = False
        self.hit_idx = 0
        self.px = 0.7
        self.py = -0.5
        self.fx = 0.0
        self.fy = 0.04
        self.fz = 0.10
        self.control_frequency = 100.0
        self.control_period = 1 / float(self.control_frequency)
        self.control_ROSRate = rospy.Rate(self.control_frequency)

        self.image_frequency = 30.0
        self.image_period = 1 / float(self.image_frequency)
        self.image_ROSRate = rospy.Rate(self.image_frequency)

        self.frames_data = Queue.Queue()

        # self.acquire_repeated_timer = RepeatedTimer(self.image_period, self.subscribe_frames_data)
        self.offline_action_filename = 'plan_point.json'
        if self.make_plan:
            self.reset()
            self.moveit_plan(execute=True)

        if self.moveit_execute:
            self.hit_traj = RobotTrajectory()
            self.hit_traj.joint_trajectory.header.frame_id = '/world'
            self.hit_traj.joint_trajectory.joint_names = self.joint_orders
            with open(self.offline_action_filename, 'r') as outfile:
                traj_data = json.load(outfile)
            for idx in xrange(len(traj_data)):
                prefix = 'point%d' % idx
                jtp = JointTrajectoryPoint()
                jtp.accelerations = traj_data[prefix]['accelerations']
                jtp.positions = traj_data[prefix]['positions']
                jtp.time_from_start.nsecs = traj_data[prefix]['time_nsecs']
                jtp.time_from_start.secs = traj_data[prefix]['time_secs']
                jtp.velocities = traj_data[prefix]['velocities']
                self.hit_traj.joint_trajectory.points.append(jtp)
        else:
            with open(self.offline_action_filename, 'r') as outfile:
                self.hit_actions = np.asarray(json.load(outfile))

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
                time.sleep(0.5)
                # self.reset()
                self.set_ball_position(px=self.px, py=self.py)
                self.hit_ball(fx=self.fx, fy=self.fy, fz=self.fz, dt=0.15)
                self.acquire_repeated_timer = RepeatedTimer(self.image_period, self.subscribe_frames_data)
            pass
        self.acquire_repeated_timer.stop()
        left_fgbg = cv2.BackgroundSubtractorMOG()
        right_fgbg = cv2.BackgroundSubtractorMOG()
        assert os.path.exists('./bkg/left-bkg.jpg'), 'Left bkg does not exist'
        assert os.path.exists('./bkg/right-bkg.jpg'), 'Right bkg does not exist'
        left_image_bkg = cv2.imread('./bkg/left-bkg.jpg')
        right_image_bkg = cv2.imread('./bkg/right-bkg.jpg')
        left_fgmask = left_fgbg.apply(left_image_bkg)
        right_fgmask = right_fgbg.apply(right_image_bkg)
        for i in xrange(discard_num):
            meta_data_one_frame = self.frames_data.get()
        while not self.frames_data.empty():
            meta_data_one_frame = self.frames_data.get()
            # ball_pos.append(meta_data_one_frame.ball_pos)
            left_fgmask = left_fgbg.apply(meta_data_one_frame.left)
            right_fgmask = right_fgbg.apply(meta_data_one_frame.right)
            left_contours, left_hierarchy = cv2.findContours(left_fgmask,
                                                             cv2.RETR_EXTERNAL,
                                                             cv2.CHAIN_APPROX_SIMPLE)
            right_contours, right_hierarchy = cv2.findContours(right_fgmask,
                                                               cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_SIMPLE)
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
        moment_time = meta_data_one_frame.timestamp
        # print('center:', np.asarray(center_pixels).shape)
        return np.asarray(center_pixels)[:num_images], moment_time, #np.asarray(ball_pos)


    def acquire_nn_response(self, center_pixels):
        rospy.wait_for_service('prediction_interface')
        try:
            prediction_interface = rospy.ServiceProxy('prediction_interface', Table_Tennis)
            req = Table_TennisRequest()
            req.inputs = np.asanyarray(center_pixels, dtype=np.int32).reshape(-1)
            resp = prediction_interface(req)
            if self.single_frame_pred:
                resp = PredictionData(position=resp.outputs[:-1], frame=resp.outputs[-1])
            else:
                resp = PredictionData(position=np.asarray(resp.outputs[:-1]).reshape(-1, 3), frame=resp.outputs[-1])
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)
        return resp

    def get_wait_time(self, resp):
        wait_time = (resp.frame + self.hit_idx) * self.image_period
        return wait_time

    def change_ee_link_dir(self, ball_pos):
        rot_angle = np.arctan2(ball_pos[-1 + self.hit_idx, 2] - ball_pos[-2 + self.hit_idx, 2],
                               ball_pos[-1 + self.hit_idx, 1] - ball_pos[-2 + self.hit_idx, 1])
        rot_angle = -np.pi / 2.0 - rot_angle
        if self.moveit_execute:
            for point in self.hit_traj.joint_trajectory.points:
                point.positions[-1] = rot_angle
        else:
            self.hit_actions[:, -1] = rot_angle
        prepare_joints = [0, -1.51, 1.87, -0.35, 1.57, rot_angle]
        start = time.time()
        while time.time() - start < 0.2:
            self._control_pub.publish(self._get_ur_trajectory_message(prepare_joints, slowness=0.01))

        return rot_angle

    def get_base_pos_to_hit_ball(self, ball_pos):
        if self.single_frame_pred:
            ball_pos = ball_pos
        else:
            max_idx = np.argmax(ball_pos, axis=0)[2]
            ball_pos_len = ball_pos.shape[0]
            self.adapt_hit_direction = True
            self.hit_idx = 0
            # if self.adapt_hit_direction:
            #     if max_idx < ball_pos_len - 3:
            #         self.hit_idx = max_idx + 4 - ball_pos_len
            ball_pos = ball_pos[self.hit_idx, :]
        origin_xyz = np.array([0.694, 2.841, 0.786])
        # origin_xyz = np.array([0.694, 2.841, 0.816])
        # ball_pos = np.asarray(ball_pos) + origin_xyz
        offset = np.asarray(ball_pos) - origin_xyz
        return offset

    def move_base(self, offset):
        client = SimpleActionClient('/position_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        client.wait_for_server()
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ['move_1_joint', 'move_2_joint', 'move_3_joint']

        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration.from_sec(0.3)
        point.positions = offset.tolist()

        # point1 = JointTrajectoryPoint()
        # point1.time_from_start = rospy.Duration.from_sec(0.3)
        # print point1.time_from_start
        # point1.positions = [0, 0, 0]

        # point2 = JointTrajectoryPoint()
        # point2.time_from_start = rospy.Duration.from_sec(0.6)
        # point2.positions = [offset_x, offset_y, offset_z]

        goal.trajectory.points.append(point)
        # goal.trajectory.points.append(point2)

        client.send_goal(goal)
        client.wait_for_result()

    def get_ee_link_pos(self, tf, target, source):
        new_time = tf.getLatestCommonTime(target, source)
        pos_now = self.get_position(tf, target, source, new_time)
        return pos_now

    def get_position(self, tf, target, source, time):
        """
        Utility function that uses tf to return the position of target
        relative to source at time
        tf: Object that implements TransformListener
        target: Valid label corresponding to target link
        source: Valid label corresponding to source link
        time: Time given in TF's time structure of secs and nsecs
        """
        position, _ = tf.lookupTransform(source, target, time)
        position = np.asarray(position)
        return position

    def moveit_plan(self, execute=False):
        from tf import TransformListener
        tf = TransformListener()
        time.sleep(1)
        ee_link_pos = self.get_ee_link_pos(tf, 'ee_link', 'world')
        current = geometry_msgs.msg.Pose()
        current.position.x = ee_link_pos[0]
        current.position.y = ee_link_pos[1]
        current.position.z = ee_link_pos[2]
        current.orientation.x = 0.655
        current.orientation.y = -0.266
        current.orientation.z = -0.655
        current.orientation.w = 0.266

        delta_y = 0.1
        delta_z = -0.0528
        waypoints = []
        for i in xrange(0, 4):
            wpose = copy.deepcopy(current)
            wpose.position.y = current.position.y - i * delta_y
            wpose.position.z = current.position.z - i * delta_z
            waypoints.append(wpose)

        (traj, fraction) = self._group.compute_cartesian_path(waypoints,  # waypoints to follow
                                                              50,        # eef_step
                                                              0.0)        # jump_threshold

        n_points = len(traj.joint_trajectory.points)
        print('MoveIt planned traj contains %d points' % n_points)
        spd = 25
        for i in xrange(n_points):
            time_from_start = traj.joint_trajectory.points[i].time_from_start.secs + \
                traj.joint_trajectory.points[i].time_from_start.nsecs / 1e9
            traj.joint_trajectory.points[i].time_from_start = rospy.Duration.from_sec(time_from_start / spd)

        if execute:
            self._group.execute(traj)
        if self.moveit_execute:
            traj_data = {}
            for idx in xrange(n_points):
                traj_data['point%d' % idx] = {
                    'accelerations': traj.joint_trajectory.points[idx].accelerations,
                    'positions': traj.joint_trajectory.points[idx].positions,
                    'time_nsecs': traj.joint_trajectory.points[idx].time_from_start.nsecs,
                    'time_secs': traj.joint_trajectory.points[idx].time_from_start.secs,
                    'velocities': traj.joint_trajectory.points[idx].velocities,
                }
            with open(self.offline_action_filename, 'w+') as f:
                json.dump(traj_data, f, indent=4)
        else:
            hit_actions = self.interpolate_traj(traj)
            print('Interpolated traj contains %d points' % hit_actions.shape[0])
            with open(self.offline_action_filename, 'w+') as f:
                json.dump(hit_actions.tolist(), f, indent=4)

        return traj

    def _get_ur_trajectory_message(self, action, slowness=1.0):
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = self.joint_orders

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        target.positions = action

        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        target.time_from_start = rospy.Duration(slowness)

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]
        return action_msg

    def interpolate_traj(self, traj):
        time_from_start = []
        actions = []
        for point in traj.joint_trajectory.points:
            time_tmp = float(point.time_from_start.secs) + float(point.time_from_start.nsecs) / 1.0e9
            time_from_start.append(time_tmp)
            actions.append(point.positions)
        actions = np.asarray(actions)
        # print('before:',actions.shape[0])
        ipl_time_from_start = np.arange(0, time_from_start[-1], step=self.control_period)
        ipl_actions = np.zeros((ipl_time_from_start.size, actions.shape[1]))
        # ipl_actions = spline(time_from_start, actions, ipl_time_from_start)
        for idx in xrange(actions.shape[1]):
            f = interp1d(time_from_start, actions[:, idx], kind='linear')
            ipl_actions[:, idx] = f(ipl_time_from_start)
        # print('after:', ipl_actions.shape[0])
        return ipl_actions

    def exe_hit_plan(self):
        if self.moveit_execute:
            self._group.execute(self.hit_traj, wait=True)
        else:
            # start = time.time()
            for idx in xrange(self.hit_actions.shape[0]):
                self._control_pub.publish(self._get_ur_trajectory_message(self.hit_actions[idx],
                                                                          slowness=self.control_period / 1.4))
                self.control_ROSRate.sleep()
                # end = time.time()
                # print('Elapsed time: ', end - start)
                # start = end
        time.sleep(1.0)

    def reset(self):
        reset_joints = [0, -1.51, 1.87, -0.35, 1.57, -0.8]
        start = time.time()
        while time.time() - start < 2:
            self._control_pub.publish(self._get_ur_trajectory_message(reset_joints, slowness=0.5))
        # if self.moveit_execute:
        #     self._group.set_joint_value_target(reset_joints)
        #     reset_plan = self._group.plan()
        #     self._group.execute(reset_plan, wait=True)
        # else:
        #     start = time.time()
        #     while time.time() - start < 2:
        #         self._control_pub.publish(self._get_ur_trajectory_message(reset_joints, slowness=0.5))
        time.sleep(0.5)


    def set_ball_position(self, px=0.7, py=-0.5):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.px = px
            self.py = py
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_position_re = SetModelStateRequest()
            pose = Pose()
            pose.position.x = px
            pose.position.y = py
            pose.position.z = 0.85
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = np.sin(np.pi / 2)
            pose.orientation.w = np.cos(np.pi / 2)
            set_position_re.model_state.model_name = 'Ping Pong Ball'
            set_position_re.model_state.pose = pose
            set_state(set_position_re)
            time.sleep(4)
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)


    def hit_ball(self, fx=0.0, fy=0.15, fz=0.0, dt=0.1):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        try:
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



class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self._br = cv_bridge.CvBridge()
        self._cameraRight_img = None
        self._cameraLeft_img = None
        self._ball_pos = None
        self._meta_data = None
        self._observation_stale = True
        # self.start = time.time()
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
        self._timestamp = time.time()
        # self.get_position_callback()
        self.cameraLeft_callback(leftImage)
        self.cameraRight_callback(rightImage)
        self._meta_data = FrameData(left=self._cameraLeft_img,
                                    right=self._cameraRight_img,
                                    timestamp=self._timestamp)
        self._observation_stale = False
        self.lock.release()
        self.end = time.time()
        # print('time:', self.end - self.start)
        # self.start = self.end


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
    rospy.init_node('table_tennis_play')
    camera = Camera()
    SINGLE_FRAME_PREDICTION = False
    RANDOM = False
    RC = RobotControl(camera=camera,
                      moveit_execute=False,
                      make_plan=False,
                      single_frame_pred=SINGLE_FRAME_PREDICTION)
    exe_time = 0.03 #0.04

    prediction = []
    trial_num = 0
    train_condition = {
        'px': np.arange(0.1, 1.3, 0.1),
        'fy': np.array([0.050, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057]),
        'fz': np.linspace(0.068, 0.076, 5)
    }
    while not rospy.is_shutdown() and trial_num < 20:
        if RANDOM:
            px = np.random.choice(train_condition['px'], 1)
            fy = np.random.choice(train_condition['fy'], 1)
            fz = np.random.choice(train_condition['fz'], 1)
        else:
            px = 0.7
            fy = 0.053#0.040
            fz = 0.072#0.10
        print('px = %.3f, fy = %.3f, fz = %.3f' % (px, fy, fz))
        RC.move_base(np.zeros(3))
        RC.reset()
        RC.set_ball_position(px=px, py=-0.5)
        RC.hit_ball(fx=0.0, fy=fy, fz=fz, dt=0.15)
        center_pixels, moment_time = RC.acquire_frames_data(num_images=14)
        print('center', center_pixels)
        resp = RC.acquire_nn_response(center_pixels=center_pixels)
        if SINGLE_FRAME_PREDICTION:
            print('    output: ', ' '.join(format(f, '.3f') for f in resp.position))
            prediction.append(resp.position)
        else:
            position = np.asarray(resp.position)
            print('    output: ', ' '.join(format(position[-1, i], '.3f') for i in xrange(position.shape[1])))
            prediction.append(position[-1])

        offset = RC.get_base_pos_to_hit_ball(ball_pos=resp.position)

        RC.change_ee_link_dir(ball_pos=resp.position)
        RC.move_base(offset)
        # time.sleep(2)
        wait_time = RC.get_wait_time(resp=resp)
        while time.time()-moment_time < wait_time - exe_time:
            pass
        RC.exe_hit_plan()
        trial_num += 1

    # prediction = np.asarray(prediction)
    # print('pred:')
    # print(prediction)
    # print(np.mean(prediction, axis=0))
    # print(np.median(prediction, axis=0))
    # fig = plt.figure()
    # plt.rcParams['axes.facecolor'] = 'white'
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(prediction[:, 0],
    #            prediction[:, 1],
    #            prediction[:, 2],
    #            marker='o', c='r', s=15)
    # utility.print_stats(prediction[:, 0], title='x')
    # utility.print_stats(prediction[:, 1], title='y')
    # utility.print_stats(prediction[:, 2], title='z')










