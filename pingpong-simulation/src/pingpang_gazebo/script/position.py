import numpy as np
import matplotlib.pyplot as plt
import math
import time
import rospy
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetModelStateRequest
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import ApplyBodyWrench
from gazebo_msgs.srv import ApplyBodyWrenchRequest
from geometry_msgs.msg import Wrench
from mpl_toolkits.mplot3d import Axes3D

import Camera
import cv2
import cv_bridge
import threading

if __name__ == '__main__':
    node = rospy.init_node('position')
    rospy.Duration(10, 0)
    thread_lock = threading.Lock()
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    try:
        set_state.wait_for_service()
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    try:
        get_state.wait_for_service()
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    apply_force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
    try:
        apply_force.wait_for_service()
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    def set_position():
        set_position_re = SetModelStateRequest()
        pose = Pose()
        pose.position.x = 0.7
        pose.position.y = -0.5
        pose.position.z = 1.28
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = np.sin(math.pi / 2)
        pose.orientation.w = np.cos(math.pi / 2)
        set_position_re.model_state.model_name = 'ping_pang_ball'
        set_position_re.model_state.pose = pose
        try:
            resp = set_state(set_position_re)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e


    def get_position():
        get_position_msg = GetModelStateRequest()
        get_position_msg.model_name = 'unit_box_0'
        state = get_state(get_position_msg)
        print(state.pose.position.x, state.pose.position.y)


    def hit_ball():
        apply_force_re = ApplyBodyWrenchRequest()
        print apply_force_re.start_time, apply_force_re.duration
        apply_force_re.body_name = 'ping_pang_ball::ball'
        wrench = Wrench()
        wrench.force.x = 0
        wrench.force.y = 0.3
        wrench.force.z = 0
        wrench.torque.x = 0
        wrench.torque.y = 0
        wrench.torque.z = 0
        apply_force_re.wrench = wrench
        apply_force_re.start_time = rospy.get_rostime()
        apply_force_re.duration = rospy.Duration(0.05)
        print apply_force_re.start_time, apply_force_re.duration
        apply_force(apply_force_re)

    time.sleep(2)
    set_position()
    get_position_msg = GetModelStateRequest()
    get_position_msg.model_name = 'ping_pang_ball'
    fig = plt.figure()
    ax = Axes3D(fig)
    t = list()
    x = list()
    y = list()
    z = list()
    _img = [None, None]
    _lock = threading.Lock()
    _b = cv_bridge.CvBridge()

    cv2.startWindowThread()
    cv2.namedWindow('w0', cv2.WINDOW_NORMAL)
    cv2.namedWindow('w1', cv2.WINDOW_NORMAL)

    t_list = ['/multi/camera/basler/left/image_raw',
              '/multi/camera/basler/right/image_raw']
    count = -1

    def callback_f(msg_0, msg_1):
        global count
        if count >= 0:
            with _lock:
                _img[0] = _b.imgmsg_to_cv2(msg_0, "bgr8")
                _img[1] = _b.imgmsg_to_cv2(msg_1, "bgr8")
                state = get_state(get_position_msg)
                x.append(state.pose.position.x)
                y.append(state.pose.position.y)
                z.append(state.pose.position.z)
                count += 1

    test = Camera.ReadCameraSyn(t_list, fc=callback_f, p=False)
    time.sleep(1)
    count = 0
    time.sleep(0.5)
    hit_ball()
    while True:
        if count > 80:
            break
        cv2.imshow('w0', _img[0])
        cv2.imshow('w1', _img[1])
        cv2.waitKey(1)

    ax.scatter(x, y, z)
    print x
    print y
    print z
    plt.show()
    cv2.destroyAllWindows()
    # set_position()
