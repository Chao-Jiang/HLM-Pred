#!/usr/bin/env python
import threading
import rospy
import time
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.srv import SwitchController
from controller_manager_msgs.srv import SwitchControllerRequest
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState
from matplotlib import pyplot as plt
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

node = rospy.init_node('pingpang_controller')
pub = rospy.Publisher('position_arm_controller/command', JointTrajectory, queue_size=5)
action_msg = JointTrajectory()
action_msg.joint_names = ["move_1_joint", "move_2_joint", "move_3_joint"]
target = JointTrajectoryPoint()
target.positions = [0, 0, 1]
target.time_from_start = rospy.Duration.from_sec(1)
action_msg.points = [target]

pub.publish(action_msg)
print "a"
time.sleep(2)

pub_1 = rospy.Publisher('arm_controller/command', JointTrajectory, queue_size=5)
action_msg = JointTrajectory()
action_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', \
                          'wrist_2_joint', 'wrist_3_joint']
target = JointTrajectoryPoint()
target.positions = [-1, 0, 1, 0, 0, 0]
target.time_from_start = rospy.Duration.from_sec(1)
action_msg.points = [target]

pub.publish(action_msg)
print "b"
time.sleep(2)
raw_input('dfe')








t = []
y1 = []
y2 = []


class PingpangControl:
    def __init__(self, mode=0):
        self.mode = mode
        self.locker = threading.Lock()
        self.list_ctrlrs = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)
        self.list_ctrlrs.wait_for_service()

        while True:
            control_list = self.list_ctrlrs.call()
            print 'len', len(control_list.controller)
            if len(control_list.controller)==11:
                break
            time.sleep(1.5)

        self.switch_controller = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
        self.switch_controller.wait_for_service()
        self.switch_ctrl(self.mode)
        time.sleep(1)
        control_list = self.list_ctrlrs.call()
        for controller in control_list.controller:
            print controller.name, controller.state



        time.sleep(0.5)  # to ensure msg can be sent out

        self.current_state = []

        self.ps_1_msg = Float64()
        self.ps_2_msg = Float64()
        self.ps_3_msg = Float64()
        self.vel_1_msg = Float64()
        self.vel_2_msg = Float64()
        self.vel_3_msg = Float64()
        self.effort_1_msg = Float64()
        self.effort_2_msg = Float64()
        self.effort_3_msg = Float64()
        self.ps_1_msg.data = 0
        self.ps_2_msg.data = 0
        self.ps_3_msg.data = 0
        self.vel_1_msg.data = 0
        self.vel_2_msg.data = 0
        self.vel_3_msg.data = 0
        self.effort_1_msg.data = 0
        self.effort_2_msg.data = 0
        self.effort_3_msg.data = 0

        self.position_1_pub = rospy.Publisher('/move_1_position_ctrlr/command', Float64, queue_size=1)

        self.velocity_1_pub = rospy.Publisher('/move_1_velocity_ctrlr/command', Float64, queue_size=1)

        self.effort_1_pub = rospy.Publisher('/move_1_effort_ctrlr/command', Float64, queue_size=1)

        self.position_2_pub = rospy.Publisher('/move_2_position_ctrlr/command', Float64, queue_size=1)

        self.velocity_2_pub = rospy.Publisher('/move_2_velocity_ctrlr/command', Float64, queue_size=1)

        self.effort_2_pub = rospy.Publisher('/move_2_effort_ctrlr/command', Float64, queue_size=1)

        self.position_3_pub = rospy.Publisher('/move_3_position_ctrlr/command', Float64, queue_size=1)

        self.velocity_3_pub = rospy.Publisher('/move_3_velocity_ctrlr/command', Float64, queue_size=1)

        self.effort_3_pub = rospy.Publisher('/move_3_effort_ctrlr/command', Float64, queue_size=1)

        self.sub_1_command = rospy.Subscriber('pingpang_pos_ctrl/command', Float64MultiArray, self.set_position_cb, queue_size=1)

        self.draw_command = rospy.Subscriber('move_2_position_ctrlr/state', JointControllerState, self.get_current_state, queue_size=1)

        self.rate = rospy.Rate(1000)
        self._thread = threading.Thread(target=self.thread_fun)
        self._thread.start()

        # self._thread_pid = threading.Thread(target=self.pid_sin)
        # self._thread_pid.start()

    def set_position_cb(self, msg):
        with self.locker:
            self.ps_1_msg.data = msg.data[0]
            self.ps_2_msg.data = msg.data[1]
            self.ps_3_msg.data = msg.data[2]

    def get_current_state(self, msg):
        global t, y1, y2
        self.current_state.append(msg.process_value)
        t.append(float(msg.header.stamp.secs + msg.header.stamp.nsecs/1.0e9))
        y1.append(msg.process_value)
        y2.append(self.ps_2_msg.data)


    def switch_ctrl(self, name=0):
        print "switch_controller"
        self.switch_controller_req = SwitchControllerRequest()
        if name == 0:
            self.switch_controller_req.start_controllers = ['move_1_position_ctrlr', 'move_2_position_ctrlr',
                                                            'move_3_position_ctrlr','joint_state_ctrlr', 'arm_controller']
            self.switch_controller_req.stop_controllers = ['move_1_velocity_ctrlr', 'move_2_velocity_ctrlr',
                                                           'move_3_velocity_ctrlr',
                                                           'move_1_effort_ctrlr', 'move_2_effort_ctrlr',
                                                           'move_3_effort_ctrlr']
        elif name == 1:
            self.switch_controller_req.start_controllers = ['move_1_velocity_ctrlr', 'move_2_velocity_ctrlr',
                                                            'move_3_velocity_ctrlr','joint_state_ctrlr','arm_controller']
            self.switch_controller_req.stop_controllers = ['move_1_position_ctrlr', 'move_2_position_ctrlr',
                                                           'move_3_position_ctrlr'
                                                           'move_1_effort_ctrlr',
                                                           'move_2_effort_ctrlr', 'move_3_effort_ctrlr']
        else:
            self.switch_controller_req.start_controllers = ['move_1_effort_ctrlr', 'move_2_effort_ctrlr',
                                                            'move_3_effort_ctrlr','joint_state_ctrlr','arm_controller']
            self.switch_controller_req.stop_controllers = ['move_1_position_ctrlr', 'move_2_position_ctrlr',
                                                           'move_3_position_ctrlr',
                                                           'move_1_velocity_ctrlr', 'move_2_velocity_ctrlr',
                                                           'move_3_velocity_ctrlr']

        self.switch_controller_req.strictness = 1
        try:
            self.switch_controller.call(self.switch_controller_req)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def thread_fun(self):
        while True:
            self.position_1_pub.publish(self.ps_1_msg)
            self.position_2_pub.publish(self.ps_2_msg)
            self.position_3_pub.publish(self.ps_3_msg)
            self.rate.sleep()

    def pid_sin(self):
        fz = 100.0
        tt = 0
        while True:
            with self.locker:
                self.ps_2_msg.data = np.sin(2*np.pi/fz*tt)
                # self.ps_2_msg.data = np.sin(2*np.pi/fz*tt)
                # self.ps_3_msg.data = np.sin(2*np.pi/fz*tt)
                tt += 1
                time.sleep(1/fz)

class PosControl:
    def __init__(self):
        self.jointState = JointState()
        self.locker = threading.Lock()
        self.pub = rospy.Publisher('pingpang_pos_ctrl/command', Float64MultiArray, queue_size=1)
        self.Joint_state = rospy.Subscriber('/joint_states', JointState, self.position_state_cb, queue_size=1)
        self.msg = Float64MultiArray()
        # self.msg.data = [-100, -100, -100]
        time.sleep(0.5)

    def position_state_cb(self, joint_msg):
        with self.locker:
            self.jointState = joint_msg
            # print self.jointState.position[0]

    def set_position(self, position=[0, 0, 0]):
        self.msg.data = position
        self.pub.publish(self.msg)

    def waitfinish(self):
        while (abs(self.jointState.position[0] - self.msg.data[0]) >= 0.001) or \
                (abs(self.jointState.position[1] - self.msg.data[1]) >= 0.001) or \
                (abs(self.jointState.position[2] - self.msg.data[2]) >= 0.001):
            time.sleep(0.05)
        print 'finish'




# def draw_test():
#     time.sleep(2)
#     fz = 1000.0
#     global t, y1, y2
#     for i in range(1000):
#         t.append(i)
#         y1.append(i*i)
#         y2.append(i*i-100*i)
#         time.sleep(1/fz)


def draw():
    plt.ion()
    fz = 2000.0
    current_t = 0
    global t, y1, y2
    while True:
        if len(t) >= 1:
            if t[-1] > current_t:
                current_t = t[-1]
                plt.scatter(t[-1], y1[-1], c="b", s=5, alpha=0.4, marker='o')
                plt.scatter(t[-1], y2[-1], c="r", s=5, alpha=0.4, marker='o')
                plt.pause(1/fz)

if __name__ == '__main__':
    node = rospy.init_node('pingpang_controller')
    print 'controller.py started'
    mode = int(rospy.get_param("~mode", 0))
    sleep_time = float(rospy.get_param("~time", 6))
    demo = bool(rospy.get_param("~demo",False))
    time.sleep(sleep_time)  # to ensure gazebo are started
    ctrl = PingpangControl(mode)
    time.sleep(0.5)

    ctrl.ps_2_msg.data = 0

    raw_input("start")

    ctrl.ps_2_msg.data = -0.1

    # th = threading.Thread(target=draw_test)
    #
    # th.start()

    draw()

    raw_input()


    # ctrl._thread.join()
    # print 'controller stop'

