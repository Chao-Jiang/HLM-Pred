#!/usr/bin/env python
import time
import sys
import rospy
from actionlib import SimpleActionClient
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint


import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

if __name__ == '__main__':
    # sleep_time = float(sys.argv[1])
    sleep_time = 1.0
    print "sleep_time", sleep_time
    time.sleep(sleep_time)
    node = rospy.init_node('action_test_1')

    # ===============================================
    '''
    1.Test unit for platform moving
    '''
    # ===============================================
    client = SimpleActionClient('/position_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()
    print 'OK'
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['move_1_joint', 'move_2_joint', 'move_3_joint']

    point1 = JointTrajectoryPoint()
    point1.time_from_start = rospy.Duration.from_sec(0.3)
    print point1.time_from_start
    point1.positions = [0, 0, 0]

    # point2 = JointTrajectoryPoint()
    # point2.time_from_start = rospy.Duration.from_sec(0.6)
    # point2.positions = [1, 0.1, -0.5]

    point2 = JointTrajectoryPoint()
    point2.time_from_start = rospy.Duration.from_sec(0.6)
    point2.positions = [1.0, 0.0, 0.0]

    goal.trajectory.points.append(point1)
    goal.trajectory.points.append(point2)

    for i in range(0, 1):
        print i
        client.send_goal(goal)
        client.wait_for_result()

    # ===============================================
    '''
    2.Test unit for robot moving
    '''
    # ===============================================
    # client_1 = SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    # client_1.wait_for_server()
    # print 'OK'
    # goal = FollowJointTrajectoryGoal()
    # goal.trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', \
    #                                'wrist_2_joint', 'wrist_3_joint']
    #
    # point1 = JointTrajectoryPoint()
    # point1.time_from_start = rospy.Duration.from_sec(2)
    # print point1.time_from_start
    # point1.positions = [1, 1, 0, 0, 0, 0]
    #
    # point2 = JointTrajectoryPoint()
    # point2.time_from_start = rospy.Duration.from_sec(4)
    # point2.positions = [0, 0, 0, 0, 0, 0]
    #
    # goal.trajectory.points.append(point1)
    # goal.trajectory.points.append(point2)
    #
    # for i in range(0, 1):
    #     print i
    #     client_1.send_goal(goal)
    #     client_1.wait_for_result()

    # ===============================================
    '''
    3.Test unit for moviet
    '''
    # ===============================================
    # robot = moveit_commander.RobotCommander()
    # scene = moveit_commander.PlanningSceneInterface()
    # group = moveit_commander.MoveGroupCommander("manipulator")
    #
    # display_trajectory_publisher = rospy.Publisher(
    #     '/move_group/display_planned_path',
    #     moveit_msgs.msg.DisplayTrajectory, queue_size=1)
    #
    # group.set_planner_id("RRTkConfigDefault")
    # group.set_num_planning_attempts(3)
    # group.set_goal_position_tolerance(0.005)
    #
    # group.set_named_target('hit')
    # plan3 = group.plan()
    # group.execute(plan3)

















