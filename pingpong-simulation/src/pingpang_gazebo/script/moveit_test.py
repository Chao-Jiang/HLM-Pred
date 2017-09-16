
import sys
import rospy

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

if __name__ == '__main__':
    print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("manipulator")

    display_trajectory_publisher = rospy.Publisher(
        '/move_group/display_planned_path',
        moveit_msgs.msg.DisplayTrajectory)

    print "============ Waiting for RVIZ..."
    rospy.sleep(2)
    print "============ Starting tutorial "
    print "============ Reference frame: %s" % group.get_planning_frame()
    print "============ Reference frame: %s" % group.get_end_effector_link()
    print "============ Robot Groups:"
    print robot.get_group_names()
    print "============ Printing robot state"
    print robot.get_current_state()
    print "============"
    group.clear_pose_targets()
    group_variable_values = group.get_current_joint_values()
    print "============ Joint values: ", group_variable_values
    group_variable_values[1] = -1.0
    print group_variable_values
    group.set_joint_value_target(group_variable_values)

    plan2 = group.plan()
    group.execute(plan2)
    print "============ Waiting while RVIZ displays plan2..."
    rospy.sleep(5)





