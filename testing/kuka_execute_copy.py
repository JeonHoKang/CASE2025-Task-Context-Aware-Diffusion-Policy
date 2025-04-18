import sys
import csv
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
import time
import subprocess
from robotiq_p import ControlRobotiq 
class KukaMotionPlanning(Node):
    def __init__(self, current_step):
        super().__init__('kuka_motion_planning')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/lbr/joint_trajectory_controller/follow_joint_trajectory')
        self.current_step = current_step
        self.joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.robotiq_gripper = ControlRobotiq()


    def send_goal(self, joint_trajectories, gripper_pos):
        for i, joint in enumerate(joint_trajectories):
            goal_msg = FollowJointTrajectory.Goal()
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names

            point = JointTrajectoryPoint()
            point.positions = joint.position.tolist()
            # point.time_from_start.sec = 1  # Set the seconds part to 0
            point.time_from_start.nanosec = int(0.4 * 1e9)  # Set the nanoseconds part to 750,000,000
            current_gripper_pos = self.robotiq_gripper.get_gripper_current_pose()
            if gripper_pos[i] < 0:
                gripper_pos[i] = 0.0
            delta_gripper = abs(current_gripper_pos - gripper_pos[i])
            if delta_gripper > 0.1:
                scaled_command = float(gripper_pos[i]*1.3)
                if scaled_command > 0.8:
                    scaled_command = 0.8
                self.robotiq_gripper.send_gripper_command(scaled_command)

            trajectory_msg.points.append(point)
        goal_msg.trajectory = trajectory_msg
        
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, self._send_goal_future)
        goal_handle = self._send_goal_future.result()

        # if not goal_handle.accepted:
        #     self.get_logger().info(f"Goal for point {i} was rejected")
        #     return
        
        # self.get_logger().info(f"Goal for point {i} was accepted")
        # Wait for the result to complete before moving to the next trajectory point
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # self.get_logger().info(f'Feedback: {feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')
        # rclpy.shutdown()