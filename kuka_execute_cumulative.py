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
class KukaMotionPlanning(Node):
    def __init__(self, current_step):
        super().__init__('kuka_motion_planning')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/lbr/joint_trajectory_controller/follow_joint_trajectory')
        self.current_step = current_step
        self.joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
      

    def send_goal(self, waypoints_list):
        # Create a FollowJointTrajectory goal message
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Iterate through each joint trajectory point
        for i, joint_values in enumerate(waypoints_list):
            # self.get_logger().info(f"Processing trajectory point {i}")

            # Create a JointTrajectoryPoint and assign positions
            point = JointTrajectoryPoint()
            point.positions = joint_values.position
            point.time_from_start = rclpy.duration.Duration(seconds=(i + 1) * 0.19).to_msg()  # Increment time for each point

            # Add the point to the trajectory
            trajectory_msg.points.append(point)

        # Assign the trajectory to the goal message
        goal_msg.trajectory = trajectory_msg

        # Wait for the action server to be available
        self._action_client.wait_for_server()

        start_exec = time.time()
        # Send the goal asynchronously
        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        # Wait for the goal to be accepted

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Trajectory goal was rejected")
            return

        self.get_logger().info("Trajectory goal was accepted")

        # Wait for the result of the action

        # get_result_future = goal_handle.get_result_async()
        # rclpy.spin_until_future_complete(self, get_result_future)
        # result = get_result_future.result().result
        # end_exec = time.time()
        # duration = end_exec - start_exec
        # print(f"Durtion execution ; {duration}")
        # self.get_logger().info(f"Trajectory execution result: {result}")

    def send_gripper_command(self, position, max_effort):
        command = (
            f'ros2 action send_goal '
            f'gripper/robotiq_gripper_controller/gripper_cmd '
            f'control_msgs/action/GripperCommand '
            f'"{{command: {{position: {position}, max_effort: {max_effort}}}}}"'
        )
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print("Command executed successfully.")
            print(result.stdout)
        else:
            print("Error executing command.")
            print(result.stderr)

    def send_goal_pink(self, joint_trajectories, vel_arr):
        current_time = time.time()
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = list(joint_trajectories)
        # point.time_from_start.sec = time_taken  # Set the seconds part to 0
        # point.time_from_start.nanosec = int(0.35 * 1e9)  # Set the nanoseconds part to 750,000,000

        trajectory_msg.points.append(point)
        goal_msg.trajectory = trajectory_msg
        trajectory_msg.points[0].velocities = vel_arr
        # trajectory_msg.points[0].accelerations = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
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
        duration = time.time() - current_time
        print(duration)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # self.get_logger().info(f'Feedback: {feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')
        # rclpy.shutdown()