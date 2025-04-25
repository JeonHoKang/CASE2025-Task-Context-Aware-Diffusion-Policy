from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import collections
from tqdm.auto import tqdm
import os
from real_robot_network import DiffusionPolicy_Real
from data_util import data_utils
from train_utils import train_utils
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation as R
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState, MoveItErrorCodes, JointConstraint, Constraints
from geometry_msgs.msg import Pose, WrenchStamped
from scipy.spatial.distance import euclidean

# from data_collection.submodules.wait_for_message import wait_for_message
import matplotlib.pyplot as plt
#@markdown ### **Loading Pretrained Checkpoint**
#@markdown Set `load_pretrained = True` to load pretrained weights.
from typing import Union
import json
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from kuka_execute_cumulative import KukaMotionPlanning
import cv2
from rotation_utils import quat_from_rot_m, rot6d_to_mat, mat_to_rot6d, quat_to_rot_m, normalize
import hydra
from omegaconf import DictConfig
from data_util import center_crop
import csv
from datetime import datetime
from robotiq_p import ControlRobotiq
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_poses_from_csv(file_path):
    """
    Reads poses (position and 6D rotation) from a CSV and returns them.
    Each row should have: x, y, z, rot_6d_1, rot_6d_2, rot_6d_3, rot_6d_4, rot_6d_5, rot_6d_6
    """
    poses = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            position = [float(row['x']), float(row['y']), float(row['z'])]
            rotation_6d = [float(row[f'rot_6d_{i+1}']) for i in range(6)]
            poses.append(position + rotation_6d)  # Combine position and rotation
    return poses


def execute_action(self, csv_file_path, gripper_action, steps):
    # Read poses from CSV
    end_effector_pos = read_poses_from_csv(csv_file_path)  # Getting poses from CSV

def convert_6d_to_rotation_matrix(r1, r2):
    """Convert 6D rotation representation to a 3x3 rotation matrix."""
    r1 = np.array(r1) / np.linalg.norm(r1)  # Normalize first vector
    r2 = np.array(r2) / np.linalg.norm(r2)  # Normalize second vector
    r3 = np.cross(r1, r2)  # Ensure orthogonality
    return np.column_stack([r1, r2, r3])  # Construct rotation matrix

def convert_rotation_matrix_to_6d(R_mat):
    """Convert a 3x3 rotation matrix back to 6D representation (first two columns)."""
    return R_mat[:, 0].tolist() + R_mat[:, 1].tolist()
def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

class EndEffectorPoseNode(Node):
    timeout_sec_ = 5.0
    joint_state_topic_ = "lbr/joint_states"
    fk_srv_name_ = "lbr/compute_fk"
    ik_srv_name_ = "lbr/compute_ik"
    base_ = "lbr/link_0"
    end_effector_ = "link_ee"


    def __init__(self, node_id: str) -> None:
        super().__init__(f"end_effector_pose_node_{node_id}")
        self.force_torque_topic_ = "/lbr/force_torque_broadcaster/wrench"
        self.joint_trajectories = read_poses_from_csv('/home/lm-2023/jeon_team_ws/lbr-stack/src/data_collection/data_collection/logs/run_1/inference_2025-04-18_13-09-52.csv')
        # Subscribe to the force/torque sensor topic
        self.force_torque_subscriber = self.create_subscription(WrenchStamped, self.force_torque_topic_, self.force_torque_callback, 10)
        
        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)
        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

    

    def force_torque_callback(self, msg):
        self.force_torque_data = msg.wrench

    def get_fk(self) -> Pose | None:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        while not current_joint_state_set:
            self.get_logger().warn("Failed to get current joint state")
            current_joint_state_set, current_joint_state = wait_for_message(
                JointState, self, self.joint_state_topic_, time_to_wait=1.0
            )
        else:
            current_robot_state = RobotState()
            current_robot_state.joint_state = current_joint_state

            request = GetPositionFK.Request()

            request.header.frame_id = self.base_
            request.header.stamp = self.get_clock().now().to_msg()

            request.fk_link_names.append(self.end_effector_)
            request.robot_state = current_robot_state

            future = self.fk_client_.call_async(request)

            rclpy.spin_until_future_complete(self, future)
            if future.result() is None:
                self.get_logger().error("Failed to get FK solution")
                return None
            
            response = future.result()
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"Failed to get FK solution: {response.error_code.val}"
                )
                return None
            
            pose = response.pose_stamped[0].pose
            position = [pose.position.x, pose.position.y, pose.position.z]
            quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            if quaternion[3] < 0:
                quaternion = [-x for x in quaternion]
            # # Convert quaternion to roll, pitch, yaw
            # rotation = R.from_quat(quaternion)
            # rpy = rotation.as_euler('xyz', degrees=False).tolist()
            return position + quaternion
    
    def get_ik(self, target_pose: Pose) -> JointState | None:
        request = GetPositionIK.Request()
    
        request.ik_request.group_name = "arm"
        # tf_prefix = self.get_namespace()[1:]
        request.ik_request.pose_stamped.header.frame_id = f"{self.base_}"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True
        constraints = Constraints()
        current_positions = []
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        for joint_name,current_position in zip(current_joint_state.name, np.array(current_joint_state.position)):
            if joint_name == "A1":
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = current_position
                joint_constraint.tolerance_below = np.pi/2
                joint_constraint.tolerance_above = np.pi/2
                joint_constraint.weight = 0.5
                constraints.joint_constraints.append(joint_constraint)
            elif joint_name == "A2":
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = current_position
                joint_constraint.tolerance_below = np.pi/2
                joint_constraint.tolerance_above = np.pi/2
                joint_constraint.weight = 0.5
                constraints.joint_constraints.append(joint_constraint)
            elif joint_name == "A3":
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = current_position
                joint_constraint.tolerance_below = np.pi/7
                joint_constraint.tolerance_above = np.pi/7
                joint_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_constraint)
            elif joint_name == "A4":
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = current_position
                joint_constraint.tolerance_below = np.pi/2
                joint_constraint.tolerance_above = np.pi/2
                joint_constraint.weight = 0.5
                constraints.joint_constraints.append(joint_constraint)
            elif joint_name == "A5":
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = current_position
                joint_constraint.tolerance_below = np.pi/2
                joint_constraint.tolerance_above = np.pi/2
                joint_constraint.weight = 0.5
                constraints.joint_constraints.append(joint_constraint)

        request.ik_request.constraints = constraints
        future = self.ik_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        while future.result() is None:
            self.get_logger().error("Failed to get IK solution")
            future = self.ik_client_.call_async(request)

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        return response.solution.joint_state
    
def execute_action(self, end_effector_pos, gripper_action, steps):
    def quaternion_multiply(q1, q2):
        x1, y1, z1, w1 = q1  # Note: [qx, qy, qz, qw]
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    def compute_next_quaternion(q_current, q_delta):
        # Changed order: delta * current instead of current * delta
        q_next = quaternion_multiply(q_delta, q_current)
        # Normalize the resulting quaternion
        return q_next / np.linalg.norm(q_next)
    
    ### Stepping function to execute action with robot
    EE_Pose_Node = EndEffectorPoseNode("exec")
    kuka_execution = KukaMotionPlanning(steps)
    waypoint_list = []
    obs_list = []
    for i, pose in enumerate(end_effector_pos):
        waypoint = [float(value) for value in pose]
        position = waypoint[:3]
        rot6d = waypoint[3:9]
        rot_m = rot6d_to_mat(np.array(rot6d))
        quaternion = quat_from_rot_m(rot_m)
        
        if self.action_def == "delta":
            current_pos = EE_Pose_Node.get_fk()
            np.array(quaternion)
            # print(f'action command {end_effector_pos} delta')
            next_rot = compute_next_quaternion(current_pos[3:], quaternion)
            # Create Pose message for IK
            target_pose = Pose()
            target_pose.position.x = current_pos[0] + position[0]
            target_pose.position.y = current_pos[1] +  position[1]
            target_pose.position.z = current_pos[2] +  position[2]
            target_pose.orientation.x = next_rot[0]
            target_pose.orientation.y = next_rot[1]
            target_pose.orientation.z = next_rot[2]
            target_pose.orientation.w = next_rot[3]
        else:
            print(f'action command {end_effector_pos} absolute')
            # Create Pose message for IK
            target_pose = Pose()
            target_pose.position.x = position[0]
            target_pose.position.y = position[1]
            target_pose.position.z = position[2]
            target_pose.orientation.x = quaternion[0]
            target_pose.orientation.y = quaternion[1]
            target_pose.orientation.z = quaternion[2]
            target_pose.orientation.w = quaternion[3]

        # Get IK solution


        joint_state = EE_Pose_Node.get_ik(target_pose)
        if joint_state is None:
            joint_state = EE_Pose_Node.get_ik(target_pose)

        waypoint_list.append(joint_state)

    if joint_state is None:
        EE_Pose_Node.get_logger().warn("Failed to get IK solution")
    else: 
        kuka_execution.send_goal(waypoint_list)
        for position_idx in range(len(end_effector_pos)):
            obs_time = time.time()
            obs = self.get_observation()
            obs_list.append(obs)
            current_gripper_pos = self.robotiq_gripper.get_gripper_current_pose()
            # print(f"gripper current pose {current_gripper_pos}")
            if gripper_action[position_idx] < 0:
                gripper_action[position_idx] = 0.0
            print(gripper_action)
            delta_gripper = abs(current_gripper_pos - gripper_action[position_idx])
            # print(f'delta gripper : {delta_gripper}')
            scaled_command = float(gripper_action[position_idx]*2.0)
            if scaled_command > 0.8:
                scaled_command = 0.8
            self.robotiq_gripper.send_gripper_command(scaled_command)
            if position_idx == len(end_effector_pos)-1:
                time.sleep(0.0)
            elif position_idx == len(end_effector_pos)-2:
                time.sleep(0.01)
            else:
                time.sleep(0.18)


