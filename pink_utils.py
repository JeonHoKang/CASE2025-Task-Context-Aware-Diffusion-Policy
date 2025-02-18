import pink
from pink import solve_ik
from pink.utils import process_collision_pairs
from pink.barriers import SelfCollisionBarrier,PositionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
import meshcat_shapes
import numpy as np
import qpsolvers
import rclpy
import time
from rclpy.node import Node
from kuka_execute import KukaMotionPlanning

from array import array
import pinocchio as pin
import copy
from sensor_msgs.msg import JointState
from submodules.wait_for_message import wait_for_message
from loop_rate_limiters import RateLimiter

import rclpy
from moveit_msgs.msg import RobotState, MoveItErrorCodes, JointConstraint, Constraints
from geometry_msgs.msg import Pose, WrenchStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK

class PinkUtils(Node):
    base_ = "lbr/link_0"
    end_effector_ = "link_ee"
    fk_srv_name_ = "lbr/compute_fk"
    timeout_sec_ = 5.0

    def __init__(self):
        super().__init__("ik_solver")
        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.joint_state_topic_ = "lbr/joint_states"

        srdf_path = "/home/lm-2023/jeon_team_ws/lbr-stack/src/lbr_fri_ros2_stack/lbr_moveit_config/iiwa14_moveit_config/config/iiwa14.srdf"
        print(srdf_path)
        model, visual_model, collision_model = pin.Model(), pin.GeometryModel(), pin.GeometryModel()

        urdf_path = "/home/lm-2023/jeon_team_ws/lbr-stack/src/lbr_fri_ros2_stack/lbr_description/urdf/iiwa14/iiwa14.urdf"

        robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=["."],
            root_joint=None,
        )
        
        print(f"URDF description successfully loaded in {robot}")
        arm_placement = pin.SE3.Identity()
        _, visual_model = pin.appendModel(
            model,
            robot.model,
            visual_model,
            robot.visual_model,
            0,
            arm_placement,
        )
        
        model, collision_model = pin.appendModel(
            model,
            robot.model,
            collision_model,
            robot.collision_model,
            0,
            arm_placement,
        )

        robot = pin.RobotWrapper(
            model,
            collision_model=collision_model,
            visual_model=visual_model,
        )
        # if yourdfpy is None:
        #     print("If you ``pip install yourdfpy``, this example will display it.")
        # else:  # yourdfpy is not None
        #     viz = yourdfpy.URDF.load(urdf_path)
        #     viz.show()
        srdf_path = "/home/lm-2023/jeon_team_ws/lbr-stack/src/lbr_fri_ros2_stack/lbr_moveit_config/iiwa14_moveit_config/config/iiwa14.srdf"
        print(srdf_path)
        

        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        ordered_joint_state = copy.deepcopy(current_joint_state.position)
        ordered_joint_state[2], ordered_joint_state[3] = ordered_joint_state[3], ordered_joint_state[2]
        q_ref1 = np.array(ordered_joint_state)
        # q_ref = np.zeros(robot.model.nq)
        # q_ref[2] += 1.4
        # q_ref[3] -= 1.2
        end_effector_task = FrameTask(
            "link_ee",
            position_cost=10.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        )

        posture_task = PostureTask(
            cost=1e-3,  # [cost] / [rad]
        )
        
        # pos_barrier = PositionBarrier(
        #     "link_ee",
        #     indices=[1],
        #     p_min=np.array([-0.4]),
        #     p_max=np.array([0.6]),
        #     gain=np.array([100.0]),
        #     safe_displacement_gain=1.0,
        # )
        # robot.collision_data = process_collision_pairs(robot.model, robot.collision_model, srdf_path)
        viz = start_meshcat_visualizer(robot)
        viewer = viz.viewer

        meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
        meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

        # barriers = [pos_barrier]

        tasks = [end_effector_task, posture_task]
        
        configuration = pink.Configuration(robot.model, robot.data, q_ref1)
        for task in tasks:
            task.set_target_from_configuration(configuration)
        viz.display(configuration.q)
        
        solver = qpsolvers.available_solvers[0]
        if "osqp" in qpsolvers.available_solvers:
            solver = "osqp"

        quaternion = self.get_fk()[3:]  # [qx, qy, qz, qw]
        position = self.get_fk()[:3]    # [x, y, z]
        # Convert quaternion to Pinocchio format (w, x, y, z)
        orientation = pin.Quaternion(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
        orientation.normalize()
        rate = RateLimiter(frequency=50.0, warn=False)
        dt = rate.period
        t = 0.0  # [s]
        time.sleep(2)
        end_effector_target = end_effector_task.transform_target_to_world
        base_pose = copy.deepcopy(end_effector_target)
        model =robot.model
        data = robot.data
        oMdes = pin.SE3(orientation.matrix(), np.array(position))
        frame_id = model.getFrameId('link_ee')
        pin.forwardKinematics(model, data, q_ref1)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[frame_id]

        # Extract translation (position)
        position = oMf.translation
        print("Position of link_ee:", position)

        # Extract rotation (orientation)
        rotation = oMf.rotation
        print("Orientation of link_ee:\n", rotation)
        # Print out the placement of each joint of the kinematic tree
        for name, oMi in zip(robot.model.names, robot.data.oMi):
            print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
        # Update task targets
        
        end_effector_target.translation[2] = base_pose.translation[2] - 0.02
        end_effector_target.translation[1] = base_pose.translation[1] + 0.02
        end_effector_target.translation[0] = base_pose.translation[0] + 0.02

        # end_effector_target.rotation[1] = base_pose.rotation[1] + 0.005

        # end_effector_target.translation[2] += 0.001
        
        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )
        start_t = time.time()
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=None,
        )

        configuration.integrate_inplace(velocity, dt)
        print(configuration.q)
        end_t = time.time()
        duration = end_t - start_t
        print(f"duration for ik: {duration}")
        fri_vel = copy.deepcopy(velocity)
        fri_vel[2], fri_vel[3] = fri_vel[3], fri_vel[2]

        # G, h = pos_barrier.compute_qp_inequalities(configuration, dt=dt)
        # distance_to_manipulator = configuration.get_transform_frame_to_world(
        #     "link_ee"
        # ).translation[1]
        # if args.verbose:
        #     print(
        #         f"Task error: {end_effector_task.compute_error(configuration)}"
        #     )
        #     print(
        #         "Position CBF value: "
        #         f"{pos_barrier.compute_barrier(configuration)[0]:0.3f} >= 0"
        #     )
        #     print(f"Distance to manipulator: {distance_to_manipulator} <= 0.6")
        #     print("-" * 60)
        goal_state = copy.deepcopy(configuration.q)
        goal_state[2], goal_state[3] = goal_state[3], goal_state[2]
        pin.computeMinverse(model, data, q_ref1)
        # pin.aba(model, data, q_ref1, velocity,)
        # # Visualize result at fixed FPS
        KukaMotionPlanning(1).send_goal_pink(array('d', configuration.q), array('d', velocity))
        # for acc in data.a:
        #     print(acc)

    def get_fk(self) -> Pose | None:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        print("joint stat current: ", current_joint_state)
        if not current_joint_state_set:
            self.get_logger().warn("Failed to get current joint state")
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
    
def test(args=None):
    rclpy.init(args=args)

    pink_utils = PinkUtils()

test()