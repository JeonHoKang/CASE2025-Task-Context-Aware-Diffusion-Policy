from geometry_msgs.msg import Pose, Point, Quaternion

import numpy as np
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation

def check_same_pose(pose_1: Pose, pose_2: Pose, pos_threshold=0.001, angle_threshold=np.deg2rad(0.1)) -> bool:
    # Position difference
    pos_diff = np.array([
        pose_1.position.x - pose_2.position.x,
        pose_1.position.y - pose_2.position.y,
        pose_1.position.z - pose_2.position.z
    ])
    pos_dist = np.linalg.norm(pos_diff)

    # Quaternion difference using rotation matrix
    q1 = [pose_1.orientation.x, pose_1.orientation.y, pose_1.orientation.z, pose_1.orientation.w]
    q2 = [pose_2.orientation.x, pose_2.orientation.y, pose_2.orientation.z, pose_2.orientation.w]

    # Convert to rotation matrix and compute angular difference
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    relative_rotation = r1.inv() * r2
    angle_diff = relative_rotation.magnitude()  # Angular distance

    # Print differences for debugging
    print(f"Position Difference: {pos_dist}, Orientation Angle Difference: {np.rad2deg(angle_diff)} degrees")

    # Return True only if both position and orientation thresholds are satisfied
    return pos_dist < pos_threshold and angle_diff < angle_threshold

# def check_same_pose(
#         pose_1: Pose, pose_2: Pose
# ) -> bool:
#     x_diff = np.abs(pose_1.position.x - pose_2.position.x)
#     y_diff = np.abs(pose_1.position.y - pose_2.position.y)
#     z_diff = np.abs(pose_1.position.z - pose_2.position.z)

#     pos_diff = np.asarray([x_diff, y_diff, z_diff])

#     qx_diff = np.abs(pose_1.orientation.x - pose_2.orientation.x)
#     qy_diff = np.abs(pose_1.orientation.y - pose_2.orientation.y)
#     qz_diff = np.abs(pose_1.orientation.z - pose_2.orientation.z)
#     qw_diff = np.abs(pose_1.orientation.w - pose_2.orientation.w)

#     quat_diff = np.asarray([qx_diff, qy_diff, qz_diff, qw_diff])
#     print(f"Position Diff: , {pos_diff}\ Orientation Diff : {quat_diff}")
#     return np.sum(np.square(pos_diff)) < 0.001 and np.sum(np.square(quat_diff)) < 0.001


# def construct_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
#     transform = np.identity(4)
#     transform[0:3, 0:3] = rotation
#     transform[0:3, [3]] = translation
    
#     return transform


def construct_pose(transform: np.ndarray) -> Pose:
    rotation = transform[0:3, 0:3]
    translation = transform[0:3, 3].reshape(1, 3)

    # scipy quat defination: quat = [x, y, z, w]
    quat = Rotation.from_matrix(rotation).as_quat()

    return Pose(
        position=Point(
            x = translation[0][0],
            y = translation[0][1],
            z = translation[0][2],
        ),
        orientation=Quaternion(
            w = quat[3],
            x = quat[0],
            y = quat[1],
            z = quat[2]
        )
    )


def get_offset_pose(base_pose: Pose, offset_dict: dict[str, float]) -> Pose:
    offset_rotation = Rotation.from_euler(
        seq="ZYX", 
        angles=[offset_dict['A'], offset_dict['B'], offset_dict['C']],
        degrees=True
    ).as_matrix()
    offset_translation = np.asarray([offset_dict['X'], offset_dict['Y'], offset_dict['Z']]).reshape(3, 1)

    transform = construct_transform_matrix(offset_rotation, offset_translation)

    base_rotation = Rotation.from_quat(
        [base_pose.orientation.x, base_pose.orientation.y, base_pose.orientation.z, base_pose.orientation.w]
    ).as_matrix()
    base_translation = np.asarray([base_pose.position.x, base_pose.position.y, base_pose.position.z]).reshape(3, 1)

    base = construct_transform_matrix(base_rotation, base_translation)

    offset_transform = np.matmul(base, transform)

    return construct_pose(offset_transform)


def frange(start, stop, step, n=None):
    """return a WYSIWYG series of float values that mimic range behavior
    by excluding the end point and not printing extraneous digits beyond
    the precision of the input numbers (controlled by n and automatically
    detected based on the string representation of the numbers passed).

    EXAMPLES
    ========

    non-WYSIWYS simple list-comprehension

    >>> [.11 + i*.1 for i in range(3)]
    [0.11, 0.21000000000000002, 0.31]

    WYSIWYG result for increasing sequence

    >>> list(frange(0.11, .33, .1))
    [0.11, 0.21, 0.31]

    and decreasing sequences

    >>> list(frange(.345, .1, -.1))
    [0.345, 0.245, 0.145]

    To hit the end point for a sequence that is divisibe by
    the step size, make the end point a little bigger by
    adding half the step size:

    >>> dx = .2
    >>> list(frange(0, 1 + dx/2, dx))
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    """
    if step == 0:
        raise ValueError('step must not be 0')
    # how many decimal places are showing?
    if n is None:
        n = max([0 if '.' not in str(i) else len(str(i).split('.')[1])
                for i in (start, stop, step)])
    if step*(stop - start) > 0:  # a non-null incr/decr range
        if step < 0:
            for i in frange(-start, -stop, -step, n):
                yield -i
        else:
            steps = round((stop - start)/step)
            while round(step*steps + start, n) < stop:
                steps += 1
            for i in range(steps):
                yield round(start + i*step, n)