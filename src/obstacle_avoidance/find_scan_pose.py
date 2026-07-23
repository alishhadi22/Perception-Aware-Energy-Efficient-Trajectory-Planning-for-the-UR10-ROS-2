#!/usr/bin/env python3
"""Search for a wrist orientation where camera_link_right's local +Y axis
(confirmed via the actual Gazebo test to be the camera's effective look
direction) points toward the obstacle region. Uses MoveIt's /compute_fk
service only -- no robot motion, no camera rendering."""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
I_POSE = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

# Obstacle centroid (average of the two known obstacles)
OBSTACLE_CENTROID = np.array([0.425, -0.075, 0.45])


def main():
    rclpy.init()
    node = Node('find_scan_pose')
    client = node.create_client(GetPositionFK, '/compute_fk')
    if not client.wait_for_service(timeout_sec=15.0):
        print("compute_fk service not available")
        sys.exit(1)

    def fk(joint_angles, link_name='camera_link_right'):
        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.fk_link_names = [link_name]
        req.robot_state.joint_state.name = JOINT_NAMES
        req.robot_state.joint_state.position = list(joint_angles)
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not resp.pose_stamped:
            return None, None
        pose = resp.pose_stamped[0].pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rot = Rotation.from_quat(quat).as_matrix()
        return pos, rot

    # First: cross-validate against the known tf2_echo measurement at I-pose
    pos, rot = fk(I_POSE, link_name='wrist_3_link')
    print("=== Cross-check vs tf2_echo ground truth (wrist_3_link @ I-pose) ===")
    print(f"compute_fk translation: {pos}")
    print(f"expected (tf2_echo):    [0.001, 0.256, 1.427]")
    print(f"compute_fk rotation:\n{rot}")
    print(f"expected Y-column (local Y -> world): should be ~[-0.005, -0.000, -1.000]")
    print()

    cam_pos, cam_rot = fk(I_POSE, link_name='camera_link_right')
    print("=== camera_link_right @ I-pose (the pose that showed ground-plane data) ===")
    print(f"position: {cam_pos}")
    print(f"local Y axis -> world: {cam_rot[:, 1]}  (confirmed look direction from real test)")
    print()

    # Now search: vary shoulder_lift, elbow, wrist_1, wrist_2 (keep shoulder_pan=0,
    # wrist_3 free too) to point camera_link_right's local Y axis at the obstacles.
    # Bounded to keep the arm in a physically sensible, forward-reaching, above-
    # ground posture -- an earlier unbounded search found a mathematically valid
    # but absurd solution with the camera below the floor (z=-0.64) behind the
    # robot base, which is not a usable pose.
    BOUNDS = [(-2.5, 0.0), (0.3, 2.5), (-3.0, 0.0), (-3.0, 0.0)]

    def objective(x):
        angles = [0.0, x[0], x[1], x[2], x[3], 0.0]
        pos, rot = fk(angles)
        if pos is None:
            return 1e6
        target_dir = OBSTACLE_CENTROID - pos
        dist = np.linalg.norm(target_dir)
        target_dir = target_dir / dist
        # CORRECTED (2026-07-15): local X is the camera's actual look axis,
        # not local Y. Confirmed by a real Gazebo test at the local-Y-optimized
        # candidate pose showing the camera facing the WRONG way; cross-checking
        # /compute_fk's local X mapping at that same pose against the observed
        # ground-plane point cloud direction matched clearly, local Y did not.
        look_dir = rot[:, 0]  # local X axis in world
        cos_sim = np.dot(look_dir, target_dir)
        orientation_cost = 1.0 - cos_sim  # 0 = perfectly aligned

        # Soft penalties for physically nonsensical solutions
        penalty = 0.0
        if pos[2] < 0.2:       # camera below/near the floor
            penalty += (0.2 - pos[2]) * 5.0
        if pos[0] < -0.1:      # camera behind the robot base, facing away from obstacles
            penalty += abs(pos[0]) * 5.0
        return orientation_cost + penalty

    x0 = np.array([-1.0, 1.5708, -1.5708, -1.5708])  # start near I-pose-ish, arm forward
    result = minimize(objective, x0, method='Nelder-Mead', bounds=BOUNDS,
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})

    best_angles = [0.0, result.x[0], result.x[1], result.x[2], result.x[3], 0.0]
    pos, rot = fk(best_angles)
    target_dir = OBSTACLE_CENTROID - pos
    target_dir = target_dir / np.linalg.norm(target_dir)
    look_dir = rot[:, 0]  # local X -- corrected axis
    angle_error_deg = np.degrees(np.arccos(np.clip(np.dot(look_dir, target_dir), -1, 1)))

    print("=== SEARCH RESULT ===")
    print(f"success: {result.success}, final objective: {result.fun:.6f}")
    print(f"best joint angles: {[round(a,4) for a in best_angles]}")
    print(f"camera position: {pos}")
    print(f"look direction (local X in world): {look_dir}")
    print(f"target direction (toward obstacle centroid): {target_dir}")
    print(f"angle error: {angle_error_deg:.2f} degrees")
    print(f"distance from camera to obstacle centroid: {np.linalg.norm(OBSTACLE_CENTROID - pos):.3f} m")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
