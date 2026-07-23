#!/usr/bin/env python3
"""
obstacle_avoidance_pipeline.py -- standalone, additive.

Combines, into one runnable end-to-end script, the two phases built and
verified separately earlier in this project:
  1. Camera-based obstacle detection (see scan_obstacles_from_cameras.py)
  2. MoveIt/OMPL collision-free path planning + execution (see
     plan_obstacle_avoiding_path.py)

Flow: move to scan pose -> capture both wrist cameras once -> detect the
two static obstacles' world-frame positions via point-cloud clustering ->
register them with MoveIt -> move to the planning start pose -> plan a
collision-free path to the goal pose using the DETECTED (not hardcoded)
obstacle positions -> execute it.

Run it the same way as this project's optimizer scripts:

    cd ~/ros2_ws/src/ur10_perception/scripts
    python3 obstacle_avoidance_pipeline.py

Requires GUI-mode Gazebo (camera rendering does not work headless on this
machine -- confirmed 2026-07-15). Thermal note: GUI mode + 2 active
cameras has been measured up to ~97-99 C CPU during extended sessions --
keep an eye on `sensors` while this runs.

This file is self-contained: it does not import from or modify
scan_obstacles_from_cameras.py, plan_obstacle_avoiding_path.py, or any
optimizer/baseline script.
"""

import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import (
    CollisionObject, PlanningScene, Constraints, JointConstraint,
    MotionPlanRequest,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Scan pose: found via a /compute_fk-based search for a wrist orientation
# whose camera look-axis (local +X) points at the obstacle region -- see
# scan_obstacles_from_cameras.py / CLAUDE.md for the full derivation.
SCAN_POSE = [0.0, -1.9824, 1.1454, -1.0228, -0.5107, 0.0]

# Planning start/goal: Path 1's I-pose -> Path 1's goal pose.
START_POSE = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
GOAL_POSE  = [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0]

PLANNING_GROUP = "ur_manipulator"
PLANNING_FRAME = "world"

WORKSPACE_BOUNDS = {  # world-frame box used to reject robot-body/ground/noise points
    'x': (0.10, 1.00),
    'y': (-0.70, 0.70),
    'z': (0.10, 1.00),
}
N_EXPECTED_OBSTACLES = 2
MIN_POINTS_FOR_DETECTION = 20
DETECTION_PADDING_M = 0.02  # per-axis safety margin, see cluster_obstacles()

GROUND_TRUTH = {  # for logging/validation only -- not used in detection
    'pipe_obstacle_1':       (0.55, 0.25, 0.50),
    'hull_panel_obstacle_2': (0.30, -0.40, 0.40),
}

JOINT_TOLERANCE = 0.01
PLANNING_TIME_S = 5.0
PLANNING_ATTEMPTS = 20
VELOCITY_SCALING = 0.3
ACCELERATION_SCALING = 0.3
MAX_PLAN_RETRIES = 5  # OMPL is randomized -- observed ~50% success per call

LOG_DIR = os.path.expanduser('~/ros2_ws/logs/obstacle_avoidance_pipeline')


class ObstacleAvoidancePipelineNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance_pipeline')
        self._client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        self._joint_states = None
        self._logged_data = []
        self._logging = False
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self._cloud_right = None
        self._cloud_left = None
        self._cam_right_sub = self.create_subscription(
            PointCloud2, '/camera/depth/right/points', self._cam_right_cb, 5)
        self._cam_left_sub = self.create_subscription(
            PointCloud2, '/camera/depth/left/points', self._cam_left_cb, 5)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

        os.makedirs(LOG_DIR, exist_ok=True)

    # --- callbacks ---
    def _js_callback(self, msg):
        self._joint_states = msg
        if self._logging:
            self._log_joint_state(msg)

    def _cam_right_cb(self, msg):
        self._cloud_right = msg

    def _cam_left_cb(self, msg):
        self._cloud_left = msg

    def _log_joint_state(self, msg):
        row = {'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for j, jname in enumerate(JOINT_NAMES):
            idx = name_to_idx.get(jname, -1)
            row[f'position_{j}'] = msg.position[idx] if idx >= 0 and idx < len(msg.position) else 0.0
            row[f'velocity_{j}'] = msg.velocity[idx] if idx >= 0 and idx < len(msg.velocity) else 0.0
            row[f'effort_{j}']   = msg.effort[idx]   if idx >= 0 and idx < len(msg.effort)   else 0.0
        self._logged_data.append(row)

    def save_log(self, name):
        if not self._logged_data:
            self.get_logger().warn('No data to save!')
            return None
        timestamp = int(time.time())
        filepath = os.path.join(LOG_DIR, f'{name}_{timestamp}.csv')
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)
        self.get_logger().info(f'Saved: {filepath}')
        return filepath

    # --- motion helpers ---
    def move_direct(self, target_pose, duration=8.0, plan_name='move'):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = target_pose
        pt.velocities = [0.0] * 6
        pt.accelerations = [0.0] * 6
        pt.time_from_start = Duration(seconds=duration).to_msg()
        traj.points = [pt]
        return self._send_and_wait(traj, plan_name)

    def wait_for_pose(self, pose, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._joint_states is None:
                continue
            name_to_idx = {n: i for i, n in enumerate(self._joint_states.name)}
            errs = []
            for j, jname in enumerate(JOINT_NAMES):
                idx = name_to_idx.get(jname, -1)
                if idx >= 0:
                    errs.append(abs(self._joint_states.position[idx] - pose[j]))
            if errs and max(errs) < 0.05:
                return True
            time.sleep(0.5)
        self.get_logger().warn('  wait_for_pose timed out -- proceeding anyway')
        return False

    def _send_and_wait(self, traj, plan_name):
        self._logged_data = []
        self._logging = True
        start_time = time.time()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._logging = False
            self.get_logger().error(f'{plan_name}: goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.time() + 60.0
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.5)
            if time.time() > deadline:
                self._logging = False
                self.get_logger().error(f'{plan_name}: execution timeout')
                return False

        self._logging = False
        duration = time.time() - start_time
        result_response = result_future.result()
        status = result_response.status if result_response else None
        error_code = result_response.result.error_code if result_response else None
        success = (status == GoalStatus.STATUS_SUCCEEDED) and (error_code == 0)

        if not success:
            self.get_logger().error(
                f'{plan_name} FAILED (status={status}, error_code={error_code}) '
                f'after {duration:.2f}s, {len(self._logged_data)} samples'
            )
            self.save_log(plan_name + '_FAILED')
            return False

        self.get_logger().info(
            f'{plan_name} completed SUCCESSFULLY in {duration:.2f}s, '
            f'{len(self._logged_data)} samples'
        )
        self.save_log(plan_name)
        return True

    # --- Phase 1: detection ---
    def wait_for_clouds(self, timeout=20.0):
        deadline = time.time() + timeout
        self._cloud_right = None
        self._cloud_left = None
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._cloud_right is not None and self._cloud_left is not None:
                return True
        return False

    def cloud_to_world_points(self, msg, source_frame):
        """source_frame is passed explicitly rather than trusting
        msg.header.frame_id, which this Gazebo sensor plugin publishes
        mismatched to any real TF frame (confirmed 2026-07-15)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                PLANNING_FRAME, source_frame, rclpy.time.Time(),
                timeout=Duration(seconds=2.0))
        except Exception as e:
            self.get_logger().error(f'  TF lookup {PLANNING_FRAME}<-{source_frame} failed: {e}')
            return np.zeros((0, 3))

        # read_points() returns a structured array (named x/y/z fields),
        # not plain tuples -- extract by name into a plain (N,3) array.
        cloud_arr = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if cloud_arr.size == 0:
            return np.zeros((0, 3))
        arr = np.column_stack(
            [cloud_arr['x'], cloud_arr['y'], cloud_arr['z']]
        ).astype(np.float64)
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        if len(arr) == 0:
            return arr

        t = tf.transform.translation
        q = tf.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return (rot @ arr.T).T + np.array([t.x, t.y, t.z])

    def filter_workspace(self, points):
        if len(points) == 0:
            return points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x >= WORKSPACE_BOUNDS['x'][0]) & (x <= WORKSPACE_BOUNDS['x'][1]) &
            (y >= WORKSPACE_BOUNDS['y'][0]) & (y <= WORKSPACE_BOUNDS['y'][1]) &
            (z >= WORKSPACE_BOUNDS['z'][0]) & (z <= WORKSPACE_BOUNDS['z'][1])
        )
        return points[mask]

    def cluster_obstacles(self, points, k=N_EXPECTED_OBSTACLES):
        """k-means into k clusters (k known in advance: 2 static obstacles).
        Returns list of dicts: centroid, yaw, size (oriented box, see
        below), aabb_min, aabb_max, n_points."""
        if len(points) < MIN_POINTS_FOR_DETECTION:
            return []
        centroids, labels = kmeans2(points, k, seed=42, minit='++')
        clusters = []
        for i in range(k):
            cluster_pts = points[labels == i]
            if len(cluster_pts) == 0:
                continue
            aabb_min = cluster_pts.min(axis=0)
            aabb_max = cluster_pts.max(axis=0)

            # Oriented (yaw-only) footprint via PCA on the cluster's XY
            # points, rather than a plain world-axis-aligned box. Confirmed
            # 2026-07-16: hull_panel_obstacle_2 is rotated 0.6 rad in the
            # world, so its true 0.05m thickness inflates to ~0.27m once
            # enclosed axis-aligned -- that extra bulk was enough to make
            # MoveIt's start-state collision check reject every planning
            # attempt instantly.
            #
            # BUT: for a near-square footprint (pipe_obstacle_1), the two
            # eigenvalues are nearly equal, so PCA's chosen axis is
            # essentially arbitrary/noise-driven -- confirmed 2026-07-16
            # this can inflate a true 0.15x0.15 box to ~0.21x0.21 (almost
            # double the area), which is exactly what broke planning again
            # after "fixing" the panel. So: compute BOTH the axis-aligned
            # box and the PCA-oriented box, and keep whichever has the
            # smaller footprint area -- this is never worse than a plain
            # axis-aligned box, while still capturing genuine rotation
            # (like the panel's) when PCA's axis is actually meaningful.
            xy = cluster_pts[:, :2]
            mean_xy = xy.mean(axis=0)
            centered = xy - mean_xy
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, np.argmax(eigvals)]
            pca_yaw = float(np.arctan2(principal[1], principal[0]))

            c_, s_ = np.cos(-pca_yaw), np.sin(-pca_yaw)
            to_local = np.array([[c_, -s_], [s_, c_]])
            local_xy = centered @ to_local.T
            pca_size_xy = local_xy.max(axis=0) - local_xy.min(axis=0)
            pca_local_center = (local_xy.max(axis=0) + local_xy.min(axis=0)) / 2.0
            c2, s2 = np.cos(pca_yaw), np.sin(pca_yaw)
            to_world = np.array([[c2, -s2], [s2, c2]])
            pca_center_xy = to_world @ pca_local_center + mean_xy

            aabb_size_xy = xy.max(axis=0) - xy.min(axis=0)
            aabb_center_xy = (xy.max(axis=0) + xy.min(axis=0)) / 2.0

            if (pca_size_xy[0] * pca_size_xy[1]) < (aabb_size_xy[0] * aabb_size_xy[1]):
                yaw = pca_yaw
                size_xy = pca_size_xy
                world_center_xy = pca_center_xy
            else:
                yaw = 0.0
                size_xy = aabb_size_xy
                world_center_xy = aabb_center_xy

            z_min, z_max = float(aabb_min[2]), float(aabb_max[2])

            clusters.append({
                'centroid': [float(world_center_xy[0]), float(world_center_xy[1]),
                             (z_min + z_max) / 2.0],
                'yaw': yaw,
                # DETECTION_PADDING_M (not the earlier 0.05m): move_group's
                # own log (2026-07-16) showed GOAL_POSE colliding with the
                # padded pipe box at 'upper_arm_link' -- a collision the
                # unpadded, hardcoded reference box in
                # plan_obstacle_avoiding_path.py doesn't have. Detection is
                # accurate to <1cm (measured 0.000-0.009m error across
                # runs), so a 5cm margin was pure excess that closed off
                # real, existing clearance. 2cm leaves comfortable headroom
                # over the measured accuracy without doing that.
                'size': [max(DETECTION_PADDING_M, float(size_xy[0])) + DETECTION_PADDING_M,
                         max(DETECTION_PADDING_M, float(size_xy[1])) + DETECTION_PADDING_M,
                         max(DETECTION_PADDING_M, z_max - z_min) + DETECTION_PADDING_M],
                'aabb_min': aabb_min.tolist(),
                'aabb_max': aabb_max.tolist(),
                'n_points': int(len(cluster_pts)),
            })
        return clusters

    def register_detected_obstacles(self, clusters):
        if not self._scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/apply_planning_scene not available')
            return False
        objs = []
        for i, c in enumerate(clusters):
            obj = CollisionObject()
            obj.header.frame_id = PLANNING_FRAME
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = f'detected_obstacle_{i}'
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = c['size']
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = c['centroid']
            pose.orientation.z = math.sin(c['yaw'] / 2.0)
            pose.orientation.w = math.cos(c['yaw'] / 2.0)
            obj.primitives.append(primitive)
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD
            objs.append(obj)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objs
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        success = result is not None and result.success
        self.get_logger().info(
            f'  Registered {len(objs)} detected obstacles with MoveIt: '
            f'{"OK" if success else "FAILED"}'
        )
        return success

    # --- Phase 2: planning ---
    def plan_path(self, goal_pose):
        if not self._plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/plan_kinematic_path not available')
            return None

        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = PLANNING_GROUP
        mpr.num_planning_attempts = PLANNING_ATTEMPTS
        mpr.allowed_planning_time = PLANNING_TIME_S
        mpr.max_velocity_scaling_factor = VELOCITY_SCALING
        mpr.max_acceleration_scaling_factor = ACCELERATION_SCALING
        mpr.start_state.joint_state = self._joint_states

        constraints = Constraints()
        for name, pos in zip(JOINT_NAMES, goal_pose):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = JOINT_TOLERANCE
            jc.tolerance_below = JOINT_TOLERANCE
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        mpr.goal_constraints = [constraints]

        req.motion_plan_request = mpr
        future = self._plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=PLANNING_TIME_S + 10.0)
        resp = future.result()
        if resp is None:
            self.get_logger().error('Planning call failed (no response)')
            return None

        mpres = resp.motion_plan_response
        if mpres.error_code.val != 1:  # 1 == SUCCESS
            self.get_logger().error(f'Planning FAILED, error_code={mpres.error_code.val}')
            return None

        traj = mpres.trajectory.joint_trajectory
        self.get_logger().info(
            f'Planning SUCCEEDED: {len(traj.points)} waypoints, '
            f'planning_time={mpres.planning_time:.3f}s'
        )
        return traj


def validate_against_ground_truth(clusters):
    """Match each detected cluster to its nearest ground-truth obstacle and
    report position error -- for honesty about detection accuracy, not used
    to influence the detection itself."""
    results = []
    gt_items = list(GROUND_TRUTH.items())
    for c in clusters:
        centroid = np.array(c['centroid'])
        best_name, best_dist = None, float('inf')
        for name, gt_pos in gt_items:
            dist = float(np.linalg.norm(centroid - np.array(gt_pos)))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        results.append({
            'detected_centroid': c['centroid'],
            'n_points': c['n_points'],
            'nearest_ground_truth': best_name,
            'position_error_m': best_dist,
        })
    return results


def main():
    rclpy.init()
    node = ObstacleAvoidancePipelineNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node._client.wait_for_server()

    node.get_logger().info('=' * 70)
    node.get_logger().info('  OBSTACLE-AVOIDANCE PIPELINE: scan -> detect -> plan -> execute')
    node.get_logger().info('=' * 70)

    # --- Phase 1/4: scan + detect ---
    node.get_logger().info('=== Phase 1/4: moving to scan pose ===')
    node.move_direct(SCAN_POSE, duration=8.0, plan_name='to_scan_pose')
    node.wait_for_pose(SCAN_POSE)

    node.get_logger().info('Waiting for both camera point clouds...')
    if not node.wait_for_clouds(timeout=20.0):
        node.get_logger().error(
            'Timed out waiting for camera point clouds -- aborting pipeline. '
            'Check both cameras are publishing: '
            'ros2 topic hz /camera/depth/right/points /camera/depth/left/points'
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    pts_right = node.cloud_to_world_points(node._cloud_right, 'camera_link_right')
    pts_left = node.cloud_to_world_points(node._cloud_left, 'camera_link_left')
    node.get_logger().info(f'  right camera: {len(pts_right)} finite points')
    node.get_logger().info(f'  left camera : {len(pts_left)} finite points')

    all_pts = (np.concatenate([pts_right, pts_left], axis=0)
               if (len(pts_right) + len(pts_left)) > 0 else np.zeros((0, 3)))
    filtered = node.filter_workspace(all_pts)
    node.get_logger().info(f'  {len(filtered)} points remain after workspace-bounds filtering')

    clusters = node.cluster_obstacles(filtered)
    if not clusters:
        node.get_logger().error(
            f'Fewer than {MIN_POINTS_FOR_DETECTION} points after filtering -- '
            'no obstacles detected. Aborting pipeline.'
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    validation = validate_against_ground_truth(clusters)
    node.get_logger().info(f'Detected {len(clusters)} cluster(s):')
    for v in validation:
        node.get_logger().info(
            f"  centroid={[round(x,3) for x in v['detected_centroid']]} "
            f"n_points={v['n_points']} nearest_gt={v['nearest_ground_truth']} "
            f"error={v['position_error_m']:.3f}m"
        )

    # --- Phase 2/4: register detected obstacles ---
    node.get_logger().info('=== Phase 2/4: registering detected obstacles with MoveIt ===')
    if not node.register_detected_obstacles(clusters):
        node.get_logger().error('Could not register detected obstacles -- aborting pipeline')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # --- Phase 3/4: move to start pose + plan ---
    node.get_logger().info('=== Phase 3/4: moving to start pose + planning collision-free path ===')
    if not node.move_direct(START_POSE, duration=8.0, plan_name='to_start_pose'):
        node.get_logger().error('Could not reach start pose -- aborting pipeline')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    time.sleep(2.0)

    traj = None
    for attempt in range(1, MAX_PLAN_RETRIES + 1):
        traj = node.plan_path(GOAL_POSE)
        if traj is not None:
            break
        node.get_logger().warn(f'Planning attempt {attempt}/{MAX_PLAN_RETRIES} failed, retrying...')
    if traj is None:
        node.get_logger().error(f'Planning failed after {MAX_PLAN_RETRIES} attempts -- aborting pipeline')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # --- Phase 4/4: execute ---
    node.get_logger().info('=== Phase 4/4: executing planned (collision-free) path ===')
    success = node._send_and_wait(traj, 'obstacle_avoiding_path')

    with open(os.path.join(LOG_DIR, 'pipeline_result.json'), 'w') as f:
        json.dump({
            'clusters': clusters,
            'validation': validation,
            'planning_waypoints': len(traj.points),
            'execution_success': success,
        }, f, indent=2)

    node.get_logger().info(f'Pipeline complete! success={success}  Logs: {LOG_DIR}/')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
