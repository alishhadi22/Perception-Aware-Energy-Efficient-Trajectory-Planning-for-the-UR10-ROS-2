#!/usr/bin/env python3
"""
scan_obstacles_from_cameras.py -- EXPERIMENTAL, standalone.

Idea: since the 2 known obstacles (pipe_obstacle_1, hull_panel_obstacle_2)
are static, there is no need for continuous live perception (the thing that
never worked reliably in the earlier Octomap debugging). Instead: move to
one scan pose, capture both wrist-mounted depth cameras ONCE, detect the
obstacles' world-frame positions via simple point-cloud clustering, then
register them as MoveIt CollisionObjects -- exactly like
add_obstacles_to_planning_scene.py already does, except the positions come
from actual camera detection instead of hardcoded ground-truth values.

This file is intentionally isolated: it does not import from or modify any
optimizer script, baseline script, or add_obstacles_to_planning_scene.py.
If this approach doesn't pan out, deleting this one file removes it cleanly.

Requires GUI-mode Gazebo (headless mode's EGL/DRI2 rendering path was found
to produce zero camera data on this machine -- confirmed 2026-07-15).
GUI mode + 2 active cameras was measured at ~97-98 C CPU temp within about
a minute, even with the robot stationary -- this script is deliberately
built to capture ONE scan and then let the caller shut Gazebo down
immediately after, not to run for an extended period.

Known ground-truth positions (for validating detection accuracy only --
NOT used in the detection itself):
  pipe_obstacle_1:       (0.55, 0.25, 0.50)
  hull_panel_obstacle_2: (0.30, -0.40, 0.40)
"""

import json
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
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
# NOT the I-pose. The I-pose was tried first and confirmed (by an actual
# camera test) to point the camera at the ground plane, not the obstacles.
# A first attempt to fix this assumed the camera's look axis was
# camera_link's local +Y -- WRONG, disproven by a second real camera test
# (the "fixed" pose still showed ground-plane data, facing backward).
# Cross-checking /compute_fk's local X vs local Y mapping against that
# test's actual point-cloud direction confirmed the true look axis is
# local +X (the standard/conventional axis, as originally guessed before
# the incorrect "correction"). This SCAN_POSE was found via a bounded
# /compute_fk-based search (no robot motion) for a wrist orientation whose
# local +X axis points at the obstacle region: 0.00 degree angle error,
# camera at (0.253, 0.264, 1.029) -- well above the floor -- 0.69m from
# the obstacle centroid (within the camera's 0.15-5.0m clip range). Not
# yet verified with a real camera test (paused for a thermal break after
# a lot of cycling) -- see CLAUDE.md's Camera-Based Obstacle Detection
# section for the full history of what was tried and ruled out.
SCAN_POSE = [0.0, -1.9824, 1.1454, -1.0228, -0.5107, 0.0]

PLANNING_FRAME = "world"
WORKSPACE_BOUNDS = {  # world-frame box used to reject robot-body/ground/noise points
    'x': (0.10, 1.00),
    'y': (-0.70, 0.70),
    'z': (0.10, 1.00),
}
N_EXPECTED_OBSTACLES = 2
MIN_POINTS_FOR_DETECTION = 20

GROUND_TRUTH = {
    'pipe_obstacle_1':       (0.55, 0.25, 0.50),
    'hull_panel_obstacle_2': (0.30, -0.40, 0.40),
}

LOG_DIR = os.path.expanduser('~/ros2_ws/logs/obstacle_scan')


class ObstacleScanNode(Node):

    def __init__(self):
        super().__init__('obstacle_scan_experimental')
        self._client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        self._joint_states = None
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

        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')

        os.makedirs(LOG_DIR, exist_ok=True)

    def _js_callback(self, msg):
        self._joint_states = msg

    def _cam_right_cb(self, msg):
        self._cloud_right = msg

    def _cam_left_cb(self, msg):
        self._cloud_left = msg

    def move_to_pose(self, pose, duration=8.0):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = pose
        pt.velocities = [0.0] * 6
        pt.accelerations = [0.0] * 6
        pt.time_from_start = Duration(seconds=duration).to_msg()
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        gh = future.result()
        if gh and gh.accepted:
            result_future = gh.get_result_async()
            deadline = time.time() + duration + 15.0
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.5)
                if time.time() > deadline:
                    self.get_logger().warn('  move_to_pose timeout -- continuing anyway')
                    break
        time.sleep(2.0)

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
                self.get_logger().info(f'  At scan pose (max_err={max(errs):.4f} rad)')
                return True
            time.sleep(0.5)
        self.get_logger().warn('  wait_for_pose timed out -- proceeding anyway')
        return False

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
        """Transform a PointCloud2 message into an (N,3) numpy array in the
        PLANNING_FRAME, filtering NaN/inf (no-return depth pixels).

        source_frame is passed explicitly rather than trusting
        msg.header.frame_id -- confirmed (2026-07-15) that this Gazebo
        sensor plugin publishes point clouds with a header.frame_id like
        "ur/wrist_3_link/depth_camera_right" that does not match any real
        published TF frame; the actual TF frames are camera_link_right/
        camera_link_left (children of wrist_3_link).
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                PLANNING_FRAME, source_frame, rclpy.time.Time(),
                timeout=Duration(seconds=2.0))
        except Exception as e:
            self.get_logger().error(f'  TF lookup {PLANNING_FRAME}<-{source_frame} failed: {e}')
            return np.zeros((0, 3))

        # Extract XYZ only and transform manually -- tf2_sensor_msgs'
        # do_transform_cloud() round-trips ALL of the message's original
        # PointFields through create_cloud(), which raised a dtype
        # mismatch AssertionError with this Gazebo bridge's field layout
        # (confirmed 2026-07-15). Since only XYZ is needed here, applying
        # the rotation/translation directly to a plain (N,3) array sidesteps
        # that entirely.
        # read_points() returns a structured array (named fields x/y/z,
        # each float32), NOT a list of plain tuples -- confirmed
        # 2026-07-15 (an earlier version of this code assumed plain
        # tuples and got a dtype cast TypeError). Extract fields by name
        # into a plain (N,3) float64 array instead.
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
        world_pts = (rot @ arr.T).T + np.array([t.x, t.y, t.z])
        return world_pts

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
        Returns list of dicts: centroid, aabb_min, aabb_max, n_points."""
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
            clusters.append({
                # AABB midpoint, NOT the raw point-cloud mean. Verified
                # 2026-07-15: for pipe_obstacle_1, the raw mean was biased
                # 0.22m off (the camera sees more surface area on the
                # object's upper-facing side, skewing the mean upward)
                # while the AABB itself was captured almost perfectly
                # (matched real geometry to ~1mm) -- its midpoint gave a
                # 0.15mm error, essentially exact.
                'centroid': ((aabb_min + aabb_max) / 2.0).tolist(),
                'aabb_min': aabb_min.tolist(),
                'aabb_max': aabb_max.tolist(),
                'n_points': int(len(cluster_pts)),
            })
        return clusters

    def register_collision_objects(self, clusters):
        if not self._scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/apply_planning_scene not available -- skipping registration')
            return False

        objs = []
        for i, c in enumerate(clusters):
            size = [max(0.05, c['aabb_max'][d] - c['aabb_min'][d]) + 0.05 for d in range(3)]
            obj = CollisionObject()
            obj.header.frame_id = PLANNING_FRAME
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = f'detected_obstacle_{i}'
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = size
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = c['centroid']
            pose.orientation.w = 1.0
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
        if success:
            self.get_logger().info(f'  Registered {len(objs)} detected obstacles with MoveIt.')
        else:
            self.get_logger().error(f'  Failed to register detected obstacles: {result}')
        return success


def validate_against_ground_truth(clusters):
    """Match each detected cluster to its nearest ground-truth obstacle and
    report the position error -- for honesty about detection accuracy, not
    used to influence the detection itself."""
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
    node = ObstacleScanNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node._client.wait_for_server()

    node.get_logger().info('=' * 65)
    node.get_logger().info('  OBSTACLE SCAN (experimental, one-shot, static obstacles)')
    node.get_logger().info('=' * 65)

    node.get_logger().info('Moving to scan pose (I-pose)...')
    node.move_to_pose(SCAN_POSE)
    node.wait_for_pose(SCAN_POSE)

    node.get_logger().info('Waiting for both camera point clouds...')
    if not node.wait_for_clouds(timeout=20.0):
        node.get_logger().error(
            'Timed out waiting for camera point clouds -- aborting. '
            '(Check both cameras are actually publishing: '
            'ros2 topic hz /camera/depth/right/points /camera/depth/left/points)'
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info('Transforming point clouds to world frame...')
    pts_right = node.cloud_to_world_points(node._cloud_right, 'camera_link_right')
    pts_left = node.cloud_to_world_points(node._cloud_left, 'camera_link_left')
    node.get_logger().info(f'  right camera: {len(pts_right)} finite points')
    node.get_logger().info(f'  left camera : {len(pts_left)} finite points')

    all_pts = np.concatenate([pts_right, pts_left], axis=0) if (len(pts_right) + len(pts_left)) > 0 else np.zeros((0, 3))
    if len(all_pts) > 0:
        node.get_logger().info(
            f'  combined raw bounding box: '
            f'x=[{all_pts[:,0].min():.3f},{all_pts[:,0].max():.3f}] '
            f'y=[{all_pts[:,1].min():.3f},{all_pts[:,1].max():.3f}] '
            f'z=[{all_pts[:,2].min():.3f},{all_pts[:,2].max():.3f}]'
        )

    filtered = node.filter_workspace(all_pts)
    node.get_logger().info(f'  {len(filtered)} points remain after workspace-bounds filtering')

    clusters = node.cluster_obstacles(filtered)
    if not clusters:
        node.get_logger().error(
            f'Fewer than {MIN_POINTS_FOR_DETECTION} points after filtering -- '
            'no obstacles detected. Nothing registered. This could mean the '
            'scan pose does not actually face the obstacles, or the cameras '
            'are not seeing them at this range/FOV -- inspect the raw '
            'bounding box above to diagnose.'
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(f'Detected {len(clusters)} cluster(s):')
    validation = validate_against_ground_truth(clusters)
    for v in validation:
        node.get_logger().info(
            f"  centroid={[round(x,3) for x in v['detected_centroid']]} "
            f"n_points={v['n_points']} nearest_gt={v['nearest_ground_truth']} "
            f"error={v['position_error_m']:.3f}m"
        )

    registered = node.register_collision_objects(clusters)

    with open(os.path.join(LOG_DIR, 'scan_result.json'), 'w') as f:
        json.dump({
            'clusters': clusters,
            'validation': validation,
            'registered': registered,
            'raw_point_counts': {'right': len(pts_right), 'left': len(pts_left)},
        }, f, indent=2)
    node.get_logger().info(f'Scan result saved: {os.path.join(LOG_DIR, "scan_result.json")}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
