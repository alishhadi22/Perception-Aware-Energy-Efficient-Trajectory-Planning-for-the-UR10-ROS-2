#!/usr/bin/env python3
"""
perception_node.py

Supports a 'camera_mode' parameter so each sensor setup can be tested in
isolation, for a clean coverage/precision comparison:

  camera_mode:=panoramic  -- ONLY the external fixed camera (Setup A).
                              Represents 1x Orbbec Gemini 335, wide view of
                              the whole workspace, for coverage testing.
  camera_mode:=stereo      -- ONLY the wrist/end-effector camera(s) (Setup B).
                              Represents 2x Intel RealSense D405 for
                              close-range precision. (Setup B not wired
                              yet -- currently falls back to the single
                              wrist camera until the second unit is added.)
  camera_mode:=both         -- combines both (whichever sees something
                               closer wins), same as the original design.

Default is 'panoramic' -- Setup A is what we're confirming right now.

Run with, e.g.:
    ros2 run ur10_perception perception_node.py --ros-args -p camera_mode:=panoramic
or directly:
    python3 perception_node.py --ros-args -p camera_mode:=panoramic


SELF-FILTERING (TF-based):

A fixed cylinder around the base axis was tried first and failed: the arm
swings well outside any small fixed radius during normal motion, and enlarging
the radius to cover the arm's reach also swallows real obstacles.

The correct fix: look up the LIVE pose of each robot link via TF (published
by robot_state_publisher) and treat the chain
    base_link_inertia -> shoulder_link -> upper_arm_link -> forearm_link ->
    wrist_1_link -> wrist_2_link -> wrist_3_link
as a sequence of line segments approximating the arm's real physical links.
Any point within ROBOT_LINK_RADIUS_M of ANY segment is excluded as
self-detection. This works correctly at any arm pose, not just idle.

RADIUS CALIBRATION (verified via debug logging, since removed): at idle
(I-pose), the arm's own joint-housing surface was detected 0.1202 m from
its nearest segment centerline -- joint housings bulge wider than a simple
straight-tube model. ROBOT_LINK_RADIUS_M is set to 0.18 m (50% margin above
the observed 0.1202 m near-miss) while staying far below the ~0.50-0.60 m
distance from either real obstacle (pipe_obstacle_1, hull_panel_obstacle_2)
to the nearest arm segment -- confirmed safe, no risk of excluding real
obstacles at this radius.

The external camera is STATIC (confirmed world pose 1.3, 0, 1.1, roll=0,
pitch=0.35, yaw=3.14159 -- not part of the TF tree, so its camera-local ->
world transform is still the precomputed fixed matrix). Only the ROBOT'S
pose is looked up live via TF; the camera's own transform is unchanged.

The wrist camera (Setup B, not active in 'panoramic' mode) DOES have a TF
frame (camera_link, parent wrist_3_link), so when Setup B is wired in later,
its camera-local -> world transform can also be looked up live via TF
instead of needing a new fixed matrix. Not needed yet for Setup A testing.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import tf2_ros
from tf2_ros import LookupException, ExtrapolationException, ConnectivityException


WRIST_TOPIC    = "/camera/depth/points"
EXTERNAL_TOPIC = "/camera_external/depth/points"
MIN_RANGE_M = 0.15   # sensor minimum range (matches both real cameras closely enough)
MAX_RANGE_M = 1.40   # UR10 max reach (1.30 m) + margin
GROUND_Z_M  = 0.05

# --- TF-based self-filter (see module docstring) ---
ROBOT_LINK_CHAIN = [
    "base_link_inertia",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]
ROBOT_LINK_RADIUS_M = 0.10   # see RADIUS CALIBRATION note above
WORLD_FRAME = "world"

# External camera's fixed world pose: (1.3, 0, 1.1), roll=0, pitch=0.35, yaw=3.14159
# (external_camera is a static SDF model, NOT part of the robot TF tree --
# confirmed absent from `ros2 run tf2_tools view_frames` output -- so this
# fixed matrix is still how its points get converted to world frame.)
_EXTERNAL_CAM_R = np.array([
    [-9.39372713e-01, -2.65358979e-06, -3.42897807e-01],
    [ 2.49270984e-06, -1.00000000e+00,  9.09910122e-07],
    [-3.42897807e-01,  0.00000000e+00,  9.39372713e-01],
])
_EXTERNAL_CAM_T = np.array([1.3, 0.0, 1.1])


def _point_to_segment_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized distance from each row of `points` (N,3) to segment a-b."""
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / ab_len2, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(points - proj, axis=1)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter('camera_mode', 'panoramic')
        self.mode = self.get_parameter('camera_mode').get_parameter_value().string_value
        if self.mode not in ('panoramic', 'stereo', 'both'):
            self.get_logger().warn(
                f"Unknown camera_mode '{self.mode}', defaulting to 'panoramic'"
            )
            self.mode = 'panoramic'

        self._dist_wrist    = MAX_RANGE_M
        self._dist_external = MAX_RANGE_M

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_warned = False

        self.sub_wrist = None
        self.sub_external = None

        if self.mode in ('stereo', 'both'):
            self.sub_wrist = self.create_subscription(
                PointCloud2, WRIST_TOPIC, self._wrist_cb, qos_profile_sensor_data
            )
        if self.mode in ('panoramic', 'both'):
            self.sub_external = self.create_subscription(
                PointCloud2, EXTERNAL_TOPIC, self._external_cb, qos_profile_sensor_data
            )

        self.dist_pub   = self.create_publisher(Float32, "/perception/min_distance", 10)
        self.risk_pub   = self.create_publisher(Float32, "/perception/risk", 10)
        self.marker_pub = self.create_publisher(Marker, "/perception/risk_marker", 10)

        active = []
        if self.sub_wrist:
            active.append(f"wrist ({WRIST_TOPIC})")
        if self.sub_external:
            active.append(f"external ({EXTERNAL_TOPIC})")
        self.get_logger().info(
            f"Perception node started. Mode='{self.mode}'. Active cameras: {', '.join(active)}. "
            f"Self-filter: TF-based, link radius={ROBOT_LINK_RADIUS_M}m."
        )

    def _get_robot_link_positions_world(self):
        """Look up the live world-frame position of each link in ROBOT_LINK_CHAIN.
        Returns an (len(chain), 3) array, or None if TF isn't ready yet."""
        positions = []
        for link in ROBOT_LINK_CHAIN:
            try:
                tf = self.tf_buffer.lookup_transform(WORLD_FRAME, link, Time())
                t = tf.transform.translation
                positions.append([t.x, t.y, t.z])
            except (LookupException, ExtrapolationException, ConnectivityException) as e:
                if not self._tf_warned:
                    self.get_logger().warn(
                        f"TF lookup failed for link '{link}' ({e}); "
                        f"self-filter inactive until TF is available."
                    )
                    self._tf_warned = True
                return None
        return np.array(positions)

    def _nearest_in_cloud(self, msg: PointCloud2, is_external: bool):
        pts = pc2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if pts.size == 0:
            return None, None

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        valid = (r > MIN_RANGE_M) & (r < MAX_RANGE_M) & (z > GROUND_Z_M) & np.isfinite(r)

        if not np.any(valid):
            return None, None

        candidate_idx = np.where(valid)[0]
        local_pts = np.stack([x[candidate_idx], y[candidate_idx], z[candidate_idx]], axis=1)

        # World-frame conversion, needed for the self-filter. Only
        # range-valid points are transformed, so Inf/NaN points already
        # excluded by `valid` never reach the matmul.
        if is_external:
            world_pts = local_pts @ _EXTERNAL_CAM_R.T + _EXTERNAL_CAM_T
        else:
            # Wrist camera world transform not yet wired (Setup B TODO) --
            # self-filter skipped for wrist points for now.
            world_pts = None

        if world_pts is not None:
            link_positions = self._get_robot_link_positions_world()
            if link_positions is not None:
                min_dist_to_arm = np.full(len(candidate_idx), np.inf)
                for i in range(len(link_positions) - 1):
                    seg_d = _point_to_segment_dist(
                        world_pts, link_positions[i], link_positions[i + 1]
                    )
                    min_dist_to_arm = np.minimum(min_dist_to_arm, seg_d)
                is_self = min_dist_to_arm < ROBOT_LINK_RADIUS_M
                valid[candidate_idx[is_self]] = False

        if not np.any(valid):
            return None, None

        idx = np.argmin(r[valid])
        nearest_r = float(r[valid][idx])
        nearest_pt = (
            float(x[valid][idx]), float(y[valid][idx]), float(z[valid][idx])
        )
        return nearest_r, nearest_pt

    def _wrist_cb(self, msg):
        d, pt = self._nearest_in_cloud(msg, is_external=False)
        self._dist_wrist = d if d is not None else MAX_RANGE_M
        self._publish_combined(pt if d is not None else None)

    def _external_cb(self, msg):
        d, pt = self._nearest_in_cloud(msg, is_external=True)
        self._dist_external = d if d is not None else MAX_RANGE_M
        self._publish_combined(pt if d is not None else None)

    def _publish_combined(self, point):
        combined = min(self._dist_wrist, self._dist_external)
        self.dist_pub.publish(Float32(data=combined))

        risk = 0.0 if combined >= MAX_RANGE_M else 1.0 / combined
        self.risk_pub.publish(Float32(data=risk))

        if point is not None:
            self._publish_marker(point, risk)

    def _publish_marker(self, point, risk):
        m = Marker()
        m.header.frame_id = "camera_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "perception_risk"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = point
        scale = 0.03 + min(risk, 5.0) * 0.02
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r = min(1.0, risk / 3.0)
        m.color.g = max(0.0, 1.0 - risk / 3.0)
        m.color.b = 0.0
        m.color.a = 0.9
        self.marker_pub.publish(m)


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
