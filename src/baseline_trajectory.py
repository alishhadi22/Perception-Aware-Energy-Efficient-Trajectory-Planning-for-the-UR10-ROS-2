#!/usr/bin/env python3
"""
Baseline Trajectory Node for UR10
Moves robot from START pose to GOAL pose and logs all joint data.
Used to establish the reference trajectory for CMA-ES optimization.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from action_msgs.msg import GoalStatus
import csv
import time
import os

# ─── TRAJECTORY DEFINITION ───────────────────────────────────────────────────
START_POSE = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]   # Home "I" position
GOAL_POSE  = [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0]
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

LOG_DIR = os.path.expanduser('~/ros2_ws/logs')

class BaselineTrajectoryNode(Node):

    def __init__(self):
        super().__init__('baseline_trajectory')
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
          '/joint_trajectory_controller/follow_joint_trajectory'
        )
        self._joint_states = None
        self._logged_data = []

        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._js_callback,
            10
        )

        self._latest_min_distance = None
        self._dist_sub = self.create_subscription(
            Float32, '/perception/min_distance', self._dist_callback, 10)

        self.get_logger().info('Baseline Trajectory Node started')

    def _js_callback(self, msg):
        self._joint_states = msg
        if len(self._logged_data) > 0 or hasattr(self, '_logging') and self._logging:
            self._log_joint_state(msg)

    def _dist_callback(self, msg):
        self._latest_min_distance = msg.data

    def _log_joint_state(self, msg):
        row = {'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for j, jname in enumerate(JOINT_NAMES):
            idx = name_to_idx.get(jname, -1)
            if idx >= 0:
                row[f'position_{j}'] = msg.position[idx] if idx < len(msg.position) else 0.0
                row[f'velocity_{j}'] = msg.velocity[idx] if idx < len(msg.velocity) else 0.0
                row[f'effort_{j}']   = msg.effort[idx]   if idx < len(msg.effort)   else 0.0
            else:
                row[f'position_{j}'] = 0.0
                row[f'velocity_{j}'] = 0.0
                row[f'effort_{j}']   = 0.0
        row['min_distance'] = self._latest_min_distance if self._latest_min_distance is not None else 1.40
        self._logged_data.append(row)

    def move_to_joint_goal(self, waypoints, plan_name="trajectory"):
        self._logged_data = []
        self._logging = True

        self.get_logger().info(f'Waiting for controller...')
        self._client.wait_for_server()

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        for i, (positions, t) in enumerate(waypoints):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
            point.time_from_start = Duration(
                seconds=int(t),
                nanoseconds=int((t % 1) * 1e9)
            ).to_msg()
            traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f'Executing {plan_name}...')
        start_time = time.time()

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            self._logging = False
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self._logging = False
        duration = time.time() - start_time

        result_response = result_future.result()
        status = result_response.status if result_response else None
        error_code = result_response.result.error_code if result_response else None
        success = (status == GoalStatus.STATUS_SUCCEEDED) and (error_code == 0)

        if not success:
            self.get_logger().error(
                f'{plan_name} FAILED (status={status}, error_code={error_code}) '
                f'after {duration:.2f}s, {len(self._logged_data)} samples -- '
                f'likely a physical collision or tolerance violation. '
                f'NOT a valid baseline -- saving with _FAILED tag.'
            )
            self.save_log(plan_name + '_FAILED', duration)
            return False

        self.get_logger().info(
            f'{plan_name} completed SUCCESSFULLY in {duration:.2f}s, '
            f'logged {len(self._logged_data)} samples'
        )

        self.save_log(plan_name, duration)
        return True

    def save_log(self, name, duration):
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = int(time.time())
        filepath = os.path.join(LOG_DIR, f'{name}_{timestamp}.csv')

        if not self._logged_data:
            self.get_logger().warn('No data to save!')
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)

        self.get_logger().info(f'Saved: {filepath}')
        self.get_logger().info(f'Duration: {duration:.2f}s | Samples: {len(self._logged_data)}')


def main():
    rclpy.init()
    node = BaselineTrajectoryNode()

    # Wait for joint states
    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info('Joint states received!')

    # Phase 1 — move to start pose
    node.get_logger().info('=== Moving to START pose ===')
    node.move_to_joint_goal([(START_POSE, 8.0)], 'start_pose')

    time.sleep(2.0)

    # Phase 2 — move to goal pose (this is the baseline)
    node.get_logger().info('=== Moving to GOAL pose (BASELINE) ===')
    node.move_to_joint_goal([(GOAL_POSE, 8.0)], 'baseline_trajectory')

    node.get_logger().info(f'Phase 2 complete! Check {LOG_DIR}/')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
