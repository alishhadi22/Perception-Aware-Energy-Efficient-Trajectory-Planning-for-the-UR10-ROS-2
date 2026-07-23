#!/usr/bin/env python3
"""
Grey Wolf Optimizer (GWO) for UR10 Trajectory Optimization
30-variable optimization: 5 params x 6 joints
Same structure as CMA-ES optimizer for fair comparison.
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
import numpy as np
import csv
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_objective import compute_objective

# ─── TRAJECTORY DEFINITION ───────────────────────────────────────────────────
START_POSE = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
GOAL_POSE  = [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0]

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

N_JOINTS      = 6
N_PARAMS      = 5
N_VARS        = N_JOINTS * N_PARAMS   # 30
MAX_EVALS     = 60
N_WOLVES      = 6
MIN_DURATION  = 2.0
LOG_DIR       = os.path.expanduser('~/ros2_ws/logs')
GWO_LOG_DIR   = os.path.join(LOG_DIR, 'gwo_30var')

BASELINE_J        = 0.770740
BASELINE_DURATION = 8.090

# ─── VARIABLE BOUNDS ─────────────────────────────────────────────────────────
LB = np.array([0.1, 0.1, 0.5, -0.3, 0.05] * N_JOINTS)
UB = np.array([1.0, 1.0, 1.0,  0.3, 1.00] * N_JOINTS)


class GWOOptimizerNode(Node):

    def __init__(self):
        super().__init__('gwo_optimizer')
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        self._joint_states  = None
        self._logged_data   = []
        self._logging       = False
        self.best_J         = float('inf')
        self.best_params    = None
        self.history        = []

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self._latest_min_distance = None
        self._dist_sub = self.create_subscription(
            Float32, '/perception/min_distance', self._dist_callback, 10)

        os.makedirs(GWO_LOG_DIR, exist_ok=True)

    def _js_callback(self, msg):
        self._joint_states = msg
        if self._logging:
            self._log_joint_state(msg)

    def _dist_callback(self, msg):
        self._latest_min_distance = msg.data

    def _log_joint_state(self, msg):
        row = {'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for j, jname in enumerate(JOINT_NAMES):
            idx = name_to_idx.get(jname, -1)
            row[f'position_{j}'] = msg.position[idx] if idx >= 0 and idx < len(msg.position) else 0.0
            row[f'velocity_{j}'] = msg.velocity[idx] if idx >= 0 and idx < len(msg.velocity) else 0.0
            row[f'effort_{j}']   = msg.effort[idx]   if idx >= 0 and idx < len(msg.effort)   else 0.0
        row['min_distance'] = self._latest_min_distance if self._latest_min_distance is not None else 1.40
        self._logged_data.append(row)

    def decode_params(self, x):
        params = []
        for i in range(N_JOINTS):
            params.append({
                'vel':      float(np.clip(x[i*5+0], 0.1, 1.0)),
                'accel':    float(np.clip(x[i*5+1], 0.1, 1.0)),
                'compress': float(np.clip(x[i*5+2], 0.5, 1.0)),
                'offset':   float(np.clip(x[i*5+3], -0.3, 0.3)),
                'weight':   float(np.clip(x[i*5+4], 0.05, 1.0)),
            })
        return params

    def return_to_home(self):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions     = START_POSE
        pt.velocities    = [0.0] * 6
        pt.accelerations = [0.0] * 6
        pt.time_from_start = Duration(seconds=8).to_msg()
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        gh = future.result()
        if gh and gh.accepted:
            result_future = gh.get_result_async()
            deadline = time.time() + 20.0
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.5)
                if time.time() > deadline:
                    self.get_logger().warn('  return_to_home timeout -- continuing')
                    try:
                        gh.cancel_goal_async()
                    except Exception:
                        pass
                    break
        time.sleep(2.0)

    def wait_for_home(self, timeout=30.0):
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
                    errs.append(abs(self._joint_states.position[idx] - START_POSE[j]))
            if errs and max(errs) < 0.05:
                self.get_logger().info(f'  [PRE-EVAL] Robot at HOME ✓ (max_err={max(errs):.4f} rad)')
                return True
            time.sleep(0.5)
        return False

    def evaluate(self, x, eval_num):
        params = self.decode_params(x)
        avg_v = sum(p['vel']      for p in params) / N_JOINTS
        avg_c = sum(p['compress'] for p in params) / N_JOINTS

        self.get_logger().info(
            f'╔══ Eval {eval_num}/{MAX_EVALS} | '
            f'avg_vel={avg_v:.3f} avg_compress={avg_c:.3f} ══╗'
        )

        self.return_to_home()
        self.wait_for_home()

        # Build trajectory
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        t1 = BASELINE_DURATION * avg_c * 0.5
        t2 = BASELINE_DURATION * avg_c

        pt1 = JointTrajectoryPoint()
        pt1.positions     = [(START_POSE[j] + GOAL_POSE[j]) / 2.0 + params[j]['offset'] for j in range(N_JOINTS)]
        pt1.velocities    = [params[j]['vel'] * params[j]['weight'] for j in range(N_JOINTS)]
        pt1.accelerations = [params[j]['accel'] for j in range(N_JOINTS)]
        pt1.time_from_start = Duration(seconds=int(t1), nanoseconds=int((t1 % 1) * 1e9)).to_msg()

        pt2 = JointTrajectoryPoint()
        pt2.positions     = GOAL_POSE
        pt2.velocities    = [0.0] * 6
        pt2.accelerations = [0.0] * 6
        pt2.time_from_start = Duration(seconds=int(t2), nanoseconds=int((t2 % 1) * 1e9)).to_msg()

        traj.points = [pt1, pt2]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self._logged_data = []
        self._logging     = True
        start_time        = time.time()

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._logging = False
            self.get_logger().warn('  Trajectory rejected -- penalizing')
            return 1.0

        result_future = goal_handle.get_result_async()
        deadline = time.time() + (t2 + 25.0)
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.5)
            if time.time() > deadline:
                self._logging = False
                self.get_logger().warn(
                    '  Result timeout -- penalizing and continuing')
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return 1.0

        self._logging = False
        T_raw         = time.time() - start_time
        samples       = len(self._logged_data)

        result_response = result_future.result()
        status = result_response.status if result_response else None
        error_code = result_response.result.error_code if result_response else None
        if status != GoalStatus.STATUS_SUCCEEDED or error_code != 0:
            self.get_logger().warn(
                f'  Trajectory ABORTED (status={status}, error_code={error_code}) '
                f'-- likely collision or tolerance violation -- penalizing'
            )
            return 1.0

        if samples < 100 or T_raw < MIN_DURATION:
            self.get_logger().warn(f'  Too few samples ({samples}) -- penalizing')
            return 1.0

        tmp_path = os.path.join(GWO_LOG_DIR, f'eval_{eval_num}.csv')
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)

        J = compute_objective(tmp_path, verbose=False, t_ref=BASELINE_DURATION)

        self.get_logger().info(f'  ┌─ RESULT: J={J:.6f} | dur={T_raw:.2f}s | samples={samples}')

        if J < self.best_J:
            self.best_J      = J
            self.best_params = params
            self.get_logger().info(f'  ★★★ NEW BEST J={J:.6f} | dur={T_raw:.2f}s ★★★')

        self.history.append({'eval': eval_num, 'J': J, 'duration': T_raw, 'samples': samples})
        return J

    def run_gwo(self):
        self.get_logger().info('=' * 65)
        self.get_logger().info('  Grey Wolf Optimizer (GWO) -- 30 Variables')
        self.get_logger().info(f'  Wolves     : {N_WOLVES}')
        self.get_logger().info(f'  Max evals  : {MAX_EVALS}')
        self.get_logger().info(f'  Baseline J : {BASELINE_J:.6f}')
        self.get_logger().info('=' * 65)

        # Initialize wolf pack randomly
        wolves = np.random.uniform(LB, UB, (N_WOLVES, N_VARS))
        fitness = np.full(N_WOLVES, float('inf'))

        # Alpha, Beta, Delta wolves (best 3)
        alpha_pos = np.zeros(N_VARS)
        beta_pos  = np.zeros(N_VARS)
        delta_pos = np.zeros(N_VARS)
        alpha_score = float('inf')
        beta_score  = float('inf')
        delta_score = float('inf')

        eval_count = 0
        iteration  = 0

        while eval_count < MAX_EVALS:
            # Evaluate all wolves
            for i in range(N_WOLVES):
                if eval_count >= MAX_EVALS:
                    break
                eval_count += 1
                fitness[i] = self.evaluate(wolves[i], eval_count)

                # Update alpha, beta, delta
                if fitness[i] < alpha_score:
                    delta_score = beta_score
                    delta_pos   = beta_pos.copy()
                    beta_score  = alpha_score
                    beta_pos    = alpha_pos.copy()
                    alpha_score = fitness[i]
                    alpha_pos   = wolves[i].copy()
                elif fitness[i] < beta_score:
                    delta_score = beta_score
                    delta_pos   = beta_pos.copy()
                    beta_score  = fitness[i]
                    beta_pos    = wolves[i].copy()
                elif fitness[i] < delta_score:
                    delta_score = fitness[i]
                    delta_pos   = wolves[i].copy()

            if eval_count >= MAX_EVALS:
                break

            iteration += 1
            # Linearly decrease a from 2 to 0
            a = 2.0 - iteration * (2.0 / (MAX_EVALS / N_WOLVES))

            # Update wolf positions
            for i in range(N_WOLVES):
                for d in range(N_VARS):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[d] - wolves[i][d])
                    X1 = alpha_pos[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[d] - wolves[i][d])
                    X2 = beta_pos[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[d] - wolves[i][d])
                    X3 = delta_pos[d] - A3 * D_delta

                    wolves[i][d] = np.clip((X1 + X2 + X3) / 3.0, LB[d], UB[d])

        # Final report
        imp = (BASELINE_J - self.best_J) / BASELINE_J * 100 if self.best_J < BASELINE_J else 0.0

        self.get_logger().info('=' * 65)
        self.get_logger().info('  GWO OPTIMIZATION COMPLETE')
        self.get_logger().info(f'  Total evals : {eval_count}')
        self.get_logger().info(f'  Baseline J  : {BASELINE_J:.6f}')
        self.get_logger().info(f'  Best J      : {self.best_J:.6f}')
        self.get_logger().info(f'  Improvement : {imp:.2f}%')
        self.get_logger().info('=' * 65)

        # Save history
        hist_path = os.path.join(GWO_LOG_DIR, 'gwo_history.csv')
        with open(hist_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['eval', 'J', 'duration', 'samples'])
            writer.writeheader()
            writer.writerows(self.history)
        self.get_logger().info(f'History saved: {hist_path}')

        # Save best params
        if self.best_params:
            best_path = os.path.join(GWO_LOG_DIR, 'best_params.csv')
            with open(best_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['joint', 'vel', 'accel', 'compress', 'offset', 'weight'])
                writer.writeheader()
                for j, p in enumerate(self.best_params):
                    writer.writerow({'joint': JOINT_NAMES[j], **p})
            self.get_logger().info(f'Best params saved: {best_path}')


def main():
    rclpy.init()
    node = GWOOptimizerNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info('Joint states received!')

    node._client.wait_for_server()
    node.run_gwo()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
