#!/usr/bin/env python3
"""
CMA-ES Trajectory Optimizer v2 for UR10
30-variable optimization: 5 params x 6 joints
Variables per joint: vel, accel, compress, offset, weight
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
POPULATION    = 6
MIN_DURATION  = 2.0
LOG_DIR       = os.path.expanduser('~/ros2_ws/logs')
CMAES_LOG_DIR = os.path.join(LOG_DIR, 'cmaes_30var')

# ─── BASELINE (updated dynamically after baseline measurement) ────────────────
BASELINE_J        = 0.770740
BASELINE_DURATION = 8.090


class CMAESOptimizerNode(Node):

    def __init__(self):
        super().__init__('cmaes_optimizer_v2')
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

        os.makedirs(CMAES_LOG_DIR, exist_ok=True)

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
                'vel':      max(0.1, min(1.0,  x[i*5+0])),
                'accel':    max(0.1, min(1.0,  x[i*5+1])),
                'compress': max(0.5, min(1.0,  x[i*5+2])),
                'offset':   max(-0.3, min(0.3, x[i*5+3])),
                'weight':   max(0.05, min(1.0, x[i*5+4])),
            })
        return params

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
            self.get_logger().info(f'  [PRE-EVAL] Waiting for home... max_err={max(errs):.4f} rad')
            time.sleep(0.5)
        return False

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
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().accepted:
            result_future = future.result().get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
        time.sleep(2.0)

    def evaluate(self, x, eval_num):
        global BASELINE_DURATION
        params = self.decode_params(x)

        avg_v = sum(p['vel']      for p in params) / N_JOINTS
        avg_c = sum(p['compress'] for p in params) / N_JOINTS
        est_dur = BASELINE_DURATION * avg_c / avg_v

        self.get_logger().info(
            f'╔══ Eval {eval_num}/{MAX_EVALS} | '
            f'avg_vel={avg_v:.3f} avg_compress={avg_c:.3f} '
            f'est_dur={est_dur:.1f}s ══╗'
        )
        self.get_logger().info(
            f'  {"Joint":<26} {"vel":>5} {"accel":>7} {"compress":>10} {"offset":>8} {"weight":>7}'
        )
        self.get_logger().info(f'  {"-"*62}')
        for j, p in enumerate(params):
            self.get_logger().info(
                f'  {JOINT_NAMES[j]:<26} '
                f'{p["vel"]:>5.3f} {p["accel"]:>7.3f} '
                f'{p["compress"]:>10.3f} {p["offset"]:>+8.3f} {p["weight"]:>7.3f}'
            )

        # Return to home
        self.return_to_home()
        self.wait_for_home()

        # Build optimized trajectory
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        # Via-point 1 (midpoint with offsets)
        pt1 = JointTrajectoryPoint()
        mid = [(START_POSE[j] + GOAL_POSE[j]) / 2.0 + params[j]['offset'] for j in range(N_JOINTS)]
        pt1.positions     = mid
        pt1.velocities    = [params[j]['vel'] * params[j]['weight'] for j in range(N_JOINTS)]
        pt1.accelerations = [params[j]['accel'] for j in range(N_JOINTS)]
        t1 = BASELINE_DURATION * avg_c * 0.5
        pt1.time_from_start = Duration(
            seconds=int(t1),
            nanoseconds=int((t1 % 1) * 1e9)
        ).to_msg()

        # Via-point 2 (goal)
        pt2 = JointTrajectoryPoint()
        pt2.positions     = GOAL_POSE
        pt2.velocities    = [0.0] * 6
        pt2.accelerations = [0.0] * 6
        t2 = BASELINE_DURATION * avg_c
        pt2.time_from_start = Duration(
            seconds=int(t2),
            nanoseconds=int((t2 % 1) * 1e9)
        ).to_msg()

        traj.points = [pt1, pt2]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Execute and log
        self._logged_data = []
        self._logging     = True
        start_time        = time.time()

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._logging = False
            self.get_logger().warn('  Trajectory failed (code=None, samples=0) -- penalizing')
            return 1.0

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

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
            self.get_logger().warn(
                f'  Trajectory failed (code=None, samples={samples}) -- penalizing'
            )
            return 1.0

        # Save temp CSV and compute J
        tmp_path = os.path.join(CMAES_LOG_DIR, f'eval_{eval_num}.csv')
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)

        J = compute_objective(tmp_path, verbose=False, t_ref=BASELINE_DURATION)

        # Check start error
        name_to_idx = {n: i for i, n in enumerate(self._joint_states.name)}
        start_err = max(
            abs(self._joint_states.position[name_to_idx.get(jname, 0)] - START_POSE[j])
            for j, jname in enumerate(JOINT_NAMES)
            if name_to_idx.get(jname, -1) >= 0
        )

        self.get_logger().info(
            f'  ┌─ RESULT: J={J:.6f} | dur={T_raw:.2f}s | samples={samples}'
        )
        self.get_logger().info(f'  └─ start_err={start_err:.3f} rad')

        if J < self.best_J:
            self.best_J      = J
            self.best_params = params
            self.get_logger().info(f'  ★★★ NEW BEST J={J:.6f} | dur={T_raw:.2f}s ★★★')

        self.history.append({
            'eval': eval_num, 'J': J, 'duration': T_raw, 'samples': samples
        })

        return J

    def run_cmaes(self):
        global BASELINE_J, BASELINE_DURATION

        self.get_logger().info('=' * 65)
        self.get_logger().info('  CMA-ES 30-Variable Optimization (v2 -- Clean)')
        self.get_logger().info(f'  Variables  : {N_PARAMS} params x {N_JOINTS} joints = {N_VARS}')
        self.get_logger().info(f'  Max evals  : {MAX_EVALS}')
        self.get_logger().info(f'  Population : {POPULATION}')
        self.get_logger().info(f'  Baseline J : {BASELINE_J:.6f} (AHP weights)')
        self.get_logger().info(f'  Min dur    : {MIN_DURATION}s')
        self.get_logger().info('=' * 65)

        # Initial solution (neutral parameters)
        x0    = np.array([0.5, 0.5, 0.7, 0.0, 0.5] * N_JOINTS)
        sigma = 0.2

        # CMA-ES implementation
        n     = len(x0)
        lam   = POPULATION
        mu    = lam // 2
        w     = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w    /= w.sum()
        mueff = 1.0 / (w ** 2).sum()

        cc    = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs    = (mueff + 2) / (n + mueff + 5)
        c1    = 2 / ((n + 1.3) ** 2 + mueff)
        cmu   = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        xmean  = x0.copy()
        C      = np.eye(n)
        pc     = np.zeros(n)
        ps     = np.zeros(n)
        chiN   = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        eval_count = 0

        while eval_count < MAX_EVALS:
            # Sample population
            arz = np.random.randn(lam, n)
            try:
                arx = xmean + sigma * (arz @ np.linalg.cholesky(C).T)
            except np.linalg.LinAlgError:
                C   = np.eye(n)
                arx = xmean + sigma * arz

            # Evaluate
            fitvals = []
            for k in range(lam):
                if eval_count >= MAX_EVALS:
                    break
                eval_count += 1
                J = self.evaluate(arx[k], eval_count)
                fitvals.append((J, k))

            if not fitvals:
                break

            # Sort by fitness
            fitvals.sort(key=lambda x: x[0])
            idx_sorted = [k for _, k in fitvals[:mu]]

            # Update mean
            xold   = xmean.copy()
            xmean  = w @ arx[idx_sorted]

            # Update evolution paths
            invsqrtC = np.linalg.inv(np.linalg.cholesky(C)).T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * eval_count / lam)) / chiN < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

            # Update covariance
            artmp = (1 / sigma) * (arx[idx_sorted] - xold)
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * artmp.T @ np.diag(w) @ artmp)

            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma  = np.clip(sigma, 0.01, 2.0)

        # Final report
        imp = (BASELINE_J - self.best_J) / BASELINE_J * 100 if self.best_J < BASELINE_J else 0.0

        self.get_logger().info('=' * 65)
        self.get_logger().info('  OPTIMIZATION COMPLETE')
        self.get_logger().info(f'  Total evals : {eval_count}')
        self.get_logger().info(f'  Baseline J  : {BASELINE_J:.6f}')
        self.get_logger().info(f'  Best J      : {self.best_J:.6f}')
        self.get_logger().info(f'  Improvement : {imp:.2f}%')

        if self.best_params:
            self.get_logger().info('  Best per-joint parameters:')
            self.get_logger().info(
                f'  {"Joint":<26} {"vel":>5} {"accel":>7} {"compress":>10} {"offset":>8} {"weight":>7}'
            )
            for j, p in enumerate(self.best_params):
                self.get_logger().info(
                    f'  {JOINT_NAMES[j]:<26} '
                    f'{p["vel"]:>5.3f} {p["accel"]:>7.3f} '
                    f'{p["compress"]:>10.3f} {p["offset"]:>+8.3f} {p["weight"]:>7.3f}'
                )
        self.get_logger().info('=' * 65)

        # Save history
        hist_path = os.path.join(CMAES_LOG_DIR, 'optimization_history.csv')
        with open(hist_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['eval', 'J', 'duration', 'samples'])
            writer.writeheader()
            writer.writerows(self.history)
        self.get_logger().info(f'History saved: {hist_path}')

        # Save best params
        if self.best_params:
            best_path = os.path.join(CMAES_LOG_DIR, 'best_params.csv')
            with open(best_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['joint', 'vel', 'accel', 'compress', 'offset', 'weight'])
                writer.writeheader()
                for j, p in enumerate(self.best_params):
                    writer.writerow({'joint': JOINT_NAMES[j], **p})
            self.get_logger().info(f'Best params saved: {best_path}')


def main():
    rclpy.init()
    node = CMAESOptimizerNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info('Joint states received!')

    node._client.wait_for_server()
    node.run_cmaes()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
