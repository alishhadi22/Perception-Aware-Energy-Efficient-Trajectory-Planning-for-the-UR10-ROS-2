#!/usr/bin/env python3
"""
Replay validation for CMA-ES Path 1 (I->_) best parameters.

Reads logs/cmaes_30var/best_params.csv (the fixed 30-variable solution
found by cmaes_optimizer.py's most recent run), re-executes it in Gazebo
3 times with NO search/CMA-ES loop, and reports J/duration per run plus
mean/std compared against the originally reported CMA-ES Path 1 mean.

Does not modify cmaes_optimizer.py. Reuses its trajectory-building /
Gazebo-execution logic (return_to_home, wait_for_home, 2-point trajectory,
t1/t2 timing formula, joint-state logging, compute_objective call)
verbatim, applied to one fixed params vector instead of a CMA-ES search.
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

# ─── TRAJECTORY DEFINITION (must match cmaes_optimizer.py's Path 1 state) ────
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
N_RUNS        = 3
MIN_DURATION  = 2.0
LOG_DIR       = os.path.expanduser('~/ros2_ws/logs')
CMAES_LOG_DIR = os.path.join(LOG_DIR, 'cmaes_30var')
BEST_PARAMS_CSV = os.path.join(CMAES_LOG_DIR, 'best_params.csv')
REPLAY_LOG_DIR  = os.path.join(CMAES_LOG_DIR, 'replay_runs')

# ─── BASELINE (Path 1 (I->_), matches cmaes_optimizer.py) ────────────────────
BASELINE_J        = 0.770740
BASELINE_DURATION = 8.090

# ─── Originally reported CMA-ES Path 1 result (for comparison only) ─────────
ORIGINAL_MEAN_J = 0.638157
ORIGINAL_STD_J  = 0.001705


def load_best_params(csv_path):
    """Read best_params.csv and rebuild the same list-of-dicts structure
    that decode_params() produces in cmaes_optimizer.py, in JOINT_NAMES order."""
    by_joint = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_joint[row['joint']] = {
                'vel':      float(row['vel']),
                'accel':    float(row['accel']),
                'compress': float(row['compress']),
                'offset':   float(row['offset']),
                'weight':   float(row['weight']),
            }
    missing = [j for j in JOINT_NAMES if j not in by_joint]
    if missing:
        raise ValueError(f'best_params.csv missing joints: {missing}')
    return [by_joint[j] for j in JOINT_NAMES]


class CMAESReplayNode(Node):

    def __init__(self):
        super().__init__('cmaes_replay_best')
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        self._joint_states  = None
        self._logged_data   = []
        self._logging       = False

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self._latest_min_distance = None
        self._dist_sub = self.create_subscription(
            Float32, '/perception/min_distance', self._dist_callback, 10)

        os.makedirs(REPLAY_LOG_DIR, exist_ok=True)

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

    def evaluate_fixed(self, params, run_num):
        """Evaluate one fixed params vector (no CMA-ES search). Mirrors
        cmaes_optimizer.py's evaluate() body exactly, minus decode_params()."""
        avg_v = sum(p['vel']      for p in params) / N_JOINTS
        avg_c = sum(p['compress'] for p in params) / N_JOINTS
        est_dur = BASELINE_DURATION * avg_c / avg_v

        self.get_logger().info(
            f'╔══ Replay run {run_num}/{N_RUNS} | '
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

        # Build fixed trajectory (identical formula to cmaes_optimizer.py's evaluate())
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
            return None, None, 0

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
            return None, None, samples

        if samples < 100 or T_raw < MIN_DURATION:
            self.get_logger().warn(
                f'  Trajectory failed (code=None, samples={samples}) -- penalizing'
            )
            return None, None, samples

        # Save this run's CSV under a dedicated replay directory (never
        # overwrites cmaes_optimizer.py's own eval_N.csv files)
        tmp_path = os.path.join(REPLAY_LOG_DIR, f'replay_run{run_num}.csv')
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)

        J = compute_objective(tmp_path, verbose=False, t_ref=BASELINE_DURATION)

        self.get_logger().info(
            f'  ┌─ RESULT: J={J:.6f} | dur={T_raw:.2f}s | samples={samples}'
        )

        return J, T_raw, samples


def main():
    print(f'Loading best params from: {BEST_PARAMS_CSV}')
    params = load_best_params(BEST_PARAMS_CSV)

    rclpy.init()
    node = CMAESReplayNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info('Joint states received!')

    node._client.wait_for_server()

    node.get_logger().info('=' * 65)
    node.get_logger().info('  CMA-ES BEST-PARAMS REPLAY (no search, fixed vector x3)')
    node.get_logger().info(f'  Baseline J : {BASELINE_J:.6f}')
    node.get_logger().info(f'  Baseline T : {BASELINE_DURATION:.3f}s')
    node.get_logger().info('=' * 65)

    results = []
    for run_num in range(1, N_RUNS + 1):
        J, dur, samples = node.evaluate_fixed(params, run_num)
        results.append({'run': run_num, 'J': J, 'duration': dur, 'samples': samples})

    node.destroy_node()
    rclpy.shutdown()

    # Save summary CSV
    summary_path = os.path.join(REPLAY_LOG_DIR, 'replay_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'J', 'duration', 'samples'])
        writer.writeheader()
        writer.writerows(results)

    # Report
    valid_J = [r['J'] for r in results if r['J'] is not None]
    print()
    print('=' * 65)
    print('  REPLAY RESULTS (fixed best_params.csv vector, 3 fresh runs)')
    print('=' * 65)
    for r in results:
        if r['J'] is not None:
            print(f"  Run {r['run']}: J={r['J']:.6f}  dur={r['duration']:.2f}s  samples={r['samples']}")
        else:
            print(f"  Run {r['run']}: FAILED/PENALIZED (samples={r['samples']})")

    if len(valid_J) == N_RUNS:
        mean_J = float(np.mean(valid_J))
        std_J  = float(np.std(valid_J))
        print('-' * 65)
        print(f'  Replay mean J   : {mean_J:.6f}')
        print(f'  Replay std J    : {std_J:.6f}')
        print(f'  Original mean J : {ORIGINAL_MEAN_J:.6f}')
        print(f'  Original std J  : {ORIGINAL_STD_J:.6f}')

        diff = abs(mean_J - ORIGINAL_MEAN_J)
        combined_band = std_J + ORIGINAL_STD_J
        overlap = diff <= combined_band
        print(f'  |diff| = {diff:.6f}  vs combined std band = {combined_band:.6f}')
        if overlap:
            print('  VERDICT: Replay mean falls within the combined std band of the '
                  'original result -- reproducible.')
        else:
            print('  VERDICT: Replay mean falls OUTSIDE the combined std band of the '
                  'original result -- investigate before trusting the original report.')
    else:
        print('-' * 65)
        print(f'  Only {len(valid_J)}/{N_RUNS} runs succeeded -- cannot compute a valid mean/std.')
    print('=' * 65)
    print(f'  Raw run CSVs + summary saved under: {REPLAY_LOG_DIR}')


if __name__ == '__main__':
    main()
