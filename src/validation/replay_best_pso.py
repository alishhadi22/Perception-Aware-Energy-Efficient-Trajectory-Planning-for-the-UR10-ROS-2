#!/usr/bin/env python3
"""
Replay validation for PSO Path 1 (I->_) best parameters.

Reads logs/pso_30var/best_params.csv (the fixed 30-variable solution found
by pso_optimizer.py's most recent run), re-executes it in Gazebo 3 times
with NO search loop, and reports J/duration per run plus mean/std compared
against that run's reported best J.

Does not modify pso_optimizer.py. Reuses its trajectory-building /
Gazebo-execution logic (return_to_home, wait_for_home, 2-point trajectory,
t1/t2 timing formula, joint-state logging, compute_objective call)
verbatim, applied to one fixed params vector instead of a PSO search.
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

# ─── TRAJECTORY DEFINITION (must match pso_optimizer.py's Path 1 state) ─────
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
PSO_LOG_DIR   = os.path.join(LOG_DIR, 'pso_30var')
BEST_PARAMS_CSV = os.path.join(PSO_LOG_DIR, 'best_params.csv')
REPLAY_LOG_DIR  = os.path.join(PSO_LOG_DIR, 'replay_runs')

# ─── BASELINE (Path 1 (I->_), matches pso_optimizer.py) ──────────────────────
BASELINE_J        = 0.770740
BASELINE_DURATION = 8.090

# ─── Reference result to compare against (filled in after the fresh PSO
# run that produces best_params.csv finishes) ────────────────────────────────
ORIGINAL_BEST_J = None  # set via --best-j CLI arg, or left None to skip the comparison


def load_best_params(csv_path):
    """Read best_params.csv and rebuild the same list-of-dicts structure
    that decode_params() produces in pso_optimizer.py, in JOINT_NAMES order."""
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


class PSOReplayNode(Node):

    def __init__(self):
        super().__init__('pso_replay_best')
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

    def evaluate_fixed(self, params, run_num):
        """Evaluate one fixed params vector (no PSO search). Mirrors
        pso_optimizer.py's evaluate() body exactly, minus decode_params()."""
        avg_v = sum(p['vel']      for p in params) / N_JOINTS
        avg_c = sum(p['compress'] for p in params) / N_JOINTS

        self.get_logger().info(
            f'╔══ Replay run {run_num}/{N_RUNS} | '
            f'avg_vel={avg_v:.3f} avg_compress={avg_c:.3f} ══╗'
        )

        self.return_to_home()
        self.wait_for_home()

        # Build fixed trajectory (identical formula to pso_optimizer.py's evaluate())
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
            return None, None, 0

        result_future = goal_handle.get_result_async()
        deadline = time.time() + (t2 + 25.0)
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.5)
            if time.time() > deadline:
                self._logging = False
                self.get_logger().warn('  Result timeout -- penalizing and continuing')
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return None, None, len(self._logged_data)

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
            self.get_logger().warn(f'  Too few samples ({samples}) -- penalizing')
            return None, None, samples

        tmp_path = os.path.join(REPLAY_LOG_DIR, f'replay_run{run_num}.csv')
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)

        J = compute_objective(tmp_path, verbose=False, t_ref=BASELINE_DURATION)

        self.get_logger().info(f'  ┌─ RESULT: J={J:.6f} | dur={T_raw:.2f}s | samples={samples}')

        return J, T_raw, samples


def main():
    global ORIGINAL_BEST_J
    if len(sys.argv) > 1:
        ORIGINAL_BEST_J = float(sys.argv[1])

    print(f'Loading best params from: {BEST_PARAMS_CSV}')
    params = load_best_params(BEST_PARAMS_CSV)

    rclpy.init()
    node = PSOReplayNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info('Joint states received!')

    node._client.wait_for_server()

    node.get_logger().info('=' * 65)
    node.get_logger().info('  PSO BEST-PARAMS REPLAY (no search, fixed vector x3)')
    node.get_logger().info(f'  Baseline J : {BASELINE_J:.6f}')
    node.get_logger().info(f'  Baseline T : {BASELINE_DURATION:.3f}s')
    node.get_logger().info('=' * 65)

    results = []
    for run_num in range(1, N_RUNS + 1):
        J, dur, samples = node.evaluate_fixed(params, run_num)
        results.append({'run': run_num, 'J': J, 'duration': dur, 'samples': samples})

    node.destroy_node()
    rclpy.shutdown()

    summary_path = os.path.join(REPLAY_LOG_DIR, 'replay_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'J', 'duration', 'samples'])
        writer.writeheader()
        writer.writerows(results)

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
        print(f'  Replay mean J     : {mean_J:.6f}')
        print(f'  Replay std J      : {std_J:.6f}')
        if ORIGINAL_BEST_J is not None:
            print(f'  Fresh-run best J  : {ORIGINAL_BEST_J:.6f}  (the single best eval that produced best_params.csv)')
            diff = abs(mean_J - ORIGINAL_BEST_J)
            print(f'  |diff| = {diff:.6f}  vs replay std = {std_J:.6f}')
            if diff <= max(std_J, 1e-9) * 3:
                print('  VERDICT: Replay mean is close to the original best J (within ~3x replay std) -- reproducible.')
            else:
                print('  VERDICT: Replay mean diverges notably from the original best J -- investigate before trusting it.')
        else:
            print('  (No original best J passed as argv[1] -- skipping comparison verdict.)')
    else:
        print('-' * 65)
        print(f'  Only {len(valid_J)}/{N_RUNS} runs succeeded -- cannot compute a valid mean/std.')
    print('=' * 65)
    print(f'  Raw run CSVs + summary saved under: {REPLAY_LOG_DIR}')


if __name__ == '__main__':
    main()
