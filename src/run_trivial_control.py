#!/usr/bin/env python3
"""
Phase 4B trivial control for Path 1 (I->_).

Executes ONE fixed trajectory, no search loop at all:
    compress = 0.500 (the CMA-ES-saturated lower bound) on all six joints
    vel      = 0.55  on all six joints
    accel    = 0.55  on all six joints
    offset   = 0.0   on all six joints
    weight   = 0.525 on all six joints

This is the control for the most serious open criticism of the report:
13 of 30 CMA-ES variables (and more on other algorithms) sit at a box
bound, and Section 15.5 shows most of the reported improvement is a time
effect. If this box-corner, no-search solution scores near CMA-ES's own
result, search contributes little beyond "go as fast as the box allows."

Also a direct test of the 2-PRE/Section 11.5 via-point-overshoot
mechanism: with offset=0.0, the via-point sits exactly at the joint-space
midpoint on every joint, so on the five kinematically inactive joints
there is no commanded offset for the vel*weight nonzero via-point
velocity to overshoot past. If the mechanism in Section 11.5 is right,
those five joints' logged position columns should stay flat (baseline-like),
not show the out-and-back excursion Figure 15.1 shows for CMA-ES's actual
winning vector. main() checks this directly from the logged CSVs at the
end of each run and prints per-joint peak excursion -- this is a read of
the logged position column, not a recomputation of J, so it does not
touch compute_objective()'s single-source-of-truth rule.

Does not modify cmaes_optimizer.py. Reuses its decode_params() and
trajectory-construction code (via-point construction, averaged
compression timing, home-reset/success checks, compute_objective()
scoring, CSV logging convention) verbatim, copied inline per this
project's established duplication convention (same pattern already used
by replay_best_cmaes.py / replay_best_gwo.py / replay_best_pso.py /
replay_best_qpso.py) -- the only thing removed is the CMA-ES search loop.

Does NOT launch Gazebo. Connects to an already-running simulation via the
same /joint_trajectory_controller/follow_joint_trajectory action every
other script in this project uses. See the bottom of this file / the
accompanying report conversation for the exact two-step launch + run
command sequence.
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
from energy_objective import compute_objective, load_csv

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
INACTIVE_JOINTS = [0, 2, 3, 4, 5]  # all but shoulder_lift_joint (index 1)

N_JOINTS      = 6
N_RUNS        = 3
MIN_DURATION  = 2.0
LOG_DIR       = os.path.expanduser('~/ros2_ws/logs')
TRIVIAL_LOG_DIR = os.path.join(LOG_DIR, 'trivial_control')

# ─── BASELINE (Path 1 (I->_)) ─────────────────────────────────────────────────
# BASELINE_DURATION is what actually enters compute_objective() as t_ref and is
# unaffected by the T_ref state-dependency bug fixed in energy_objective.py.
# BASELINE_J below is the CORRECTED value (Table 10.2, post-fix, 0.770853),
# used here only for the console "improvement over baseline" display -- it
# does not feed compute_objective() or affect any logged J. cmaes_optimizer.py
# itself still has the pre-fix 0.770740 hardcoded in its own BASELINE_J
# constant (a display-only value there too); this script deliberately uses
# the corrected figure instead so its own printed "improvement" is
# consistent with the report this feeds into.
BASELINE_J        = 0.770853
BASELINE_DURATION = 8.090

# ─── Fixed trivial-control parameter vector (identical on all six joints) ────
TRIVIAL_VEL      = 0.55
TRIVIAL_ACCEL    = 0.55
TRIVIAL_COMPRESS = 0.500
TRIVIAL_OFFSET   = 0.0
TRIVIAL_WEIGHT   = 0.525


class TrivialControlNode(Node):

    def __init__(self):
        super().__init__('trivial_control')
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

        os.makedirs(TRIVIAL_LOG_DIR, exist_ok=True)

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

    # ─── verbatim from cmaes_optimizer.py ────────────────────────────────────
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

    # ─── verbatim from cmaes_optimizer.py ────────────────────────────────────
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

    # ─── verbatim from cmaes_optimizer.py ────────────────────────────────────
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
        """Evaluate the fixed trivial-control params vector (no search).
        Mirrors cmaes_optimizer.py's evaluate() body verbatim from the
        via-point construction onward, minus the CMA-ES loop around it."""
        avg_v = sum(p['vel']      for p in params) / N_JOINTS
        avg_c = sum(p['compress'] for p in params) / N_JOINTS
        est_dur = BASELINE_DURATION * avg_c / avg_v

        self.get_logger().info(
            f'╔══ Trivial control run {run_num}/{N_RUNS} | '
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
            return None, None, 0, None

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
            return None, None, samples, None

        if samples < 100 or T_raw < MIN_DURATION:
            self.get_logger().warn(
                f'  Trajectory failed (code=None, samples={samples}) -- penalizing'
            )
            return None, None, samples, None

        # Save this run's CSV under its own dedicated directory (never
        # overwrites cmaes_optimizer.py's own eval_N.csv files)
        csv_path = os.path.join(TRIVIAL_LOG_DIR, f'trivial_run{run_num}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)

        # Same single-source-of-truth call as every other script, explicit t_ref
        J = compute_objective(csv_path, verbose=False, t_ref=BASELINE_DURATION)

        self.get_logger().info(
            f'  ┌─ RESULT: J={J:.6f} | dur={T_raw:.2f}s | samples={samples}'
        )

        return J, T_raw, samples, csv_path


def check_inactive_joint_excursions(csv_path):
    """Direct test of the 2-PRE/Section 11.5 mechanism: with offset=0.0 there
    is no commanded via-point offset for the vel*weight via-point velocity to
    overshoot past, so the five kinematically inactive joints should stay
    flat. Reads the logged position column only -- not a J recomputation."""
    data = load_csv(csv_path)
    peaks = {}
    for j in INACTIVE_JOINTS:
        peak = max(abs(r.get(f'position_{j}', 0.0) - START_POSE[j]) for r in data)
        peaks[JOINT_NAMES[j]] = peak
    return peaks


def main():
    rclpy.init()
    node = TrivialControlNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info('Joint states received!')

    node._client.wait_for_server()

    x = [TRIVIAL_VEL, TRIVIAL_ACCEL, TRIVIAL_COMPRESS, TRIVIAL_OFFSET, TRIVIAL_WEIGHT] * N_JOINTS
    params = node.decode_params(x)

    node.get_logger().info('=' * 65)
    node.get_logger().info('  PHASE 4B TRIVIAL CONTROL (no search, fixed box-corner vector x3)')
    node.get_logger().info(f'  compress={TRIVIAL_COMPRESS}  vel={TRIVIAL_VEL}  accel={TRIVIAL_ACCEL}  '
                            f'offset={TRIVIAL_OFFSET}  weight={TRIVIAL_WEIGHT}  (all six joints)')
    node.get_logger().info(f'  Baseline J : {BASELINE_J:.6f}')
    node.get_logger().info(f'  Baseline T : {BASELINE_DURATION:.3f}s')
    node.get_logger().info('=' * 65)

    results = []
    for run_num in range(1, N_RUNS + 1):
        J, dur, samples, csv_path = node.evaluate_fixed(params, run_num)
        results.append({'run': run_num, 'J': J, 'duration': dur, 'samples': samples, 'csv': csv_path})

    node.destroy_node()
    rclpy.shutdown()

    # Save summary CSV
    summary_path = os.path.join(TRIVIAL_LOG_DIR, 'trivial_control_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'J', 'duration', 'samples'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ('run', 'J', 'duration', 'samples')} for r in results)

    # Report
    valid = [r for r in results if r['J'] is not None]
    print()
    print('=' * 65)
    print('  TRIVIAL CONTROL RESULTS (fixed box-corner vector, 3 fresh runs)')
    print('=' * 65)
    for r in results:
        if r['J'] is not None:
            print(f"  Run {r['run']}: J={r['J']:.6f}  dur={r['duration']:.2f}s  samples={r['samples']}")
        else:
            print(f"  Run {r['run']}: FAILED/PENALIZED (samples={r['samples']})")

    if len(valid) == N_RUNS:
        Js = [r['J'] for r in valid]
        durs = [r['duration'] for r in valid]
        mean_J, std_J = float(np.mean(Js)), float(np.std(Js))
        mean_dur, std_dur = float(np.mean(durs)), float(np.std(durs))
        imp = (BASELINE_J - mean_J) / BASELINE_J * 100
        print('-' * 65)
        print(f'  Mean J        : {mean_J:.6f}')
        print(f'  Std J         : {std_J:.6f}')
        print(f'  Mean duration : {mean_dur:.3f}s')
        print(f'  Std duration  : {std_dur:.3f}s')
        print(f'  Improvement over baseline ({BASELINE_J:.6f}): {imp:.2f}%')
    else:
        print('-' * 65)
        print(f'  Only {len(valid)}/{N_RUNS} runs succeeded -- cannot compute a valid mean/std.')

    print('=' * 65)
    print('  INACTIVE-JOINT EXCURSION CHECK (offset=0.0 -> should stay flat)')
    print('=' * 65)
    for r in valid:
        peaks = check_inactive_joint_excursions(r['csv'])
        print(f"  Run {r['run']}:")
        for jname, peak in peaks.items():
            flag = '  <-- NOT flat, unexplained' if peak > 0.01 else '  flat, as predicted'
            print(f"    {jname:<22} peak excursion = {peak:.4f} rad{flag}")
    print('=' * 65)
    print(f'  Raw run CSVs + summary saved under: {TRIVIAL_LOG_DIR}')


if __name__ == '__main__':
    main()
