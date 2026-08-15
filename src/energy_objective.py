#!/usr/bin/env python3
"""
Energy Objective Function for UR10 Trajectory Optimization
Computes J = wE*E + wT*T + wS*S + wR*R
AHP weights: wE=0.4896, wT=0.3054, wS=0.1264, wR=0.0786 (CR=0.018)
"""

import csv
import math
import numpy as np

# ─── AHP WEIGHTS (CR = 0.018) ───────────────────────────────────────────────
W_E = 0.4896   # Energy
W_T = 0.3054   # Time
W_S = 0.1264   # Smoothness
W_R = 0.0786   # Risk (obstacle proximity, perception-aware)

# ─── NORMALIZATION REFERENCES ────────────────────────────────────────────────
E_ref = 130.0    # Reference energy (J) — max expected
T_ref = 8.042    # Reference time (s) — will be updated after baseline measurement
S_ref = 16000000000.0  # Reference smoothness (jerk integral)
R_ref = 6.67    # Reference risk = 1 / MIN_RANGE_M (0.15 m) -- worst-case sensor-limit risk

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

def compute_energy(data):
    """Compute energy as integral of |torque * velocity| over time."""
    total = 0.0
    for i in range(1, len(data)):
        dt = data[i]['time'] - data[i-1]['time']
        if dt <= 0:
            continue
        for j in range(6):
            torque = data[i].get(f'effort_{j}', 0.0)
            vel    = data[i].get(f'velocity_{j}', 0.0)
            total += abs(torque * vel) * dt
    return total

def compute_time(data):
    """Compute total trajectory duration."""
    if len(data) < 2:
        return 0.0
    return data[-1]['time'] - data[0]['time']

def compute_smoothness(data):
    """Compute smoothness as sum of squared jerk across all joints."""
    total = 0.0
    for i in range(2, len(data)):
        dt1 = data[i]['time']   - data[i-1]['time']
        dt2 = data[i-1]['time'] - data[i-2]['time']
        if dt1 <= 0 or dt2 <= 0:
            continue
        for j in range(6):
            v2 = data[i].get(f'velocity_{j}', 0.0)
            v1 = data[i-1].get(f'velocity_{j}', 0.0)
            v0 = data[i-2].get(f'velocity_{j}', 0.0)
            a1 = (v2 - v1) / dt1
            a0 = (v1 - v0) / dt2
            jerk = (a1 - a0) / ((dt1 + dt2) / 2.0)
            total += jerk ** 2
    return total

def compute_risk(data):
    """
    Compute risk as R = 1 / min_distance to the nearest obstacle, using the
    CLOSEST approach seen at any point during the trajectory (worst case).

    This replaces the earlier joint-limit-proximity definition of R. Joint
    limit safety is already enforced independently by the UR's built-in
    safety_limits controller (safety_pos_margin / safety_k_position in
    ur_sim_control.launch.py), so this term is free to represent exactly
    what nothing else in the stack covers: external obstacle collision risk,
    fed live by the Orbbec Gemini 335 depth camera via perception_node.py.

    min_distance is logged per joint-state sample (see _log_joint_state()).
    If no obstacle was ever seen (min_distance stayed at MAX_RANGE_M for the
    whole trajectory), R = 0, matching the original obstacle-free baseline.
    """
    MAX_RANGE_M = 1.40  # must match perception_node.py's MAX_RANGE_M
    closest = MAX_RANGE_M
    for row in data:
        d = row.get('min_distance', MAX_RANGE_M)
        try:
            d = float(d)
        except (TypeError, ValueError):
            continue
        if d < closest:
            closest = d
    if closest >= MAX_RANGE_M:
        return 0.0
    return 1.0 / closest

def normalize(value, ref):
    """Normalize value to [0, 1] range."""
    return min(value / max(ref, 1e-9), 1.0)

def load_csv(filepath):
    """Load trajectory CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {}
            for key, val in row.items():
                try:
                    entry[key] = float(val)
                except (ValueError, TypeError):
                    entry[key] = val
            data.append(entry)
    return data

def compute_objective(filepath, verbose=True, t_ref=None, return_breakdown=False):
    """
    Compute J = wE*E + wT*T + wS*S + wR*R from a trajectory CSV.

    Args:
        filepath: path to trajectory CSV
        verbose: print detailed results
        t_ref: override T_ref (for dynamic baseline)
        return_breakdown: if True, return a dict with the raw and normalized
            per-term values and their weighted contributions, instead of the
            scalar J. Default False preserves the original return type and
            all existing call sites unchanged.

    Returns:
        J value (float), or a breakdown dict if return_breakdown=True:
        {
          'J': float,
          'E_raw': float, 'T_raw': float, 'S_raw': float, 'R_raw': float,
          'E_norm': float, 'T_norm': float, 'S_norm': float, 'R_norm': float,
          'E_weighted': float, 'T_weighted': float, 'S_weighted': float, 'R_weighted': float,
        }
    """
    effective_t_ref = t_ref if t_ref is not None else T_ref

    data = load_csv(filepath)
    if len(data) < 2:
        return {'J': 1.0} if return_breakdown else 1.0

    E_raw = compute_energy(data)
    T_raw = compute_time(data)
    S_raw = compute_smoothness(data)
    R_raw = compute_risk(data)

    E_norm = normalize(E_raw, E_ref)
    T_norm = normalize(T_raw, effective_t_ref)
    S_norm = normalize(S_raw, S_ref)
    R_norm = normalize(R_raw, R_ref)

    J = W_E * E_norm + W_T * T_norm + W_S * S_norm + W_R * R_norm

    if verbose:
        print("=" * 55)
        print("  ENERGY OBJECTIVE FUNCTION RESULTS")
        print("=" * 55)
        print(f"  File     : {filepath.split('/')[-1]}")
        print(f"  Samples  : {len(data)}")
        print(f"  Duration : {T_raw:.3f} s")
        print("-" * 55)
        print(f"  E (Energy)     raw={E_raw:.4f}  norm={E_norm:.4f}  weight={W_E}")
        print(f"  T (Time)       raw={T_raw:.4f}  norm={T_norm:.4f}  weight={W_T}")
        print(f"  S (Smoothness) raw={S_raw:.4f}  norm={S_norm:.4f}  weight={W_S}")
        print(f"  R (Risk)       raw={R_raw:.4f}  norm={R_norm:.4f}  weight={W_R}")
        print("-" * 55)
        print(f"  J = {W_E}*{E_norm:.4f} + {W_T}*{T_norm:.4f} + {W_S}*{S_norm:.4f} + {W_R}*{R_norm:.4f}")
        print(f"  J = {J:.6f}  (lower is better)")
        print("=" * 55)

    if return_breakdown:
        return {
            'J': J,
            'E_raw': E_raw, 'T_raw': T_raw, 'S_raw': S_raw, 'R_raw': R_raw,
            'E_norm': E_norm, 'T_norm': T_norm, 'S_norm': S_norm, 'R_norm': R_norm,
            'E_weighted': W_E * E_norm, 'T_weighted': W_T * T_norm,
            'S_weighted': W_S * S_norm, 'R_weighted': W_R * R_norm,
        }
    return J


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        compute_objective(sys.argv[1])
    else:
        print("Usage: python3 energy_objective.py <trajectory.csv>")
