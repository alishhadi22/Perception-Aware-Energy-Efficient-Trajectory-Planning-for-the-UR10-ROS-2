#!/usr/bin/env python3
"""
Newton-Euler (RNEA) Dynamics Model Validation

Validates the hand-derived Recursive Newton-Euler dynamics model (Chapter 7)
against Gazebo's own DART physics engine. This is DIFFERENT from the
Chapter 16 three-level validation (which checks that the reported objective
J is correctly computed/reproducible/hardware-consistent) -- this instead
checks whether the derived rigid-body dynamics MATH actually predicts what
the simulator internally computes, which is what justifies treating logged
/joint_states effort as physically meaningful torque in energy_objective.py.

Method: an independent, standalone RNEA implementation is fed the exact
logged q(t), qdot(t) from a baseline trajectory CSV (qddot is recovered by
central finite-differencing qdot, since it is not itself logged). The
resulting predicted joint torques tau_rnea(t) are compared, joint-by-joint,
against Gazebo's own logged effort_j(t) at the same timestep -- the ground
truth energy_objective.py's E term already relies on. Per-joint RMS error
is reported.

Kinematic/inertial parameters (link masses, centers of mass, inertia
tensors, joint origins) are NOT re-derived from the raw
default_kinematics.yaml/physical_parameters.yaml config files by hand.
Instead they are read directly from the exact URDF `xacro` expands into
Gazebo (verified: every joint <origin> in that URDF matches the DH-derived
offsets in default_kinematics.yaml exactly, e.g. shoulder_pan_joint's
z=0.1273 = kinematics.yaml's shoulder.z). Pulling them from the resolved
URDF avoids re-deriving the same model-frame-vs-DH-frame axis remapping
ur_description's own xacro macros already had to resolve once (the
physical_parameters.yaml CoM/rotation entries are annotated "model.x",
"-model.z", etc. -- a nontrivial permutation that is easy to get wrong by
hand and does not add any independent verification value here).

Uses the CSV logged 2026-07-08 for Path 1 (I->_), before the 2026-07-15
camera/obstacle additions -- so it is a clean, unloaded, non-colliding
open-chain motion (no external contact wrench term needed in the RNEA,
and no extra 0.2kg camera-pair mass on wrist_3_link to account for).
Correspondingly, link parameters are taken from the pre-camera URDF
(`ur_gz.urdf.xacro.bak_no_cameras`), matching what was physically true in
Gazebo when that CSV was recorded.

Confirmed from the URDF: all 6 joints have <dynamics damping="0"
friction="0"/> -- Gazebo models NO joint friction/damping, so a pure
rigid-body RNEA (no friction term) is in fact the exact correct comparison
model, not an approximation.
"""

import csv
import math
import os
import sys

import numpy as np

# ─── LINK CHAIN PARAMETERS ────────────────────────────────────────────────
# All values read directly from the pre-camera Gazebo URDF (`xacro
# ur_gz.urdf.xacro.bak_no_cameras name:=ur ur_type:=ur10`), 2026-07-29.
# joint_origin_xyz/rpy: fixed offset from parent link's frame to this
#   joint's frame (URDF <joint><origin>), BEFORE the joint's own qi rotation
#   about local +z is applied. This offset does not move when qi changes.
# com_xyz/rpy: link's <inertial><origin> -- position/orientation of the
#   inertia tensor's principal-axis frame, expressed in the LINK's own frame
#   (which is identical to the joint frame above, confirmed empirically:
#   joint origins in the URDF match default_kinematics.yaml's DH offsets
#   exactly).
# inertia_xx..zz: <inertial><inertia>, expressed in the rotated com_rpy frame.

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Per-joint max_effort (N*m), from ur_description's own
# config/ur10/joint_limits.yaml. Used to detect actuator saturation: when
# Gazebo's reported effort sits at this ceiling, it reflects the
# actuator's rated torque limit, not the physically-required torque RNEA
# computes -- comparing the two at a saturated sample is not a fair test
# of the dynamics model (found 2026-07-29 investigating a ~100% RMS error
# on Path 2, which demands much more holding torque than Path 1/3 near the
# end of its trajectory; confirmed the "error" there was
# shoulder_pan/shoulder_lift/elbow all pinned exactly at 330/330/150 N*m
# for the same ~55/810 samples across all 3 runs, not a modeling bug).
MAX_EFFORT = [330.0, 330.0, 150.0, 56.0, 56.0, 56.0]
SATURATION_MARGIN = 0.5  # N*m headroom below max_effort still counted as saturated

LINKS = [
    # name              joint_origin_xyz        joint_origin_rpy                                              mass   com_xyz                  com_rpy                 inertia (ixx,ixy,ixz,iyy,iyz,izz)
    dict(name='shoulder_link',
         j_xyz=(0.0, 0.0, 0.1273), j_rpy=(0.0, 0.0, 0.0),
         mass=7.1, com=(0.021, -0.027, 0.0), com_rpy=(1.570796326794897, 0.0, 0.0),
         I=(0.03408, 0.00002, -0.00425, 0.03529, 0.00008, 0.02156)),
    dict(name='upper_arm_link',
         j_xyz=(0.0, 0.0, 0.0), j_rpy=(1.570796327, 0.0, 0.0),
         mass=12.7, com=(-0.232, 0.0, 0.158), com_rpy=(0.0, 0.0, 0.0),
         I=(0.02814, 0.00005, -0.01561, 0.77068, 0.00002, 0.76943)),
    dict(name='forearm_link',
         j_xyz=(-0.612, 0.0, 0.0), j_rpy=(0.0, 0.0, 0.0),
         mass=4.27, com=(-0.3323, 0.0, 0.068), com_rpy=(0.0, 0.0, 0.0),
         I=(0.01014, 0.00008, 0.00916, 0.30928, 0.0, 0.30646)),
    dict(name='wrist_1_link',
         j_xyz=(-0.5723, 0.0, 0.163941), j_rpy=(0.0, 0.0, 0.0),
         mass=2.0, com=(0.0, -0.018, 0.007), com_rpy=(1.570796326794897, 0.0, 0.0),
         I=(0.00296, -0.00001, 0.0, 0.00222, -0.00024, 0.00258)),
    dict(name='wrist_2_link',
         j_xyz=(0.0, -0.1157, 0.0), j_rpy=(1.570796327, 0.0, 0.0),
         mass=2.0, com=(0.0, 0.018, -0.007), com_rpy=(-1.570796326794897, 0.0, 0.0),
         I=(0.00296, -0.00001, 0.0, 0.00222, -0.00024, 0.00258)),
    dict(name='wrist_3_link',
         j_xyz=(0.0, 0.0922, 0.0), j_rpy=(1.570796326589793, 3.141592653589793, 3.141592653589793),
         mass=0.365, com=(0.0, 0.0, -0.026), com_rpy=(0.0, 0.0, 0.0),
         I=(0.0004, 0.0, 0.0, 0.00041, 0.0, 0.00034)),
]

Z_AXIS = np.array([0.0, 0.0, 1.0])

# SDFormat's documented default <world><gravity> is "0 0 -9.8" (not 9.81 or
# 9.80665) when a world file does not override it -- confirmed:
# ur10_sensors.sdf has no <gravity> element, so Gazebo uses this default.
GRAVITY = 9.8


def rpy_to_matrix(roll, pitch, yaw):
    """URDF <origin rpy="r p y"/> convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def inertia_vec_to_matrix(I):
    ixx, ixy, ixz, iyy, iyz, izz = I
    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])


# Precompute fixed per-joint quantities once.
_R_ORIGIN = [rpy_to_matrix(*link['j_rpy']) for link in LINKS]          # R_origin_i
_P_ORIGIN = [np.array(link['j_xyz']) for link in LINKS]                # p_origin_i, in frame i-1
_COM = [np.array(link['com']) for link in LINKS]                       # p_ci, in frame i
_R_COM = [rpy_to_matrix(*link['com_rpy']) for link in LINKS]            # orientation of inertia frame in frame i
_I_COM_LINK_FRAME = [
    _R_COM[i] @ inertia_vec_to_matrix(LINKS[i]['I']) @ _R_COM[i].T
    for i in range(6)
]  # inertia tensor about CoM, expressed in link i's own frame
_MASS = [link['mass'] for link in LINKS]


def rnea(q, qdot, qddot, gravity=GRAVITY):
    """
    Recursive Newton-Euler inverse dynamics for the UR10's 6R open chain.

    q, qdot, qddot: length-6 arrays (rad, rad/s, rad/s^2), joint order
    matching JOINT_NAMES.

    Returns: length-6 array of joint torques (N*m), same order.

    Derivation (all quantities for link i expressed IN FRAME i unless noted):
      R_{i-1,i} = R_origin_i @ Rz(q_i)          (frame i -> frame i-1)
      R_{i,i-1} = R_{i-1,i}.T                   (frame i-1 -> frame i)
      p_i       = p_origin_i                    (frame i-1 origin -> frame i
                                                  origin, expressed in frame
                                                  i-1; constant, independent
                                                  of q_i since a pure
                                                  rotation about frame i's
                                                  own z does not move its
                                                  origin)

    Outward (i=1..6), gravity folded in via the standard trick of setting
    the base's linear acceleration to +g (world frame) instead of 0:
      omega_i = R_{i,i-1} @ omega_{i-1} + qdot_i * z
      alpha_i = R_{i,i-1} @ alpha_{i-1} + (R_{i,i-1} @ omega_{i-1}) x (qdot_i*z) + qddot_i*z
      a_i     = R_{i,i-1} @ (alpha_{i-1} x p_i + omega_{i-1} x (omega_{i-1} x p_i) + a_{i-1})
      a_ci    = a_i + alpha_i x p_ci + omega_i x (omega_i x p_ci)
      F_i     = m_i * a_ci
      N_i     = I_ci @ alpha_i + omega_i x (I_ci @ omega_i)

    Inward (i=6..1), f_7 = n_7 = 0 (no external tool/tip wrench):
      f_i = R_{i,i+1} @ f_{i+1} + F_i
      n_i = N_i + R_{i,i+1} @ n_{i+1} + p_ci x F_i + p_origin_{i+1} x (R_{i,i+1} @ f_{i+1})
      tau_i = n_i . z
    """
    omega = [np.zeros(3)]  # index 0 = base
    alpha = [np.zeros(3)]
    a = [np.array([0.0, 0.0, gravity])]  # fictitious base accel = +g

    R_out = []  # R_{i-1,i}, frame i -> frame i-1 (i.e. "outward"/child-to-parent rotation)

    for i in range(6):
        Rz_qi = rpy_to_matrix(0.0, 0.0, q[i])
        R_iminus1_i = _R_ORIGIN[i] @ Rz_qi          # frame i -> frame i-1
        R_out.append(R_iminus1_i)
        R_i_iminus1 = R_iminus1_i.T                  # frame i-1 -> frame i

        omega_prev_in_i = R_i_iminus1 @ omega[i]
        omega_i = omega_prev_in_i + qdot[i] * Z_AXIS

        alpha_i = (R_i_iminus1 @ alpha[i]
                   + np.cross(omega_prev_in_i, qdot[i] * Z_AXIS)
                   + qddot[i] * Z_AXIS)

        p_i = _P_ORIGIN[i]
        a_i = R_i_iminus1 @ (np.cross(alpha[i], p_i)
                             + np.cross(omega[i], np.cross(omega[i], p_i))
                             + a[i])

        omega.append(omega_i)
        alpha.append(alpha_i)
        a.append(a_i)

    F = [None] * 6
    N = [None] * 6
    for i in range(6):
        p_ci = _COM[i]
        omega_i = omega[i + 1]
        alpha_i = alpha[i + 1]
        a_ci = a[i + 1] + np.cross(alpha_i, p_ci) + np.cross(omega_i, np.cross(omega_i, p_ci))
        F[i] = _MASS[i] * a_ci
        I_ci = _I_COM_LINK_FRAME[i]
        N[i] = I_ci @ alpha_i + np.cross(omega_i, I_ci @ omega_i)

    f_next = np.zeros(3)
    n_next = np.zeros(3)
    tau = np.zeros(6)
    for i in range(5, -1, -1):
        if i < 5:
            R_i_iplus1 = R_out[i + 1]  # frame i+1 -> frame i
            p_origin_next = _P_ORIGIN[i + 1]
        else:
            R_i_iplus1 = np.zeros((3, 3))
            p_origin_next = np.zeros(3)

        f_i = (R_i_iplus1 @ f_next) + F[i]
        n_i = (N[i] + R_i_iplus1 @ n_next
               + np.cross(_COM[i], F[i])
               + np.cross(p_origin_next, R_i_iplus1 @ f_next))

        tau[i] = n_i @ Z_AXIS
        f_next = f_i
        n_next = n_i

    return tau


# ─── CSV I/O + FINITE-DIFFERENCE ACCELERATION ─────────────────────────────

def load_csv(filepath):
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


def extract_arrays(data):
    t = np.array([row['time'] for row in data])
    q = np.array([[row[f'position_{j}'] for j in range(6)] for row in data])
    qdot = np.array([[row[f'velocity_{j}'] for j in range(6)] for row in data])
    effort = np.array([[row[f'effort_{j}'] for j in range(6)] for row in data])
    return t, q, qdot, effort


def central_diff_accel(t, qdot):
    """qddot via central differences of the LOGGED velocity (not position --
    velocity is Gazebo's own reported joint state, not itself finite-
    differenced by this script, so only one differentiation step is
    introduced here, not two)."""
    n = len(t)
    qddot = np.zeros_like(qdot)
    for i in range(1, n - 1):
        dt = t[i + 1] - t[i - 1]
        if dt <= 0:
            qddot[i] = qddot[i - 1]
            continue
        qddot[i] = (qdot[i + 1] - qdot[i - 1]) / dt
    if n >= 2:
        qddot[0] = qddot[1] if n > 2 else 0.0
        qddot[-1] = qddot[-2] if n > 2 else 0.0
    return qddot


def savgol_diff_accel(t, qdot, window=11, polyorder=3):
    """
    qddot via a Savitzky-Golay smoothed derivative of the logged velocity
    (still a single differentiation step, same as central_diff_accel --
    this only changes HOW that one derivative is estimated). Plain central
    differences amplify the real per-step jitter already present in
    Gazebo/DART's reported velocity (visible as high-frequency noise on
    both the predicted AND actual torque once differentiated twice
    overall -- position -> velocity is Gazebo's own internal state, so
    RNEA's differentiation of velocity is the only extra derivative taken
    here). Local polynomial smoothing before differencing is standard
    practice for numerically differentiating noisy sampled data and does
    not change what is being measured, only how robustly the derivative is
    estimated. window/polyorder are deliberately modest (0.1s window at
    100Hz, cubic fit) -- large enough to reject sample-to-sample jitter,
    small enough not to blur the actual several-second trajectory shape.
    """
    from scipy.signal import savgol_filter
    dt = float(np.median(np.diff(t)))
    n = len(t)
    w = min(window, n - (1 - n % 2))
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return central_diff_accel(t, qdot)
    qddot = np.zeros_like(qdot)
    for j in range(qdot.shape[1]):
        qddot[:, j] = savgol_filter(qdot[:, j], w, polyorder, deriv=1, delta=dt)
    return qddot


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def validate(filepath, gravity=GRAVITY, trim=5, out_csv=None, accel_method='central'):
    """
    trim: number of samples dropped from each end of the trajectory before
    scoring. The controller briefly settles at the very start/end of a
    FollowJointTrajectory execution (near-zero velocity, PID-transient
    effort) where finite-differenced acceleration is least reliable and
    the "effort" reading is dominated by settling noise rather than
    tracking a smooth commanded motion -- excluding these avoids scoring
    an artifact of the differencing/settling process rather than the
    dynamics model itself. Kept small and stated explicitly so it cannot
    quietly hide a real mismatch.
    """
    data = load_csv(filepath)
    t, q, qdot, effort = extract_arrays(data)
    qddot = (savgol_diff_accel(t, qdot) if accel_method == 'savgol'
             else central_diff_accel(t, qdot))

    n = len(t)
    tau_rnea = np.zeros((n, 6))
    for i in range(n):
        tau_rnea[i] = rnea(q[i], qdot[i], qddot[i], gravity=gravity)

    lo, hi = trim, n - trim
    tau_rnea_s = tau_rnea[lo:hi]
    effort_s = effort[lo:hi]
    err = tau_rnea_s - effort_s

    # Actuator saturation mask: True where Gazebo's reported effort sits at
    # that joint's rated max_effort ceiling. At those samples the actuator
    # is physically incapable of applying more torque than commanded, so
    # "effort" no longer reflects the physically-required torque RNEA
    # computes -- scoring RNEA against a clipped reading there would blame
    # the dynamics model for an actuator limit, not a modeling error.
    # Exclusion is applied per-TIMESTEP across ALL joints, not just the
    # saturated joint's own column: when one actuator saturates, the whole
    # arm's real trajectory deviates from a smooth commanded motion (it is
    # physically failing to track/hold the commanded state), which
    # corrupts the finite-differenced qddot and the dynamic coupling terms
    # for every joint at that instant, confirmed by inspection -- e.g.
    # wrist_1/wrist_2 (never themselves saturated) showed their own worst
    # errors exactly at the same timesteps shoulder_pan/lift/elbow were
    # pinned at their limits.
    per_joint_sat = np.zeros_like(effort_s, dtype=bool)
    for j in range(6):
        per_joint_sat[:, j] = np.abs(effort_s[:, j]) >= (MAX_EFFORT[j] - SATURATION_MARGIN)
    any_sat = per_joint_sat.any(axis=1)
    sat_mask = np.tile(any_sat[:, None], (1, 6))

    print("=" * 70)
    print("  NEWTON-EULER (RNEA) DYNAMICS MODEL VALIDATION vs GAZEBO/DART")
    print("=" * 70)
    print(f"  File     : {os.path.basename(filepath)}")
    print(f"  Samples  : {n} total, {hi - lo} scored (first/last {trim} trimmed)")
    print(f"  Gravity  : {gravity} m/s^2")
    print(f"  Accel method: {accel_method}")
    if sat_mask.any():
        sat_joints = [JOINT_NAMES[j] for j in range(6) if per_joint_sat[:, j].any()]
        print(f"  WARNING: actuator saturation detected on {', '.join(sat_joints)}")
        print(f"           -- excluding {int(any_sat.sum())}/{len(any_sat)} affected timesteps")
        print(f"           from ALL joints' RMS scoring (see below)")
    print("-" * 70)
    print(f"  {'Joint':<20}{'RMS err (N*m)':>16}{'RMS |tau_gz| (N*m)':>20}{'% of scale':>14}{'  sat.excl.':>12}")
    for j in range(6):
        keep = ~sat_mask[:, j]
        n_excl = int(sat_mask[:, j].sum())
        rms_err = rms(err[keep, j])
        scale = rms(effort_s[keep, j])
        pct = (rms_err / scale * 100.0) if scale > 1e-9 else float('nan')
        excl_str = f'{n_excl}/{len(keep)}' if n_excl > 0 else '-'
        print(f"  {JOINT_NAMES[j]:<20}{rms_err:>16.4f}{scale:>20.4f}{pct:>13.2f}%{excl_str:>12}")
    keep_all = ~sat_mask.any(axis=1)
    overall = rms(err[keep_all].flatten())
    overall_scale = rms(effort_s[keep_all].flatten())
    n_excl_any = int((~keep_all).sum())
    print("-" * 70)
    excl_note = f" ({n_excl_any}/{len(keep_all)} samples excluded, any-joint saturated)" if n_excl_any else ""
    print(f"  Overall RMS error: {overall:.4f} N*m "
          f"({overall / overall_scale * 100.0:.2f}% of overall RMS |tau_gazebo|){excl_note}")
    print("=" * 70)

    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time'] + [f'tau_rnea_{j}' for j in range(6)] + [f'tau_gazebo_{j}' for j in range(6)]
            writer.writerow(header)
            for i in range(lo, hi):
                writer.writerow([t[i]] + list(tau_rnea[i]) + list(effort[i]))
        print(f"  Per-timestep comparison saved: {out_csv}")

    return tau_rnea, effort, t


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 validate_dynamics_rnea.py <baseline_trajectory.csv> [out_comparison.csv]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    validate(sys.argv[1], out_csv=out)
