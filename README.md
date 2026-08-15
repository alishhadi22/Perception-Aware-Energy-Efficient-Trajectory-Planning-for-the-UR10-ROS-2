# Perception-Aware, Energy-Efficient Trajectory Planning for the UR10 Cobot

A Final Year Project at the Lebanese University, investigating how
metaheuristic optimization can reduce the energy consumption of a UR10
collaborative robot's trajectories, and how camera-based obstacle
perception can be integrated into the motion planning process. The full
system is simulated in ROS 2 Jazzy with Gazebo Harmonic and MoveIt 2.

## Overview

Industrial robot trajectories are usually programmed for reachability and
cycle time, with little regard for the energy they consume. This project
treats energy as an explicit optimization objective: four population-based
metaheuristics are used to search a parameterized trajectory space and are
compared against a common, non-optimized baseline and against each other.

A second, complementary line of work looks at perception: rather than
assuming a known, static environment, the robot detects obstacles from
onboard cameras and plans a collision-free path around them, closing the
loop between what the robot sees and how it moves.

## Optimization approach

Each optimizer searches the same 30-dimensional space (five parameters per
joint, across six joints: velocity, acceleration, compression, offset, and
blending weight) and is evaluated against an identical objective function:

```
J = wE·E + wT·T + wS·S + wR·R
```

where `E`, `T`, `S`, and `R` are the trajectory's energy consumption,
execution time, smoothness (jerk), and obstacle-proximity risk, and the
weights are derived through an AHP (Analytic Hierarchy Process) pairwise
comparison. The objective function and its weighting are implemented once,
in `energy_objective.py`, and shared by all four optimizers so that results
are directly comparable.

The four algorithms compared are:

- **CMA-ES** – Covariance Matrix Adaptation Evolution Strategy
- **GWO** – Grey Wolf Optimizer
- **PSO** – Particle Swarm Optimization, with hyperparameters matched to
  El Hachem et al., *"A physics-based design-stage digital twin for
  time-energy trajectory optimization of an industrial robotic
  manipulator,"* Digital Engineering 11 (2026) 100119, for consistency with
  prior work from the same research group
- **QPSO** – Quantum-behaved Particle Swarm Optimization, following Sun,
  Feng, and Xu, *"Particle swarm optimization with particles having
  quantum behavior,"* CEC 2004, pp. 325–331

Each is run against two distinct trajectories (referred to as Path 1 and
Path 2) and evaluated over three independent runs, all under a fixed
evaluation budget so the comparison between algorithms is fair.

## Dynamics validation

The energy term relies on the joint torque Gazebo reports during
simulation, so before trusting that value, the underlying dynamics model
is checked independently. `validate_dynamics_rnea.py` implements the
manipulator's Newton-Euler inverse dynamics from the UR10's own URDF (link
masses, inertia tensors, joint origins), predicts the torque a logged
trajectory should have produced, and compares it sample-by-sample against
what Gazebo actually logged.

The comparison runs on three independent trajectories — the two
optimizer baselines and the Cartesian lift used for the hardware-trend
check below — and lands at 1–14% RMS error across all of them, with the
one outlier traced to actuator saturation (three joints pinned at their
rated torque limit for part of the motion) rather than a modelling error.
The logs each run was checked against are in `src/validation/logs/`:

| Trajectory | Log |
|---|---|
| Path 1 baseline | `path1_baseline_run1.csv` |
| Path 2 baseline | `path2_baseline_run1.csv` |
| Cartesian lift, 0.7 m/s | `path3_vertical_lift_v0.7_run1.csv` |

The link masses, inertia tensors, and joint origins the model is built
from come from the resolved URDF rather than being hand-copied out of
`ur_description`'s own config files — see
`src/validation/reference/` for why, and for unmodified copies of those
config files if you want to check the comparison yourself.

## Obstacle-aware perception and planning

To move beyond optimizing a fixed, known trajectory, this part of the
project adds a perception layer:

- **Detection** – two wrist-mounted cameras observe the workspace, and the
  resulting point cloud is clustered (k-means, k = 2) into individual
  obstacles. Each obstacle's footprint is fitted with a minimum-area
  bounding box, using PCA to estimate its orientation and falling back to
  an axis-aligned box when the PCA axis becomes unstable on near-square
  clusters. This avoids the common failure mode of representing a rotated
  object as an oversized, axis-aligned box.
- **Planning** – detected obstacles are registered as collision objects in
  the MoveIt planning scene, and a collision-free path is planned with
  OMPL's RRTConnect planner before being executed on the robot.
- **Combined pipeline** – detection, registration, planning, and execution
  are chained into a single script that runs the entire process end to end.

This component is still under active development; obstacle padding and
safety-margin tuning is the current focus. See the project's internal
`CLAUDE.md` notes for the detailed debugging history if you're continuing
this work.

## Repository structure

```
src/
├── cmaes_optimizer.py, gwo_optimizer.py,
│   pso_optimizer.py, qpso_optimizer.py   Four optimizers, identically structured
├── energy_objective.py                   Shared objective function and AHP weights
├── baseline_trajectory.py                Non-optimized reference trajectory
├── obstacle_avoidance/
│   ├── scan_obstacles_from_cameras.py    Camera-based obstacle detection
│   ├── add_obstacles_to_planning_scene.py Registers known obstacles for planning
│   ├── plan_obstacle_avoiding_path.py    Obstacle-aware planning and execution
│   ├── obstacle_avoidance_pipeline.py    End-to-end detect → plan → execute
│   └── find_scan_pose.py                 Utility: finds a camera pose for scanning
└── validation/
    ├── validate_dynamics_rnea.py         Independent Newton-Euler (RNEA) dynamics
    │                                      check against Gazebo/DART's own logged
    │                                      joint effort, per-joint and overall RMS
    │                                      error, across baseline trajectories
    ├── replay_best_cmaes.py,             Re-executes each optimizer's stored best
    │   replay_best_gwo.py,               parameter vector with no search loop, to
    │   replay_best_pso.py                isolate simulation execution noise from
    │                                     search-to-search variability
    ├── logs/                            The three trajectory logs used by
    │                                     validate_dynamics_rnea.py (see Dynamics
    │                                     validation below)
    └── reference/                       Unmodified UR10 config files from
                                          ur_description, kept for comparison
                                          against (BSD-3-Clause, see its README)

results/                                  Per-evaluation result pages (see below)
```

## Results

Each optimizer's full per-evaluation, per-joint search history is available
as an interactive page, hosted directly from this repository via GitHub
Pages:

| Algorithm | Path 1 | Path 2 |
|---|---|---|
| CMA-ES | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/cmaes_path1_evals.html) | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/cmaes_path2_evals.html) |
| GWO | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/gwo_path1_evals.html) | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/gwo_path2_evals.html) |
| PSO | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/pso_path1_evals.html) | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/pso_path2_evals.html) |
| QPSO | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/qpso_path1_evals.html) | [view](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/qpso_path2_evals.html) |

Each page shows, per run, the best trajectory found, its objective value,
and execution duration for every evaluation. CMA-ES's pages additionally
break this down per joint (velocity, acceleration, compression, offset,
weight); GWO, PSO, and QPSO report swarm-level average velocity and
compression instead, since that is the granularity their logged data
provides.

## Requirements

- ROS 2 Jazzy
- Gazebo Harmonic
- MoveIt 2
- A UR10 description and MoveIt configuration (`ur_description`,
  `ur_moveit_config`)

Each script is a standalone ROS 2 node, run directly with Python:

```bash
python3 cmaes_optimizer.py
python3 src/obstacle_avoidance/obstacle_avoidance_pipeline.py
```
