# Perception-Aware, Energy-Efficient Trajectory Planning for the UR10 Cobot (ROS 2)

Final Year Project (Lebanese University) comparing four metaheuristic
optimizers for energy-efficient trajectory optimization of a UR10 arm,
simulated in ROS 2 Jazzy / Gazebo Harmonic / MoveIt 2.

## Optimizers compared

- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
- **GWO** (Grey Wolf Optimizer)
- **PSO** (Particle Swarm Optimization) — hyperparameters matched to
  El Hachem et al., *"A physics-based design-stage digital twin for
  time-energy trajectory optimization of an industrial robotic
  manipulator,"* Digital Engineering 11 (2026) 100119
- **QPSO** (Quantum-behaved Particle Swarm Optimization) — Sun, Feng, Xu,
  *"Particle swarm optimization with particles having quantum behavior,"*
  CEC 2004, pp. 325-331

Each optimizer searches a 30-variable space (5 parameters × 6 joints:
velocity, acceleration, compress, offset, weight) to minimize a weighted
objective `J = wE*E + wT*T + wS*S + wR*R` (energy, time, smoothness,
jerk/reversal) with AHP-derived weights (`energy_objective.py`).

## Obstacle-aware perception and planning

A second line of work adds camera-based obstacle perception feeding into
collision-free motion planning, so the "perception-aware" half of the
project is not just energy optimization on a fixed path:

- **Detection**: two wrist-mounted cameras are used to cluster the point
  cloud of two static obstacles in the workspace (k-means, k=2), then fit
  each a minimum-area oriented bounding box (PCA on the cluster's XY
  footprint, falling back to an axis-aligned box when the PCA axis is
  unstable on near-square clusters) so rotated real-world obstacles are
  registered accurately rather than as an inflated axis-aligned box.
- **Planning**: detected obstacles are registered as MoveIt collision
  objects and a collision-free path is planned with OMPL/RRTConnect
  (MoveIt's default planner) via `/plan_kinematic_path`, then executed.
- **Combined pipeline**: scan → detect → register → plan → execute is
  chained into one runnable script end-to-end.

This part of the project is still being actively debugged (padding/margin
tuning around detected obstacles is the current open issue) — see the
project's own `CLAUDE.md` (not included here) for the full bug-by-bug
history if you're picking this back up.

## Contents

- `src/` — optimizer implementations and shared dependencies:
  - `cmaes_optimizer.py`, `gwo_optimizer.py`, `pso_optimizer.py`,
    `qpso_optimizer.py` — the four optimizers, structured identically for a
    fair comparison.
  - `energy_objective.py` — the shared objective function and AHP weights.
  - `baseline_trajectory.py` — establishes the reference (non-optimized)
    trajectory each optimizer is compared against.
- `src/obstacle_avoidance/` — camera-based obstacle detection and
  obstacle-aware motion planning:
  - `scan_obstacles_from_cameras.py` — moves to a scan pose, clusters the
    two obstacles from camera point clouds, and estimates each one's
    oriented bounding box.
  - `add_obstacles_to_planning_scene.py` — registers the two known static
    obstacles (ground-truth geometry) as MoveIt collision objects, so
    planning can be tested independently of the camera pipeline.
  - `plan_obstacle_avoiding_path.py` — plans and executes a collision-free
    path (OMPL/RRTConnect) around already-registered obstacles.
  - `obstacle_avoidance_pipeline.py` — combines detection, registration,
    planning, and execution into one end-to-end script.
  - `find_scan_pose.py` — utility used to derive a wrist orientation where
    the camera actually looks at the obstacle region (via `/compute_fk`,
    no robot motion).
- `results/` — HTML evaluation-history plots for each optimizer, on two
  trajectories (Path 1, Path 2). Viewable directly in-browser (via GitHub
  Pages) rather than downloading:
  - [CMA-ES — Path 1](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/cmaes_path1_evals.html) · [Path 2](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/cmaes_path2_evals.html) (rebuilt on Path 1's template — run cards + full per-joint detail)
  - [GWO — Path 1](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/gwo_path1_evals.html) · [Path 2](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/gwo_path2_evals.html)
  - [PSO — Path 1](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/pso_path1_evals.html) · [Path 2](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/pso_path2_evals.html)
  - [QPSO — Path 1](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/qpso_path1_evals.html) · [Path 2](https://alishhadi22.github.io/Perception-Aware-Energy-Efficient-Trajectory-Planning-for-the-UR10-ROS-2/results/qpso_path2_evals.html)

## Requirements

ROS 2 Jazzy, Gazebo Harmonic, MoveIt 2, and a UR10 description/moveit config
(`ur_description`, `ur_moveit_config`). Each script is a standalone ROS 2
node run individually, e.g.:

```
python3 cmaes_optimizer.py
python3 src/obstacle_avoidance/obstacle_avoidance_pipeline.py
```
