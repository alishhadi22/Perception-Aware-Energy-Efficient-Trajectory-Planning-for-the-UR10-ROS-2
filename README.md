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

## Contents

- `src/` — optimizer implementations and shared dependencies:
  - `cmaes_optimizer.py`, `gwo_optimizer.py`, `pso_optimizer.py`,
    `qpso_optimizer.py` — the four optimizers, structured identically for a
    fair comparison.
  - `energy_objective.py` — the shared objective function and AHP weights.
  - `baseline_trajectory.py` — establishes the reference (non-optimized)
    trajectory each optimizer is compared against.
- `results/` — HTML evaluation-history plots for each optimizer, on two
  trajectories (Path 1, Path 2).

## Requirements

ROS 2 Jazzy, Gazebo Harmonic, MoveIt 2, and a UR10 description/moveit config
(`ur_description`, `ur_moveit_config`). Each script is a standalone ROS 2
node run individually, e.g.:

```
python3 cmaes_optimizer.py
```
