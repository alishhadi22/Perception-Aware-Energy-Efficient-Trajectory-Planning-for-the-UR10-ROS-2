# Reference: UR10 kinematic/inertial parameters

Unmodified copies of `default_kinematics.yaml` and
`physical_parameters.yaml` from the `ur_description` package
(`ur10` config), included so the discussion in the report's dynamics
chapter is checkable without needing a local ROS install.

`validate_dynamics_rnea.py` does not read these files directly. Its
joint origins, link masses, centres of mass, and inertia tensors are
parsed from the fully resolved URDF `xacro` expands for Gazebo instead,
specifically to avoid re-deriving `physical_parameters.yaml`'s
"model-frame" axis convention by hand (see the annotated `x`/`y`/`z`
comments below, e.g. `y: -0.027 # -model.z`) — a manual remapping step
that adds transcription risk without adding any independent
verification value.

Source: [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description),
`urdf/ur10/`, BSD-3-Clause license, Copyright (c) Universal Robots A/S
and contributors.
