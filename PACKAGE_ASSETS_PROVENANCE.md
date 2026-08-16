# Provenance: worlds/, config/, urdf/, and two of the three launch files

These directories, plus `launch/ur_sim_control.launch.py` and
`launch/ur_sim_moveit.launch.py`, originate from
[UniversalRobots/Universal_Robots_ROS2_GZ_Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation)
(package `ur_simulation_gz`), BSD-3-Clause license, Copyright (c)
Universal Robots A/S and contributors. `launch/ur10_perception.launch.py`
is this project's own file (forwards into the two files above) and is
covered by the top-level `LICENSE` instead.

`worlds/ur10_sensors.sdf`, `config/ur_controllers.yaml`, and
`urdf/ur_gz.urdf.xacro` are modified from the upstream package, not used
unmodified:

- **`config/ur_controllers.yaml`**: `update_rate` changed from the
  upstream default to 100 Hz, aligned with the world file's physics
  profile below (Section 9.5 of the report).
- **`worlds/ur10_sensors.sdf`**: an explicit `<physics name="100hz">`
  profile added (`max_step_size=0.01`, `real_time_update_rate=100`) --
  the upstream world has no explicit physics profile at all, which
  caused the timing mismatch described in the report's Section 9.5.
  Two static obstacles (`pipe_obstacle_1`, `hull_panel_obstacle_2`) were
  also added for the obstacle-avoidance work (Chapter 17).
- **`urdf/ur_gz.urdf.xacro`**: a stereo depth-camera pair
  (`camera_link_right`, `camera_link_left`, 60 mm baseline) rigidly
  attached to `wrist_3_link`, used by the perception/obstacle-detection
  pipeline (Chapter 17). Each camera has a real `<inertial>` element, so
  its mass is included in the simulated arm dynamics.

`launch/ur_sim_control.launch.py` and `launch/ur_sim_moveit.launch.py`
are also modified, not used unmodified:

- **`launch/ur_sim_control.launch.py`**: the `world_file` default
  changed from the upstream `empty.sdf` to `ur10_sensors.sdf` above, and
  the `ros_gz_bridge` argument list extended with the two stereo camera
  point-cloud topics (`/camera/depth/right/points`,
  `/camera/depth/left/points`) so the cameras added to the URDF above
  are actually bridged into ROS 2.
- **`launch/ur_sim_moveit.launch.py`**: a `gazebo_gui` launch argument
  added and threaded through to the control launch file (used
  throughout the report's hardware-constraints discussion, Chapter 6,
  to run headless for long unattended optimizer campaigns); `launch_rviz`
  changed from the upstream default `"true"` to `"false"` (RViz is pure
  visualization, not used by anything in this pipeline, and disabling it
  measurably reduced GUI-mode thermal load).

`launch/ur10_perception.launch.py` (this project's own file, covered by
the top-level `LICENSE` instead) forwards into `ur_sim_moveit.launch.py`
above.
