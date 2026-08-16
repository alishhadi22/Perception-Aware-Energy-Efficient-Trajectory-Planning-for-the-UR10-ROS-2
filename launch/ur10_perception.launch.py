#!/usr/bin/env python3
"""
Launch file for UR10 Perception-Aware Trajectory Optimization
Launches Gazebo simulation + MoveIt 2 for UR10, with Octomap built live
from the external camera's point cloud. Supports gazebo_gui:=false for
headless Gazebo (no 3D render window) to reduce CPU/thermal load during
data-collection runs.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
def generate_launch_description():
    gazebo_gui = LaunchConfiguration('gazebo_gui')
    declare_gazebo_gui = DeclareLaunchArgument(
        'gazebo_gui',
        default_value='true',
        description='Start gazebo with GUI? Use false for headless (less CPU/thermal load).'
    )
    ur_sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_simulation_gz'),
                'launch',
                'ur_sim_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur10',
            'gazebo_gui': gazebo_gui,
            'moveit_launch_file': PathJoinSubstitution([
                FindPackageShare('ur10_perception'),
                'launch',
                'ur_moveit_with_octomap.launch.py'
            ]),
        }.items()
    )
    return LaunchDescription([
        declare_gazebo_gui,
        ur_sim_moveit,
    ])
