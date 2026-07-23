#!/usr/bin/env python3
"""
add_obstacles_to_planning_scene.py

Registers the two static Gazebo obstacles (pipe_obstacle_1,
hull_panel_obstacle_2) as MoveIt collision objects, so the OMPL/RRTConnect
planner actually routes around them -- instead of only existing in Gazebo's
physics (where the arm discovers them by colliding, since MoveIt's planning
scene and Gazebo's physics world are separate and were never linked).

Geometry taken directly from ur10_sensors.sdf:
  pipe_obstacle_1:        box 0.15 x 0.15 x 0.60 @ (0.55, 0.25, 0.50), no rotation
  hull_panel_obstacle_2:  box 0.40 x 0.05 x 0.40 @ (0.30, -0.40, 0.40), yaw=0.6 rad

Run once before planning/executing trajectories:
    python3 add_obstacles_to_planning_scene.py
"""

import math
import sys

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


# Planning frame candidates to try, in order of likelihood for this setup.
# NOTE: if the service call reports success but RViz/MoveIt still doesn't
# show the obstacles, the planning frame may differ -- check with:
#   ros2 topic echo /planning_scene --once | grep -A2 header
# and adjust PLANNING_FRAME accordingly.
PLANNING_FRAME = "world"


def yaw_to_quaternion(yaw: float):
    """Quaternion for a rotation about Z only (matches SDF 'yaw' convention)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def make_box_collision_object(node: Node, obj_id: str, size_xyz, position_xyz, yaw=0.0):
    obj = CollisionObject()
    obj.header.frame_id = PLANNING_FRAME
    obj.header.stamp = node.get_clock().now().to_msg()
    obj.id = obj_id

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(size_xyz)

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position_xyz
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


def main():
    rclpy.init()
    node = Node("add_obstacles_to_planning_scene")

    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    node.get_logger().info("Waiting for /apply_planning_scene service...")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            "/apply_planning_scene service not available after 10s -- "
            "is move_group running? Aborting."
        )
        rclpy.shutdown()
        sys.exit(1)

    pipe_obj = make_box_collision_object(
        node, "pipe_obstacle_1",
        size_xyz=(0.15, 0.15, 0.60),
        position_xyz=(0.55, 0.25, 0.50),
        yaw=0.0,
    )
    hull_obj = make_box_collision_object(
        node, "hull_panel_obstacle_2",
        size_xyz=(0.40, 0.05, 0.40),
        position_xyz=(0.30, -0.40, 0.40),
        yaw=0.6,
    )

    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = [pipe_obj, hull_obj]

    request = ApplyPlanningScene.Request()
    request.scene = scene

    node.get_logger().info(
        f"Applying planning scene with 2 collision objects (frame='{PLANNING_FRAME}')..."
    )
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

    if future.result() is not None and future.result().success:
        node.get_logger().info(
            "SUCCESS: pipe_obstacle_1 and hull_panel_obstacle_2 registered "
            "with MoveIt's planning scene. The planner should now route "
            "around them instead of colliding."
        )
    else:
        node.get_logger().error(
            f"FAILED to apply planning scene. Result: {future.result()}"
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
