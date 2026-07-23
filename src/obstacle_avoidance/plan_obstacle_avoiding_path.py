#!/usr/bin/env python3
"""
plan_obstacle_avoiding_path.py -- standalone, additive.

Unlike baseline_trajectory.py / cmaes_optimizer.py / gwo_optimizer.py /
pso_optimizer.py / qpso_optimizer.py / path3_vertical_lift.py, which all
send a fixed 2-point JointTrajectory directly to the controller (bypassing
MoveIt's motion planner entirely), THIS script uses MoveIt's real OMPL
planning pipeline (/plan_kinematic_path) to compute a collision-free path
from the I-pose to Path 1's goal pose, given the two known static
obstacles registered in the planning scene. Confirmed prerequisite
(2026-07-15): the SAME I-pose -> GOAL_POSE motion, sent as a naive direct
2-point trajectory, physically collides with these obstacles -- this
script exists to actually avoid that.

Obstacle positions are the known/hardcoded ground-truth values (same as
add_obstacles_to_planning_scene.py), registered directly in this script so
it's fully self-contained -- not the camera-detected values from
scan_obstacles_from_cameras.py, to keep this independent of that (also
experimental) pipeline and avoid the GUI-mode thermal cost of camera
rendering for something that doesn't need it: the obstacles are static
and their true positions are already known from the world SDF.

This file does not modify or import from any of the scripts above.
"""

import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import (
    CollisionObject, PlanningScene, Constraints, JointConstraint,
    MotionPlanRequest, RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import csv
import os

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
START_POSE = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]   # Path 1 I-pose
GOAL_POSE  = [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0]        # Path 1 goal

PLANNING_GROUP = "ur_manipulator"
PLANNING_FRAME = "world"
LOG_DIR = os.path.expanduser('~/ros2_ws/logs/obstacle_avoiding_plan')

JOINT_TOLERANCE = 0.01   # rad, for the goal JointConstraint
PLANNING_TIME_S = 5.0
PLANNING_ATTEMPTS = 20
# Reduced from 1.0: a first attempt at full speed found a candidate path
# that OMPL accepted during search but MoveIt's post-hoc ValidateSolution
# adapter rejected (2 of 17 waypoints invalid after time-parameterization)
# -- likely spline overshoot near an obstacle edge introduced during
# retiming. Slower retiming reduces that overshoot.
VELOCITY_SCALING = 0.3
ACCELERATION_SCALING = 0.3


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def make_box(obj_id, size_xyz, position_xyz, yaw=0.0):
    obj = CollisionObject()
    obj.header.frame_id = PLANNING_FRAME
    obj.id = obj_id
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(size_xyz)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position_xyz
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


class ObstacleAvoidingPlannerNode(Node):

    def __init__(self):
        super().__init__('plan_obstacle_avoiding_path')
        self._client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        self._joint_states = None
        self._logged_data = []
        self._logging = False
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self._scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

        os.makedirs(LOG_DIR, exist_ok=True)

    def _js_callback(self, msg):
        self._joint_states = msg
        if self._logging:
            self._log_joint_state(msg)

    def _log_joint_state(self, msg):
        row = {'time': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for j, jname in enumerate(JOINT_NAMES):
            idx = name_to_idx.get(jname, -1)
            row[f'position_{j}'] = msg.position[idx] if idx >= 0 and idx < len(msg.position) else 0.0
            row[f'velocity_{j}'] = msg.velocity[idx] if idx >= 0 and idx < len(msg.velocity) else 0.0
            row[f'effort_{j}']   = msg.effort[idx]   if idx >= 0 and idx < len(msg.effort)   else 0.0
        self._logged_data.append(row)

    def save_log(self, name):
        if not self._logged_data:
            self.get_logger().warn('No data to save!')
            return None
        timestamp = int(time.time())
        filepath = os.path.join(LOG_DIR, f'{name}_{timestamp}.csv')
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self._logged_data)
        self.get_logger().info(f'Saved: {filepath}')
        return filepath

    def register_obstacles(self):
        if not self._scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/apply_planning_scene not available -- aborting')
            return False
        pipe_obj = make_box(
            'pipe_obstacle_1', (0.15, 0.15, 0.60), (0.55, 0.25, 0.50), yaw=0.0)
        hull_obj = make_box(
            'hull_panel_obstacle_2', (0.40, 0.05, 0.40), (0.30, -0.40, 0.40), yaw=0.6)
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [pipe_obj, hull_obj]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        ok = result is not None and result.success
        self.get_logger().info(f'Obstacle registration: {"OK" if ok else "FAILED"}')
        return ok

    def move_direct(self, target_pose, duration=8.0, plan_name='move'):
        """Direct 2-point send, no planning -- used only for the initial
        move to START_POSE, which is known obstacle-free."""
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = target_pose
        pt.velocities = [0.0] * 6
        pt.accelerations = [0.0] * 6
        pt.time_from_start = Duration(seconds=duration).to_msg()
        traj.points = [pt]
        return self._send_and_wait(traj, plan_name)

    def plan_path(self, goal_pose):
        if not self._plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/plan_kinematic_path not available')
            return None

        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = PLANNING_GROUP
        mpr.num_planning_attempts = PLANNING_ATTEMPTS
        mpr.allowed_planning_time = PLANNING_TIME_S
        mpr.max_velocity_scaling_factor = VELOCITY_SCALING
        mpr.max_acceleration_scaling_factor = ACCELERATION_SCALING

        # Explicit current state -- an earlier script (path3_vertical_lift.py)
        # found /compute_cartesian_path silently fails ("Found empty
        # JointState message") if start_state is left at its default empty
        # value; populating it explicitly avoids the same issue here.
        mpr.start_state.joint_state = self._joint_states

        constraints = Constraints()
        for name, pos in zip(JOINT_NAMES, goal_pose):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = JOINT_TOLERANCE
            jc.tolerance_below = JOINT_TOLERANCE
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        mpr.goal_constraints = [constraints]

        req.motion_plan_request = mpr
        future = self._plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=PLANNING_TIME_S + 10.0)
        resp = future.result()
        if resp is None:
            self.get_logger().error('Planning call failed (no response)')
            return None

        mpres = resp.motion_plan_response
        if mpres.error_code.val != 1:  # 1 == SUCCESS
            self.get_logger().error(
                f'Planning FAILED, error_code={mpres.error_code.val} '
                '(see moveit_msgs/MoveItErrorCodes for the meaning)'
            )
            return None

        traj = mpres.trajectory.joint_trajectory
        self.get_logger().info(
            f'Planning SUCCEEDED: {len(traj.points)} waypoints, '
            f'planning_time={mpres.planning_time:.3f}s'
        )
        return traj

    def _send_and_wait(self, traj, plan_name):
        self._logged_data = []
        self._logging = True
        start_time = time.time()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._logging = False
            self.get_logger().error(f'{plan_name}: goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.time() + 60.0
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.5)
            if time.time() > deadline:
                self._logging = False
                self.get_logger().error(f'{plan_name}: execution timeout')
                return False

        self._logging = False
        duration = time.time() - start_time
        result_response = result_future.result()
        status = result_response.status if result_response else None
        error_code = result_response.result.error_code if result_response else None
        success = (status == GoalStatus.STATUS_SUCCEEDED) and (error_code == 0)

        if not success:
            self.get_logger().error(
                f'{plan_name} FAILED (status={status}, error_code={error_code}) '
                f'after {duration:.2f}s, {len(self._logged_data)} samples -- '
                'likely a physical collision or tolerance violation'
            )
            self.save_log(plan_name + '_FAILED')
            return False

        self.get_logger().info(
            f'{plan_name} completed SUCCESSFULLY in {duration:.2f}s, '
            f'{len(self._logged_data)} samples'
        )
        self.save_log(plan_name)
        return True


def main():
    rclpy.init()
    node = ObstacleAvoidingPlannerNode()

    node.get_logger().info('Waiting for joint states...')
    while node._joint_states is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node._client.wait_for_server()

    node.get_logger().info('=' * 65)
    node.get_logger().info('  OBSTACLE-AVOIDING PATH PLANNING (OMPL, real motion planner)')
    node.get_logger().info('=' * 65)

    if not node.register_obstacles():
        node.get_logger().error('Could not register obstacles -- aborting')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info('=== Moving to START pose (direct, known obstacle-free) ===')
    if not node.move_direct(START_POSE, duration=8.0, plan_name='start_pose'):
        node.get_logger().error('Could not reach start pose -- aborting')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    time.sleep(2.0)

    node.get_logger().info('=== Planning collision-free path to GOAL pose ===')
    # OMPL is a randomized planner -- observed (2026-07-15) to succeed on
    # roughly half of individual calls with these parameters (tight-ish
    # clearance near the obstacles), even though each call already retries
    # internally via num_planning_attempts. Retrying the whole planning
    # call a few more times is cheap (sub-second each) and makes this
    # reliable without needing to manually re-run the script.
    MAX_PLAN_RETRIES = 5
    traj = None
    for attempt in range(1, MAX_PLAN_RETRIES + 1):
        traj = node.plan_path(GOAL_POSE)
        if traj is not None:
            break
        node.get_logger().warn(f'Planning attempt {attempt}/{MAX_PLAN_RETRIES} failed, retrying...')
    if traj is None:
        node.get_logger().error(f'Planning failed after {MAX_PLAN_RETRIES} attempts -- nothing to execute')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info('=== Executing planned (collision-free) path ===')
    success = node._send_and_wait(traj, 'obstacle_avoiding_path')

    node.get_logger().info(f'Phase complete! success={success} Check {LOG_DIR}/')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
