#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name waypoint_server which implements
an action server to navigate the Swift Pico Drone to the given waypoints.
It uses the LQR controller from Task 1C.
'''

import time
import math
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import control specific libraries
import numpy as np
import control as ct

# Import the custom action
# !! IMPORTANT !!
# You must first create the 'waypoint_navigation' package and define
# 'NavToWaypoint.action' inside its 'action/' folder.
# Then build your workspace (colcon build)
try:
    from waypoint_navigation.action import NavToWaypoint
except ImportError:
    print("CRITICAL: 'waypoint_navigation' package or 'NavToWaypoint' action not found.")
    print("Please create this package and action, then 'colcon build' your workspace.")
    exit(1)

# Pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from error_msg.msg import Error
from controller_msg.msg import PIDTune  # Included from boilerplate, not used by LQR
from nav_msgs.msg import Odometry

# LQR Controller Constants
MIN_ROLL = 1000
BASE_ROLL = 1500
MAX_ROLL = 2000

MIN_PITCH = 1000
BASE_PITCH = 1500
MAX_PITCH = 2000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1530
MAX_THROTTLE = 2000

class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')
        self.get_logger().info('LQR Waypoint Server Node is running...')

        # Use ReentrantCallbackGroups to allow callbacks to run in parallel
        # This is crucial so the action server, controller, and subscribers don't block each other
        self.controller_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.subscriber_callback_group = ReentrantCallbackGroup()

        # Action server state variables
        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0
        self.dtime = self.get_clock().now().nanoseconds / 1e9

        # Drone state variables
        self.yaw = 0.0

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1400

        # ==================================================================
        # LQR CONTROLLER INITIALIZATION (from Task 1C)
        # ==================================================================
        self.m = 0.152  # Quadcopter mass (kg)
        self.g = 9.81   # Gravity (m/s^2)

        # Desired state [x, y, z, x_dot, y_dot, z_dot]
        # This will be updated by the action server goal
        self.desired_state = np.array([-0.7, 0.0, 2.0, 0.0, 0.0, 0.0])   # Initial hover state
        
        # Current state [x ,y ,z ,x_dot ,y_dot ,z_dot]
        self.current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Variables for velocity calculation
        self.prev_position = np.array([0.0, 0.0, 0.0])
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # System dynamics matrices A and B
        self.A = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

        self.B = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, -self.g, 0],
                           [self.g, 0, 0],
                           [0, 0, -1/self.m]]) # Z-axis is inverted

        # LQR Q and R tuning matrices
        self.Q = np.diag([1, 1, 5, 1, 1, 20])  # Penalize state errors
        self.R = np.diag([5, 5, 0.9])          # Penalize control effort

        # Integral altitude error
        self.integral_error_z = 0.0
        self.Ki_z = 1e-5
        self.max_integral = 5

        # Filter parameters (currently off, as alpha is 0.0)
        self.alpha_pos = 0.0    # Position filter coefficient (0-1)
        self.alpha_vel = 0.0    # Velocity filter coefficient (0-1)
        self.fpos = np.array([0.0, 0.0, 0.0])
        self.fvel = np.array([0.0, 0.0, 0.0])

        # Compute the LQR gain matrix K
        try:
            self.K, _, _ = ct.lqr(self.A, self.B, self.Q, self.R)
            self.get_logger().info('LQR gain matrix K computed successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to compute LQR gain: {e}')
            rclpy.shutdown()

        # P-gain for Yaw controller
        self.Kp_yaw = 0.8  # Tune this value
        # ==================================================================
        # END LQR INITIALIZATION
        # ==================================================================

        self.sample_time = 0.01666  # approx 60Hz

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/position_error', 10)

        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1,
                                 callback_group=self.subscriber_callback_group)
        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10,
                                 callback_group=self.subscriber_callback_group)

        # Create the Action Server
        self.get_logger().info("Creating Action Server 'waypoint_navigation'...")
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            self.execute_callback,
            callback_group=self.action_callback_group
        )
        self.get_logger().info("Action Server created.")
        
        self.arm()
        
        # Create the controller timer
        self.timer = self.create_timer(self.sample_time, self.controller, 
                                       callback_group=self.controller_callback_group)
        self.get_logger().info("Controller timer started.")

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.get_logger().info("Arming drone...")
        self.disarm()
        time.sleep(0.5) # Small delay
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500 # Set throttle to 1500 for arming
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        time.sleep(0.5) # Small delay
        # Set throttle back to base after arming
        self.cmd.rc_throttle = 1400 
        self.command_pub.publish(self.cmd)
        self.get_logger().info("Drone ARMED.")

    # ==================================================================
    # LQR CONTROLLER FUNCTIONS (from Task 1C)
    # ==================================================================

    def filter_whycon_data(self, pos, dt):
        # Applies low-pass filtering to position and velocity data
        self.fpos = self.alpha_pos * self.fpos + (1 - self.alpha_pos) * pos
        raw_vel = (pos - self.prev_position) / dt
        self.fvel = self.alpha_vel * self.fvel + (1 - self.alpha_vel) * raw_vel
        return self.fpos, self.fvel

    def whycon_callback(self, msg):
        if not msg.poses or len(msg.poses) == 0:
            self.get_logger().warn("No poses received in whycon_callback", throttle_duration_sec=1.0)
            return
        
        # Update timestamp for hover-checking logic
        self.dtime = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # Extract position (LQR uses meters)
        pos = np.array([
            msg.poses[0].position.x / 10.0,
            msg.poses[0].position.y / 10.0,
            msg.poses[0].position.z / 10.0
        ])

        # Calculate dt for velocity
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero
        self.last_time = current_time

        # Get filtered position and velocity
        fpos, fvel = self.filter_whycon_data(pos, dt)
        
        # Update the 6D current state vector
        self.current_state[0:3] = fpos
        self.current_state[3:6] = fvel

        # Update previous position for next velocity calculation
        self.prev_position = pos

    def linear_map(self, x):
        # Maps angle in radians to 1000-2000 RC range
        y = ((1000 / math.pi) * x) + 1500
        return y
    
    def force_to_throttle_linear(self, thrust_force):
        # Maps LQR thrust force deviation to 1250-2000 RC range
        hover_thrust = self.m * self.g
        total_thrust = hover_thrust + thrust_force  
        max_thrust = 2.0 * hover_thrust # Assume max thrust is 2x hover
        
        # Calculate slope
        slope = (MAX_THROTTLE - BASE_THROTTLE) / (max_thrust - hover_thrust)
        throttle = BASE_THROTTLE + slope * (total_thrust - hover_thrust)
        
        return int(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

    def odometry_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw_rad = euler_from_quaternion(orientation_list)
        
        # self.yaw is now in degrees, as in boilerplate
        self.yaw = math.degrees(yaw_rad)

    def controller(self):
        # This function runs at ~60Hz from the timer

        # 1. Compute State Error
        # Note: LQR convention is often (x - x_ref), so error = current - desired
        error = self.current_state - self.desired_state

        # 2. Compute LQR Control Input (u = -K * error)
        u = -1 * self.K @ error  # u = [roll_cmd, pitch_cmd, throttle_force_cmd]

        # 3. Process Roll and Pitch
        roll = np.clip(u[0], -math.pi/4, math.pi/4)  # Clip to +/- 45 degrees
        pitch = np.clip(-u[1], -math.pi/12, math.pi/12) # Clip to +/- 15 degrees
                                                      # Negative sign on u[1] corrects for B-matrix sign convention
        
        # Deadband for small roll/pitch commands to prevent twitching
        roll = 0.0 if abs(roll) < 0.01 else roll
        pitch = 0.0 if abs(pitch) < 0.01 else pitch

        # 4. Process Throttle (with Integral-control)
        u[2] += self.Ki_z * self.integral_error_z
        thr = self.force_to_throttle_linear(self.m * u[2])

        # Anti-windup for integral term
        # Only integrate if altitude error is significant and throttle is not saturated
        if abs(error[2]) > 0.04 and MIN_THROTTLE < thr < MAX_THROTTLE:
            self.integral_error_z -= error[2] * self.sample_time
            self.integral_error_z = np.clip(self.integral_error_z, -self.max_integral, self.max_integral)

        # 5. Process Yaw (New for Task 2a)
        # Simple P-controller to hold yaw at 0 degrees
        # Error = Desired (0) - Current (self.yaw)
        yaw_error_deg = 0.0 - self.yaw
        yaw_command = self.Kp_yaw * yaw_error_deg
        
        # Add to base yaw (1500) and clip
        yaw_rc = 1500 + int(yaw_command)
        yaw_rc = int(np.clip(yaw_rc, 1000, 2000))

        # 6. Set and Publish Commands
        
        # !! BUG FIX from Task 1C !!
        # Use the calculated roll and pitch instead of 0.0
        self.cmd.rc_roll = int(self.linear_map(roll))
        self.cmd.rc_pitch = int(self.linear_map(pitch))
        
        self.cmd.rc_throttle = thr
        self.cmd.rc_yaw = yaw_rc # Set the new yaw command

        # Publish the command
        self.command_pub.publish(self.cmd)

        # 7. Publish LQR error for debugging
        pos_error = Error()
        pos_error.pitch_error = error[0]  # x error
        pos_error.roll_error = error[1]   # y error
        pos_error.throttle_error = error[2] # z error
        self.pos_error_pub.publish(pos_error)

    # ==================================================================
    # ACTION SERVER CALLBACK
    # ==================================================================
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing new waypoint goal...')
        
        # Set the LQR controller's desired state from the goal
        # LQR uses meters, so we must scale the whycon coordinates
        # Task 2a waypoints are in (e.g., -7.00), which is 10x meters.
        # So we divide by 10, just like in whycon_callback.
        #
        # ** Correction based on problem statement: **
        # "wp1: [-7.64, 3.06, 29.22]" - These are in *decimeters*.
        # The LQR model from Task 1C was built using *meters*.
        # We must be consistent. We will convert all inputs (whycon) and
        # setpoints (goals) to meters.
        
        self.desired_state[0] = goal_handle.request.waypoint.position.x / 10.0
        self.desired_state[1] = goal_handle.request.waypoint.position.y / 10.0
        self.desired_state[2] = goal_handle.request.waypoint.position.z / 10.0
        # Desired velocities remain 0.0 for hovering
        self.desired_state[3:6] = 0.0

        self.get_logger().info(f'New Waypoint Set (in meters): {self.desired_state[0:3]}')
        
        # Reset hover-time variables
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime # Store the start time of this goal

        # Create Feedback and Result objects
        feedback_msg = NavToWaypoint.Feedback()
        result = NavToWaypoint.Result()

        # Loop until the drone has hovered for 3 seconds
        while True:
            # Publish feedback
            feedback_msg.current_waypoint.pose.position.x = self.current_state[0] * 10.0 # Convert back to decimeters for feedback
            feedback_msg.current_waypoint.pose.position.y = self.current_state[1] * 10.0
            feedback_msg.current_waypoint.pose.position.z = self.current_state[2] * 10.0
            feedback_msg.current_waypoint.header.stamp.sec = int(self.max_time_inside_sphere)
            goal_handle.publish_feedback(feedback_msg)

            # Check if drone is in the sphere
            # Use 0.08m tolerance for LQR (which is 0.8 in whycon decimeters)
            drone_is_in_sphere = self.is_drone_in_sphere(self.current_state[0:3], goal_handle, 0.08) 

            # State machine for checking hover time
            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                # State 1: Outside sphere, timer not started. Do nothing.
                pass
            
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                # State 2: Just entered sphere. Start the timer.
                self.point_in_sphere_start_time = self.dtime
                # self.get_logger().info('Drone in sphere, starting timer...')
            
            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                # State 3: Inside sphere, timer is running.
                self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                # self.get_logger().info(f'Drone in sphere for {self.time_inside_sphere:.2f}s')
            
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                # State 4: Was in sphere, but left. Reset timer.
                # self.get_logger().info('Drone left sphere, resetting timer.')
                self.point_in_sphere_start_time = None
                self.time_inside_sphere = 0

            # Update max time
            if self.time_inside_sphere > self.max_time_inside_sphere:
                self.max_time_inside_sphere = self.time_inside_sphere

            # Check for goal completion
            if self.max_time_inside_sphere >= 3:
                self.get_logger().info('Hover successful for 3 seconds.')
                break
            
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled.')
                return NavToWaypoint.Result()

            # Sleep briefly to prevent this loop from spinning too fast
            # The controller is running in its own timer thread
            time.sleep(0.1)

        goal_handle.succeed()

        result.hov_time = float(self.dtime - self.duration) # Total time for this goal
        self.get_logger().info(f'Goal Succeeded! Total time: {result.hov_time:.2f}s')
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center_goal, radius_meters):
        # Note: drone_pos is in meters, goal is in decimeters
        # We must compare them in the same unit. We'll use meters.
        goal_x_meters = sphere_center_goal.request.waypoint.position.x / 10.0
        goal_y_meters = sphere_center_goal.request.waypoint.position.y / 10.0
        goal_z_meters = sphere_center_goal.request.waypoint.position.z / 10.0
        
        dist_sq = (
            (drone_pos[0] - goal_x_meters) ** 2
            + (drone_pos[1] - goal_y_meters) ** 2
            + (drone_pos[2] - goal_z_meters) ** 2
        )
        return dist_sq <= (radius_meters ** 2)

# ==================================================================
# MAIN EXECUTION
# ==================================================================

def main(args=None):
    rclpy.init(args=args)

    waypoint_server = WayPointServer()
    
    # Use a MultiThreadedExecutor to handle all the callbacks concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_server.disarm() # Disarm the drone on exit
        executor.shutdown()
        waypoint_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
