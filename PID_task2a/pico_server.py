#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name waypoint_server which implements
an action server to navigate the Swift Pico Drone to the given waypoints.
It uses the PID controller from Task 1C as its baseline.
'''

import time
import math
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import the custom action
try:
    from waypoint_navigation.action import NavToWaypoint
except ImportError:
    print("CRITICAL: 'waypoint_navigation' package or 'NavToWaypoint' action not found.")
    print("Please create this package and action, then 'colcon build' your workspace.")
    exit(1)

# Pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray, Point # Point is for debug
from error_msg.msg import Error
from controller_msg.msg import PIDTune # For PID tuning
from nav_msgs.msg import Odometry

# RC Command Limits
MAX_ROLL = 2000
MIN_ROLL = 1000
MAX_PITCH = 2000
MIN_PITCH = 1000
MAX_THROTTLE = 2000
MIN_THROTTLE = 1000


class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')
        self.get_logger().info('PID Waypoint Server Node is running...')

        # Use ReentrantCallbackGroups to allow callbacks to run in parallel
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
        self.cmd.rc_throttle = 1500 # Start at 1500 for arming
        self.cmd.rc_aux4 = 1000

        # ==================================================================
        # PID CONTROLLER INITIALIZATION (from Task 1C Baseline)
        # ==================================================================
        
        # Drone states
        self.current_state = [0.0, 0.0, 0.0]  # x, y, z (in decimeters/Whycon units)
        # Initial desired state (will be updated by action goals)
        # Set to the first hover point from the service
        self.desired_state = [-7.00, 0.00, 29.22]

        # PID parameters
        self.Kp = [0.0, 0.0, 0.0]  # roll, pitch, throttle
        self.Ki = [0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0]

        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]

        # Min/max RC values
        self.max_values = [MAX_ROLL, MAX_PITCH, MAX_THROTTLE]
        self.min_values = [MIN_ROLL, MIN_PITCH, MIN_THROTTLE]

        # P-gain for Yaw controller (Task 2a requirement)
        self.Kp_yaw = 0.8  # Tune this value

        self.sample_time = 0.033  # 30 Hz (from PID baseline)
        # ==================================================================
        # END PID INITIALIZATION
        # ==================================================================

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        # Debug publisher for plotting desired position in PlotJuggler
        self.desired_pos_pub = self.create_publisher(Point, '/desired_position_debug', 10)


        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1,
                                 callback_group=self.subscriber_callback_group)
        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10,
                                 callback_group=self.subscriber_callback_group)
        
        # PID Tuning Subscribers (from baseline)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1,
                                 callback_group=self.subscriber_callback_group)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1,
                                 callback_group=self.subscriber_callback_group)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1,
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
        
        # Create the controller timer, now pointing to self.pid
        self.timer = self.create_timer(self.sample_time, self.pid, 
                                       callback_group=self.controller_callback_group)
        self.get_logger().info(f"PID Controller timer started at {1/self.sample_time:.1f} Hz.")

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
        self.get_logger().info("Drone ARMED.")

    # ==================================================================
    # PID CONTROLLER FUNCTIONS (from Task 1C Baseline)
    # ==================================================================

    def whycon_callback(self, msg):
        """Updates current x, y, z position from WhyCon (in decimeters)"""
        if not msg.poses or len(msg.poses) == 0:
            self.get_logger().warn("No poses received in whycon_callback", throttle_duration_sec=1.0)
            return
        
        # Update timestamp for hover-checking logic
        self.dtime = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # Use raw Whycon units (decimeters), as expected by the PID controller
        self.current_state[0] = msg.poses[0].position.x
        self.current_state[1] = msg.poses[0].position.y
        self.current_state[2] = msg.poses[0].position.z

    def odometry_callback(self, msg):
        """Updates current yaw angle from odometry"""
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw_rad = euler_from_quaternion(orientation_list)
        self.yaw = math.degrees(yaw_rad)

    # --- PID Tuning Callbacks ---
    def altitude_set_pid(self, alt):
        """Tuning callback for throttle PID"""
        self.Kp[2] = alt.kp * 0.03
        self.Ki[2] = alt.ki * 0.002
        self.Kd[2] = alt.kd * 0.6

    def pitch_set_pid(self, pitch):
        """Tuning callback for pitch PID"""
        self.Kp[1] = pitch.kp * 0.03
        self.Ki[1] = pitch.ki * 0.008
        self.Kd[1] = pitch.kd * 0.6

    def roll_set_pid(self, roll):
        """Tuning callback for roll PID"""
        self.Kp[0] = roll.kp * 0.03
        self.Ki[0] = roll.ki * 0.008
        self.Kd[0] = roll.kd * 0.6
    # --- End Tuning Callbacks ---

    def pid(self):
        """
        The main PID control loop.
        Runs at ~30Hz from the timer.
        """

        # 1. Calculate Error (Desired - Current)
        error = [
            self.desired_state[0] - self.current_state[0],  # roll (X)
            self.desired_state[1] - self.current_state[1],  # pitch (Y)
            self.desired_state[2] - self.current_state[2],  # throttle (Z)
        ]

        # 2. Calculate PID Output for each axis
        output = [0.0, 0.0, 0.0]
        for i in range(3):
            # Proportional term
            p_term = self.Kp[i] * error[i]

            # Integral term
            self.error_sum[i] += error[i] * self.sample_time
            i_term = self.Ki[i] * self.error_sum[i]
            
            # Derivative term
            error_diff = (error[i] - self.prev_error[i]) / self.sample_time
            d_term = self.Kd[i] * error_diff

            # *** CRITICAL BUG FIX from Baseline ***
            # The baseline PID code SUBTRACTED the integral term.
            # The correct PID formula ADDS it.
            output[i] = p_term + i_term + d_term

            # Update previous error for next loop
            self.prev_error[i] = error[i]

        # 3. Assign RC Commands
        # Note: Pitch (Y) and Throttle (Z) are inverted relative to their error
        self.cmd.rc_roll = int(1500 + output[0])      # Roll control (X)
        self.cmd.rc_pitch = int(1500 - output[1])     # Pitch control (Y)
        self.cmd.rc_throttle = int(1500 - output[2])  # Throttle control (Z)

        # 4. Add Yaw Control (Task 2a Requirement)
        yaw_error_deg = 0.0 - self.yaw
        yaw_command = self.Kp_yaw * yaw_error_deg
        self.cmd.rc_yaw = int(np.clip(1500 + int(yaw_command), 1000, 2000))

        # 5. Constrain all RC values
        self.cmd.rc_roll = int(np.clip(self.cmd.rc_roll, self.min_values[0], self.max_values[0]))
        self.cmd.rc_pitch = int(np.clip(self.cmd.rc_pitch, self.min_values[1], self.max_values[1]))
        self.cmd.rc_throttle = int(np.clip(self.cmd.rc_throttle, self.min_values[2], self.max_values[2]))

        # 6. Publish RC commands
        self.command_pub.publish(self.cmd)

        # 7. Publish Error and Debug Info (for PlotJuggler)
        self.publish_debug_info(error)
    
    def publish_debug_info(self, error):
        """Publishes error and setpoint topics for debugging."""
        
        # Publish position error
        pos_msg = Error()
        pos_msg.x_error = error[0]
        pos_msg.y_error = error[1]
        pos_msg.z_error = error[2]
        pos_msg.x_current = self.current_state[0]
        pos_msg.y_current = self.current_state[1]
        pos_msg.z_current = self.current_state[2]
        pos_msg.x_setpoint = self.desired_state[0]
        pos_msg.y_setpoint = self.desired_state[1]
        pos_msg.z_setpoint = self.desired_state[2]
        self.pos_error_pub.publish(pos_msg)

        # Publish desired position
        desired_msg = Point()
        desired_msg.x = self.desired_state[0]
        desired_msg.y = self.desired_state[1]
        desired_msg.z = self.desired_state[2]
        self.desired_pos_pub.publish(desired_msg)

    # ==================================================================
    # ACTION SERVER CALLBACK
    # ==================================================================
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing new waypoint goal...')
        
        # Set the PID controller's desired state from the goal
        # The coordinates are already in decimeters, which PID expects.
        # No scaling is needed.
        self.desired_state[0] = goal_handle.request.waypoint.position.x
        self.desired_state[1] = goal_handle.request.waypoint.position.y
        self.desired_state[2] = goal_handle.request.waypoint.position.z

        self.get_logger().info(f'New Waypoint Set (in decimeters): {self.desired_state}')
        
        # *** CRITICAL ***
        # Reset PID state variables (Integral and Derivative) for the new goal
        # This prevents old errors from "leaking" into the new calculation
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        
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
            # Publish feedback (no scaling needed)
            feedback_msg.current_waypoint.pose.position.x = self.current_state[0]
            feedback_msg.current_waypoint.pose.position.y = self.current_state[1]
            feedback_msg.current_waypoint.pose.position.z = self.current_state[2]
            feedback_msg.current_waypoint.header.stamp.sec = int(self.max_time_inside_sphere)
            goal_handle.publish_feedback(feedback_msg)

            # Check if drone is in the sphere
            # Use 0.4 decimeter tolerance as per Task 2a prompt
            drone_is_in_sphere = self.is_drone_in_sphere(self.current_state, goal_handle, 0.4) 

            # State machine for checking hover time
            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                pass
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                self.point_in_sphere_start_time = self.dtime
            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
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
                # Reset PID state on cancel
                self.prev_error = [0.0, 0.0, 0.0]
                self.error_sum = [0.0, 0.0, 0.0]
                return NavToWaypoint.Result()

            time.sleep(0.1)

        goal_handle.succeed()

        result.hov_time = float(self.dtime - self.duration) # Total time for this goal
        self.get_logger().info(f'Goal Succeeded! Total time: {result.hov_time:.2f}s')
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center_goal, radius_decimeters):
        # All coordinates are already in decimeters. No conversion needed.
        goal_x = sphere_center_goal.request.waypoint.position.x
        goal_y = sphere_center_goal.request.waypoint.position.y
        goal_z = sphere_center_goal.request.waypoint.position.z
        
        dist_sq = (
            (drone_pos[0] - goal_x) ** 2
            + (drone_pos[1] - goal_y) ** 2
            + (drone_pos[2] - goal_z) ** 2
        )
        return dist_sq <= (radius_decimeters ** 2)

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

