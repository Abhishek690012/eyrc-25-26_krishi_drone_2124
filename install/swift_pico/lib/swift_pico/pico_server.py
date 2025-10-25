#!/usr/bin/env python3

# This python file runs a ROS 2-node of name waypoint_server which implements an action server to navigate the Swift Pico Drone to the given waypoints.
# You can use either PID or LQR controller to navigate the drone to the given waypoints.


import time
import math
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

#import control specific libraries
import numpy as np
import control as ct

#import the action
from waypoint_navigation.action import NavToWaypoint

#pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from error_msg.msg import Error
from controller_msg.msg import PIDTune
from nav_msgs.msg import Odometry

MIN_THROTTLE = 1250
BASE_THROTTLE = 1530  # 4.3 #Changed according to testing
MAX_THROTTLE = 2000

class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')

        self.pid_or_lqr_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.odometry_callback_group = ReentrantCallbackGroup()

        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0


        self.yaw = 0.0
        self.xyz = [0.0, 0.0, 0.0, 0.0]
        self.dtime = 0

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1400


        #Initiate or declare other variables here depending upon whether you are implementing PID or LQR controller.

        self.m = 0.152  # Quadcopter mass (kg)
        self.g = 9.81 # Gravity (m/s^2)

		# Desired state [x, y, z, x_dot, y_dot, z_dot]
		# These Points are in meters and meters/second
        self.desired_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # Target state vector

        # Current state [x ,y ,z ,x_dot ,y_dot ,z_dot]
        #  whycon marker at the position of the drone given in the scene. 
        self.current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Variables for velocity calculation
        self.prev_position = np.array([0.0, 0.0, 0.0])
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        ##############################################################################################
        # Define the system dynamics matrices A and B matrix for the state-space representation of the quadcopter
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
                            [0, 0, -1/self.m]])
        # larger z means the drone is closer to the ground, and lowering z means flying upwards.
        # z=32 at ground, z=27 at hover height
        #####################################################################################################
        # State vector: [x, y, z, x_dot, y_dot, z_dot] (Q matrix corresponds to these states)
        # Input vector: [roll, pitch, throttle]  (R matrix corresponds to these inputs)
        # Q matrix penalizes deviations from the desired state
        # R matrix penalizes control effort

        self.Q = np.diag([0.1, 0.1, 1, 1, 1, 80])           #  Define the Q matrix
        self.R = np.diag([5, 5, 0.9])           # Define the R matrix
        
        # Initialize integral altitude error
        self.integral_error_z = 0.0
        # Integral gain – tune this value carefully
        self.Ki_z = 1e-5
        # for integral windup
        self.max_integral = 5

        # Adding filter parameters
        self.alpha_pos = 0.0    # Position filter coefficient (0-1)
        self.alpha_vel = 0.0    # Velocity filter coefficient (0-1)
        self.fpos = np.array([0.0, 0.0, 0.0])
        self.fvel = np.array([0.0, 0.0, 0.0])
        #####################################################################################################

        # Compute the LQR gain matrix K 
        self.K, _, _ = ct.lqr(self.A, self.B, self.Q, self.R)           # Calculate the LQR gain matrix K
        # K matrix Gain is computed using the LQR method using the A, B, Q, R matrices defined above



        self.sample_time = 0.01666 #put the appropriate value according to your controller

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/position_error', 10)
        # A publisher for debugging the integral
        self.integral_debug_pub = self.create_publisher(Error, '/integral_debug', 10)

        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        #Add other sunscribers here

        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10, callback_group=self.odometry_callback_group)

        #create an action server for the action 'NavToWaypoint'. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        #action name should 'waypoint_navigation'.
        #include the action_callback_group in the action server. Refer to executors in ROS 2 concepts
        self.waypoint_action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            self.execute_callback,
            callback_group=self.action_callback_group)

        
        self.arm()
        #define the function to be run inside the timer callback. This function will implement the PID or LQR algorithm
        self.timer = self.create_timer(self.sample_time, self.controller, callback_group=self.pid_or_lqr_callback_group)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)


    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)


	# Whycon callback function
	# The function gets executed each time when /whycon node publishes /whycon/poses 
    def whycon_callback(self, msg):

        if not msg.poses or len(msg.poses) == 0:
            self.get_logger().warn("No poses received")
            return

        else:
            #complete the function according to your controller as you did in task 1c.
            self.current_state[0] = (msg.poses[0].position.x / 10)  # x position in meters
            #--------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------
            self.current_state[1] = (msg.poses[0].position.y / 10) 	# y position in meters
            self.current_state[2] = (msg.poses[0].position.z / 10) 	# z position in meters
            
            current_time = self.get_clock().now().nanoseconds / 1e9
            dt = current_time - self.last_time if (current_time - self.last_time) > 0 else 1e-6  # Avoid division by zero
            self.last_time = current_time

            # Calculate velocities (x_dot, y_dot, z_dot) from the position data (and position)
            # Hint : You can use the previous position and current position to calculate velocity
            # Velocity = (Current Position - Previous Position) / Sample Time
            pos = self.current_state[0:3].copy()
            fpos, fvel = self.filter_whycon_data(pos, dt) #Filtered Position and Velocity
            
            # After the calculation of the velocities update them into the self.current_state variable accordingly
            self.current_state[0:3] = fpos
            self.current_state[3:6] = fvel
            #---------------------------------------------------------------------------------------------------------------

            # Update previous position
            self.prev_position = pos


        self.dtime = msg.header.stamp.sec



    # If you are using PID controller, then define callback function like altitide_set_pid to tune pitch, roll.
    #If you are using LQR controller, then define a functions which were given in the boiler plate code of task 1c and the ones which you have defined on yur own.
    # This function maps angles to the RC controlller stick range
    def linear_map(self,x):
        y = ((1000/math.pi) * x) + 1500
        return y


    # This function maps the thrust force to the RC controller stick range
    def force_to_throttle_linear(self, thrust_force):
        hover_thrust = self.m * self.g
        total_thrust = hover_thrust + thrust_force  
        max_thrust = 2.0 * hover_thrust 
        slope = (MAX_THROTTLE - BASE_THROTTLE) / (max_thrust - hover_thrust)
        throttle = BASE_THROTTLE + slope * (total_thrust - hover_thrust)
        return int(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))
    
    def filter_whycon_data(self, pos, dt):
        # Applies low-pass filtering to position and velocity data
        # pos (np.array): Raw position data from WhyCon [x, y, z]
        # Returns tuple: Filtered position and velocity (np.array, np.array)
        
        # Filter position
        self.fpos = self.alpha_pos * self.fpos + (1 - self.alpha_pos) * pos

        # Calculate raw velocity
        raw_vel = (pos - self.prev_position) / dt

        # Filter velocity
        self.fvel = self.alpha_vel * self.fvel + \
                        (1 - self.alpha_vel) * raw_vel

        return self.fpos, self.fvel


    def odometry_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)
        self.yaw = self.yaw_deg		

    #define the function to be run inside the timer callback. This function will implement the PID or LQR algorithm. 
    # This will be either 'pid' finction or 'controller' function as you can see in the boiler plate for PID or LQR for task 1c.
    def controller(self):
        # ===============================
        # 1. Compute State Error
        # =============================== 

        error = self.current_state - self.desired_state

        # ===============================
        # 2. Compute Control Input
        # ===============================

        # The U matrix is the control input matrix for the LQR controller
        # It is computed as the product of the state error and the LQR gain matrix K (u = -K * error)

        u = -1 * self.K @ error               # Control input from LQR  (Explore How the Matrix Multiplication is done in the necessary library you are using)

        # The Clampped values of roll, pitch and throttle are then mapped to the RC controller stick range using the linear_map and force_to_throttle_linear functions defined above
        roll  = np.clip(u[0], -math.pi/4, math.pi/4)
        pitch = np.clip(-u[1], -math.pi/12, math.pi/12)

        # Deadband for small roll/pitch commands
        roll = 0.0 if abs(roll) < 0.01 else roll
        pitch = 0.0 if abs(pitch) < 0.01 else pitch

        # integral correction on throttle
        u[2] += self.Ki_z * self.integral_error_z

        thr = int(self.force_to_throttle_linear(self.m * u[2]))

        # Anti-windup for integral term. Only integrate if throttle is not saturated.
        # Deadband for integral term: +-0.4
        if abs(error[2]) > 0.04 and MIN_THROTTLE < thr < MAX_THROTTLE:
            self.integral_error_z -= error[2] * self.sample_time
            self.integral_error_z = np.clip(self.integral_error_z, -self.max_integral, self.max_integral)
        self.integral_debug_pub.publish(Error(throttle_error=self.integral_error_z))

        self.cmd.rc_roll = int(self.linear_map(roll))	
        self.cmd.rc_pitch = int(self.linear_map(pitch))

        self.cmd.rc_throttle = thr

        # Publishing /drone_command
        self.command_pub.publish(self.cmd)


        # ===============================
        # Publish LQR error
        # ===============================

        # Publish LQR error
        pos_error = Error()
        pos_error.pitch_error = error[0]
        #------------------------------------------------------------------------------------------------------------------------
        # fill in the remaining error values for roll and throttle
        pos_error.roll_error = error[1]
        pos_error.throttle_error = error[2]
        #------------------------------------------------------------------------------------------------------------------------
        # calculate throttle error, pitch error and roll error, then publish it accordingly
        self.pos_error_pub.publish(pos_error)






    def execute_callback(self, goal_handle):

        self.get_logger().info('Executing goal...')
        self.desired_state[0] = goal_handle.request.waypoint.position.x / 10.0
        self.desired_state[1] = goal_handle.request.waypoint.position.y / 10.0
        self.desired_state[2] = goal_handle.request.waypoint.position.z / 10.0

        self.get_logger().info(f'New Waypoint Set: {self.desired_state}')
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime

        #create a NavToWaypoint feedback object. Refer to Writing an action server and client (Python) in ROS 2 tutorials.
        feedback_msg = NavToWaypoint.Feedback()
        result = NavToWaypoint.Result()

        #--------The script given below checks whether you are hovering at each of the waypoints(goals) for max of 3s---------#
        # This will help you to analyse the drone behaviour and help you to tune the PID better.

        while True:
			#current whycon poses
            feedback_msg.current_waypoint.pose.position.x = self.current_state[0]
            feedback_msg.current_waypoint.pose.position.y = self.current_state[1]
            feedback_msg.current_waypoint.pose.position.z = self.current_state[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            goal_handle.publish_feedback(feedback_msg)
            
            # Add a small delay to prevent a tight loop and allow other callbacks to run.
            time.sleep(0.1)

            drone_is_in_sphere = self.is_drone_in_sphere(self.current_state[0:3], goal_handle, 0.08) 

            # The value 0.08 is in meters used for as error range in Whycon coordinates for LQR controller. If you are using PID controller, it will become 0.8 as PID Whycon coordinates are in decimeters.

            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        pass
            
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        self.point_in_sphere_start_time = self.dtime
                        self.get_logger().info('Drone in sphere for 1st time')                        #you can choose to comment this out to get a better look at other logs

            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                        self.get_logger().info('Drone in sphere')                                     #you can choose to comment this out to get a better look at other logs
                             
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.get_logger().info('Drone out of sphere')                                 #you can choose to comment this out to get a better look at other logs
                        self.point_in_sphere_start_time = None

            if self.time_inside_sphere > self.max_time_inside_sphere:
                 self.max_time_inside_sphere = self.time_inside_sphere

            if self.max_time_inside_sphere >= 3:
                 break
                        

        goal_handle.succeed()

        #create a NavToWaypoint result object. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        result = NavToWaypoint.Result()

        result.hov_time = int(self.dtime - self.duration) #this is the total time taken by the drone in trying to stabilize at a point
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        return (
            (drone_pos[0] - sphere_center.request.waypoint.position.x) ** 2
            + (drone_pos[1] - sphere_center.request.waypoint.position.y) ** 2
            + (drone_pos[2] - sphere_center.request.waypoint.position.z) ** 2
        ) <= radius**2


def main(args=None):
    rclpy.init(args=args)

    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)
    
    try:
         executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
         waypoint_server.destroy_node()
         rclpy.shutdown()


if __name__ == '__main__':
    main()
