#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given drone.
This node publishes and subsribes the following topics:

		PUBLICATIONS			SUBSCRIPTIONS
		/drone_command			/whycon/poses
		/position_error

Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.	
'''

# Import Necessary Libraries for LQR
##############################################
import numpy as np
import control as ct
##############################################
import math
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from error_msg.msg import Error

import rclpy
from rclpy.node import Node

MIN_ROLL = 1000
BASE_ROLL = 1500
MAX_ROLL = 2000
SUM_ERROR_ROLL_LIMIT = 5000

MIN_PITCH = 1000
BASE_PITCH = 1500
MAX_PITCH = 2000
SUM_ERROR_PITCH_LIMIT = 5000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1530  # 4.3 #Changed according to testing
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 5000


class Swift_Pico(Node):
	def __init__(self):
		super().__init__('pico_controller')  # initializing ros node with name pico_controller

		self.m = 0.152  # Quadcopter mass (kg)
		self.g = 9.81 # Gravity (m/s^2)

  
  

  
		# Desired state [x, y, z, x_dot, y_dot, z_dot]
		# These Points are in meters and meters/second
		# and [-0.7, 0.0, 2.7, 0.0, 0.0, 0.0] corresponds to Whycon coordinates [x= -7, y= 0, z= 27]
		self.desired_state = np.array([-0.7, 0.0, 2.0, 0.0, 0.0, 0.0])   # Target state vector



		# Current state [x ,y ,z ,x_dot ,y_dot ,z_dot]
		#  whycon marker at the position of the drone given in the scene. Make the whycon marker associated with position_to_hold drone renderable and make changes accordingly
		self.current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])              # Current position [x ,y ,z ,x_dot ,y_dot ,z_dot]

		# Variables for velocity calculation
		self.prev_position = np.array([0.0, 0.0, 0.0])
		self.last_time = self.get_clock().now().nanoseconds / 1e9

		##############################################################################################
		# Derive the system dynamics matrices A and B 
		# Define the A matrix and B matrix for the state-space representation of the quadcopter
		
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
		# Understand the the state vector and input vector
		# State vector: [x, y, z, x_dot, y_dot, z_dot] (Q matrix corresponds to these states)
		# Input vector: [roll, pitch, throttle]  (R matrix corresponds to these inputs)
		
		# Define the Q and R matrices for the LQR controller
		# Q matrix penalizes deviations from the desired state
		# R matrix penalizes control effor
			
		
		# The Q and R matrices for LQR are defined to balance state error and control effort
		# These Matrices are Diagonal Matrices
		# You need to tune the values in these matrices to get the desired performance
		
		
		self.Q = np.diag([1, 1, 5, 1, 1, 20])           #  Define the Q matrix

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
		
		
		# Compute the LQR gain matrix K using the necessary library (explore the necessary library for this e.g: control etc.)
		self.K, _, _ = ct.lqr(self.A, self.B, self.Q, self.R)           # Calculate the LQR gain matrix K
		# K matrix Gain is computed using the LQR method using the A, B, Q, R matrices defined above





		# Declaring a cmd of message type swift_msgs and initializing values
		self.cmd = SwiftMsgs()
		self.cmd.rc_roll = 1500
		self.cmd.rc_pitch = 1500
		self.cmd.rc_yaw = 1500
		self.cmd.rc_throttle = 1400










		# Hint : Add variables for storing previous errors in each axis, like self.prev_error = [0,0,0] where corresponds to [pitch, roll, throttle]
		# self.prev_error = np.array([0.0, 0.0, 0.0])
		# Add variables for limiting the values like self.max_values = [2000,2000,2000] corresponding to [roll, pitch, throttle]
		# self.max_values = np.array([2000,2000,2000])
		# self.min_values = [1000,1000,1000] corresponding to [pitch, roll, throttle]
		# self.min_values = np.array([1000,1000,1000])
		# You can change the upper limit and lower limit accordingly. 
		#----------------------------------------------------------------------------------------------------------
		# A publisher for debugging the integral
		self.integral_debug_pub = self.create_publisher(Error, '/integral_debug', 10)

		#------------------------Add other ROS 2 Publishers here-----------------------------------------------------
  
  
		# # This is the sample time in which you need to run pid. Choose any time which you seem fit.
	
		self.sample_time = 0.01666  # in seconds
	
 
 
		# ===============================
		#  Publishers and Subscribers
		# ===============================
		# Publishing /drone_command, /position_error
		self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
		self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)


		# Subscribing to /whycon/poses
		self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
	
		self.arm()  # ARMING THE DRONE
  


		# Creating a timer to run the pid function periodically, refer ROS 2 tutorials on how to create a publisher subscriber(Python)
		self.timer = self.create_timer(self.sample_time, self.controller)

		
        


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
		self.command_pub.publish(self.cmd)  # Publishing /drone_command

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

	# Whycon callback function
	# The function gets executed each time when /whycon node publishes /whycon/poses 
	def whycon_callback(self, msg):

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
		# Hint : You can define a function for filtering the whycon data if you feel the need for it


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

		self.cmd.rc_roll = int(self.linear_map(0.0))	
		self.cmd.rc_pitch = int(self.linear_map(0.0))

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



def main(args=None):
	rclpy.init(args=args)
	swift_pico = Swift_Pico()
 
	try:
		rclpy.spin(swift_pico)
	except KeyboardInterrupt:
		swift_pico.get_logger().info('KeyboardInterrupt, shutting down.\n')
	finally:
		swift_pico.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
