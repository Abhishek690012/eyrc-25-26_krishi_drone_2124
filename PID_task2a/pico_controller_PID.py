#!/usr/bin/env python3

'''
Swift Pico Drone Controller Node

Publications:
    /drone_command           (swift_msgs/SwiftMsgs)
    /pos_error               (error_msg/Error)
    /desired_position_debug  (geometry_msgs/Point)

Subscriptions:
    /whycon/poses            (geometry_msgs/PoseArray)
    /throttle_pid            (controller_msg/PIDTune)
    /pitch_pid               (controller_msg/PIDTune)
    /roll_pid                (controller_msg/PIDTune)
'''

from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray, Point
from controller_msg.msg import PIDTune
from error_msg.msg import Error
import rclpy
from rclpy.node import Node


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')

        # Drone states
        self.current_state = [0.0, 0.0, 0.0]  # x, y, z
        self.desired_state = [-10.0, 2.0, 20.0]  # fixed target

        # PID parameters
        self.Kp = [0.0, 0.0, 0.0]  # roll, pitch, throttle
        self.Ki = [0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0]

        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]

        self.sample_time = 0.033  # 30 Hz

        # RC command setup
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 1000

        # Min/max RC values
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.desired_pos_pub = self.create_publisher(Point, '/desired_position_debug', 10)

        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

        # PID timer
        self.timer = self.create_timer(self.sample_time, self.pid)

        # Arm drone
        self.arm()

    # ---------------------- DRONE CONTROL METHODS ------------------------

    def arm(self):
        self.disarm()
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Drone Armed')

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Drone Disarmed')

    # ---------------------- CALLBACKS ------------------------

    def whycon_callback(self, msg):
        """Updates current x, y, z position from WhyCon"""
        if len(msg.poses) > 0:
            self.current_state[0] = msg.poses[0].position.x
            self.current_state[1] = msg.poses[0].position.y
            self.current_state[2] = msg.poses[0].position.z

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

    # ---------------------- PID CONTROL ------------------------

    def pid(self):
        error = [
            self.desired_state[0] - self.current_state[0],  # roll (X)
            self.desired_state[1] - self.current_state[1],  # pitch (Y)
            self.desired_state[2] - self.current_state[2],  # throttle (Z)
        ]

        output = [0.0, 0.0, 0.0]
        for i in range(3):
            self.error_sum[i] += error[i] * self.sample_time
            error_diff = (error[i] - self.prev_error[i]) / self.sample_time

            output[i] = (
                self.Kp[i] * error[i]
                - self.Ki[i] * self.error_sum[i]
                + self.Kd[i] * error_diff
            )

            self.prev_error[i] = error[i]

        # Command computation
        self.cmd.rc_roll = int(1500 + output[0])      # Roll control (X)
        self.cmd.rc_pitch = int(1500 + output[1])     # Pitch control (Y)
        self.cmd.rc_throttle = int(1500 - output[2])  # Throttle control (Z)

        # Constrain RC values
        self.cmd.rc_roll = max(self.min_values[0], min(self.max_values[0], self.cmd.rc_roll))
        self.cmd.rc_pitch = max(self.min_values[1], min(self.max_values[1], self.cmd.rc_pitch))
        self.cmd.rc_throttle = max(self.min_values[2], min(self.max_values[2], self.cmd.rc_throttle))

        # Publish RC commands
        self.command_pub.publish(self.cmd)

        # ---------------------- PUBLISH POSITION & ERROR ------------------------
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

        # ---------------------- PUBLISH DESIRED POSITION (for PlotJuggler) ------------------------
        desired_msg = Point()
        desired_msg.x = self.desired_state[0]
        desired_msg.y = self.desired_state[1]
        desired_msg.z = self.desired_state[2]
        self.desired_pos_pub.publish(desired_msg)

    # ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    try:
        rclpy.spin(swift_pico)
    except KeyboardInterrupt:
        swift_pico.get_logger().info('KeyboardInterrupt, shutting down...')
    finally:
        swift_pico.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

