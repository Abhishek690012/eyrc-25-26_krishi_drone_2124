#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# Import the action and service
# !! IMPORTANT !!
# You must first create the 'waypoint_navigation' package and define
# 'NavToWaypoint.action' and 'GetWaypoints.srv'
# Then build your workspace (colcon build)
try:
    from waypoint_navigation.action import NavToWaypoint
    from waypoint_navigation.srv import GetWaypoints
except ImportError:
    print("CRITICAL: 'waypoint_navigation' package, action, or service not found.")
    print("Please create this package, action, and service, then 'colcon build' your workspace.")
    exit(1)


class WayPointClient(Node):

    def __init__(self):
        super().__init__('waypoint_client')
        self.get_logger().info('Waypoint Client Node is running...')
        self.goals = []
        self.goal_index = 0
        
        # Create an action client for the action 'NavToWaypoint'.
        # Action name is 'waypoint_navigation'.
        self._action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')

        
        # Create a client for the service 'GetWaypoints'.
        # Service name is 'waypoints'
        self.cli = self.create_client(GetWaypoints, 'waypoints')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service "waypoints" not available, waiting again...')

        # Create a request object for GetWaypoints service.
        self.req = GetWaypoints.Request()
        self.get_logger().info('Action client and Service client created.')

    
    ### Action client functions

    def send_goal(self, waypoint):
        self.get_logger().info(f'Sending goal {self.goal_index + 1}/{len(self.goals)}: {waypoint}')

        # Create a NavToWaypoint goal object.
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = waypoint[0]
        goal_msg.waypoint.position.y = waypoint[1]
        goal_msg.waypoint.position.z = waypoint[2]

        # Wait for the action server to be available.
        self.get_logger().info('Waiting for action server "waypoint_navigation"...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server available.')

        self.send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)    
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        # Callback for when the server accepts or rejects the goal
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        # Request the result (this is a new future)
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # Callback for when the goal is finished
        
        # Complete the missing line
        result = future.result().result # Get the result from the future
        
        self.get_logger().info('Goal Finished!')
        self.get_logger().info(f'Result (Total hover stability time): {result.hov_time:.2f}s')

        # Increment goal index and send the next goal if one exists
        self.goal_index += 1
        if self.goal_index < len(self.goals):
            self.send_goal(self.goals[self.goal_index])
        else:
            self.get_logger().info('All waypoints have been reached successfully!')
            # You can add rclpy.shutdown() here if you want the node to exit
            # rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        # Callback for receiving periodic feedback from the server
        
        # Complete the missing line
        feedback = feedback_msg.feedback
        
        x = feedback.current_waypoint.pose.position.x
        y = feedback.current_waypoint.pose.position.y
        z = feedback.current_waypoint.pose.position.z
        t = feedback.current_waypoint.header.stamp.sec
        
        self.get_logger().info(f'Feedback: Pos=[{x:.2f}, {y:.2f}, {z:.2f}], Max Hover Time=[{t}s]', 
                             throttle_duration_sec=1.0) # Throttle to 1Hz


    # Service client functions

    def send_request(self):
        # Send the service request and return the future
        self.get_logger().info('Requesting waypoints from service...')
        self.future = self.cli.call_async(self.req)
        return self.future
    
    def receive_goals(self):
        future = self.send_request()
        
        # Wait until the service call is complete
        rclpy.spin_until_future_complete(self, future)
        
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return

        self.get_logger().info('Waypoints received by the action client.')

        # Populate the self.goals list
        for pose in response.waypoints.poses:
            waypoint = [pose.position.x, pose.position.y, pose.position.z]
            self.goals.append(waypoint)
            self.get_logger().info(f'Added Waypoint: {waypoint}')
        
        self.get_logger().info(f'Total {len(self.goals)} waypoints loaded.')

        # Send the first goal
        if self.goals:
            self.send_goal(self.goals[0])
        else:
            self.get_logger().warn('No waypoints received. Client will idle.')
    

def main(args=None):
    rclpy.init(args=args)

    waypoint_client = WayPointClient()
    
    # Call the service to get waypoints, which then sends the first goal
    waypoint_client.receive_goals()

    try:
        # Spin to keep the node alive to receive callbacks
        rclpy.spin(waypoint_client)
    except KeyboardInterrupt:
        waypoint_client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_client.destroy_node()
        rclpy.shutdown()
    
    # This rclpy.shutdown() is redundant if one is in the finally block
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
