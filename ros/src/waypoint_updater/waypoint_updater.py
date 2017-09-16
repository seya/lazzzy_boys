#!/usr/bin/env python

import sys
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = []
        self.base_velocities = []
        self.traffic_light_waypoint_index = -1
        self.red_light_on = True

        rospy.spin()

    def pose_cb(self, msg):
        if len(self.waypoints) == 0:
            return
        
        # Find the index of the point that is nearest to ego vehicle
        start_index = 0
        min_dist = sys.maxint
        dist = lambda a, b: (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2
        for i in range(len(self.waypoints)):
            cur_dist = dist(msg.pose.position, self.waypoints[i].pose.pose.position)
            if cur_dist < min_dist:
                min_dist = cur_dist
                start_index = i
        
        # Create new sets of waypoints (starting from nearest waypoint)
        lane = Lane()
        lane.waypoints = []
        next_index = start_index
        for _ in range(LOOKAHEAD_WPS):
            if next_index >= len(self.waypoints):
                next_index = 0
            next_waypoint = self.waypoints[next_index]

            # set the velocity
            velocity = self.base_velocities[next_index]
            if self.red_light_on == True:
                steps_to_light = self.traffic_light_waypoint_index - next_index
                if  steps_to_light > 0 and steps_to_light < 100:
                    velocity = 0

            self.set_waypoint_velocity(self.waypoints, next_index, velocity)

            lane.waypoints.append(self.waypoints[next_index])
            next_index += 1

        # publish final waypoints
        self.final_waypoints_pub.publish(lane)


    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.base_velocities = [self.waypoints[i].twist.twist.linear.x for i in range(len(self.waypoints))]

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.red_light_on = True
        if msg.data == -1:
            self.red_light_on = False
        else:
            self.red_light_on = True
            self.traffic_light_waypoint_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
