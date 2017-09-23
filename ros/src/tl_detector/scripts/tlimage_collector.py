#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import math
import sys

STATE_COUNT_THRESHOLD = 3

class TLImageCollector(object):
    def __init__(self):
        rospy.init_node('tlimage_collector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        

        self.bridge = CvBridge()
        self.image_id = 0
        self.last_state = TrafficLight.UNKNOWN
       
        
        # Create directory to store images captured from simulator
        self.img_folder = '/home/student/workspace/system_integration/traffic_light_classifier/data/sim_images/'   
        for folder_name in ["", "RED", "YELLOW", "GREEN", "UNKNOWN"]:
            directory_name = self.img_folder + folder_name
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        return
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        return

    def traffic_cb(self, msg):
        self.lights = msg.lights
        return
    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
    
        # Find index of closest waypoint
        closest_waypoint = 0
        min_dist = sys.maxint
        dist = lambda a, b: (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2
        for i in range(len(self.waypoints.waypoints)):
            cur_dist = dist(pose.position, 
            self.waypoints.waypoints[i].pose.pose.position)
            if cur_dist < min_dist:
                min_dist = cur_dist
                closest_waypoint = i
        return closest_waypoint

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.waypoints is None or self.pose is None or self.lights is None:
            return None
        
        camera_image = msg
        car_pose = self.pose.pose
        light_array = self.lights
       
        light_id, light = self.get_clostest_trafficlight(light_array, car_pose)
        light_state = light.state
        
        if (self.last_state != light_state):
            self.last_state = light_state
            return
        self.last_state = light_state
        dist_func = lambda a, b: math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
        meter_dist = dist_func(light.pose.pose.position, car_pose.position)
            
        
        if (meter_dist <25 or meter_dist > 150):
            # the car is too near or too far away from the traffic light
            return
        
        rospy.loginfo("ligth_id={}, distance={:.1f}, light_state={}".format(light_id, meter_dist, light_state))
        #store the image
        cv_image = self.bridge.imgmsg_to_cv2(camera_image, 'bgr8')
        
        if light_state == TrafficLight.RED:
            folder = "RED/"
        elif light_state == TrafficLight.YELLOW:
            folder = "YELLOW/"

        elif light_state == TrafficLight.GREEN:
            folder = "GREEN/"
        else:
            folder = "UNKNOWN/"
        
        
        folder = self.img_folder + folder
        img_file_name = "{}{:07d}_{}.jpg".format(folder, self.image_id,int(meter_dist))
        cv2.imwrite(img_file_name, cv_image)
        self.image_id += 1
        
        return


    def get_clostest_trafficlight(self, light_array, car_pose):
        """Identifies the closest traffic light in front of car
        Args:
            lights: the list of lights
            car: car's current pose

        Returns:
            float, the distance between the car and the cloest traffic light
            light, the cloest traffic light

        """
        
        car_position = self.get_closest_waypoint(car_pose)
        lights = []
        for tl in light_array:
            closest_wp_position = self.get_closest_waypoint(tl.pose.pose)
            lights.append((closest_wp_position, tl))
        lights.sort()

        first_tl_position = lights[0][0]
        last_tl_position = lights[-1][0]

        # Find the closest traffic light to the car
        closest_light_wp_position = len(self.waypoints.waypoints) # set the max number
        for tl_position, tl in lights:
            if ((tl_position >= car_position) and (tl_position < closest_light_wp_position) or
                (car_position > last_tl_position) and (tl_position == first_tl_position)): 
                closest_light_wp_position = tl_position
                light = tl
        
        
        return closest_light_wp_position, light
       


    

if __name__ == '__main__':
    try:
        TLImageCollector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
