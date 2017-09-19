#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import cv2

STATE_COUNT_THRESHOLD = 3

class TLImageCollector(object):
    def __init__(self):
        rospy.init_node('tlimage_collector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.image_id = 0
       
        
        # Create directory to store images captured from simulator
        self.img_folder = '/home/student/workspace/system_integration/traffic_light_classifier/data/sim_images/'   
        for folder_name in ["", "RED", "YELLOW", "GREEN", "UNKNOWN"]:
            directory_name = self.img_folder + folder_name
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        camera_image = msg
       
        dist, id, light = self.get_clostest_trafficlight(self.lights, self.pose)
        if(dist <3 or dist > 400):
            # the car is too near or too far away from the traffic light
            return
        rospy.loginfo("ligth_id={}, distance={}, light={}".format(id, dist, light))
        #store the image
        cv_image = self.bridge.imgmsg_to_cv2(camera_image, 'bgr8')
        
        if light.state == TrafficLight.RED:
            folder = "RED/"
        elif light.state == TrafficLight.YELLOW:
            folder = "YELLOW/"

        elif light.state == TrafficLight.GREEN:
            folder = "GREEN/"
        else:
            folder = "UNKNOWN/"
        
        
        folder = self.img_folder + folder
        img_file_name = "{}_{:07d}.jpg".format(folder, self.image_id)
        cv2.imwrite(img_file_name, cv_image)
        self.image_id += 1
        
        return
    def get_dist(self, a, b):
        """
        the distance between two two point (position) 
        """
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
        


    def get_clostest_trafficlight(self, lights, car):
        """Identifies the closest traffic light in front of car
        Args:
            lights: the list of lights
            car: car's current pose

        Returns:
            float, the distance between the car and the cloest traffic light
            light, the cloest traffic light

        """
        car_light_dist = []
        last_id = len(lights) - 1
        for id, light in enumerate(lights):
            dist = self.get_dist(light.pose.pose.position, car.position)
            car_light_dist.append((dist, id, light))
        car_light_dist = sorted(car_light_dist)
        closest_0_id = car_light_dist[0][1]
        closest_1_id = car_light_dist[1][1]
        if (closest_0_id in [0, last_id]) and (closest_1_id in [0, last_id]):
            #if the car is between the last and the first light, return the first light
            if (closest_0_id == 0):
                return car_light_dist[0]
            else:
                return car_light_dist[1]
        if car_light_dist[0][0] == 0:
            #if the car happens to be on a taffic light position, return this traffic light
            return car_light_dist[0]
        if closest_0_id < closest_1_id:
            #selct the light that is in front of the car
            return car_light_dist[1]
        else:
            return car_light_dist[0]
       


    

if __name__ == '__main__':
    try:
        TLImageCollector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
