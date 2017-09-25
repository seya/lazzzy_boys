import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import traffic_light_classifier.nets.inception_v4 as inception_v4


from tensorflow.contrib import slim

from traffic_light_classifier.preprocessing import inception_preprocessing

class TLClassifier(object):
    def __init__(self):
        # Input placeholder.
       
        
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_size = inception_v4.inception_v4.default_image_size
        image_pre = inception_preprocessing.preprocess_image(self.img_input, image_size, image_size, is_training=False)
       
        
        
        # Define the SSD model.
        net = TLModel()
        net.input = tf.expand_dims(image_pre, 0)
        net.build_eval_graph()
        probabilities = tf.nn.softmax(net.output)
        self.predictions = tf.argmax(probabilities, 1)
        
        # Restore SSD model.
#         ckpt_filename = tf.train.latest_checkpoint('../logs/finetune/')
        ckpt_filename = tf.train.latest_checkpoint('./tlmodel/model.ckpt-26000')
        
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, ckpt_filename)
        return

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        predictons = isess.run([self.predictions],feed_dict={self.img_input: image})
        return predictons[0]
    
    def run(self):
        img_path = '/home/student/workspace/system_integration/traffic_light_classifier/data/sim_images/RED/0000000_41_318.jpg'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        res = self.get_classification(img)
        print("traffic light={}".format(res))
        return


if __name__ == "__main__":   
    obj= TLClassifier()
    obj.run()
