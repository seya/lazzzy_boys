import tensorflow as tf
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../traffic_light_classifier')))
# print(sys.path)
from nets import inception_v4
from preprocessing import inception_preprocessing
from datasets import trafficlight_to_tfrecords
from nets.tl_model import TLModel
import time
import glob

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
        ckpt_filename = os.path.dirname(__file__) + '/tlmodel/model.ckpt-26000'
        
        isess = tf.InteractiveSession()
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, ckpt_filename)
        self.isess = isess
        return

    def get_classification(self, image,show_time=False):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        start = time.time()
        predictons = self.isess.run(self.predictions,feed_dict={self.img_input: image})
        elapsed = time.time()
        elapsed = elapsed - start
        if show_time:
            print('inference : {:.3f} seconds.'.format(elapsed))
        
        return predictons[0]
    
    def run(self):
        traffic_clasiffication_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../traffic_light_classifier/data'))
        dataset_dir = traffic_clasiffication_path + '/sim_images/GREEN'
        img_paths = glob.glob(dataset_dir + '/**/*.jpg', recursive=True)
#         img_paths = ['sim_images/GREEN/0000120_147_784.jpg',
#                          'sim_images/GREEN/0000119_148_784.jpg',
#                          'sim_images/GREEN/0000119_148_784.jpg',
#                          'sim_images/RED/0000052_40_318.jpg',
#                          'sim_images/YELLOW/0000164_103_784.jpg']
#         img_paths = [traffic_clasiffication_path + '/'+ img_path for img_path in img_paths]
        correct_pred = 0
        for img_path in img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            pred_label = self.get_classification(img) 
            gt_label = trafficlight_to_tfrecords.class_names_to_ids[img_paths[0].split('/')[-2]]
            
            if pred_label == gt_label:
                correct_pred += 1.0
            else:
                print("actual={},pred={},image={}".format(gt_label, pred_label,img_path))
        print("acc={}".format(correct_pred/len(img_paths)))
            
            
        return


if __name__ == "__main__":   
    obj= TLClassifier()
    obj.run()
