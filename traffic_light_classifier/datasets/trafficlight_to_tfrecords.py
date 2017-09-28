
"""Converts traffic light data to TFRecords file format with Example protos.

The raw traffic light image data set is expected to reside in JPEG files located in the
directory specified.

This TensorFlow script converts the training and evaluation data.

Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/filename: string, specifying the file name
    image/label':interger, iamge label
.
"""
import os
import tensorflow as tf
from datasets.dataset_utils import int64_feature,bytes_feature
import math
import glob
from sklearn.model_selection import train_test_split

TEST_SIZE_RATIO = 0.2
class_names_to_ids = {"RED":0, "YELLOW":1, "GREEN":2}





class TL2Tfrecords(object):
    def __init__(self):
        self.dataset_dirs = ["../data/traffic_light_bag_files/images"]
        self.name='site'
        self.output_dir = "../data/tfrecords/"
        self.num_per_shard = 1000
        
#         self.dataset_dirs = ["../data/sim_images"]
#         self.name='sim'
#         self.output_dir = "../data/tfrecords/"
#         self.num_per_shard = 2500
        return

    
    def __get_dataset_filename(self, output_dir, name, shard_id, num_shard, records_num):
        output_filename = '%s_%05d-of-%05d-total%05d.tfrecord' % (name, shard_id + 1, num_shard,records_num)
        return os.path.join(output_dir, output_filename)
    def __write2tffiles(self, filenames, name, output_dir):
        num_per_shard = self.num_per_shard
        num_shard = int(math.ceil(len(filenames) / float(num_per_shard)))
         
        for shard_id in range(num_shard):
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
            records_num = end_ndx - start_ndx
            tf_filename =self.__get_dataset_filename(output_dir, name, shard_id, num_shard, records_num)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                for i in range(start_ndx, end_ndx):
                    filename = filenames[i]
                     
                    print('Converting image %d/%d %s shard %d' % (i+1, len(filenames), filename[:-4], shard_id+1))
                    #save the file to tfrecords
                    self.__add_to_tfrecord(filename, tfrecord_writer)
        return
    def run(self):
        X_train, X_test = self.split_traineval()
        self.__write2tffiles(X_train, self.name + "_train", self.output_dir)
        self.__write2tffiles(X_test, self.name + "_eval", self.output_dir)
        print('\nFinished, train={}, eval={}'.format(len(X_train), len(X_test)))
        
        return
    def __process_image(self, filename):
        
        image_data = tf.gfile.FastGFile(filename, 'rb').read()
    
        label = filename.split('/')[-2]
        label = class_names_to_ids[label]
       
        image_format = filename.split('/')[-1].split('.')[-1]
        image_format = image_format.encode('utf-8')
        filename = filename.split('/')[-1][:-4]
        return image_data, label, filename,image_format
    def __convert_to_example(self, image_data, label, name,image_format):
        example = tf.train.Example(features=tf.train.Features(feature={
                'image/label': int64_feature(label),
                'image/filename': bytes_feature(name.encode('utf-8')),
                'image/format': bytes_feature(image_format),
                'image/encoded': bytes_feature(image_data)}))
        return example
    def __add_to_tfrecord(self, filename, tfrecord_writer):
        """Loads data from image files and add them to a TFRecord.
    
        Args:
          dataset_dir: Dataset directory;
          name: Image name to add to the TFRecord;
          tfrecord_writer: The TFRecord writer to use for writing.
        """
        image_data, labels, filename,image_format = self.__process_image(filename)
        example = self.__convert_to_example(image_data, labels, filename,image_format)
        tfrecord_writer.write(example.SerializeToString())
    
    def split_traineval(self):
        X = []
        y = []
        
        for dataset_dir in self.dataset_dirs:
            for filename in glob.iglob(dataset_dir + '/**/*.jpg', recursive=True):
                if 'UNKNOWN' in filename:
                    continue
#                 print(filename)
                base_file_name = filename.split('/')[-1][:-4]
                if  base_file_name.endswith('_318'):
                    light_distance = int(base_file_name.split('_')[-2])
                    if light_distance > 80:
                        # for traffic light 318, the traffic light is not within camera image when it's too far away
                        continue
                    
                X.append(filename)
                y.append(filename.split('/')[-2])
        X_train, X_test, _, _ = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size = TEST_SIZE_RATIO)
        return X_train, X_test
    

if __name__ == "__main__":   
    obj= TL2Tfrecords()
    obj.run()
    
    
    
    
    
