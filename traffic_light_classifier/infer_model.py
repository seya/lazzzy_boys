import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.tl_model import TLModel
from preparedata import PrepareData
import math
import argparse
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np




class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)  
       
        return
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--split_name',  help='which split of dataset to use',  default="train")
        parser.add_argument('-c', '--checkpoint_path',  help='which checkpoint to use',  default="./logs/finetune/")
        parser.add_argument('-b', '--batch_size',  help='batch size to use',  type=int,default=1)
        args = parser.parse_args()
        
       
        self.checkpoint_path = args.checkpoint_path
        self.split_name = args.split_name
        self.batch_size = args.batch_size
            
        return
    
    
    def run(self): 
        self.parse_param()
       
        tf.logging.set_verbosity(tf.logging.INFO)
        net = TLModel()
        _ = slim.get_or_create_global_step()
        
      
        net.input, _ , net.labels,_ = self.get_input(self.split_name, is_training=False,batch_size=self.batch_size)
        net.build_eval_graph()
        
        
        num_batches = int(math.ceil(self.dataset.num_samples / float(self.batch_size)))
        # Standard evaluation loop.
       
        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_file = self.checkpoint_path
        tf.logging.info('Evaluating checkpoint_path={}, split={}'.format(checkpoint_file, self.split_name))
        
        probabilities = tf.nn.softmax(net.output)
        predictions = tf.argmax(probabilities, 1)
        init_fn = slim.assign_from_checkpoint_fn(
                checkpoint_file,
                slim.get_variables_to_restore())
       
       
        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.initialize_local_variables())
                init_fn(sess)
                
#                 np_predictions, np_images_raw, np_labels = sess.run([predictions, images_raw, labels])
                y_true = []
                y_pred = []
                for i in range(num_batches):
                    start = time.time()
                    np_predictions, np_labels = sess.run([predictions,net.labels])
                    elapsed = time.time()
                    elapsed = elapsed - start
                    print('{}/{}, {:.4f} seconds.'.format(i, num_batches, elapsed))
                    y_true.extend(np_labels.tolist())
                    y_pred.extend(np_predictions.tolist())
                # Log time spent.
                y_true  = np.array(y_true)
                y_pred = np.array(y_pred)
               
                
#                 print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))
#                 print('Time spent per image: %.3f seconds.' % (elapsed / (self.batch_size*num_batches)))
                
                print("f1_score={}".format(f1_score(y_true, y_pred, average=None)))
                print("accuracy_score={}".format(accuracy_score(y_true, y_pred)))
                
        return
    
    


if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()