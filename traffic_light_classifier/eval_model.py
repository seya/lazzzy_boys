import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.tl_model import TLModel
from preparedata import PrepareData
import math
import argparse
import time



class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)  
        self.batch_size = 32
        
        return
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--split_name',  help='which split of dataset to use',  default="train")
        parser.add_argument('-c', '--checkpoint_path',  help='which checkpoint to use',  default="./logs/")
        args = parser.parse_args()
        
       
        self.checkpoint_path = args.checkpoint_path
        self.split_name = args.split_name
            
        return
    
    
    def run(self): 
        self.parse_param()
       
        tf.logging.set_verbosity(tf.logging.INFO)
        net = TLModel()
        _ = slim.get_or_create_global_step()
        
      
        net.input, _ , net.labels,_ = self.get_input(self.split_name, is_training=False,batch_size=self.batch_size)
        net.build_eval_graph()
        
        
        num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
        # Standard evaluation loop.
       
        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_file = self.checkpoint_path
        tf.logging.info('Evaluating checkpoint_path={}, split={}'.format(checkpoint_file, self.split_name))
       
       
        logdir = './logs/evals/' + self.split_name
        start = time.time()
        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_file,
            logdir=logdir,
            num_evals=num_batches,
            eval_op=net.names_to_updates ,
            variables_to_restore=slim.get_variables_to_restore())
        # Log time spent.
        elapsed = time.time()
        elapsed = elapsed - start
        print('Time spent : %.3f seconds.' % elapsed)
        print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))
       
                    
        
        
        return
    
    


if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()