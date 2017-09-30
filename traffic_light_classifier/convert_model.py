import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.tl_model import TLModel
from preparedata import PrepareData
from nets import inception_v4
import cv2
from preprocessing import inception_preprocessing




class ConvertModel(PrepareData):
    def __init__(self):
        return
    def convert(self):
        net = TLModel()
       
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3), name="inputs")
        # Evaluation pre-processing: resize to SSD net shape.
        image_size = inception_v4.inception_v4.default_image_size
        image_pre = inception_preprocessing.preprocess_image(self.img_input, image_size, image_size, is_training=False)
        net.input = tf.expand_dims(image_pre, 0)
        net.build_eval_graph()
        
        
      
        # Standard evaluation loop.
       
        checkpoint_file = './logs/finetune/model.ckpt-27774'
        tf.logging.info('convert checkpoint_path={}'.format(checkpoint_file))
        
        probabilities = tf.nn.softmax(net.output)
        predictions = tf.argmax(probabilities, 1,name="predictions")
        init_fn = slim.assign_from_checkpoint_fn(
                checkpoint_file,
                slim.get_variables_to_restore())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

       
       
        with tf.Session() as sess:
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            # Save GraphDef
            tf.train.write_graph(sess.graph_def,'./logs/deploy','graph.pb')
            # Save checkpoint
            save_path = saver.save(sess, "./logs/deploy/tlmodel")
            print("Model saved in file: %s" % save_path)
        return
    def use_optimized_model(self):
        model_file = './logs/deploy/graph_optimized.pb'
        with tf.gfile.GFile(model_file, 'rb') as f:
            graph_def_optimized = tf.GraphDef()
            graph_def_optimized.ParseFromString(f.read())
        
        G = tf.Graph()
        
        with tf.Session(graph=G) as sess:
            tf.import_graph_def(graph_def_optimized, name='')
            predictions = G.get_tensor_by_name('predictions:0')
            inputs = G.get_tensor_by_name('inputs:0')
#             print('Operations in Optimized Graph:')
#             print([op.name for op in G.get_operations()])
            img_path = './data/sim_images/GREEN/0000118_149_784.jpg'
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            res = sess.run(predictions,feed_dict={inputs: img})
            print(res)
#             x = G.get_tensor_by_name('import/x:0')
#             tf.global_variables_initializer().run()
#             out = sess.run(y, feed_dict={x: 1.0})
#             print(out)
        return
    def use_raw_model(self):
        return
    
    
    def run(self):
        self.convert() 
#         self.use_optimized_model()
#         self.use_raw_model()
                
        return
    
"""
Freeze and optimize for inference
python -m tensorflow.python.tools.freeze_graph --input_graph graph.pb --input_checkpoint ./tlmodel --output_graph graph_frozen.pb --output_node_names=predictions

python -m tensorflow.python.tools.optimize_for_inference --input graph_frozen.pb --output graph_optimized.pb --input_names=inputs --output_names=predictions --placeholder_type_enum=4
"""


if __name__ == "__main__":   
    obj= ConvertModel()
    obj.run()