from nets.tl_model import TLModel
from nets.train_model_base import TrainModelBase



class TrainModel(TrainModelBase):
    def __init__(self):
        TrainModelBase.__init__(self)
        
        return
    def config_training(self):
        #customization
         
        self.max_number_of_steps = 1000
        self.learning_rate = 0.01
        self.learning_rate_decay_type = 'fixed'
        self.optimizer = 'adam'
        self.weight_decay = 0.0004 # for model regularization
        self.train_dir = './logs'
        self.checkpoint_path = './data/models/inception_v4.ckpt'
        self.checkpoint_exclude_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
        self.trainable_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
        
        
        fine_tune = True
        if fine_tune:
            #fine tune
            self.train_dir = './logs/finetune'
            self.checkpoint_path =  './logs'
            self.checkpoint_exclude_scopes = None
            self.trainable_scopes = None
            self.max_number_of_steps += 28000
            self.learning_rate = 0.0001
        
        
        
        return
    def setup_model(self):
        net = TLModel()
        #preapre input, label,
        net.input, _ , net.labels,_ = self.get_input("train", is_training=True, batch_size= self.batch_size)  
        net.build_train_graph()
        return net
    


if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()