from triplegan import *
from utils import *
import tensorflow.compat.v1 as tf1


c_dir = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/Checkpoints/"
l_dir = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/Logs/"
r_dir = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/Results/"

def main():
    
    with tf1.Session(config = tf1.ConfigProto(allow_soft_placement=True)) as sess:

    	#same learning rate for generater and discriminator. 
        gan = TripleGAN(sess, epoch = 250, batch_size = 16, unlabel_batch_size = 8, latent_dim = 100, gan_lr = 2e-4, cla_lr = 2e-3,
                        checkpoint_dir = c_dir, result_dir = r_dir, log_dir = l_dir)
        
        
        gan.build_model()
        
        gan.train()
        
        print("[*] Training Finished..")
        
        gan.visualize_results(999)
        

if __name__ == "__main__":
    main()
    