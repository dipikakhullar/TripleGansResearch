from triplegan import *
from utils import *
import tensorflow.compat.v1 as tf1


import warnings
warnings.filterwarnings("ignore")


c_dir = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/Checkpoints/"
l_dir = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/Logs/"
r_dir = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/Results/"
d_path = "/Users/dipika/Desktop/TripleGansResearch/tgan-code/cifar10/all_train_int_0/"

# c_dir = "/content/drive/My Drive/TripleGansResearch/tgan-code/Checkpoints/"
# l_dir = "/content/drive/My Drive/TripleGansResearch/tgan-code/Logs/"
# r_dir = "/content/drive/My Drive/TripleGansResearch/tgan-code/Results/"
# d_path = "/content/drive/My Drive/TripleGansResearch/tgan-code/UTKFace/"


def main():
    
    with tf1.Session(config = tf1.ConfigProto(allow_soft_placement=True)) as sess:
        gan = TripleGAN(sess, epoch = 500, batch_size = 8, unlabel_batch_size = 8, latent_dim = 100, gan_lr = 2e-4, cla_lr = 2e-3,
                        checkpoint_dir = c_dir, result_dir = r_dir, log_dir = l_dir, path = d_path)
        
        
        gan.build_model()
        
        gan.train()
        
        print("[*] Training Finished..")
        
        gan.visualize_results(999)
        

if __name__ == "__main__":
    main()
    