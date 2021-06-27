import tensorflow as tf
import keras
import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior() 

import numpy as np
#import tqdm as tqdm
#import pickle
import os
from utils import *
import time
from PIL import Image
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, Dropout, Flatten, GaussianNoise, Softmax
from tensorflow.keras.layers import LeakyReLU, concatenate, Reshape, Conv2DTranspose, AveragePooling2D, MaxPooling2D


tf.compat.v1.disable_eager_execution()

from utk_data_loader import *

x_train, y_train, x_test, y_test = load_utk_data('./UTKFace')
print("UTK X TRAIN",np.shape(x_train))
print("UTK X TEST", np.shape(x_test))
print("UTK Y TRAIN", np.shape(y_train))
print("UTK Y TEST", np.shape(y_test))

x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)

x_train/=255
x_test/=255
# print(x_train)
# print(y_train)

print("*** DATA LOADING COMPLETE ***")

labelled_x, labelled_y, unlabelled_x, unlabelled_y, test_x, test_y = create_data_subsets(0.5, 
    x_train = x_train, y_train = y_train,
    x_test = x_test, y_test = y_test)
# labelled_x, labelled_y, unlabelled_x, unlabelled_y, test_x, test_y = create_data_subsets(0.5)

def generate_one_batch(data, batchsize,
                       labelled_x = labelled_x, labelled_y = labelled_y,
                       unlabelled_x = unlabelled_x, unlabelled_y = unlabelled_y,
                       test_x = test_x, test_y = test_y):
    
    if data == "labelled":
        dataset_x = labelled_x
        dataset_y = labelled_y
    if data == "unlabelled":
        dataset_x = unlabelled_x
        dataset_y = unlabelled_y
    if data == "test":
        dataset_x = test_x
        dataset_y = test_y
    
    sample_ids = random.sample(range(len(dataset_x)), batchsize)
    X_id = dataset_x[sample_ids]
    Y_id = dataset_y[sample_ids]
    
    return X_id, Y_id


class TripleGAN(object):
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, latent_dim, gan_lr, cla_lr, checkpoint_dir, result_dir, log_dir):
        
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.unlabelled_batch_size = unlabel_batch_size
        self.test_batch_size = 32
        self.model_name = "Triple GAN"
        self.input_height = 224
        self.input_width = 224
        self.output_height = 224
        self.output_width = 224
        self.latent_dim = latent_dim
        self.num_classes = 10
        self.c_dim = 3
        #self.path = path
        
        self.learning_rate = gan_lr
        self.cla_learning_rate = cla_lr
        self.GAN_beta1 = 0.5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # alpha, epsilon, beta.
        self.alpha = 0.5

        # extra term in the classifier loss. Rp in page 5 of paper. 
        self.alpha_cla_adv = 0.01
        self.init_alpha_p = 0.0
        self.apply_alpha_p = 0.1
        self.apply_epoch = 200
        self.decay_epoch = 50
        
        self.sample_num = 64
        self.visual_num = 100
        self.len_discrete_code = 10
        self.num_batches = 25000//self.batch_size
        
    def discriminator(self, x, y_, is_train):
        x = GaussianNoise(0.15, name = 'dis0')(x)
        x = Dropout(0.2, name = 'dis1')(x, training = is_train)
        y = Reshape([1, 1, self.num_classes], name = 'dis2')(y_)
        x = conv_concat(x, y)

        x = Conv2D(filters = 32, kernel_size = 3, name = 'dis3')(x)
        x = LeakyReLU(0.2, name = 'dis4')(x)
        x = conv_concat(x, y)
    
        x = Conv2D(filters = 32, kernel_size = 3, name = 'dis5')(x)
        x = LeakyReLU(0.2, name = 'dis6')(x)
        x = Dropout(0.2, name = 'dis7')(x, training = is_train)
        x = conv_concat(x, y)
            
        x = Conv2D(filters = 64, kernel_size = 3, name = 'dis8')(x)
        x = LeakyReLU(0.2, name = 'dis9')(x)
        x = conv_concat(x, y)
        
        x = Conv2D(filters = 64, kernel_size = 3, strides = 2, name = 'dis10')(x)
        x = LeakyReLU(0.2, name = 'dis11')(x)
        x = Dropout(0.2, name = 'dis12')(x, training = is_train)
        x = conv_concat(x, y)
        
        x = Conv2D(filters = 128, kernel_size = 3, name = 'dis13')(x)
        x = LeakyReLU(0.2, name = 'dis14')(x)
        x = conv_concat(x, y) 
        
        x = Conv2D(filters = 128, kernel_size = 3, name = 'dis15')(x)
        x = LeakyReLU(0.2, name = 'dis16')(x)
        x = conv_concat(x, y)  
        
        x = AveragePooling2D(name = 'dis17')(x)
        x = Flatten(name = 'dis18')(x)
        x = concatenate([x, y_], name = 'dis19')
        out_logit = Dense(1, name = 'dis20')(x)
        out = Softmax(1, name = 'dis21')(out_logit)
        
        return out, out_logit 
                
    def classifier(self, x, is_train):
        x = GaussianNoise(0.15, name = 'cla1')(x)
    
        x = Conv2D(filters = 128, kernel_size = 3, name = 'cla2')(x)
        x = LeakyReLU(0.2, name = 'cla3')(x)
            
        x = Conv2D(filters = 128, kernel_size = 3, name = 'cla4')(x)
        x = LeakyReLU(0.2, name = 'cla5')(x)
            
        x = Conv2D(filters = 128, kernel_size = 3, name = 'cla6')(x)
        x = LeakyReLU(0.2, name = 'cla7')(x)
        
        x = MaxPooling2D(pool_size = 2, strides = 2, name = 'cla8')(x)
        x = Dropout(0.3, name = 'cla9')(x, training = is_train)
        
        x = Conv2D(filters = 256, kernel_size = 3, name = 'cla10')(x)
        x = LeakyReLU(0.2, name = 'cla11')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, name = 'cla12')(x)
        x = LeakyReLU(0.2, name = 'cla13')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, name = 'cla14')(x)
        x = LeakyReLU(0.2, name = 'cla15')(x)
        
        x = MaxPooling2D(pool_size = 2, strides = 2, name = 'cla16')(x)
        x = Dropout(0.3, name = 'cla17')(x, training = is_train)
        
        x = AveragePooling2D(name = 'cla18')(x)
        x = Flatten(name = 'cla19')(x)
        
        x = Dense(512, activation = 'relu', name = 'cla20')(x)
        x = Dense(128, activation = 'relu', name = 'cla21')(x)
        
        label = Dense(10, name = 'cla22')(x)
        
        return label
    def generator(self, y_, z, is_train):
        x = concatenate([y_,z], name = 'gen1')
        x = Dense(512*4*4, activation = 'relu', name = 'gen2')(x)
        x = BatchNormalization(name = 'gen3')(x, training = is_train)
        
        x = Reshape([4, 4, 512], name = 'gen4')(x)
        #print(y_)
        y = Reshape([1, 1, self.num_classes], name = 'gen5')(y_)
        x = conv_concat(x,y)
        
        x = Conv2DTranspose(kernel_size = 4, filters = 512, strides = 2, activation = 'relu', name = 'gen6')(x)
        x = BatchNormalization(name = 'gen7')(x, training = is_train)
        x = conv_concat(x, y)
        
        x = Conv2DTranspose(kernel_size = 5, filters = 128, strides =2, activation = 'relu', name = 'gen8')(x)
        x = BatchNormalization(name = 'gen9')(x, training = is_train)
        x = conv_concat(x, y)
        
        x = Conv2DTranspose(kernel_size = 5, filters = 64, strides = 2, activation = 'relu', name = 'gen10')(x)
        x = BatchNormalization(name = 'gen11')(x, training = is_train)
        x = conv_concat(x, y)
        
        x = Conv2DTranspose(kernel_size = 3, filters = 32, strides = 2, activation = 'relu', name = 'gen12')(x)
        x = BatchNormalization(name = 'gen13')(x, training = is_train)
        x = conv_concat(x, y)
        
        img = Conv2DTranspose(kernel_size = 4, filters = 3, strides = 2, activation = 'tanh', name = 'gen14')(x)
        
        return img 
    
    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        unlabel_bs = self.unlabelled_batch_size
        test_bs = self.test_batch_size
        alpha = self.alpha
        alpha_cla_adv = self.alpha_cla_adv
        self.alpha_p = tf1.placeholder(tf.float32, name = 'alpha_p')
        self.gan_lr = tf1.placeholder(tf.float32, name='gan_lr')
        self.cla_lr = tf1.placeholder(tf.float32, name='cla_lr')
        self.unsup_weight = tf1.placeholder(tf.float32, name='unsup_weight')
        self.c_beta1 = tf1.placeholder(tf.float32, name='c_beta1')
        
        #images
        self.inputs = tf1.placeholder(tf.float32, [bs] + image_dims, name = 'real_images')
        self.unlabelled_inputs = tf1.placeholder(tf.float32, [unlabel_bs] + image_dims, name='unlabelled_images')
        self.test_inputs = tf1.placeholder(tf.float32, [test_bs] + image_dims, name='test_images')

        # labels
        self.y = tf1.placeholder(tf.float32, [bs, self.num_classes], name='y')
        self.unlabelled_inputs_y = tf1.placeholder(tf.float32, [unlabel_bs, self.num_classes])
        self.test_label = tf1.placeholder(tf.float32, [test_bs, self.num_classes], name='test_label')
        self.visual_y = tf1.placeholder(tf.float32, [self.visual_num, self.num_classes], name='visual_y')
       # print(self.visual_y)
        
        # noises
        self.z = tf1.placeholder(tf.float32, [bs, self.latent_dim], name='z')
        self.visual_z = tf1.placeholder(tf.float32, [self.visual_num, self.latent_dim], name='visual_z')
        
        '''Loss function'''
        
        ###Discriminator Output for real images
        D_real, D_real_logits = self.discriminator(self.inputs, self.y, is_train = True)
        ###Discriminator Output for generator images
        G = self.generator(self.y, self.z, is_train = True)
        D_fake, D_fake_logits = self.discriminator(G, self.y, is_train = True)

        ###Classifier Output for real images
        C_real_logits = self.classifier(self.inputs, is_train = True)
        #L_cla_real = tf.reduce_sum(categorical_crossentropy(y, C_real))
        L_cla_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=C_real_logits))
        
        
        ###Classifier Output for unlabelled images
        C_unlabelled = self.classifier(self.unlabelled_inputs, is_train = True)
        D_unlabelled, D_unlabelled_logits = self.discriminator(self.unlabelled_inputs, C_unlabelled, is_train = True)
        
        ###Classifier Output for generator images
        C_fake_logits = self.classifier(G, is_train = True)
        #L_cla_fake = tf.reduce_sum(categorical_crossentropy(y, C_fake))
        L_cla_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=C_fake_logits))
        
        
        ##Get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        d_loss_cla = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_unlabelled_logits, labels=tf.zeros_like(D_unlabelled)))
        self.d_loss = d_loss_real + (1-alpha)*d_loss_fake + alpha*d_loss_cla
    
        # get loss for generator
        #### Zeros or ones
        self.g_loss = (1-alpha)*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))  
        
        # test loss for classifier
        test_Y = self.classifier(self.test_inputs, is_train=False)
        correct_prediction = tf.equal(tf.argmax(test_Y, 1), tf.argmax(self.test_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        # get loss for classifier
        max_c = tf.cast(tf.argmax(C_unlabelled, axis=1), tf.float32)
        c_loss_dis = tf.reduce_mean(max_c * tf.nn.softmax_cross_entropy_with_logits(logits=D_unlabelled_logits, labels=tf.ones_like(D_unlabelled)))
        # self.c_loss = alpha * c_loss_dis + R_L + self.alpha_p*R_P
        
        # R_UL = self.unsup_weight * tf.reduce_mean(tf.squared_difference(Y_c, self.unlabelled_inputs_y))
        self.c_loss = alpha_cla_adv * alpha * c_loss_dis + L_cla_real + self.alpha_p*L_cla_fake
        
        
        '''Training'''
        #Divide training variables into a group for D and G
        t_vars = tf1.trainable_variables()
        
        d_vars = [var for var in t_vars if 'dis' in var.name]
        g_vars = [var for var in t_vars if 'gen' in var.name]
        c_vars = [var for var in t_vars if 'cla' in var.name]
        
        self.d_optim = tf1.train.AdamOptimizer(self.gan_lr, beta1=self.GAN_beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf1.train.AdamOptimizer(self.gan_lr, beta1=self.GAN_beta1).minimize(self.g_loss, var_list=g_vars)
        self.c_optim = tf1.train.AdamOptimizer(self.cla_lr, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.c_loss, var_list=c_vars)
        
        '''Testing'''
        self.fake_images = self.generator(self.visual_y, self.visual_z, is_train=False)
        
        """ Summary """
        d_loss_real_sum = tf1.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf1.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_cla_sum = tf1.summary.scalar("d_loss_cla", d_loss_cla)

        d_loss_sum = tf1.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf1.summary.scalar("g_loss", self.g_loss)
        c_loss_sum = tf1.summary.scalar("c_loss", self.c_loss)



        # final summary operations
        self.g_sum = tf1.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf1.summary.merge([d_loss_real_sum, d_loss_sum])
        self.c_sum = tf1.summary.merge([d_loss_cla_sum, c_loss_sum])
        
        
        
    def train(self):
        self.sess.run(tf1.global_variables_initializer())
        gan_lr = self.learning_rate
        cla_lr = self.cla_learning_rate
        
        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.visual_num, self.latent_dim))



        '''
        def generate_one_batch(data, batchsize,
                       labelled_x = labelled_x, labelled_y = labelled_y,
                       unlabelled_x = unlabelled_x, unlabelled_y = unlabelled_y,
                       test_x = test_x, test_y = test_y):
                       '''
        self.test_samples, self.test_codes = generate_one_batch('labelled', self.visual_num)
        
        #saver to save model
        self.saver = tf1.train.Saver()
        
        #summary writer
        self.writer = tf1.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)
        
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            with open('lr_logs.txt', 'r') as f :
                line = f.readlines()
                line = line[-1]
                gan_lr = float(line.split()[0])
                cla_lr = float(line.split()[1])
                print("gan_lr : ", gan_lr)
                print("cla_lr : ", cla_lr)
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
            
        # loop for epoch
        start_time = time.time()
        
        for epoch in range(start_epoch, self.epoch):

            if epoch >= self.decay_epoch :
                gan_lr *= 0.995
                cla_lr *= 0.99
                print("**** learning rate DECAY ****")
                print(gan_lr)
                print(cla_lr)

            if epoch >= self.apply_epoch :
                alpha_p = self.apply_alpha_p
            else :
                alpha_p = self.init_alpha_p

            rampup_value = rampup(epoch - 1)
            unsup_weight = rampup_value * 100.0 if epoch > 1 else 0
            
        
        
            for idx in range(start_batch_id, self.num_batches):
            # for idx in range(0, 10):
                batch_images, batch_codes = generate_one_batch( "labelled", self.batch_size)
                batch_unlabelled_images, batch_unlabelled_images_y = generate_one_batch("unlabelled", self.unlabelled_batch_size)
                batch_z = np.random.uniform(1, size = (self.batch_size, self.latent_dim))

                feed_dict = {
                        self.inputs: batch_images, self.y: batch_codes,
                        self.unlabelled_inputs: batch_unlabelled_images,
                        self.unlabelled_inputs_y: batch_unlabelled_images_y,
                        self.z: batch_z, self.alpha_p: alpha_p,
                        self.gan_lr: gan_lr, self.cla_lr: cla_lr,
                        self.unsup_weight : unsup_weight
                        }
                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str_g, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_g, counter)
                
                # update C network
                _, summary_str_c, c_loss = self.sess.run([self.c_optim, self.c_sum, self.c_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_c, counter)
                
                # display training status
                counter += 1
                if idx%500 ==0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, c_loss))
                
                # save training results for every 100 steps
                """
                if np.mod(counter, 100) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))
                    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                                './' + check_folder(
                                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                                epoch, idx))
                    """   
                    
            # classifier test
            test_acc = 0.0

            for idx in range(10) :
                test_batch_x, test_batch_y = generate_one_batch('test', self.test_batch_size)
                
                acc_ = self.sess.run(self.accuracy, feed_dict={
                        self.test_inputs: test_batch_x,
                        self.test_label: test_batch_y
                        })

                test_acc += acc_
            test_acc /= 10

            summary_test = tf1.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
            self.writer.add_summary(summary_test, epoch)

            line = "Epoch: [%2d], test_acc: %.4f\n" % (epoch, test_acc)
            print(line)
            lr = "{} {}".format(gan_lr, cla_lr)
            with open('logs.txt', 'a') as f:
                f.write(line)
            with open('lr_logs.txt', 'a') as f :
                f.write(lr+'\n')

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            self.save(self.checkpoint_dir, counter)
            
            # show temporal results
            self.visualize_results(epoch)
        
        # save model for final step
        self.save(self.checkpoint_dir, counter)
    
    def visualize_results(self, epoch):
        image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))
        z_sample = np.random.uniform(1, size = (self.visual_num, self.latent_dim))
        
        y = np.random.choice(self.len_discrete_code, self.visual_num)
        
        #Generate 10 labels with batch size
        y_one_hot = np.zeros((self.visual_num, self.num_classes))
        y_one_hot[np.arange(self.visual_num), y] = 1
        
        samples = self.sess.run(self.fake_images, feed_dict = {self.visual_z:z_sample, self.visual_y:y_one_hot})
        
        save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim],
                    check_folder(
                            self.result_dir + '/' + self.model_dir + '/all_classes') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
        
        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.visual_num, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.visual_num, dtype=np.int64) + l
            y_one_hot = np.zeros((self.visual_num, self.num_classes))
            y_one_hot[np.arange(self.visual_num), y] = 1

            samples = self.sess.run(self.fake_images, feed_dict={self.visual_z: z_sample, self.visual_y: y_one_hot})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        check_folder(
                            self.result_dir + '/' + self.model_dir + '/class_%d' % l) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)            
        
        
                """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.len_discrete_code],
                    check_folder(
                        self.result_dir + '/' + self.model_dir + '/all_classes_style_by_style') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.batch_size, self.latent_dim)
    
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step = step)
        
    def load(self, checkpoint_dir):
        import re
        print("[*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        
        ckpt = tf1.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        
        
        
        