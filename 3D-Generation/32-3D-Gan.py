import tensorflow as tf 
import os
import sys 
sys.path.insert(0, '../')
import tensorlayer as tl
import numpy as np
import random 
import argparse
import scripts
from scripts.GANutils import *
from scripts.models import *




parser = argparse.ArgumentParser(description='3D-GAN implementation for 32*32*32 voxel output')
parser.add_argument('-n','--name', default='Test', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='data/train/chair', help ='The location fo the object voxel models.' )
parser.add_argument('-e','--epochs', default=1500, help ='The number of epochs to run for.', type=int)
parser.add_argument('-b','--batchsize', default=256, help ='The batch size.', type=int)
parser.add_argument('-sample', default= 5, help='How often generated obejcts are sampled and saved.', type= int)
parser.add_argument('-save', default= 5, help='How often the network models are saved.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= '', help='The epoch to number to be loaded from.', type=str)
parser.add_argument('-glr','--genorator_learning_rate', default=0.0025, help ='The genorator learning rate.', type=int)
parser.add_argument('-dlr','--discriminator_learning_rate', default=0.00005, help ='The discriminator learning rate.', type=int)

args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
save_dir =  "savepoint/" + args.name +'/'
output_size = 32 


######### make directories ############################

make_directories(checkpoint_dir,save_dir)

####### inputs  ###################
real_models = tf.placeholder(tf.float32, [args.batchsize, output_size, output_size, output_size] , name='real_models')
z           = tf.random_normal((args.batchsize, 200), 0, 1)
a = tf.Print(z, [z], message="This is a: ")
########## network computations #######################

net_g , G_train     = generator_32(z, is_train=True, reuse = False, sig= True, batch_size=args.batchsize)


else: 
    dis = discriminator
net_d , D_fake      = dis(G_train, output_size, batch_size= args.batchsize, sig = True, is_train = True, reuse = False)
net_d2, D_legit     = dis(real_models,  output_size, batch_size= args.batchsize, sig = True, is_train= True, reuse = True)
net_d2, D_eval      = dis(real_models,  output_size, batch_size= args.batchsize, sig = True, is_train= False, reuse = True) # this is for desciding weather to train the discriminator


########### Loss calculations #########################
d_loss = -tf.reduce_mean(tf.log(D_legit) + tf.log(1. - D_fake))
g_loss = -tf.reduce_mean(tf.log(D_fake))


############ Optimization #############
g_vars = net_g.all_params   
d_vars = net_d.all_params  



g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)

d_optim = tf.train.AdamOptimizer(args.discriminator_learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(args.genorator_learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)


####### Training ################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
sess.run(tf.global_variables_initializer())

if args.load: 
    load_networks(checkpoint_dir, sess, net_g, net_d, epoch = args.load_epoch)
 
files,iter_counter = grab_files(args.data) 
Train_Dis = True 
if len(args.load_epoch)>1: 
    start = int(args.load_epoch)
else: 
    start = 0 
for epoch in range(start, args.epochs):
    random.shuffle(files)
    for idx in xrange(0, len(files)/args.batchsize):
        file_batch = files[idx*args.batchsize:(idx+1)*args.batchsize]
        models, start_time = make_inputs(file_batch)
        #training the discriminator and the VAE's encoder 
        if Train_Dis: 
            errD,_,ones = sess.run([d_loss, d_optim, D_legit] ,feed_dict={real_models: models}) 
        else: 
            ones = sess.run([D_eval] ,feed_dict={real_models: models}) 
        errG,_,zeros,objects = sess.run([g_loss, g_optim, D_fake, G_train], feed_dict={})    
        Train_Dis = (cal_acc(zeros,ones)<0.95)# only train discriminator at certain level of accuracy 
       
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, args.epochs, idx, len(files)/args.batchsize , time.time() - start_time, errD, errG))
    #saving the model 
    if np.mod(epoch, args.save) == 0:
        save_networks(checkpoint_dir,sess, net_g, net_d, epoch)
    #saving generated objects
    if np.mod(epoch, args.sample ) == 0:     
        save_voxels(save_dir,objects, epoch )
    

    
