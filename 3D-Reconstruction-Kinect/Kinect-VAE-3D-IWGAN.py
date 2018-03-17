import tensorflow as tf  
import os
import sys 
sys.path.insert(0, '../')
import tensorlayer as tl
import numpy as np
import random 
import argparse
from scripts.GANutils import *
from scripts.models import *


parser = argparse.ArgumentParser(description='3D-GAN implementation for 32*32*32 voxel output')
parser.add_argument('-n','--name', default='Test', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='data/train/chair', help ='The location for the object voxel models.' )
parser.add_argument('-sf','--surfaces', default='data/surfaces/train/chair', help ='The location for the Kinext depth surfaces.' )
parser.add_argument('-e','--epochs', default=1500, help ='The number of epochs to run for.', type=int)
parser.add_argument('-b','--batchsize', default=30, help ='The batch size.', type=int)
parser.add_argument('-sample', default= 5, help='How often generated obejcts are sampled and saved.', type= int)
parser.add_argument('-save', default= 5, help='How often the network models are saved.', type= int)
parser.add_argument('-graph', default= 5, help='How often the discriminator loss and the reconstruction loss graphs are saved.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= '', help='The epoch to number to be loaded from.', type=str)
parser.add_argument('-vsf','--valid_surfaces', default='data/surfaces/valid/chair', help ='The location for the validation set of the Kinext depth surfaces.' )

args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
save_dir =  "savepoint/" + args.name +'/'
output_size = 32 

 



######### make directories ############################
make_directories(checkpoint_dir,save_dir)

############ Model Generatrion ##########

####### inputs  ###################
surfaces      = tf.placeholder(tf.float32, [args.batchsize, output_size, output_size, output_size], name='surfaces')
real_models = tf.placeholder(tf.float32, [args.batchsize, output_size, output_size, output_size] , name='real_models')
z           = tf.random_normal((args.batchsize, 200), 0, 1)
eps         = tf.random_normal((args.batchsize, 200), 0, 1)
########## network computations #######################

net_m, net_s, means, sigmas = surface_VAE(surfaces, batch_size=args.batchsize, output_size = output_size) # means in the input vector, variance is used for error 
z_x = tf.add(means,  tf.multiply(sigmas, eps))

net_g, G_dec        = generator_32(z_x, is_train=True, reuse = False, batch_size = args.batchsize)
net_d  , D_dec_fake  = discriminator(G_dec, output_size,improved = True ,is_train = True, reuse= False, batch_size = args.batchsize)

net_g2, G_train     = generator_32(z, is_train = True, reuse=True, batch_size = args.batchsize)
net_d2 , D_fake      = discriminator(G_train, output_size, improved = True, is_train = True, reuse = True, batch_size = args.batchsize)
net_d2 , D_legit     = discriminator(real_models,  output_size, improved = True, is_train= True, reuse = True, batch_size = args.batchsize)

########## Gradient penalty calculations ##############
alpha               = tf.random_uniform(shape=[args.batchsize,1] ,minval =0., maxval=1.)
difference          = G_train - real_models
inter               = []
for i in range(args.batchsize): 
    inter.append(difference[i] *alpha[i])
inter = tf.unstack(inter)
interpolates        = real_models + inter
gradients           = tf.gradients(discriminator(interpolates, output_size, improved = True, is_train = False, batch_size = args.batchsize, reuse= True)[1],[interpolates])[0]
slopes              = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
gradient_penalty    = tf.reduce_mean((slopes-1.)**2.)


########### Loss calculations #########################

kl_loss             = tf.reduce_mean(-sigmas +.5*(-1.+tf.exp(2.*sigmas)+tf.square(means)))  
recon_loss          = tf.reduce_mean(tf.square(real_models-G_dec))/2.
d_loss              = -tf.reduce_mean(D_legit) + tf.reduce_mean(D_fake) + 10.*gradient_penalty
g_loss              = -tf.reduce_mean(D_fake)+(5)*recon_loss
v_loss              = kl_loss + recon_loss 

############ Optimization #############
v_vars = tl.layers.get_variables_with_name('vae', True, True)
g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)

net_g.print_params(False)
net_d.print_params(False)
net_m.print_params(False)
net_s.print_params(False)

d_optim = tf.train.AdamOptimizer( learning_rate = 1e-4, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer( learning_rate = 1e-4, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)
v_optim = tf.train.AdamOptimizer( learning_rate = 1e-4, beta1=0.5, beta2=0.9).minimize(v_loss, var_list=v_vars)



####### Training ################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
sess.run(tf.initialize_all_variables())

# load checkpoints
if args.load: 
    load_networks(checkpoint_dir, sess, net_g, net_d, net_m= net_m, net_s= net_s, epoch = args.load_epoch)
    #these keep track of the discriminaotrs loss, the reconstruction loss, and the reconstruction loss on a validatino set s
    if len(args.load_epoch)>1: 
        track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, iter_counter  = [],[],[],[],[],[],0
    else: 
        track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, iter_counter = load_values(save_dir, recon = True ,valid = True )
else:     
    track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss, iter_counter  = [],[],[],[],[],[],0
  

 
files = grab_files_surfaces(args.surfaces, args.data)
valid = grab_files_surfaces(args.valid_surfaces, args.data)[:args.batchsize]
random.shuffle(valid)
valid_models, valid_surfaces, _ = make_inputs_and_surfaces(valid, args.data)

if len(args.load_epoch)>1: 
    start = int(args.load_epoch)
else: 
    start = 0 
for epoch in range(start, args.epochs):
    random.shuffle(files)
    for idx in xrange(0, len(files)/args.batchsize):
        file_batch = files[idx*args.batchsize:(idx+1)*args.batchsize]
        models, batch_surfaces, start_time = make_inputs_and_surfaces(file_batch, args.data)

        #training the discriminator and the VAE's encoder 
        errD,_,errV,_,r_loss = sess.run([d_loss, d_optim, v_loss, v_optim, recon_loss] ,feed_dict={surfaces: batch_surfaces, real_models: models}) 
        track_d_loss.append(-errD)
        track_d_loss_iter.append(iter_counter)
    
        
        #training the gen / decoder and the encoder 
        if iter_counter% 5 ==0:
            errG,_,r_loss= sess.run([g_loss, g_optim, recon_loss], feed_dict={surfaces: batch_surfaces, real_models:models })    
        track_recon_loss.append(r_loss)
        track_recon_loss_iter.append(iter_counter)
       
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f, v_loss: %.4f, r_loss: %.4f" % (epoch, args.epochs, idx, len(files)/args.batchsize, time.time() - start_time, errD, errG, errV, r_loss))           
        iter_counter += 1
        sys.stdout.flush()

    #saving the model 
    if np.mod(epoch, args.save) == 0:
        save_networks(checkpoint_dir,sess, net_g, net_d, epoch, net_m,net_s)
    #saving generated objects
    if np.mod(epoch, args.sample) == 0:
        models,recon_models = sess.run([net_g2.outputs,net_g.outputs], feed_dict={surfaces:batch_surfaces})       
        save_voxels(save_dir, models, epoch, recon_models )
    #saving learning info 
    if np.mod(epoch, args.graph) == 0: 
        r_loss = sess.run([recon_loss], feed_dict={surfaces:valid_surfaces, real_models: valid_models})
        track_valid_loss.append(r_loss[0])
        track_valid_loss_iter.append(iter_counter)
        render_graphs(save_dir, epoch, track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss)
        save_values(save_dir,track_d_loss_iter, track_d_loss, track_recon_loss_iter, track_recon_loss, track_valid_loss_iter, track_valid_loss)


    
