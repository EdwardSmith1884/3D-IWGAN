import tensorflow as tf 
import os
import sys 
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import random 
from glob import glob


def Deconv(inputs, f_dim_in, dim, net, batch_size, f_dim_out = None, stride = 2 ):
	if f_dim_out is None: 
		f_dim_out = f_dim_in/2 
	return tl.layers.DeConv3dLayer(inputs,
								shape = [4, 4, 4, f_dim_out, f_dim_in],
								output_shape = [batch_size, dim, dim, dim, f_dim_out],
								strides=[1, stride, stride, stride, 1],
								W_init = tf.random_normal_initializer(stddev=0.02),
								act=tf.identity, name='g/net_' + net + '/deconv')

def Conv3D(inputs, f_dim_out, net, f_dim_in = None, batch_norm = False, is_train = True):
	if f_dim_in is None: 
		f_dim_in = f_dim_out/2
	layer = tl.layers.Conv3dLayer(inputs, 
								   shape=[4, 4, 4, f_dim_in, f_dim_out],
								   W_init = tf.random_normal_initializer(stddev=0.02),
								   strides=[1, 2, 2, 2, 1], name= 'd/net_' + net + '/conv')
	if batch_norm: 
		return tl.layers.BatchNormLayer(layer, is_train=is_train, name='d/net_' + net + '/batch_norm')
	else:
		return layer


def generator_20(inputs, is_train=True, reuse=False, batch_size = 128):
	output_size, half, forth = 20, 10, 5 
	gf_dim = 128 # Dimension of gen filters in first conv layer
	with tf.variable_scope("gen", reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)

		net_0 = tl.layers.InputLayer(inputs, name='g/net_0/in')

		net_1 = tl.layers.DenseLayer(net_0, n_units = gf_dim*forth*forth*forth, W_init = tf.random_normal_initializer(stddev=0.02), act = tf.identity, name='g/net_1/dense')
		net_1 = tl.layers.ReshapeLayer(net_1, shape = [-1, forth, forth, forth, gf_dim], name='g/net_1/reshape')
		net_1 = tl.layers.BatchNormLayer(net_1, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_1/batch_norm')
		net_1.outputs = tf.nn.relu(net_1.outputs, name='g/net_1/relu')
	 
		net_2 = Deconv(net_1, gf_dim, half,'2', batch_size)
		net_2 = tl.layers.BatchNormLayer(net_2, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_2/batch_norm')
		net_2.outputs = tf.nn.relu(net_2.outputs, name='g/net_2/relu')

		net_3 = Deconv(net_2, gf_dim/2, output_size, '3', batch_size)
		net_3 = tl.layers.BatchNormLayer(net_3, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_3/batch_norm')
		net_3.outputs = tf.nn.relu(net_3.outputs, name='g/net_3/relu')

		net_4 = Deconv(net_3,gf_dim/4, output_size, '4', batch_size, f_dim_out = 1, stride = 1) 
		net_4.outputs = tf.reshape(net_4.outputs,[batch_size,output_size,output_size,output_size], name='g/net_4/reshape')
		net_4.outputs = tf.nn.tanh(net_4.outputs, name='g/net_4/tanh')
		return net_4, net_4.outputs

def generator_32(inputs, is_train=True, reuse=False, batch_size = 128, sig = False):
	output_size, half, forth, eighth, sixteenth = 32, 16, 8, 4, 2
	gf_dim = 256 # Dimension of gen filters in first conv layer
	with tf.variable_scope("gen", reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)

		net_0 = tl.layers.InputLayer(inputs, name='g/net_0/in')

		net_1 = tl.layers.DenseLayer(net_0, n_units = gf_dim*sixteenth*sixteenth*sixteenth, W_init = tf.random_normal_initializer(stddev=0.02), act = tf.identity, name='g/net_1/dense')
		net_1 = tl.layers.ReshapeLayer(net_1, shape = [-1, sixteenth, sixteenth, sixteenth, gf_dim], name='g/net_1/reshape')
		net_1 = tl.layers.BatchNormLayer(net_1, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_1/batch_norm')
		net_1.outputs = tf.nn.relu(net_1.outputs, name='g/net_1/relu')

		net_2 = Deconv(net_1, gf_dim, eighth, '2', batch_size) 
		net_2 = tl.layers.BatchNormLayer(net_2, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_2/batch_norm')
		net_2.outputs = tf.nn.relu(net_2.outputs, name='g/net_2/relu')

		net_3 = Deconv(net_2, gf_dim/2, forth, '3', batch_size)
		net_3 = tl.layers.BatchNormLayer(net_3, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_3/batch_norm')
		net_3.outputs = tf.nn.relu(net_3.outputs, name='g/net_3/relu')
		
		net_4 = Deconv(net_3, gf_dim/4, half, '4', batch_size)
		net_4 = tl.layers.BatchNormLayer(net_4, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_4/batch_norm')
		net_4.outputs = tf.nn.relu(net_4.outputs, name='g/net_4/relu')
	   
		net_5 = Deconv(net_4, gf_dim/8, output_size, '5', batch_size, f_dim_out = 1)
		net_5.outputs = tf.reshape(net_5.outputs,[batch_size,output_size,output_size,output_size])
		if sig: 
			net_5.outputs = tf.nn.sigmoid(net_5.outputs)
		else: 
			net_5.outputs = tf.nn.tanh(net_5.outputs)
		
		return net_5, net_5.outputs

def discriminator(inputs ,output_size, improved = False, VAE_loss = False, sig = False, is_train=True, reuse=False, batch_size=128, output_units= 1):
	inputs = tf.reshape(inputs,[batch_size,output_size,output_size,output_size,1])
	df_dim = output_size # Dimension of discrim filters in first conv layer

	with tf.variable_scope("dis", reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)

		net_0 = tl.layers.InputLayer(inputs, name='d/net_0/in')

		net_1 = Conv3D(net_0, df_dim, '1', f_dim_in = 1 , batch_norm = False ) 
		net_1.outputs = tl.activation.leaky_relu(net_1.outputs, alpha=0.2, name='d/net_1/lrelu')
		
		net_2 = Conv3D(net_1, df_dim*2, '2', batch_norm = not improved, is_train = is_train,) 
		net_2.outputs = tl.activation.leaky_relu(net_2.outputs, alpha=0.2, name='d/net_2/lrelu')
		
		net_3 = Conv3D(net_2, df_dim*4, '3', batch_norm = not improved, is_train = is_train)  
		net_3.outputs = tl.activation.leaky_relu(net_3.outputs, alpha=0.2, name='d/net_3/lrelu')
		
		net_4 = Conv3D(net_3, df_dim*8, '4', batch_norm = not improved, is_train = is_train)   
		net_4.outputs = tl.activation.leaky_relu(net_4.outputs, alpha=0.2, name='d/net_4/lrelu')
		
		net_5 = FlattenLayer(net_4, name='d/net_5/flatten')
		net_5 = tl.layers.DenseLayer(net_5, n_units=output_units, act=tf.identity,
										W_init = tf.random_normal_initializer(stddev=0.02),
										name='d/net_5/dense')
		if sig: 
			return net_5, tf.nn.sigmoid(net_5.outputs)
		else: 
			return net_5, net_5.outputs 


 


def VAE(images, is_train = True):
	sizes = [64,128,256,512,400]
	with tf.variable_scope("vae") as vs:

		net_0 = tl.layers.InputLayer(images, name='v/net_0/in')

		net_1 = tl.layers.Conv2dLayer(net_0, shape=[11, 11, 3, sizes[0]],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 4, 4, 1], name='v/net_1/conv2d')
		net_1.outputs = tl.activation.leaky_relu(net_1.outputs, alpha=0.2, name='v/net_1/lrelu')

		net_2 = tl.layers.Conv2dLayer(net_1, shape=[5, 5, sizes[0], sizes[1]],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 4, 4, 1], name='v/net_2/conv2d')
		net_2 = tl.layers.BatchNormLayer(net_2, is_train=is_train, name='v/net_2/batch_norm')
		net_2.outputs = tl.activation.leaky_relu(net_2.outputs, alpha=0.2, name='v/net_2/lrelu')

		net_3 = tl.layers.Conv2dLayer(net_2, shape=[5, 5, sizes[1], sizes[2]],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 2, 2, 1], name='v/net_3/conv2d')
		net_3 = tl.layers.BatchNormLayer(net_3, is_train=is_train, name='v/net_3/batch_norm')
		net_3.outputs = tl.activation.leaky_relu(net_3.outputs, alpha=0.2, name='v/net_3/lrelu')

		net_4 = tl.layers.Conv2dLayer(net_3, shape=[5, 5, sizes[2], sizes[3]],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 2, 2, 1], name='v/net_4/conv2d')
		net_4 = tl.layers.BatchNormLayer(net_4, is_train=is_train, name='v/net_4/batch_norm')
		net_4.outputs = tl.activation.leaky_relu(net_4.outputs, alpha=0.2, name='v/net_4/lrelu')

		net_5 = tl.layers.Conv2dLayer(net_4, shape=[8, 8, sizes[3], sizes[4]],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 1, 1, 1], name='v/net_5/conv2d')
		net_5 = tl.layers.BatchNormLayer(net_5, is_train=is_train, name='v/net_5/batch_norm')
		net_5.outputs = tl.activation.leaky_relu(net_5.outputs, alpha=0.2, name='v/net_5/lrelu')
		net_6 = FlattenLayer(net_5, name='v/net_6/flatten')
		means = tl.layers.DenseLayer(net_6, n_units= 200, act=tf.identity,
											W_init = tf.random_normal_initializer(stddev=0.02),
											name='v/means')
		sigmas = tl.layers.DenseLayer(net_6, n_units= 200, act=tf.tanh,
											W_init = tf.random_normal_initializer(stddev=0.02),
											name='v/sigmas')
		return means,sigmas,means.outputs,sigmas.outputs

	
def surface_VAE(inputs, is_train = True, batch_size= 128, output_size = 20):
	
	with tf.variable_scope("vae") as vs:

		inputs = tf.reshape(inputs,[batch_size,output_size,output_size,output_size,1])
		df_dim = output_size # Dimension of discrim filters in first conv layer. [64]
	 
		net_0 = tl.layers.InputLayer(inputs, name='v/net_0/in')

		net_1 = tl.layers.Conv3dLayer(net_0, shape=[4, 4, 4, 1, df_dim],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 2, 2, 2, 1], name='v/net_1/conv2d')
		net_1.outputs = tl.activation.leaky_relu(net_1.outputs, alpha=0.2, name='v/net_1/lrelu')
		
		net_2 = tl.layers.Conv3dLayer(net_1, shape=[4, 4, 4, df_dim, df_dim*2],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 2, 2, 2, 1], name='v/net_2/conv2d')
		net_2 = tl.layers.BatchNormLayer(net_2, is_train=is_train, name='v/net_2/batch_norm')
		net_2.outputs = tl.activation.leaky_relu(net_2.outputs, alpha=0.2, name='v/net_2/lrelu')
		
		net_3 = tl.layers.Conv3dLayer(net_2, shape=[4, 4, 4, df_dim*2, df_dim*4],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 2, 2, 2, 1], name='v/net_3/conv2d')
		net_3 = tl.layers.BatchNormLayer(net_3, is_train=is_train, name='v/net_3/batch_norm')
		net_3.outputs = tl.activation.leaky_relu(net_3.outputs, alpha=0.2, name='v/net_3/lrelu')
		
		net_4 = tl.layers.Conv3dLayer(net_3, shape=[4, 4, 4, df_dim*4, df_dim*8],
									   W_init = tf.random_normal_initializer(stddev=0.02),
									   strides=[1, 2, 2, 2, 1], name='v/net_4/conv2d')
		net_4.outputs = tl.activation.leaky_relu(net_4.outputs, alpha=0.2, name='v/net_4/lrelu')
		net_4 = tl.layers.BatchNormLayer(net_4, is_train=is_train, name='v/net_4/batch_norm')
		
		net_5 = FlattenLayer(net_4, name='v/net_5/flatten')

		means = tl.layers.DenseLayer(net_5, n_units= 200, act=tf.identity,
											W_init = tf.random_normal_initializer(stddev=0.02),
											name='v/means/id')
		sigmas = tl.layers.DenseLayer(net_5, n_units= 200, act=tf.tanh,
											W_init = tf.random_normal_initializer(stddev=0.02),
											name='v/sigmas/id')
		return means,sigmas,means.outputs,sigmas.outputs
