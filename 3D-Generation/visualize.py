import numpy as np
import sys 
import os 
import mayavi.mlab
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from scripts.Vis2 import * 
from scripts.util import * 

connect = 2
threshold = 0.1
index = 1 
ind = index -1 
filename = ' '
downsample_factor = 1
downsample_method = 'max'
uniform_size = 0.9
use_colormap = False


if len(sys.argv) <1:
	print('you need to specify what set of voxels to use')


def my_viz():

	models = np.load(sys.argv[1])
	print models.shape
	for i,m in enumerate(models):
		xx, yy, zz = np.where(m >= 0.3)

		mayavi.mlab.points3d(xx, yy, zz,
							 
							 color=(.1, 0, 1),
							 scale_factor=1)

		mayavi.mlab.show()
			

def old_viz(): 

	objects = np.load(sys.argv[1])
	if len(objects.shape)==3: 
		objects = [objects]
	for voxels in objects:
		print voxels.shape

		if connect > 0: 
			voxels_keep = (voxels >= threshold)
			voxels_keep = max_connected(voxels_keep, connect)
			voxels[np.logical_not(voxels_keep)] = 0
		if downsample_factor > 1:
			print "==> Performing downsample: factor: "+str(downsample_factor)+" method: "+downsample_method,
			voxels = downsample(voxels, downsample_factor, method=downsample_method)
		print "Done"
		visualization(voxels, threshold, title=str(ind+1), uniform_size=uniform_size, use_colormap=use_colormap)

old_viz()
