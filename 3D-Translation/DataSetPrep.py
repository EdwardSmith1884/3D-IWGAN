import os 
import urllib
from multiprocessing import Pool
from progress.bar import Bar
import sys
sys.path.insert(0, '../')
import scripts.binvox_rw
from scripts.global_variables import *
import numpy as np 
from tqdm import tqdm
from glob import glob
from datetime import datetime
import random 
import shutil
import Image
import ImageOps
import scripts
import argparse

# this is the dataset for object translation, it will download the object files, convert then into numpy matricies, and overlay them onto pictures from the sun dataset 

parser = argparse.ArgumentParser(description='Dataset prep for image to 3D object translation, downloads and creates objects and image overlays.')
parser.add_argument('-o','--objects', default=['chair'], help='List of object classes to be used downloaded and converted.', nargs='+' )
parser.add_argument('-no','--num_objects', default=1000, help='number of objects to be converted', type = int)
parser.add_argument('-ni','--num_images', default=15, help='number of images to be created for each object', type = int)
parser.add_argument('-b','--backgrounds', default='sun/', help='location of the background images')
parser.add_argument('-t','--textures', default='dtd/', help='location of the textures to place onto the objects')
args = parser.parse_args()


#labels for the union of the core shapenet classes and the ikea dataset classes 
labels = {'03001627' : 'chair', '04256520': 'sofa', '04379243': 'table', '02858304':'boat', '02958343':'car',  '02691156': 'plane', '02808440': 'bathtub', '03085219': 'monitor', '02871439': 'bookcase', '1':'bed','2':'desk' }

# indicate here with set you want to use 
wanted_classes=[]
for l in labels: 
	if labels[l] in args.objects:
		wanted_classes.append(l)


debug_mode = 1 # change to make all of the called scripts print their errors and warnings 
if debug_mode:
	io_redirect = ''
else:
	io_redirect = ' > /dev/null 2>&1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'../'))

if not os.path.exists('data/voxels/'):
	os.makedirs('data/voxels/')
if not os.path.exists('data/objects/'):
	os.makedirs('data/objects/')


def download():
	with open('obj_locations/binvox_file_locations.txt','rb') as f: # location of all the binvoxes for shapenet's core classes 
		content = f.readlines()
	for s in wanted_classes: 
		obj = 'data/objects/' + labels[s]+'/'
		if not os.path.exists(obj):
			os.makedirs(obj)
		voxes = 'data/voxels/' + labels[s]+'/'
		if not os.path.exists(voxes):
			os.makedirs(voxes)

	binvox_urls = []
	obj_urls = []
	for file in content: 
		current_class = file.split('/')
		if current_class[1] in wanted_classes:  
			if '_' in current_class[3]: continue 
			if 'presolid' in current_class[3]: continue 
			#get urls for objects 
			obj_urls.append(['http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'+file.split('/')[1]+'/'+file.split('/')[2]+'/model.obj', 'data/objects/'+labels[current_class[1]]+ '/'+ current_class[2]+'.obj'])
	
	#sofas and desk are subset of the beds and tables classes and so I need special locations for these to differentiate them
	for s in wanted_classes: 
		if labels[s] == 'desk': 
			with open('obj_locations/desk_locations.txt','rb') as f: 
				content = f.readlines()
			for line in content: 
				current_class = line.split(',')
				obj_urls.append(['http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/04379243/'+current_class[0][4:]+'/model.obj', 'data/objects/desk/'+ current_class[0][4:]+'.obj'])
		if labels[s] == 'bed': 
			with open('obj_locations/bed_locations.txt','rb') as f: 
				content = f.readlines()
			for line in content: 
				current_class = line.split(',')
				obj_urls.append(['http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/02818832/'+current_class[0][4:]+'/model.obj', 'data/objects/bed/'+ current_class[0][4:]+'.obj'])
	random.shuffle(obj_urls)
	final_urls = []
	dictionary = {}
	for o in obj_urls:
		obj_class = o[1].split('/')[-2]
		if obj_class in dictionary: 
			dictionary[obj_class] += 1
			if dictionary[obj_class]> args.num_objects: 
				continue
		else: 
			dictionary[obj_class] = 1
		final_urls.append(o)  
	pool = Pool()
	pool.map(down, final_urls)

# these are two simple fucntions for parallel processing, down downloads in parallel, and call calls functions in parallel
# there work in conjuntion with pool.map()
def down(url):
	urllib.urlretrieve(url[0], url[1])
def call(command):
	os.system('%s %s' % (command, io_redirect))


def binvox(): # converts .obj files to .binvox files, intermidiate step before converting to voxel .npy files 
	for s in wanted_classes: 
		dirs = glob('data/objects/' + labels[s]+'/*.obj')
		commands =[]
		count = 0 
		for d in tqdm(dirs):
			command = './binvox -d 100 -pb ' + d # this executable can be found at http://www.patrickmin.com/binvox/ ,  -d 100 idicates resoltuion will be 100 by 100 by 100 , -pb is to stop the visualization
			commands.append(command)
			if count %200 == 0  and count != 0: #again parallelize to make quicker, be careful, while this runs your computer will not be useable!
				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
			count +=1 
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()


def convert(): # converts .binvox files to .npy voxels grids. A 5 times downsampling is also applied here. 
			   # I apply a downsampling instead of rendering the binvox file at 20 by 20 by 20 resolution as I found that 
			   # the binvoxer makes things to skinny and will often miss out sections of objects entirely if they are not large enough 
			   # to avoid this I render at  high resolution and then my downsampling 'encourages' all of the object to be seen at this resoltuion
	for directory in wanted_classes:
		directory = 'data/voxels/'+labels[directory]+'/' 
		if not os.path.exists(directory):
			os.makedirs(directory)
	for num in wanted_classes: 
		mods = glob('data/objects/'+labels[num]+'/*.binvox')
		for m  in tqdm(mods):  
			with open(m , 'rb') as f:
				try: 
					model = scripts.binvox_rw.read_as_3d_array(f)
				except ValueError:
					continue
			data = model.data.astype(int)
			down = 5 # how
			smaller_data = np.zeros([20,20,20])

			a,b,c = np.where(data==1)
			
			for [x,y,z] in zip(a,b,c): 
				count = 0
				br = False 
				u = x 
				uu=y
				uuu=z
				if x%2 ==1: u-=1
				if y%2 ==1: uu-=1
				if z%2 ==1: uuu-=1
				if smaller_data[u/down][uu/down][uuu/down]==1: continue 
				for i in range(down): 
					for j in range(down): 
						for k in range(down):
							try: 
								count += data[x+i][y+j][z+k] 
							except IndexError: 
								w = 0 
							if count >= 1: 
								if x%2 ==1: x-=1
								if y%2 ==1: y-=1
								if z%2 ==1: z-=1
								smaller_data[x/down][y/down][z/down]= 1 
								br = True 
								break 
						if br: break 
					if br: break 
			xx,yy,zz = np.where(smaller_data==1)
			if len(xx)<200: continue # this avoid objects whihc have been heavily distorted
			np.save( 'data/voxels/'+labels[num]+'/'+m.split('/')[-1][:-7], smaller_data) 				



def render(): # code for rendering the cad models as images 
	for s in wanted_classes: 
		img_dir = 'data/images/'+labels[s]+ '/' 
		if not os.path.exists(img_dir):
			os.makedirs(img_dir)
		Model_dir = 'data/objects/'+labels[s]+ '/'
		models = glob(Model_dir+'*.obj')
		textures = glob(args.textures + '/*') # loaction of all the texture you with to use, I hand pruned a texture dataset for this, though any set of metalic and wood based pictures will do 
		l=0
		for i in range(args.num_images):
			commands = []
			print str(datetime.now())
			print 'Image set number: ' +str(i)
			for model in tqdm(models): 
				model_name = model.split('/')[-1].split('.')[0]
				img_name = model_name + '_' + str(i) + '.png'
				if not os.path.exists(os.path.join(img_dir, model_name)):
					os.mkdir(os.path.join(img_dir, model_name))
				v = [ random.uniform(0.0,12.0) * 360./12., random.uniform(0.0,3.0)*15.0,  random.uniform(-5.,10.), random.uniform(2.0,5.0)] # specifies ranges for azimuth, elevation, light level, and size 
				# these setting work for me to get things large and varied enough
				# the size setting is low number for large objects and high numbers for small, be careful, it takes much longer to render larger object, I have it working so that 
				# I render smaller then desired and then I crop the image to enlarge it 
				python_cmd = 'python %s -a %s -e %s -t %s -d %s -o %s -m %s -tex %s ' % ('../scripts/render_class_view.py', 
					str(v[0]), str(v[1]), str(v[2]), str(v[3]), os.path.join(img_dir,model_name, img_name ), model, random.choice(textures))
				commands.append(python_cmd)

				# I do parallel processing here, very important unless you want this to take an order of magnitude longer!
				if l%50 == 0: 
					pool = Pool()
					pool.map(call, commands)
					pool.close()
					pool.join()
					commands = []
				l+=1
			pool = Pool()
			pool.map(call, commands)
			pool.close()
			pool.join()
			commands = []

def overlay(): # for overlaying the intermediate cad images onto backgrounds 
	for s in wanted_classes:
		overlay_dir = 'data/overlays/'+labels[s]+ '/'
		if not os.path.exists(overlay_dir):
			os.makedirs(overlay_dir)
		Model_dir = 'data/objects/'+labels[s]+ '/'
		background_list  = os.path.join(args.backgrounds, 'filelist.txt') # txt file with each line the location of each image you wish to use as background 
		
		# I do something weird here to get the matlab code which I didnt origionally write,though did alter, to work
		# The script usually does not convert all of the desired images, though as you can see below the script takes as input a directory not a file 
		# so the easiest fix I found for this was moving all the the files to a temp directory images.temp, running the script on this directory, 
		# and then moving back the images which have been coverted, this may seem silly, but the conversions take a while and this allows for stopping and starting 
		# as well it ensures the images are not harmed, which is good as they take hours to produce 
		overlays =glob('data/overlays/'+ labels[s]+ '/*')
		overlays = [f.split('/')[-1] for f in overlays]
		images = glob('data/images/' + labels[s]+ '/*')
		images = [f.split('/')[-1] for f in images]
		missing = [f for f in images  if f not in overlays] # all of the unconverted images 

		img_dir = 'data/images.temp/' + labels[s] + '/'
		if not os.path.exists(img_dir):
			os.makedirs(img_dir)
		for m in missing: 
			shutil.move('data/images/'+labels[s] + '/' + m, 'data/images.temp/'+ labels[s] + '/' + m) # move unconverted to temp directory 

		while len(missing) >0 :
			for m in missing: 
				i = 0 
			matlab_cmd = "addpath('%s'); overlay_background('%s','%s','%s', '%s', %f, 1);" % (os.path.join(g_render4cnn_root_folder, 'render_pipeline'), img_dir,  overlay_dir, background_list, backgrounds, 1.0)
			print ">> Starting MATLAB ... to run background overlaying command: \n \t %s" % matlab_cmd
			os.system('%s -nodisplay -r "try %s ; catch; end; quit;" %s' % (g_matlab_executable_path, matlab_cmd, io_redirect)) # run script 
			overlays = glob('data/overlays/'+ labels[s] + '/*')
			overlays = [f.split('/')[-1] for f in overlays]
			for m in missing: 
				if m in overlays: # repace converted images 
					shutil.move('data/images.temp/' + labels[s] + '/'+ m, 'data/images/'+ labels[s] + '/' + m)
					missing.remove(m)

def resize(): # for resizing images to 256 by 256  and zooming in on the objects 
	for s in wanted_classes: 
		files = glob('data/overlays/' + labels[s]+'/*') 
		for f in tqdm(files):
			images = glob(f+'/*')
			for im in images:
				# here I am maintianing the origional images and renaming them with a large_ prefix 
				actual = im
				new = f+'/orig_' + im.split('/')[-1]
				shutil.move(actual,new)
				im = Image.open(new)
				x,y = im.size
				diff = (x-y)/2
				im = im.crop((diff+100,100, x-diff-100, y-100)) # getting a square aspect ratio and zooming in on object by 100 pixels 
				im = ImageOps.fit(im, (256,256), Image.ANTIALIAS) # reshaping to 256 by 256 
				im.save(actual)


backgrounds = 'sun/' # location of desired background images, I just used the sun dataset 

print '------------'
print'downloading'
download()
print'------------'
print'rendering'
print'------------'
render()
print'------------'
print'overlaying'
print'------------'
overlay()
print'------------'
print'resizing'
print'------------'
resize()
print'------------'
print'converting binvoxes from objects'
print'------------'
binvox()
print'------------'
print 'converting binvoxes to voxels'
print'------------'
convert()
print'------------'
print 'we are all done, Yay'


