import os
import sys 
import numpy as np
import random 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageOps
import time
import psutil
from sklearn import metrics
import math 
from scipy.spatial import distance



def visit_all(orig ,gx0, gy0, gz0, gx1, gy1, gz1, model ):
    if gx0 == gx1 and gy0 == gy1 and gz0 == gz1: 
        return True 
    if .99<model[gx0, gy0, gz0]<1.01 and  orig != (gx0,gy0,gz0): 
        return False 
    miner = 1000000
    move= [0,0,0]
    for i in [-1,0,1]: 
        for j in [-1,0,1]: 
            for k in [-1,0,1]:
                dist = distance.euclidean((gx0+i,gy0+j,gz0+k), (gx1,gy1,gz1))  
                if dist < miner:  
                    miner = dist
                    move = (gx0+i,gy0+j,gz0+k)
    return visit_all(orig, move[0],move[1],move[2],gx1, gy1, gz1, model) 


def visit(command):
    return visit_all(command[0],command[1],command[2],command[3],command[4],command[5],command[6],command[7])

def make_surface(B): 
    B = np.load(B)
    xp= 0.
    yp = 0.
    zp = 0.
    a,b,c = np.where(B>.3)
    t = zip(a,b,c)
    l = float(len(t))

    for k in a:
        xp+=float(k)/l
    for k in b: 
        yp+= float(k)/l
    for k in c:  
        zp+= float(k)/l
    B = ndimage.interpolation.shift(B,(9-int(xp),9-int(yp),9-int(zp)))
    surfaces = []
    for j in range(surface_num): 

        camera = [16+random.randint(10,15)*(-1)**random.randint(0,1),16+random.randint(10,15)*(-1)**random.randint(0,1), random.randint(16,25)]
        A = np.zeros([32,32,32])
        a,b,c = np.where(B>.3)
        t = zip(a,b,c)
        commands = []
        for a,b,c in t:     
            commands.append([(a,b,c),a,b,c, camera[0], camera[1], camera[2], B])
        pool = Pool()
        checks = pool.map(visit, commands)
        pool.close()
        pool.join()
        i = 0
        for a,b,c in t: 
            if checks[i]: 
                A[a,b,c] = 1 
            i+=1
        surfaces.append(A)
    return surfaces

def surface(obj_type): 
    files = glob('data/train/' + obj_type+ '/*')[:obj_num]
    dest = 'data/surfaces/'
    if not os.path.exists(dest + 'train/' + obj_type):
        os.makedirs(dest + 'train/' + obj_type)
    if not os.path.exists(dest + 'valid/' + obj_type):
        os.makedirs(dest + 'valid/' + obj_type)
    surfaces = [] 
    print 'making training set...'
    for f in tqdm(files[:int(len(files)*.9)]):
        surfaces = make_surface(f) 
        for i,sur in enumerate(surfaces): 
            np.save(dest + 'train/' + obj_type + '/' +  f.split('/')[-1][:-4] + '_' + str(i) , sur )
    print 'making validation set...'
    for f in tqdm(files[int(len(files)*.9):]):
        surfaces = make_surface(f) 
        for i,sur in enumerate(surfaces): 
            np.save(dest + 'valid/' + obj_type + '/' +  f.split('/')[-1][:-4] + '_' + str(i) , sur )    




surface_num = 15 # number of surfaces produced for each object 
obj_num = 800 # number of objects used to produce surfaces, you will want to increase this to train properly

surface(sys.argv[1])
print 'We are all done!!!!!'




