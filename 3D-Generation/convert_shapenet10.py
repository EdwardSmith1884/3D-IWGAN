import numpy as np
import sys
import os
import scipy.io
from path import Path
from tqdm import tqdm

if sys.argv[-1] == '-v': # this will allow you to visualize the models as they are made, more of a sanity check 
    import mayavi.mlab
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from mpl_toolkits.mplot3d import Axes3D


instances = {}
class_id_to_name = {
    "1": "bathtub",
    "2": "bed",
    "3": "chair",
    "4": "desk",
    "5": "dresser",
    "6": "monitor",
    "7": "night_stand",
    "8": "sofa",
    "9": "table",
    "10": "toilet"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())


if not os.path.exists('data/train/'):
    os.makedirs('data/train/')
if not os.path.exists('data/test/'):
    os.makedirs('data/test/')
base_dir = Path(sys.argv[1]).expand()

for fname in tqdm(sorted(base_dir.walkfiles('*.mat'))):
    if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'): 
        continue
    elts = fname.splitall()
    info = Path(elts[-1]).stripext().split('_')
    if len(info)<3: continue  
    if info[0] == 'discriminative' or info[0] == 'generative' : continue 
    instance = info[1]
    rot = int(info[2])
    split = elts[-2]
    classname = elts[-4].strip()
    if classname in class_names:
        dest = 'data/'+split+'/' + classname + '/'
        if not os.path.exists(dest):
            os.makedirs(dest)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        matrix = np.zeros((32,)*3, dtype=np.uint8)
        matrix[1:-1,1:-1,1:-1] = arr
        if sys.argv[-1] == '-v':
            xx, yy, zz = np.where(matrix>= 0.3)
            mayavi.mlab.points3d(xx, yy, zz,
                                 
                                 color=(.1, 0, 1),
                                 scale_factor=1)

            mayavi.mlab.show()
        # saves the models by instance name, and then rotation 
        np.save(dest +  instance + '_' + str(rot) , matrix)
        


