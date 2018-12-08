import numpy as np
import sys 
import os 
from subprocess import call


# this is mostly from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py 
# though I sped up the voxel2mesh function considerably, now only surface voxels are saved
# this is only really important for very large models 


def voxel2mesh(voxels, threshold=.3):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > threshold) # recieves position of all voxels
    offpositions = np.where(voxels < threshold) # recieves position of all voxels
    voxels[positions] = 1 # sets all voxels values to 1 
    voxels[offpositions] = 0 
    for i,j,k in zip(*positions):
        if np.sum(voxels[i-1:i+2,j-1:j+2,k-1:k+2])< 27 : #identifies if current voxels has an exposed face 
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)   
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, threshold=.3):
    verts, faces = voxel2mesh(pred, threshold )
    write_obj(filename, verts, faces)


if len(sys.argv) < 2:
    print('you need to specify what set of voxels to use')
models = np.load(sys.argv[1])
print models.shape
for i,m in enumerate(models):
	voxel2obj('current.obj', m)
	print '-----------'
	print 'displaying'
	print '-----------'
	call(['meshlab', 'current.obj'],  stdout=open(os.devnull, 'wb'))

