import os
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import time

# making soem subdirectories, chekcpoint stores the models,
# savpoint saves some created objects at each epoch
def make_directories(checkpoint,savepoint): 
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    if not os.path.exists(savepoint):
        os.makedirs(savepoint)

# loads the images and objects into memory 
def make_inputs_and_images(file_batch, voxel_dir):
    voxel_dir+='/'
    models = []
    images = []
    for i,fil in enumerate(file_batch): 
        split = fil.split('/')
        models.append(np.load( voxel_dir+ split[-1].split('_')[0] +'.npy'))
        img = Image.open(fil)
        images.append(np.asarray(img,dtype='uint8'))
    models = np.array(models)
    images = np.array(images)
    start_time = time.time()
    return models, images, start_time

def make_inputs(file_batch):
    models = [np.load(f) for f in file_batch]
    models = np.array(models)
    start_time = time.time()
    return models, start_time

def make_inputs_and_surfaces(file_batch, voxel_dir):
    models = []
    surfaces = []
    for fil in file_batch: 
        name = '/' + fil.split('/')[-1].split('_')[-2] + '.npy'
        models.append(np.load( voxel_dir+ name ))
        surfaces.append(np.load(fil))
    models = np.array(models)
    surfaces = np.array(surfaces)
    start_time = time.time()
    return models, surfaces, start_time


def save_voxels(save_dir, models, epock, recon_models = None): 
    print "Saving the model"
    np.save(save_dir+str(epock)  , models) 
    if recon_models is not None: 
        np.save(save_dir+str(epock) + '_VAE', recon_models) 
      

        

def grab_files_images(image_dir, voxel_dir): 
    files = []
    pattern  = "*.jpg"
    image_dir+='/'
    voxel_dir+='/'
    for dir,_,_ in os.walk(image_dir):
        files.extend(glob(os.path.join(dir,pattern))) 
    voxels = [ v.split('/')[-1].split('.')[0] for v in glob(voxel_dir + '*')]
    
    temp = []
    for f in files: 
        if 'orig_' in f: 
            continue 
        if f.split('/')[-2] not in voxels: continue
        temp.append(f)
    files = []
    valid = [] 
    for i,t in enumerate(temp): 
        if len(valid) < 128 and i %33 == 0 :  
            valid.append(t)
        else:
            files.append(t)
    return files,valid

def grab_files(voxel_dir): 
    voxel_dir+='/'
    return [f for f in glob(voxel_dir + '*.npy')], 0 


def grab_files_surfaces(surface_dir, voxel_dir): 
    surface_dir+='/'
    voxel_dir+='/'
    files = glob(surface_dir+'*')
    voxels = [ v.split('/')[-1].split('_')[-1] for v in glob(voxel_dir + '*')]
    temp = []
    for f in files: 
        if f.split('/')[-1].split('_')[-2] + '.npy' not in voxels: continue
        temp.append(f)
    return temp



def save_networks(checkpoint_dir, sess, net_g, net_d, epoch, net_m = None, net_s=None):
    print("[*] Saving checkpoints...")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # this saves as the latest version location
    net_g_name = os.path.join(checkpoint_dir, 'net_g.npz')
    net_d_name = os.path.join(checkpoint_dir, 'net_d.npz')
    # this saves as a backlog of models
    net_g_iter_name = os.path.join(checkpoint_dir, 'net_g_%d.npz' % epoch)
    net_d_iter_name = os.path.join(checkpoint_dir, 'net_d_%d.npz' % epoch)
    tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
    tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
    tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
    tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)


    if net_m is not None and net_s is not None :
        net_m_name = os.path.join(checkpoint_dir, 'net_m.npz')
        net_s_name = os.path.join(checkpoint_dir, 'net_s.npz')
        net_m_iter_name = os.path.join(checkpoint_dir, 'net_m_%d.npz' % epoch)
        net_s_iter_name = os.path.join(checkpoint_dir, 'net_s_%d.npz' % epoch)
        tl.files.save_npz(net_m.all_params, name=net_m_name, sess=sess)
        tl.files.save_npz(net_s.all_params, name=net_s_name, sess=sess)
        tl.files.save_npz(net_m.all_params, name=net_m_iter_name, sess=sess)
        tl.files.save_npz(net_s.all_params, name=net_s_iter_name, sess=sess)
    print("[*] Saving checkpoints SUCCESS!")




def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def load_networks(checkpoint_dir, sess, net_g, net_d, net_m = None, net_s = None, epoch = ''): 
    print("[*] Loading checkpoints...")
    if len(epoch) >=1: epoch = '_' + epoch
    # load the latest checkpoints
    net_g_name = os.path.join(checkpoint_dir, 'net_g'+epoch+'.npz')
    net_d_name = os.path.join(checkpoint_dir, 'net_d'+epoch+'.npz')
    
    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Loading checkpoints failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        print("[*] Loading Generator and Discriminator checkpoints SUCCESS!")
        

    if net_m is not None and net_s is not None: 
        net_m_name = os.path.join(checkpoint_dir, 'net_m'+epoch+'.npz')
        net_s_name = os.path.join(checkpoint_dir, 'net_s'+epoch+'.npz')
        if not (os.path.exists(net_m_name) and os.path.exists(net_s_name)):
            print("[!] Loading VAE checkpoints failed!")
        else: 
            net_m_loaded_params = tl.files.load_npz(name=net_m_name)
            net_s_loaded_params = tl.files.load_npz(name=net_s_name)
            tl.files.assign_params(sess, net_m_loaded_params, net_m)
            tl.files.assign_params(sess, net_s_loaded_params, net_s)
            print("[*] Loading VAE checkpoints SUCCESS!")
def load_values(save_dir, recon = False, valid = False):
    outputs = []
    outputs.append(list(np.load(save_dir+'/plots/track_d_loss_iter.npy')))
    outputs.append(list(np.load(save_dir+'/plots/track_d_loss.npy')))
    if recon: 
        outputs.append( list(np.load(save_dir+'/plots/track_recon_loss_iter.npy')))
        outputs.append( list(np.load(save_dir+'/plots/track_recon_loss.npy')))
    if valid:
        outputs.append( list(np.load(save_dir+'/plots/track_valid_loss_iter.npy')))
        outputs.append( list(np.load(save_dir+'/plots/track_valid_loss.npy')))
    outputs.append(outputs[0][-1] )
    return outputs 
def render_graphs(save_dir,epoch, track_d_loss_iter, track_d_loss, track_recon_loss_iter = None, track_recon_loss=None, track_valid_loss_iter=None, track_valid_loss=None): 
    if not os.path.exists(save_dir+'/plots/'):
        os.makedirs(save_dir+'/plots/')
    if track_recon_loss is not None:
        if len(track_recon_loss)>51: 
            smoothed_recon = savitzky_golay(track_recon_loss, 51, 3)
            plt.plot(track_recon_loss_iter, track_recon_loss,color='blue') 
            plt.plot(track_recon_loss_iter,smoothed_recon , color = 'red')
            if track_valid_loss is not None:
                plt.plot(track_valid_loss_iter, track_valid_loss ,color='green')
            plt.savefig(save_dir+'/plots/recon_' + str(epoch)+'.png' )
            plt.clf()
    if len(track_d_loss)> 51: 
        smoothed_d_loss = savitzky_golay(track_d_loss, 51, 3)
        plt.plot(track_d_loss_iter, track_d_loss)
        plt.plot(track_d_loss_iter, smoothed_d_loss, color = 'red')
        plt.savefig(save_dir+'/plots/' + str(epoch)+'.png' )
        plt.clf()

def save_values(save_dir,track_d_loss_iter, track_d_loss, track_recon_loss_iter = None, track_recon_loss=None, track_valid_loss_iter=None, track_valid_loss=None):
    np.save(save_dir+'/plots/track_d_loss_iter', track_d_loss_iter)
    np.save(save_dir+'/plots/track_d_loss', track_d_loss)
    if track_recon_loss is not None:
        np.save(save_dir+'/plots/track_recon_loss_iter', track_recon_loss_iter)
        np.save(save_dir+'/plots/track_recon_loss', track_recon_loss)
    if track_valid_loss is not None: 
        np.save(save_dir+'/plots/track_valid_loss_iter', track_valid_loss_iter)
        np.save(save_dir+'/plots/track_valid_loss', track_valid_loss)


def cal_acc(zeros,ones): 
    accuracy = 0.0
    for example in zeros:

        if example[0]<0.5: accuracy += 1.0
        
    for example in ones:
        if example[0]>0.5: accuracy += 1.0 
    accuracy = accuracy/(float(len(zeros) + len(ones))) 
    print 'The accuracy of the discriminator is: ' + str(accuracy)
    return accuracy
