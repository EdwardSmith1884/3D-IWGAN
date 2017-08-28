# 3D Reconstruction
This is the directory for recovering 3D object shape from single perspective depth scans using my 3D-VAE-IWGAN method described in: https://arxiv.org/abs/1707.09557 .

**Dataset**:
* This system is set up to only run on recovering chairs.
* To set up the dataset call ./3D_generation.sh .
  * This will download the ShapeNet models needed, and produce the chair surfaces. 
  * Look into 3D_generation.sh to alter the object class used.
  * Look into Kinect_setup.py to change the details of how many objects and viewing directions are used. 

**Training**: <br />
To run the model on this data call Kinect-VAE-3D-IWGAN.py, there are other parameters which can be specified, call with -h to view them. 

**Additional Details**:
* To evaluate the output you can use the visualize.py script.
  * To use this just call python visualize.py VOXEL_FILE, where voxel file is the file created during training and saved to savepoint/NAME/EPOCH.npy. 
* Graphs will be created which allow you to track the discriminator's loss
  * The loss will at first raise rapidly but then begin to decrease..
  * This decreasing loss can be used to track convergence. 
* Let me know if you have any issues at edward.smith@mail.mcgill.ca, I am happy to help.
