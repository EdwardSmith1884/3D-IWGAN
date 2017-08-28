# 3D Generation  
Two models are available: 
* The first is my implementation of the 3D-GAN paper, found here: http://3dgan.csail.mit.edu/
* The second is my own model, called 3D-IWGAN, released for my paper: https://arxiv.org/abs/1707.09557 

![Chairs](../imgs/IWGAN_chairs2.png?raw=true "Title")
Example generation results using the 3D-IWGAN model, from a distribution consisting of the chair class from ModelNet10 in 12 orientations.

**Dataset**: <br />
The dataset you can use to train is the modelNet10 dataset which can be downloaded and converted using the Make_Data.sh script. This will not take long.

**Training**: <br />
* You can train by calling python 32-3D-IWGan.py
* The parameters are [-h] [-n NAME] [-d DATA] [-e EPOCHS] [-b BATCHSIZE][-sample SAMPLE] [-save SAVE] [-l] [-le LOAD_EPOCH] [-graph GRAPH] * There is a description for all of these parameters by applying the '-h' parameter
* Without parameters the model will start training on the chair class with 12 orientations. 

**Additional Details**: <br />
* To evaluate the output you can use the visualize.py script.
  * To use this just call python visualize.py VOXEL_FILE, where voxel_file is the file created during training and saved to         savepoint/NAME/EPOCH.npy.
* For the 3D-IWGAN model, graphs will also be created which allow you to track the discriminator's loss
  * The loss should at first raise rapidly but then begin to decrease
  * This decreasing loss can be used to track convergence
  * To train on all of the training classes call 32-3D-IWGan.py -d 'data/train/*'. 
* Let me know if you have any issues at edward.smith@mail.mcgill.ca, I am happy to help. 


![graph](../imgs/graph.png?raw=true "Title") <br />
An example graph of the discriminator's loss during training. This can be used to track convergence, as can be observed in this image. 

