# 3D Translation 
This is the directory for translating images to 3D models using my method described in: https://arxiv.org/abs/1707.09557.

![Recovered](../imgs/rgbtovoxel.png?raw=true "Title")
Example reconstruction results using the 3D-VAE-IWGAN model, from a distribution consisting of the chair class from the ShapeNet Core dataset. In the 1st and 4th column are the rgb input, in the 2nd and 5th are my reconstructions, and in the 3rd and 6th are the ground truth objects.

**Dataset** 
* To set up the dataset call python DataSetPrep.py  [-h] [-o OBJECTS] [-no NUM_OBJECTS] [-ni NUM_IMAGES][-b BACKGROUNDS] [-t TEXTURES]. 
* Only enter one object at a time, for example python DataSetPrep.py -o chair -no 2000 -ni 15, or it will take far too long
  * Even a single class will take a few hours
* You need blender and matlab installed for these to work
* You can set the code to run in debug mode to see if something is not working. 
  * Debug mode is set on by default as when you first run there's a good chance something won't be installed properly
  * To change this just manually set the debug_mode variable to 0. 
* You will also need the binvox executable which can be found at this executable can be found at http://www.patrickmin.com/binvox/. 
  * I have included the executable in this directory for convenience
  * You will need to change the permissions on this executable to allow it to run. 
* The script above will not set up the background images or the object textures however, these must be completed yourself, though there are many datasets available online for textures and background images. 
  * Place the background images into the sun directory and run the script convert_sun.py from the model directory to set the backgrounds up properly. I have supplied a small sample to see how this works. 
  * Place the textures into dtd directory. 
* The DataSetPrep.py script is set up by default to produce the chair dataset, with a small number of textures and backgrounds. 
* Please let me know if you are having any problems with this part. 
* This section may be useful for other projects as it provides a complete pipeline for producing a dataset with images of objects and their corresponding 3D models in CAD and voxel format. 

**Training**
* To create the GAN model call python 20-VAE-3D-IWGAN.py
  * It has parameters which can be discovered using the -h parameters. 
  * It will, by default, train on the chair dataset, which is set up by defualt in the DataSetPrep.py script so as long as everything has been installed properly, and this script is called first everything should train as is. 
* To evaluate the output you can use the visualize.py script. This will require you to download Meshlab. To use this just call python visualize.py VOXEL_FILE, where voxel file is the file created during training and saved to savepoint/NAME/EPOCH.npy. 
* Graphs will be created which allow you to track the discriminator's loss, which will at first raise rapidly but then begin to decrease, and this decreasing loss can be used to track convergence. Let me know if you have any issues at edward.smith@mail.mcgill.ca
