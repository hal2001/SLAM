“””
README file for 
Project 4 SLAM and Texture Map
by Hantian Liu
“””


1. Run SLAM.py to do SLAM based on LIDAR and joint data. Videos showing occupancy grids, robot position (in red), and raw odometry (in blue) would be saved as 'slam.py'. 

Set the test boolean to be True, or False when using training datasets.
Set the rgbd_flag boolean to be True, to show the texture map video, and it would be saved as 'texture.py'.

Set the if_mulriple_data boolean to be True if there exists multiple datasets, i.e. there would be two '_' in RGB mat filenames.
Set the datanum to be the number of the LIDAR and joint datasets, 
	e.g. test_lidar0.mat has a 'datanum' of 0.

Set the if_mulriple_mat boolean to be True if there exists multiple mats of RGB images.
Set the rgbnum to be the TOTAL number of the RGB-D datasets, 
	e.g. RGB_3_0.mat to RGB_3_5.mat, has a 'datanum' of 3, and an 'rgbnum' of 5. 

Put the LIDAR and joint datasets into “./test_data” subfolder, or change the folder name for LIDAR and joint test files. 
Put the RGB-D datasets into "./test_rgb" subfolder, or change the folder name for RGB-D test files.


2. Run mapping.py, if needs mapping based on only the raw odometer. Video showing map and robot position would be saved as 'mapping.py'.

Set the test boolean to be True, or False when using training datasets.
Set the datanum to be the number of the LIDAR and joint datasets, 
	e.g. test_lidar0.mat has a 'datanum' of 0.

Put the LIDAR and joint datasets into “./test_data” subfolder, or change the folder name for LIDAR and joint test files. 
