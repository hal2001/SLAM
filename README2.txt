“””
README file cont.for 
Project 4 SLAM and Texture Map
by Hantian Liu
“””

Note that SLAM_for_trajectory.py uses the same SLAM code in SLAM.py and functions in other .py files that I submitted before. The only difference is that I added some lines to save the robot trajectory for faster texture mapping. 

Please add these python files together with the code I submitted before in order to run normally. 

1. Run SLAM_for_trajectory.py to save the robot trajectories and final occupancy grid map, of SLAM based on LIDAR and joint data. Robot poses and maps would be saved as npy in local folder.

2. Run TM.py to generate texture maps, using the saved robot trajectories and maps.



######Below is the explanation for modifying parameters for folders and filenames, which are the same as in the README file I submitted before######  

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
