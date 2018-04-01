#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/21/18 11:35 PM 

@author: Hantian Liu
"""
import numpy as np
from math import pi, sin, cos
from rot_util import rot, rotmat2D, rotmat3D, rotx, roty, rotz
import load_data as ld
from mapping import mToCell, updateMap, raycast, bodyToGlobal
import random, pdb, cv2, os, pickle, time
from localization import motionUpdate, initMap, initParticles, resample, compCorr
from rgbd import fit_plane, alignCams, img_to_world, world_to_img
import matplotlib
matplotlib.use('TkAgg')
#from matplotlib.path import Path
import matplotlib.pyplot as plt


############################
## MODIFY THESE VARIABLES ##
############################
test=True
data_folder_name="./test_data"

rgbd_flag=False # whether to show rgb-d texture map
rgbd_folder_name="./test_rgb"

if_multiple_data=False
datanum='' # data number

if_multiple_mat=True
rgbnum=9 # rgb mat number
############################

def SLAM(data_folder_name, datanum, rgbd_flag):
	if test:
		name0='test'
	else:
		name0='train'
	if not if_multiple_data:
		datanum=''

	#load data
	lidar_new = ld.get_lidar(os.path.join(data_folder_name, name0+'_lidar'+str(datanum)))
	joint_new = ld.get_joint(os.path.join(data_folder_name, name0+'_joint'+str(datanum)))

	# init map
	map = {}
	map=initMap(map)

	# init particles
	pr = {}
	pr = initParticles(pr)

	# starting time and intervals for data scan
	mints=0
	int_skip=10

	# init video
	videoname = 'slam' + str(datanum)+'.mp4'
	video = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
							15, (map['sizey'], map['sizex']))
	cv2.namedWindow(videoname, cv2.WINDOW_NORMAL)

	#############################################
	path_x=[]
	path_y=[]
	path_theta=[]
	path_ts=[]
	#############################################

	# SLAM
	for i in np.arange(mints, len(lidar_new), int_skip):
		print('\nscan ' + str(i))

		# sync time stamps
		ts_diff = abs(joint_new['ts'] - lidar_new[i]['t'])[0, :]
		ind_j = np.where(ts_diff == min(ts_diff))[0][0]
		#print('lidar'+str(i)+str(lidar_new[i]['t']))
		#print('joint'+str(ind_j)+ str(joint_new['ts'][:, ind_j]))

		start = time.time()

		# first scan to update map only
		if i==mints:
			ind_j_first = ind_j
			pr['best']=lidar_new[i]['pose']
			pr['best']=pr['best'].astype('float')
			pr['odom']=np.tile(pr['best'],[pr['num'],1])
			ray_g= raycast(joint_new, lidar_new, map, i, ind_j, ind_j_first, pr['best'])
			map = updateMap(map, pr['best'], ray_g)
			continue

		# standard deviation for gaussian noise in motion update
		binary_map = (map['occu_grids'] > 0).astype('uint8')
		delta_pose=lidar_new[i]['pose']-lidar_new[i - int_skip]['pose']
		stdx=max(0.001*int_skip, abs(delta_pose[0,0])/3)
		stdy=max(0.001*int_skip, abs(delta_pose[0,1])/3)
		stdtheta=max(0.005*int_skip, abs(delta_pose[0,2])/3)

		# for each particle
		for p in range(pr['num']):
			# particle propagation
			motion_noise=np.random.normal(0, [[stdx], [stdy], [stdtheta]], (3,1))
			pr['odom'][p, :] =motionUpdate(lidar_new[i]['pose'], lidar_new[i - int_skip]['pose'], pr['odom'][p, :], motion_noise)

			# use map correlation to update particle weights
			ray_g = raycast(joint_new, lidar_new, map, i, ind_j, ind_j_first, pr['odom'][p,:][np.newaxis,:]) #3*n
			ray_cell_x, ray_cell_y = mToCell(ray_g[0, :], ray_g[1, :], map)
			corrsum=compCorr(binary_map, map, ray_cell_x, ray_cell_y)
			pr['weights'][p, :] = pr['weights'][p, :] * corrsum

		# find best particle with maximum weights
		ind_best=np.where(pr['weights']==max(pr['weights']))
		ind_best=ind_best[0][0]
		pr['best']=pr['odom'][ind_best][np.newaxis, :]

		# weights normalization
		pr['weights'] = pr['weights'] / sum(pr['weights'])

		# update map with best particle only
		ray_g = raycast(joint_new, lidar_new, map, i, ind_j, ind_j_first, pr['best'])
		map = updateMap(map, pr['best'], ray_g)

		#print('...num of particles with max weights: '+str(np.where(pr['weights'] == max(pr['weights']))[0].size))
		print('...max weights: ' + str(max(pr['weights'])))
		print('...best pose: '+str(pr['best']))

		# resample particles when effective particles is less than threshold
		Neff=sum(pr['weights'])**2/sum(pr['weights']**2)
		print('...effective particle number: '+str(Neff))
		if Neff<pr['Neff']:
			pr=resample(pr)
			print('\nresampling done!')

		end = time.time()
		print(end - start)

		# visulization
		# occupancy grids
		display = (map['occu_grids']).copy()
		displayrgb=np.tile(display[:,:,np.newaxis], (1,1,3))
		displayrgb= (displayrgb + 100) / 200 * 255
		displayrgb = displayrgb.astype('uint8')
		# texts
		font = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (20, 20)
		fontScale = 0.4
		fontColor = (255, 255, 255)
		cv2.putText(displayrgb, 'Time passed: ' + str(
			(lidar_new[i]['t'][0, :] - lidar_new[0]['t'][0, :])) + '   Robot pose: ' + str(pr['best'][0, :]),
					bottomLeftCornerOfText,
					font, fontScale, fontColor, thickness = 1)
		# ray ends of each scan
		#displayrgb[ray_cell_x, ray_cell_y, 0] = 255
		#displayrgb[ray_cell_x, ray_cell_y, 1]= 0
		#displayrgb[ray_cell_x, ray_cell_y, 2] = 0

		# all particles
		'''
		for p in range(pr['num']):
			# meter to cell
			cell_x, cell_y = mToCell(pr['odom'][p, 0], pr['odom'][p, 1], map)
			cv2.circle(displayrgb, (cell_y, cell_x), 1, (255, 0, 255), -11)
		'''

		# meter to cell
		rob_cell_x, rob_cell_y = mToCell(pr['best'][:, 0], pr['best'][:, 1], map)
		raw_cell_x, raw_cell_y= mToCell(lidar_new[i]['pose'][:,0], lidar_new[i]['pose'][:,1], map)

		#########################################################
		path_x.append(rob_cell_x)
		path_y.append(rob_cell_y)
		path_theta.append(pr['best'][:,2])
		path_ts.append(lidar_new[i]['t'])

		# show trajectory
		arrowlen = 7
		for j in range(len(path_x)):
			'''
			cv2.circle(displayrgb, (path_ry[j], path_rx[j]), 2, (255,10,0), -11)
			end_x_raw = raw_cell_x + arrowlen * cos(lidar_new[i]['pose'][:, 2])
			end_y_raw = raw_cell_y + arrowlen * sin(lidar_new[i]['pose'][:, 2])
			cv2.arrowedLine(displayrgb, (raw_cell_y, raw_cell_x), \
							(int(end_y_raw), int(end_x_raw)), (255, 10, 0), 5)
			'''

			# optimized odometry (best particle)
			cv2.circle(displayrgb, (path_y[j], path_x[j]), 2, (0,10, 255), -11)
			end_x = rob_cell_x + arrowlen * cos(pr['best'][:, 2])
			end_y = rob_cell_y + arrowlen * sin(pr['best'][:, 2])
			cv2.arrowedLine(displayrgb, (rob_cell_y, rob_cell_x), \
							(int(end_y), int(end_x)), (0, 10, 255), 5)

		#######################################################

		#display = cv2.applyColorMap(display, cv2.COLORMAP_BONE)
		cv2.imshow(videoname, displayrgb)
		key = cv2.waitKey(100)
		video.write(displayrgb)

	video.release()
	cv2.destroyAllWindows()

	########################################################
	np.save('map'+str(datanum)+'.npy', map['occu_grids'])
	#np.save('path_rx0.npy', np.asarray(path_rx))
	#np.save('path_ry0.npy', np.asarray(path_ry))
	np.save('path_x'+str(datanum)+'.npy', np.asarray(path_x))
	np.save('path_y'+str(datanum)+'.npy', np.asarray(path_y))
	np.save('path_theta'+str(datanum)+'.npy', np.asarray(path_theta))
	np.save('path_ts'+str(datanum)+'.npy', np.asarray(path_ts))
	########################################################



if __name__ == '__main__':
	SLAM(data_folder_name, datanum, rgbd_flag)

