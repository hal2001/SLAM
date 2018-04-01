#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/15/18 5:17 PM 

@author: Hantian Liu
"""
import numpy as np
from rot_util import rot, rotx, roty, rotz, rotmat2D, rotmat3D
from math import cos, sin, pi
import load_data as ld
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib.path import Path
#import matplotlib.pyplot as plt
import cv2, pdb, os

############################
## MODIFY THESE VARIABLES ##
############################
test=True
data_folder_name="./test_data"
datanum='' # data number
############################

def motionUpdate0(raw_odom_curr, raw_odom_prev, odom_prev, noise):
	"""

	:param raw_odom_curr:
	:param raw_odom_prev:
	:param odom_prev:
	:param noise:
	:return:
	"""
	print('raw   '+str(raw_odom_curr))
	delta_odom=np.dot(rotmat2D(raw_odom_prev[:,2]).T, (raw_odom_curr[:,0:2]-raw_odom_prev[:,0:2]).T)
	pos_curr=np.dot(rotmat2D(odom_prev[2]), delta_odom)+odom_prev[0:2, np.newaxis]+noise[0:2,:]
	yaw_curr=(raw_odom_curr[:,2]-raw_odom_prev[:,2])+odom_prev[2]+noise[2,:]
	odom_curr=np.vstack((pos_curr, yaw_curr)).T
	print('global   '+str(odom_curr))
	return odom_curr


def scanToBody(ranges, angles, head_angles):
	"""given ranges and angles of casting-rays, get their polar coordinates
	in the LIDAR frame, and convert lidar frame to head frame,
	then finally convert to body frame(CoM) of the robot,
	remove the rays hitting ground

	:param ranges: n
	:param angles: n
	:param head_angles: 2
	:return: 4*n in body frame
	"""
	r_h=np.zeros([3, len(ranges.T)])
	r_h[0,:]=ranges*np.cos(angles)
	r_h[1,:]=ranges*np.sin(angles)

	# plot original lidar points
	#plt.figure()
	#plt.plot(r_h[1,:], r_h[0,:])
	#plt.show()
	yaw=head_angles[0]
	pitch=head_angles[1]
	d_neck=np.array([[0],[0],[0.15]])
	d_body=np.array([[0],[0],[0.33]])

	# r_b=np.dot(rotz(yaw), np.dot(roty(pitch), r_h)+d_neck)+d_body

	#lidar to head
	T_l_to_h=np.zeros([4,4])
	T_l_to_h[0:3, 0:3]=roty(pitch)
	T_l_to_h[0:3, 3:4]=d_neck
	T_l_to_h[-1,-1]=1

	#head to body/IMU
	T_h_to_b=np.zeros([4,4])
	T_h_to_b[0:3,0:3]=rotz(yaw)
	T_h_to_b[0:3,3:4]=d_body
	T_h_to_b[-1,-1]=1

	r_homo = np.vstack((r_h, np.ones([1, np.shape(r_h)[1]])))
	r_b=np.dot(T_h_to_b, np.dot(T_l_to_h, r_homo))

	#remove ground hits
	indValid=(r_b[2,:]>=-0.93)
	r_b=r_b[:,indValid]
	return r_b, np.dot(T_h_to_b, T_l_to_h)

def imu_rot(r, rpy):
	"""based on IMU rpy readings, apply transformation
	from original body frame, to rotated body frame to points

	:param r: 4*n, body frame
	:param rpy: IMU readings
	:return: 4*n in rotated body frame
	"""
	T=np.zeros([4,4])
	T[-1,-1]=1
	T[0:3, 0:3]=np.dot(roty(rpy[1]), rotx(rpy[0]))
	r=np.dot(T, r)
	return r

def bodyToGlobal(r, odom_curr, rpy):
	"""based on current odometry in global frame,
	apply transformation from body frame to world frame
	to points

	:param r: 4*n, body frame (CoM)
	:param odom_curr: 3, world frame: x, y ,theta(yaw)
	:param rpy: 3, world frame: IMU roll pitch yaw
	:return: 4*n in world frame
	"""
	T_b_to_g=np.zeros([4,4])
	T_b_to_g[0:3,0:3]=rotz(odom_curr[2])
	#print('...odom: '+str(odom_curr[2]))
	#T_b_to_g[0:3, 0:3] = rotz(rpy[2])
	#print('...IMU: '+str(rpy[2]))
	angle = (odom_curr[2] + rpy[2]) / 2
	#T_b_to_g[0:3, 0:3]=rotz(angle)
	T_b_to_g[0:2,3]=odom_curr[0:2].T
	T_b_to_g[2,3]=0.93
	T_b_to_g[-1,-1]=1
	#r_homo = np.vstack((r, np.ones([1, np.shape(r)[1]])))
	r_g=np.dot(T_b_to_g, r)
	return r_g, T_b_to_g


def mToCell(x,y, map):
	"""convert meter to corresponding cells in occupancy grids

	:param x: n
	:param y: n
	:param map: dict
	:return: n, n
	"""
	xis = np.ceil((x - map['xmin']) / map['res']).astype(np.int16) - 1
	yis = np.ceil((y - map['ymin']) / map['res']).astype(np.int16) - 1
	return xis, yis

def saturationCheck(map):
	"""thresholding maximum log likelihood in occupancy grids

	:param map: dict
	:return: dict
	"""
	log_odds_free_sat = -100
	log_odds_occupied_sat = 100

	map[map>=log_odds_occupied_sat]=log_odds_occupied_sat
	map[map<=log_odds_free_sat]=log_odds_free_sat
	return map


def raycast(joint_new, lidar_new, map, i, ind_j, ind_j_first, odom):
	"""given head angles, scan info, and current odometry,
	remove invalid scan info, get the coordinate of rays's ends
	in global/world frame, with units in meter

	:param joint_new: dict synced  (load_data get_joints)
	:param lidar_new: list synced (load_data get_lidar)
	:param map: dict
	:param i: ith scan of lidar
	:param ind_j: of joint
	:param ind_j_first: of joint, for deleting bias of IMU rpy
	:param odom: 1*3
	:return: 3*n [meter]
	"""

	# read scanned data
	ranges = lidar_new[i]['scan']
	angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.])  # .T

	# remove invalid range
	indInvalid = np.logical_or((ranges >= 30), (ranges <= 0.1))
	ranges[indInvalid] = 0
	#indValid = np.logical_and((ranges < 30), (ranges > 0.1))
	#ranges = ranges[indValid]
	#angles = angles[indValid]

	# convert rays' ends from LIDAR frame to body frame,
	# then from body frame to global frame based on current odometry
	head_angles = joint_new['head_angles'][:, ind_j]
	rpy=joint_new['rpy'][:, ind_j]-joint_new['rpy'][:,ind_j_first]
	ray_b, T_h_to_b = scanToBody(ranges, angles, head_angles)
	#print(ray_b)
	ray_b=imu_rot(ray_b, rpy)
	#print(ray_b)
	ray_g, T_b_to_g = bodyToGlobal(ray_b, odom[0,:], rpy)
	T_h_to_g= np.dot(T_b_to_g, T_h_to_b)
	ray_g=ray_g[0:3,:]

	return ray_g


def updateMap(map, odom, ray_g):
	"""update occupancy grids' log likelihood based on given coordinate
	of rays' ends with units in meter

	:param map: dict
	:param odom: [meter] 1*3
	:param ray_g: [meter] 3*1
	:return: dict
	"""
	log_odds_free_update = 2.5/4
	log_odds_occupied_update = 2.5

	# convert units of robot pos (odometry) and rays' ends
	# from meter to cell
	#rob_cell_x, rob_cell_y = mToCell(odom[:, 0], odom[:, 1], map)
	ray_cell_x, ray_cell_y = mToCell(ray_g[0, :], ray_g[1, :], map)

	# find free cells based on rays' ends and robot pos
	ver=np.hstack((ray_cell_x[:,np.newaxis], ray_cell_y[:, np.newaxis]))
	poly = ver.reshape(-1, 1, 2).astype(np.int32)
	matrix = np.zeros((map['sizex'], map['sizey']), dtype = np.int32)
	#pdb.set_trace()
	cv2.drawContours(matrix, [poly], -1, (1), thickness = -1)
	matrix=matrix.T

	'''
	list_of_points_indices = np.nonzero(matrix)
	free_cell_y2 = np.asarray(list_of_points_indices[0])
	free_cell_x2 = np.asarray(list_of_points_indices[1])
	'''
	#free_cell_x, free_cell_y = getMapCellsFromRay(rob_cell_x, rob_cell_y, ray_cell_x, ray_cell_y)

	# log odds update
	map['occu_grids'][ray_cell_x, ray_cell_y] = \
		map['occu_grids'][ray_cell_x, ray_cell_y] + log_odds_occupied_update +log_odds_free_update
	# map['occu_grids'][free_mask]=map['occu_grids'][free_mask]-log_odds_free_update
	#map['occu_grids'][free_cell_x, free_cell_y] = \
	#	map['occu_grids'][free_cell_x, free_cell_y] - log_odds_free_update
	map['occu_grids'][matrix>0]=map['occu_grids'][matrix>0]-log_odds_free_update

	# saturation limit of log odds
	map['occu_grids'] = saturationCheck(map['occu_grids'])

	return map

if __name__ == '__main__':
	if test:
		name0='test'
	else:
		name0='train'

	lidar_new = ld.get_lidar(os.path.join(data_folder_name, name0 + '_lidar' + str(datanum)))
	joint_new = ld.get_joint(os.path.join(data_folder_name, name0 + '_joint' + str(datanum)))

	log_odds_free_update=0.75
	log_odds_occupied_update=1.5
	log_odds_free_sat=-100
	log_odds_occupied_sat=100

	# init map
	map = {}
	map['res'] = 0.1  # meters
	map['xmin'] = -30  # meters
	map['ymin'] = -30
	map['xmax'] = 30
	map['ymax'] = 30
	map['sizex'] = int(np.ceil((map['xmax'] - map['xmin']) / map['res'] + 1))  # cells
	map['sizey'] = int(np.ceil((map['ymax'] - map['ymin']) / map['res'] + 1))
	map['occu_grids'] = np.zeros((map['sizex'], map['sizey']))  # DATA TYPE: char or int8

	allx, ally = np.meshgrid(np.arange(map['sizex']), np.arange(map['sizey']))
	allx, ally = allx.flatten(), ally.flatten()
	all_points = np.vstack((allx, ally)).T

	videoname = 'mapping' + '.mp4'
	video = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
							15, (map['sizey'], map['sizex']))
	cv2.namedWindow(videoname, cv2.WINDOW_NORMAL)

	motion_noise=np.zeros([3,1])
	#fig = plt.figure()

	int_skip = 10

	#odom = np.zeros([5500, 3])
	for i in np.arange(0, len(lidar_new), int_skip):

		print('iteration '+str(i))
		odom=lidar_new[i]['pose'][0,:][np.newaxis,:]

		ts_diff = abs(joint_new['ts'] - lidar_new[i]['t'])[0, :]
		ind_j = np.where(ts_diff == min(ts_diff))[0][0]

		if i==0:
			ind_j_first=ind_j

		ray_g=raycast(joint_new, lidar_new, map, i, ind_j, ind_j_first, odom)
		map=updateMap(map, odom, ray_g)

		# meter to cell
		rob_cell_x, rob_cell_y = mToCell(odom[:, 0], odom[:, 1], map)

		display=map['occu_grids'].copy()
		display=(display+400)/800*255
		display=display.astype('uint8')
		cv2.circle(display, (rob_cell_y, rob_cell_x), 5, (255), -11)

		arrowlen=7
		end_x=rob_cell_x+arrowlen*cos(odom[:,2])
		end_y=rob_cell_y+arrowlen*sin(odom[:,2])
		cv2.arrowedLine(display, (rob_cell_y, rob_cell_x), \
						(int(end_y), int(end_x)), (255), 5)
		display=cv2.applyColorMap(display, cv2.COLORMAP_BONE)
		cv2.imshow(videoname, display)
		key = cv2.waitKey(100)
		video.write(display)

	video.release()
	cv2.destroyAllWindows()



