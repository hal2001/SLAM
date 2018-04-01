#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/23/18 6:11 PM 

@author: Hantian Liu
"""

import numpy as np
from math import pi, sin, cos
from rot_util import rot, rotmat2D, rotmat3D, rotx, roty, rotz
import load_data as ld
from mapping import mToCell, updateMap, raycast, bodyToGlobal
import random, pdb, cv2, os, pickle, time
from localization import motionUpdate, initMap, initParticles, resample, compCorr
#from rgbd import fit_plane, alignCams, img_to_world, world_to_img

############################
## MODIFY THESE VARIABLES ##
############################
test = True
data_folder_name = "./test_data"

rgbd_flag = True  # whether to show rgb-d texture map
rgbd_folder_name = "./test_rgb"

if_multiple_data = False
datanum = 0  # data number

if_multiple_mat = True
rgbnum = 9  # rgb mat number
############################


def img_to_world(fx, fy, px, py, u, v, Z):
	"""given camera intrinsic parameters, depth and 2D coordinates in camera frame,
	convert to 3D coordinates in camera frame

	:param fx: 1
	:param fy: 1
	:param px: 1
	:param py: 1
	:param u: n
	:param v: n
	:param Z: n
	:return: n*3
	"""
	u=u.flatten()
	v=v.flatten()
	Z=Z.flatten()
	u2=u-px
	v2=v-py
	X=u2*Z/fx
	Y=v2*Z/fy
	ind=(Z>0.5) # invalid depth readings if Z>5m or Z<50cm
	ind=ind*(Z<5)
	Z=Z[ind]
	X=X[ind]
	Y=Y[ind]
	num=len(Z)
	pts=np.zeros([num,3])
	pts[:,0]=X
	pts[:,1]=Y
	pts[:,2]=Z
	return pts

def world_to_img(fx, fy, px, py, pts):
	"""given camera intrinsic parameters, 3D coordinates in camera frame,
	convert to 2D coordinates in camera frame

	:param fx: 1
	:param fy: 1
	:param pts: n*3
	:return: n, n
	"""
	u=fx*pts[:,0]/pts[:,2]+px
	v=fy*pts[:,1]/pts[:,2]+py
	return u, v

def kinectToGlobal(head_angles, rpy, odom_curr):
	"""Given IMU data and robot pose, get the transformation
	from Kinect sensor to global/world frame

	:param head_angles: 2
	:param rpy: 3
	:param odom_curr: 3
	:return: 4*4
	"""
	yaw=head_angles[0]
	pitch=head_angles[1]
	d_neck=np.array([[0],[0],[0.07]])
	d_body=np.array([[0],[0],[0.33]])

	#kinect to head
	T_l_to_h=np.zeros([4,4])
	T_l_to_h[0:3, 0:3]=roty(pitch)
	T_l_to_h[0:3, 3:4]=d_neck
	T_l_to_h[-1,-1]=1

	#head to body/IMU
	T_h_to_b=np.zeros([4,4])
	T_h_to_b[0:3,0:3]=rotz(yaw)
	T_h_to_b[0:3,3:4]=d_body
	T_h_to_b[-1,-1]=1

	#body to rotated body
	T = np.zeros([4, 4])
	T[-1, -1] = 1
	T[0:3, 0:3] = np.dot(roty(rpy[1]), rotx(rpy[0]))
	T=np.dot(T, np.dot(T_h_to_b, T_l_to_h))

	#rotated body to ground
	T_b_to_g=np.zeros([4,4])
	#T_b_to_g[0:3,0:3]=rotz(odom_curr[2])
	#print('...odom: '+str(odom_curr[2]))
	T_b_to_g[0:3, 0:3] = rotz(rpy[2])
	#print('...IMU: '+str(rpy[2]))
	T_b_to_g[0:2,3]=odom_curr[0:2].T
	T_b_to_g[2,3]=0.93
	T_b_to_g[-1,-1]=1

	#pdb.set_trace()
	return np.dot(T_b_to_g, T)


def TM(data_folder_name, datanum, rgbd_flag):
	if test:
		name0 = 'test'
	else:
		name0 = 'train'
	if not if_multiple_data:
		datanum = ''

	'''
	# Load kinect parameters
	fex = open('./cameraParam/exParams.pkl', 'rb')
	fIR = open('./cameraParam/IRcam_Calib_result.pkl', 'rb')
	fRGB = open('./cameraParam/RGBcamera_Calib_result.pkl', 'rb')
	exParams = pickle.load(fex, encoding = 'bytes')
	irParams = pickle.load(fIR, encoding = 'bytes')
	rgbParams = pickle.load(fRGB, encoding = 'bytes')
	# get camera intrinsic parameters
	fx_ir = irParams[b'fc'][0]
	fy_ir = irParams[b'fc'][1]
	px_ir = irParams[b'cc'][0]
	py_ir = irParams[b'cc'][1]
	fx_rgb = rgbParams[b'fc'][0]
	fy_rgb = rgbParams[b'fc'][1]
	px_rgb = rgbParams[b'cc'][0]
	py_rgb = rgbParams[b'cc'][1]
	'''

	fx_ir, fy_ir = 364.457362486, 364.542810627
	px_ir, py_ir = 258.422487562, 202.48713994
	fx_rgb, fy_rgb = 1049.3317526, 1051.31847629
	px_rgb, py_rgb = 956.910516428, 533.452032441
	# get transformation from depth/IR camera to RGB camera
	# T_alignCams = alignCams(exParams[b'R'], exParams[b'T'])
	T_alignCams = np.array([[0.99996855, 0.00589981, 0.00529992, 52.2682/1000],
							[-0.00589406, 0.99998202, -0.00109998, 1.5192/1000],
							[-0.00530632, 0.00106871, 0.99998535, -0.6059/1000],
							[0., 0., 0., 1.]])

	# load data
	lidar_new = ld.get_lidar(os.path.join(data_folder_name, name0 + '_lidar' + str(datanum)))
	joint_new = ld.get_joint(os.path.join(data_folder_name, name0 + '_joint' + str(datanum)))

	dpt_name = os.path.join(rgbd_folder_name, 'DEPTH' + (if_multiple_data * ('_' + str(datanum))))
	dpt_mat = ld.get_depth(dpt_name)

	image_name = os.path.join(rgbd_folder_name, 'RGB' + (if_multiple_data * ('_' + str(datanum))) + ( \
		if_multiple_mat * ('_' + str(1))))
	rgb = ld.get_rgb(image_name)

	# starting time and intervals for data scan
	mints = 0
	int_skip = 1

	#prev_img_ind = 0
	h = 424  # depth image height & width
	w = 512
	xr = np.arange(w)
	yr = np.arange(h)
	u_dpt, v_dpt = np.meshgrid(xr, yr)
	# v=rows in mat/y-axis
	# u=cols in mat/x-axis

	# threshold of plane fitting
	eps = 0.02

	# init video for texture map
	map = np.load('map' + str(datanum) + '.npy')
	texture_videoname = 'goodtexture' + str(datanum) + '.mp4'
	video_t = cv2.VideoWriter(texture_videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
							15, (np.shape(map)[1], np.shape(map)[0]))
	cv2.namedWindow(texture_videoname, cv2.WINDOW_NORMAL)

	img_videoname = 'grounddetection' + str(datanum) + '.mp4'
	video_img=cv2.VideoWriter(img_videoname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), \
							15, (1920, 1080))
	cv2.namedWindow('im_to_show', cv2.WINDOW_NORMAL)

	display_floor = np.tile((map > 0)[:, :, np.newaxis], (1, 1, 3)) * 125
	display_floor = display_floor.astype('uint8')

	#path_rx = np.load('path_rx'+str(datanum)+'.npy')
	#path_ry = np.load('path_ry'+str(datanum)+'.npy')
	path_x = np.load('path_x'+str(datanum)+'.npy')
	path_y = np.load('path_y'+str(datanum)+'.npy')
	path_theta=np.load('path_theta'+str(datanum)+'.npy')
	path_ts=np.load('path_ts'+str(datanum)+'.npy')

	# cell to meter
	path_x=(path_x+1)*0.1-50
	path_y = (path_y + 1) * 0.1 - 50

	# SLAM
	for i in np.arange(mints, len(dpt_mat), int_skip):
		i=int(i)
		print('\ndepth image ' + str(i))

		# check whether to get frames from different RGB mat
		samergb=(i+1)%300
		this_rgb=int((i+1)/300)
		if not samergb:
			if this_rgb ==rgbnum:
				print('\n...texture mapping end!')
				video_t.release()
				video_img.release()
				break
			print('\nTime to go with another RGB mat...')
			rgb=ld.get_rgb(os.path.join(rgbd_folder_name, 'RGB' + (if_multiple_data * ('_' + str(datanum))) + ( \
				if_multiple_mat * ('_' + str(this_rgb+1)))))

		# sync time stamps
		ts_diff = abs(joint_new['ts'] - dpt_mat[i]['t'])[0, :]
		ind_j = np.where(ts_diff == min(ts_diff))[0][0]
		ts_diff = abs(path_ts- dpt_mat[i]['t'])
		ind_p = np.where(ts_diff == min(ts_diff))[0]
		#pdb.set_trace()

		# load image
		#pdb.set_trace()
		image=rgb[i - 300*(this_rgb)]['image']
		dpt=dpt_mat[i]['depth']
		dpt = dpt / 1000  # depth in meter now
		head_angles = joint_new['head_angles'][:, ind_j]
		rpy = joint_new['rpy'][:, ind_j] - joint_new['rpy'][:, 0]

		# 3D in dpt frame
		Pts_dpt = img_to_world(fx_ir, fy_ir, px_ir, py_ir, u_dpt, v_dpt, dpt[v_dpt, u_dpt])
		# 3D in cam frame
		valid_homo = np.hstack((Pts_dpt, np.ones([np.shape(Pts_dpt)[0], 1])))# n*4
		Pts_rgb = np.dot(T_alignCams, valid_homo.T)  # 4*n
		# 2D in cam frame
		u_rgb, v_rgb = world_to_img(fx_rgb, fy_rgb, px_rgb, py_rgb, Pts_rgb.T)
		u_rgb = u_rgb.astype(np.int)
		v_rgb = v_rgb.astype(np.int)
		# get valid indices
		ind_max = np.logical_and(u_rgb < 1920, v_rgb < 1080)
		ind_min = np.logical_and(u_rgb >= 0, v_rgb >= 0)
		ind_range=np.logical_and(ind_max, ind_min)

		# 3D in world frame
		T=kinectToGlobal(head_angles, rpy, \
						 np.array([path_x[ind_p], path_y[ind_p], path_theta[ind_p]]))
		#pdb.set_trace()
		R = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]])
		Pts_w=np.dot(T, np.dot(R,Pts_rgb)) #4*n
		#pdb.set_trace()

		# get points close to ground plane
		mask_g=(Pts_w[2,:]<eps)
		ind_valid=np.logical_and(ind_range, mask_g)
		true_pos=Pts_w[:2, ind_valid]

		# pos in grid cells
		x_true_pos= np.ceil((true_pos[0,:] - (-50)) / 0.1).astype(np.int16) - 1
		y_true_pos = np.ceil((true_pos[1,:] - (-50)) / 0.1).astype(np.int16) - 1
		x_true_pos = x_true_pos.astype(np.int)
		y_true_pos = y_true_pos.astype(np.int)

		# visualization
		img_base = np.zeros_like(image)
		img_base[v_rgb[ind_valid], u_rgb[ind_valid]] += 1
		img_base = cv2.dilate(img_base, kernel = np.ones((5,5),np.uint8), iterations = 1)
		gmask = img_base[:, :, 2].copy()
		g = np.stack((np.zeros_like(gmask), np.zeros_like(gmask), gmask), 2)
		im_to_show = image.copy()+g*255*0.3
		cv2.imshow('im_to_show', im_to_show.astype(np.uint8))
		key = cv2.waitKey(100)
		video_img.write(im_to_show.astype(np.uint8))

		# adding texture
		display_floor[x_true_pos, y_true_pos, 2] = image[v_rgb[ind_valid], u_rgb[ind_valid], 0]
		display_floor[x_true_pos, y_true_pos, 1] = image[v_rgb[ind_valid], u_rgb[ind_valid], 1]
		display_floor[x_true_pos, y_true_pos, 0] = image[v_rgb[ind_valid], u_rgb[ind_valid], 2]
		cv2.imshow(texture_videoname, display_floor)
		key = cv2.waitKey(100)
		video_t.write(display_floor)

if __name__ == '__main__':
	TM(data_folder_name, datanum, rgbd_flag)