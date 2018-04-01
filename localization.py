#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/16/18 4:57 PM 

@author: Hantian Liu
"""
import numpy as np
from math import pi, sin, cos
from rot_util import rot, rotmat2D, rotmat3D, rotx, roty, rotz
import load_data as ld
from mapping import mToCell, updateMap, raycast
import random, pdb, cv2
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib.path import Path
#import matplotlib.pyplot as plt
#import matplotlib.animation as manimation


def initMap(map):
	"""initialize map

	:param map: dict
	:return: dict
	"""
	map['res'] = 0.1  # meters
	map['xmin'] = -50  # meters
	map['ymin'] = -50
	map['xmax'] = 50
	map['ymax'] = 50
	map['sizex'] = int(np.ceil((map['xmax'] - map['xmin']) / map['res'] + 1))  # cells
	map['sizey'] = int(np.ceil((map['ymax'] - map['ymin']) / map['res'] + 1))
	map['occu_grids'] = np.zeros((map['sizex'], map['sizey']))
	return map

def initParticles(particles):
	"""initialize particles

	:param particles: dict
	:return: dict
	"""
	particles['num']=100
	particles['stdx']=1
	particles['stdy']=1
	particles['stdtheta']=pi/2
	particles['Neff']=particles['num']*0.4
	particles['weights']=1/particles['num']*np.ones([particles['num'],1])
	particles['odom'] = np.zeros([particles['num'], 3])  # N*3 odom of particles
	particles['best']=np.zeros([1,3]) # odom of best particle
	return particles

def motionUpdate(raw_odom_curr, raw_odom_prev, odom_prev, noise):
	"""propagate the particles' states based on odometry

	:param raw_odom_curr: 1*3
	:param raw_odom_prev: 1*3
	:param odom_prev: 3
	:param noise: 3*1
	:return: 1*3
	"""
	#print('raw' + str(raw_odom_prev))
	#print(odom_prev)
	#print('raw' +str(raw_odom_curr))
	delta_odom = np.dot(rotmat2D(raw_odom_prev[:, 2]).T, (raw_odom_curr[:, 0:2] - raw_odom_prev[:, 0:2]).T)
	pos_curr = np.dot(rotmat2D(odom_prev[2]), delta_odom) + odom_prev[0:2, np.newaxis] + noise[0:2, :]
	yaw_curr = (raw_odom_curr[:, 2] - raw_odom_prev[:, 2]) + odom_prev[2] + noise[2, :]
	odom_curr = np.vstack((pos_curr, yaw_curr)).T
	#print('updated  '+str(odom_curr[0,:]))
	return odom_curr[0,:]

def resample(pr):
	"""resample the particles when the effective number of the particles
	becomes small

	:param pr: dict
	:return: updated pr
	"""
	cumsum=np.cumsum(pr['weights'])
	#idx=np.mod(random.random()+np.arange(pr['num'])/pr['num'], 1)
	#for p in range(pr['num']):
	#	pr['odom'][p,:] = pr['odom'][np.where(idx[p] <= cumsum)[0][0],:]

	for p in range(pr['num']):
		pr['odom'][p,:] = pr['odom'][np.where(random.random() <= cumsum)[0][0],:]
	pr['weights'] = 1 / pr['num'] * np.ones([pr['num'], 1])
	return pr

def compCorr(binary_map, map, ray_cell_x, ray_cell_y):
	"""compute correlation between the scan with ray ends' locations in grids
	and the previous occupancy grids map

	:param binary_map:
	:param map: dict
	:param ray_cell_x: n
	:param ray_cell_y: n
	:return: overlapped cells between the scan and the previous map
	"""
	'''
	x, y=np.shape(binary_map)
	xr=np.arange(x)
	yr=np.arange(y)
	yv, xv=np.meshgrid(xr, yr)
	obs_mask=binary_map[xv, yv]
	free_mask=np.ones(np.shape(obs_mask))-obs_mask
	obs_corr=sum(obs_mask[ray_cell_x, ray_cell_y])

	# free_cell_x, free_cell_y = getMapCellsFromRay(rob_cell_x, rob_cell_y, ray_cell_x, ray_cell_y)
	'''
	ver = np.hstack((ray_cell_x[:, np.newaxis], ray_cell_y[:, np.newaxis]))
	poly = ver.reshape(-1, 1, 2).astype(np.int32)
	matrix = np.zeros((map['sizex'], map['sizey']), dtype = np.int32)
	# pdb.set_trace()
	cv2.drawContours(matrix, [poly], -1, (1), thickness = -1)
	matrix = matrix.T
	# list_of_points_indices = np.nonzero(matrix)
	# free_cell_y = np.asarray(list_of_points_indices[0])
	# free_cell_x = np.asarray(list_of_points_indices[1])
	# free_corr = sum(free_mask[matrix > 0])
	obs=np.sum(map['occu_grids'][ray_cell_x, ray_cell_y]>0)
	free=np.sum(map['occu_grids'][matrix>0]<0)-np.sum(map['occu_grids'][ray_cell_x, ray_cell_y]<0)

	return (obs+free)






