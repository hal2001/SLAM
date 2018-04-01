#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/20/18 2:46 PM 

@author: Hantian Liu
"""
import load_data as ld
import numpy as np
from rot_util import roty, rotx
from math import cos
import pickle, os, pdb
from mapping import bodyToGlobal
import matplotlib
matplotlib.use('TkAgg')
#from matplotlib.path import Path
import matplotlib.pyplot as plt

def fit_plane(pitch, pts, eps):
	"""by setting up a threshold of eps
	determine whether the points fit the ground plane

	:param pitch: 1
	:param pts: n*3
	:param eps: 1, threshold for plane fitting
	:return: k, valid index of points on the ground
	"""

	pts_num=np.shape(pts)[0]

	# rotated normal vector to the ground plane
	nm_v=np.array([[0],[-1],[0]])
	new_nm_v=np.dot(rotx(pitch), nm_v)

	plane=np.zeros([1,4])
	plane[0,0:3]=new_nm_v.flatten()
	plane[0,3]=(0.93+0.33+0.07*cos(pitch))
	#pts_homo=np.vstack((pts, np.ones([1, np.shape(pts)[1]])))
	pts_homo=np.hstack((pts, np.ones([pts_num, 1])))
	plane_mat=np.tile(plane, (pts_num, 1))
	to_zero=plane_mat*pts_homo #n*4
	to_zero=np.sum(to_zero, axis=1)
	to_zero=abs(to_zero) #n
	ind_valid=np.where(to_zero<=eps)[0]
	return ind_valid


def alignCams(R, t):
	"""get transformation from the depth/IR camera
	to the RGB camera

	:param R: from depth/IR cam to RGB cam
	:param t:
	:return: T 4*4
	"""
	T=np.zeros([4,4])
	T[0:3, 0:3]=R
	T[0:3,3]=t.flatten()/1000 # in meter now
	T[-1,-1]=1
	return T

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
	u2=u-512/2 #px
	v2=v-424/2 #py
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
	u=fx*pts[:,0]/pts[:,2]+1920/2#px
	v=fy*pts[:,1]/pts[:,2]+1080/2#py
	return u, v




