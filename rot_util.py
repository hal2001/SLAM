#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/15/18 4:38 PM 

@author: Hantian Liu
"""

import numpy as np
from math import cos, pi, sin

def rotx(theta):
	"""3D rotation around x-axis

	:param x: 3*n
	:param theta: angle in radian
	:return: 3*n after rotation theta around x-axis
	"""
	R=np.array([[1,0,0], [0,cos(theta),-sin(theta)], [0, sin(theta), cos(theta)]])
	return R

def roty(theta):
	"""3D rotation around y-axis

	:param x: 3*n
	:param theta: angle in radian
	:return: 3*n after rotation theta around y-axis
	"""
	R=np.array([[cos(theta),0, sin(theta)], [0,1,0],  [-sin(theta), 0, cos(theta)]])
	return R

def rotz(theta):
	"""3D rotation around z-axis

	:param x: 3*n
	:param theta: angle in radian
	:return: 3*n after rotation theta around y-axis
	"""
	R=np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta),0], [0,0,1]])
	return R

def rot(x, theta):
	"""apply 2D rotation to a vector

	:param x: 2*n
	:param theta: angle in radian
	:return: 2*n after rotation theta
	"""
	R=np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	return np.dot(R,x)


def rotmat2D(theta):
	"""2D rotation matrix

	:param theta: angle in radian
	:return:
	"""
	R=np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	return R


def rotmat3D(theta):
	"""3D rotation matrix around z-axis

	:param theta: angle in radian
	:return:
	"""
	R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
	return R