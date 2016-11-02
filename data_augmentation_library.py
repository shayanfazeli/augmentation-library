#!/usr/bin/python
"""
DATA AUGMENTATION LIBRARY

- Different color perturbations, rotations with different capabilities, fancy PCA, random projections, etc.
- All of the above strategies have been implemented in this code, which can be exploited in a caffe python layer or when creating an lmdb dataset to reduce the chance of overfitting.

"""
__date__= "2016"
__credits__ = ["Shayan Fazeli"]
__maintainer = "Shayan Fazeli"
__email__ = "shayan.fazeli@gmail.com"

import timeit
import numpy as np
import math
import cv2
import pdb
from IPython import embed
from sklearn.decomposition import PCA
from IPython import embed
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from numpy import linalg as LA
import copy
from scipy import stats
import pickle
from random import shuffle

def da_key_maker():
	thelist = []
	for i in range(1000000):
		thelist.append(i+1)
	shuffle(thelist)
	return thelist

def mat_normalizer(matrix):
	matrix = matrix - matrix.min()
	matrix = matrix / matrix.max()
	matrix = matrix * 255.0
	return matrix.astype('uint8')

def da_vectorized_image(image):
	return image.reshape((image.shape[0] * image.shape[1], 3))

def da_pca_model_finder(input_image):
	data = da_vectorized_image(input_image)
	pca = PCA()
	pca.fit(data)
	
	print "Writing down the PCA model...\n"
	pickle.dump(pca, open('PCA_fitted.pkl', 'wb'))
	print "PCA model is successfully written to HDD.\n"
	return pca
	
def da_fancy_cov(input_image):
	data = da_vectorized_image(input_image)
	covariance_matrix = np.cov(data.transpose() - np.repeat(np.reshape(np.mean(data,0), (1,3)), data.shape[0], axis=0).transpose())
	eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
	lambda1 = eigen_values[0]
	lambda2 = eigen_values[1]
	lambda3 = eigen_values[2]
	offset1 = lambda1 * (np.random.normal(0, 0.1, 1)[0])
	offset2 = lambda2 * (np.random.normal(0, 0.1, 1)[0])
	offset3 = lambda3 * (np.random.normal(0, 0.1, 1)[0])
	offset_cache = np.array([[offset1], [offset2], [offset3]])
	offset_cache = np.dot(eigen_vectors, offset_cache)
	offset1 = offset_cache[0]
	offset2 = offset_cache[1]
	offset3 = offset_cache[2]
	offset_matrix = np.ones(data.shape)
	offset_matrix[:,0] = offset_matrix[:,0] * offset1
	offset_matrix[:,1] = offset_matrix[:,1] * offset2
	offset_matrix[:,2] = offset_matrix[:,2] * offset3
	data = data + offset_matrix
	output_image = np.reshape(data, (input_image.shape[0], input_image.shape[1], input_image.shape[2]))
	return output_image.astype('uint8')

def da_fancy_cov_batch_pca(input_image):
	data = da_vectorized_image(input_image)
	covariance_matrix = pickle.load(open("covariance_matrix.pkl","rb"))
	eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
	lambda1 = eigen_values[0]
	lambda2 = eigen_values[1]
	lambda3 = eigen_values[2]
	
	
	offset1 = lambda1 * (np.random.normal(0, 0.1, 1)[0])
	offset2 = lambda2 * (np.random.normal(0, 0.1, 1)[0])
	offset3 = lambda3 * (np.random.normal(0, 0.1, 1)[0])
	
	offset_cache = np.array([[offset1], [offset2], [offset3]])
	offset_cache = np.dot(eigen_vectors, offset_cache)
	offset1 = offset_cache[0]
	offset2 = offset_cache[1]
	offset3 = offset_cache[2]
	offset_matrix = np.ones(data.shape)
	offset_matrix[:,0] = offset_matrix[:,0] * offset1
	offset_matrix[:,1] = offset_matrix[:,1] * offset2
	offset_matrix[:,2] = offset_matrix[:,2] * offset3
	data = data + offset_matrix
	data = data - data.min()
	data = data / data.max()
	data = data * 255
	#pdb.set_trace()
	output_image = np.reshape(data, (input_image.shape[0], input_image.shape[1], input_image.shape[2]))
	return output_image.astype('uint8')










def da_fancy_pca(input_image, pca):
	data = da_vectorized_image(input_image)
	new_data = pca.transform(data)
	#offset  finder 1:
	
	covariance_matrix = pca.get_covariance()
	eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
	lambda1 = eigen_values[0]
	lambda2 = eigen_values[1]
	lambda3 = eigen_values[2]
	offset1 = lambda1 * (np.random.normal(0, 0.1, 1)[0])
	offset2 = lambda2 * (np.random.normal(0, 0.1, 1)[0])
	offset3 = lambda3 * (np.random.normal(0, 0.1, 1)[0])
	offset_cache = np.array([[offset1], [offset2], [offset3]])
	offset_cache = np.dot(eigen_vectors, offset_cache)
	offset1 = offset_cache[0]
	offset2 = offset_cache[1]
	offset3 = offset_cache[2]
	
	
	#offset finder 2:
	"""
	eigen_values, eigen_vectors = np.linalg.eig(np.cov(new_data.transpose()))
	lambda1 = eigen_values[0]
	lambda2 = eigen_values[1]
	lambda3 = eigen_values[2]
	offset1 = lambda1 * (np.random.normal(0, 0.001, 1)[0])
	offset2 = lambda2 * (np.random.normal(0, 0.001, 1)[0])
	offset3 = lambda3 * (np.random.normal(0, 0.001, 1)[0])
	offset_cache = np.array([[offset1], [offset2], [offset3]])
	offset_cache = np.dot(eigen_vectors.transpose(), offset_cache)
	offset1 = offset_cache[0]
	offset2 = offset_cache[1]
	offset3 = offset_cache[2]
	"""
	
	#offset finder 3:
	"""
	offset1 = np.mean(np.abs(new_data[:,0])) * (np.random.normal(0, 1, 1)[0])
	offset2 = np.mean(np.abs(new_data[:,1])) * (np.random.normal(0, 1, 1)[0])
	offset3 = np.mean(np.abs(new_data[:,2])) * (np.random.normal(0, 1, 1)[0])
	"""
	offset_matrix = np.ones(new_data.shape)
	offset_matrix[:,0] = offset_matrix[:,0] * offset1
	offset_matrix[:,1] = offset_matrix[:,1] * offset2
	offset_matrix[:,2] = offset_matrix[:,2] * offset3
	new_data = new_data + offset_matrix
	data2 = pca.inverse_transform(new_data)
	output_image = np.reshape(data2, (input_image.shape[0], input_image.shape[1], input_image.shape[2]))
	pdb.set_trace()
	return output_image.astype('uint8')


def show_image(input_image):
	cv2.imshow('image', input_image)
	cv2.waitKey()

def da_flip(input_image):
	return np.fliplr(input_image)

def da_hsv_t1(input_image, offset):
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	hsv[:,:,0] = hsv[:,:,0] + offset
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def da_hsv_t2(input_image, channel, coefficient):
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	if channel == 's':
		hsv[:,:,1] = hsv[:,:,1] * coefficient
	elif channel == 'v':
		hsv[:,:,2] = hsv[:,:,2] * coefficient
		hsv[:,:,2] = 244
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def da_hsv_t3(input_image, channel, power, med_filter):
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	if channel == 's':
		hsv[:,:,1] = (255*((hsv[:,:,1]/255.0) ** power))
	elif channel == 'v':
		hsv[:,:,2] = (255*((hsv[:,:,2]/255.0) ** power))
	return cv2.medianBlur(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), med_filter)

def da_random_projective(input_image):
	height, width, depth = input_image.shape
	
	#obtaining the color to fit the black margins with:
	#################################################################
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	margin_h = int(0.05*height)
	margin_w = int(0.05*width)
	h_values1 = hsv[0:margin_h, 0:margin_w, 0]
	h_values2 = hsv[0:margin_h, -1*margin_w:, 0]
	h_values3 = hsv[-1*margin_h:, 0:margin_w, 0]
	h_values4 = hsv[-1*margin_h:, -1*margin_w:, 0]
	h_values_stacked1 = np.hstack((h_values1, h_values2))
	h_values_stacked2 = np.hstack((h_values3, h_values4))
	h_values_stacked = np.hstack((h_values_stacked1, h_values_stacked2))
	H_VALUE = np.median(h_values_stacked.ravel())
	H_VALUE_UP = np.median(h_values_stacked1.ravel())
	H_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	s_values1 = hsv[0:margin_h, 0:margin_w, 1]
	s_values2 = hsv[0:margin_h, -1*margin_w:, 1]
	s_values3 = hsv[-1*margin_h:, 0:margin_w, 1]
	s_values4 = hsv[-1*margin_h:, -1*margin_w:, 1]
	s_values_stacked = np.hstack((s_values1, s_values2))
	s_values_stacked1 = np.hstack((s_values1, s_values2))
	s_values_stacked2 = np.hstack((s_values3, s_values4))
	s_values_stacked = np.hstack((s_values_stacked1, s_values_stacked2))
	S_VALUE = np.median(s_values_stacked.ravel())
	S_VALUE_UP = np.median(h_values_stacked1.ravel())
	S_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	v_values1 = hsv[0:margin_h, 0:margin_w, 2]
	v_values2 = hsv[0:margin_h, -1*margin_w:, 2]
	v_values3 = hsv[-1*margin_h:, 0:margin_w, 2]
	v_values4 = hsv[-1*margin_h:, -1*margin_w:, 2]
	v_values_stacked = np.hstack((v_values1, v_values2))
	v_values_stacked1 = np.hstack((v_values1, v_values2))
	v_values_stacked2 = np.hstack((v_values3, v_values4))
	v_values_stacked = np.hstack((v_values_stacked1, v_values_stacked2))
	V_VALUE = np.median(v_values_stacked.ravel())
	V_VALUE_UP = np.median(v_values_stacked1.ravel())
	V_VALUE_DOWN = np.median(v_values_stacked2.ravel())
	######################################################################
	
	input_image = np.lib.pad(input_image, ((int(0.5*height), int(0.5*height)), (int(0.5*width), (int(0.5*width))), (0,0)), 'constant')
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	center_row = int(num_of_rows / 2.0)
	center_col = int(num_of_cols / 2.0)
	#enumerating the corners clockwise, and obtaining the correspondences:
	p1x = center_col - int(width / 2.0)
	p1y = center_row - int(height / 2.0)
	p2x = center_col + int(width / 2.0)
	p2y = center_row - int(height / 2.0)
	p3x = center_col + int(width / 2.0)
	p3y = center_row + int(height / 2.0)
	p4x = center_col - int(width / 2.0)
	p4y = center_row + int(height / 2.0)
	
	offset_x = 0.05 * int(width)
	offset_y = 0.05 * int(height)
	
	deltaq1x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq1y = offset_y * (np.random.normal(0,0.5,1)[0])
	deltaq2x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq2y = offset_y * (np.random.normal(0,0.5,1)[0])
	deltaq3x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq3y = offset_y * (np.random.normal(0,0.5,1)[0])
	deltaq4x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq4y = offset_y * (np.random.normal(0,0.5,1)[0])

	q1x = p1x + deltaq1x
	q1y = p1y + deltaq1y
	q2x = p2x + deltaq2x
	q2y = p2y + deltaq2y
	q3x = p3x + deltaq3x
	q3y = p3y + deltaq3y
	q4x = p4x + deltaq4x
	q4y = p4y + deltaq4y
	
	height_new = np.maximum((q4y - q1y), (q3y - q2y))
	width_new = np.maximum((q2x-q1x), (q3x-q4x))
	pts1 = np.float32([ [p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y] ])
	pts2 = np.float32([ [q1x, q1y], [q2x, q2y], [q3x, q3y], [q4x, q4y] ])
	
	M = cv2.getPerspectiveTransform(pts1, pts2)
	warped_image = cv2.warpPerspective(input_image, M, (num_of_cols, num_of_rows))
	padded_center_x = int(num_of_cols / 2.0)
	padded_center_y = int(num_of_rows / 2.0)
	
	#return warped_image
	
	#now filling the black parts:
	SEPARATED_UP_AND_DOWN = False
	hsv_copy = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
	size = hsv_copy.shape
	
	size0 = np.hstack((np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1), size[0] - 1 - np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1)))
	size1 = np.hstack((np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1), size[1] - 1 - np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1)))
	
	"""
	Inefficient but complete filling:
	for i in range(size[0]):
		for j in range(size[1]):
			px = float(j)
			py = float(i)
			C1 = py < (+1.0)*(((q2y-q1y)/(q2x-q1x))*(px-q1x)+q1y +1)
			if np.sign(q1x - q4x) > 0:
				C2 = py <  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y+1) 
			else:
				C2 = py >  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y-1) 
			#C2 = np.sign(q1x - q4x) * py <  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y+1) 
			C3 = py > (+1.0)*(((q3y-q4y)/(q3x-q4x))*(px-q3x)+q3y -1)
			if np.sign(q2x-q3x) > 0:
				C4 = py > ( ((q2y-q3y)/(q2x-q3x))*(px-q2x) +q2y-1)
			else:
				C4 = py < ( ((q2y-q3y)/(q2x-q3x))*(px-q2x) +q2y+1)
			if C1 or C2 or C3 or C4:
				if SEPARATED_UP_AND_DOWN == False:
					hsv_copy[i,j,0] = H_VALUE
					hsv_copy[i,j,1] = S_VALUE
					hsv_copy[i,j,2] = V_VALUE
				else:
					if i > int(size[0]/2):
						hsv_copy[i,j,0] = H_VALUE_DOWN
						hsv_copy[i,j,1] = S_VALUE_DOWN
						hsv_copy[i,j,2] = V_VALUE_DOWN
					else:
						hsv_copy[i,j,0] = H_VALUE_UP
						hsv_copy[i,j,1] = S_VALUE_UP
						hsv_copy[i,j,2] = V_VALUE_UP
	"""
	for i in size0.astype('int'):
		for j in range(size[1]):
			px = float(j)
			py = float(i)
			C1 = py < (+1.0)*(((q2y-q1y)/(q2x-q1x))*(px-q1x)+q1y +1)
			if np.sign(q1x - q4x) > 0:
				C2 = py <  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y+1) 
			else:
				C2 = py >  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y-1) 
			#C2 = np.sign(q1x - q4x) * py <  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y+1) 
			C3 = py > (+1.0)*(((q3y-q4y)/(q3x-q4x))*(px-q3x)+q3y -1)
			if np.sign(q2x-q3x) > 0:
				C4 = py > ( ((q2y-q3y)/(q2x-q3x))*(px-q2x) +q2y-1)
			else:
				C4 = py < ( ((q2y-q3y)/(q2x-q3x))*(px-q2x) +q2y+1)
			if C1 or C2 or C3 or C4:
				if SEPARATED_UP_AND_DOWN == False:
					hsv_copy[i,j,0] = H_VALUE
					hsv_copy[i,j,1] = S_VALUE
					hsv_copy[i,j,2] = V_VALUE
				else:
					if i > int(size[0]/2):
						hsv_copy[i,j,0] = H_VALUE_DOWN
						hsv_copy[i,j,1] = S_VALUE_DOWN
						hsv_copy[i,j,2] = V_VALUE_DOWN
					else:
						hsv_copy[i,j,0] = H_VALUE_UP
						hsv_copy[i,j,1] = S_VALUE_UP
						hsv_copy[i,j,2] = V_VALUE_UP
	for i in range(size[0]):
		for j in size1.astype('int'):
			px = float(j)
			py = float(i)
			C1 = py < (+1.0)*(((q2y-q1y)/(q2x-q1x))*(px-q1x)+q1y +1)
			if np.sign(q1x - q4x) > 0:
				C2 = py <  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y+1) 
			else:
				C2 = py >  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y-1) 
			#C2 = np.sign(q1x - q4x) * py <  ( ((q4y-q1y)/(q4x-q1x))*(px-q4x) +q4y+1) 
			C3 = py > (+1.0)*(((q3y-q4y)/(q3x-q4x))*(px-q3x)+q3y -1)
			if np.sign(q2x-q3x) > 0:
				C4 = py > ( ((q2y-q3y)/(q2x-q3x))*(px-q2x) +q2y-1)
			else:
				C4 = py < ( ((q2y-q3y)/(q2x-q3x))*(px-q2x) +q2y+1)
			if C1 or C2 or C3 or C4:
				if SEPARATED_UP_AND_DOWN == False:
					hsv_copy[i,j,0] = H_VALUE
					hsv_copy[i,j,1] = S_VALUE
					hsv_copy[i,j,2] = V_VALUE
				else:
					if i > int(size[0]/2):
						hsv_copy[i,j,0] = H_VALUE_DOWN
						hsv_copy[i,j,1] = S_VALUE_DOWN
						hsv_copy[i,j,2] = V_VALUE_DOWN
					else:
						hsv_copy[i,j,0] = H_VALUE_UP
						hsv_copy[i,j,1] = S_VALUE_UP
						hsv_copy[i,j,2] = V_VALUE_UP
	
	warped_image = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)
	
	warped_image = warped_image[int(np.minimum(q1y, q2y)):int(np.maximum(q3y, q4y)), int(np.minimum(q1x, q4x)): int(np.maximum(q2x, q3x)),:]
	
	
	
	#pdb.set_trace()
	return warped_image

def da_rotation(input_image, angle, TRUNCATE):
	#obtaining the size and making a copy of it:
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	height = num_of_rows
	width = num_of_cols
	if TRUNCATE == False:
		M = cv2.getRotationMatrix2D(((num_of_cols / 2), (num_of_rows / 2)), angle, 1)
		rotated_image = cv2.warpAffine(input_image, M, (num_of_cols, num_of_rows))
		return rotated_image
	else:
		input_image = np.lib.pad(input_image, ((50,50),(50,50), (0,0)), 'constant')
		num_of_rows, num_of_cols, num_of_channels = input_image.shape
		M = cv2.getRotationMatrix2D(((num_of_cols / 2), (num_of_rows / 2)), angle, 1)
		rotated_image = cv2.warpAffine(input_image, M, (num_of_cols, num_of_rows))
		angle_r = angle * (3.141596/180.0)
		t_width = width - abs((2 * math.tan(angle_r)) * height)
		t_height = height - abs((2 * math.tan(angle_r)) * width)
		truncated_image = rotated_image[int((num_of_rows - t_height)/2.0):int((num_of_rows + t_height)/2.0), int((num_of_cols - t_width)/2.0):int((num_of_cols + t_width)/2.0), :]
		return truncated_image

	
def da_advanced_rotation(input_image, angle):
	SEPARATED_UP_AND_DOWN = False
	#obtaining the size and making a copy of it:
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	height = num_of_rows
	width = num_of_cols
	angle_r = angle * (np.pi / 180.0)
	#obtaining the color to fit the black margins with:
	#################################################################
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	margin_h = int(0.05*height)
	margin_w = int(0.05*width)
	h_values1 = hsv[0:margin_h, 0:margin_w, 0]
	h_values2 = hsv[0:margin_h, -1*margin_w:, 0]
	h_values3 = hsv[-1*margin_h:, 0:margin_w, 0]
	h_values4 = hsv[-1*margin_h:, -1*margin_w:, 0]
	h_values_stacked1 = np.hstack((h_values1, h_values2))
	h_values_stacked2 = np.hstack((h_values3, h_values4))
	h_values_stacked = np.hstack((h_values_stacked1, h_values_stacked2))
	H_VALUE = np.median(h_values_stacked.ravel())
	H_VALUE_UP = np.median(h_values_stacked1.ravel())
	H_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	s_values1 = hsv[0:margin_h, 0:margin_w, 1]
	s_values2 = hsv[0:margin_h, -1*margin_w:, 1]
	s_values3 = hsv[-1*margin_h:, 0:margin_w, 1]
	s_values4 = hsv[-1*margin_h:, -1*margin_w:, 1]
	s_values_stacked = np.hstack((s_values1, s_values2))
	s_values_stacked1 = np.hstack((s_values1, s_values2))
	s_values_stacked2 = np.hstack((s_values3, s_values4))
	s_values_stacked = np.hstack((s_values_stacked1, s_values_stacked2))
	S_VALUE = np.median(s_values_stacked.ravel())
	S_VALUE_UP = np.median(h_values_stacked1.ravel())
	S_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	v_values1 = hsv[0:margin_h, 0:margin_w, 2]
	v_values2 = hsv[0:margin_h, -1*margin_w:, 2]
	v_values3 = hsv[-1*margin_h:, 0:margin_w, 2]
	v_values4 = hsv[-1*margin_h:, -1*margin_w:, 2]
	v_values_stacked = np.hstack((v_values1, v_values2))
	v_values_stacked1 = np.hstack((v_values1, v_values2))
	v_values_stacked2 = np.hstack((v_values3, v_values4))
	v_values_stacked = np.hstack((v_values_stacked1, v_values_stacked2))
	V_VALUE = np.median(v_values_stacked.ravel())
	V_VALUE_UP = np.median(v_values_stacked1.ravel())
	V_VALUE_DOWN = np.median(v_values_stacked2.ravel())
	######################################################################
	#finding out the lines and a coordinates, to develope a classifier
	L = 0.5 * math.sqrt((width*width) + (height*height))
	beta = math.atan(float(height)/float(width))
	angle_r = -1.0 * angle_r
	Lcos = L * (math.cos(beta-angle_r))
	Lsin = L * (math.sin(beta-angle_r))
	p1x = -1.0 * L * (math.cos(beta-angle_r))
	p2x = -1.0 * L * (math.cos(beta+angle_r))
	p3x = L * (math.cos(beta-angle_r))
	p4x = L * (math.cos(beta+angle_r))
	y1 = L * (math.sin(beta-angle_r))
	y2 = -1.0 * L * (math.sin(beta+angle_r))
	y3 = -1.0 * L * (math.sin(beta-angle_r))
	y4 = L * (math.sin(beta+angle_r))
	input_image = np.lib.pad(input_image, ((50,50),(50,50), (0,0)), 'constant')
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	M = cv2.getRotationMatrix2D(((num_of_cols / 2), (num_of_rows / 2)), angle, 1)
	rotated_image = cv2.warpAffine(input_image, M, (num_of_cols, num_of_rows))
	
	#now we need to apply those median colors to black margins:
	hsv_copy = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
	size = hsv_copy.shape
	"""
	for i in range(size[0]):
		for j in range(size[1]):
			px = float(j - (size[1]/2.0))
			py = float(i - (size[0]/2.0))
			if angle_r >= 0:
				C1 = py > (+1.0)*((y4-y1)/(p4x-p1x))*(px-p1x)+y1 -1
				C2 = py > (+1.0)*((y3-y4)/(p3x-p4x))*(px-p4x)+y4 -1
				C3 = py < (+1.0)*((y2-y3)/(p2x-p3x))*(px-p3x)+y3 +1 
				C4 = py < (+1.0)*((y1-y2)/(p1x-p2x))*(px-p2x)+y2 +1
			else:
				C1 = py > (+1.0)*((y4-y1)/(p4x-p1x))*(px-p1x)+y1 -1
				C2 = py < (+1.0)*((y3-y4)/(p3x-p4x))*(px-p4x)+y4 +1
				C3 = py < (+1.0)*((y2-y3)/(p2x-p3x))*(px-p3x)+y3 +1
				C4 = py > (+1.0)*((y1-y2)/(p1x-p2x))*(px-p2x)+y2 -1
			if C1 or C2 or C3 or C4:
				if SEPARATED_UP_AND_DOWN == False:
					hsv_copy[i,j,0] = H_VALUE
					hsv_copy[i,j,1] = S_VALUE
					hsv_copy[i,j,2] = V_VALUE
				else:
					if i > int(size[0]/2):
						hsv_copy[i,j,0] = H_VALUE_DOWN
						hsv_copy[i,j,1] = S_VALUE_DOWN
						hsv_copy[i,j,2] = V_VALUE_DOWN
					else:
						hsv_copy[i,j,0] = H_VALUE_UP
						hsv_copy[i,j,1] = S_VALUE_UP
						hsv_copy[i,j,2] = V_VALUE_UP
	"""
	size0 = np.hstack((np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1), size[0] - 1 - np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1)))
	size1 = np.hstack((np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1), size[1] - 1 - np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1)))
	
	for i in size0.astype('int'):
		for j in range(size[1]):
			px = float(j - (size[1]/2.0))
			py = float(i - (size[0]/2.0))
			if angle_r >= 0:
				C1 = py > (+1.0)*((y4-y1)/(p4x-p1x))*(px-p1x)+y1 -1
				C2 = py > (+1.0)*((y3-y4)/(p3x-p4x))*(px-p4x)+y4 -1
				C3 = py < (+1.0)*((y2-y3)/(p2x-p3x))*(px-p3x)+y3 +1 
				C4 = py < (+1.0)*((y1-y2)/(p1x-p2x))*(px-p2x)+y2 +1
			else:
				C1 = py > (+1.0)*((y4-y1)/(p4x-p1x))*(px-p1x)+y1 -1
				C2 = py < (+1.0)*((y3-y4)/(p3x-p4x))*(px-p4x)+y4 +1
				C3 = py < (+1.0)*((y2-y3)/(p2x-p3x))*(px-p3x)+y3 +1
				C4 = py > (+1.0)*((y1-y2)/(p1x-p2x))*(px-p2x)+y2 -1
			if C1 or C2 or C3 or C4:
				if SEPARATED_UP_AND_DOWN == False:
					hsv_copy[i,j,0] = H_VALUE
					hsv_copy[i,j,1] = S_VALUE
					hsv_copy[i,j,2] = V_VALUE
				else:
					if i > int(size[0]/2):
						hsv_copy[i,j,0] = H_VALUE_DOWN
						hsv_copy[i,j,1] = S_VALUE_DOWN
						hsv_copy[i,j,2] = V_VALUE_DOWN
					else:
						hsv_copy[i,j,0] = H_VALUE_UP
						hsv_copy[i,j,1] = S_VALUE_UP
						hsv_copy[i,j,2] = V_VALUE_UP
						
	for i in range(size[0]):
		for j in size1.astype('int'):
			px = float(j - (size[1]/2.0))
			py = float(i - (size[0]/2.0))
			if angle_r >= 0:
				C1 = py > (+1.0)*((y4-y1)/(p4x-p1x))*(px-p1x)+y1 -1
				C2 = py > (+1.0)*((y3-y4)/(p3x-p4x))*(px-p4x)+y4 -1
				C3 = py < (+1.0)*((y2-y3)/(p2x-p3x))*(px-p3x)+y3 +1 
				C4 = py < (+1.0)*((y1-y2)/(p1x-p2x))*(px-p2x)+y2 +1
			else:
				C1 = py > (+1.0)*((y4-y1)/(p4x-p1x))*(px-p1x)+y1 -1
				C2 = py < (+1.0)*((y3-y4)/(p3x-p4x))*(px-p4x)+y4 +1
				C3 = py < (+1.0)*((y2-y3)/(p2x-p3x))*(px-p3x)+y3 +1
				C4 = py > (+1.0)*((y1-y2)/(p1x-p2x))*(px-p2x)+y2 -1
			if C1 or C2 or C3 or C4:
				if SEPARATED_UP_AND_DOWN == False:
					hsv_copy[i,j,0] = H_VALUE
					hsv_copy[i,j,1] = S_VALUE
					hsv_copy[i,j,2] = V_VALUE
				else:
					if i > int(size[0]/2):
						hsv_copy[i,j,0] = H_VALUE_DOWN
						hsv_copy[i,j,1] = S_VALUE_DOWN
						hsv_copy[i,j,2] = V_VALUE_DOWN
					else:
						hsv_copy[i,j,0] = H_VALUE_UP
						hsv_copy[i,j,1] = S_VALUE_UP
						hsv_copy[i,j,2] = V_VALUE_UP
	
	
	out = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)
	half_height = int(L * math.sin(abs(beta) + abs(angle_r)))+1
	half_width = int(L *math.cos(abs(beta)+abs(angle_r)))+1
	out = out[int(size[0]/2.0)-half_height:int(size[0]/2.0)+half_height, int(size[1]/2.0)-half_width:int(size[1]/2.0)+half_width, :]
	return out
	
	
	
	
	
	
	
	
	
	
	
	
	
def da_advanced_rotation_turbo(input_image, angle):
	if angle == 0:
		return input_image
	SEPARATED_UP_AND_DOWN = False
	#obtaining the size and making a copy of it:
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	height = num_of_rows
	width = num_of_cols
	angle_r = angle * (np.pi / 180.0)
	#obtaining the color to fit the black margins with:
	#################################################################
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	margin_h = int(0.05*height)
	margin_w = int(0.05*width)
	h_values1 = hsv[0:margin_h, 0:margin_w, 0]
	h_values2 = hsv[0:margin_h, -1*margin_w:, 0]
	h_values3 = hsv[-1*margin_h:, 0:margin_w, 0]
	h_values4 = hsv[-1*margin_h:, -1*margin_w:, 0]
	h_values_stacked1 = np.hstack((h_values1, h_values2))
	h_values_stacked2 = np.hstack((h_values3, h_values4))
	h_values_stacked = np.hstack((h_values_stacked1, h_values_stacked2))
	H_VALUE = np.median(h_values_stacked.ravel())
	H_VALUE_UP = np.median(h_values_stacked1.ravel())
	H_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	s_values1 = hsv[0:margin_h, 0:margin_w, 1]
	s_values2 = hsv[0:margin_h, -1*margin_w:, 1]
	s_values3 = hsv[-1*margin_h:, 0:margin_w, 1]
	s_values4 = hsv[-1*margin_h:, -1*margin_w:, 1]
	s_values_stacked = np.hstack((s_values1, s_values2))
	s_values_stacked1 = np.hstack((s_values1, s_values2))
	s_values_stacked2 = np.hstack((s_values3, s_values4))
	s_values_stacked = np.hstack((s_values_stacked1, s_values_stacked2))
	S_VALUE = np.median(s_values_stacked.ravel())
	S_VALUE_UP = np.median(h_values_stacked1.ravel())
	S_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	v_values1 = hsv[0:margin_h, 0:margin_w, 2]
	v_values2 = hsv[0:margin_h, -1*margin_w:, 2]
	v_values3 = hsv[-1*margin_h:, 0:margin_w, 2]
	v_values4 = hsv[-1*margin_h:, -1*margin_w:, 2]
	v_values_stacked = np.hstack((v_values1, v_values2))
	v_values_stacked1 = np.hstack((v_values1, v_values2))
	v_values_stacked2 = np.hstack((v_values3, v_values4))
	v_values_stacked = np.hstack((v_values_stacked1, v_values_stacked2))
	V_VALUE = np.median(v_values_stacked.ravel())
	V_VALUE_UP = np.median(v_values_stacked1.ravel())
	V_VALUE_DOWN = np.median(v_values_stacked2.ravel())
	######################################################################
	#finding out the lines and a coordinates, to develope a classifier
	L = 0.5 * math.sqrt((width*width) + (height*height))
	beta = math.atan(float(height)/float(width))
	angle_r = -1.0 * angle_r
	Lcos = L * (math.cos(beta-angle_r))
	Lsin = L * (math.sin(beta-angle_r))
	p1x = -1.0 * L * (math.cos(beta-angle_r))
	p2x = -1.0 * L * (math.cos(beta+angle_r))
	p3x = L * (math.cos(beta-angle_r))
	p4x = L * (math.cos(beta+angle_r))
	y1 = L * (math.sin(beta-angle_r))
	y2 = -1.0 * L * (math.sin(beta+angle_r))
	y3 = -1.0 * L * (math.sin(beta-angle_r))
	y4 = L * (math.sin(beta+angle_r))
	input_image = np.lib.pad(input_image, ((int(0.5*height), int(0.5*height)), (int(0.5*width), (int(0.5*width))), (0,0)), 'constant')
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	M = cv2.getRotationMatrix2D(((num_of_cols / 2), (num_of_rows / 2)), angle, 1)
	rotated_image = cv2.warpAffine(input_image, M, (num_of_cols, num_of_rows))
	
	#now we need to apply those median colors to black margins:
	hsv_copy = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
	size = hsv_copy.shape

	size0 = np.hstack((np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1), size[0] - 1 - np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1)))
	size1 = np.hstack((np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1), size[1] - 1 - np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1)))
	
	
	x_matrix = np.repeat( ( np.linspace(0, size[1]-1, size[1]).reshape(1, size[1]) - (size[1]/2.0) ), size[0], axis = 0)
	y_matrix = np.repeat( ( np.linspace(0, size[0]-1, size[0]).reshape(size[0], 1) - (size[0]/2.0) ), size[1], axis = 1)
	
	if angle_r >= 0:
		C1_matrix = y_matrix > ((y4-y1)/(p4x-p1x))*(x_matrix-p1x)+y1 -1
		C2_matrix = y_matrix > ((y3-y4)/(p3x-p4x))*(x_matrix-p4x)+y4 -1
		C3_matrix = y_matrix < ((y2-y3)/(p2x-p3x))*(x_matrix-p3x)+y3 +1 
		C4_matrix = y_matrix < ((y1-y2)/(p1x-p2x))*(x_matrix-p2x)+y2 +1
	else:
		C1_matrix = y_matrix > ((y4-y1)/(p4x-p1x))*(x_matrix-p1x)+y1 -1
		C2_matrix = y_matrix < ((y3-y4)/(p3x-p4x))*(x_matrix-p4x)+y4 +1
		C3_matrix = y_matrix < ((y2-y3)/(p2x-p3x))*(x_matrix-p3x)+y3 +1
		C4_matrix = y_matrix > ((y1-y2)/(p1x-p2x))*(x_matrix-p2x)+y2 -1
	
	
	
	C_matrix = C1_matrix + C2_matrix + C3_matrix + C4_matrix
	C_matrix = C_matrix.astype('bool')
	C_matrix = C_matrix.astype('float')
	hsv_copy[:,:,0] = (1-C_matrix) * hsv_copy[:,:,0] + C_matrix * H_VALUE
	hsv_copy[:,:,1] = (1-C_matrix) * hsv_copy[:,:,1] + C_matrix * S_VALUE
	hsv_copy[:,:,2] = (1-C_matrix) * hsv_copy[:,:,2] + C_matrix * V_VALUE
	
	out = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)
	half_height = int(L * math.sin(abs(beta) + abs(angle_r)))+1
	half_width = int(L *math.cos(abs(beta)+abs(angle_r)))+1
	#pdb.set_trace()
	out = out[int(size[0]/2.0)-half_height:int(size[0]/2.0)+half_height, int(size[1]/2.0)-half_width:int(size[1]/2.0)+half_width, :]
	return out
	
	
	
	
	
	
	
	
	
def da_random_projective_turbo(input_image):
	height, width, depth = input_image.shape
	
	#obtaining the color to fit the black margins with:
	#################################################################
	hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
	margin_h = int(0.05*height)
	margin_w = int(0.05*width)
	h_values1 = hsv[0:margin_h, 0:margin_w, 0]
	h_values2 = hsv[0:margin_h, -1*margin_w:, 0]
	h_values3 = hsv[-1*margin_h:, 0:margin_w, 0]
	h_values4 = hsv[-1*margin_h:, -1*margin_w:, 0]
	h_values_stacked1 = np.hstack((h_values1, h_values2))
	h_values_stacked2 = np.hstack((h_values3, h_values4))
	h_values_stacked = np.hstack((h_values_stacked1, h_values_stacked2))
	H_VALUE = np.median(h_values_stacked.ravel())
	H_VALUE_UP = np.median(h_values_stacked1.ravel())
	H_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	s_values1 = hsv[0:margin_h, 0:margin_w, 1]
	s_values2 = hsv[0:margin_h, -1*margin_w:, 1]
	s_values3 = hsv[-1*margin_h:, 0:margin_w, 1]
	s_values4 = hsv[-1*margin_h:, -1*margin_w:, 1]
	s_values_stacked = np.hstack((s_values1, s_values2))
	s_values_stacked1 = np.hstack((s_values1, s_values2))
	s_values_stacked2 = np.hstack((s_values3, s_values4))
	s_values_stacked = np.hstack((s_values_stacked1, s_values_stacked2))
	S_VALUE = np.median(s_values_stacked.ravel())
	S_VALUE_UP = np.median(h_values_stacked1.ravel())
	S_VALUE_DOWN = np.median(h_values_stacked2.ravel())
	v_values1 = hsv[0:margin_h, 0:margin_w, 2]
	v_values2 = hsv[0:margin_h, -1*margin_w:, 2]
	v_values3 = hsv[-1*margin_h:, 0:margin_w, 2]
	v_values4 = hsv[-1*margin_h:, -1*margin_w:, 2]
	v_values_stacked = np.hstack((v_values1, v_values2))
	v_values_stacked1 = np.hstack((v_values1, v_values2))
	v_values_stacked2 = np.hstack((v_values3, v_values4))
	v_values_stacked = np.hstack((v_values_stacked1, v_values_stacked2))
	V_VALUE = np.median(v_values_stacked.ravel())
	V_VALUE_UP = np.median(v_values_stacked1.ravel())
	V_VALUE_DOWN = np.median(v_values_stacked2.ravel())
	######################################################################
	
	input_image = np.lib.pad(input_image, ((int(0.5*height), int(0.5*height)), (int(0.5*width), (int(0.5*width))), (0,0)), 'constant')
	num_of_rows, num_of_cols, num_of_channels = input_image.shape
	center_row = int(num_of_rows / 2.0)
	center_col = int(num_of_cols / 2.0)
	#enumerating the corners clockwise, and obtaining the correspondences:
	p1x = center_col - int(width / 2.0)
	p1y = center_row - int(height / 2.0)
	p2x = center_col + int(width / 2.0)
	p2y = center_row - int(height / 2.0)
	p3x = center_col + int(width / 2.0)
	p3y = center_row + int(height / 2.0)
	p4x = center_col - int(width / 2.0)
	p4y = center_row + int(height / 2.0)
	
	offset_x = 0.05 * int(width)
	offset_y = 0.05 * int(height)
	
	deltaq1x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq1y = offset_y * (np.random.normal(0,0.5,1)[0])
	deltaq2x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq2y = offset_y * (np.random.normal(0,0.5,1)[0])
	deltaq3x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq3y = offset_y * (np.random.normal(0,0.5,1)[0])
	deltaq4x = offset_x * (np.random.normal(0,0.5,1)[0])
	deltaq4y = offset_y * (np.random.normal(0,0.5,1)[0])

	q1x = p1x + deltaq1x
	q1y = p1y + deltaq1y
	q2x = p2x + deltaq2x
	q2y = p2y + deltaq2y
	q3x = p3x + deltaq3x
	q3y = p3y + deltaq3y
	q4x = p4x + deltaq4x
	q4y = p4y + deltaq4y
	
	height_new = np.maximum((q4y - q1y), (q3y - q2y))
	width_new = np.maximum((q2x-q1x), (q3x-q4x))
	pts1 = np.float32([ [p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y] ])
	pts2 = np.float32([ [q1x, q1y], [q2x, q2y], [q3x, q3y], [q4x, q4y] ])
	
	M = cv2.getPerspectiveTransform(pts1, pts2)
	warped_image = cv2.warpPerspective(input_image, M, (num_of_cols, num_of_rows))
	padded_center_x = int(num_of_cols / 2.0)
	padded_center_y = int(num_of_rows / 2.0)
	
	#now filling the black parts:
	SEPARATED_UP_AND_DOWN = False
	hsv_copy = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
	size = hsv_copy.shape
	
	size0 = np.hstack((np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1), size[0] - 1 - np.linspace(0,int(0.2 * size[0]), int(0.2*size[0])+1)))
	size1 = np.hstack((np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1), size[1] - 1 - np.linspace(0,int(0.2*size[1]), int(0.2*size[1])+1)))
	x_matrix = np.repeat( ( np.linspace(0, size[1]-1, size[1]).reshape(1, size[1])), size[0], axis = 0)
	y_matrix = np.repeat( ( np.linspace(0, size[0]-1, size[0]).reshape(size[0], 1)), size[1], axis = 1)
	
	C1_matrix = y_matrix < (+1.0)*(((q2y-q1y)/(q2x-q1x))*(x_matrix-q1x)+q1y +1)
	if np.sign(q1x - q4x) > 0:
		C2_matrix = y_matrix <  ( ((q4y-q1y)/(q4x-q1x))*(x_matrix-q4x) +q4y+1) 
	else:
		C2_matrix = y_matrix >  ( ((q4y-q1y)/(q4x-q1x))*(x_matrix-q4x) +q4y-1)
	C3_matrix = y_matrix > (+1.0)*(((q3y-q4y)/(q3x-q4x))*(x_matrix-q3x)+q3y -1)
	if np.sign(q2x-q3x) > 0:
		C4_matrix = y_matrix > ( ((q2y-q3y)/(q2x-q3x))*(x_matrix-q2x) +q2y-1)
	else:
		C4_matrix = y_matrix < ( ((q2y-q3y)/(q2x-q3x))*(x_matrix-q2x) +q2y+1)
	C_matrix = C1_matrix + C2_matrix + C3_matrix + C4_matrix
	C_matrix = C_matrix.astype('bool')
	C_matrix = C_matrix.astype('float')
	hsv_copy[:,:,0] = (1-C_matrix) * hsv_copy[:,:,0] + C_matrix * H_VALUE
	hsv_copy[:,:,1] = (1-C_matrix) * hsv_copy[:,:,1] + C_matrix * S_VALUE
	hsv_copy[:,:,2] = (1-C_matrix) * hsv_copy[:,:,2] + C_matrix * V_VALUE
	warped_image = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)
	warped_image = warped_image[int(np.minimum(q1y, q2y)):int(np.maximum(q3y, q4y)), int(np.minimum(q1x, q4x)): int(np.maximum(q2x, q3x)),:]
	return warped_image

def read_myimage(index):
	return cv2.imread("data/" + "{:0>5d}".format(int(index+1)) + '.jpg',1)


	
	
	
	
	
	
"""
TESTS AND EVALUATION FUNCTIONS:
------------------------------------------------------------------------
"""
def test_t1(NUMBER_OF_CHANGES = 10, NUMBER_OF_IMAGES = 10):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - Adding an offset value to Hue')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_hsv_t1(input_image, -100 + (float(i)/float(NUMBER_OF_CHANGES)) * 200)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
			
def test_t2_s(NUMBER_OF_CHANGES = 10, NUMBER_OF_IMAGES = 10):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - Multiplying S by a coefficient')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_hsv_t2(input_image, 's', 0.5 + (float(i)/float(NUMBER_OF_CHANGES)) * 0.9)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
	
def test_t2_v(NUMBER_OF_CHANGES = 10, NUMBER_OF_IMAGES = 10):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - Multiplying V by a coefficient')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_hsv_t2(input_image, 'v', -0.5 + (float(i)/float(NUMBER_OF_CHANGES)) * 3.1)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()

def test_t3_s(NUMBER_OF_CHANGES = 20, NUMBER_OF_IMAGES = 10):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - Raising S to a power')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_hsv_t3(input_image, 's', 0.15 + (float(i)/float(NUMBER_OF_CHANGES)) * 7.3, 1)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
	
def test_t3_v(NUMBER_OF_CHANGES = 20, NUMBER_OF_IMAGES = 10):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - Raising v to a power')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_hsv_t3(input_image, 'v', 0.15 + (float(i)/float(NUMBER_OF_CHANGES)) * 7.3, 1)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
	
def test_rotation(NUMBER_OF_CHANGES = 10, NUMBER_OF_IMAGES = 10):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - Rotation')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_advanced_rotation_turbo(input_image, -5 + (float(i)/float(NUMBER_OF_CHANGES)) * 10)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()

			
def test_pca(NUMBER_OF_CHANGES = 5, NUMBER_OF_IMAGES = 2):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - FANCY PCA')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			pca = da_pca_model_finder(input_image)
			temp_image = da_fancy_pca(input_image, pca)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
			print "i: " + str(i) + " | " +"k: " + str(k)
			
def test_cov(NUMBER_OF_CHANGES = 5, NUMBER_OF_IMAGES = 2):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - FANCY PCA')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_fancy_cov_batch_pca(input_image)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
			print "i: " + str(i) + " | " +"k: " + str(k)
	
	
def test_spatial_transforms(NUMBER_OF_CHANGES = 5, NUMBER_OF_IMAGES = 2):
	#reading sample files:
	plt.ion()
	plt.figure()
	plt.suptitle('Data Augmentation - FANCY PCA')
	#HSV - TRANSFORMATION NUMBER 1:
	for k in range(NUMBER_OF_IMAGES):
		input_image = read_myimage(k)
		for i in range(NUMBER_OF_CHANGES):
			j = i + 1
			plt.subplot(NUMBER_OF_IMAGES, NUMBER_OF_CHANGES, j + k*NUMBER_OF_CHANGES)
			temp_image = da_random_projective_turbo(input_image)
			plt.axis("off")
			plt.imshow(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
			plt.show()
			print "i: " + str(i) + " | " +"k: " + str(k)

#embed()
			
