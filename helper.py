#!/usr/bin/python -tt
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import h5py
from numpy.lib.stride_tricks import as_strided
from skimage.util.shape import view_as_windows
from skimage.util import crop
import dicom
import os
## get the path list of MR and CT
def getMRCTdir(path):
    lstMR = []
    lstCT = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if '.dcm' in filename.lower(): 
                if 'MR' in filename:
                    lstMR.append(os.path.join(dirName,filename))
                    lstMR.sort()
                if 'CT' in filename:
                    lstCT.append(os.path.join(dirName,filename))
                    lstCT.sort()
    return lstMR, lstCT

# transform dicom to numpy.array
def load_pixel_data(path):
    ds = dicom.read_file(s)
    image= ds.pixel_array
    return image

def cropPatches(path, windowsize=(24,24,24),stride = 4):
	ds = dicom.read_file(path) # read medical data
	image= ds.pixel_array # to array
	#image = image.reshape((image.shape[1],image.shape[2],image.shape[0]))
	image_crop= view_as_windows(image,windowsize,stride) #crop the data to small patches
	shape = image_crop.shape
	newshape = (shape[0]*shape[1]*shape[2],windowsize[0],windowsize[1],windowsize[2])
	image_crop = image_crop.reshape(newshape)
	return image_crop,image

# Crop MR and CT to different sizes	
def cropCTPatches(CT_path, MR_windowsize=(24,24,24), CT_windowsize=(4,4,4), stride = 4):
    ds_CT = dicom.read_file(CT_path) # read medical data
    image_CT= ds_CT.pixel_array # to array
    diff = np.subtract(MR_windowsize,CT_windowsize)/2
    image_CT = crop(image_CT,(diff[0], ), (diff[1], ), (diff[2], ))
    image_crop_CT= view_as_windows(image_CT,CT_windowsize,stride) #crop the data to small patches   
    shape_CT = image_crop_CT.shape
    newshape_CT = (shape_CT[0]*shape_CT[1]*shape_CT[2],CT_windowsize[0],CT_windowsize[1],CT_windowsize[2])
    image_crop_CT = image_crop_CT.reshape(newshape_CT)
    return image_crop_CT

#zero mean and unit variance
def normalization(patch):
	deviation = np.std(patch)
	mu = np.mean(patch)
	patch = (patch-mu)/deviation
	patch = patch.reshape((patch.shape+(1,)))
	return patch, mu, deviation

def fuseCTPatches(patches, step = (4,4,4), out_shape = (40,128,128)):
	out = np.zeros(out_shape, patches.dtype)
	denom = np.zeros(out_shape, patches.dtype)
	patch_shape = patches.shape[-3:]
	#as_strided(x, shape=None, strides=None)
	#Make an ndarray from the given array with the given shape and strides
	patches_6D = as_strided(out, (((out.shape[0] - patch_shape[0]) // step[0] + 1),
								  ((out.shape[1] - patch_shape[1]) // step[1] + 1),
								  ((out.shape[2] - patch_shape[2]) // step[2] + 1),
								  patch_shape[0], patch_shape[1], patch_shape[2]), #shape
						   (out.strides[0] * step[0], out.strides[1] * step[1], out.strides[2] * step[2],
							out.strides[0], out.strides[1],out.strides[2]))
	denom_6D = as_strided(denom, (((denom.shape[0] - patch_shape[0]) // step[0] + 1),
								  ((denom.shape[1] - patch_shape[1]) // step[1] + 1),
								  ((denom.shape[2] - patch_shape[2]) // step[2] + 1),
								  patch_shape[0], patch_shape[1], patch_shape[2]),
						 (denom.strides[0] * step[0], denom.strides[1] * step[1],denom.strides[2] * step[2],
						  denom.strides[0], denom.strides[1],denom.strides[2]))
	np.add.at(patches_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), patches.ravel())
	np.add.at(denom_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), 1)
	return out/denom
def trueCT(path):
	ds = dicom.read_file(path) # read medical data
	image= ds.pixel_array # to array
	return image

def virtualCT(pathMR,model,CT_std,CT_mu):
	# crop MR to patches
	MR_patch,MR_for_predict = cropMRPatches(pathMR)
	# normalize the data
	MR_patch_nor,_,_ = normalization(MR_patch)
	# predict
	predict_CT_patch = model.predict(MR_patch_nor)
	# reshape
	predict_CT_patch = predict_CT_patch.reshape(predict_CT_patch.shape[:-1])
	# convert to CT value
	predict_CT_patch = predict_CT_patch*CT_std+CT_mu
	# fuse the patches
	predict_CT = fuseCTPatches(predict_CT_patch)
	return MR_for_predict, predict_CT