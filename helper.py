#!/usr/bin/python -tt
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dicom
import h5py
import os
from six.moves import cPickle as pickle
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided


def getMRCTdir(path):
	"""
	Get the path list of MR and CT
	Return croped patches and original image
	:param path: path of the folder containing MR and CT dicom
	:return: lstMR(list of path for MR), lstCT(list of path for CT)
	"""
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

def getModelHistorydir(path):
	"""
	Get the path list of trained model and history
	Return croped patches and original image
	:param path: path of the folder containing MR and CT dicom
	:return: lstMR(list of path for MR), lstCT(list of path for CT)
	"""
	lstModel =[]
	lstHistory =[]
	for dirName, subdirList, fileList in os.walk(Path):
		for filename in fileList:
			if "model" in filename.lower():
				lstModel.append(os.path.join(dirName,filename))
			if "history" in filename.lower():
            lstHistory.append(os.path.join(dirName,filename))
    return lstModel, lstHistory

def dicom2array(path):
	ds = dicom.read_file(path) # read medical data
	image= ds.pixel_array # to array
	return image

def cropPatches(path, windowsize=(24,24,24),stride = 4):
	"""
	:param path: path the dicom 
	:return: image_crop(croped patches)
	"""
	ds = dicom.read_file(path) # read medical data
	image= ds.pixel_array # to array
	image_crop= view_as_windows(image,windowsize,stride) #crop the data to small patches
	shape = image_crop.shape
	newshape = (shape[0]*shape[1]*shape[2],windowsize[0],windowsize[1],windowsize[2])
	image_crop = image_crop.reshape(newshape)
	return image_crop


def standardization(patch):
	"""
	return the standardized data (zero mean and unit variance) and reshape to one channel
	data=format = "channels_last"
	"""
	deviation = np.std(patch)
	mu = np.mean(patch)
	patch = (patch-mu)/deviation
	patch = patch.reshape((patch.shape+(1,))).astype('float32')
	return patch, mu, deviation

def fusePatches(patches, step = (4,4,4), out_shape = (40,128,128)):
	"""
	return the mean fusion volume
	"""
	out = np.zeros(out_shape, patches.dtype)
	denom = np.zeros(out_shape, patches.dtype)
	patch_shape = patches.shape[-3:]
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

def virtualCT(pathMR,model,CT_std,CT_mu):
	# crop MR to patches
	MR_patch = cropPatches(pathMR)
	# normalize the data
	MR_patch_nor = standardization(MR_patch)[0]
	# predict
	predict_CT_patch = model.predict(MR_patch_nor)
	# reshape
	predict_CT_patch = predict_CT_patch.reshape(predict_CT_patch.shape[:-1])
	# convert to CT value
	predict_CT_patch = predict_CT_patch*CT_std+CT_mu
	# fuse the patches
	predict_CT = fuseCTPatches(predict_CT_patch)
	return predict_CT
	
def compareResult(MR, vCT, CT, patientid, rows=3, cols=6, startwith=2, savefig_dir):
	norm_ct = mcolors.Normalize(vmax = np.amax(np.maximum(vCT, CT)), 
                                vmin = np.amin(np.minimum(vCT, CT)))
	norm_mr = mcolors.Normalize(vmax = np.amax(MR), vmin = np.amin(MR))
    fig,ax = plt.subplots(rows,cols, figsize = [24,24])
    for i in range(cols):
        step = (MR.shape[0]-startwith)/cols
        ind = startwith + i*step
        ax[0,i].set_title('MR s%d' % ind)
        ax[0,i].imshow(MR[ind,:,:],cmap='gray', norm = norm_mr)
        ax[0,i].axis('off')
        ax[1,i].set_title('vCT s%d' % ind)
        ax[1,i].imshow(vCT[ind,:,:],cmap='gray', norm = norm_ct)
        ax[1,i].axis('off')
        ax[2,i].set_title('CT s%d' % ind)
        ax[2,i].imshow(CT[ind,:,:],cmap='gray', norm = norm_ct)
        ax[2,i].axis('off')
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    fig.suptitle('patient ' + str(patientid), fontsize=30, fontweight='bold')
    figname = os.path.join(savefig_dir, 'patient_%d.png' %(patientid))
    plt.savefig(figname)

# def MAEafterFusion(vCT, true_CT):
#     mae = np.mean(np.absolute((vCT.astype("float") - true_CT.astype("float"))))
#     return mae
def MAEforPrediction(model, lstMR, lstCT, CT_std, CT_mu):
	MAE = []
	# prediction for patients
	for i, pathMR in enumerate(lstMR):
		vCT = virtualCT(pathMR, model, CT_std, CT_mu)
		MR = dicom2array(pathMR)
		true_CT = dicom2array(CT_path[i])
		mae = np.mean(np.absolute((vCT.astype("float") - true_CT.astype("float"))))
		MAE.append(mae)
	return np.mean(MAE)

# #save the patches
# pickle_file = '../Data/Exp14_128/MRICT_patch_%d_%d.pickle' %(cropMR.shape[1], cropCT.shape[1])
# try:
#     f = open(pickle_file,'wb')
#     save = {
#         'MRI': cropMR,
#         'CT': cropCT,
#         'num_patients':len(lstMR),
#     }
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#     f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)