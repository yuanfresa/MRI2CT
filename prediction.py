#!/usr/bin/python -tt
# Yuan Zhou
from __future__ import print_function
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from loss_function import combine_loss,gdl_loss
import helper

if __name__ == '__main__':

    PathDicom = '../Data/14patient_Nonrigid/128x128x40'
    lstMR,lstCT = helper.getMRCTdir(PathDicom)


    # Load Model list and history list
    PathModel = '../Models/model_0503/combine_loss'
    lstModel, lstHistory = helper.getModelHistorydir(PathModel)

    # Load trained model

    model = load_model(lstModel[0], custom_objects={'combine_loss': combine_loss})
    # model = load_model(lstModel[0])
    # Load the training sets mean and std
    with open('../Data/Exp14_128/non-rigid/MRICT_7folds_mean_std.pickle', "rb") as f:
        data = pickle.load(f)

    CT_mean = data['CT_mean'][0]
    CT_std = data['CT_std'][0]
    del data

    true_CT = np.stack([helper.dicom2array(s) for s in lstCT ])
    virtual_CT = np.stack([helper.virtualCT(s, model, CT_mean, CT_std) for s in lstMR ])
    error_CT = np.abs(np.subtract(true_CT, virtual_CT))
    vCT_file = '../Data/Exp14_128/non-rigid/combine_loss_fold1_prediction.hdf5'
    try:
        with h5py.File(vCT_file,'w') as f:
            f.create_dataset("virtual_CT", data = virtual_CT)
            f.create_dataset("error_CT", data = error_CT)
            f.close()
    except Exception as e:
        print('Unable to save data to', vCT_file, ':', e)
