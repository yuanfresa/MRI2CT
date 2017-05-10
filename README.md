## helper.py

1. Get the path lists of MR and CT `lstMR, lstCT = getMRCTdir(path)`

2. Get the path lists of trained model and history `lstModel, lstHistory = getMRCTdir(path)`

3. Convert Dicom to Numpy.array `trueCT = dicom2array(lstCT)`

4. Crop the images to small patches, if `stride = 4` ,  One big 3D volume (40,128,128) results in 3645 smaller patches (24, 24, 24) 

   ```python
   crop_image= cropPatches(path, windowsize=(24,24,24), stride=4)
   ```

5. Zero mean and unit variance`patch,mu,std = standardization(patch)`

6. Any preprocessing statistics  **must only be computed on the training data**, and then applied to the validation / test data. Returned the normalized pathes of train and test data.

   `train, test= normalize_train_test(patch_train, patch_test)`

7. Mean fusion of the patches `volume = fusePatches(patches)`

8. Create virtualCT`predict_CT = virtualCT(pathMR, model, CT_std, CT_mu, MR_std, MR_mu)`

  Standard deviation and mean are from training set to normalize the testing data

## model.py

1. Base model, 9 (conv3d+BN+ReLU) layers, "he initialization"  `model_FCN9_base`
2. `model_FCN9_xavier` change the initialization of base model to xavier
3. `model_FCN9_BNafterReLU` change the order of BN and ReLU