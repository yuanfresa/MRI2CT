## helper.py

1. Get the path lists of MR and CT `lstMR, lstCT = getMRCTdir(path)`

2. Get the path lists of trained model and history `lstModel, lstHistory = getMRCTdir(path)`

3. Convert Dicom to Numpy.array `trueCT = dicom2array(lstCT)`

4. Crop the images to small patches, if `stride = 4` ,  One big 3D volume (40,128,128) results in 3645 smaller patches (24, 24, 24) 

   ```python
   crop_image= cropPatches(path, windowsize=(24,24,24), stride=4)
   ```

5. Zero mean and unit variance`std_patch = standardization(patch)`

6. Mean fusion of the patches `volume = fusePatches(patches)`

7. Create virtualCT`predict_CT = virtualCT(pathMR, model, CT_std, CT_mu)`

8. A simple way to visualize the vCT, true CT and MR

   `compareResult(MR, vCT, CT, savefig_dir)`

9. Calculate the Mean Absolute Error for several patients of one model

   ` MeanMAEforPatients = MAEforPrediction(model, lstMR, lstCT, CT_std, CT_mu)`