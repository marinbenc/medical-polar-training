# Todo

## Datasets

### Liver

Data obtained from LiTS - Liver Tumor Segmentation Challenge. 
Link: https://competitions.codalab.org/competitions/17094#participate

The project was trained on the training data, with a (101, 15, 15) train-test-valid split. Download the dataset and add it the scans as follows:

```
datasets/
  liver/
    scans/
      train/
        segmentation-100.nii
        volume-100.nii
        ...
      test/
        ...
      valid/
        ...
```

Then run `python datasets/liver/scans_to_images.py`.

### Polyp

Data obtained from CVC-ClinicDB.  
Link: https://polyp.grand-challenge.org/Databases/

We use the version from Kaggle since it's in color and uses PNG: https://www.kaggle.com/balraj98/cvcclinicdb

Download the dataset and add it as follows:

```
datasets/
  polyp/
    CVC-ClinicDB/
      Original/
        612.png
        ...
      Ground Truth/
        ...
```

Then run `python datasets/polyp/split_dataset.py`.

Citation:  
Bernal, J., Tajkbaksh, N., Sánchez, F.J., Matuszewski, B., Chen H., Yu, L., Angermann, Q., Romain, O., Rustad, B., Balasingham, I., Pogorelov, K., Choi, S., Debard, Q., Maier-Hein, L., Speidel, S., Stoyanov, D., Brandao, P., Cordova, H., Sánchez-Montes, C., Gurudu, S.R., Fernández-Esparrach, G., Dray, X.,  Liang, J. and Histace, A. "Comparative Validation of Polyp Detection Methods in Video Colonoscopy: Results from the MICCAI 2015 Endoscopic Vision Challenge", IEEE Transactions on Medical Imaging, 2017, Issue 99