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

Dataset citation:

Bernal, J., Tajkbaksh, N., Sánchez, F.J., Matuszewski, B., Chen H., Yu, L., Angermann, Q., Romain, O., Rustad, B., Balasingham, I., Pogorelov, K., Choi, S., Debard, Q., Maier-Hein, L., Speidel, S., Stoyanov, D., Brandao, P., Cordova, H., Sánchez-Montes, C., Gurudu, S.R., Fernández-Esparrach, G., Dray, X.,  Liang, J. and Histace, A. "Comparative Validation of Polyp Detection Methods in Video Colonoscopy: Results from the MICCAI 2015 Endoscopic Vision Challenge", IEEE Transactions on Medical Imaging, 2017, Issue 99

### Lesion

Dataset obtained from: https://challenge2018.isic-archive.com/task1/
Download link: https://challenge.isic-archive.com/data#2018

Download the validation and training input and GT for Task 1 and extract the folders as follows:

```
datasets/
  lesion/
    ISIC2018_Task1-2_Validation_Input/
    ISIC2018_Task1-2_Training_Input/
    ISIC2018_Task1_Validation_GroundTruth/
    ISIC2018_Task1_Training_GroundTruth/
```

Then, navigate to `datasets/lesion` and run `python make_dataset.py`.

Dataset citation:

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).