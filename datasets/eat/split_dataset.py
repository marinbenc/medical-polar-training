import sys
import os.path as p

import numpy as np
import cv2 as cv

sys.path.append('../../')
import helpers as h

labels_folder = 'data/label'
inputs_folder = 'data/input'

train, valid, test = (10, 5, 5)
patients = np.array(h.listdir(labels_folder))
patients.sort()

np.random.seed(42)
np.random.shuffle(patients)

train_patients = patients[:train]
valid_patients = patients[train:train + valid]
test_patients = patients[-test:]

assert(len(train_patients) + len(valid_patients) + len(test_patients) == len(patients))

folders = {
  'train': train_patients,
  'valid': valid_patients,
  'test': test_patients
}

print(folders)

for (folder, folder_patients) in folders.items():
  h.mkdir(p.join(folder, 'input'))
  h.mkdir(p.join(folder, 'label'))

  for patient in folder_patients:
    patient_labels = h.listdir(p.join(labels_folder, patient))
    patient_labels.sort()

    for label_file in patient_labels:
      print(f'{patient}_{label_file}')
      label = cv.imread(p.join(labels_folder, patient, label_file), cv.IMREAD_GRAYSCALE)
      input = cv.imread(p.join(inputs_folder, patient, label_file), cv.IMREAD_GRAYSCALE)

      input = cv.resize(input, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
      label = cv.resize(label, dsize=(128, 128), interpolation=cv.INTER_CUBIC)

      file_name = f'{patient}_{label_file}'
      cv.imwrite(p.join(folder, 'input', file_name), input)
      cv.imwrite(p.join(folder, 'label', file_name), label)



