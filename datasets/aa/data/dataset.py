import os

import matplotlib.pyplot as plt
import torch
import torchio as tio
from torch.utils.data import DataLoader

def get_transforms(target_image):
  transform = tio.transforms.Compose([
    tio.ToCanonical(),
    tio.Resample(target=target_image),
    tio.RescaleIntensity(out_min_max=(-1, 1), in_min_max=(-200, 400))
  ])
  return transform

def get_data_loaders(prefix='./', patch_size=128, patches_per_scan=30, batch_size=8):
  data_folder = os.path.join(prefix, 'avt')
  all_files = os.listdir(data_folder)
  all_files.sort()

  label_files = [os.path.join(data_folder, f) for f in all_files if 'seg' in f]
  scan_files = [f.replace('seg.', '') for f in label_files]

  label_files_train = label_files[:40]
  scan_files_train = scan_files[:40]

  label_files_valid = label_files[40:]
  scan_files_valid = scan_files[40:]

  subjects_train = [tio.Subject(scan=tio.ScalarImage(s), label=tio.LabelMap(l)) for s, l in zip(label_files_train, scan_files_train)]

  target_image = subjects_train[0]['scan']
  dataset_train = tio.SubjectsDataset(subjects_train, transform=get_transforms(target_image))

  subjects_valid = [tio.Subject(scan=tio.ScalarImage(s), label=tio.LabelMap(l)) for s, l in zip(label_files_valid, scan_files_valid)]
  dataset_valid = tio.SubjectsDataset(subjects_valid, transform=get_transforms(target_image))

  datasets = [dataset_train, dataset_valid]

  # viz_slice = dataset[0]['scan'][tio.DATA][0][:, :, 128]
  # print(viz_slice.shape)
  # plt.imshow(viz_slice)
  # plt.show()

  sampler = tio.data.UniformSampler(patch_size)

  loaders = []

  for dataset in datasets:
    patches_queue = tio.Queue(
      dataset,
      max_length=60,
      samples_per_volume=patches_per_scan,
      sampler=sampler,
      num_workers=2,
    )

    patches_loader = DataLoader(
      patches_queue,
      batch_size=batch_size,
      num_workers=0,  # this must be 0
    )

    loaders.append(patches_loader)

  return loaders


if __name__ == '__main__':
  loader = get_data_loader()
  batch = iter(loader).next()
  scan = batch['scan'][tio.DATA][0]
  print(scan.shape)
  slice = scan[:, :, 32][0]
  plt.imshow(slice)
  plt.show()
