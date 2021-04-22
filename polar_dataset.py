import torch
from torch.utils.data import Dataset

class PolarDataset(Dataset):
  def __init__(self, segmentation_dataset, heatmap_dataset):
    self.segmentation_dataset = segmentation_dataset
    self.heatmap_dataset = heatmap_dataset

  def __len__(self):
    return 32
    return self.heatmap_dataset.__len__()

  def __getitem__(self, idx):
    input_segmentation, segmentation = self.segmentation_dataset.__getitem__(idx)
    input_heatmap, heatmap = self.heatmap_dataset.__getitem__(idx)
    return (input_heatmap, input_segmentation), (heatmap, segmentation)

