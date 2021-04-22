import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from polar_transformations import centroid
from torch.nn.functional import mse_loss

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        dscs = torch.zeros(y_pred.shape[1])

        for i in range(y_pred.shape[1]):
          y_pred_ch = y_pred[:, i].contiguous().view(-1)
          y_true_ch = y_true[:, i].contiguous().view(-1)
          intersection = (y_pred_ch * y_true_ch).sum()
          dscs[i] = (2. * intersection + self.smooth) / (
              y_pred_ch.sum() + y_true_ch.sum() + self.smooth
          )
        return 1. - torch.mean(dscs)
