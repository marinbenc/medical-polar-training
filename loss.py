import torch.nn as nn
import torch
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

class CenterPointLoss(nn.Module):

    def __init__(self):
        super(CenterPointLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        
        y_centers = torch.zeros((y_pred.shape[0], 2), dtype=torch.float)
        y_pred_centers = torch.zeros((y_pred.shape[0], 2), dtype=torch.float)

        for i in range(y_pred.shape[0]):
            y_centers[i] = torch.Tensor(centroid(y_true[i].detach().cpu().numpy().squeeze()))
            y_pred_centers[i] = torch.Tensor(centroid(y_pred[i].detach().cpu().numpy().squeeze()))

        mse = mse_loss(y_pred_centers, y_centers)
        dsc = self.dice_loss(y_pred, y_true)

        return dsc + mse

