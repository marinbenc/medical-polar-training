'''
Based on Stacked Hourglass Networks for Human Pose Estimation. 
Alejandro Newell, Kaiyu Yang, and Jia Deng. 
European Conference on Computer Vision (ECCV), 2016. Github

Adopted from PyTorch code by Chris Rockwell; 
based on: Associative Embedding: End-to-end Learning for Joint Detection and Grouping. 
Alejandro Newell, Zhiao Huang, and Jia Deng. 
Neural Information Processing Systems (NeurIPS), 2017. Github

source: https://github.com/princeton-vl/pytorch_stacked_hourglass/
'''

import torch
from torch import nn
from layers import Conv, Hourglass, Pool, Residual
import matplotlib.pyplot as plt

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        losses = torch.zeros(pred.shape[1])
        for i in range(pred.shape[1]):
            l = ((pred[:, i, :, :] - gt)**2)
            l = l.mean(dim=3).mean(dim=2).mean(dim=1)
            losses[i] = l.mean()
        return losses.mean()

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class StackedHourglass(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, in_channels=3, bn=False, increase=0, **kwargs):
        super(StackedHourglass, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(in_channels, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        x = self.pre(imgs)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        # result = torch.stack(combined_hm_preds, 1)
        # batch = result[0]
        # for img in batch:
        #     print(img.shape)
        #     plt.imshow(img.detach().cpu().numpy().squeeze())
        #     plt.show()

        return torch.stack(combined_hm_preds, 1)
