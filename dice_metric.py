from ignite.metrics import Metric
from helpers import dsc
import numpy as np

class DiceMetric(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu"):
      self._validation_pred = []
      self._validation_true = []
      super(DiceMetric, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
      self._validation_pred = []
      self._validation_true = []

    def update(self, output):
      y_pred, y = output[0].detach().cpu().numpy(), output[1].detach().cpu().numpy()
      self._validation_pred.extend(
        [y_pred[s] for s in range(y_pred.shape[0])])
      self._validation_true.extend(
        [y[s] for s in range(y.shape[0])])
    
    def compute(self):
      return np.mean([
        dsc(y_pred, y_true) 
        for (y_pred, y_true) 
        in zip(self._validation_pred, self._validation_true)])
