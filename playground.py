import numpy as np
from helpers import dsc
import sys
import matplotlib.pyplot as plt
from dice_metric import DiceMetric
from loss import DiceLoss

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset

dataset = LiverDataset('datasets/liver/valid')
image, label = dataset[29]
_, label_2 = dataset[31]
_, label_3 = dataset[14]

# plt.imshow(label.squeeze())
# plt.show()
# plt.imshow(label_2.squeeze())
# plt.show()

metric = DiceMetric()
metric.update((label, label_2))
print(metric.compute())

loss = DiceLoss()
print(loss.forward(label, label_2).item())


label = label.detach().cpu().numpy()
label_2 = label_2.detach().cpu().numpy()
label_3 = label_3.detach().cpu().numpy()


print(np.mean([dsc(label, label_2), dsc(label, label_3)]))