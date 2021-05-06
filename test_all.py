import os
import sys
import numpy as np

import train
import test
import test_center_from_model
import test_centerpoint_model
import helpers as h

weights_folder = 'logs'

models = train.model_choices[:1]
datasets = train.dataset_choices[:1]

class DisablePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Args(dict):
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_best_model(folder):
  models = h.listdir(folder)
  models = [m for m in  models if 'best_model_' in m]
  epochs = [int(m.split('_')[2]) for m in models]
  sorting = np.argsort(np.array(epochs))
  models = np.array(models)[sorting]
  return os.path.join(folder, models[-1])

def print_output(name, output):
  dsc, iou, prec, rec = output
  print(f'& {name} & {dsc:.4f} & {iou:.4f} & {prec:.4f} & {rec:.4f} \\\\')

for dataset in datasets:
  print('\n------\n')
  print(dataset)
  print('--')
  
  for model in models:

    # print(f' - model: {model}')
    # args = {
    #   'weights': get_best_model(f'logs/{dataset}_{model}_non_polar/'),
    #   'model': model,
    #   'dataset': dataset,
    #   'polar': False,
    # }
    # with DisablePrint():
    #   output = test.main(Args(args))
    # print_output('Cartesian', output)

    # args = {
    #   'weights': get_best_model(f'logs/{dataset}_{model}_polar_aug/'),
    #   'model': model,
    #   'dataset': dataset,
    #   'polar': True,
    # }
    # with DisablePrint():
    #   output = test.main(Args(args))
    # print_output('GT centers', output)

    # args = {
    #   'non_polar_weights': get_best_model(f'logs/{dataset}_{model}_non_polar/'),
    #   'polar_weights': get_best_model(f'logs/{dataset}_{model}_polar_aug/'),
    #   'model': model,
    #   'dataset': dataset,
    # }

    # with DisablePrint():
    #   output = test_center_from_model.main(Args(args))
    # print_output('Cart. centers', output)

    args = {
      'centerpoint_weights': get_best_model(f'logs/{dataset}_stacked_hourglass_sigma_8/'),
      'polar_weights': get_best_model(f'logs/{dataset}_{model}_polar_aug/'),
      'model': model,
      'dataset': dataset,
      'nstacks': 8,
    }

    with DisablePrint():
      output = test_centerpoint_model.main(Args(args))
    print_output('Model centers', output)
    print(' ---')
