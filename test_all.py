weights_folder = 'logs'

import train
import test

models = train.model_choices
datasets = train.dataset_choices

def get_best_model(folder):
  models = h.listdir(folder)
  models.sort()
  print(models)
  return os.path.join(folder, models[-2])

def print_output(name, output):
  dsc, iou, prec, rec = output
  f'{name} & {dsc:.4f} & {iou:.4f} & {prec:.4f} & {rec:.4f}'

for dataset in datasets:
  print('\n------\n')
  print(dataset)
  print('--')
  
  for model in models:

    print(f' - model: {model}')
    args = {
      'weights': best_model(f'logs/{dataset}_{model}_non_polar/'),
      'model': model,
      'dataset': dataset,
      'polar': False,
    }
    print_output('Cartesian', test.main(args))

    args = {
      'weights': best_model(f'logs/{dataset}_{model}_polar_aug/'),
      'model': model,
      'dataset': dataset,
      'polar': True,
    }
    print_output('Polar GT centers', test.main(args))

    args = {
      'non_polar_weights': best_model(f'logs/{dataset}_{model}_non_polar/'),
      'polar_weights': best_model(f'logs/{dataset}_{model}_polar_aug/'),
      'model': model,
      'dataset': dataset,
    }
    print_output('Cart. centers', test.main(args))

    # TODO:
    args = {
      'non_polar_weights': best_model(f'logs/{dataset}_{model}_non_polar/'),
      'polar_weights': best_model(f'logs/{dataset}_{model}_polar_aug/'),
      'model': model,
      'dataset': dataset,
    }
    print_output('Model centers', test.main(args))
    print(' ---')


  args = ['']
