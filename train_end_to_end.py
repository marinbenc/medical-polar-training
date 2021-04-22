'''
Adapted from:
Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski,
Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm,
Computers in Biology and Medicine, Volume 109, 2019, Pages 218-225, ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2019.05.002.
https://github.com/mateuszbuda/brain-segmentation-pytorch
'''

import argparse
import json
import os
import sys
import datetime
from pprint import pprint

import segmentation_models_pytorch as smp
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient, MeanSquaredError
from ignite.contrib.metrics.regression import MedianAbsolutePercentageError
import albumentations as A

import helpers as h
from loss import DiceLoss
from helpers import dsc
from dice_metric import DiceMetric

sys.path.append('models')
from unet_plain import UNet
from polar_net import PolarNet
from stacked_hourglass import StackedHourglass, HeatmapLoss
from polar_dataset import PolarDataset
from heatmap_dataset import HeatmapDataset

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset
sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset
sys.path.append('datasets/lesion')
from lesion_dataset import LesionDataset
sys.path.append('datasets/eat')
from eat_dataset import EATDataset


dataset_choices = ['liver', 'polyp', 'lesion', 'eat']
model_choices = ['unet', 'resunetpp', 'deeplab']

def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    dataset_class = get_dataset_class(args)
    loader_train, loader_valid = data_loaders(args, dataset_class)

    model = get_model(args, dataset_class, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    heatmap_loss = HeatmapLoss()
    dice_loss = DiceLoss()

    def criterion(y_pred, y_true):
      heatmaps_pred, segmentation_pred = y_pred
      heatmaps_true, segmentation_true = y_true
      hl = heatmap_loss(heatmaps_pred, heatmaps_true)
      dsc = dice_loss(segmentation_pred, segmentation_true) 
      return hl * 1 + dsc * 1

    def output_transform_heatmap(data):
      y_pred, y_true = data
      heatmaps_pred, segmentation_pred = y_pred
      heatmaps_true, segmentation_true = y_true
      return (heatmaps_pred, heatmaps_true)

    def output_transform_segmentation(data):
      y_pred, y_true = data
      heatmaps_pred, segmentation_pred = y_pred
      heatmaps_true, segmentation_true = y_true
      return (segmentation_pred, segmentation_true)


    heatmap_metric = Loss(heatmap_loss, output_transform=output_transform_heatmap)
    segmentation_metric = Loss(dice_loss, output_transform=output_transform_segmentation)
    metrics = {
      'heatmap_loss': heatmap_metric,
      'dsc_loss': segmentation_metric,
      'loss': heatmap_metric + segmentation_metric
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger('Train Evaluator')
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    best_dsc = 0

    @trainer.on(Events.GET_BATCH_COMPLETED(once=1))
    def plot_batch(engine):
        x, y = engine.state.batch
        images = [x[0], y[0]]
        # for image in images:
        #     if image.shape[0] > 1:
        #         image = image.numpy()
        #         image = image.transpose(1, 2, 0)
        #         image += 0.5
        #     plt.imshow(image.squeeze())
        #     plt.show()

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        nonlocal best_dsc
        train_evaluator.run(loader_train)
        validation_evaluator.run(loader_valid)

    log_dir = f'logs/{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    tb_logger = TensorboardLogger(log_dir=log_dir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='training',
        output_transform=lambda loss: {'batchloss': loss},
        metric_names='all',
    )

    for tag, evaluator in [('training', train_evaluator), ('validation', validation_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names='all',
            global_step_transform=global_step_from_engine(trainer),
        )

    def score_function(engine):
        return -engine.state.metrics['loss']

    model_checkpoint = ModelCheckpoint(
        log_dir,
        n_saved=2,
        filename_prefix='best',
        score_function=score_function,
        score_name='loss',
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})

    trainer.run(loader_train, max_epochs=args.epochs)
    tb_logger.close()

def get_dataset_class(args):
    mapping = {
        'liver':    LiverDataset,
        'polyp':    PolypDataset,
        'lesion':   LesionDataset,
        'eat':      EATDataset,
    }
    return mapping[args.dataset]

def get_model(args, dataset_class, device):
    if args.model == 'unet':
        seg_model = UNet(
            in_channels=dataset_class.in_channels, 
            out_channels=dataset_class.out_channels, 
            device=device)
    elif args.model == 'resunetpp':
        seg_model = smp.UnetPlusPlus(
            in_channels=dataset_class.in_channels,
            classes=dataset_class.out_channels,
            encoder_weights=None,
            activation='sigmoid')
    elif args.model == 'deeplab':
        seg_model = smp.DeepLabV3Plus(
            in_channels=dataset_class.in_channels,
            classes=dataset_class.out_channels,
            activation='sigmoid')
    
    center_model = StackedHourglass(nstack=args.nstacks, inp_dim=256, oup_dim=1, in_channels=dataset_class.in_channels)
    model = PolarNet(center_model, seg_model)
    return model

def data_loaders(args, dataset_class):
    dataset_train, dataset_valid = datasets(args, dataset_class)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid

def datasets(args, dataset_class):

    train_heatmap = HeatmapDataset(
      dataset_name=args.dataset,
      directory='train',
      transform=A.Compose([
          A.HorizontalFlip(p=0.5),
          A.ShiftScaleRotate(p=0.3),
          A.GridDistortion(p=0.3),
      ]))

    train_seg = dataset_class(
      directory='train',
      polar=False,
      center_augmentation=False
    )
    train = PolarDataset(train_seg, train_heatmap)
    

    valid_heatmap = HeatmapDataset(
        dataset_name=args.dataset, 
        directory='valid')
    valid_seg = dataset_class(
      directory='valid',
      polar=False,
    )
    valid = PolarDataset(valid_seg, valid_heatmap)
    return train, valid

def makedirs(args):
    os.makedirs(args.logs, exist_ok=True)

def snapshotargs(args):
    args_file = os.path.join(args.logs, 'args.json')
    with open(args_file, 'w') as fp:
        json.dump(vars(args), fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--logs', type=str, default='./logs', help='folder to save logs'
    )
    parser.add_argument(
        '--model', type=str, choices=model_choices, default='unet', help='which model architecture to use'
    )
    parser.add_argument(
        '--dataset', type=str, choices=dataset_choices, default='liver', help='which dataset to use'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for data loading (default: 4)',
    )
    parser.add_argument(
      '--polar', 
      action='store_true',
      help='use polar coordinates')
    parser.add_argument(
        '--nstacks',
        type=int,
        default=8,
        help='number of hourglass stacks',
    )
    args = parser.parse_args()
    main(args)