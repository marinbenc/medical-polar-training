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

sys.path.append('models/stacked_hourglass')
from stacked_hourglass import StackedHourglass, HeatmapLoss
from heatmap_dataset import HeatmapDataset

from helpers import dsc
from dice_metric import DiceMetric

from train import dataset_choices, makedirs, snapshotargs, get_dataset_class

def main(args):
    args.polar = False

    makedirs(args)
    snapshotargs(args)
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    dataset_class = get_dataset_class(args)
    loader_train, loader_valid = data_loaders(args)

    model = get_model(args, dataset_class, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HeatmapLoss()

    metrics = {
      'loss': Loss(criterion)
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger('Train Evaluator')
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
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
        score_name='dsc',
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})

    trainer.run(loader_train, max_epochs=args.epochs)

    tb_logger.close()

def get_model(args, dataset_class, device):
    return StackedHourglass(nstack=args.nstacks, inp_dim=256, oup_dim=1, in_channels=dataset_class.in_channels)

def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

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

def datasets(args):
    train = HeatmapDataset(
        dataset_name=args.dataset,
        directory='train',
        transform=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.3),
            A.GridDistortion(p=0.3),
        ]))
    valid = HeatmapDataset(
        dataset_name=args.dataset, 
        directory='valid')

    return train, valid

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
        '--dataset', type=str, choices=dataset_choices, default='liver', help='which dataset to use'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for data loading (default: 4)',
    )
    parser.add_argument(
        '--nstacks',
        type=int,
        default=8,
        help='number of hourglass stacks',
    )
    args = parser.parse_args()
    main(args)