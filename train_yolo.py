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
import torchvision
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ignite.engine import Engine
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient, MeanSquaredError, Average
from ignite.contrib.metrics.regression import MedianAbsolutePercentageError

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import helpers as h
from loss import DiceLoss, CenterPointLoss

sys.path.append('models')
from unet_plain import UNet

sys.path.append('datasets/liver')
from liver_dataset import LiverDataset
sys.path.append('datasets/polyp')
from polyp_dataset import PolypDataset

from helpers import dsc
from dice_metric import DiceMetric
from bbox_dataset import BBOXDataset

dataset_choices = ['liver', 'polyp']
model_choices = ['unet', 'resunetpp', 'deeplab']

def get_iou(bb1, bb2):

    bb1 = { 'x1': bb1[0], 'y1': bb1[1], 'x2': bb1[2], 'y2': bb1[3] }
    bb2 = { 'x1': bb2[0], 'y1': bb2[1], 'x2': bb2[2], 'y2': bb2[3] }

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    if args.dataset == 'liver':
        dataset_class = LiverDataset
    elif args.dataset == 'polyp':
        dataset_class = PolypDataset

    loader_train, loader_valid = data_loaders(args, dataset_class)

    model = get_model(args, dataset_class, device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def evaluate(engine, batch):
        with torch.no_grad():
            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            model.eval()

            preds = model(images)
            ious = torch.zeros(len(images))

            for i, (pred, target) in enumerate(zip(preds, targets)):
                if len(pred['boxes']) <= 0:
                    iou = 0
                else:
                    iou = get_iou(pred['boxes'][0], target['boxes'][0])
                ious[i] = iou
            
            return { 'loss': losses.item(), 'iou': ious.mean().item() }

    def train_one_epoch(engine, batch):
        model.train()

        images, targets = batch
        images = list(image.to(device).detach() for image in images)
        targets = [{k: v.to(device).detach() for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        del images
        
        return losses.item()

    trainer = Engine(train_one_epoch)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = Engine(evaluate)
    train_evaluator.logger = setup_logger('Train Evaluator')

    validation_evaluator =  Engine(evaluate)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    avg_loss = Average(output_transform=lambda output: output['loss'])
    avg_loss.attach(validation_evaluator, 'loss')

    avg_iou = Average(output_transform=lambda output: output['iou'])
    avg_iou.attach(validation_evaluator, 'iou')
    avg_iou.attach(train_evaluator, 'iou')

    best_dsc = 0

    # @trainer.on(Events.GET_BATCH_COMPLETED(once=1))
    # def plot_batch(engine):
    #     x, y = engine.state.batch

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        nonlocal best_dsc
        train_evaluator.run(loader_train)
        validation_evaluator.run(loader_valid)

        # curr_dsc = validation_evaluator.state.metrics['dsc']
        # if curr_dsc > best_dsc:
        #     best_dsc = curr_dsc

    log_dir = f'logs/{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    tb_logger = TensorboardLogger(log_dir=log_dir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='training',
        output_transform=lambda loss: {'loss': loss},
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
        return engine.state.metrics['iou']

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

    return best_dsc

    print(f'Mean CV DSC: {mean_dsc:.4f}')

def get_model(args, dataset_class, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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
        collate_fn=lambda batch: list(zip(*batch))
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        collate_fn=lambda batch: list(zip(*batch))
    )

    return loader_train, loader_valid

def datasets(args, dataset_class):
    train = dataset_class(
      directory='train',
      polar=args.polar
    )
    valid = dataset_class(
      directory='valid',
      polar=args.polar
    )
    return BBOXDataset(train), BBOXDataset(valid)

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
    args = parser.parse_args()
    main(args)