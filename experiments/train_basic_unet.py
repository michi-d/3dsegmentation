
import argparse
import os
import sys

from pathlib import Path
import utils
from utils.data import SegmentationFake2DDataset
from utils.misc import Hyperparameters
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np

from seg2d.unet import Unet
from seg2d.stackedhourglass import StackedUnet
from utils.misc import count_trainable_parameters

from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_title", type=str, default='my_experiment')
    parser.add_argument("--log_path", type=str, default='./test')
    parser.add_argument("--checkpoint_policy", type=str, default='best')

    parser.add_argument("--L_train", type=int, default='128')
    parser.add_argument("--L_validate", type=int, default='32')
    parser.add_argument("--L_test", type=int, default='32')

    parser.add_argument("--train_file", type=str, default='./train_dataset.h5')
    parser.add_argument("--validation_file", type=str, default='./validation_dataset.h5')
    parser.add_argument("--test_file", type=str, default='./test_dataset.h5')

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--arch", type=str, default='unet')
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--start_channels", type=int, default=4)
    parser.add_argument("--num_stacks", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--loss_type", type=str, default='dice')

    parser.add_argument("--start_lr", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.20)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    return args


def main():

    # get device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE: {DEVICE}')

    # get hyperparameters
    args = parse_args()
    hparams = Hyperparameters(**vars(args))
    hparams.add_params(DEVICE=DEVICE.type)

    # get data
    train_dataset = SegmentationFake2DDataset(L=hparams.L_train, seed=111, h5path=hparams.train_file)
    validation_dataset = SegmentationFake2DDataset(L=hparams.L_validate, seed=999, h5path=hparams.validation_file)
    test_dataset = SegmentationFake2DDataset(L=hparams.L_test, seed=2222, h5path=hparams.test_file)

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size)
    valid_loader = DataLoader(validation_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # build model
    if hparams.arch == 'unet':
        model = Unet(depth=hparams.depth, start_channels=hparams.start_channels, input_channels=1)
    elif hparams.arch == 'stacked_unet':
        model = StackedUnet(depth=hparams.depth, start_channels=hparams.start_channels, input_channels=1,
                            num_stacks=hparams.num_stacks)
    trainable_parameters = count_trainable_parameters(model)
    print(f'Trainable parameters: {trainable_parameters}')
    if trainable_parameters > 10e6:
        print(f'Number of parameters too high, aborting execution ....')
        sys.exit()

    # set loss function
    if hparams.loss_type == 'dice':
        loss = utils.losses.DiceLoss(activation='sigmoid')
    elif hparams.loss_type == 'bce':
        loss = torch.nn.BCEWithLogitsLoss()
    elif hparams.loss_type == 'weighted_bce':
        n_pos = (np.stack(train_dataset.masks[:1000], 0) == 1).sum()
        n_neg = (np.stack(train_dataset.masks[:1000], 0) == 0).sum()
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([n_neg / n_pos]))

    # set metrics
    metrics = [
        utils.metrics.IoU(threshold=0.5, activation='sigmoid'),
        utils.metrics.TotalError()
    ]

    # set optimizer
    if hparams.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=hparams.start_lr),
        ])
    elif hparams.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop([
            dict(params=model.parameters(), lr=hparams.start_lr),
        ])

    # set scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     factor=hparams.lr_scheduler_factor,
                                     patience=hparams.lr_scheduler_patience,
                                     verbose=True)

    # set logger
    script = os.path.abspath(__file__)
    log_path = Path(hparams.log_path) / hparams.experiment_title
    logger = utils.logger.Logger(str(log_path), model, title=hparams.experiment_title, script=script,
                                 checkpoint_policy=hparams.checkpoint_policy,
                                 eval_score_name='valid_total_error', eval_mode='min', verbose=True,
                                 resume=False)
    logger.save_hyperparameters(**hparams.as_dict())

    # prepare training
    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model
    for i in range(0, hparams.epochs):
        print('\nEpoch: {}'.format(i))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        lr_scheduler.step(valid_logs['loss'])

        logger.log_epoch(train_loss=train_logs['loss'],
                         valid_loss=valid_logs['loss'],
                         train_iou=train_logs['iou_score'],
                         valid_iou=valid_logs['iou_score'],
                         train_total_error=train_logs['total_error'],
                         valid_total_error=valid_logs['total_error'],
                         lr=optimizer.param_groups[0]['lr'])


if __name__ == '__main__':

    main()
