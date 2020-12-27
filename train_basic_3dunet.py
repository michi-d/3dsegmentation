
import argparse
import os
import sys

from pathlib import Path
import utils

from utils.misc import Hyperparameters
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
import traceback

from seg3d.unet import Unet
from seg3d.data import Fake3DDataset
from seg3d.stackedhourglass import StackedUnet
from utils.misc import count_trainable_parameters

from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_title", type=str, default='my_experiment')
    parser.add_argument("--log_path", type=str, default='./test')
    parser.add_argument("--checkpoint_policy", type=str, default='best')

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--arch", type=str, default='unet')
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--start_channels", type=int, default=4)
    parser.add_argument("--num_stacks", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--loss_type", type=str, default='dice')
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--start_lr", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.20)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15)

    parser.add_argument("--MAX_PARAMS", type=int, default=30e6)
    parser.add_argument("--MIN_PARAMS", type=int, default=0e6)

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
    print(f'BATCH SIZE: {hparams.batch_size}')
  
    # check if already exists
    script = os.path.abspath(__file__)
    log_path = Path(hparams.log_path) / hparams.experiment_title
    if log_path.is_dir():
        print(f'Experiment already done previously.')
        sys.exit()    

    # get data
    train_files = ['vol_data/vol_train_set1.h5',
                   'vol_data/vol_train_set2.h5',
                   'vol_data/vol_train_set3.h5',
                   'vol_data/vol_train_set4.h5']
    val_files = ['vol_data/vol_val_set.h5']

    train_dataset = Fake3DDataset(h5_files=train_files)
    validation_dataset = Fake3DDataset(h5_files=val_files)

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size)
    valid_loader = DataLoader(validation_dataset, batch_size=1)

    # build model
    if hparams.arch == 'unet':
        model = Unet(depth=hparams.depth, start_channels=hparams.start_channels, input_channels=1)
    elif hparams.arch == 'stacked_unet':
        print('Not yet implemented.')
        model = StackedUnet(depth=hparams.depth, start_channels=hparams.start_channels, input_channels=1,
                            num_stacks=hparams.num_stacks)
    trainable_parameters = count_trainable_parameters(model)
    print(f'Trainable parameters: {trainable_parameters}')
    if (trainable_parameters >= hparams.MAX_PARAMS) or (trainable_parameters <= hparams.MIN_PARAMS):
        print(f'Number of parameters too high or too low, aborting execution ....')
        sys.exit()

    # set loss function
    if hparams.loss_type == 'dice':
        loss = utils.losses.DiceLoss(activation='sigmoid')
    elif hparams.loss_type == 'bce':
        loss = torch.nn.BCEWithLogitsLoss()
    elif hparams.loss_type == 'weighted_bce':
        # get fraction of positive labels from train set
        n_pos = 0
        for i in range(100):
            vol_data, vol_labels = train_dataset[i]
            n_pos += (vol_labels == 1).sum()
        frac_pos = n_pos / vol_labels.numel() / 100
        print(f'Fraction of positive labels: {frac_pos:.2f}')
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=(1-frac_pos)/frac_pos)

    # set metrics
    metrics = [
        utils.metrics.IoU(threshold=0.5, activation='sigmoid'),
        utils.metrics.TotalError()
    ]

    # set optimizer
    if hparams.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=hparams.start_lr, weight_decay=hparams.weight_decay),
        ])
    elif hparams.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop([
            dict(params=model.parameters(), lr=hparams.start_lr, weight_decay=hparams.weight_decay),
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
    min_val_loss = np.inf
    epochs_no_improve = 0
    for i in range(0, hparams.epochs):
        print('\nEpoch: {}'.format(i))

        try:
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

            # early stopping
            if valid_logs['loss'] < min_val_loss:
                min_val_loss = valid_logs['loss']
                epochs_no_improve = 0
            else:
                epochs_no_improve +=1
            if epochs_no_improve >= hparams.early_stopping_patience:
                print('Early Stopping!')
                logger.set_exit_message('Early stopping')
                logger._save_logs()
                break

        except Exception as err: # save any error message
            print('Some error occurred during training:')
            traceback.print_tb(err.__traceback__)
            msg = traceback.format_exc()
            logger.set_exit_message(msg)
            logger._save_logs()
            break



if __name__ == '__main__':

    main()
