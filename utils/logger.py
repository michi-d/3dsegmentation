
__author__ = ['Michael Drews']

import torch
import json
import warnings
from utils.misc import count_trainable_parameters
from pathlib import Path
import shutil
import datetime
import os
from numpy import inf
from numpy import min, max


class Logger(object):
    """Logs all training parameters and process"""

    def __init__(self, fpath, model,
                 title='', hyperparameters={}, script=None, checkpoint_policy=None,
                 resume=False, eval_score_name=None, eval_mode='min', verbose=False):

        self.fpath = Path(fpath)
        self.parent_directory = self.fpath#.parent
        if not self.parent_directory.exists():
            self.parent_directory.mkdir(parents=True)

        self.checkpoint_path_dir = self.parent_directory / 'checkpoints'
        if not self.checkpoint_path_dir.exists():
            self.checkpoint_path_dir.mkdir(parents=True)

        self.pythonfile_path_dir = self.parent_directory / 'pythonfile'
        if not self.pythonfile_path_dir.exists():
            self.pythonfile_path_dir.mkdir()

        self.resume = resume
        self.model = model
        self.title = title
        self.script = script
        self.checkpoint_policy = checkpoint_policy
        self.eval_score_name = eval_score_name
        self.verbose = verbose
        self.eval_mode = eval_mode
        if self.eval_mode == 'min':
            self.best_score = inf
        if self.eval_mode == 'max':
            self.best_score = -inf

        if fpath is not None:
            if resume:
                self._load_data_()
            else:
                self._init_data()
                self.save_hyperparameters(**hyperparameters)
                self._save_pythonfile()
                self._save_logs()

    def _init_data(self):
        """
        Initializes data dictionary with empty fields.
        """
        self.data = dict()
        self.data['logs'] = dict()
        self.data['hyperparameters'] = dict()
        self.data['title'] = self.title
        self.data['parameter_count'] = None
        self.data['checkpoint_path'] = str(self.checkpoint_path_dir)
        self.data['pythonfile_path'] = str(self.pythonfile_path_dir)
        self.data['epochs_trained'] = 0
        self.data['epochs_timing'] = dict()
        self.data['eval_score_name'] = self.eval_score_name
        self.data['eval_mode'] = self.eval_mode
        self.data['checkpoint_policy'] = self.checkpoint_policy

    def _load_data_(self):
        """
        Loads the data dictionary from external file and sets internal variables accordingly.
        """
        with open(self.fpath / 'logs.json', 'r') as fp:
            self.data = json.load(fp)

        self.title = self.data['title']
        self.script = list(Path(self.data['pythonfile_path']).glob('*.py'))[0]
        self.checkpoint_policy = self.data['checkpoint_policy']
        self.eval_score_name = self.data['eval_score_name']
        self.eval_mode = self.data['eval_mode']
        if self.eval_mode == 'min':
            self.best_score = min(self.data['logs'][self.eval_score_name])
        elif self.eval_mode == 'max':
            self.best_score = max(self.data['logs'][self.eval_score_name])

    def _check_log_length(self):
        for i, (name, value) in enumerate(self.data['logs'].items()):
            n = len(self.data['logs'][name])
            if i == 0:
                n_logs = n
            else:
                if n != n_logs:
                    return False
        return True

    def log_epoch(self, **items):
        """Appends new log entries

        Args:
            **items: (name, value) pairs of logging values
        """
        for i, (name, value) in enumerate(items.items()):
            if name in self.data['logs']:
                self.data['logs'][name].append(value)
            else:
                self.data['logs'][name] = [value]

        if not self._check_log_length():
            print('WARNING: Log length not the same for all items.')
            warnings.warn('WARNING: Log length not the same for all items.')

        # save time
        epoch = int(self.data['epochs_trained'])
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.data['epochs_timing'][epoch] = now

        # save checkpoint
        if self.checkpoint_policy == 'all':
            name = f'model_{now}'
            self._save_checkpoint(name)
        elif self.checkpoint_policy == 'last':
            self._save_checkpoint('model')
        elif self.checkpoint_policy == 'best':
            if self._is_better_model(**items):
                self._save_checkpoint('best_model')

        self._save_logs()
        self.data['epochs_trained'] = epoch + 1

    def _save_logs(self):
        with open(self.fpath / 'logs.json', 'w+') as fp:
            json.dump(self.data, fp)

    def _is_better_model(self, **items):
        if self.eval_score_name:
            for i, (name, value) in enumerate(items.items()):
                if name == self.eval_score_name:
                    if self.eval_mode == 'min':
                        if value < self.best_score:
                            self.best_score = value
                            return True
                        else:
                            return False
                    elif self.eval_mode == 'max':
                        if value > self.best_score:
                            self.best_score = value
                            return True
                        else:
                            return False
        print(f'Model saving not possible: no metric with this name: {self.eval_score_name}')
        warnings.warn(f'Model saving not possible: no metric with this name: {self.eval_score_name}')
        return False

    def save_hyperparameters(self, **items):
        """Sets new values for the hyperparameter field in self.data

        Args:
            **items: (name, value) pairs of hyperparameters
        """
        for i, (name, value) in enumerate(items.items()):
            self.data['hyperparameters'][name] = value
        self.data['parameter_count'] = str(int(count_trainable_parameters(self.model)))
        self._save_logs()

    def _save_checkpoint(self, name='model'):
        """
        Saves model checkpoint under the given name
        """
        torch.save(self.model, (self.checkpoint_path_dir / name).with_suffix('.pth'))
        if self.verbose:
            print('Model saved!')

    def _save_pythonfile(self, name='train'):
        """
        Saves training script under the given name
        """
        if self.script:
            #now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            #name = f'{name}_{now}'
            shutil.copy(self.script, (self.pythonfile_path_dir / name).with_suffix('.py'))



