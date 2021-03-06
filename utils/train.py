
__credits__ = ['Pavel Yakubovskiy, https://github.com/qubvel/segmentation_models.pytorch']

import sys
import torch
from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        """Should return loss and model prediction
        """
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for element in iterator:
                x = element[0]
                y = element[1]
                if len(element) == 3: # for keypoint detection
                    z = element[2]
                    z.to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                #loss_logs = {self.loss.__name__: loss_meter.mean}
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    if metric_fn.requires_z: # for keypoint detection
                        gt = z
                    else:
                        gt = y
                    metric_value = metric_fn(y_pred, gt).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                # update grad logs
                # grads = self.model.get_gradients()

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        output = self.model.forward(x)

        if isinstance(output, list):
            # multiple outputs (intermediate supervision)
            loss = 0
            for o in output:
                loss += self.loss(o, y)
            prediction = output[-1]
        else:
            # single output tensor
            prediction = output
            loss = self.loss(prediction, y)

        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            output = self.model.forward(x)
            if isinstance(output, list):
                # multiple outputs
                prediction = output[-1]
            else:
                # single output
                prediction = output
            loss = self.loss(prediction, y)
        return loss, prediction
    
    
#class History():
#
#    def __init__(self):
#        pass#
#
#    def _loss_update(self):

