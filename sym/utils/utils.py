import torch
import pytorch_lightning as pl
from tqdm import tqdm
import sys
from sklearn.metrics import classification_report
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard.summary import hparams
from typing import Any, Dict, Optional, Union


@torch.jit.script
def Augment(imgs):
    nums = []
    for img in imgs:
        i = torch.randint(0, 10, (1,)).item()
        j = torch.randint(0, 10, (1,)).item()
        istart = i * 28
        iend = (i+1) * 28
        jstart = i * 28
        jend = (i+1) * 28
        zeros = torch.zeros(280, 280)
        zeros[istart:iend, jstart:jend] = img
        nums.append(zeros)
    return torch.stack(nums)


class ProgressBar(pl.callbacks.ProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(
            desc='Validation ...',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            ascii=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        return bar

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        return bar


def Classification_report(model):
    with torch.no_grad():
        model.eval()
        model.to(device)
        pred = []
        target = []
        for x, y in model.val_dataloader():
            x = x.to(device)
            pred.append(model(x).cpu().numpy())
            target.append(y.numpy())
    pred = np.concatenate(pred)
    target = np.concatenate(target)
    out = classification_report(target, pred.argmax(axis=1))
    print(out)

class Logger(pl.loggers.TensorBoardLogger):
    def __init__(self, save_dir: str,
                 name: Union[str, None] = 'default',
                 version: Union[int, str, None] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True,
                 **kwargs):
        super().__init__(save_dir, name, version, log_graph, default_hp_metric, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
       # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)
            
            
def projBv(B,v):
    return torch.matmul(torch.outer(v,v),B)/torch.matmul(v,v)

def Misalignment(A,B):
    d = A.size(0)
    evalA,evecA = torch.eig(A,True)
    M = 0
    for evec in evecA:
        tr1 = projBv(B,evec)
        tr2 = projBv(torch.inverse(B),evec)
        M += torch.sqrt(torch.trace(tr1)*torch.trace(tr2))
    return M-d

def Misalignment2(A,B):
    evalA,evecA = torch.eig(A,True)
    M = 0
    for evec in evecA:
        a = torch.dot(evec,torch.matmul(B,evec))
        b = torch.dot(evec,torch.matmul(torch.inverse(B),evec))
        M += torch.sqrt(a*b)
    return M
