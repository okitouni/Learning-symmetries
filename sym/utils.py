import torch
import pytorch_lightning as pl
from tqdm import tqdm
import sys
from sklearn.metrics import classification_report

@torch.jit.script
def Augment(imgs):
    nums = []
    for img in imgs:
        i = torch.randint(0,10,(1,)).item()
        j = torch.randint(0,10,(1,)).item()
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
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        bar.set_description('running validation ...')
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
            ascii=Truerue
        )
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
            ascii=True
        )
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
