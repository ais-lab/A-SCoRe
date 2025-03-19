from __future__ import division

import os
import torch

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def adjust_lr(optimizer, init_lr, c_iter, n_iter):
    lr = init_lr * (0.5 ** ((c_iter + 200000 - n_iter) // 50000 + 1 if (c_iter 
         + 200000 - n_iter) >= 0 else 0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr    

def save_state(savepath, epoch, model, optimizer, name='model.pkl'):
    state = {'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()}
    filepath = os.path.join(savepath, name)
    torch.save(state, filepath)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
