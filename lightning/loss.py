from __future__ import division

import torch
import torch.nn as nn

def get_loss(name):
    return {
            'euclidean_loss'    : EuclideanLoss,
            'regcoord_loss'     : RegCoordLoss
            }[name]


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, pred, target, mask):
        n, c, h, w = pred.size()

        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        pred = pred[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        pred = pred.view(-1, c)

        target = target.transpose(1,2).transpose(2,3).contiguous().view(-1, c)
        target  = target[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        target = target.view(-1,c)

        loss = self.pdist(pred, target)
        loss = torch.sum(loss, 0)
        loss /= mask.sum()

        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred, target, mask):
        loss = self.celoss(pred, target)
        return (loss*mask).sum() / mask.sum()



class RegCoordLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred:dict, gt:dict):
        #import pdb; pdb.set_trace()
        batch_size, _, _ = pred['scenecoord'].shape
        sc_mask = gt['sfm_sc_mask']

        sc_loss_l2 = torch.norm((pred['scenecoord'] - gt['sfm_sc'].permute(0, 2, 1)), 
                                dim=1)
        sc_loss_l2 = torch.sum(sc_mask * sc_loss_l2) / batch_size

        return sc_loss_l2



    
        

