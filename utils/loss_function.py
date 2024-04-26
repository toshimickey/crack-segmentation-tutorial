import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # model output = (X,1,224,224)
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, 1e-7, 1-1e-7)
        inputs = inputs.view(-1)
        # annotation mask = (X,1,224,224)
        targets = targets.view(-1)

        bce_weight = 0.5
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, inputs, targets):
        # model output = (X,1,224,224)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        # model output2 = (X,1,224,224)
        targets = torch.sigmoid(targets)
        targets = targets.view(-1)

        loss = torch.sqrt(torch.mean((inputs-targets)**2))
        return loss