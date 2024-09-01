import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Loss_Config

LOSS_CONFIG = Loss_Config()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = LOSS_CONFIG.SMOOTH

    def forward(self, x:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        x = F.softmax(x, dim=1)

        num_classes = x.size(1)
        label_one_hot = F.one_hot(label, num_classes).permute(0,3,1,2).float()

        # calculate Dice cofficient
        intersection = (x * label_one_hot).sum(dim=(2,3))
        union = x.sum(dim=(2,3)) + label_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = LOSS_CONFIG.CE_WEIGHT
        self.dice_weight = LOSS_CONFIG.DICE_WEIGHT

    def forward(self, x:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        ce_loss = self.loss(x, label)
        dice_loss = self.dice_loss(x, label)
        combine_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return combine_loss
