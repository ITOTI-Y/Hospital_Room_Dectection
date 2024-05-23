import torch.nn as nn
from transformers import Swinv2Config, Swinv2Model, AutoImageProcessor

config = Swinv2Config(image_size=256)

class Backbone(nn.Module):
    def __init__(self,):
        super().__init__()
        self.config = Swinv2Config(image_size=256)
        self.model = Swinv2Model(config).from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')

    def forward(self, x):
        return self.model(x)