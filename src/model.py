import torch.nn as nn
from transformers import Swinv2Config, Swinv2Model, AutoImageProcessor
from .config import Model_Config

class Backbone(nn.Module):
    def __init__(self,):
        super().__init__()
        self.config = Swinv2Config(image_size=Model_Config.IMAGE_SIZE)
        self.model = Swinv2Model(self.config).from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')

    def forward(self, x):
        return self.model(x)