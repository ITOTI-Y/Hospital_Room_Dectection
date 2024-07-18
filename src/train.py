import torch
from torch.utils.data import DataLoader
from .dataset import *
from .config import Train_Config
from .model import get_model

TRAIN_CONFIG = Train_Config()

class Train():
    
    def __init__(self, dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = DataLoader(dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=TRAIN_CONFIG.SHUFFLE)
        self.model = get_model().to(self.device)
    
    def _step_epoch(self):
        for image, label in self.data:
            image, label = image.to(self.device), label.to(self.device)
            outputs = self.model(pixel_values=image, labels=label)
            loss = outputs.loss
            print(image.shape, label.shape)
        pass
