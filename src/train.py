import torch
from .dataset import *

class Train():
    
    def __init__(self, dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
    
    def _step_epoch(self):
        image, label = self.dataset[0]
        pass
