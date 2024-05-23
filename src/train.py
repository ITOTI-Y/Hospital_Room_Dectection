import torch
from .dataset import *

class Train():
    
    def __init__(self, dataset, model = None, bathc_size:int = 1, epochs:int = 100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model = model.to(self.device)
        self.dataloader = RoomDataLoader(dataset = dataset, batch_size = bathc_size)
        
    
    def _step_train(self):
        for image,label in self.dataloader:
            image = image.to(device=self.device)
            label = label.to(device=self.device)
            out = self.model(image)
            pass
