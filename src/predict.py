import torch
import numpy as np
from PIL import Image
from typing import List

class Predict:
    def __init__(self, model, image:List[np.ndarray|Image.Image|torch.Tensor]):
        self.model = model
        self.image = image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for img in self.image:
                if isinstance(img, np.ndarray):
                    pass
