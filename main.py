import src
import src.utils
import torch
from torch.utils.data import random_split
from src.config import COLOR_MAP
from src.train import *

CONFIG = src.Train_Config()

if __name__ == "__main__":
    # torch.manual_seed(CONFIG.SEED)
    dataset = src.Mask2FormerDataset()

    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[CONFIG.TRAIN_SIZE, CONFIG.VAL_SIZE])
    # src.utils.visualize_dataset(dataset)
    
    src.Train(train_dataset,val_dataset).train()
    # predict = Predict(model_path='./models/best_model.pth', image_path='./data/val_image/1f.jpg')
    # predict.run()