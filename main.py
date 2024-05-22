import pathlib
import torch
from src.train import *

if __name__ == "__main__":
    DATA_PATH = pathlib.Path('./data')
    dataset = RoomDataset(data_path=DATA_PATH)
    model = None
    Train(dataset=dataset, model=model, bathc_size=1, epochs = 100)._step_train()