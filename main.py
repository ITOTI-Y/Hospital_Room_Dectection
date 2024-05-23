import pathlib
import torch
import src


if __name__ == "__main__":
    DATA_PATH = pathlib.Path('./data')
    dataset = src.RoomDataset(data_path=DATA_PATH)
    model = src.Backbone()
    src.Train(dataset=dataset, model=model, bathc_size=2, epochs = 100)._step_train()