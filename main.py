import src
import src.utils
from src.config import COLOR_MAP

CONFIG = src.Train_Config()

if __name__ == "__main__":
    dataset = src.RoomDataset()
    # src.utils.visualize_dataset(dataset)
    src.Train(dataset=dataset)._step_epoch()