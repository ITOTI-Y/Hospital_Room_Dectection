import pathlib
import torch
from typing import Tuple, Generator
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

class RoomDataset(Dataset):
    def __init__(self, data_path: pathlib.Path):
        super().__init__()
        self.data_path = data_path
        self.data = {}
        self._prep_data()
        self.init_transforms = v2.Compose([
            v2.ToImage(), # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.Resize(size=(512,512), antialias=True),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _prep_data(self):
        image_path = self.data_path.rglob('*.jpg')
        for path in image_path:
            if '_mask' in path.name:
                name = path.stem.removesuffix('_mask')
                if name in self.data:
                    self.data[name]['label'] = Image.open(path)
                else:
                    self.data[name] = {'label': Image.open(path)}
            else:
                name = path.stem
                if name in self.data:
                    self.data[name]['image'] = Image.open(path)
                else:
                    self.data[name] = {'image': Image.open(path)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor,torch.Tensor]:
        name = list(self.data.keys())[idx]
        image = self.data[name]['image']
        label = self.data[name]['label']
        return self.init_transforms(image), self.init_transforms(label)
    
class RoomDataLoader(DataLoader):
    def __init__(self, dataset: RoomDataset, batch_size: 1, **kwargs):
        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def __iter__(self) -> Generator[Tuple[torch.Tensor,torch.Tensor], None, None]:
        for data in super().__iter__():
            yield data

