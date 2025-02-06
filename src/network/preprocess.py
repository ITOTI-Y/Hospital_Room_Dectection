import numpy as np
import cv2
import napari
import os
import unicodedata
from PIL import Image
from scipy.spatial import KDTree
from src.config import COLOR_MAP

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    image = find_nearest_color(image, COLOR_MAP)
    return image

def morphology_operation(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5,5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def find_nearest_color(image: np.ndarray, color_map: dict) -> np.ndarray:
    color_map = COLOR_MAP
    pixels = image.reshape(-1, 3)
    kdtree = KDTree(list(color_map.keys()))
    _, closest_indices = kdtree.query(pixels)
    color_values = np.array(list(color_map.keys()))
    new_pixels = color_values[closest_indices]
    new_image = new_pixels.reshape(image.shape).astype(np.uint8)
    return new_image

class DebugImage:
    count = 0
    def __init__(self, image: Image.Image,save:bool = False, show: bool = True, suffix: str = ''):
        self.image = np.array(image)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        if save:
            self._save(suffix)
        if show:
            self._show()

    def _save(self, suffix: str = ''):
        # 原图转为RGB
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        os.makedirs('./debug', exist_ok=True)
        Image.fromarray(self.image).save(f'./debug/debug_{self.count}_{suffix}.png')
        DebugImage.count += 1

    def _show(self):
        viewer = napari.Viewer()
        viewer.add_image(self.image)
        napari.run()

