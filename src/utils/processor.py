"""Handles image loading, preprocessing, and basic morphological operations."""

import cv2
from src.rl_optimizer.utils.setup import setup_logger
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from typing import Tuple, Dict, Any, Optional

from src.config import NetworkConfig

logger = setup_logger(__name__)


class ImageProcessor:
    """
    Provides functionalities for image loading, color quantization,
    and morphological operations.
    """

    def __init__(self, config: NetworkConfig, color_map_data: Dict[Tuple[int, int, int], Dict[str, Any]]):
        """
        Initializes the ImageProcessor.

        Args:
            config: The NetworkConfig object.
            color_map_data: The color map dictionary.
        """
        self.config = config
        self.color_map_data = color_map_data
        self._current_image_data: Optional[np.ndarray] = None
        self._image_height: Optional[int] = None
        self._image_width: Optional[int] = None

    def load_and_prepare_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image, rotates it, and stores its dimensions.

        The processed image is stored internally and also returned.

        Args:
            image_path: Path to the image file.

        Returns:
            A NumPy array representing the processed image (RGB).

        Raises:
            FileNotFoundError: If the image_path does not exist.
            IOError: If the image cannot be opened or read.
        """
        try:
            img = Image.open(image_path).convert('RGB')  # Ensure RGB
            if self.config.IMAGE_ROTATE != 0:
                img = img.rotate(self.config.IMAGE_ROTATE)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except IOError:
            raise IOError(f"Could not open or read image file: {image_path}")

        self._current_image_data = np.asarray(img, dtype=np.uint8)
        if self._current_image_data is None:
            raise ValueError(
                f"Failed to convert image to numpy array: {image_path}")

        self._image_height, self._image_width = self._current_image_data.shape[:2]
        return self._current_image_data.copy()  # 返回副本避免意外修改

    def get_image_dimensions(self) -> Tuple[int, int]:
        """
        Returns the dimensions of the last loaded image.

        Returns:
            A tuple (height, width).

        Raises:
            ValueError: If no image has been loaded yet.
        """
        if self._image_height is None or self._image_width is None:
            raise ValueError(
                "Image not loaded. Call load_and_prepare_image() first.")
        return self._image_height, self._image_width

    def quantize_colors(self, image_data: np.ndarray) -> np.ndarray:
        """
        Replaces each pixel's color in the image with the nearest color
        from the provided color_map using a KDTree for efficiency.

        Args:
            image_data: A NumPy array representing the image (H, W, 3) in RGB.

        Returns:
            A NumPy array of the same shape with colors replaced.
        """
        if not self.color_map_data:
            # 如果颜色映射为空，返回原始图像以避免错误
            logger.warning(
                "Warning: Color map is empty. Returning original image from quantize_colors.")
            return image_data.copy()

        pixels = image_data.reshape(-1, 3)
        map_colors_rgb = list(self.color_map_data.keys())

        if not map_colors_rgb:  # 如果self.color_map_data不为空，这种情况不应该发生
            logger.warning(
                "Warning: No colors in color_map_data keys. Returning original image.")
            return image_data.copy()

        kdtree = KDTree(map_colors_rgb)
        _, closest_indices = kdtree.query(pixels)

        new_pixels = np.array(map_colors_rgb, dtype=np.uint8)[closest_indices]
        new_image = new_pixels.reshape(image_data.shape).astype(np.uint8)
        return new_image

    def apply_morphology(self, mask: np.ndarray, operation: str = 'close_open',
                         kernel_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Applies morphological operations to a binary mask.

        Args:
            mask: The input binary mask (NumPy array, dtype=uint8).
            operation: The type of operation.
                       'close_open': MORPH_CLOSE then MORPH_OPEN (default)
                       'open': MORPH_OPEN
                       'close': MORPH_CLOSE
                       'dilate': MORPH_DILATE
                       'erode': MORPH_ERODE
            kernel_size: Tuple (k_height, k_width) for the morphological kernel.
                         Defaults to `config.MORPHOLOGY_KERNEL_SIZE`.

        Returns:
            The processed binary mask.
        """
        if kernel_size is None:
            k_size = self.config.MORPHOLOGY_KERNEL_SIZE
        else:
            k_size = kernel_size

        kernel = np.ones(k_size, np.uint8)
        processed_mask = mask.copy()

        if operation == 'close_open':
            processed_mask = cv2.morphologyEx(
                processed_mask, cv2.MORPH_CLOSE, kernel)
            processed_mask = cv2.morphologyEx(
                processed_mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'open':
            processed_mask = cv2.morphologyEx(
                processed_mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            processed_mask = cv2.morphologyEx(
                processed_mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)
        elif operation == 'erode':
            processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
        else:
            raise ValueError(
                f"Unsupported morphological operation: {operation}")

        return processed_mask
    

class DebugImage:
    """Helper class for saving and displaying debug images."""
    count = 0

    def __init__(self, image_data: np.ndarray, save: bool = False,
                 show_napari: bool = False, suffix: str = '',
                 config: NetworkConfig = NetworkConfig()):
        """
        Initializes DebugImage.

        Args:
            image_data: NumPy array of the image to debug.
            save: If True, saves the image.
            show_napari: If True, shows the image using napari (requires napari installed).
            suffix: Suffix for the saved filename.
            debug_path_base: Base directory for saving debug images.
        """
        self.image_to_debug = image_data.copy()  # 使用副本工作
        self.debug_path = config.DEBUG_PATH
        self.debug_path.mkdir(parents=True, exist_ok=True)

        if save:
            self._save_image(suffix)
        if show_napari:
            self._show_with_napari()

    def _save_image(self, suffix: str = ''):
        """Saves the debug image."""
        filename = f'debug_{DebugImage.count}_{suffix}.png'
        save_path = self.debug_path / filename
        try:
            # 如果是RGB格式，转换为BGR以供Pillow保存，或处理灰度图
            if self.image_to_debug.ndim == 3 and self.image_to_debug.shape[2] == 3:
                # Assume RGB from PIL, convert to BGR for OpenCV-style saving or save as is with PIL
                img_to_save_pil = Image.fromarray(self.image_to_debug)
            elif self.image_to_debug.ndim == 2:  # 灰度图像
                 img_to_save_pil = Image.fromarray(self.image_to_debug, mode='L')
            else:
                logger.warning(f"Warning: Unsupported image format for saving: {self.image_to_debug.shape}")
                return

            img_to_save_pil.save(save_path)
            DebugImage.count += 1
            logger.info(f"Debug image saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving debug image {save_path}: {e}")


    def _show_with_napari(self):
        """Shows the image using napari."""
        try:
            import napari
            viewer = napari.Viewer()
            viewer.add_image(self.image_to_debug)
            napari.run()
        except ImportError:
            logger.warning("Napari is not installed. Skipping napari display.")
        except Exception as e:
            logger.error(f"Error showing image with napari: {e}")
