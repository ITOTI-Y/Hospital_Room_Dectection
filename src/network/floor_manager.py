"""
Manages floor detection from filenames and Z-level calculations for SuperNetwork.
"""

import pathlib
import re

from loguru import logger

logger = logger.bind(module=__name__)


class FloorManager:
    """
    Handles detection of floor numbers from image filenames and calculation
    of their corresponding Z-coordinate levels.
    """

    def __init__(self, base_floor_default: int = 0, default_floor_height: float = 10.0):
        """
        Initializes the FloorManager.

        Args:
            base_floor_default: The default starting floor number if none can be detected.
            default_floor_height: The default height difference between adjacent floors.
        """
        self.base_floor_default = base_floor_default
        self.default_floor_height = default_floor_height
        self._floor_patterns = {
            # Order matters: more specific patterns first
            # B-1, B1, b-1, b1 -> negative
            r"([Bb])-?(\d+)": lambda m: -int(m.group(2)),
            # -1F, -2f -> negative
            r"-(\d+)[Ff]": lambda m: -int(m.group(1)),
            r"([Ll])(\d+)": lambda m: int(m.group(2))
            - 1,  # L1, l2 -> positive, 1-based to 0-based
            r"(\d+)[Ff]": lambda m: int(m.group(1))
            - 1,  # 1F, 2f -> positive, 1-based to 0-based
            # Add more general patterns if needed, like just a number if no prefix/suffix
            # r'_(\d+)_': lambda m: int(m.group(1)) # Example: floor_1_plan.png
        }

    def detect_floor_from_filename(self, file_path: pathlib.Path) -> int | None:
        """
        Detects the floor number from a filename.

        Args:
            file_path: Path object of the image file.

        Returns:
            The detected floor number (integer) or None if not detected.
        """
        filename = file_path.stem  # Get filename without extension
        for pattern, converter in self._floor_patterns.items():
            match = re.search(pattern, filename)
            if match:
                try:
                    return converter(match)
                except ValueError:
                    continue  # Conversion failed, try next pattern
        return None

    def auto_assign_floors(
        self, image_paths: list[pathlib.Path]
    ) -> tuple[dict[pathlib.Path, int], dict[int, pathlib.Path]]:
        """
        Assigns floor numbers to a list of image paths.

        Attempts to detect from filenames first. If unsuccessful for some or all,
        assigns sequentially based on sort order or a defined starting floor.

        Args:
            image_paths: A list of Path objects for the images.

        Returns:
            A tuple containing:
                - path_to_floor_map (Dict[pathlib.Path, int]): Maps image path to floor number.
                - floor_to_path_map (Dict[int, pathlib.Path]): Maps floor number to image path.
                                                               (Assumes one image per floor for this map)
        """
        path_to_floor_map: dict[pathlib.Path, int] = {}
        detected_floors: dict[pathlib.Path, int] = {}
        undetected_paths: list[pathlib.Path] = []

        for p_path in image_paths:
            floor = self.detect_floor_from_filename(p_path)
            if floor is not None:
                if floor in path_to_floor_map.values():
                    logger.warning(
                        f"Warning: Duplicate floor number {floor} detected. Check filenames."
                    )
                detected_floors[p_path] = floor
            else:
                undetected_paths.append(p_path)

        path_to_floor_map.update(detected_floors)

        if undetected_paths:
            undetected_paths.sort()

            if detected_floors:
                all_detected_nos = list(detected_floors.values())
                if all(f < 0 for f in all_detected_nos):
                    start_floor = (
                        min(all_detected_nos) - 1
                        if min(all_detected_nos) - 1 not in all_detected_nos
                        else max(all_detected_nos) + 1
                    )
                else:
                    start_floor = max(all_detected_nos) + 1

                while start_floor in path_to_floor_map.values():
                    start_floor += 1
            else:
                start_floor = self.base_floor_default

            for i, p_path in enumerate(undetected_paths):
                current_assigned_floor = start_floor + i
                while (
                    current_assigned_floor in path_to_floor_map.values()
                ):  # Avoid collision
                    current_assigned_floor += 1
                path_to_floor_map[p_path] = current_assigned_floor

        floor_to_path_map: dict[int, pathlib.Path] = {
            v: k for k, v in path_to_floor_map.items()
        }

        if len(floor_to_path_map) != len(path_to_floor_map):
            logger.warning(
                "Non-unique floor numbers assigned or detected, "
                "floor_to_path_map may not represent all images."
            )

        return path_to_floor_map, floor_to_path_map

    def calculate_z_levels(
        self, floor_to_path_map: dict[int, pathlib.Path]
    ) -> dict[int, float]:
        """
        Calculates the Z-coordinate for each floor.

        Args:
            floor_to_path_map: A map from floor number to image path (used to get sorted floors).

        Returns:
            A dictionary mapping floor number to its Z-coordinate.
        """
        if not floor_to_path_map:
            return {}

        sorted_floor_numbers = sorted(floor_to_path_map.keys())

        revision_num = 0

        if 0 not in sorted_floor_numbers:
            revision_num = 1

        z_levels: dict[int, float] = {
            floor_num: float(
                (floor_num - revision_num) * self.default_floor_height
                if floor_num > 0
                else floor_num * self.default_floor_height
            )
            for floor_num in sorted_floor_numbers
        }
        return z_levels
