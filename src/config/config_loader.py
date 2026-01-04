from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


class ConfigLoader:
    _instance = None
    configs = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.configs = cls._load_configs()
        return cls._instance

    def __init__(self):
        pass

    @classmethod
    def _load_configs(cls) -> Any:
        config_dir = Path(__file__).parent.parent.parent / 'configs'

        for config_path in config_dir.glob('*.yaml'):
            config_name = config_path.stem
            config_data = OmegaConf.load(config_path)
            setattr(cls, config_name, config_data)

    def __getattr__(self, name):
        raise AttributeError(
            f"Config '{name}' not found"
            f'Available configs: {[key for key in self.__class__.__dict__ if not key.startswith("_")]}'
        )
