from omegaconf import OmegaConf
from pathlib import Path
from typing import Any

class ConfigLoader:

    agent: Any
    constraints: Any
    graph_config: Any
    paths: Any
    pathways: Any
    plotter: Any

    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_configs()
        return cls._instance
    
    def _load_configs(self) -> None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        
        for config_path in config_dir.glob("*.yaml"):
            config_name = config_path.stem
            config_data = OmegaConf.load(config_path)
            setattr(self, config_name, config_data)

    def __getattr__(self, name):
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")