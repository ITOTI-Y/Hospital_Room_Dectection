from src.pipeline import PathwayGenerator
from src.config.config_loader import ConfigLoader

class OptimizeManager:
    def __init__(self, config: ConfigLoader, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.pathway_generator = PathwayGenerator(self.config)

    def run(self):
        self.pathways = self.pathway_generator.generate_all()
        pass