import random

from src.pipeline import PathwayGenerator, CostManager
from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger

class OptimizeManager:
    def __init__(self, config: ConfigLoader, **kwargs):
        self.logger = setup_logger(__name__)
        self.config = config
        self.kwargs = kwargs
        self.pathway_generator = PathwayGenerator(self.config)
        self.cost_manager = CostManager(self.config)

    def run(self):
        self.pathways = self.pathway_generator.generate_all()
        self.cost_manager.initialize(pathways=self.pathways)
        self.original_travel_cost = self.cost_manager.current_travel_cost
        self.layout = self.cost_manager.layout

        # for i in range(1000):
        #     dept1, dept2 = random.sample(list(self.layout.keys()), 2)
        #     if self.cost_manager.swap(dept1, dept2) > 0:
        #         self.logger.info(f"Swap {dept1} and {dept2}, new cost: {self.cost_manager.current_travel_cost}")
        #         pass