import numpy as np

from src.pipeline import PathwayGenerator, CostManager
from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.rl.env import LayoutEnv

class OptimizeManager:
    def __init__(self, config: ConfigLoader, **kwargs):
        self.logger = setup_logger(__name__)
        self.config = config
        self.kwargs = kwargs
        self.pathway_generator = PathwayGenerator(self.config)
        self.cost_manager = CostManager(self.config, is_shuffle=True)
        self.env = LayoutEnv(config, max_departments=100, max_step=1000)

    def run(self):
        self.pathways = self.pathway_generator.generate_all()
        self.cost_manager.initialize(pathways=self.pathways)
        self.env.reset()
        self.env.step(np.array([0, 1]))

        for _ in range(1000):
            actions = self.env.action_space.sample()
            self.env.step(actions)




        # original_cost = cost_engine.current_travel_cost
        # current_cost = cost_engine.current_travel_cost
        # optimize_layout = None
        # for i in range(50000):
        #     dept1, dept2 = random.sample(list(cost_engine.layout), 2)
        #     if cost_engine.swap(dept1, dept2):
        #         self.logger.info(f"Swap {dept1} and {dept2}, new cost: {cost_engine.current_travel_cost}")
        #         if cost_engine.current_travel_cost < current_cost:
        #             current_cost = cost_engine.current_travel_cost
        #             optimize_layout = cost_engine.layout.copy()
        # pass