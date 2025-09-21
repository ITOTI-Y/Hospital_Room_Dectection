import random
from typing import Dict, Any, List, Tuple
from src.utils.logger import setup_logger
from src.config.config_loader import ConfigLoader

class PathwayGenerator:

    def __init__(self, config: ConfigLoader):
        self.process_id = 1
        self.paths = config.paths
        self.pools = config.pathways.training_generation.department_pools
        self.fragments = config.pathways.training_generation.sequence_fragments
        self.meta_rules = config.pathways.training_generation.meta_rules
        self.pathways_number = config.pathways.pathways_number
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("PathwayGenerator initialized")


    def generate_all(self) -> Dict[str, Dict[str, Any]]:
        pathways: Dict[str, Dict[str, Any]] = {}
        count = 0
        while count < self.pathways_number:
            full_sequence_str, final_pathway = self.generate()
            if full_sequence_str in pathways:
                continue
            pathways[full_sequence_str] = final_pathway
            count += 1
        return pathways


    def generate(self) -> Tuple[str, Dict[str, Any]]:
        rule_weights = [rule.base_weight for rule in self.meta_rules]
        chosen_rule = random.choices(self.meta_rules, weights=rule_weights, k=1)[0]

        generated_sequence: List[str] = []
        context: Dict[str, str] = {}

        for step in chosen_rule.structure:
            self._parse_step(step, context, generated_sequence)

        final_pathway = {
            "process_id": f"PROC_GEN_{chosen_rule['id']}_{self.process_id}",
            "description": f"Programmatic Generation: {chosen_rule['description']}",
            "core_sequence": generated_sequence,
            "start_nodes": chosen_rule.start_nodes,
            "end_nodes": chosen_rule.end_nodes,
            "base_weight": chosen_rule.base_weight,
        }

        full_sequence_str = (
            f"{' -> '.join(final_pathway['start_nodes'])} -> "
            f"{' -> '.join(generated_sequence)} -> "
            f"{' -> '.join(final_pathway['end_nodes'])}"
        )
        self.logger.info(
            f"Generated pathway: {final_pathway['process_id']}, Full sequence: {full_sequence_str}"
        )

        self.process_id += 1

        return full_sequence_str, final_pathway

    def _parse_step(self, step: Any, context: Dict[str, Any], generated_sequence: List[str]):
        step_type = step.type

        if step_type == "primary_department":
            from_pool_value = step.from_pool
            department_list = []
            if isinstance(from_pool_value, str) and from_pool_value in self.pools:
                department_list = self.pools[from_pool_value]
            elif isinstance(from_pool_value, list):
                department_list = from_pool_value

            if department_list:
                dept = random.choice(department_list)
                context["primary_department"] = dept
                generated_sequence.append(dept)

        elif step_type == "fixed":
            dept_name = step.name
            if dept_name:
                generated_sequence.append(dept_name)

        elif step_type == "pool":
            pool_name = step.pool_name
            if pool_name in self.pools:
                generated_sequence.append(random.choice(self.pools[pool_name]))

        elif step_type == "optional":
            if random.random() < step.probability:
                self._parse_step(step.step, context, generated_sequence)

        elif step_type == "fragment":
            fragment_name = step.fragment_name
            if fragment_name in self.fragments:
                fragment_steps = self.fragments[fragment_name]
                for frag_step in fragment_steps:
                    self._parse_step(frag_step, context, generated_sequence)

        elif step_type == "repeat":
            target = step.target
            if target in context:
                generated_sequence.append(context[target])