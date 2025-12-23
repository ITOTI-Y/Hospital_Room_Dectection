import random
from typing import Any

from loguru import logger

from src.config.config_loader import ConfigLoader


class PathwayGenerator:
    def __init__(self, config: ConfigLoader, is_training: bool = True, eval_mix_ratio: float = 0.3):
        """
        Initialize PathwayGenerator.

        Args:
            config: Configuration loader
            is_training: Whether this is for training mode
            eval_mix_ratio: Ratio of evaluation pathways to mix in during training (0.0 to 1.0)
        """
        self.process_id = 1
        self.paths = config.paths
        self.is_training = is_training
        self.eval_mix_ratio = eval_mix_ratio if is_training else 0.0

        if is_training:
            self.meta_rules = config.pathways.training.meta_rules
            self.pathways_number = config.pathways.pathways_number
            # Load evaluation rules for mixing
            self.eval_meta_rules = config.pathways.evaluation.smart.meta_rules
        else:
            self.meta_rules = config.pathways.evaluation.smart.meta_rules
            self.pathways_number = len(self.meta_rules)
            self.eval_meta_rules = []

        self.pools = config.pathways.training.department_pools
        self.fragments = config.pathways.training.sequence_fragments
        self.logger = logger.bind(module=self.__class__.__name__)
        self.logger.info(
            f"PathwayGenerator initialized (training={is_training}, "
            f"eval_mix_ratio={eval_mix_ratio if is_training else 'N/A'})"
        )

    def generate_all(self) -> dict[str, dict[str, Any]]:
        pathways: dict[str, dict[str, Any]] = {}
        count = 0

        # In training mode, mix evaluation pathways with random pathways
        if self.is_training and self.eval_mix_ratio > 0 and self.eval_meta_rules:
            # Calculate number of evaluation pathways to include
            num_eval_pathways = int(self.pathways_number * self.eval_mix_ratio)
            num_random_pathways = self.pathways_number - num_eval_pathways

            # First, generate evaluation-based pathways
            eval_count = 0
            while eval_count < num_eval_pathways and eval_count < len(self.eval_meta_rules):
                full_sequence_str, final_pathway = self.generate_from_rule(
                    self.eval_meta_rules[eval_count % len(self.eval_meta_rules)]
                )
                if full_sequence_str not in pathways:
                    pathways[full_sequence_str] = final_pathway
                    eval_count += 1
                count += 1
                if count > self.pathways_number * 2:  # Prevent infinite loop
                    break

            # Then generate random pathways
            random_count = 0
            while random_count < num_random_pathways:
                full_sequence_str, final_pathway = self.generate()
                if full_sequence_str not in pathways:
                    pathways[full_sequence_str] = final_pathway
                    random_count += 1
                count += 1
                if count > self.pathways_number * 3:  # Prevent infinite loop
                    break
        else:
            # Original behavior for evaluation or no mixing
            while count < self.pathways_number:
                full_sequence_str, final_pathway = self.generate()
                if full_sequence_str in pathways:
                    continue
                pathways[full_sequence_str] = final_pathway
                count += 1

        return pathways

    def generate_from_rule(self, rule: Any) -> tuple[str, dict[str, Any]]:
        """Generate a pathway from a specific rule (used for evaluation pathways)."""
        generated_sequence: list[str] = []
        context: dict[str, str] = {}

        for step in rule.structure:
            self._parse_step(step, context, generated_sequence)

        final_pathway = {
            "process_id": f"PROC_EVAL_{rule['id']}_{self.process_id}",
            "description": f"Evaluation Rule: {rule['description']}",
            "core_sequence": generated_sequence,
            "start_nodes": rule.start_nodes,
            "end_nodes": rule.end_nodes,
            "base_weight": rule.base_weight,
        }

        full_sequence_str = (
            f"{' -> '.join(final_pathway['start_nodes'])} -> "
            f"{' -> '.join(generated_sequence)} -> "
            f"{' -> '.join(final_pathway['end_nodes'])}"
        )
        self.logger.debug(
            f"Generated eval pathway: {final_pathway['process_id']}, Full sequence: {full_sequence_str}"
        )

        self.process_id += 1

        return full_sequence_str, final_pathway

    def generate(self) -> tuple[str, dict[str, Any]]:
        rule_weights = [rule.base_weight for rule in self.meta_rules]
        chosen_rule = random.choices(self.meta_rules, weights=rule_weights, k=1)[0]

        generated_sequence: list[str] = []
        context: dict[str, str] = {}

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
        self.logger.debug(
            f"Generated pathway: {final_pathway['process_id']}, Full sequence: {full_sequence_str}"
        )

        self.process_id += 1

        return full_sequence_str, final_pathway

    def _parse_step(
        self, step: Any, context: dict[str, Any], generated_sequence: list[str]
    ):
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
