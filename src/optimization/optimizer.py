"""
Module for facility layout optimization by reassigning functional types
to physical locations to minimize total travel times for defined workflows.
"""

import copy
import logging
from typing import List, Dict, Optional, Tuple, Set, Any, Sequence
import os
from joblib import Parallel, delayed

from src.analysis.process_flow import PeopleFlow, PathFinder
from src.config import NetworkConfig

logger = logging.getLogger(__name__)


class PhysicalLocation:
    """Represents a physical space/node in the facility.

    Attributes:
        name_id: The unique identifier of the physical location (e.g., 'RoomType_123').
        original_functional_type: The functional type initially associated with this
            physical location based on the input data (e.g., ' Radiology').
        area: The area of this physical location.
        is_swappable: Boolean indicating if this location can have its function reassigned.
                      Typically, connection types like 'Door' are not swappable.
    """

    def __init__(self, name_id: str, original_functional_type: str, area: float, is_swappable: bool = True):
        self.name_id: str = name_id
        self.original_functional_type: str = original_functional_type
        self.area: float = area
        self.is_swappable: bool = is_swappable

    def __repr__(self) -> str:
        return (f"PhysicalLocation(name_id='{self.name_id}', "
                f"original_type='{self.original_functional_type}', area={self.area:.2f}, "
                f"swappable={self.is_swappable})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhysicalLocation):
            return NotImplemented
        return self.name_id == other.name_id

    def __hash__(self) -> int:
        return hash(self.name_id)


class FunctionalAssignment:
    """Manages the assignment of functional types to lists of physical locations.

    This class represents the current "layout" by defining which physical
    locations (Name_IDs) are currently serving each functional type.
    It supports creating new assignment states by swapping the functional roles
    of two physical locations.

    Attributes:
        assignment_map: A dictionary where keys are functional type strings (e.g., "Radiology")
                        and values are lists of Name_ID strings of PhysicalLocations
                        currently assigned to that function.
    """

    def __init__(self, initial_assignment_map: Dict[str, List[str]]):
        # Deepcopy to ensure independence from the source map
        self.assignment_map: Dict[str, List[str]
                                  ] = copy.deepcopy(initial_assignment_map)

    def get_physical_ids_for_function(self, functional_type: str) -> List[str]:
        """Returns the list of physical Name_IDs assigned to the given functional type."""
        return self.assignment_map.get(functional_type, [])

    def get_functional_type_at_physical_id(self, physical_name_id: str) -> Optional[str]:
        """Finds which functional type, if any, the given physical_name_id is currently assigned to."""
        for func_type, id_list in self.assignment_map.items():
            if physical_name_id in id_list:
                return func_type
        return None

    def get_map_copy(self) -> Dict[str, List[str]]:
        """Returns a deep copy of the internal assignment map."""
        return copy.deepcopy(self.assignment_map)

    def apply_functional_swap(self, phys_loc_A_id: str, phys_loc_B_id: str) -> 'FunctionalAssignment':
        """Creates a new FunctionalAssignment representing the state after swapping
        the functional roles currently hosted at phys_loc_A_id and phys_loc_B_id.

        For example, if A hosts 'Radiology' and B hosts 'Lab', the new assignment
        will have A hosting 'Lab' and B hosting 'Radiology'.
        If one location is "unassigned" (not in any list in assignment_map),
        the function from the other location moves to it, and the original location
        becomes unassigned for that function.

        Args:
            phys_loc_A_id: Name_ID of the first physical location.
            phys_loc_B_id: Name_ID of the second physical location.

        Returns:
            A new FunctionalAssignment object with the swapped roles.
        """
        new_map = self.get_map_copy()

        func_type_at_A = self.get_functional_type_at_physical_id(phys_loc_A_id)
        func_type_at_B = self.get_functional_type_at_physical_id(phys_loc_B_id)

        # Remove A from its current function's list (if any) and add B instead.
        if func_type_at_A:
            if phys_loc_A_id in new_map.get(func_type_at_A, []):
                new_map[func_type_at_A].remove(phys_loc_A_id)
            if func_type_at_A not in new_map:
                new_map[func_type_at_A] = []  # Should not happen if found
            new_map[func_type_at_A].append(phys_loc_B_id)
            if not new_map[func_type_at_A]:  # If list became empty
                del new_map[func_type_at_A]

        # Remove B from its current function's list (if any) and add A instead.
        if func_type_at_B:
            # Check if B was already moved (e.g. A and B had same func type)
            if phys_loc_B_id in new_map.get(func_type_at_B, []):
                if func_type_at_A == func_type_at_B:
                    pass
                else:
                    new_map[func_type_at_B].remove(phys_loc_B_id)

            if func_type_at_B not in new_map:
                new_map[func_type_at_B] = []
            new_map[func_type_at_B].append(phys_loc_A_id)
            if not new_map[func_type_at_B]:
                del new_map[func_type_at_B]

        for func_type in list(new_map.keys()):
            if new_map[func_type]:
                new_map[func_type] = sorted(list(set(new_map[func_type])))
            else:
                del new_map[func_type]

        return FunctionalAssignment(new_map)

    def __repr__(self) -> str:
        return f"FunctionalAssignment(map_size={len(self.assignment_map)})"


class WorkflowDefinition:
    """Defines a patient/staff workflow to be evaluated.

    Attributes:
        workflow_id: A unique identifier for this workflow (e.g., 'OutpatientVisit').
        functional_sequence: An ordered list of functional type names representing
                             the steps in the workflow (e.g., ['Reception', 'ConsultRoom', 'Pharmacy']).
        weight: A float representing the importance or frequency of this workflow,
                used in the objective function.
    """

    def __init__(self, workflow_id: str, functional_sequence: List[str], weight: float = 1.0):
        self.workflow_id: str = workflow_id
        self.functional_sequence: List[str] = functional_sequence
        self.weight: float = weight

    def __repr__(self) -> str:
        return (f"WorkflowDefinition(id='{self.workflow_id}', "
                f"seq_len={len(self.functional_sequence)}, weight={self.weight})")


class EvaluatedWorkflowOutcome:
    """Stores the outcome of evaluating a WorkflowDefinition under a specific assignment.

    Attributes:
        workflow_definition: The WorkflowDefinition that was evaluated.
        average_time: The average travel time of all valid PeopleFlows for this workflow
                      under the given assignment. Can be float('inf') if any potential path
                      is unroutable or if no valid paths exist.
        shortest_flow: The actual PeopleFlow object representing the shortest path found
                       for this workflow under the given assignment. Can be None if no
                       valid path was found or if average_time is inf.
        num_paths_considered: The number of valid PeopleFlows used to calculate the average_time.
        all_path_times: A list of travel times for all valid PeopleFlows. Can be None.
    """

    def __init__(self,
                 workflow_definition: WorkflowDefinition,
                 average_time: Optional[float],
                 # Still store the shortest for reference
                 shortest_flow: Optional[PeopleFlow],
                 num_paths_considered: int,
                 all_path_times: Optional[List[float]]):
        self.workflow_definition: WorkflowDefinition = workflow_definition
        self.average_time: float = average_time if average_time is not None else float(
            'inf')
        self.shortest_flow: Optional[PeopleFlow] = shortest_flow
        self.num_paths_considered: int = num_paths_considered
        self.all_path_times: Optional[List[float]] = all_path_times

    def __repr__(self) -> str:
        path_str = "N/A"
        if self.shortest_flow and self.shortest_flow.actual_node_id_sequence:
            shortest_time_val = self.shortest_flow.total_time if self.shortest_flow.total_time is not None else float(
                'inf')
            path_str = (f"Shortest: {self.shortest_flow.actual_node_id_sequence[0]}..."
                        f"{self.shortest_flow.actual_node_id_sequence[-1]} (Time: {shortest_time_val:.2f})")

        return (f"EvaluatedWorkflow(id='{self.workflow_definition.workflow_id}', "
                f"avg_time={self.average_time:.2f}, num_paths={self.num_paths_considered}, {path_str})")

# Helper function for joblib to process a single flow
# This function will be called in parallel for each flow.
# It needs access to a PathFinder instance.


def _calculate_flow_time_joblib_task(flow_to_process: PeopleFlow, path_finder_instance: PathFinder) -> Tuple[Optional[float], Any, List[str]]:
    """
    Calculates total time for a flow and returns time, flow identifier, and sequence.
    The flow_to_process.total_time will be updated by calculate_flow_total_time.
    """
    time = path_finder_instance.calculate_flow_total_time(flow_to_process)
    # Return time and identifiers to re-associate with original flow object if needed,
    # and to allow the main thread to update the original flow objects.
    return time, flow_to_process.identify, flow_to_process.actual_node_id_sequence


class LayoutObjectiveCalculator:
    """Calculates the overall objective value for a given functional assignment.

    The objective is typically the sum of weighted average travel times for all defined workflows.
    """

    def __init__(self,
                 workflow_definitions: Sequence[WorkflowDefinition],
                 path_finder: PathFinder,
                 n_jobs_for_flows: int = -1):  # Default to all cores for flow calculation
        self.workflow_definitions: Sequence[WorkflowDefinition] = workflow_definitions
        self.path_finder: PathFinder = path_finder
        self.n_jobs_for_flows: int = n_jobs_for_flows if n_jobs_for_flows != 0 else 1
        if self.n_jobs_for_flows == -1:  # Check if os.cpu_count() is available
            cpu_count = os.cpu_count()
            self.n_jobs_for_flows = cpu_count if cpu_count is not None else 1
        elif self.n_jobs_for_flows == 0:  # Explicitly set to 1 if 0 is passed.
            self.n_jobs_for_flows = 1

    def evaluate(self, assignment: FunctionalAssignment) -> Tuple[float, List[EvaluatedWorkflowOutcome]]:
        """Evaluates the given FunctionalAssignment.

        Args:
            assignment: The FunctionalAssignment to evaluate.

        Returns:
            A tuple containing:
                - total_objective_value (float): The sum of weighted average travel times.
                  Can be float('inf') if any critical workflow is unroutable.
                - evaluated_outcomes (List[EvaluatedWorkflowOutcome]): A list of outcomes
                  for each workflow definition.
        """
        total_weighted_time: float = 0.0
        evaluated_outcomes: List[EvaluatedWorkflowOutcome] = []

        for wf_def in self.workflow_definitions:
            generated_flows = self.path_finder.generate_flows(
                workflow_names=wf_def.functional_sequence,
                workflow_identifier=wf_def.workflow_id,
                custom_assignment_map=assignment.get_map_copy()
            )

            workflow_average_time: float = float('inf')
            shortest_flow_for_wf: Optional[PeopleFlow] = None
            shortest_time_val_for_wf: float = float('inf')
            valid_flow_times_for_wf: List[float] = []
            num_paths_considered_for_wf: int = 0

            if not generated_flows:
                logger.debug(
                    f"Workflow '{wf_def.workflow_id}' generated no flows for current assignment. Avg Time is Inf.")
            else:
                # Use joblib to parallelize the calculation of total_time for each flow
                # The _calculate_flow_time_joblib_task helper will be used.
                # PathFinder instance needs to be passed to the helper.
                try:
                    # Using 'loky' backend which is robust for multiprocessing
                    # prefer='threads' could be used if tasks are I/O bound and GIL is an issue,
                    # but here it's likely CPU bound due to DataFrame lookups.
                    # 'threading' backend is limited by GIL for CPU-bound tasks in CPython.
                    # 'loky' or 'multiprocessing' are better for CPU-bound.
                    with Parallel(n_jobs=self.n_jobs_for_flows, backend='loky') as parallel:
                        results = parallel(
                            delayed(_calculate_flow_time_joblib_task)(
                                flow, self.path_finder)
                            for flow in generated_flows
                        )
                    # results is a list of (time, flow_identify, flow_sequence)

                except Exception as e:  # Catch potential joblib/pickling errors
                    logger.error(
                        f"Error during parallel flow calculation for workflow '{wf_def.workflow_id}': {e}. Falling back to sequential.")
                    results = [_calculate_flow_time_joblib_task(
                        flow, self.path_finder) for flow in generated_flows]

                all_paths_routable = True
                temp_flow_times: List[float] = []

                # Create a mapping from (identify, tuple(sequence)) to original flow object for quick lookup
                flow_map_for_update: Dict[Tuple[Any, Tuple[str, ...]], PeopleFlow] = {
                    (f.identify, tuple(f.actual_node_id_sequence)): f for f in generated_flows
                }

                for current_flow_time, flow_identify, flow_sequence_list in results:
                    # Ensure hashable for dict key
                    flow_sequence_tuple = tuple(flow_sequence_list)
                    original_flow_obj = flow_map_for_update.get(
                        (flow_identify, flow_sequence_tuple))

                    if original_flow_obj:
                        # Update the total_time on the original PeopleFlow object
                        # Note: _calculate_flow_time_joblib_task already calls calculate_flow_total_time
                        # which updates the flow object passed to it (which is a copy in multiprocessing).
                        # So, we need to update the original object in the main process's list.
                        original_flow_obj.update_total_time(
                            current_flow_time if current_flow_time is not None else float('inf'))
                    else:
                        logger.warning(
                            f"Could not find original flow for id {flow_identify} and sequence {flow_sequence_list} after parallel processing.")
                        # This should ideally not happen if identifiers are unique and sequences match.

                    if current_flow_time is None:
                        logger.debug(f"Flow identified by '{flow_identify}' (sequence: {'->'.join(flow_sequence_list[:2])}... ) "
                                     f"for workflow '{wf_def.workflow_id}' is unroutable. "
                                     f"Workflow average time will be Inf (strict mode).")
                        all_paths_routable = False
                        break  # Strict mode

                    temp_flow_times.append(current_flow_time)
                    if original_flow_obj and current_flow_time < shortest_time_val_for_wf:
                        shortest_time_val_for_wf = current_flow_time
                        shortest_flow_for_wf = original_flow_obj

                if all_paths_routable:
                    if temp_flow_times:
                        valid_flow_times_for_wf = temp_flow_times
                        workflow_average_time = sum(
                            valid_flow_times_for_wf) / len(valid_flow_times_for_wf)
                        num_paths_considered_for_wf = len(
                            valid_flow_times_for_wf)
                    else:
                        logger.warning(
                            f"Workflow '{wf_def.workflow_id}': all_paths_routable is True, but temp_flow_times is empty.")
                        workflow_average_time = float('inf')
                else:
                    workflow_average_time = float('inf')
                    shortest_flow_for_wf = None
                    valid_flow_times_for_wf = []
                    num_paths_considered_for_wf = 0

            outcome = EvaluatedWorkflowOutcome(
                workflow_definition=wf_def,
                average_time=workflow_average_time,
                shortest_flow=shortest_flow_for_wf,
                num_paths_considered=num_paths_considered_for_wf,
                all_path_times=valid_flow_times_for_wf if workflow_average_time != float(
                    'inf') else None
            )
            evaluated_outcomes.append(outcome)

            if workflow_average_time == float('inf'):
                total_weighted_time = float('inf')

            if total_weighted_time != float('inf'):
                total_weighted_time += workflow_average_time * wf_def.weight

        return total_weighted_time, evaluated_outcomes


class LayoutOptimizer:
    """
    Optimizes facility layout by reassigning functional types to physical locations.

    Uses a greedy iterative approach (best-swap local search) to minimize the
    total weighted travel time of predefined workflows.
    """

    def __init__(self,
                 path_finder: PathFinder,
                 workflow_definitions: Sequence[WorkflowDefinition],
                 config: NetworkConfig,
                 area_tolerance_ratio: float = 0.2):
        """
        Initializes the LayoutOptimizer.

        Args:
            path_finder: An initialized PathFinder instance, used for path generation
                         and time calculation. It must have its travel_times_df loaded.
            workflow_definitions: A list of WorkflowDefinition objects that define
                                  the paths and their importance.
            config: The NetworkConfig object, used to identify non-swappable
                    connection types.
            area_tolerance_ratio: The maximum allowed relative area difference between
                                  two physical locations for them to be considered swappable.
                                  E.g., 0.2 means areas can differ by at most 20%.
        """
        self.path_finder: PathFinder = path_finder
        self.config: NetworkConfig = config
        self.objective_calculator: LayoutObjectiveCalculator = LayoutObjectiveCalculator(
            workflow_definitions, path_finder
        )
        self.area_tolerance_ratio: float = area_tolerance_ratio
        self.all_physical_locations: List[PhysicalLocation] = self._initialize_physical_locations(
        )
        self.swappable_locations: List[PhysicalLocation] = [
            loc for loc in self.all_physical_locations if loc.is_swappable
        ]

        logger.info(f"Optimizer initialized. Found {len(self.all_physical_locations)} total physical locations, "
                    f"{len(self.swappable_locations)} are swappable.")

    def _initialize_physical_locations(self) -> List[PhysicalLocation]:
        """
        Creates PhysicalLocation objects from the PathFinder's data.
        """
        locations = []
        if self.path_finder.travel_times_df is None:
            logger.error(
                "PathFinder's travel_times_df is not loaded. Cannot initialize physical locations.")
            return []

        if '面积' not in self.path_finder.travel_times_df.index:
            logger.warning(
                "Optimizer: '面积' row not found in travel_times_df. Areas will default to 0 for locations.")

        for name_id_str in self.path_finder.all_name_ids:
            parts = name_id_str.split('_', 1)
            original_func_type = parts[0]

            area = 0.0
            if '面积' in self.path_finder.travel_times_df.index and \
               name_id_str in self.path_finder.travel_times_df.columns:
                try:
                    area = float(
                        self.path_finder.travel_times_df.loc['面积', name_id_str])
                except ValueError:
                    logger.warning(
                        f"Could not parse area for {name_id_str}. Defaulting to 0.")

            is_swappable = original_func_type not in self.config.CONNECTION_TYPES
            locations.append(PhysicalLocation(
                name_id_str, original_func_type, area, is_swappable))
        return locations

    def _get_valid_swap_pairs(self) -> List[Tuple[PhysicalLocation, PhysicalLocation]]:
        """Generates pairs of swappable physical locations that satisfy area constraints."""
        valid_pairs: List[Tuple[PhysicalLocation, PhysicalLocation]] = []
        num_swappable = len(self.swappable_locations)
        for i in range(num_swappable):
            for j in range(i + 1, num_swappable):
                loc_A = self.swappable_locations[i]
                loc_B = self.swappable_locations[j]

                area_A, area_B = loc_A.area, loc_B.area
                if area_A > 0 and area_B > 0:
                    if abs(area_A - area_B) / max(area_A, area_B) > self.area_tolerance_ratio:
                        continue
                elif (area_A == 0 and area_B > 0) or (area_A > 0 and area_B == 0):
                    if self.area_tolerance_ratio < 1.0:
                        continue

                valid_pairs.append((loc_A, loc_B))
        return valid_pairs

    def run_optimization(self,
                         initial_assignment: FunctionalAssignment,
                         max_iterations: int = 100
                         ) -> Tuple[FunctionalAssignment, float, List[EvaluatedWorkflowOutcome]]:
        """
        Runs the iterative layout optimization process.

        Args:
            initial_assignment: The starting FunctionalAssignment.
            max_iterations: The maximum number of iterations to perform.

        Returns:
            A tuple containing:
                - best_assignment (FunctionalAssignment): The optimized assignment.
                - best_objective_value (float): The objective value of the best_assignment.
                - best_outcomes (List[EvaluatedWorkflowOutcome]): Detailed outcomes for the best_assignment.
        """
        logger.info("Starting layout optimization...")

        current_assignment = initial_assignment
        current_best_objective, current_best_outcomes = self.objective_calculator.evaluate(
            current_assignment)

        logger.info(f"Initial Objective Value: {current_best_objective:.2f}")
        if current_best_objective == float('inf'):
            logger.error(
                "Initial assignment results in unroutable workflows. Optimization aborted.")
            return current_assignment, current_best_objective, current_best_outcomes

        valid_swap_pairs = self._get_valid_swap_pairs()
        if not valid_swap_pairs:
            logger.warning(
                "No valid pairs of physical locations to swap. Optimization cannot proceed.")
            return current_assignment, current_best_objective, current_best_outcomes

        for iteration in range(max_iterations):
            logger.info(f"--- Iteration {iteration + 1}/{max_iterations} ---")

            best_swap_this_iteration: Optional[Tuple[PhysicalLocation,
                                                     PhysicalLocation]] = None
            best_candidate_assignment_this_iteration: Optional[FunctionalAssignment] = None
            objective_after_best_swap_this_iteration = current_best_objective
            outcomes_for_best_swap_this_iteration = current_best_outcomes

            num_evaluated_swaps = 0
            for loc_A, loc_B in valid_swap_pairs:
                candidate_assignment = current_assignment.apply_functional_swap(
                    loc_A.name_id, loc_B.name_id)

                candidate_objective, candidate_outcomes = self.objective_calculator.evaluate(
                    candidate_assignment)
                num_evaluated_swaps += 1
                if num_evaluated_swaps % 100 == 0:
                    logger.debug(
                        f"  Evaluated {num_evaluated_swaps}/{len(valid_swap_pairs)} potential swaps in iteration {iteration+1}...")

                if candidate_objective < objective_after_best_swap_this_iteration:
                    objective_after_best_swap_this_iteration = candidate_objective
                    best_swap_this_iteration = (loc_A, loc_B)
                    best_candidate_assignment_this_iteration = candidate_assignment
                    outcomes_for_best_swap_this_iteration = candidate_outcomes

            if best_swap_this_iteration and best_candidate_assignment_this_iteration is not None:
                swapped_loc_A, swapped_loc_B = best_swap_this_iteration
                improvement = current_best_objective - objective_after_best_swap_this_iteration

                logger.info(
                    f"Improvement found! Swapping functions at '{swapped_loc_A.name_id}' "
                    f"(original: {swapped_loc_A.original_functional_type}, area: {swapped_loc_A.area:.0f}) and "
                    f"'{swapped_loc_B.name_id}' (original: {swapped_loc_B.original_functional_type}, area: {swapped_loc_B.area:.0f})."
                )
                logger.info(
                    f"Objective improved from {current_best_objective:.2f} to {objective_after_best_swap_this_iteration:.2f} (Gain: {improvement:.2f}).")

                current_assignment = best_candidate_assignment_this_iteration
                current_best_objective = objective_after_best_swap_this_iteration
                current_best_outcomes = outcomes_for_best_swap_this_iteration
            else:
                logger.info(
                    "No further improvement found in this iteration. Optimization stopped.")
                break

        if iteration == max_iterations - 1 and best_swap_this_iteration:
            logger.info(
                f"Reached maximum number of iterations ({max_iterations}).")

        logger.info("Layout optimization finished.")
        logger.info(
            f"Final Best Objective Value: {current_best_objective:.2f}")
        return current_assignment, current_best_objective, current_best_outcomes
