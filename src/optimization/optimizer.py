"""
Module for facility layout optimization by reassigning functional types
to physical locations to minimize total travel times for defined workflows.
"""

import copy
import logging
from typing import List, Dict, Optional, Tuple, Set, Any, Sequence
import pandas as pd
import os
from joblib import Parallel, delayed

from src.analysis.process_flow import PeopleFlow, PathFinder
from src.config import NetworkConfig

# Code Time Profiling
import cProfile
import pstats
profiler = cProfile.Profile()

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
        new_map = self.get_map_copy()  # Start with a copy of the current assignment

        func_type_at_A = self.get_functional_type_at_physical_id(phys_loc_A_id)
        func_type_at_B = self.get_functional_type_at_physical_id(phys_loc_B_id)

        # Handle func_type_at_A (moving func_type_at_A from A to B, if B is involved)
        if func_type_at_A is not None:
            if phys_loc_A_id in new_map.get(func_type_at_A, []):
                new_map[func_type_at_A].remove(phys_loc_A_id)
            # If B is now supposed to host what A had
            if func_type_at_B != func_type_at_A:  # Avoid issues if A and B had same func type initially
                if func_type_at_A not in new_map:  # Should not be needed if remove was successful
                    new_map[func_type_at_A] = []
                # Add B to host A's original function
                if phys_loc_B_id not in new_map[func_type_at_A]:
                    new_map[func_type_at_A].append(phys_loc_B_id)

        # Handle func_type_at_B (moving func_type_at_B from B to A, if A is involved)
        if func_type_at_B is not None:
            if phys_loc_B_id in new_map.get(func_type_at_B, []):
                new_map[func_type_at_B].remove(phys_loc_B_id)
            # If A is now supposed to host what B had
            # if func_type_at_A != func_type_at_B: # Redundant due to first block, but helps clarity
            if func_type_at_B not in new_map:
                new_map[func_type_at_B] = []
            # Add A to host B's original function
            if phys_loc_A_id not in new_map[func_type_at_B]:
                new_map[func_type_at_B].append(phys_loc_A_id)

        # Clean up: remove empty lists and ensure lists are sorted and unique
        # Iterate over a copy of keys for safe deletion
        for func_type in list(new_map.keys()):
            if new_map[func_type]:
                new_map[func_type] = sorted(list(set(new_map[func_type])))
            else:
                # Remove function type if it no longer has any locations
                del new_map[func_type]

        return FunctionalAssignment(new_map)

    def __repr__(self) -> str:
        return f"FunctionalAssignment(map_size={len(self.assignment_map)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionalAssignment):
            return NotImplemented

        # Check if the keys (functional types) are the same
        if set(self.assignment_map.keys()) != set(other.assignment_map.keys()):
            return False

        # Check if the lists of physical IDs for each functional type are the same
        # Comparing sorted lists ensures that order of IDs within the list doesn't affect equality
        for func_type, phys_ids_self in self.assignment_map.items():
            # Already know func_type exists in other due to key check
            phys_ids_other = other.assignment_map.get(func_type)
            if sorted(phys_ids_self) != sorted(phys_ids_other):
                return False
        return True


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

def _calculate_batch_flow_times(flows_batch, path_finder):
    results = []
    for flow in flows_batch:
        time = path_finder.calculate_flow_total_time(flow)
        results.append((time, flow.identify, flow.actual_node_id_sequence))
    return results

class LayoutObjectiveCalculator:
    """
    Calculates the overall objective value for a given functional assignment.
    Supports incremental calculation.
    """

    def __init__(self,
                 workflow_definitions: Sequence[WorkflowDefinition],
                 path_finder: PathFinder,
                 n_jobs_for_flows: int = -1):
        self.workflow_definitions: Sequence[WorkflowDefinition] = workflow_definitions
        self.path_finder: PathFinder = path_finder

        if n_jobs_for_flows == 0:
            self.n_jobs_for_flows = 1
        elif n_jobs_for_flows == -1:
            cpu_count = os.cpu_count()
            self.n_jobs_for_flows = cpu_count if cpu_count is not None else 1
        else:
            self.n_jobs_for_flows = n_jobs_for_flows

        logger.info(
            f"LayoutObjectiveCalculator initialized with n_jobs_for_flows={self.n_jobs_for_flows}")

        self.cached_evaluated_outcomes: Optional[List[EvaluatedWorkflowOutcome]] = None
        self.cached_total_objective_value: Optional[float] = None
        # Stores the FunctionalAssignment object
        self.cached_assignment_obj: Optional[FunctionalAssignment] = None

    def _evaluate_single_workflow_flows(self,
                                        generated_flows: List[PeopleFlow],
                                        workflow_id_for_logging: str
                                        ) -> Tuple[float, Optional[PeopleFlow], int, Optional[List[float]]]:
        # ... (Implementation from the previous response, unchanged) ...
        workflow_average_time: float = float('inf')
        shortest_flow: Optional[PeopleFlow] = None
        shortest_time_val: float = float('inf')
        valid_flow_times: List[float] = []
        num_paths_considered: int = 0

        if not generated_flows:
            logger.debug(
                f"Single WF Eval: Workflow '{workflow_id_for_logging}' generated no flows. Avg Time is Inf.")
            return float('inf'), None, 0, None

        try:
            parallel_results_wf: List[Tuple[Optional[float], Any, List[str]]] = [
            ]
            backend_to_use = 'loky'
            if self.n_jobs_for_flows > 1 and len(generated_flows) > 1:
                batch_size = 600
                batch_results = []
                flow_batches = [generated_flows[i:i + batch_size] for i in range(0, len(generated_flows), batch_size)]
                # Ensure _calculate_batch_flow_times is globally defined or properly imported
                with Parallel(n_jobs=self.n_jobs_for_flows, backend=backend_to_use) as parallel:
                    results_from_parallel = parallel(
                        delayed(_calculate_batch_flow_times)(
                            flow_batch, self.path_finder)
                        for flow_batch in flow_batches
                    )
                for single_batch_result in results_from_parallel:
                    batch_results.extend(single_batch_result)
                parallel_results_wf = batch_results
            else:
                for flow in generated_flows:
                    parallel_results_wf.append(
                        _calculate_flow_time_joblib_task(flow, self.path_finder))
        except Exception as e:
            logger.error(
                f"Error during parallel flow calculation for single workflow '{workflow_id_for_logging}': {e}. Falling back to sequential.", exc_info=True)
            parallel_results_wf = [_calculate_flow_time_joblib_task(
                flow, self.path_finder) for flow in generated_flows]

        all_paths_routable = True
        temp_flow_times: List[float] = []
        flow_map_for_update: Dict[Tuple[Any, Tuple[str, ...]], PeopleFlow] = {
            (f.identify, tuple(f.actual_node_id_sequence)): f for f in generated_flows
        }

        for flow_time, flow_id, flow_seq_list in parallel_results_wf:
            flow_seq_tuple = tuple(flow_seq_list)
            original_flow = flow_map_for_update.get((flow_id, flow_seq_tuple))
            if original_flow:
                original_flow.update_total_time(
                    flow_time if flow_time is not None else float('inf'))

            if flow_time is None:
                all_paths_routable = False
                break
            temp_flow_times.append(flow_time)
            if original_flow and flow_time < shortest_time_val:
                shortest_time_val = flow_time
                shortest_flow = original_flow

        if all_paths_routable:
            if temp_flow_times:
                valid_flow_times = temp_flow_times
                workflow_average_time = sum(
                    valid_flow_times) / len(valid_flow_times)
                num_paths_considered = len(valid_flow_times)
        else:
            workflow_average_time = float('inf')
            shortest_flow = None
            valid_flow_times = []
            num_paths_considered = 0

        return workflow_average_time, shortest_flow, num_paths_considered, (valid_flow_times if workflow_average_time != float('inf') else None)

    def _perform_full_evaluation(self,
                                 current_assignment: FunctionalAssignment
                                 ) -> Tuple[float, List[EvaluatedWorkflowOutcome]]:
        # ... (Implementation from the previous response, unchanged) ...
        logger.debug(
            f"Performing full evaluation for assignment (map_size={len(current_assignment.assignment_map)}).")
        total_weighted_time: float = 0.0
        evaluated_outcomes: List[EvaluatedWorkflowOutcome] = []

        for wf_def in self.workflow_definitions:
            generated_flows = self.path_finder.generate_flows(
                workflow_names=wf_def.functional_sequence,
                workflow_identifier=wf_def.workflow_id,
                custom_assignment_map=current_assignment.get_map_copy()
            )

            wf_avg_time, wf_shortest_flow, wf_num_paths, wf_all_times = \
                self._evaluate_single_workflow_flows(
                    generated_flows, wf_def.workflow_id)

            outcome = EvaluatedWorkflowOutcome(
                workflow_definition=wf_def,
                average_time=wf_avg_time,
                shortest_flow=wf_shortest_flow,
                num_paths_considered=wf_num_paths,
                all_path_times=wf_all_times
            )
            evaluated_outcomes.append(outcome)

            if total_weighted_time != float('inf'):
                if wf_avg_time == float('inf'):
                    total_weighted_time = float('inf')
                else:
                    total_weighted_time += wf_avg_time * wf_def.weight

        return total_weighted_time, evaluated_outcomes

    def evaluate(self,
                 assignment_to_evaluate: FunctionalAssignment,
                 base_assignment_for_cache: Optional[FunctionalAssignment] = None,
                 changed_locations: Optional[Tuple[str, str]] = None
                 ) -> Tuple[float, List[EvaluatedWorkflowOutcome]]:
        """Evaluates the given FunctionalAssignment.

        If `base_assignment_for_cache` and `changed_locations` are provided, and
        `base_assignment_for_cache` matches the internally cached assignment object,
        an incremental update is attempted. Otherwise, a full evaluation is performed.
        """
        can_do_incremental = (
            base_assignment_for_cache is not None and
            changed_locations is not None and
            self.cached_assignment_obj is not None and
            self.cached_evaluated_outcomes is not None and
            # Use the __eq__ method of FunctionalAssignment for comparison
            self.cached_assignment_obj == base_assignment_for_cache
        )

        if not can_do_incremental:
            if base_assignment_for_cache is not None and changed_locations is not None and self.cached_assignment_obj is not None:
                # More specific logging if an attempt at incremental was made but failed due to cache mismatch
                logger.debug(
                    "Cache mismatch for incremental: `base_assignment_for_cache` does not match `cached_assignment_obj`. Performing full evaluation.")
            else:
                logger.debug(
                    "Performing full evaluation (no valid base for incremental or cache empty).")
            total_objective, outcomes = self._perform_full_evaluation(
                assignment_to_evaluate)
        else:
            # Incremental calculation
            logger.debug(
                f"Attempting incremental evaluation. Changed locations: {changed_locations}")
            loc_A_id, loc_B_id = changed_locations

            func_type_at_A_prev = base_assignment_for_cache.get_functional_type_at_physical_id(
                loc_A_id)
            func_type_at_B_prev = base_assignment_for_cache.get_functional_type_at_physical_id(
                loc_B_id)
            func_type_at_A_curr = assignment_to_evaluate.get_functional_type_at_physical_id(
                loc_A_id)
            func_type_at_B_curr = assignment_to_evaluate.get_functional_type_at_physical_id(
                loc_B_id)

            affected_functional_types: Set[str] = set()
            if func_type_at_A_prev:
                affected_functional_types.add(func_type_at_A_prev)
            if func_type_at_B_prev:
                affected_functional_types.add(func_type_at_B_prev)
            if func_type_at_A_curr:
                affected_functional_types.add(func_type_at_A_curr)
            if func_type_at_B_curr:
                affected_functional_types.add(func_type_at_B_curr)

            logger.debug(f"Swap details for incremental: LocA ({loc_A_id}): {func_type_at_A_prev} -> {func_type_at_A_curr}. "
                         f"LocB ({loc_B_id}): {func_type_at_B_prev} -> {func_type_at_B_curr}.")
            logger.debug(
                f"Functional types triggering re-evaluation: {affected_functional_types}")

            new_evaluated_outcomes: List[EvaluatedWorkflowOutcome] = []

            for i, wf_def in enumerate(self.workflow_definitions):
                workflow_is_potentially_affected = any(
                    func_step in affected_functional_types for func_step in wf_def.functional_sequence
                )

                if workflow_is_potentially_affected:
                    # logger.debug(f"Workflow '{wf_def.workflow_id}' is affected by swap. Re-evaluating.")
                    generated_flows_wf = self.path_finder.generate_flows(
                        workflow_names=wf_def.functional_sequence,
                        workflow_identifier=wf_def.workflow_id,
                        custom_assignment_map=assignment_to_evaluate.get_map_copy()
                    )
                    wf_avg_time, wf_shortest_flow, wf_num_paths, wf_all_times = \
                        self._evaluate_single_workflow_flows(
                            generated_flows_wf, wf_def.workflow_id)

                    current_outcome = EvaluatedWorkflowOutcome(
                        workflow_definition=wf_def, average_time=wf_avg_time,
                        shortest_flow=wf_shortest_flow, num_paths_considered=wf_num_paths,
                        all_path_times=wf_all_times)
                    new_evaluated_outcomes.append(current_outcome)
                else:
                    # Ensure self.cached_evaluated_outcomes is not None (already checked by can_do_incremental)
                    new_evaluated_outcomes.append(
                        self.cached_evaluated_outcomes[i])  # type: ignore

            current_total_weighted_time_calc = 0.0
            has_inf_path_incrementally = False
            for outcome in new_evaluated_outcomes:
                if outcome.average_time == float('inf'):
                    has_inf_path_incrementally = True
                    break
                current_total_weighted_time_calc += outcome.average_time * \
                    outcome.workflow_definition.weight

            if has_inf_path_incrementally:
                total_objective = float('inf')
            else:
                total_objective = current_total_weighted_time_calc

            outcomes = new_evaluated_outcomes
            # Log difference for significant changes to monitor incremental logic
            if self.cached_total_objective_value is not None and abs(total_objective - self.cached_total_objective_value) > 1e-5:
                logger.info(
                    f"Incremental eval result. Prev Obj: {self.cached_total_objective_value:.2f}, New Obj: {total_objective:.2f}. Diff: {total_objective - self.cached_total_objective_value:.2f}")
            elif self.cached_total_objective_value is None:
                logger.info(
                    f"Incremental eval result (no prev obj for diff). New Obj: {total_objective:.2f}")

        # Update cache with the results of *this* evaluation, for the *assignment_to_evaluate*
        self.cached_total_objective_value = total_objective
        self.cached_evaluated_outcomes = outcomes
        # Store a deepcopy to prevent external modifications to assignment_to_evaluate from affecting the cache
        self.cached_assignment_obj = copy.deepcopy(assignment_to_evaluate)

        return total_objective, outcomes

    def reset_cache(self):
        """Resets the cache. Call when the base state for comparison changes significantly."""
        logger.info("Resetting LayoutObjectiveCalculator cache.")
        self.cached_evaluated_outcomes = None
        self.cached_total_objective_value = None
        self.cached_assignment_obj = None


class LayoutOptimizer:
    """
    Optimizes facility layout by reassigning functional types to physical locations.
    Uses a greedy iterative approach (best-swap local search).
    """

    def __init__(self,
                 path_finder: PathFinder,
                 workflow_definitions: Sequence[WorkflowDefinition],
                 config: NetworkConfig,  # Used by _initialize_physical_locations
                 area_tolerance_ratio: float = 0.2):
        self.path_finder: PathFinder = path_finder
        # For BAN_TYPES, CONNECTION_TYPES etc.
        self.config: NetworkConfig = config
        self.objective_calculator: LayoutObjectiveCalculator = LayoutObjectiveCalculator(
            workflow_definitions, path_finder, n_jobs_for_flows=config.N_JOBS_FOR_OPTIMIZER_FLOWS if hasattr(
                config, 'N_JOBS_FOR_OPTIMIZER_FLOWS') else -1
        )  # Pass n_jobs from config if available
        self.area_tolerance_ratio: float = area_tolerance_ratio
        self.all_physical_locations: List[PhysicalLocation] = self._initialize_physical_locations(
        )
        self.swappable_locations: List[PhysicalLocation] = [
            loc for loc in self.all_physical_locations if loc.is_swappable
        ]
        logger.info(f"Optimizer initialized. Found {len(self.all_physical_locations)} total physical locations, "
                    f"{len(self.swappable_locations)} are swappable.")

    def _initialize_physical_locations(self) -> List[PhysicalLocation]:
        locations = []
        if self.path_finder.travel_times_df is None:
            logger.error(
                "PathFinder's travel_times_df is not loaded. Cannot initialize physical locations.")
            return []
        # Ensure '面积' key exists or handle gracefully
        has_area_info = '面积' in self.path_finder.travel_times_df.index

        for name_id_str in self.path_finder.all_name_ids:
            parts = name_id_str.split('_', 1)
            original_func_type = parts[0]
            area = 0.0  # Default area
            if has_area_info and name_id_str in self.path_finder.travel_times_df.columns:
                try:
                    area_val = self.path_finder.travel_times_df.loc['面积', name_id_str]
                    if pd.notna(area_val):  # Check for NaN
                        area = float(area_val)
                    else:
                        logger.debug(
                            f"Area for {name_id_str} is NaN. Defaulting to 0.")
                except ValueError:
                    logger.warning(
                        f"Could not parse area for {name_id_str}. Defaulting to 0.")
                except KeyError:  # Should be caught by `name_id_str in ...columns` but as a safeguard
                    logger.warning(
                        f"KeyError for area of {name_id_str}. Defaulting to 0.")

            # Determine swappability (e.g., '门' is a connection type, not swappable for a room function)
            # This uses self.config.CONNECTION_TYPES from your NetworkConfig
            is_swappable = original_func_type not in self.config.CONNECTION_TYPES and \
                original_func_type not in getattr(self.config, 'BAN_TYPES_FOR_SWAP', [
                ])  # Example of more specific ban types

            locations.append(PhysicalLocation(
                name_id_str, original_func_type, area, is_swappable))
        return locations

    def _get_valid_swap_pairs(self) -> List[Tuple[PhysicalLocation, PhysicalLocation]]:
        valid_pairs: List[Tuple[PhysicalLocation, PhysicalLocation]] = []
        num_swappable = len(self.swappable_locations)
        for i in range(num_swappable):
            for j in range(i + 1, num_swappable):
                loc_A = self.swappable_locations[i]
                loc_B = self.swappable_locations[j]

                # Ensure original functions are different, or allow swapping same if it makes sense
                # For now, let's assume swapping locations with the same original function is allowed
                # if their areas match, effectively just moving a function between two identical spots.
                # If they must have different functions to be considered a meaningful swap, add:
                # if loc_A.original_functional_type == loc_B.original_functional_type:
                #     continue

                area_A, area_B = loc_A.area, loc_B.area
                # Handle area_A or area_B being 0 or very small if that's possible
                if area_A > 1e-6 and area_B > 1e-6:  # Use a small epsilon for float comparison
                    if abs(area_A - area_B) / max(area_A, area_B) > self.area_tolerance_ratio:
                        continue
                elif (area_A <= 1e-6 and area_B > 1e-6) or \
                     (area_A > 1e-6 and area_B <= 1e-6):
                    # If one area is (near) zero and the other isn't,
                    # only allow swap if tolerance is very high (e.g., >= 1.0 means ignore area diff)
                    if self.area_tolerance_ratio < 1.0:  # A stricter interpretation
                        continue
                # If both areas are (near) zero, they are considered compatible in terms of area

                valid_pairs.append((loc_A, loc_B))
        logger.info(
            f"Generated {len(valid_pairs)} valid swap pairs based on area tolerance {self.area_tolerance_ratio}.")
        return valid_pairs

    def run_optimization(self,
                         initial_assignment: FunctionalAssignment,
                         max_iterations: int = 100
                         ) -> Tuple[FunctionalAssignment, float, List[EvaluatedWorkflowOutcome]]:
        logger.info("Starting layout optimization...")

        current_best_assignment = copy.deepcopy(
            initial_assignment)  # Start with a copy

        # Initial full evaluation and cache setup
        self.objective_calculator.reset_cache()  # Clear any old cache
        current_best_objective, current_best_outcomes = self.objective_calculator.evaluate(
            assignment_to_evaluate=current_best_assignment
            # No base_assignment or changed_locations, so it performs a full evaluation
            # and populates its internal cache with current_best_assignment
        )
        logger.info(f"Initial Objective Value: {current_best_objective:.2f}")

        if current_best_objective == float('inf'):
            logger.error(
                "Initial assignment results in unroutable workflows. Optimization aborted.")
            return current_best_assignment, current_best_objective, current_best_outcomes

        valid_swap_pairs = self._get_valid_swap_pairs()
        if not valid_swap_pairs:
            logger.warning(
                "No valid pairs of physical locations to swap. Optimization cannot proceed.")
            return current_best_assignment, current_best_objective, current_best_outcomes

        for iteration in range(max_iterations):
            logger.info(f"--- Iteration {iteration + 1}/{max_iterations} ---")

            best_swap_candidate_in_iteration: Optional[FunctionalAssignment] = None
            # Initialize with current best
            best_swap_objective_in_iteration = current_best_objective
            best_swap_outcomes_in_iteration = current_best_outcomes
            best_swapped_pair_info: Optional[Tuple[PhysicalLocation,
                                                   PhysicalLocation]] = None

            num_evaluated_swaps = 0
            for loc_A, loc_B in valid_swap_pairs:
                # Create a candidate assignment by swapping functions based on the *current_best_assignment*
                candidate_assignment = current_best_assignment.apply_functional_swap(
                    loc_A.name_id, loc_B.name_id
                )

                # Evaluate this candidate, providing the base for incremental calculation
                candidate_objective, candidate_outcomes = self.objective_calculator.evaluate(
                    assignment_to_evaluate=candidate_assignment,
                    base_assignment_for_cache=current_best_assignment,  # The state *before* this swap
                    # The specific swap made
                    changed_locations=(loc_A.name_id, loc_B.name_id)
                )
                num_evaluated_swaps += 1
                if num_evaluated_swaps % 200 == 0 and num_evaluated_swaps > 0:  # Log progress periodically
                    logger.debug(
                        f"  Evaluated {num_evaluated_swaps}/{len(valid_swap_pairs)} potential swaps in iteration {iteration+1}...")

                if candidate_objective < best_swap_objective_in_iteration:
                    best_swap_objective_in_iteration = candidate_objective
                    best_swap_candidate_in_iteration = candidate_assignment  # This is a new object
                    best_swap_outcomes_in_iteration = candidate_outcomes
                    best_swapped_pair_info = (loc_A, loc_B)

            if best_swap_candidate_in_iteration is not None and best_swap_objective_in_iteration < current_best_objective:
                # An improvement was found in this iteration
                improvement = current_best_objective - best_swap_objective_in_iteration
                # Should not be None if candidate is not None
                sw_loc_A, sw_loc_B = best_swapped_pair_info

                logger.info(
                    f"Improvement found! Swapping functions related to '{sw_loc_A.name_id}' "
                    f"(Orig Func: {sw_loc_A.original_functional_type}, Area: {sw_loc_A.area:.0f}) and "
                    f"'{sw_loc_B.name_id}' (Orig Func: {sw_loc_B.original_functional_type}, Area: {sw_loc_B.area:.0f})."
                )
                logger.info(
                    f"Objective improved from {current_best_objective:.2f} to {best_swap_objective_in_iteration:.2f} (Gain: {improvement:.2f}).")

                # Update the current best state for the next iteration
                current_best_assignment = copy.deepcopy(
                    best_swap_candidate_in_iteration)  # Store the new best assignment
                current_best_objective = best_swap_objective_in_iteration
                current_best_outcomes = best_swap_outcomes_in_iteration

                # The objective_calculator's cache was already updated to reflect best_swap_candidate_in_iteration
                # when it was evaluated and found to be the best in the inner loop, because evaluate() always
                # updates its cache with the assignment_to_evaluate.
                # So, when the next iteration starts, current_best_assignment will match the calculator's cache.

            else:
                logger.info(
                    "No further improvement found in this iteration. Optimization stopped.")
                break  # Stop if no improvement in this iteration

        if iteration == max_iterations - 1 and best_swap_candidate_in_iteration is not None and best_swap_objective_in_iteration < current_best_objective:
            logger.info(
                f"Reached maximum number of iterations ({max_iterations}), but was still improving.")
        elif iteration == max_iterations - 1:
            logger.info(
                f"Reached maximum number of iterations ({max_iterations}).")

        logger.info("Layout optimization finished.")
        logger.info(
            f"Final Best Objective Value: {current_best_objective:.2f}")
        # The current_best_outcomes corresponds to current_best_assignment
        return current_best_assignment, current_best_objective, current_best_outcomes
