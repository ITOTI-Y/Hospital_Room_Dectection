import pandas as pd
from src.rl_optimizer.utils.setup import setup_logger
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any, Mapping
from itertools import product

from src.analysis.word_detect import WordDetect
from src.config import NetworkConfig

logger = setup_logger(__name__)


class PeopleFlow:
    """
    Represents a specific, resolved flow of an entity through a sequence of nodes.

    Each node in the flow is represented by its unique "Name_ID" string.
    This class primarily stores a fully determined path.

    Attributes:
        identify (Any): An identifier for this specific flow instance or the
            workflow it belongs to.
        actual_node_id_sequence (List[str]): The complete, ordered list of
            "Name_ID" strings representing the flow.
        _cached_hash (Optional[int]): Cached hash value for performance.
    """

    def __init__(self, identify: Any, actual_node_id_sequence: List[str]):
        """
        Initializes a PeopleFlow instance with a specific node ID sequence.

        Args:
            identify (Any): An identifier for this flow.
            actual_node_id_sequence (List[str]): The fully resolved sequence
                of "Name_ID" strings for this flow.
        """
        self.identify: Any = identify
        self.actual_node_id_sequence: List[str] = actual_node_id_sequence
        self.total_time: Optional[float] = None
        self._cached_hash: Optional[int] = None

    def update_total_time(self, total_time: float) -> None:
        """
        Updates the total travel time for this flow.
        """
        self.total_time = total_time

    @property
    def start_node_id(self) -> Optional[str]:
        """Optional[str]: The 'Name_ID' of the first node in the sequence, if any."""
        return self.actual_node_id_sequence[0] if self.actual_node_id_sequence else None

    @property
    def end_node_id(self) -> Optional[str]:
        """Optional[str]: The 'Name_ID' of the last node in the sequence, if any."""
        if len(self.actual_node_id_sequence) > 0:
            return self.actual_node_id_sequence[-1]
        return None

    @property
    def intermediate_node_ids(self) -> List[str]:
        """List[str]: A list of 'Name_ID's for intermediate nodes."""
        if len(self.actual_node_id_sequence) > 2:
            return self.actual_node_id_sequence[1:-1]
        return []

    def __eq__(self, other: object) -> bool:
        """
        Checks equality based on the `identify` and `actual_node_id_sequence`.
        """
        if not isinstance(other, PeopleFlow):
            return NotImplemented
        return (self.identify == other.identify and
                self.actual_node_id_sequence == other.actual_node_id_sequence)

    def __hash__(self) -> int:
        """
        Computes hash based on `identify` and the tuple of `actual_node_id_sequence`.
        """
        if self._cached_hash is None:
            # Making sequence a tuple makes it hashable
            self._cached_hash = hash(
                (self.identify, tuple(self.actual_node_id_sequence)))
        return self._cached_hash

    def __repr__(self) -> str:
        """
        Returns a string representation of the PeopleFlow instance.
        """
        flow_str = " -> ".join(
            self.actual_node_id_sequence) if self.actual_node_id_sequence else "Empty"
        return (
            f"PeopleFlow(identify={self.identify}, "
            f"path=[{flow_str}], "
            f"total_time={self.total_time})"
        )


class PathFinder:
    """
    Finds all possible PeopleFlow paths based on a sequence of node names
    and a CSV file defining available "Name_ID"s and their connections/travel times.
    """

    def __init__(self, csv_filepath: str | None  = None, config: NetworkConfig | None = None):
        """
        Initializes the PathFinder by loading and processing the CSV file.

        Args:
            csv_filepath (str): Path to the CSV file. The CSV should have
                "Name_ID" formatted strings as column headers and row index.
        """
        if config:
            self.config: NetworkConfig = config
        else:
            self.config: NetworkConfig = NetworkConfig()

        if csv_filepath:
            self.csv_filepath: Path = Path(csv_filepath)
        else:
            self.csv_filepath: Path = self.config.RESULT_PATH / 'super_network_travel_times.csv'

        self.name_to_ids_map: Dict[str, List[str]] = {}
        self.all_name_ids: Set[str] = set()
        self.travel_times_df: Optional[pd.DataFrame] = None
        self._load_and_process_csv()

    def _parse_name_id(self, name_id_str: str) -> Tuple[str, str]:
        """
        Parses a "Name_ID" string into its name and ID components.

        Args:
            name_id_str (str): The string in "Name_ID" format (e.g., "Door_11072").

        Returns:
            Tuple[str, str]: (name, id)
        """
        parts = name_id_str.split('_', 1)
        name = parts[0]
        # if no '_', id is same as name
        node_id = parts[1] if len(parts) > 1 else name
        return name, node_id

    def _load_and_process_csv(self) -> None:
        """
        Loads the CSV, extracts "Name_ID"s, and populates the name_to_ids_map.
        """
        try:
            # Assuming the first column is the index and also contains Name_ID
            df = pd.read_csv(self.csv_filepath, index_col=0)
            self.travel_times_df = df
        except FileNotFoundError:
            logger.error(f"Error: CSV file not found at {self.csv_filepath}")
            return
        except Exception as e:
            logger.error(f"Error loading CSV {self.csv_filepath}: {e}")
            return

        # Process column headers (assuming they are the primary source of Name_IDs)
        # and also the index if it's different or more comprehensive
        all_headers = list(df.columns)
        if df.index.name is not None or df.index.dtype == 'object':  # Check if index is meaningful
            all_headers.extend(list(df.index))

        unique_name_ids = sorted(list(set(all_headers)))  # Get unique Name_IDs

        for name_id_str in unique_name_ids:
            if not isinstance(name_id_str, str) or '_' not in name_id_str:
                # Skip non-string headers or headers not in expected format, like "面积"
                # print(f"Skipping header/index: {name_id_str} as it's not in Name_ID format.")
                continue

            self.all_name_ids.add(name_id_str)
            name, _ = self._parse_name_id(name_id_str)
            if name not in self.name_to_ids_map:
                self.name_to_ids_map[name] = []
            # Ensure no duplicates if a Name_ID appears in both columns and index
            if name_id_str not in self.name_to_ids_map[name]:
                self.name_to_ids_map[name].append(name_id_str)

    def transform_workflow_name(self, workflow_names: List[str]) -> str:
        """
        Transforms a workflow name by detecting the nearest word in the CSV.
        """
        word_detect = WordDetect(config=self.config)
        all_type_names = self.config.ALL_TYPES
        return word_detect.detect_nearest_word(workflow_names, all_type_names)

    def generate_flows(self,
                       workflow_names: List[str],
                       workflow_identifier: Any = 0,
                       # Changed Dict to Mapping for broader type hint
                       custom_assignment_map: Optional[Mapping[str,
                                                               List[str]]] = None
                       ) -> List[PeopleFlow]:
        """Generates all possible PeopleFlow objects for a given sequence of node names.

        Uses custom_assignment_map if provided for resolving functional names to
        physical Name_IDs, otherwise defaults to the instance's self.name_to_ids_map
        which is loaded from the CSV during initialization.

        Args:
            workflow_names: An ordered list of functional node names.
            workflow_identifier: An identifier for the generated PeopleFlow objects.
            custom_assignment_map: An optional mapping where keys are functional type
                names (e.g., '妇科') and values are lists of physical Name_ID strings
                (e.g., ['妇科_101', '妇科_102']) that currently fulfill that function.

        Returns:
            A list of PeopleFlow objects.
        """
        if not workflow_names:
            return []

        # Use the provided custom_assignment_map if available, otherwise use the default one
        assignment_to_use = custom_assignment_map if custom_assignment_map is not None else self.name_to_ids_map

        # Convert node names in `workflow_names` to the closest node types (e.g., from config.ALL_TYPES)
        # This step helps normalize user input if workflow_names might not exactly match official types.

        possible_ids_per_step: List[List[str]] = []
        for name in workflow_names:
            # Use the determined assignment map
            ids_for_name = assignment_to_use.get(name)
            if not ids_for_name:
                logger.warning(
                    f"Functional type '{name}' (from original '{workflow_names[workflow_names.index(name)] if name in workflow_names else 'N/A'}') "
                    f"not found in the current assignment map. Workflow '{workflow_identifier}' cannot be fully resolved."
                )
                return []  # Cannot generate flows if a step is unresolvable
            possible_ids_per_step.append(ids_for_name)

        # Use itertools.product to get all combinations of Name_IDs
        all_possible_id_sequences = product(*possible_ids_per_step)

        generated_flows: List[PeopleFlow] = []
        # Use a local counter for unique flow IDs within this specific generation call,
        # prefixed by the workflow_identifier.
        flow_counter_for_identifier = 0
        for id_sequence_tuple in all_possible_id_sequences:
            # Create a more unique identifier for the flow if multiple flows are generated for one workflow_identifier
            current_flow_id = f"{workflow_identifier}_{flow_counter_for_identifier}" \
                if isinstance(workflow_identifier, str) else (workflow_identifier, flow_counter_for_identifier)

            flow = PeopleFlow(identify=current_flow_id,
                              actual_node_id_sequence=list(id_sequence_tuple))
            flow_counter_for_identifier += 1
            generated_flows.append(flow)

        return generated_flows

    def get_travel_time(self, from_name_id: str, to_name_id: str) -> Optional[float]:
        """
        Gets the travel time between two specific "Name_ID" nodes.

        Args:
            from_name_id (str): The "Name_ID" of the source node.
            to_name_id (str): The "Name_ID" of the target node.

        Returns:
            Optional[float]: The travel time if connection exists, else None.
                           Returns None also if dataframe isn't loaded.
        """
        if self.travel_times_df is None:
            logger.warning("Warning: Travel times DataFrame not loaded.")
            return None
        try:
            # Ensure both from_name_id and to_name_id are in the DataFrame's
            # index and columns to avoid KeyError
            if from_name_id in self.travel_times_df.index and \
               to_name_id in self.travel_times_df.columns:
                time = self.travel_times_df.loc[from_name_id, to_name_id]
                return float(time)  # Ensure it's a float
            else:
                logger.warning(
                    f"Warning: Node {from_name_id} or {to_name_id} not in travel matrix.")
                return None  # Or handle as an error, e.g., raise ValueError
        except KeyError:
            logger.warning(
                f"KeyError: Node {from_name_id} or {to_name_id} not found in travel matrix.")
            return None
        except ValueError:  # If conversion to float fails for some reason
            logger.warning(
                f"ValueError: Travel time for {from_name_id} to {to_name_id} is not a valid number.")
            return None

    def calculate_flow_total_time(self, flow: PeopleFlow) -> Optional[float]:
        """
        Calculates the total travel time for a given PeopleFlow.

        Args:
            flow (PeopleFlow): The PeopleFlow object.

        Returns:
            Optional[float]: The total travel time for the flow.
                             Returns None if any segment has no travel time.
                             Returns 0.0 for single-node flows.
        """
        if not flow.actual_node_id_sequence:
            return 0.0
        if len(flow.actual_node_id_sequence) == 1:
            return 0.0  # No travel for a single point

        total_time = 0.0
        for i in range(len(flow.actual_node_id_sequence) - 1):
            from_node = flow.actual_node_id_sequence[i]
            to_node = flow.actual_node_id_sequence[i+1]
            segment_time = self.get_travel_time(from_node, to_node)
            if segment_time is None:
                logger.warning(
                    f"Warning: No travel time for segment {from_node} -> {to_node} in flow {flow.identify}.")
                return None  # Or handle this case differently, e.g. infinite time
            total_time += segment_time
        flow.update_total_time(total_time)
        return total_time
