import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from itertools import product # 用于生成笛卡尔积，简化路径组合

from src.analysis.word_detect import WordDetect
from src.config import NetworkConfig

logger = logging.getLogger(__name__)

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
            self._cached_hash = hash((self.identify, tuple(self.actual_node_id_sequence)))
        return self._cached_hash
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the PeopleFlow instance.
        """
        flow_str = " -> ".join(self.actual_node_id_sequence) if self.actual_node_id_sequence else "Empty"
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
    def __init__(self, csv_filepath: str = None, config: NetworkConfig = None):
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
            self.csv_filepath: str = csv_filepath
        else:
            self.csv_filepath: str = self.config.RESULT_PATH / 'super_network_travel_times.csv'
        
        self.name_to_ids_map: Dict[str, List[str]] = {}
        self.all_name_ids: Set[str] = set()
        self.travel_times_df: Optional[pd.DataFrame] = None
        self.word_detect = WordDetect()
        self._load_and_process_csv()

    def _parse_name_id(self, name_id_str: str) -> Tuple[str, str]:
        """
        Parses a "Name_ID" string into its name and ID components.

        Args:
            name_id_str (str): The string in "Name_ID" format (e.g., "门_11072").

        Returns:
            Tuple[str, str]: (name, id)
        """
        parts = name_id_str.split('_', 1)
        name = parts[0]
        node_id = parts[1] if len(parts) > 1 else name # if no '_', id is same as name
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
        if df.index.name is not None or df.index.dtype == 'object': # Check if index is meaningful
            all_headers.extend(list(df.index))

        unique_name_ids = sorted(list(set(all_headers))) # Get unique Name_IDs

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

    def _transform_workflow_name(self, workflow_names: List[str]) -> str:
        """
        Transforms a workflow name by detecting the nearest word in the CSV.
        """
        all_type_names = self.config.ALL_TYPES
        workflow_names = [self.word_detect.detect_nearest_word(name, all_type_names) for name in workflow_names]
        return workflow_names
    
    def generate_flows(self, workflow_names: List[str], workflow_identifier: Any = 0) -> List[PeopleFlow]:
        """
        Generates all possible PeopleFlow objects for a given sequence of node names.

        Args:
            workflow_names (List[str]): An ordered list of node names representing
                the desired workflow (e.g., ['门', '妇科', '门']).
            workflow_identifier (Any, optional): An identifier to assign to the
                generated PeopleFlow objects. Defaults to 0.

        Returns:
            List[PeopleFlow]: A list of all valid PeopleFlow objects representing
            the different path combinations. Returns empty list if a name in
            workflow_names is not found or if the workflow is empty.
        """
        if not workflow_names:
            return []
        
        # Convert node names in `workflow_names` to the closest node types.
        workflow_names = self._transform_workflow_name(workflow_names)

        possible_ids_per_step: List[List[str]] = []
        for name in workflow_names:
            ids_for_name = self.name_to_ids_map.get(name)
            if not ids_for_name:
                # If any name in the workflow is not found in our CSV data,
                # no valid paths can be generated for this workflow.
                print(f"Warning: Node name '{name}' not found in CSV data. No flows generated for this workflow.")
                return []
            possible_ids_per_step.append(ids_for_name)

        # Use itertools.product to get all combinations of IDs
        # Example: if possible_ids_per_step = [['ID_A1', 'ID_A2'], ['ID_B1'], ['ID_C1', 'ID_C2']]
        # product will yield:
        #   ('ID_A1', 'ID_B1', 'ID_C1')
        #   ('ID_A1', 'ID_B1', 'ID_C2')
        #   ('ID_A2', 'ID_B1', 'ID_C1')
        #   ('ID_A2', 'ID_B1', 'ID_C2')
        all_possible_id_sequences = product(*possible_ids_per_step)

        generated_flows: List[PeopleFlow] = []
        for id_sequence_tuple in all_possible_id_sequences:
            # id_sequence_tuple is a tuple of Name_ID strings
            # Convert it to a list for PeopleFlow constructor
            flow = PeopleFlow(identify=workflow_identifier, actual_node_id_sequence=list(id_sequence_tuple))
            workflow_identifier += 1
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
                return float(time) # Ensure it's a float
            else:
                logger.warning(f"Warning: Node {from_name_id} or {to_name_id} not in travel matrix.")
                return None # Or handle as an error, e.g., raise ValueError
        except KeyError:
            logger.warning(f"KeyError: Node {from_name_id} or {to_name_id} not found in travel matrix.")
            return None
        except ValueError: # If conversion to float fails for some reason
            logger.warning(f"ValueError: Travel time for {from_name_id} to {to_name_id} is not a valid number.")
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
            return 0.0 # No travel for a single point

        total_time = 0.0
        for i in range(len(flow.actual_node_id_sequence) - 1):
            from_node = flow.actual_node_id_sequence[i]
            to_node = flow.actual_node_id_sequence[i+1]
            segment_time = self.get_travel_time(from_node, to_node)
            if segment_time is None:
                logger.warning(f"Warning: No travel time for segment {from_node} -> {to_node} in flow {flow.identify}.")
                return None # Or handle this case differently, e.g. infinite time
            total_time += segment_time
        flow.update_total_time(total_time)
        return total_time
