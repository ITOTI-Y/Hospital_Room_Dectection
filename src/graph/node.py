"""Defines the Node class for the network graph."""

from typing import Tuple, Optional

class Node:
    """
    Represents a node in the network graph.

    Attributes:
        id (int): A unique identifier for the node.
        node_type (str): The type of the node (e.g., 'Room', 'Door', 'Corridor').
        pos (Tuple[int, int, int]): The (x, y, z) coordinates of the node.
        time (float): The time cost associated with traversing this node.
        door_type (Optional[str]): Specifies the type of door connection
            (e.g., 'room', 'in', 'out'), if the node is a door.
            Defaults to None.
        area (float): The area occupied by the node in pixel units.
                      For point-like nodes (e.g., mesh centroids), this might be
                      an estimated representative area or a standard small value.
    """
    def __init__(self, node_id: int, node_type: str, pos: Tuple[int, int, int],
                 default_time: float, area: float = 1.0): # Default area to 1 pixel if not specified
        """
        Initializes a Node object.

        Args:
            node_id: The unique identifier for the node.
            node_type: The type of the node.
            pos: The (x, y, z) coordinates of the node.
            default_time: The default time cost for this node type.
            area: The area of the node in pixel units.
        """
        self.id: int = node_id
        self.node_type: str = node_type
        self.pos: Tuple[int, int, int] = pos
        self.time: float = default_time
        self.door_type: Optional[str] = None
        self.area: float = area

    def __repr__(self) -> str:
        """Returns a string representation of the Node."""
        return (f"Node(id={self.id}, type='{self.node_type}', pos={self.pos}, "
                f"time={self.time:.2f}, area={self.area:.2f}, door_type='{self.door_type}')")
    
    def __hash__(self) -> int:
        """Returns the hash of the node based on its ID."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Checks equality with another Node based on ID."""
        if isinstance(other, Node):
            return self.id == other.id
        return False