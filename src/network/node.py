from typing import Tuple
class Node:
    _id_counter = 1
    def __init__(self, node_type:str, node_pos:Tuple[int, int]):
        self.type = node_type
        self.pos = node_pos
        self.id = Node._id_counter
        Node._id_counter += 1

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type}, pos={self.pos})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False