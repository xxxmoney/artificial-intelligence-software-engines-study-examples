from idlelib.tree import TreeNode
from typing import Generic, TypeVar, List, Optional, Any

T = TypeVar('T')


class TreeNode(Generic[T]):
    """
    A generic Tree Node for game states
    T: The type of the game state (e.g., board matrix, config dict)
    """

    def __init__(self, state: T, parent: Optional['TreeNode'] = None):
        self.state = state
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.best_child: Optional[TreeNode] = None

        # Metadata for algorithms (Minimax/MCTS)
        self.score: float = 0.0
        self.visits: int = 0

    @property
    def score_visits_ratio(self) -> float:
        return 0 if not self.visits else self.score / self.visits

    def add_child(self, state: T) -> 'TreeNode':
        """ Creates and appends a child node with the given state """
        child = TreeNode(state, parent=self)
        self.children.append(child)
        return child

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_root(self) -> 'TreeNode':
        node = self
        while node.parent:
            node = node.parent
        return node

    def __repr__(self) -> str:
        return f"TreeNode(state={self.state}, children={len(self.children)})"

    def print_tree(self, level: int = 0):
        """ Visualizes the tree structure """
        indent = "    " * level
        print(f"{indent} -> {self.state} || (Score: {self.score}) (Visits: {self.visits})")
        for child in self.children:
            child.print_tree(level + 1)


if __name__ == "__main__":
    #
    # Example Usage
    #

    root = TreeNode("Start")

    # Simulate moves
    move_a = root.add_child("Move A")
    move_b = root.add_child("Move B")

    move_a.add_child("Move A-1")
    move_a.add_child("Move A-2")

    root.print_tree()
