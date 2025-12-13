from typing import List
import copy


#
# Game is essentially made of two vital parts - its states and turns
# State is basically a "frame" of a game - its one point in the game - for example in chess, state is how the chessboard is at a moment
# Turn (move) is what creates a new state - we move from one state from another
# Game is played by going from initial state to another state by making a turn - and this continues until the end of the game
# Tree graph can be used to represent this - nodes are states and branches are turns (moves)
#



#
# Simple representation of game tick tack toe - me cross, opponent circle
# Overall playing of the game is represented by the Tree
# Each node represents current state with matrix (each value is number: 1 is cross, 0 is none, -1 is circle
#

class TreeNode:
    def __init__(self, is_current_player_me: bool, board: List[List[int]] = None):
        # 0: Empty, 1: Me (Cross), -1: Opponent (Circle)
        if board is None:
            self.board = [[0 for _ in range(3)] for _ in range(3)]
        else:
            self.board = board
        self.children: List[TreeNode] = []
        self.is_current_player_me = is_current_player_me

    def generate_children(self):
        """ Generates all possible next moves (states) from current state """
        player_val = 1 if self.is_current_player_me else -1

        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    # Create deep copy of board for the new state
                    new_board = copy.deepcopy(self.board)
                    new_board[r][c] = player_val

                    # Create child node (switch turn)
                    child = TreeNode(not self.is_current_player_me, new_board)
                    self.children.append(child)


def print_tree(node: TreeNode, depth: int = 0, max_depth: int = 2):
    """ Recursively prints the game tree up to max_depth """
    if depth > max_depth:
        return

    indent = "    " * depth
    # Symbol map: 1 -> X, -1 -> O, 0 -> .
    symbols = {1: "X", -1: "O", 0: "."}

    # Flatten board for single-line display in tree
    board_str = "|".join(["".join([symbols[x] for x in row]) for row in node.board])
    turn = "My Turn" if node.is_current_player_me else "Opponent"
    print(f"{indent}State: [{board_str}] ({turn})")

    for child in node.children:
        print_tree(child, depth + 1, max_depth)

# Usage
root = TreeNode(is_current_player_me=True) # My turn (X)

# Generate 1st layer (My moves)
root.generate_children()

# Generate 2nd layer (Opponent responses for each of my moves)
for child in root.children:
    child.generate_children()

print("Game Tree (Depth 2):")
print_tree(root, max_depth=2)

# ... And so on other moves

# Each move created new state - so for each of these states, this would repeat - other moves would be made to create new states, etc
# This all would repeat until each "branching" would finish the game

