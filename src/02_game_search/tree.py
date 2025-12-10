from idlelib.tree import TreeNode
from typing import List


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
    board: List[List[int]]
    moves: List[TreeNode]
    isCurrentPlayerMe: bool

    def __init__(self, isCurrentPlayerMe: bool, cols: int = 3, rows: int = 3, default_value: int = 0):
        self.board = [[default_value for _ in range(cols)] for _ in range(rows)]
        self.moves = []
        self.isCurrentPlayerMe = isCurrentPlayerMe

# Initial game state - and I am starting
tree: TreeNode = TreeNode(True)

# Some of the moves I can do in this turn - turn 1 - from initial state to another state
move_one = TreeNode(False) # This one will be opponents turn
move_one.board[0][0] = 1 # I made a move of cross in the left upper corner in this specific move
tree.moves.append(move_one)

move_two = TreeNode(False) # # This one will also be opponents turn
move_two.board[0][2] = 1 # I made a move of cross in the right upper corner in this specific move
tree.moves.append(move_two)

# ... And so on other moves

# Each move created new state - so for each of these states, this would repeat - other moves would be made to create new states, etc
# This all would repeat until each "branching" would finish the game

