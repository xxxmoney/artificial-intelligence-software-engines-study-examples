from dataclasses import Field
from enum import Enum
from typing import Optional
import copy
from src.tree import TreeNode


#
# Tic Tac Toe - simplified version
# This is a simplified version - only with one row, player only needs to have N besides each other to win
# Me is CROSS, opponent is CIRCLE
#


SEQUENCE_COUNT_TO_WIN = 2
DEFAULT_SIZE = 5

class Field(Enum):
    CROSS = 0
    CIRCLE = 1

    def __str__(self):
        if self == Field.CROSS:
            return "X"
        elif self == Field.CIRCLE:
            return "O"

        return ""

    def opposite(self) -> Field:
        if self == Field.CIRCLE:
            return Field.CROSS
        elif self == Field.CROSS:
            return Field.CIRCLE

        raise ValueError("Invalid field")

    def is_me(self):
        return self == Field.CROSS

    def is_opponent(self):
        return self == Field.CIRCLE

    @staticmethod
    def me():
        return Field.CROSS

    @staticmethod
    def opponent():
        return Field.CIRCLE


class TicTacToeState:
    board: list[Optional[Field]] # Board of fields
    current: Field # Current player

    def __init__(self, current: Field, board: Optional[list[Field]] = None):
        self.current = current

        if board is None:
            self.board = [None for _ in range(DEFAULT_SIZE)]
        else:
            self.board = board

    def __str__(self) -> str:
        return str(self.board)

    def generate_possible_states(self) -> list['TicTacToeState']:
        states = []

        opposite = self.current.opposite()

        for i, field in enumerate(self.board):
            if field is None:
                # Set opponent move to this field
                board_copy = copy.deepcopy(self.board)
                board_copy[i] = opposite

                # Append state with the board copy as opposite player
                states.append(TicTacToeState(opposite, board_copy))

        return states


if __name__ == "__main__":
    #
    # Example Usage
    #

    # Start with me, empty board
    root = TreeNode(TicTacToeState(Field.me(), None))

    # Generate possible states - with opponent
    for state in root.state.generate_possible_states():
        child = root.add_child(state)

        for child_state in state.generate_possible_states():
            child.add_child(child_state)

    root.print_tree()
