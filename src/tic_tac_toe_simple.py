from dataclasses import Field
from enum import Enum
from typing import Optional
import copy

from src.game_state import IGameState, IHasEvaluableState, GameStatus

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

    def inverse(self) -> Field:
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


class TicTacToeState(IGameState, IHasEvaluableState):
    board: list[Optional[Field]] # Board of fields
    current: Field # Current player

    def __init__(self, current: Field, board: Optional[list[Field]] = None):
        self.current = current

        if board is None:
            self.board = [None for _ in range(DEFAULT_SIZE)]
        else:
            self.board = board

    def generate_possible_states(self) -> list['TicTacToeState']:
        states = []

        opposite = self.current.inverse()

        for i, field in enumerate(self.board):
            if field is None:
                # Set opponent move to this field
                board_copy = copy.deepcopy(self.board)
                board_copy[i] = opposite

                # Append state with the board copy as opposite player
                states.append(TicTacToeState(opposite, board_copy))

        return states

    def get_status(self) -> GameStatus:
        win_status = self._is_win()

        # We won
        if win_status:
            return GameStatus.WIN
        # Opponent won
        elif not win_status and win_status is not None:
            return GameStatus.DEFEAT
        # None won and board is full - draw
        elif win_status is None and self._is_board_full():
            return GameStatus.DRAW
        # None won and board is not full - the game is still running
        else:
            return GameStatus.RUNNING

    def evaluate(self) -> float:
        return 0 # TODO

    def __str__(self) -> str:
        return f"SCORE: {self.evaluate()} | STATUS: {self.get_status()} | BOARD: {str(self.board)}"

    def _is_board_full(self) -> bool:
        # Board has no None fields
        return len([field for field in self.board if field is None]) == 0

    def _is_win(self) -> Optional[bool]:
        last_field = None
        sequence_count = 1 # Default sequence number is 1

        # Check whether there is sequence for winning
        for field in self.board:
            # If field is not defined, skip it
            if field is None:
                last_field = field
                continue

            # Sequence increase by one
            if field == last_field:
                sequence_count += 1
            # Sequence interrupted, set current field and reset sequence count
            else:
                sequence_count = 1
                last_field = field

            if sequence_count >= SEQUENCE_COUNT_TO_WIN:
                # Winner is determined whether me is the sequence field or not
                return field == Field.me()

        # Draw
        return None


if __name__ == "__main__":
    #
    # Example Usage
    #

    # Running
    print("Running:")
    running_state = TicTacToeState(Field.me(), None)
    print(running_state)

    # Cheated win
    print("Cheated Win:")
    cheated_winning_state = TicTacToeState(Field.me(), None)
    for i in range(len(cheated_winning_state.board)):
        cheated_winning_state.board[i] = Field.me()
    print(cheated_winning_state)

    # Less Cheated win
    print("Less Cheated Win:")
    less_cheated_winning_state = TicTacToeState(Field.me(), None)
    for i in range(SEQUENCE_COUNT_TO_WIN):
        less_cheated_winning_state.board[i] = Field.me()
    print(less_cheated_winning_state)

    # Cheated defeat
    print("Cheated Defeat:")
    cheated_defeat_state = TicTacToeState(Field.me(), None)
    for i in range(len(cheated_defeat_state.board)):
        cheated_defeat_state.board[i] = Field.opponent()
    print(cheated_defeat_state)

    # Less Cheated defeat
    print("Less Cheated Defeat:")
    less_cheated_defeat_state = TicTacToeState(Field.me(), None)
    for i in range(SEQUENCE_COUNT_TO_WIN):
        less_cheated_defeat_state.board[i] = Field.opponent()
    print(less_cheated_defeat_state)

    # Draw
    print("Draw:")
    draw_state = TicTacToeState(Field.me(), None)
    last_field = Field.me()
    for i in range(len(draw_state.board)):
        draw_state.board[i] = last_field
        last_field = last_field.inverse()
    print(draw_state)

