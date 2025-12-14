from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar
from enum import Enum

T = TypeVar('T')


class GameStatus(Enum):
    RUNNING = 1
    WIN = 2
    DEFEAT = 3
    DRAW = 4

class IHasEvaluableState(ABC):
    @abstractmethod
    def evaluate(self) -> float:
        """ Returns an evaluation of the game state """
        pass

class IGameState(ABC):
    @property
    @abstractmethod
    def get_status(self) -> GameStatus:
        """ Get game status """
        pass

    @property
    @abstractmethod
    def possible_next_states(self) -> List['IGameState']:
        """ Get next possible game states """
        pass

    def is_me(self) -> bool:
        """ Whether the current player is me """
        pass

    def is_opponent(self) -> bool:
        """ Whether the current player is opponent """
        pass