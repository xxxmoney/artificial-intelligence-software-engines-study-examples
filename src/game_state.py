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
    @abstractmethod
    def get_status(self) -> GameStatus:
        """ Get game status """
        pass

    @abstractmethod
    def generate_possible_states(self) -> List['IGameState']:
        """ Get next possible game states """
        pass
