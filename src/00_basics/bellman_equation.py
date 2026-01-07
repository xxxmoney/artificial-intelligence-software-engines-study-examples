import copy
import enum
from typing import Optional

GAMMA = 0.9
DEFAULT_VALUE = 0.0

class StateType(enum.Enum):
    START = 1
    HALLWAY = 2
    TREASURE = 3
    LAVA = 4

class ActionType(enum.Enum):
    GO_LEFT = 1
    GO_RIGHT = 2

class State:
    _type: StateType
    _left: Optional['State']
    _right: Optional['State']
    _value: float

    def __init__(self, type: StateType, left: Optional['State'] = None, right: Optional['State'] = None):
        self._type = type
        self._left = left
        self._right = right
        self._value = DEFAULT_VALUE

    @property
    def type(self) -> StateType:
        return self._type

    @property
    def left(self) -> Optional['State']:
        return self._left

    @property
    def right(self) -> Optional['State']:
        return self._right

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    def get_next(self, action: ActionType) -> Optional['State']:
        if action == ActionType.GO_LEFT:
            return self._left
        if action == ActionType.GO_RIGHT:
            return self._right

        raise ValueError('Invalid action')

    def is_last(self):
        return self._left is None and self._right is None

    def __str__(self):
        return f"{type} -> ({self.left}, {self.right}): {self.value}"

def print_state(state: State):
    states_queue = [state]
    while states_queue:
        current_state = states_queue.pop(0)
        if current_state.left is not None:
            states_queue.append(current_state.left)
        if current_state.right is not None:
            states_queue.append(current_state.right)

        print(current_state)

ACTIONS = [ActionType.GO_LEFT, ActionType.GO_RIGHT]
REWARDS = {
    StateType.START: 0,
    StateType.HALLWAY: 0,
    StateType.TREASURE: 100,
    StateType.LAVA: -100
}


def bellman_update(node: State) -> State:
    max_state = copy.deepcopy(node)

    state_queue = [node]
    while state_queue:
        state = state_queue.pop(0)

        if state.is_last():
            state.value = REWARDS[state.type]
            continue

        for action in ACTIONS:
            next_state = state.get_next(action)
            state_queue.append(next_state)

            reward = 0 # TODO: why 0, reward function?
            value_of_action = reward + GAMMA * bellman_update(next_state).value # Use the bellman equation here
            if value_of_action > max_state.value:
                max_state = copy.deepcopy(next_state)

    node.value = max_state.value

    return max_state

if __name__ == "__main__":
    root = State(
        StateType.START,
        State(
            StateType.HALLWAY,
            State(StateType.LAVA, None, None),
            State(StateType.LAVA, None, None)
        ),
        State(
            StateType.HALLWAY,
            State(StateType.TREASURE, None, None),
            State(StateType.LAVA, None, None)
        )
    )

    for iteration in range(10):
        print(f"[ITERATION #{iteration}]")

        best_next_state = bellman_update(root)
        print_state(root)




