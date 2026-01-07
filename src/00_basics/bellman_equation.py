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
        return None

    def is_last(self):
        return self._left is None and self._right is None

    def __str__(self):
        left_value = f"{self.left.value:.1f}" if self.left else "None"
        right_value = f"{self.right.value:.1f}" if self.right else "None"
        return f"State({self.type.name}) Value: {self.value:.2f} | Left->{left_value} | Right->{right_value}"


ACTIONS = [ActionType.GO_LEFT, ActionType.GO_RIGHT]

# Rewards definitions
# Reward = Value gained instantly upon entering/terminating in a state
REWARDS = {
    StateType.START: 0,
    StateType.HALLWAY: 0,
    StateType.TREASURE: 100,
    StateType.LAVA: -100
}


def get_all_states_bfs(root: State) -> list[State]:
    """ Get all states contained from root """

    states = []
    queue = [root]
    visited = set()

    while queue:
        s = queue.pop(0)
        if s in visited: continue
        visited.add(s)
        states.append(s)

        if s.left: queue.append(s.left)
        if s.right: queue.append(s.right)
    return states


def bellman_update_all(root_node: State):
    """
    Performs ONE iteration of Bellman update across all states
    Updates state values based on their neighbors' values
    """

    # 1. Get all states to iterate over them
    all_states = get_all_states_bfs(root_node)

    # 2. Iterate and update each state
    for state in all_states:

        # TERMINAL STATES (Leaves)
        # Its value is based only on reward by type
        if state.is_last():
            state.value = REWARDS[state.type]
            continue

        # NON-TERMINAL STATES
        # Find best action using Bellman Equation: max(R + gamma * V_next) - ie: max(transition_reward + gamma - next_value)
        best_value = -float('inf')

        for action in ACTIONS:
            next_state = state.get_next(action)

            if next_state is None:
                continue  # Invalid move

            # --- THE BELLMAN EQUATION ---
            # R(s,a): Reward of action - cost of move (0 here)
            # V(s'):  Value of neighbor (state.value from memory)

            immediate_reward = 0
            # We only use next_state.value because this is deterministic - with true bellman equation, there could be something like: future_value = GAMMA * (0.8 * val_success + 0.2 * val_fail)
            future_value = GAMMA * next_state.value
            total_value = immediate_reward + future_value

            if total_value > best_value:
                best_value = total_value

        # Update current state with the calculated best value
        state.value = best_value


if __name__ == "__main__":
    # Tree Structure Setup
    #       START
    #      /     \
    #  HALLWAY  HALLWAY
    #  /    \    /    \
    # LAVA LAVA TREASURE LAVA

    root = State(
        StateType.START,
        left=State(  # Left Path -> Bad ending
            StateType.HALLWAY,
            State(StateType.LAVA),
            State(StateType.LAVA)
        ),
        right=State(  # Right Path -> Good ending
            StateType.HALLWAY,
            State(StateType.TREASURE),
            State(StateType.LAVA)
        )
    )

    print("--- START LEARNING ---")
    # Iterative learning loop (Value Iteration)
    # Alternative to recursion from true bellman equation
    for iteration in range(1, 6):
        bellman_update_all(root)

        print(f"\n[ITERATION #{iteration}]")
        print(f"ROOT (Start): {root.value}")
        print(f"LEFT (Hallway): {root.left.value}")
        print(f"RIGHT (Hallway): {root.right.value}")
        print(f"  -> Right Child (Treasure): {root.right.left.value}")