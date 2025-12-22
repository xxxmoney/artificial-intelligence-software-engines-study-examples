#
# Behavior Tree is a principle of managing NPC behavior - an improved principle from RBS or FSM
# It works on basis of tree of nodes and leafs and state, that is returned from nodes
# Each node can selector or sequence
# - Selector - list of nodes - successful if any of the nodes returns SUCCESSFUL or RUNNING state (so it basically takes the first successful one)
# - Sequence - list of nodes - all of them have to be successful (SUCCESSFUL/RUNNING)
# Each leaf can be condition or action
# - Condition - simply returns SUCCESS or FAILURE (like checking if has enough health, etc)
# - Action - takes action - attacks enemy, heals, etc
#

from enum import Enum
from typing import Callable

from game_simple import GameSimple

class Status(Enum):
    SUCCESS = 0,
    FAILURE = 1,
    RUNNING = 2

class Node:
    def tick(self, game: GameSimple) -> Status:
        raise NotImplementedError("Method tick not implemented")

class Selector(Node):
    children: list[Node]

    def __init__(self, children: list[Node]):
        self.children = []
#
    def tick(self, game: GameSimple):
        # Checks for the first child that is not FAILURE
        for child in self.children:
            status = child.tick(game)
            if status != Status.FAILURE:
                return status

        return Status.FAILURE

class Sequence(Node):
    children: list[Node]

    def __init__(self, children: list[Node]):
        self.children = children

    def tick(self, game: GameSimple):
        # Stops checking if any child is not SUCCESS
        for child in self.children:
            status = child.tick(game)
            if status != Status.SUCCESS:
                return status

        return Status.SUCCESS

class Condition(Node):
    predicate_fn: Callable[[GameSimple], bool]

    def __init__(self, predicate_fn):
        self.predicate_fn = predicate_fn

    def tick(self, game: GameSimple):
        return Status.SUCCESS if self.predicate_fn(game) else Status.FAILURE

class Action(Node):
    action_fn: Callable[[GameSimple], None]

    def __init__(self, action_fn):
        self.action_fn = action_fn

    def tick(self, game: GameSimple):
        try:
            self.action_fn(game)
            return Status.SUCCESS
        except:
            return Status.FAILURE

if __name__ == "__main__":
    game = GameSimple()

    # TODO: use behavior tree for the game