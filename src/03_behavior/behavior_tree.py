#
# Behavior Tree is a principle of managing NPC behavior - an improved principle from RBS or FSM
# It works on basis of tree of nodes and leafs and state, that is returned from nodes
# Each node can selector or sequence
# - Selector - list of nodes - successful if any of the nodes returns SUCCESSFUL or RUNNING state (so it basically takes the first successful one)
# - Sequence - list of nodes - all of them have to be successful (SUCCESSFUL/RUNNING)
# Each leaf can be condition or action
# - Condition - simply returns TRUE or FALSE (like checking if has enough health, etc)
# - Action - takes action - attacks enemy, heals, etc
#

from enum import Enum
from game_simple import GameSimple

class Status(Enum):
    SUCCESS = 0,
    FAILURE = 1,
    RUNNING = 2

