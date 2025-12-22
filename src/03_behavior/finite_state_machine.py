#
# Finite State machine is a simple principle of managing NPC behavior with loads of if statements and a state memory
# Finite State Machine is an improvement of Rule Based System - while it's still a mess of ifs, it now has memory - state
# State acts as a memory in sense that we know in which state we are - whether attacking, retreating, healing, etc
# Said state reduces the amount of "nested ifs" and gives us an ability to write cleaner, more readable code
#

from enum import Enum
from game_simple import GameSimple

class GameSimpleState(Enum):
    DEFAULT = 0,
    ENGAGING = 1,
    HEALING = 2,
    RELOADING = 3

class FSM:
    game: GameSimple
    state: GameSimpleState

    def __init__(self, game):
        self.game = game
        self.state = GameSimpleState.DEFAULT

    # Demon example, non working function - showcases rule based system for "NPC behavior"
    # Nested if statements are for the demonstration
    def check(self):
        # Check to engage new enemy
        if self.state == GameSimpleState.DEFAULT:
            if self.game.has_enough_health():
                self.game.engage_enemy()
                self.state = GameSimpleState.ENGAGING

        if self.state == GameSimpleState.RELOADING:
            if self.game.has_enough_ammunition():
                self.game.cancel_block()
                self.state = GameSimpleState.ENGAGING
            else:
                self.game.reload()

        if self.state == GameSimpleState.ENGAGING:
            if self.game.has_enough_health():
                if self.game.has_enough_ammunition():
                    if self.game.close_enemy:
                        self.game.attack()
                    else:
                        self.state = GameSimpleState.DEFAULT
                else:
                    self.game.block()
                    self.state = GameSimpleState.RELOADING
            else:
                self.game.cancel_engage_enemy()
                self.state = GameSimpleState.HEALING

        if self.state == GameSimpleState.HEALING:
            if self.game.has_enough_health():
                self.state = GameSimpleState.DEFAULT
            else:
                self.game.heal()

        # ... Other conditions could be defined here



if __name__ == "__main__":
    game = GameSimple()
    fsm = FSM(game)

    play_game_for_ticks = 100

    # Play a game for a while
    for _ in range(play_game_for_ticks):
        game.tick()
        fsm.check()
