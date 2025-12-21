#
# Rule Based System is a simple principle of managing NPC behavior with loads of if statements
# Basically, what we need to do is to check for circumstances for what the NPC can do
# Great disadvantage is the repetition of same conditions across multiple if statements, along with not memory of previous state
#

from game_simple import GameSimple, Weapon

class RBS:
    game: GameSimple

    def __init__(self, game: GameSimple):
        self.game = game

    # Demon example, non working function - showcases rule based system for "NPC behavior"
    # Nested if statements are for the demonstration
    def check(self):
        # Check to engage new enemy
        if not self.game.close_enemy:
            if self.game.has_enough_health():
                self.game.engage_enemy()

        # Check to attack current enemy
        if self.game.close_enemy:
            if self.game.has_enough_health():
                if self.game.has_enough_ammunition():
                    self.game.attack()
                else:
                    if not self.game.is_blocking:
                        self.game.block()
                    else:
                        self.game.reload()
                        self.game.cancel_block()

        # Check to cancel engaging current enemy:
        if not self.game.has_enough_health():
            if self.game.close_enemy:
                self.game.cancel_engage_enemy()

            self.game.heal()

        # ... Other conditions could be defined here

if __name__ == "__main__":
    game = GameSimple()
    rbs = RBS(game)

    play_game_for_ticks = 100

    # Play a game for a while
    for _ in range(play_game_for_ticks):
        game.tick()
        rbs.check()


