from dataclasses import dataclass
from enum import Enum
from typing import Optional

MAX_HEALTH = 100
MIN_SAFE_HEALTH = 50
HEAL_STEP = 5
ATTACK_POWER = 10
MAX_AMMUNITION = 5
AMMUNITION_STEP = 1

class Weapon(Enum):
    PISTOL = 0

class Enemy:
    health: int
    attack_power: int

    def __init__(self):
        self.health = MAX_HEALTH
        self.attack_power = ATTACK_POWER

class GameSimple:
    health: int
    attack_power: int
    is_blocking: bool
    ammunition: int
    distant_enemy: Optional[Enemy]
    close_enemy: Optional[Enemy]
    weapon: Weapon

    def __init__(self):
        self.health = MAX_HEALTH
        self.attack_power = ATTACK_POWER
        self.is_blocking = False
        self.ammunition = MAX_AMMUNITION
        self.distant_enemy = Enemy()
        self.close_enemy = None
        self.weapon = Weapon.PISTOL

    def engage_enemy(self):
        print("[Engaging new enemy]")

        self.close_enemy = self.distant_enemy
        self.distant_enemy = None

    def cancel_engage_enemy(self):
        print("[Retreating from current enemy]")

        self.distant_enemy = self.close_enemy
        self.close_enemy = None

    def attack(self):
        print(f"[Attacking close enemy for {self.attack_power}]")

        self.close_enemy.health -= self.attack_power
        self.ammunition -= AMMUNITION_STEP

        print(f"[Close enemy health {self.close_enemy.health}]")

        print(f"[Current ammunition {self.ammunition}]")

    def block(self):
        print("[Activated blocking]")

        self.is_blocking = True

    def cancel_block(self):
        print("[Deactivated blocking]")

        self.is_blocking = False

    def heal(self):
        print(f"[Healing for {HEAL_STEP}]")

        self.health += HEAL_STEP

        print(f"[Current health {self.health}]")

    def reload(self):
        print("[Reloading]")

        self.ammunition = MAX_AMMUNITION

    def has_enough_health(self):
        print("Checking if has enough health")

        return self.health >= MIN_SAFE_HEALTH

    def has_full_health(self):
        print("Checking if has full health")

        return self.health >= MAX_HEALTH

    def has_enough_ammunition(self):
        print("Checking if has enough ammunition")

        return self.ammunition > 0

    def tick(self):
        print("\n\n[[Ticking game]]")

        # Handle dead enemy
        if self.close_enemy and self.close_enemy.health <= 0:
            print("[Close enemy is dead, new enemy is approaching in distance]")

            self.close_enemy = None
            self.distant_enemy = Enemy()

        # Enemy attack
        if self.close_enemy:
            # Cannot attack if blocking
            if self.is_blocking:
                print("[Blocking is activated, enemy cannot attack]")
            else:
                print(f"[Enemy attacking for {ATTACK_POWER}]")

                self.health -= ATTACK_POWER

                print(f"[Current health {self.health}]")