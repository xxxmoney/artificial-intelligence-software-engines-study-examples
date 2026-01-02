import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional


#
# WFC - Wave Function Collapse
# It's a way to procedurally generate maps for games based on rules
# How it works:
# - Imagine we have tiles grid - 10x10 - each tile can be of some type (mountain, grass, lake, etc) and each tile type has rules which another tile type can border with it
# - So each tile has superposition - it can be multiple positions at once - at start, there are no positions, so each tile can be anything
# - This is represented by possible tile types - each tile can "evolve" into any of these types - thus possible
# - We start by defining one tile randomly or set it predefined
# - Reduce (of possible tile types)
#   - Because we now set one tile, we may have broken compatibility of neighbouring rules (because previously anything could be anything, any tile could neighbour with any tile)
#   - This now changed - so we need to start removing (reducing) the neighbours' possible tile types
#   - For example if we define a specific type for tile, we need to check all neighbours and remove not compatible possible types with this tiles
#   - The same goes for neighbours - their possible types changed, so they also need to check their own neighbours, etc
#   - One recursion end is in the moment the neighbour won't change - its already compatible
# - Collapse (of tile type)
#   - This now defines the defined tile's neighbours - each neighbour now has less superposition (ie entropy - the number of possible states)
#   - We take the one with least entropy and do choose randomly from available
#   - So we could have now 3 neighbours that have 4 types and 1 which has 2 - we choose the on with 2 and randomly choose its type
#   - And this now cascades further - and this repeats until the whole grid is defined
#

class TileType(Enum):
    MOUNTAIN = 1
    GRASS = 2
    LAKE = 3
    CASTLE = 4

    @staticmethod
    def get_all() -> set['TileType']:
        return set(list(TileType))

    @staticmethod
    def get_connections() -> list[tuple['TileType', 'TileType']]:
        return [
            (TileType.MOUNTAIN, TileType.GRASS),
            (TileType.MOUNTAIN, TileType.CASTLE),
            (TileType.GRASS, TileType.LAKE),
            (TileType.CASTLE, TileType.LAKE),
            (TileType.GRASS, TileType.GRASS)
        ]

    @staticmethod
    def get_rules() -> dict['TileType', set['TileType']]:
        rules = defaultdict(set)
        for a, b in TileType.get_connections():
            rules[a].add(b)
            rules[b].add(a) # A <-> B symmetry
        return dict(rules)

    @staticmethod
    def are_compatible(first_type: 'TileType', second_type: 'TileType') -> bool:
        return second_type in TileType.get_rules()[first_type]

    def __str__(self):
        match self:
            case TileType.MOUNTAIN:
                return "M"
            case TileType.GRASS:
                return "G"
            case TileType.LAKE:
                return "L"
            case TileType.CASTLE:
                return "C"
            case _:
                return "_"

class Tile:
    _type: Optional[TileType] # The type of this tile
    _possible_types: set[TileType] # Represents which types this can turn into - if not None, has only one - the type

    def __init__(self, type: Optional[TileType]):
        self._type = type
        self._possible_types = {type} if type is not None else TileType.get_all()

    def reduce_noncompatible_possible_types(self, other_tile: 'Tile') -> bool:
        """
        Removes not compatible possible types with other tile

        :param other_tile: Tile to check compatibility with
        :return: Whether this tile's possible states have changed or not
        """
        changed = False
        reduced_types = []

        for type in self.possible_types:
            # Possible type of the other tile is not compatible - add to list to be removed
            if not other_tile.is_compatible(type):
                reduced_types.append(type)
                changed = True

        # Remove not compatible with tile
        self._reduce_possible_types(reduced_types)

        return changed

    def is_compatible(self, other_type: TileType) -> bool:
        """
        Checks whether other type is compatible with any possible types of this tile

        :param other_type: Type to be checked against
        :return: Whether is compatible or not
        """
        for type in self.possible_types:
            if TileType.are_compatible(type, other_type):
                return True

        return False

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type: TileType):
        if self._type is not None:
            raise Exception('Cannot set type twice')

        self._type = type
        self._possible_types = {type}

    @property
    def possible_types(self):
        return self._possible_types

    @property
    def possible_types_count(self):
        return len(self.possible_types)

    def _reduce_possible_types(self, reduced_types: list[TileType]):
        if self.type is not None:
            raise Exception('Cannot reduce types for defined type')

        self._possible_types -= set(reduced_types)

    def __str__(self):
        return f"[{self.type}: {self.possible_types}]"

@dataclass
class TileWithPosition:
    row: int
    column: int
    tile: Tile

class Grid:
    row_count: int
    column_count: int
    _tiles: list[list[Tile]] # First list - rows - - each list for each row is columns

    def __init__(self, row_count: int, column_count: int):
        self.row_count = row_count
        self.column_count = column_count

        # Set default grid by row count and column count
        self._tiles = [[Tile(None) for _ in range(row_count)] for _ in range(column_count)]

    def reduce_neighbours_possible_tile_types(self, row: int, column: int) -> None:
        """
        Starts reductions (from specific position) of possible tile types - ie for the whole board removes non-compatible possible types for neighbours

        :param row: Start row
        :param column: Start column
        :return: None
        """

        tile = self.get_tile(row, column)

        for row, column in self._get_neighbours_positions(row, column):
            neighbour = self.get_tile(row, column)

            # We don't reduce states for already defined tiles
            if neighbour.type is not None:
                continue

            # Explore neighbours if changed (removed some possible type)
            if neighbour.reduce_noncompatible_possible_types(tile):
                self.reduce_neighbours_possible_tile_types(row, column)

    def get_neighbours(self, row: int, column: int) -> list[TileWithPosition]:
        return [TileWithPosition(row = neighbour_row, column = neighbour_column, tile = self.get_tile(neighbour_row, neighbour_column)) for neighbour_row, neighbour_column in self._get_neighbours_positions(row, column)]

    def get_tile_type(self, row: int, column: int) -> TileType:
        return self.get_tile(row, column).type

    def set_tile_type(self, row: int, column: int, type: TileType):
        self._set_tile_type(self.get_tile(row, column), type)

    def set_random_tile_type(self, row: int, column: int):
        tile = self.get_tile(row, column)
        self._set_tile_type(tile, random.choice(list(tile.possible_types)))

    def has_tile(self, row: int, column: int) -> bool:
        return self.get_tile_type(row, column) is not None

    def get_tile(self, row: int, column: int) -> Tile:
        return self._tiles[row][column]

    def has_any_tile(self) -> bool:
        for row in self._tiles:
            for tile in row:
                if tile is not None:
                    return True

        return False

    def _set_tile_type(self, tile: Tile, type: TileType):
        tile.type = type
        # Tile type change - we need to reduce the possible types of neighbours
        self.reduce_neighbours_possible_tile_types(row, column)

    def _get_neighbours_positions(self, row: int, column: int) ->  list[tuple[int, int]]:
        """
        Get positions of neighbours for specific position

        :param row: Row position
        :param column: Column position
        :return: List of positions (row, column)
        """

        positions: list[tuple[int, int]] = []

        # Left
        if row > 0:
            positions.append((row - 1, column))
        # Right
        if row < self.row_count - 1:
            positions.append((row + 1, column))
        # Up
        if column > 0:
            positions.append((row, column - 1))
        # Down
        if column < self.column_count - 1:
            positions.append((row, column + 1))

        return positions

    def __str__(self):
        rows = []

        for row in self._tiles:
            rows.append("\t".join([str(tile) for tile in row]))

        return "\n".join(rows)

class WaveFunctionCollapse:
    grid: Grid

    def __init__(self, grid: Grid):
        self.grid = grid

    # TODO: test collapsing
    def collapse(self, row: int, column: int):
        # Neighbour with least possible types count (or first one of the least)
        candidate: Optional[TileWithPosition] = None

        # Need to get the on with least possible types - thats the one we will be choosing specific types first
        for neighbour in self.grid.get_neighbours(row, column):
            # Neighbour has to have none type set yet and less possible types then the current candidate - or current candidate is none
            if neighbour.tile.type is None and (candidate is None or neighbour.tile.possible_types_count < candidate.tile.possible_types_count):
                candidate = neighbour

        if candidate is not None:
            # We have one - choose random specific type for it
            self.grid.set_random_tile_type(candidate.row, candidate.column)

            # call collapse on the candidate now
            self.collapse(candidate.row, candidate.column)


if __name__ == "__main__":
    grid = Grid(5, 5)

    row, column = (0, 0)
    grid.set_tile_type(row, column, TileType.MOUNTAIN)

    wfc = WaveFunctionCollapse(grid)
    wfc.collapse(row, column)

    print(grid)
