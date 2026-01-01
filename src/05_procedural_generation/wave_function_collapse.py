from collections import defaultdict
from enum import Enum
from typing import Optional


#
# WFC - Wave Function Collapse
# It's a way to procedurally generate maps for games based on rules
# How it works:
# - Imagine we have tiles grid - 10x10 - each tile can be of some type (mountain, grass, lake, etc) and each tile type has rules which another tile type can border with it
# - So each tile has superposition - it can be multiple positions at once - at start, there are no positions, so each tile can be anything
# - We start by defining one tile randomly or set it predefined
# - This now defines the defined tile's neighbours - each neighbour now has less superposition (ie entropy - the number of possible states)
# - We take the one with least entropy and do choose randomly from available
# - So we could have now 3 neighbours that have 4 types and 1 which has 2 - we choose the on with 2 and randomly choose its type
# - And this now cascades further - and this repeats until the whole grid is defined
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

    def reduce_possible_types(self, reduced_types: list[TileType]):
        if self._type is not None:
            raise Exception('Cannot reduce types for defined type')

        self._possible_types -= set(reduced_types)

    def is_compatible(self, other_type: TileType) -> bool:
        for type in self._possible_types:
            if TileType.are_compatible(type, other_type):
                return True

        return False

    def __str__(self):
        return f"[{self.type}: {self.possible_types}]"

class Grid:
    row_count: int
    column_count: int
    _tiles: list[list[Tile]] # First list - rows - - each list for each row is columns

    def __init__(self, row_count: int, column_count: int):
        self.row_count = row_count
        self.column_count = column_count

        # Set default grid by row count and column count
        self._tiles = [[Tile(None) for _ in range(row_count)] for _ in range(column_count)]

    def get_tile_type(self, row: int, column: int) -> TileType:
        return self._tiles[row][column].type

    def set_tile_type(self, row: int, column: int, type: TileType):
        self._tiles[row][column].type = type

    def has_tile(self, row: int, column: int) -> bool:
        return self.get_tile_type(row, column) is not None

    def has_any_tile(self) -> bool:
        for row in self._tiles:
            for tile in row:
                if tile is not None:
                    return True

        return False

    def reduce_possible_tile_types(self, row: int, column: int):
        tile = self._tiles[row][column]

        for row, column in self.get_neighbour_positions(row, column):
            neighbour = self._tiles[row][column]

            # We don't reduce states for already defined tiles
            if neighbour.type is not None:
                continue

            changed = False
            reduced_types = []
            for neighbour_type in neighbour.possible_types:
                # Possible type of the neighbour tile is not compatible - add to list to be removed
                if not tile.is_compatible(neighbour_type):
                    reduced_types.append(neighbour_type)
                    changed = True

            # Remove not compatible with tile
            neighbour.reduce_possible_types(reduced_types)

            # Explore neighbours if changed (removed some possible type)
            if changed:
                self.reduce_possible_tile_types(row, column)


    def get_neighbour_positions(self, row: int, column: int) ->  list[tuple[int, int]]:
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


if __name__ == "__main__":
    grid = Grid(5, 5)

    grid.set_tile_type(0, 0, TileType.MOUNTAIN)
    grid.reduce_possible_tile_types(0, 0)

    #wfc = WaveFunctionCollapse(grid)

    print(grid)
