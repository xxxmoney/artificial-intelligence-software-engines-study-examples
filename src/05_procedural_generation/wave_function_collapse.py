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



