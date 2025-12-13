from src.tic_tac_toe_simple import TicTacToeState, Field
from src.tree import TreeNode

#
# MINIMAX is an algorithm for determining best possible move when we can evaluate each state with defined number
# Principle of MINIMAX is defined by using MIN and MAX functions
# MAX function is used when we play as the player - we want to maximize our score
# MIN function is used when opponent plays - opponent wants to minize out score
# The MINIMAX function is called recursively upon each state - its called until the limit is reached
# - The limit is either game end (win max value, loose min value) or max depth (evaluation of state)
# - Upon reaching the limit of call tree of MINIMAX, the last call returns the value which is then propagated through recursion upwards)
#



if __name__ == "__main__":
    #
    # Example Usage
    #

    # TODO: use MINIMAX

    # Start with me, empty board
    root = TreeNode(TicTacToeState(Field.me(), None))

    # Generate possible states - with opponent
    for state in root.state.generate_possible_states():
        child = root.add_child(state)

        for child_state in state.generate_possible_states():
            child.add_child(child_state)

    root.print_tree()

