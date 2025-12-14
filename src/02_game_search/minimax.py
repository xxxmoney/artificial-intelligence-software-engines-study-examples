from src.game_state import IGameState, GameStatus, IHasEvaluableState
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


def minimax(node: TreeNode[IGameState], depth: int = 5) -> float:
    node.visits += 1

    # Either the game has finished or the max depth was reached
    if node.state.get_status != GameStatus.RUNNING or depth <= 0:
        if not isinstance(node.state, IHasEvaluableState):
            raise TypeError("State must be of type IHasEvaluableState")
        node.score = node.state.evaluate()

        return node.score

    # We are playing this state, so we want to find max score
    if node.state.is_me():
        max_score = float('-inf')

        for state in node.state.possible_next_states:
            child_node = node.add_child(state)
            child_node.score = minimax(child_node, depth - 1)

            if child_node.score > max_score:
                max_score = child_node.score
                node.best_child = child_node
                node.score = child_node.score

    # Opponent is playing this state, and they want to find theirs max - so ours min
    else:
        min_score = float('inf')

        for state in node.state.possible_next_states:
            child_node = node.add_child(state)
            child_node.score = minimax(child_node, depth - 1)

            if child_node.score < min_score:
                min_score = child_node.score
                node.best_child = child_node
                node.score = child_node.score

    return node.score


if __name__ == "__main__":
    #
    # Example Usage with simpler simple tick tack toe
    #

    # Start with me, empty board
    root = TreeNode(TicTacToeState(Field.me(), None))

    # Run minimax
    minimax(root)
    best_state = root.best_child

    print(f"Best state:\n{best_state} \n\n")
    print("Tree:")
    root.print_tree()



