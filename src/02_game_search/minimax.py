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


def minimax(node: TreeNode[IGameState], depth: int = 5) -> IGameState:
    max_score = float('-inf')
    best_state = None

    for state in node.state.generate_possible_states():
        state_max_score = find_score(node, depth)
        if state_max_score > max_score:
            max_score = state_max_score
            best_state = state

    return best_state

def find_score(node: TreeNode[IGameState], depth) -> float:
    # Either the game has finished or the max depth was reached
    if node.state.get_status != GameStatus.RUNNING or depth <= 0:
        if not isinstance(node.state, IHasEvaluableState):
            raise TypeError("State must be of type IHasEvaluableState")
        score = node.state.evaluate()
        node.score = score
        return score

    # We are playing this state, so we want to find max score
    if not node.state.is_me():
        return find_max_score(node, depth - 1)
    # Opponent is playing this state, and they want to find theirs max - so ours min
    else:
        return find_min_score(node, depth - 1)

def find_max_score(node: TreeNode[IGameState], depth) -> float:
    max_score = float('-inf')

    for state in node.state.generate_possible_states():
        child_node = node.add_child(state)

        max_score = max(
            max_score,
            find_score(child_node, depth - 1)
        )

    return max_score

def find_min_score(node: TreeNode[IGameState], depth) -> float:
    min_score = float('inf')

    for state in node.state.generate_possible_states():
        child_node = node.add_child(state)

        min_score = min(
            min_score,
            find_score(child_node, depth - 1)
        )

    return min_score



if __name__ == "__main__":
    #
    # Example Usage with simpler simple tick tack toe
    #

    # Start with me, empty board
    root = TreeNode(TicTacToeState(Field.me(), None))

    # Run minimax
    best_state = minimax(root)

    print(f"Best state:\n{best_state} \n\n")
    print("Tree:")
    root.print_tree()



