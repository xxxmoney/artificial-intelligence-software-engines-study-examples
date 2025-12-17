import random

from src.game_state import IGameState, GameStatus
from src.tic_tac_toe_simple import TicTacToeState, Field
from src.tree import TreeNode


#
# MCTS - Monte Carlo Tree Search is a principle of finding best move in the game based on statistics
# In contrast to MINIMAX, it doesn't need an evaluation function for each state, instead, it uses rather "statistically based brute-forcy" approach
# This algorithm plays thousands and thousands of pseudo-random games, until the finish - so either win/defeat games
# For each state, it remembers the number of play-thorughs and the number of wins of its descendants
# It then uses statistics to choose the one with the most likelyhood
#

def mcts(node: TreeNode[IGameState], try_count: int) -> TreeNode[IGameState]:
    for i in range(try_count):
        explore_path(node)

    return node.best_child

def explore_path(node: TreeNode[IGameState]) -> int:
    # Make sure children are present
    check_children(node)

    # Choose child to be explored
    visited = choose_node(node.children)

    # If visited is still running, explore further
    if visited.state.status == GameStatus.RUNNING:
        score = explore_path(visited)
    # Win, positive score
    elif visited.state.status == GameStatus.WIN:
        score = 1
    # Defeat, negative score
    elif visited.state.status == GameStatus.DEFEAT:
        score = -1
    # Draw, neutral score
    else:
        score = 0

    node.score += score
    node.visits += 1

    # Visited is final, set also its score
    if visited.state.status != GameStatus.RUNNING:
        visited.score += score
        visited.visits += 1

    # Set the best child to the node
    visited_children = [node for node in node.children if node.visits]
    node.best_child = max(visited_children, key=lambda child: child.score_visits_ratio)

    return score

def check_children(node: TreeNode[IGameState]) -> None:
    if not node.children:
        for state in node.state.possible_next_states:
            node.add_child(state)

def choose_node(nodes: list[TreeNode[IGameState]]) -> TreeNode[IGameState]:
    # Get not visited states
    selected_nodes = [node for node in nodes if not node.visits]

    # If all nodes were visited, we will choose from the visited
    if not selected_nodes:
        selected_nodes = nodes

    # Firstly prefer winning node
    win_node = next((node for node in selected_nodes if node.state == GameStatus.WIN), None)
    if win_node:
        return win_node

    # If there is no winning node, then try draw
    draw_node = next((node for node in selected_nodes if node.state == GameStatus.DRAW), None)
    if draw_node:
        return draw_node

    # If none is winning
    defeat_node = next((node for node in selected_nodes if node.state == GameStatus.DEFEAT), None)
    if defeat_node:
        return defeat_node

    return random.choice(selected_nodes)


if __name__ == "__main__":
    #
    # Example Usage with simpler simple tick tack toe
    #

    # Start with me, empty board
    root = TreeNode(TicTacToeState(Field.me(), None))

    # MCTS builds the tree and sets the score (wins) and visits
    best_state = mcts(root, 500)

    print(f"Best state:\n{best_state} \n\n")
    print("Tree:")
    root.print_tree()

