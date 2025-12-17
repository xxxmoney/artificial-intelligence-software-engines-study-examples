import math
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
# An important part of the MCTS is UCT - this is an algorithm choosed for choosing which child node to explore
# UCT is basically in simple terms - while I want to prefer winning nodes, I also want to try explore unexplored nodes
# UCT uses two parts - exploitation (tendency to choose the winning nodes) and exploration (tendency to choose the unknown nodes)
# Balance between these two is maintained by the EXPLORATION_CONSTANT - this says how much we want to try and explore unknown nodes
#

# Sets how much we want try try exploring new nodes
# Lower values (less then 0.5) - really focused more towards the "sure way" - not trying new ways much
# Higher values (more then 2) - really trying out new ways even though they could fail
# Middle ground (around 1.414) - one-armed bandit - a compromise
EXPLORATION_CONSTANT = 1.414

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

def choose_node(nodes: list[TreeNode[IGameState]]) -> TreeNode[IGameState]:
    not_visited_nodes = [node for node in nodes if node.visits == 0]

    # We want to try not visited at least once
    if not_visited_nodes:
        return random.choice(not_visited_nodes)

    # We want to use the one with max UCT
    return max(nodes, key=get_uct_score)

def get_uct_score(child: TreeNode[IGameState]) -> float:
    # How good is this state (turn)? - goes towards successful ones (we want winning ones)
    exploitation = child.score_visits_ratio

    # How few times have we been here? - goes towards lower explored ones (we want to try unexplored ones)
    exploration = EXPLORATION_CONSTANT * math.sqrt(math.log(child.parent.visits) / child.visits)

    # Combination of both - while we want to have successful ones, we also want to try unexplored ones
    return exploitation + exploration

def check_children(node: TreeNode[IGameState]) -> None:
    if not node.children:
        for state in node.state.possible_next_states:
            node.add_child(state)


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

