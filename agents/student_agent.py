=# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import (
    random_move,
    count_capture,
    execute_move,
    check_endgame,
    get_valid_moves,
    get_directions,
)

@register_agent("student_agent")
class StudentAgent(Agent):

    """
    I have included helper functions for the heuristics, as well as the main logic for my agent.

    Functions:
        count_frontier_discs    - counts the number of discs that are adjacent to empty squares
        count_corner_discs      - counts the number of discs present on corners, i.e., (0,0), (0, N - 1), (N - 1, 0), (N - 1, N -1)
        count_stable_discs      - counts the number of discs that cannot be outflanked
            neighbors                   - returns all the squares around a disc
            is_connected_to_corner      - returns if a disc is connected to a corner (only used for an edge disc)
            is_stable_disc              - helper function that determines if a disc is stable
        evaluate_board          - evaluates the board given the heuristics and their multipliers
        create_move_ordering    - given valid moves, this function sorts and prunes moves
        minimax                 - minimax function with alpha-beta pruning
        step                    - calls minimax and uses iterative deepening search

    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True

    def count_frontier_discs(self, chess_board, player):
        """
        Frontier discs are defined as discs that border one or more empty squares.

        Returns
        -------
        int
            Number of discs adjacent to empty squares
        """

        # Get directions
        directions = get_directions()
        frontier_count = 0
        N = np.size(chess_board, 0)

        # Iterate through the chess board and count frontier discs
        for r in range(N):
            for c in range(N):
                if chess_board[r, c] == player:
                    for (dr, dc) in directions:
                        (nr, nc) = (r + dr, c + dc)
                        if 0 <= nr < N and 0 <= nc < N and chess_board[nr, nc] == 0:
                            frontier_count += 1
                            break
        return frontier_count

    def count_corner_discs(self, chess_board, player):
        """
        Counts the number of discs on corners

        Returns
        -------
        int
            Number of discs on corners

        """

        N = np.size(chess_board, 0)
        corners = [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]

        return sum(1 for (r, c) in corners if chess_board[r, c] == player)

    def count_stable_discs(self, chess_board, player):
        """
        Stable discs are discs that cannot be turned over. Usually present on the edges where we hold a corner.

        Returns
        -------
        int
            Number of stable discs
        """

        # Get the board size (e.g., 8 for an 8x8 board)
        N = np.size(chess_board, 0)

        def neighbors(row, col):
            """
            Neighbors are discs surrounding you

            Returns
            -------
            int
                Number of neighbors
            """

            return [
                (row + dr, col + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
            ]

        # Identify corners and edges
        corners = [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]
        edges = [(i, j) for i in [0, N - 1] for j in range(1, N - 1)] + [
            (i, j) for i in range(1, N - 1) for j in [0, N - 1]
        ]

        def is_connected_to_corner(row, col):
            """
            Check if a disc on the edge is connected to a corner via a chain of discs of the same player.

            Returns
            -------
            boolean
                If a disc is connected to a corner (used on edges)
            """

            # Corner case
            if (row, col) in corners:
                return True

            visited = set()
            to_visit = [(row, col)]

            # DFS implementaton
            while to_visit:
                (r, c) = to_visit.pop()
                if (r, c) in visited:
                    continue
                visited.add((r, c))

                # If we reach a corner, the disc is stable
                if (r, c) in corners:
                    return True

                # Visit all appropriate neighboring squares
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    (nr, nc) = (r + dr, c + dc)
                    if 0 <= nr < N and 0 <= nc < N and chess_board[nr, nc] == player:
                        to_visit.append((nr, nc))
            return False

        # Helper function to check if a disc is stable
        def is_stable_disk(row, col):
            """
            A disc is stable if:
            - It is in a corner (always stable).
            - It is in an edge and connected to a corner.
            - It is surrounded by same-player discs (for inner discs).

            Returns
            -------
            boolean
                If a disc is stable
            """

            # Check if the disc is in a corner
            if (row, col) in corners:
                return True

            # Check if it's an edge disc connected to a corner
            if (row, col) in edges:
                return is_connected_to_corner(row, col)

            # For non-edge discs, check if all neighbors are the same player
            return all(chess_board[r, c] == player for (r, c) in neighbors(row, col))

        # Calculate the stable disc count for the player
        stable_count = 0
        for r in range(N):
            for c in range(N):
                if chess_board[r, c] == player and is_stable_disk(r, c):
                    stable_count += 1

        return stable_count

    def evaluate_board(self, chess_board, player):
        """
        Evaluate the board state based on the current game phase.
        The higher the score, the better the board is for the player

        Returns
        -------
        int
            Score of board
        """

        total_moves = np.count_nonzero(chess_board)
        max_moves = np.size(chess_board)
        opponent = 3 - player
        score = 0

        # Heuristics computations
        mobility_count = len(get_valid_moves(chess_board, player)) - len(
            get_valid_moves(chess_board, opponent)
        )
        corner_count = self.count_corner_discs(
            chess_board, player
        ) - self.count_corner_discs(chess_board, opponent)
        stable_count = self.count_stable_discs(
            chess_board, player
        ) - self.count_stable_discs(chess_board, opponent)
        frontier_count = self.count_frontier_discs(
            chess_board, player
        ) - self.count_frontier_discs(chess_board, opponent)
        discs_count = np.sum(chess_board == player) - np.sum(chess_board == opponent)

        (is_game_finished, _, _) = check_endgame(chess_board, player, 3 - player)

        # At the end only the number of discs matter
        if is_game_finished:
            return discs_count

        # Opening
        elif total_moves < max_moves * 0.25:

            score += corner_count * 15
            score += stable_count * 1
            score += mobility_count * 10
            score += frontier_count * -5

        # Middle Game
        elif max_moves * 0.25 <= total_moves <= max_moves * 0.75:

            score += corner_count * 25
            score += stable_count * 10
            score += mobility_count * 15
            score += frontier_count * -4
            score += discs_count * 4

        # End game
        else:
            score += corner_count * 15
            score += stable_count * 10
            score += mobility_count * 5
            score += discs_count * 30

        return score

    def create_move_ordering(
        self,
        chess_board,
        valid_moves,
        player,
    ):
        """
        Given all valid moves, this function prunes and orders them.
        The order is:
        - corners
        - safe x and c moves
        - edges
        - all but risky x and c moves

        Returns
        -------
        [(tuple)]
            Up to 5 tuples

        """

        N = np.size(chess_board, 0)

        # Define different sets of squares
        corners = [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]

        edges = [(i, j) for i in [0, N - 1] for j in range(N)] + [
            (i, j) for i in range(N) for j in [0, N - 1]
        ]

        c_squares = [
            (0, 1),
            (1, 0),
            (0, N - 2),
            (1, N - 1),
            (N - 1, 1),
            (N - 2, 0),
            (N - 1, N - 2),
            (N - 2, N - 1),
        ]

        edges_without_corner_and_c = [
            pos for pos in edges if pos not in c_squares and pos not in corners
        ]

        corner_moves = [move for move in valid_moves if move in corners]
        edges_without_corner_and_c_moves = [
            move for move in valid_moves if move in edges_without_corner_and_c
        ]

        # Define the X-square positions and their corresponding corners
        x_squares_with_corners = {
            (1, 1): (0, 0),
            (1, N - 2): (0, N - 1),
            (N - 2, 1): (N - 1, 0),
            (N - 2, N - 2): (N - 1, N - 1),
        }

        # Define the C-square positions and their corresponding corners
        c_squares_with_corners = {
            (0, 1): (0, 0),
            (1, 0): (0, 0),
            (0, N - 2): (0, N - 1),
            (1, N - 1): (0, N - 1),
            (N - 1, 1): (N - 1, 0),
            (N - 2, 0): (N - 1, 0),
            (N - 1, N - 2): (N - 1, N - 1),
            (N - 2, N - 1): (N - 1, N - 1),
        }

        ordered_moves = []
        rest_moves = []

        # Corners are top priority
        for move in corner_moves:
            ordered_moves.append(move)

        # "X" and "C" squares with their corner controlled are also very good
        for move in valid_moves:
            if move in x_squares_with_corners:

                # If it's an X-square, ensure the corresponding corner is controlled by the player
                if (
                    chess_board[
                        x_squares_with_corners[move][0], x_squares_with_corners[move][1]
                    ]
                    == player
                ):
                    ordered_moves.append(move)

            elif move in c_squares_with_corners:
                # If it's a C-square, ensure the corresponding corner is controlled by the player
                if (
                    chess_board[
                        c_squares_with_corners[move][0], c_squares_with_corners[move][1]
                    ]
                    == player
                ):
                    ordered_moves.append(move)
            elif (
                move not in edges_without_corner_and_c_moves
                and move not in corner_moves
            ):

                rest_moves.append(move)

        # Other edge moves should be considered next
        for move in edges_without_corner_and_c_moves:
            ordered_moves.append(move)

        # Other safe moves are then considered
        for move in rest_moves:
            ordered_moves.append(move)

        # Return at most 5 moves
        return ordered_moves[:5]

    def minimax(
        self,
        chess_board,
        depth,
        player,
        alpha,
        beta,
        maximizing_player,
        start_time,
        time_limit,
    ):
        """
        Performs minimax with alpha beta pruning while respecting time constraints

        Returns
        -------
            int, (tuple)
            The score and the following move to play
        """

        if time.time() - start_time > time_limit:
            return (None, None)

        # Base case: end of search depth or game over
        (is_endgame, _, _) = check_endgame(chess_board, player, 3 - player)

        if depth == 0 or is_endgame:
            if maximizing_player:
                return (self.evaluate_board(chess_board, player), None)
            else:
                # Important to negate the evaluation as the min player, so that it chooses the most negative option possible
                return (-1 * self.evaluate_board(chess_board, player), None)

        valid_moves = get_valid_moves(chess_board, player)

        if not valid_moves:

            # No valid moves; opponent plays
            return (
                self.minimax(
                    chess_board,
                    depth - 1,
                    3 - player,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    time_limit,
                )[0],
                None,
            )

        # Prune and order moves
        # In both branches, I use numpy.ndarray.copy() instead of deepcopy.copy()
        # as it is more efficient (integers are not reference types so creating shallow copies is ok)

        ordered_moves = self.create_move_ordering(chess_board, valid_moves, player)
        best_move = None

        if maximizing_player:

            max_eval = float("-inf")

            for move in ordered_moves:

                # Simulate move
                temp_board = chess_board.copy()
                execute_move(temp_board, move, player)

                (eval, _) = self.minimax(
                    temp_board,
                    depth - 1,
                    3 - player,
                    alpha,
                    beta,
                    False,
                    start_time,
                    time_limit,
                )

                if eval is None:
                    return (None, None)

                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, eval)

                if beta <= alpha:
                    break  # Beta cutoff
            return (max_eval, best_move)
        
        else:

            min_eval = float("inf")

            for move in ordered_moves:

                # Simulate move
                temp_board = chess_board.copy()
                execute_move(temp_board, move, player)

                (eval, _) = self.minimax(
                    temp_board,
                    depth - 1,
                    3 - player,
                    alpha,
                    beta,
                    True,
                    start_time,
                    time_limit,
                )

                if eval is None:
                    return (None, None)

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)

                if beta <= alpha:
                    break  # Alpha cutoff

            return (min_eval, best_move)

    def step(
        self,
        chess_board,
        player,
        opponent,
    ):
        """
        The main function the agent executes to make a move

        Returns
        -------
            (tuple)
            The next move to make
        """

        # Capture the time when the move search starts
        start_time = time.time()

        # Maximum time (in seconds) allowed for each move
        time_limit = 1.985

        total_moves = np.count_nonzero(chess_board)
        max_moves = np.size(chess_board)

        # Depending on the stage of the game, the maximum depth will be different
        if total_moves < max_moves * 0.25:
            max_depth = 2

        elif max_moves * 0.25 <= total_moves <= max_moves * 0.75:
            max_depth = 3

        else:
            max_depth = 5

        best_move = None
        best_score = float("-inf")

        # Get all valid moves upfront as a fallback option
        valid_moves = get_valid_moves(chess_board, player)

        # If no valid moves exist, return None
        if not valid_moves:
            return None

        # Use the first valid move as the fallback initially
        fallback_move = valid_moves[0]

        # Start the search loop with iterative deepening
        for depth in range(1, max_depth + 1):

            # Check if we've exceeded the time limit
            if time.time() - start_time >= time_limit:
                break

            # Perform minimax search with alpha-beta pruning for the current depth
            alpha = float("-inf")
            beta = float("inf")
            (score, move) = self.minimax(
                chess_board,
                depth,
                player,
                alpha,
                beta,
                True,
                start_time,
                time_limit,
            )

            # If time limit exceeded, stop
            if score is None:
                break

            # Update the best move if the score is better
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else fallback_move
