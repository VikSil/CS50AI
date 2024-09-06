"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    X-es get the first move.
    """
    flat_list = [cell for row in board for cell in row]
    empty_cells = flat_list.count(None)
    if empty_cells % 2 == 1:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    Possible actions are any cells that do not already
    have an X or an O in them.
    """
    actions = set()
    for index_i, row in enumerate(board):
        for index_j, cell in enumerate(row):
            if cell == EMPTY:
                move = (index_i, index_j)
                actions.add(move)

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    next_board = copy.deepcopy(board)
    i, j = action

    if i < 0 or j < 0:
        raise IndexError('Action out of bounds')

    if next_board[i][j] != EMPTY:
        raise ValueError('Illegal Move')

    next_board[i][j] = player(board)

    return next_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    players = [X, O]

    for player in players:
        # check rows
        for row in board:
            if all(cell == player for cell in row):
                return player

        # check columns
        for j in range(3):
            column = [board[i][j] for i in range(3)]
            if all(cell == player for cell in column):
                return player

        # check diagonals
        dexter = [board[i][i] for i in range(3)]
        if all(cell == player for cell in dexter):
            return player

        sinister = [board[-(i + 1)][i] for i in range(3)]
        if all(cell == player for cell in sinister):
            return player

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True

    flat_list = [cell for row in board for cell in row]
    empty_cells = flat_list.count(None)
    if empty_cells == 0:
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    next_player = player(board)
    next_action = None

    if terminal(board):
        return next_action

    elif next_player == X:
        v = -math.inf
        for action in actions(board):
            game_result = min_value(result(board, action))
            if game_result > v:
                v = game_result
                next_action = action

    else:
        v = math.inf
        for action in actions(board):
            game_result = max_value(result(board, action))
            if game_result < v:
                v = game_result
                next_action = action

    return next_action


def max_value(board):
    """
    Implements board evaluation from max player perspective
    """
    if terminal(board):
        return utility(board)

    else:
        v = -math.inf
        for action in actions(board):
            v = max(v, min_value(result(board, action)))
        return v


def min_value(board):
    """
    Implements board evaluation from min player perspective
    """
    if terminal(board):
        return utility(board)

    else:
        v = math.inf
        for action in actions(board):
            v = min(v, max_value(result(board, action)))
        return v
