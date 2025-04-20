import sys
import time
import random
import pickle
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS
)

# --- Constants ---
PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']
Q_TABLE_FILE = 'qtable.pkl'

# --- Load Q-table ---
def load_q_table(filename=Q_TABLE_FILE):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Q-table not found. AI will play randomly.")
        return {}

# --- Get state representation for Q-table lookup ---
def get_state(board):
    return tuple(board[p] for p in PIT_ORDER)

# --- Choose best or random action ---
def choose_action(state, q_table, player, board, epsilon=0.0):
    valid_moves = [p for p in (PLAYER_1_PITS if player == '1' else PLAYER_2_PITS) if board[p] > 0]
    if not valid_moves:
        return None
    if random.random() < epsilon:
        return random.choice(valid_moves)
    q_values = [q_table.get((state, a), 0) for a in valid_moves]
    return valid_moves[q_values.index(max(q_values))]

# --- Get human player input ---
def ask_for_player_move(player, board):
    pits = PLAYER_1_PITS if player == '1' else PLAYER_2_PITS
    while True:
        move = input(f"Your move (Player {player}) {pits} or QUIT: ").upper().strip()
        if move == 'QUIT':
            print("Thanks for playing!")
            sys.exit()
        if move not in pits:
            print("Invalid pit. Choose one on your side.")
            continue
        if board[move] == 0:
            print("That pit is empty. Try again.")
            continue
        return move

# --- Main Game Loop ---
def main():
    print("Welcome to Mancala: Play against AI!")
    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'
    q_table = load_q_table()

    board = getNewBoard()
    player_turn = '1'

    while True:
        print('\n' * 60)
        displayBoard(board)

        if player_turn == human:
            move = ask_for_player_move(player_turn, board)
        else:
            print(f"AI ({ai}) is thinking...")
            time.sleep(1)
            state = get_state(board)
            move = choose_action(state, q_table, player_turn, board)
            if move is None:
                print("AI has no valid moves.")
                break
            print(f"AI chooses: {move}")

        player_turn, board = makeMove(board, player_turn, move)

        winner = checkForWinner(board)
        if winner in ['1', '2', 'tie']:
            print('\nFinal Board:')
            displayBoard(board)
            if winner == 'tie':
                print("It's a tie!")
            else:
                print(f"Player {winner} wins!")
            break

if __name__ == '__main__':
    main()
