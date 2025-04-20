import sys
import time
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS
)

from sarsa import SARSAAgent 

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']

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


def main():
    print("Welcome to Mancala: Play against SARSA AI!")

    while True:
        try:
            num_episodes = int(input("How many episodes should the AI train for before playing? (e.g. 5000): "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'

    print(f"Training SARSA AI for {num_episodes} episodes...")
    agent = SARSAAgent()
    agent.train(num_episodes)

    print("Training complete. Starting game!")

    board = getNewBoard()
    player_turn = '1'

    while True:
        print('\n' * 60)
        displayBoard(board)

        if player_turn == human:
            move = ask_for_player_move(player_turn, board)
        else:
            print(f"SARSA AI ({ai}) is thinking...")
            time.sleep(1)
            state = agent.get_state(board)
            pits = PLAYER_1_PITS if player_turn == '1' else PLAYER_2_PITS
            valid_moves = [p for p in pits if board[p] > 0]
            if not valid_moves:
                print("AI has no valid moves.")
                break
            move = agent.epsilon_greedy(state, valid_moves)
            print(f"SARSA AI chooses: {move}")

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
