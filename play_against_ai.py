import time
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS, askForPlayerMove
)
from sarsa import SARSAAgent

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']

def main():
    print("Welcome to Mancala: Play against SARSA AI!")
    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'

    # Load pretrained AI Q-table
    agent = SARSAAgent(q_table_file='qtable-200k.pkl', load_existing=True)

    board = getNewBoard()
    player_turn = '1'

    while True:
        print('\n' * 60)
        displayBoard(board)

        if player_turn == human:
            move = askForPlayerMove(player_turn, board)
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
