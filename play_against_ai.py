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
            num_episodes = int(input(
                "How many episodes should the AI train for before playing?\nThe more trained, the harder the difficulty (e.g. Easy-2000, Med-5000, Hard-8000): "))
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

    ai_moves = []  # Track ai moves
    human_moves = [] # Track human moves

    while True:
        print('\n' * 60)
        displayBoard(board)

        if player_turn == human:
            state = agent.get_state(board)  # Capture the state before the move
            move = ask_for_player_move(player_turn, board)
            player_turn, board = makeMove(board, player_turn, move)
            next_state = agent.get_state(board)  # Capture the new state
            human_moves.append((state, move, next_state))  # Log the move

        else:
            print(f"SARSA AI ({ai}) is thinking...")
            time.sleep(1)

            # Step 2: Get state & valid moves
            state = agent.get_state(board)
            pits = PLAYER_1_PITS if player_turn == '1' else PLAYER_2_PITS
            valid_moves = [p for p in pits if board[p] > 0]

            if not valid_moves:
                print("AI has no valid moves.")
                break

            move = agent.epsilon_greedy(state, valid_moves)
            print(f"SARSA AI chooses: {move}")

            # Step 3: Make move, then get next state
            player_turn, board = makeMove(board, player_turn, move)
            next_state = agent.get_state(board)

            ai_moves.append((state, move, next_state))

        player_turn, board = makeMove(board, player_turn, move)

        winner = checkForWinner(board)
        if winner in ['1', '2', 'tie']:
            print('\nFinal Board:')
            displayBoard(board)
            if winner == 'tie':
                print("It's a tie!")
            else:
                print(f"Player {winner} wins!")
                print("\n--- Game Move History ---")
                total_turns = len(human_moves) + len(ai_moves)
                human_idx = 0
                ai_idx = 0
                player_turn = '1'  # Start from Player 1 (adjust if needed)

                for turn in range(total_turns):
                    if player_turn == human and human_idx < len(human_moves):
                        state, action, next_state = human_moves[human_idx]
                        print(f"Turn {turn+1} - Human (Player {human}):")
                        print(f"  State: {state}")
                        print(f"  Action: {action}")
                        print(f"  Next State: {next_state}")
                        human_idx += 1
                        player_turn = ai  # switch to AI
                    elif player_turn == ai and ai_idx < len(ai_moves):
                        state, action, next_state = ai_moves[ai_idx]
                        print(f"Turn {turn+1} - AI (Player {ai}):")
                        print(f"  State: {state}")
                        print(f"  Action: {action}")
                        print(f"  Next State: {next_state}")
                        ai_idx += 1
                        player_turn = human  # switch to Human
            break

if __name__ == '__main__':
    main()
