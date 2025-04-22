import numpy as np
import pickle
import random
import time
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS, askForPlayerMove
)

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']

class QAgent:
    def __init__(self, player):
        assert player in ['1', '2']
        self.player = player
        self.q_table_file = f'qtable-newp{player}.pkl'
        try:
            with open(self.q_table_file, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"[INFO] Loaded Q-table for Player {player} with {len(self.q_table)} entries.")
        except FileNotFoundError:
            print(f"[ERROR] Could not find Q-table: {self.q_table_file}")
            self.q_table = {}

    def get_state(self, board):
        return tuple(board[p] for p in PIT_ORDER)

    def approximate_q_value(self, state, action, tolerance=1):
        candidates = [
            q_val for (s, a), q_val in self.q_table.items()
            if a == action and len(s) == len(state)
            and all(abs(s[i] - state[i]) <= tolerance for i in range(len(state)))
        ]
        if candidates:
            return sum(candidates) / len(candidates)
        else:
            return 0

    def choose_action(self, board):
        pits = PLAYER_1_PITS if self.player == '1' else PLAYER_2_PITS
        valid_moves = [p for p in pits if board[p] > 0]
        state = self.get_state(board)

        print(f"[DEBUG] Current state: {state}")
        found_any = False
        for p in valid_moves:
            q = self.q_table.get((state, p), 'MISSING')
            print(f"  Q({p}) = {q}")
            if q != 'MISSING':
                found_any = True

        q_values = np.array([
            self.q_table.get((state, a), self.approximate_q_value(state, a))
            for a in valid_moves
        ])

        if not found_any:
            print(f"[APPROX] Using approximated Q-values (fallback).")

        if all(q == 0 for q in q_values):
            print("[INFO] All Q-values are 0 — choosing arbitrarily.")
            return random.choice(valid_moves)

        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        print("[Q-CHOICES]", dict(zip(valid_moves, np.round(probs, 3))))

        best_move = valid_moves[np.argmax(q_values)]
        print(f"[SARSA AI chooses: {best_move}]")
        return best_move


def main():
    print("Welcome to Mancala: Play against SARSA AI!")
    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'
    agent = QAgent(player=ai)

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
            move = agent.choose_action(board)
            if move is None:
                print("AI has no valid moves.")
                break
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
    import random
    main()
