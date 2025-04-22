import numpy as np
import random
import pickle
import argparse
from mancala import getNewBoard, makeMove, checkForWinner, PLAYER_1_PITS, PLAYER_2_PITS, OPPOSITE_PIT

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']
PLAYER_1_STORE = '1'
PLAYER_2_STORE = '2'

class SARSAAgent:
    def __init__(self, player, alpha=0.1, gamma=0.95, epsilon=0.1, episodes=300000):
        assert player in ['1', '2']
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table_file = f'qtable-p{player}.pkl'
        self.q_table = self.load_q_table() or {}
        print(f"[INFO] Loaded Q-table with {len(self.q_table)} entries for Player {player}.")

    def epsilon_greedy(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.q_table.get((state, a), 0) for a in actions]
        return actions[np.argmax(q_values)]

    def get_state(self, board):
        return tuple(board[pit] for pit in PIT_ORDER)

    def load_q_table(self):
        try:
            with open(self.q_table_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def opponent_move(self, board, pits):
        valid = [p for p in pits if board[p] > 0]
        if not valid:
            return None
        return max(valid, key=lambda p: (board[p], p))

    def train(self):
        epsilon_decay = 0.9998
        epsilon_min = 0.05
        wins = 0

        for ep in range(1, self.episodes + 1):
            board = getNewBoard()
            state = self.get_state(board)
            player_turn = '1'
            done = False
            action = None
            seen_states = set()

            if ep > 100:
                self.epsilon = max(self.epsilon * epsilon_decay, epsilon_min)
                self.alpha = max(self.alpha * 0.999, 0.01)

            while not done:
                state_key = tuple(board[p] for p in PIT_ORDER)
                if state_key in seen_states:
                    break
                seen_states.add(state_key)

                is_ai_turn = (player_turn == self.player)
                store = PLAYER_1_STORE if self.player == '1' else PLAYER_2_STORE
                pits = PLAYER_1_PITS if player_turn == '1' else PLAYER_2_PITS

                if is_ai_turn:
                    valid_moves = [p for p in pits if board[p] > 0]
                    if not valid_moves:
                        break

                    prev_diff = board[PLAYER_1_STORE] - board[PLAYER_2_STORE]
                    action = self.epsilon_greedy(state, valid_moves)
                    next_turn, next_board = makeMove(board.copy(), player_turn, action)
                    next_state = self.get_state(next_board)

                    new_diff = next_board[PLAYER_1_STORE] - next_board[PLAYER_2_STORE]
                    reward = 1 if new_diff > prev_diff else -1 if new_diff < prev_diff else 0
                    if board[action] == 1 and OPPOSITE_PIT.get(action) in (
                        PLAYER_2_PITS if self.player == '1' else PLAYER_1_PITS
                    ) and board[OPPOSITE_PIT[action]] > 0:
                        reward += 2
                    if next_turn == self.player and action == store:
                        reward += 1

                    next_valid_moves = [p for p in pits if next_board[p] > 0]
                    next_action = self.epsilon_greedy(next_state, next_valid_moves) if next_valid_moves else action

                    winner = checkForWinner(next_board)
                    if winner != 'no winner':
                        final_reward = 20 if winner == self.player else -20 if winner != 'tie' else 0
                        reward += final_reward
                        q_val = self.q_table.get((state, action), 0)
                        self.q_table[(state, action)] = q_val + self.alpha * (reward - q_val)
                        if winner == self.player:
                            wins += 1
                        break
                    else:
                        q_val = self.q_table.get((state, action), 0)
                        next_q_val = self.q_table.get((next_state, next_action), 0)
                        self.q_table[(state, action)] = q_val + self.alpha * (reward + self.gamma * next_q_val - q_val)

                    board = next_board
                    state = next_state
                    action = next_action
                    player_turn = next_turn
                else:
                    move = self.opponent_move(board, pits)
                    if not move:
                        break
                    player_turn, board = makeMove(board, player_turn, move)

                    winner = checkForWinner(board)
                    if winner != 'no winner':
                        reward = 20 if winner == self.player else -20 if winner != 'tie' else 0
                        if action:
                            q_val = self.q_table.get((state, action), 0)
                            self.q_table[(state, action)] = q_val + self.alpha * (reward - q_val)
                        if winner == self.player:
                            wins += 1
                        break

            if ep % 10000 == 0:
                print(f"[TRAIN] Episode {ep}/{self.episodes} | Q-table size: {len(self.q_table)} | Epsilon: {self.epsilon:.4f}")

        self.save_q_table()
        print(f"[DONE] Training complete for Player {self.player}.")
        print(f"[RESULT] Wins: {wins}/{self.episodes} ({wins/self.episodes:.2%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--player', type=str, choices=['1', '2'], required=True, help="Train for player 1 or 2")
    parser.add_argument('--episodes', type=int, default=300000, help="Number of training episodes")
    args = parser.parse_args()

    agent = SARSAAgent(player=args.player, episodes=args.episodes)
    agent.train()
