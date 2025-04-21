import numpy as np
import random
import pickle
import os
from mancala import getNewBoard, makeMove, checkForWinner, PLAYER_1_PITS, PLAYER_2_PITS, OPPOSITE_PIT

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']
PLAYER_1_STORE = '1'
PLAYER_2_STORE = '2'

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1,
                 use_greedy_opponent=True, q_table_file='qtable-200k.pkl', load_existing=True):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_greedy_opponent = use_greedy_opponent
        self.q_table_file = q_table_file
        self.q_table = {} if not load_existing else self.load_q_table() or {}

    def epsilon_greedy(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.q_table.get((state, a), 0) for a in actions]
        return actions[np.argmax(q_values)]

    def get_state(self, board):
        return tuple(board[pit] for pit in PIT_ORDER)

    def reset_game(self):
        board = getNewBoard()
        return board, self.get_state(board)

    def opponent_move(self, board, pits):
        valid = [p for p in pits if board[p] > 0]
        if not valid:
            return None
        return max(valid, key=lambda p: (board[p], p))

    def load_q_table(self):
        try:
            with open(self.q_table_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def train(self, episodes):
        epsilon_decay = 0.9995
        epsilon_min = 0.01
        wins = 0

        for ep in range(episodes):
            board, state = self.reset_game()
            player_turn = '1'
            done = False
            steps = 0
            total_reward = 0
            action = None
            seen_states = set()

            if ep > 100:
                self.epsilon = max(self.epsilon * epsilon_decay, epsilon_min)
                self.alpha = max(self.alpha * 0.999, 0.01)

            while not done:
                state_key = tuple(board[p] for p in PIT_ORDER)
                if state_key in seen_states:
                    reward = -10
                    total_reward += reward
                    break
                seen_states.add(state_key)

                if player_turn == '1':
                    valid_moves = [p for p in PLAYER_1_PITS if board[p] > 0]
                    if not valid_moves:
                        reward = -10
                        total_reward += reward
                        done = True
                        break

                    prev_diff = board[PLAYER_1_STORE] - board[PLAYER_2_STORE]
                    action = self.epsilon_greedy(state, valid_moves)
                    next_turn, next_board = makeMove(board.copy(), '1', action)
                    next_state = self.get_state(next_board)

                    new_diff = next_board[PLAYER_1_STORE] - next_board[PLAYER_2_STORE]
                    shaping = 1 if new_diff > prev_diff else -1 if new_diff < prev_diff else 0
                    reward = shaping

                    if board[action] == 1 and OPPOSITE_PIT.get(action) in PLAYER_2_PITS and board[OPPOSITE_PIT[action]] > 0:
                        reward += 2
                    if next_turn == '1' and action == PLAYER_1_STORE:
                        reward += 1

                    next_valid_moves = [p for p in PLAYER_1_PITS if next_board[p] > 0]
                    next_action = self.epsilon_greedy(next_state, next_valid_moves) if next_valid_moves else action

                    winner = checkForWinner(next_board)
                    if winner != 'no winner':
                        if winner == '1':
                            final_reward = 20
                        elif winner == '2':
                            final_reward = -20
                        else:
                            final_reward = 0
                        reward += final_reward
                        total_reward += reward
                        q_val = self.q_table.get((state, action), 0)
                        self.q_table[(state, action)] = q_val + self.alpha * (reward - q_val)
                        if winner == '1':
                            wins += 1
                        done = True
                        break
                    else:
                        q_val = self.q_table.get((state, action), 0)
                        next_q_val = self.q_table.get((next_state, next_action), 0)
                        self.q_table[(state, action)] = q_val + self.alpha * (reward + self.gamma * next_q_val - q_val)

                    board = next_board
                    state = next_state
                    action = next_action
                    player_turn = next_turn
                    total_reward += reward

                else:
                    valid_moves = [p for p in PLAYER_2_PITS if board[p] > 0]
                    if not valid_moves:
                        done = True
                        break
                    move = self.opponent_move(board, PLAYER_2_PITS)
                    player_turn, board = makeMove(board, '2', move)

                    winner = checkForWinner(board)
                    if winner != 'no winner':
                        if winner == '1':
                            final_reward = 20
                        elif winner == '2':
                            final_reward = -20
                        else:
                            final_reward = 0
                        reward = final_reward
                        total_reward += reward
                        if action is not None:
                            q_val = self.q_table.get((state, action), 0)
                            self.q_table[(state, action)] = q_val + self.alpha * (reward - q_val)
                        if winner == '1':
                            wins += 1
                        done = True
                        break

                steps += 1

            
        self.save_q_table()


if __name__ == '__main__':
    agent = SARSAAgent(q_table_file='qtable-200k.pkl', load_existing=False)
    agent.train(8000)
