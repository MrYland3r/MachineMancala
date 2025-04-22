import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
from mancala import (
    getNewBoard, makeMove, checkForWinner,
    PLAYER_1_PITS, PLAYER_2_PITS
)

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']

class QNetwork(nn.Module):
    def __init__(self, input_dim=14, output_dim=6):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DeepSARSAAgent:
    def __init__(self, player, alpha=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        assert player in ['1', '2']
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.pits = PLAYER_1_PITS if player == '1' else PLAYER_2_PITS
        self.store = '1' if player == '1' else '2'

        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

        self.model_path = f'deep_sarsa_p{player}.pt'
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"[INFO] Loaded model from {self.model_path}")

        self.total_wins = 0
        self.total_games = 0

    def get_state_vector(self, board):
        return torch.FloatTensor([board[p] for p in PIT_ORDER])

    def select_action(self, board):
        state_vec = self.get_state_vector(board)
        valid_moves = [i for i, p in enumerate(self.pits) if board[p] > 0]
        if not valid_moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        with torch.no_grad():
            q_values = self.model(state_vec)
            mask = torch.tensor([float('-inf') if i not in valid_moves else 0.0 for i in range(6)])
            q_values += mask
            return torch.argmax(q_values).item()

    def calculate_reward(self, board, next_board):
        before = next_board[self.store] - next_board['2' if self.store == '1' else '1']
        after = board[self.store] - board['2' if self.store == '1' else '1']
        return 1 if before > after else -1 if before < after else 0

    def train(self, episodes):
        for ep in range(1, episodes + 1):
            board = getNewBoard()
            state = self.get_state_vector(board)
            player_turn = '1'
            action_idx = self.select_action(board)

            while True:
                if player_turn != self.player:
                    opp_pits = PLAYER_2_PITS if self.player == '1' else PLAYER_1_PITS
                    opp_moves = [p for p in opp_pits if board[p] > 0]
                    if not opp_moves:
                        break
                    opp_move = max(opp_moves, key=lambda p: (board[p], p))
                    player_turn, board = makeMove(board, player_turn, opp_move)
                    if checkForWinner(board) != 'no winner':
                        break
                    continue

                move = self.pits[action_idx]
                player_turn, next_board = makeMove(board.copy(), self.player, move)

                reward = self.calculate_reward(board, next_board)
                next_state = self.get_state_vector(next_board)
                next_action_idx = self.select_action(next_board)

                q_values = self.model(state)
                q_next = self.model(next_state)
                target = q_values.clone()

                winner = checkForWinner(next_board)
                if winner != 'no winner':
                    reward += 20 if winner == self.player else -20
                    target[action_idx] = reward
                else:
                    target[action_idx] = reward + self.gamma * q_next[next_action_idx].item()

                loss = self.criterion(q_values, target.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                board = next_board
                state = next_state
                action_idx = next_action_idx

                if winner != 'no winner':
                    if winner == self.player:
                        self.total_wins += 1
                    self.total_games += 1
                    break

            if ep > 100:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if ep % 1000 == 0:
                print(f"[TRAIN] Episode {ep}/{episodes} | Epsilon: {self.epsilon:.4f}")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"[DONE] Training complete for Player {self.player}.")
        print(f"[RESULT] Wins: {self.total_wins}/{self.total_games} ({(self.total_wins / self.total_games) * 100:.2f}%)")
        print(f"[SAVE] Model saved to {self.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--player', type=str, choices=['1', '2'], required=True)
    parser.add_argument('--episodes', type=int, default=100000)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    agent = DeepSARSAAgent(player=args.player, epsilon=args.epsilon, alpha=args.lr)
    agent.train(episodes=args.episodes)
