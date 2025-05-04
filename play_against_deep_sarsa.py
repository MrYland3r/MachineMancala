import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS, askForPlayerMove
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

class DeepSARSAPlayer:
    def __init__(self, player, lr=0.001, gamma=0.95):
        assert player in ['1', '2']
        self.player = player
        self.pits = PLAYER_1_PITS if player == '1' else PLAYER_2_PITS
        self.store = '1' if player == '1' else '2'
        self.model_path = f'deep_sarsa_p{player}.pt'
        self.gamma = gamma

        self.model = QNetwork()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"[INFO] Loaded model for Player {player} from {self.model_path}")
        except FileNotFoundError:
            print(f"[INFO] No existing model found. Starting fresh.")

        self.model.train()

    def get_state_vector(self, board):
        vec = [board[p] for p in PIT_ORDER]
        return torch.FloatTensor(vec).cuda() if torch.cuda.is_available() else torch.FloatTensor(vec)

    def choose_action(self, board, epsilon=0.0, show_q=True):
        state_vec = self.get_state_vector(board)
        valid_indices = [i for i, p in enumerate(self.pits) if board[p] > 0]
        if not valid_indices:
            return None, None

        with torch.no_grad():
            q_values = self.model(state_vec)
            if show_q:
                print("[Q-VALUES]:")
                for i in valid_indices:
                    print(f"  {self.pits[i]}: {q_values[i].item():.2f}")

            mask = torch.tensor([float('-inf') if i not in valid_indices else 0.0 for i in range(6)])
            if torch.cuda.is_available():
                mask = mask.cuda()
            q_values += mask

            best_index = torch.argmax(q_values).item()
            return self.pits[best_index], best_index

    def update(self, s, a, r, s_next, a_next_idx, done):
        q_vals = self.model(s)
        target = q_vals.clone().detach()
        next_q_vals = self.model(s_next)

        if done:
            target[a] = r
        else:
            target[a] = r + self.gamma * next_q_vals[a_next_idx].item()

        loss = self.criterion(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.save(self.model.state_dict(), self.model_path)

def main():
    print("Welcome to Mancala: Play against Deep SARSA AI (learning mode)!")
    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'
    agent = DeepSARSAPlayer(player=ai)

    board = getNewBoard()
    player_turn = '1'

    s, a_idx = None, None  # For tracking the previous state/action

    while True:
        print('\n' * 60)
        displayBoard(board)

        if player_turn == human:
            move = askForPlayerMove(player_turn, board)
        else:
            print(f"Deep SARSA AI ({ai}) is thinking...")
            time.sleep(1)

            s = agent.get_state_vector(board)
            move, a_idx = agent.choose_action(board, show_q=True)
            if move is None:
                print("AI has no valid moves.")
                break
            print(f"Deep SARSA AI chooses: {move}")

        player_turn, next_board = makeMove(board.copy(), player_turn, move)

        winner = checkForWinner(next_board)
        done = winner in ['1', '2', 'tie']

        if player_turn == ai and a_idx is not None and not done:
            s_next = agent.get_state_vector(next_board)
            _, a_next_idx = agent.choose_action(next_board)
            reward = agent_reward(board, next_board, ai)
            agent.update(s, a_idx, reward, s_next, a_next_idx, done)

        board = next_board

        if done:
            print('\nFinal Board:')
            displayBoard(board)
            print("It's a tie!" if winner == 'tie' else f"Player {winner} wins!")
            break

def agent_reward(prev, curr, ai_store):
    ai_store = '1' if ai_store == '1' else '2'
    opp_store = '2' if ai_store == '1' else '1'
    return (curr[ai_store] - prev[ai_store]) - (curr[opp_store] - prev[opp_store])

if __name__ == '__main__':
    torch.manual_seed(42)
    main()
