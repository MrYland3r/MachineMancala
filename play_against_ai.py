import torch
import torch.nn as nn
import time
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS, askForPlayerMove
)

PIT_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']

# Deep Q Network (same as in training)
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

class DeepQAgent:
    def __init__(self, player, model_path):
        assert player in ['1', '2']
        self.player = player
        self.pits = PLAYER_1_PITS if player == '1' else PLAYER_2_PITS

        self.model = QNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"[INFO] Loaded Deep SARSA model from {model_path}")

    def get_state_vector(self, board):
        return torch.FloatTensor([board[p] for p in PIT_ORDER])

    def choose_action(self, board):
        state_vec = self.get_state_vector(board)
        valid_moves = [i for i, p in enumerate(self.pits) if board[p] > 0]
        if not valid_moves:
            return None

        with torch.no_grad():
            q_values = self.model(state_vec)
            mask = torch.tensor([float('-inf') if i not in valid_moves else 0.0 for i in range(6)])
            q_values += mask
            return self.pits[torch.argmax(q_values).item()]

def main():
    print("Welcome to Mancala: Play against Deep SARSA AI!")
    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'
    model_path = f"checkpoints/deep_sarsa_p{ai}_latest.pt"
    agent = DeepQAgent(player=ai, model_path=model_path)

    board = getNewBoard()
    player_turn = '1'

    while True:
        print('\n' * 60)
        displayBoard(board)

        if player_turn == human:
            move = askForPlayerMove(player_turn, board)
        else:
            print(f"Deep SARSA AI ({ai}) is thinking...")
            time.sleep(1)
            move = agent.choose_action(board)
            if move is None:
                print("AI has no valid moves.")
                break
            print(f"Deep SARSA AI chooses: {move}")

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
