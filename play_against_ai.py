import torch
import torch.nn as nn
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
    def __init__(self, player):
        assert player in ['1', '2']
        self.player = player
        self.pits = PLAYER_1_PITS if player == '1' else PLAYER_2_PITS
        self.model_path = f'deep_sarsa_p{player}.pt'

        self.model = QNetwork()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print(f"[INFO] Loaded trained model for Player {player} from {self.model_path}")

    def get_state_vector(self, board):
        return torch.FloatTensor([board[p] for p in PIT_ORDER])

    def choose_action(self, board):
        state_vec = self.get_state_vector(board)
        valid_indices = [i for i, p in enumerate(self.pits) if board[p] > 0]
        if not valid_indices:
            return None

        with torch.no_grad():
            q_values = self.model(state_vec)
            for i in valid_indices:
                print(f"  Q({self.pits[i]}) = {q_values[i].item():.2f}")
            mask = torch.tensor([float('-inf') if i not in valid_indices else 0.0 for i in range(6)])
            q_values += mask
            chosen_idx = torch.argmax(q_values).item()
            return self.pits[chosen_idx]

def main():
    print("Welcome to Mancala: Play against Deep SARSA AI!")
    human = ''
    while human not in ['1', '2']:
        human = input("Choose your side: Player 1 or 2? (Enter 1 or 2): ").strip()

    ai = '2' if human == '1' else '1'
    agent = DeepSARSAPlayer(player=ai)

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
            print("It's a tie!" if winner == 'tie' else f"Player {winner} wins!")
            break

if __name__ == '__main__':
    torch.manual_seed(42)
    main()
