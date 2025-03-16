"""This is for sarsa model"""
import numpy as np 
import random
import time
from mancala import getNewBoard, displayBoard, askForPlayerMove, makeMove, checkForWinner, PLAYER_1_PITS, PLAYER_2_PITS

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self.initialize_q_table()
        self.stats = {
                'episode_rewards': [],
                'episode_lengths': [],
                'win_rates': [],
                'execution_times': []
                }

    def initialize_q_table(self):
        return {}

    def epsilon_greedy(self, state, possible_input):
        if random.random() < self.epsilon:
            return random.choice(possible_input)
        else:
            q_values = [self.q_table.get((state, action), 0) for action in possible_input]
            return possible_input[np.argmax(q_values)]
    
    def reset_ep(self):
        board = getNewBoard()
        state = tuple(board.values())
        action=self.epsilon_greedy(state, PLAYER_1_PITS)
        return board, state, action
        
    def train(self, episodes):
        wins = 0
        losses = 0
        ties = 0
        epsilon_decay = 0.99  # Example decay rate
        epsilon_min = 0.01  # Minimum epsilon value
        epsilon = self.epsilon
        start_time = time.time()
        wins = 0
        for episode in range(episodes):
            episode_start_time = time.time()
            board, state, action = self.reset_ep()
            player_turn = '1'
            done = False
            rewards = 0
            episode_length = 0
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)
            self.epsilon = epsilon

            while not done:
                possible_input = PLAYER_1_PITS if player_turn == '1' else PLAYER_2_PITS

                action = self.epsilon_greedy(state, possible_input)

                next_player_turn, next_board = makeMove(board.copy(), player_turn, action)
                next_state = tuple(next_board.values())
                reward = 0 
                winner = checkForWinner(next_board)

                if winner == '1' or winner == '2':
                    reward = 1 if winner == player_turn else -1
                    done = True
                    if winner == player_turn:
                        wins += 1
                    else:
                        losses += 1
                elif winner == 'tie':
                    reward = 0 
                    done = True
                    ties += 1

                next_action = self.epsilon_greedy(next_state, PLAYER_1_PITS if player_turn == '2' else PLAYER_2_PITS)
                q_value = self.q_table.get((state, action), 0)
                next_q_value = self.q_table.get((next_state, next_action), 0)
                new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)

                self.q_table[(state, action)] = new_q_value
                board = next_board
                state = next_state
                action = next_action
                player_turn = next_player_turn
                rewards += reward
                episode_length += 1

            episode_end_time = time.time()

            self.stats['episode_rewards'].append(rewards)
            self.stats['episode_lengths'].append(episode_length)
            self.stats['win_rates'].append(wins / (episode + 1))
            self.stats['execution_times'].append(episode_end_time - episode_start_time)

            print(f"Episode {episode+1}, Reward: {rewards}, Episode Length: {episode_length}, Win Rate: {wins / (episode + 1):.2f}, Time: {episode_end_time - episode_start_time:.2f} seconds")

        end_time = time.time()
        print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    sarsa_agent = SARSAAgent()
    sarsa_agent.train(1000)

