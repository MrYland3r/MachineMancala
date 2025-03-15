"""This is for q_learning model"""

import numpy as np

from mancala import getNewBoard, displayBoard, askForPlayerMove, makeMove, checkForWinner


"""
Q-Learning Agent for Mancala using TensorFlow/Keras
This file uses the model defined in qnetwork_tf.py.
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from qagent import create_q_model  # Import the model creation function

# Import your game functions/constants
from mancala import getNewBoard, displayBoard, makeMove, checkForWinner, PLAYER_1_PITS, PLAYER_2_PITS

def get_observation(board, playerTurn):
    """
    Converts the board state into a 6-element numpy array for the current player.
    """
    if playerTurn == '1':
        return np.array([board[p] for p in PLAYER_1_PITS])
    else:
        return np.array([board[p] for p in PLAYER_2_PITS])

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.1, model_path=None):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration probability
        if model_path is None:
            self.model = create_q_model()  # Create a new model using our external function
        else:
            self.model = load_model(model_path)
    
    def epsilon_greedy(self, state, possible_actions, playerTurn):
        """
        Chooses an action based on an epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            if playerTurn == '1':
                indices = [PLAYER_1_PITS.index(a) for a in possible_actions]
            else:
                indices = [PLAYER_2_PITS.index(a) for a in possible_actions]
            valid_moves = [(a, q_values[i]) for i, a in zip(indices, possible_actions) if state[i] > 0]
            return max(valid_moves, key=lambda x: x[1])[0] if valid_moves else random.choice(possible_actions)
    
    def train(self, episodes=1000):
        """
        Trains the agent over a specified number of episodes.
        """
        for episode in range(episodes):
            board = getNewBoard()
            player_turn = '1'
            state = get_observation(board, player_turn)
            total_reward = 0
            done = False

            while not done:
                possible_actions = PLAYER_1_PITS if player_turn == '1' else PLAYER_2_PITS
                action = self.epsilon_greedy(state, possible_actions, player_turn)
                
                next_player_turn, next_board = makeMove(board.copy(), player_turn, action)
                reward = 0
                winner = checkForWinner(next_board)
                if winner == player_turn:
                    reward = 1
                    done = True
                elif winner != 'no winner':
                    reward = -1
                    done = True

                next_state = get_observation(next_board, next_player_turn)
                target = reward
                if not done:
                    next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                    target += self.gamma * np.max(next_q_values)
                
                # Get current Q-values and update the chosen action's Q-value.
                q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                action_index = PLAYER_1_PITS.index(action) if player_turn == '1' else PLAYER_2_PITS.index(action)
                q_values[action_index] += self.alpha * (target - q_values[action_index])
                
                # Train the model on the updated Q-values for the current state.
                self.model.fit(state.reshape(1, -1), q_values.reshape(1, -1), epochs=1, verbose=0)
                
                board = next_board
                state = next_state
                player_turn = next_player_turn
                total_reward += reward

            print(f"Episode {episode+1}, Total Reward: {total_reward}")

if __name__ == '__main__':
    agent = QLearningAgent()
    agent.train(episodes=1000)
