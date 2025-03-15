"""This is for sarsa model"""
import numpy as np 
import random

from mancala import getNewBoard, displayBoard, askForPlayerMove, makeMove, checkForWinner

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        """not sure how to implement lol"""
        return
    def epsilon_greedy(self, state, possible_input):
        if random.random() < self.epsilon:
            return random.choice(possible_input)
        else:
            q_values = [self.q_table.get((state, actions), 0) for action in possible_input]
            return possible_input[np.argmax(q_values)]



