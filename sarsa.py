"""This is for sarsa model"""
import numpy as np 
import random

from mancala import getNewBoard, displayBoard, askForPlayerMove, makeMove, checkForWinner

class SARSAAgent:
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        return {}

    def epsilon_greedy(self, state, possible_input):
        if random.random() < self.epsilon:
            return random.choice(possible_input)
        else:
            q_values = [self.q_table.get((state, actions), 0) for action in possible_input]
            return possible_input[np.argmax(q_values)]
    
    def reset_ep(self):
        state=getNewBoard()
        action=self.epsilon_greedy(state)
        return state, action
        
    def train(self, episodes):
        for episode in range(episodes):
            state, action = self.reset_ep()
            player_turn = '1'
            done = False
            rewards = 0

            while not done:
                state = tuple(board.values())
                possible_input = PLYAER_1_PITS if player_turn == '1' else PLAYER_2_PITS

                action = self.epsilon_greedy(state, possible_input)

                next_board = makeMove(board.copy(), player_turn, action)
                next_state = tuple(next_board.values())
                reward = 0 
                winner = checkForWinner(next_board)
                if winner == '1' or winner == '2':
                    reward = 1 if winner == player_turn else -1
                    done = True
                elif winner == 'tie':
                    reward = 0 
                    done = True

                next_move = self..episolon_greedy(next_state)
                self.q_table[(state, action)] = self.q_table.get((state, action), 0) + self.alpha * (
                    reward + self.gamma * self.q_table.get((next_state, next_action), 0) - self.q_table.get((state, action), 0)
                )

                state = next_state
                action = next_action 


