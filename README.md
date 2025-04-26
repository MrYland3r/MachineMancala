# MachineMancala

This is a project with the purpose of introducing a model that can learn how to play Mancala over time and get better at it. The code runs in python as a .py program

## Development

In order to make this run, create a fork (for independent development), or a branch as needed. Make sure to not merge with Main until changes are correct and operational

### Prerequisites

First off, make sure you install python in your local machine // development environment. Keep in mind that as the project progresses more dependencies may be needed. Consult pertinent documentation for your OS instructions, or ask around!

Next, make sure to clone this repo locally:

```sh
git clone https://github.com/MrYland3r/MachineMancala.git
```

Afterwards, install the following python library:

```sh
pip install numpy
```
```sh
pip install torch
```
## Files
deep_sarsa_agent.py - A neural network version of SARSA (State Action Rewards State Action) algorithm. It is trained off simulations of the mancala game.

deep_sarsa_p1.pt - Contains the training data (saved weight and biases) of the deep sarsa player playing as player 1

deep_sara_p2.pt - Contains the training data (saved weight and biases) of the deep sarsa player playing as player 2

deep_sarsa_test.py - It tests the deep sarsa player against a random player.

mancala.py - The base mancala game itself (capture gamemode). It is text-based with the gameboard and inputs in the terminal.

play_against_deep_sarsa.py - It is where the user can choose to play as player 1 or 2, and they can play against the deep sarsa player. The deep sarsa player also continously learns from every game.


## Acknowledgements

Credit for the initial Mancala game code goes to **Al Sweigart al@inventwithpython.com**
