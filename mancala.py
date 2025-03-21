"""
Mancala, by Al Sweigart al@inventwithpython.com
The ancient seed-sowing game.
This code is available at https://nostarch.com/big-book-small-python-programming
Tags: large, board game, game, two-player

Modifications may be made for the purpose of this non-profit project to experiment with machine learning in mancala
"""

import sys

PLAYER_1_PITS = ('A', 'B', 'C', 'D', 'E', 'F')
PLAYER_2_PITS = ('G', 'H', 'I', 'J', 'K', 'L')

OPPOSITE_PIT = {'A': 'G', 'B': 'H', 'C': 'I', 'D': 'J', 
                'E': 'K', 'F': 'L', 'G': 'A', 'H': 'B',
                'I': 'C', 'J': 'D', 'K': 'E', 'L': 'F'}

NEXT_PIT = {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 
            'E': 'F', 'F': '1', '1': 'L', 'L': 'K',
            'K': 'J', 'J': 'I', 'I': 'H', 'H': 'G',
            'G': '2', '2': 'A'}

PIT_LABELS = 'ABCDEF1LKJIHG2'

STARTING_SEED_NUMBER = 4

def main():
    print('''Welcome to Mancala! Original code by Al Sweigart and adapted for project by MachineMancala development team.''')
    input('''Press ENTER to begin! ''')
    gameBoard = getNewBoard()
    playerTurn = '1'

    while True:
        print('\n' * 60)
        displayBoard(gameBoard)
        playerMove = askForPlayerMove(playerTurn,
                                      gameBoard)

        playerTurn = makeMove(gameBoard, playerTurn,
                              playerMove)

        winner = checkForWinner(gameBoard)
        if winner == '1' or winner == '2':
            #displayBoard(gameBoard)
            print('Player ' + winner + ' has won! Congratulations')
            sys.exit()
        elif winner == 'tie':
            displayBoard(gameBoard)
            print('There is a tie :0')
            sys.exit()

def getNewBoard():
    s = STARTING_SEED_NUMBER

    return {'1': 0, '2': 0, 'A': s, 'B': s, 'C': s, 'D': s,
            'E': s, 'F': s, 'G': s, 'H': s, 'I': s, 'J': s,
            'K': s, 'L': s}

def displayBoard(board):
    seedAmounts = []

    for pit in 'GHIJKL21ABCDEF':
        numSeedsInThisPit = str(board[pit]).rjust(2)
        seedAmounts.append(numSeedsInThisPit)

    print("""
+------+------+--<<<<<-Player 2----+------+------+------+
2      |G     |H     |I     |J     |K     |L     |      1 
       |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |       
S      |      |      |      |      |      |      |      S 
T  {}  +------+------+------+------+------+------+  {}  T 
O      |A     |B     |C     |D     |E     |F     |      O 
R      |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |      R 
E      |      |      |      |      |      |      |      E 
+------+------+------+-Player 1>>>>>------+------+------+
    """.format(*seedAmounts))

def askForPlayerMove(playerTurn, board):
    while True:
        if playerTurn == '1':
            print('Player 1, choose move: A-F (or QUIT)')
        elif playerTurn == '2':
            print('Player 2, choose move: G-L (or QUIT)')
        response = input('> ').upper().strip()

        if response == 'QUIT':
            print('Thanks for playing :D')
            sys.exit()

        if (playerTurn == '1' and response not in PLAYER_1_PITS) or (playerTurn == '2' and response not in PLAYER_2_PITS):
            print('Please pick a letter on your side of the board!')
            continue
        if board.get(response) == 0:
            print('Please pick a non-empty pit.')
            continue
        return response

def makeMove(board, playerTurn, pit):
    seedsToSow = board[pit]
    board[pit] = 0

    while seedsToSow > 0:
        pit = NEXT_PIT[pit]
        if (playerTurn == '1' and pit == '2') or (playerTurn == '2' and pit == '1'):
            continue
        board[pit] += 1 
        seedsToSow -= 1 

    if (pit == playerTurn == '1') or (pit == playerTurn == '2'):
        return playerTurn

    if playerTurn == '1' and pit in PLAYER_1_PITS and board[pit] == 1:
        oppositePit = OPPOSITE_PIT[pit]
        board['1'] += board[oppositePit]
        board[oppositePit] = 0
    elif playerTurn == '2' and pit in PLAYER_2_PITS and board[pit] == 1:
        oppositePit = OPPOSITE_PIT[pit]
        board['2'] += board[oppositePit]
        board[oppositePit] = 0

    if playerTurn == '1':
        return '2'
    elif playerTurn == '2':
        return '1'

def checkForWinner(board):
    player1Total = board['A'] + board['B'] + board['C']
    player1Total += board['D'] + board['E'] + board['F']
    player2Total = board['G'] + board['H'] + board['I']
    player2Total += board['J'] + board['K'] + board['L']

    if player1Total == 0:
        board['2'] += player2Total
        for pit in PLAYER_2_PITS:
            board[pit] = 0
    elif player2Total == 0:
        board['1'] += player1Total
        for pit in PLAYER_1_PITS:
            board[pit] = 0
    else:
        return 'no winner'

    if board['1'] > board['2']:
        return '1'
    elif board['1'] < board['2']:
        return '2'
    else:
        return 'tie'

if __name__ == '__main__':
    main()
    
