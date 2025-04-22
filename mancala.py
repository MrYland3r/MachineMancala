import sys

PLAYER_1_PITS = ('A', 'B', 'C', 'D', 'E', 'F')
PLAYER_2_PITS = ('G', 'H', 'I', 'J', 'K', 'L')

OPPOSITE_PIT = {
    'A': 'G', 'B': 'H', 'C': 'I', 'D': 'J', 'E': 'K', 'F': 'L',
    'G': 'A', 'H': 'B', 'I': 'C', 'J': 'D', 'K': 'E', 'L': 'F'
}

NEXT_PIT = {
    'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F', 'F': '1',
    '1': 'L', 'L': 'K', 'K': 'J', 'J': 'I', 'I': 'H', 'H': 'G',
    'G': '2', '2': 'A'
}

STARTING_SEED_NUMBER = 4

def getNewBoard():
    s = STARTING_SEED_NUMBER
    return {
        '1': 0, '2': 0,
        'A': s, 'B': s, 'C': s, 'D': s, 'E': s, 'F': s,
        'G': s, 'H': s, 'I': s, 'J': s, 'K': s, 'L': s
    }

def displayBoard(board):
    seedAmounts = [str(board[p]).rjust(2) for p in 'GHIJKL21ABCDEF']
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
        pits = PLAYER_1_PITS if playerTurn == '1' else PLAYER_2_PITS
        print(f'Player {playerTurn}, choose move: {"-".join(pits)} (or QUIT)')
        response = input('> ').upper().strip()

        if response == 'QUIT':
            print('Thanks for playing :D')
            sys.exit()

        if response not in pits:
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

    # Extra turn if ending in own store
    if (playerTurn == '1' and pit == '1') or (playerTurn == '2' and pit == '2'):
        return playerTurn, board

    # Capture logic
    if playerTurn == '1' and pit in PLAYER_1_PITS and board[pit] == 1:
        oppositePit = OPPOSITE_PIT[pit]
        if board[oppositePit] > 0:
            board['1'] += board[pit] + board[oppositePit]
            board[pit] = 0
            board[oppositePit] = 0
    elif playerTurn == '2' and pit in PLAYER_2_PITS and board[pit] == 1:
        oppositePit = OPPOSITE_PIT[pit]
        if board[oppositePit] > 0:
            board['2'] += board[pit] + board[oppositePit]
            board[pit] = 0
            board[oppositePit] = 0

    next_player = '2' if playerTurn == '1' else '1'
    return next_player, board

def checkForWinner(board):
    player1Total = sum(board[p] for p in PLAYER_1_PITS)
    player2Total = sum(board[p] for p in PLAYER_2_PITS)

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
    print("Launching standalone Mancala game...")
    gameBoard = getNewBoard()
    playerTurn = '1'

    while True:
        print('\n' * 60)
        displayBoard(gameBoard)
        playerMove = askForPlayerMove(playerTurn, gameBoard)
        playerTurn, gameBoard = makeMove(gameBoard, playerTurn, playerMove)

        winner = checkForWinner(gameBoard)
        if winner in ['1', '2']:
            print('\nFinal Board:')
            displayBoard(gameBoard)
            print(f'Player {winner} has won! 🎉')
            sys.exit()
        elif winner == 'tie':
            displayBoard(gameBoard)
            print("It's a tie!")
            sys.exit()
