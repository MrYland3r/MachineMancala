import sys
import torch
import numpy as np
from mancala import (
    getNewBoard, makeMove, checkForWinner, displayBoard,
    PLAYER_1_PITS, PLAYER_2_PITS, OPPOSITE_PIT, NEXT_PIT
)

# Import Deep SARSA Player class
from deep_sarsa_agent import QNetwork, DeepSARSAPlayer

def test_deep_sarsa_consecutive_moves():
    print("\n=== TESTING DEEP SARSA AI CONSECUTIVE MOVES ===")
    
    # Set up an agent
    try:
        agent = DeepSARSAPlayer(player='2')
        print("Deep SARSA AI agent loaded.")
    except Exception as e:
        print(f"Failed to load Deep SARSA AI agent: {e}")
        return False
    
    # Setup a board 
    board = getNewBoard()
    board['H'] = 0    
    i = 1
    for pit in ['G', 'H','I', 'J', 'K', 'L']:
        board[pit] = i
        i += 1
    
    print("Initial board:")
    displayBoard(board)
    print("AI thinking about move...")
    
    # Get AI's choice
    move = agent.choose_action(board)
    print(f"AI chose move: {move}")
    
    # Execute the move
    player_turn, new_board = makeMove(board, '2', move)
    
    print("\nBoard after first move:")
    displayBoard(new_board)
    print(f"Next player: {player_turn}")
    
    # Check if AI gets another turn
    ai_got_extra_turn = (player_turn == '2')
    print(f"AI got extra turn: {ai_got_extra_turn}")
    
    if player_turn == '2':
        print("AI gets another turn, as expected!")
        
        # see if AI can make another valid move
        try:
            next_move = agent.choose_action(new_board)
            print(f"AI's second move: {next_move}")
            player_turn, final_board = makeMove(new_board, '2', next_move)
            print("\nBoard after second move:")
            displayBoard(final_board)
            print(f"Next player: {player_turn}")
            return ai_got_extra_turn
        except Exception as e:
            print(f"AI failed to make second move: {e}")
            return False
    else:
        print("AI did not get extra turn when landing in store!")
        return False

def test_deep_sarsa_capture():
    print("\n=== TESTING DEEP SARSA AI CAPTURE LOGIC ===")
    
    # Set up an agent
    try:
        agent = DeepSARSAPlayer(player='2')
        print("Deep SARSA AI agent loaded.")
    except Exception as e:
        print(f"Failed to load Deep SARSA AI agent: {e}")
        return False
    
    # Setup a board where so AI can make a capture move
    board = getNewBoard()
    board['I'] = 1     
    board['H'] = 0     
    board['B'] = 5    
    
    for pit in ['G', 'J', 'K', 'L']:
        board[pit] = 0
    
    print("Initial board (forcing AI to make a capture move):")
    displayBoard(board)
    print("AI thinking about move...")
    
    # Get AI's choice
    move = agent.choose_action(board)
    print(f"AI chose move: {move}")
    
    # Execute the move
    player_turn, new_board = makeMove(board, '2', move)
    
    print("\nBoard after move:")
    displayBoard(new_board)
    
    # Check if capture worked
    capture_worked = (new_board['2'] == 6 and new_board['H'] == 0 and new_board['B'] == 0)
    print(f"AI capture worked: {capture_worked}")
    
    return capture_worked

def test_deep_sarsa_vs_random():
    print("\n=== TESTING DEEP SARSA AI VS RANDOM PLAYER ===")
    
    # Set up an agent
    try:
        agent = DeepSARSAPlayer(player='2')
        print("Deep SARSA AI agent loaded.")
    except Exception as e:
        print(f"Failed to load Deep SARSA AI agent: {e}")
        return False
    
    # Play multiple games and record wins
    num_games = 1
    wins = 0
    draws = 0
    total_moves = 0
    
    for game in range(num_games):
        print(f"\nGame {game+1}/{num_games}")
        board = getNewBoard()
        player_turn = '1'
        moves = 0
        max_moves = 200  # Prevent infinite games
        
        while moves < max_moves:
            moves += 1
            total_moves += 1
            
            if player_turn == '1':
                # Random player chooses a random valid move
                valid_moves = [p for p in PLAYER_1_PITS if board[p] > 0]
                if not valid_moves:
                    break
                move = np.random.choice(valid_moves)
                print(f"Random player chose: {move}")
            else:
                # Deep SARSA AI chooses a move
                move = agent.choose_action(board)
                if move is None:
                    break
                print(f"Deep SARSA AI chose: {move}")
            displayBoard(board)
            player_turn, board = makeMove(board, player_turn, move)
            
            # Check for winner
            winner = checkForWinner(board)
            if winner != 'no winner':
                break
        
        print(f"Game ended after {moves} moves")
        winner = checkForWinner(board)
        
        if winner == '2':
            wins += 1
            print("Deep SARSA AI won!")
        elif winner == 'tie':
            draws += 1
            print("Game ended in a draw!")
        else:
            print("Random player won!")
        
        print("Final board:")
        displayBoard(board)
    
    win_rate = wins / num_games
    print(f"\nDeep SARSA AI won {wins}/{num_games} games ({win_rate:.0%})")
    print(f"Draws: {draws}/{num_games} ({draws/num_games:.0%})")
    print(f"Average moves per game: {total_moves/num_games:.1f}")
    
    return win_rate > 0.5  # Test passes if win rate > 50%

def run_all_tests():
    """Run all tests and report results."""
    results = {
        "Consecutive Moves": test_deep_sarsa_consecutive_moves(),
        "Capture Logic": test_deep_sarsa_capture(),
        "VS Random Player": test_deep_sarsa_vs_random()
    }
    
    print("\n=== TEST RESULTS ===")
    all_passed = True
    for test_name, passed in results.items():
        print(f"{test_name}: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nall tests passed")
    else:
        print("\nsome tests failed")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_all_tests()
