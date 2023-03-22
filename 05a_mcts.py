import math
from agents.mcts import MCTS
from environments.tictactoe import TicTacToeEnv


if __name__ == "__main__":    
    env = TicTacToeEnv()
    mcts = MCTS(model = TicTacToeEnv(), c_uct = 1/math.sqrt(2.0), player = 2, constraint = 't1')
    
    state = env.reset()
    env.render()
    done = False
    
    while True:
        row_col = input('Enter your action (row,col): ')
        row, col = map(int, row_col.split(','))
        action = 3 * (row - 1) + (col - 1)
        print(f'Player action: {action}')
        
        if action not in env.possible_actions():
            raise RuntimeError(f'Invalid action: {action}')
        
        state, done, reward, info = env.step(action)
        
        if done:
            break
        
        env.render()
        
        action = mcts.get_action(state, reset = False)
        print(f'MCTS action: {action}')
        
        if action not in env.possible_actions():
            raise RuntimeError(f'Invalid action: {action}')
        
        state, done, reward, info = env.step(action)
        
        if done:
            break
        
        env.render()
    
    state, done, reward, info = env.step(1)
    env.render()
