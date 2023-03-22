from .gridworld import Gridworld_Lecture, Gridworld_Book, GridWorldEnv
from .onestate import OneStateEnv
from .windygridworld import WindyGridWorldEnv
from .kingwindygridworld import KingWindyGridWorldEnv
from .stochasticwindygridworld import StochasticWindyGridWorldEnv
from .cliffwalking import CliffWalkingEnv
from .tictactoe import TicTacToeEnv


environments = [
    'Gridworld_Lecture',
    'Gridworld_Book',
    'GridWorldEnv',
    'OneStateEnv',
    'WindyGridWorldEnv',
    'KingWindyGridWorldEnv',
    'StochasticWindyGridWorldEnv',
    'CliffWalkingEnv',
    'TicTacToeEnv'
]

__all__ = environments
