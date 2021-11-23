hard_dependencies = (
    'gym',
    'ipycanvas',
    'IPython',
    'ipywidgets',
    'itertools',
    'numpy',
    'os',
    'random',
)
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
    
del hard_dependencies, dependency, missing_dependencies


from .bandits.bandit import Bandit

from .environments.gridworld import Gridworld_Lecture, Gridworld_Book, GridWorldEnv
from .environments.windygridworld import WindyGridWorldEnv
from .environments.kingwindygridworld import KingWindyGridWorldEnv
from .environments.stochasticwindygridworld import StochasticWindyGridWorldEnv
from .environments.cliffwalking import CliffWalkingEnv

from .agents.dynamic_programming import Random_Agent
from .agents.monte_carlo import OnPolicy_Agent, OffPolicy_Agent
from .agents.temporal_difference import Sarsa_Agent, ExpectedSarsa_Agent, QLearning_Agent, DoubleQLearning_Agent


bandits = ['Bandit']

environments = [
    'Gridworld_Lecture',
    'Gridworld_Book',
    'GridWorldEnv',
    'WindyGridWorldEnv',
    'KingWindyGridWorldEnv',
    'StochasticWindyGridWorldEnv',
    'CliffWalkingEnv'
]

dp_agents = ['Random_Agent']
mc_agents = ['OnPolicy_Agent', 'OffPolicy_Agent']
td_agents = ['Sarsa_Agent', 'ExpectedSarsa_Agent', 'QLearning_Agent', 'DoubleQLearning_Agent']
agents = dp_agents + mc_agents + td_agents

__all__ = bandits + environments + agents
