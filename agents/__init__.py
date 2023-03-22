from .dynamic_programming import Random_Agent
from .monte_carlo import OnPolicy_Agent, OffPolicy_Agent
from .temporal_difference import Sarsa_Agent, ExpectedSarsa_Agent, QLearning_Agent, DoubleQLearning_Agent
from .mcts import MCTS


dp_agents = ['Random_Agent']
mc_agents = ['OnPolicy_Agent', 'OffPolicy_Agent']
td_agents = ['Sarsa_Agent', 'ExpectedSarsa_Agent', 'QLearning_Agent', 'DoubleQLearning_Agent']
mcts_agents = ['MCTS']

__all__ = dp_agents + mc_agents + td_agents + mcts_agents
