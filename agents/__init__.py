from .dynamic_programming import Random_Agent
from .monte_carlo import OnPolicy_Agent, OffPolicy_Agent
from .temporal_difference import Sarsa_Agent, ExpectedSarsa_Agent, QLearning_Agent, DoubleQLearning_Agent


dp_agents = ['Random_Agent']
mc_agents = ['OnPolicy_Agent', 'OffPolicy_Agent']
td_agents = ['Sarsa_Agent', 'ExpectedSarsa_Agent', 'QLearning_Agent', 'DoubleQLearning_Agent']

__all__ = dp_agents + mc_agents + td_agents