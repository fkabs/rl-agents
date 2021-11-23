# rl-agents
This repository holds implementations of Reinforcement Learning agents based on [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) [1].

## Multi-armed Bandits
Multi-armed Bandits are implemented with stationary and non-stationary environments using following action-selection methods:
- Static
- Random
- Greedy
- $\epsilon$-greedy
- Split
- Linear decay $\epsilon$-greedy
- Optimistic
- UCB
- Gradient (w/ and w/o baseline)

## Dynamic Programming
The Dynamic Programming implementation consists of an algebraic solution as well as an random agent with seperate case and in-place iterative solutions.

## Monte Carlo Methods
Following agents are already implemented:
  - First-Visit / Every-Visit
  - On-Policy / Off-Policy
  
## Temporal-Difference Learning
Following Agents are already implemented:
- Sarsa
- Expected Sarsa
- Q-Learning
- Double Q-Learning

# References
[1] R. S. Sutton and A. G. Barto, Reinforcement learning: an introduction, Second edition. Cambridge, Massachusetts: The MIT Press, 2018.
