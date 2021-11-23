import numpy as np

class OnPolicy_Agent():
    
    def __init__(self, state_space = None, action_space = None, epsilon = 0.1, decay = 0, first_visit = True):
        self.state_space = state_space
        self.action_space = action_space
        self.positions = np.arange(np.prod(state_space.nvec)).reshape(tuple(state_space.nvec)[::-1])
        
        self.epsilon = epsilon if decay == 0 else 1
        self.min_epsilon = epsilon
        self.decay = decay
        self.first_visit = first_visit
        
        self.is_learning = True
        
        self.reset()
    
    def reset(self):
        self.epsilon = self.min_epsilon if self.decay == 0 else 1
        state_action_shape = (np.prod(self.state_space.nvec), self.action_space.n)
        self.Q = np.zeros(state_action_shape, dtype = float)
        self.N = np.zeros(state_action_shape, dtype = float)
        self.target_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
        self.max_updates = []
    
    def state_to_int(self, s):
        return self.positions[tuple(s)]
    
    def learn(self, episode, gamma = 0.99):
        G = 0
        max_update = 0
        state_actions_in_episode = [(self.state_to_int(s), a) for (s, a, r, t) in episode]

        for step, (s, a, r, t) in enumerate(reversed(episode)):
            (s, a, r, t) = (self.state_to_int(s), a, r, self.state_to_int(t))
            old_estimate = self.Q[s, a]
            G = gamma*G + r

            if not self.first_visit or (self.first_visit and (s, a) not in state_actions_in_episode[:step]):
                self.N[s, a] += 1
                self.Q[s, a] += 1/self.N[s, a] * (G - self.Q[s, a])
                
                max_update = max(abs(old_estimate - self.Q[s, a]), max_update)
        
        if self.decay > 0:
            self.epsilon = np.max(np.array([self.epsilon-self.decay, self.min_epsilon]))
        
        self.max_updates.append(max_update)
        self.policy_improvement()
    
    def policy_improvement(self):        
        for s in range(np.prod(self.state_space.nvec)):
            if np.sum(self.Q[s]) == 0:
                self.target_policy[s] = 1 / self.action_space.n
            else:
                self.update_state_policy(s)
    
    def update_state_policy(self, s):
        a_idx = np.argmax(self.Q[s])
        self.target_policy[s] = self.epsilon / self.action_space.n
        self.target_policy[s, a_idx] = 1 - self.epsilon + self.epsilon / self.action_space.n
    
    def get_action(self, s):
        return np.random.choice(self.action_space.n, p = self.target_policy[self.state_to_int(s)])


class OffPolicy_Agent():
    
    def __init__(self, state_space = None, action_space = None, epsilon = 0, first_visit = False):
        self.state_space = state_space
        self.action_space = action_space
        self.positions = np.arange(np.prod(state_space.nvec)).reshape(tuple(state_space.nvec)[::-1])
        
        self.epsilon = epsilon
        self.first_visit = first_visit
        
        self.is_learning = True
        
        self.reset()
    
    def reset(self):
        state_action_shape = (np.prod(self.state_space.nvec), self.action_space.n)
        self.Q = np.zeros(state_action_shape, dtype = float)
        self.C = np.zeros(state_action_shape, dtype = float)
        self.behavior_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
        self.target_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
        self.max_updates = []
    
    def state_to_int(self, s):
        return self.positions[tuple(s)]
    
    def learn(self, episode, gamma = 0.99):
        G = 0.0
        W = 1.0
        max_update = 0
        state_actions_in_episode = [(self.state_to_int(s), a) for (s, a, r, t) in episode]

        for step, (s, a, r, t) in enumerate(reversed(episode)):
            (s, a, r, t) = (self.state_to_int(s), a, r, self.state_to_int(t))
            old_estimate = self.Q[s, a]
            G = gamma*G + r
            
            if not self.first_visit or (self.first_visit and (s, a) not in state_actions_in_episode[:step]):
                self.C[s, a] += W
                self.Q[s, a] += W/self.C[s, a] * (G - self.Q[s, a])
                
                max_update = max(abs(old_estimate - self.Q[s, a]), max_update)
                
                if a != np.argmax(self.target_policy[s]):
                    break
                
                W = W * self.target_policy[s, a]/self.behavior_policy[s, a]
        
        self.max_updates.append(max_update)
        self.policy_improvement()
    
    def policy_improvement(self):     
        for s in range(np.prod(self.state_space.nvec)):
            if np.sum(self.Q[s]) == 0:
                self.target_policy[s] = 1 / self.action_space.n
            else:
                self.update_state_policy(s)
    
    def update_state_policy(self, s):
        a_idx = np.argmax(self.Q[s])
        self.target_policy[s] = self.epsilon / self.action_space.n
        self.target_policy[s, a_idx] = 1 - self.epsilon + self.epsilon / self.action_space.n
    
    def get_action(self, s):
        if self.is_learning:
            return np.random.choice(self.action_space.n, p = self.behavior_policy[self.state_to_int(s)])
        else:
            return np.random.choice(self.action_space.n, p = self.target_policy[self.state_to_int(s)])
