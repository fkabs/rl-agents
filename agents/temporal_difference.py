import numpy as np

class Sarsa_Agent():
    
    def __init__(self, state_space = None, action_space = None, epsilon = 0.1, decay = 0):
        self.state_space = state_space
        self.action_space = action_space
        self.positions = np.arange(np.prod(state_space.nvec)).reshape(tuple(state_space.nvec)[::-1])
        
        self.epsilon = epsilon if decay == 0 else 1
        self.min_epsilon = epsilon
        self.decay = decay
        
        self.is_learning = True
        
        self.reset()
    
    def reset(self):
        self.epsilon = self.min_epsilon if self.decay == 0 else 1
        state_action_shape = (np.prod(self.state_space.nvec), self.action_space.n)
        self.Q = np.zeros(state_action_shape, dtype = float)
        self.N = np.zeros(state_action_shape, dtype = float)
        self.target_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
    
    def state_to_int(self, s):
        return self.positions[tuple(s)]
    
    def int_to_state(self, s):
        return np.argwhere(self.positions == s).flatten()
    
    def learn(self, step, alpha = 0.1, gamma = 0.99):
        (s, a, r, t) = (self.state_to_int(step[0]), step[1], step[2], self.state_to_int(step[3]))
        b = self.get_action(step[3])
        
        if self.is_learning:
            self.N[s, a] += 1
            self.Q[s, a] += alpha * (r + gamma*self.Q[t, b] - self.Q[s, a])
            
            self.update_state_policy(s)
            
            if self.decay > 0:
                self.epsilon = np.max(np.array([self.epsilon-self.decay, self.min_epsilon]))
        
        return self.int_to_state(t), b
            
    def update_state_policy(self, s):
        a_idx = np.argmax(self.Q[s])
        self.target_policy[s] = self.epsilon / self.action_space.n
        self.target_policy[s, a_idx] = 1 - self.epsilon + self.epsilon / self.action_space.n
    
    def policy_improvement(self):        
        for s in range(np.prod(self.state_space.nvec)):
            if np.sum(self.Q[s]) == 0:
                self.target_policy[s] = 1 / self.action_space.n
            else:
                self.update_state_policy(s)
    
    def get_action(self, s):
        return np.random.choice(self.action_space.n, p = self.target_policy[self.state_to_int(s)])


class ExpectedSarsa_Agent():
    
    def __init__(self, state_space = None, action_space = None, epsilon = 0.1, decay = 0):
        self.state_space = state_space
        self.action_space = action_space
        self.positions = np.arange(np.prod(state_space.nvec)).reshape(tuple(state_space.nvec)[::-1])
        
        self.epsilon = epsilon if decay == 0 else 1
        self.min_epsilon = epsilon
        self.decay = decay
        
        self.is_learning = True
        
        self.reset()
    
    def reset(self):
        self.epsilon = self.min_epsilon if self.decay == 0 else 1
        state_action_shape = (np.prod(self.state_space.nvec), self.action_space.n)
        self.Q = np.zeros(state_action_shape, dtype = float)
        self.N = np.zeros(state_action_shape, dtype = float)
        self.target_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
    
    def state_to_int(self, s):
        return self.positions[tuple(s)]
    
    def int_to_state(self, s):
        return np.argwhere(self.positions == s).flatten()
    
    def learn(self, step, alpha = 0.1, gamma = 0.99):
        (s, a, r, t) = (self.state_to_int(step[0]), step[1], step[2], self.state_to_int(step[3]))
        
        if self.is_learning:
            self.N[s, a] += 1
            self.Q[s, a] += alpha * (r + gamma*np.sum(np.multiply(self.Q[t], self.target_policy[t])) - self.Q[s, a])
            
            self.update_state_policy(s)
            
            if self.decay > 0:
                self.epsilon = np.max(np.array([self.epsilon-self.decay, self.min_epsilon]))
        
        return self.int_to_state(t), self.get_action(step[3])
            
    def update_state_policy(self, s):
        a_idx = np.argmax(self.Q[s])
        self.target_policy[s] = self.epsilon / self.action_space.n
        self.target_policy[s, a_idx] = 1 - self.epsilon + self.epsilon / self.action_space.n
    
    def policy_improvement(self):        
        for s in range(np.prod(self.state_space.nvec)):
            if np.sum(self.Q[s]) == 0:
                self.target_policy[s] = 1 / self.action_space.n
            else:
                self.update_state_policy(s)
    
    def get_action(self, s):
        return np.random.choice(self.action_space.n, p = self.target_policy[self.state_to_int(s)])


class QLearning_Agent():
    
    def __init__(self, state_space = None, action_space = None, epsilon = 0.1, decay = 0):
        self.state_space = state_space
        self.action_space = action_space
        self.positions = np.arange(np.prod(state_space.nvec)).reshape(tuple(state_space.nvec)[::-1])
        
        self.epsilon = epsilon if decay == 0 else 1
        self.min_epsilon = epsilon
        self.decay = decay
        
        self.is_learning = True
        
        self.reset()
    
    def reset(self):
        self.epsilon = self.min_epsilon if self.decay == 0 else 1
        state_action_shape = (np.prod(self.state_space.nvec), self.action_space.n)
        self.Q = np.zeros(state_action_shape, dtype = float)
        self.N = np.zeros(state_action_shape, dtype = float)
        self.target_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
        self.behavior_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
    
    def state_to_int(self, s):
        return self.positions[tuple(s)]
    
    def int_to_state(self, s):
        return np.argwhere(self.positions == s).flatten()
    
    def learn(self, step, alpha = 0.1, gamma = 0.99):
        (s, a, r, t) = (self.state_to_int(step[0]), step[1], step[2], self.state_to_int(step[3]))
        
        if self.is_learning:
            self.N[s, a] += 1
            self.Q[s, a] += alpha * (r + gamma*np.max(self.Q[t, :]) - self.Q[s, a])
            
            self.update_state_policy(s)
            
            if self.decay > 0:
                self.epsilon = np.max(np.array([self.epsilon-self.decay, self.min_epsilon]))
        
        return self.int_to_state(t), self.get_action(step[3])
            
    def update_state_policy(self, s):
        a_idx = np.argmax(self.Q[s])
        
        self.behavior_policy[s] = self.epsilon / self.action_space.n
        self.behavior_policy[s, a_idx] = 1 - self.epsilon + self.epsilon / self.action_space.n
        
        self.target_policy[s] = 0
        self.target_policy[s, a_idx] = 1
    
    def policy_improvement(self):        
        for s in range(np.prod(self.state_space.nvec)):
            if np.sum(self.Q[s]) == 0:
                self.behavior_policy[s] = 1 / self.action_space.n
                self.target_policy[s] = 1 / self.action_space.n
            else:
                self.update_state_policy(s)
    
    def get_action(self, s):
        if self.is_learning:
            return np.random.choice(self.action_space.n, p = self.behavior_policy[self.state_to_int(s)])
        else:
            return np.random.choice(self.action_space.n, p = self.target_policy[self.state_to_int(s)])


class DoubleQLearning_Agent():
    
    def __init__(self, state_space = None, action_space = None, epsilon = 0.1, decay = 0, load = ''):
        self.state_space = state_space
        self.action_space = action_space
        self.positions = np.arange(np.prod(state_space.nvec)).reshape(tuple(state_space.nvec)[::-1])
        
        self.epsilon = epsilon if decay == 0 else 1
        self.min_epsilon = epsilon
        self.decay = decay
        
        self.is_learning = True
        
        self.reset()
    
    def reset(self):
        self.epsilon = self.min_epsilon if self.decay == 0 else 1
        state_action_shape = (np.prod(self.state_space.nvec), self.action_space.n)
        self.Q1 = np.zeros(state_action_shape, dtype = float)
        self.Q2 = np.zeros(state_action_shape, dtype = float)
        self.N = np.zeros(state_action_shape, dtype = float)
        self.target_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
        self.behavior_policy = np.ones(state_action_shape, dtype = float) / self.action_space.n
    
    def state_to_int(self, s):
        return self.positions[tuple(s)]
    
    def int_to_state(self, s):
        return np.argwhere(self.positions == s).flatten()
    
    def learn(self, step, alpha = 0.1, gamma = 0.99):
        (s, a, r, t) = (self.state_to_int(step[0]), step[1], step[2], self.state_to_int(step[3]))
        
        if self.is_learning:
            self.N[s, a] += 1
            
            if np.random.rand() < 0.5:
                self.Q1[s, a] += alpha * (r + gamma*self.Q2[t, np.argmax(self.Q1[t])] - self.Q1[s, a])
            else:
                self.Q2[s, a] += alpha * (r + gamma*self.Q1[t, np.argmax(self.Q2[t])] - self.Q2[s, a])
            
            self.update_state_policy(s)
            
            if self.decay > 0:
                self.epsilon = np.max(np.array([self.epsilon-self.decay, self.min_epsilon]))
        
        return self.int_to_state(t), self.get_action(step[3])
            
    def update_state_policy(self, s):
        a_idx = np.argmax(self.Q1[s] + self.Q2[s])
        
        self.behavior_policy[s] = self.epsilon / self.action_space.n
        self.behavior_policy[s, a_idx] = 1 - self.epsilon + self.epsilon / self.action_space.n
        
        self.target_policy[s] = 0
        self.target_policy[s, a_idx] = 1
    
    def policy_improvement(self):        
        for s in range(np.prod(self.state_space.nvec)):
            if np.sum(self.Q1[s] + self.Q2[s]) == 0:
                self.behavior_policy[s] = 1 / self.action_space.n
                self.target_policy[s] = 1 / self.action_space.n
            else:
                self.update_state_policy(s)
    
    def get_action(self, s):
        if self.is_learning:
            return np.random.choice(self.action_space.n, p = self.behavior_policy[self.state_to_int(s)])
        else:
            return np.random.choice(self.action_space.n, p = self.target_policy[self.state_to_int(s)])
