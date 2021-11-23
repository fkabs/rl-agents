import numpy as np

class Bandit:
    '''
    @k_arm:             # of arms (possible actions)
    @epsilon:           probability for exploration in epsilon-greedy policies
    @linear_decay:      epsilon follows a linear decay (starting at 100%, stoping at epsilon% after 10% of episodes are played)
    @split:             set split agent (pure exploration in first 10% of episodes, greedy afterwards)
    @stationary:        type for the environment (True: stationary, False: non-stationary)
    @sample_averages:   if True, update estimates using sample averages
    @alpha:             if sample_averages is False update estimates using constant step size alpha (also used for step size in gradient bandits)
    @optimistic:        initialize estimates with a bias
    @UCB:               if not 0, upper-confidence-bound action selection (defines c for amount of exploration)
    @gradient:          if True, use gradient based bandit
    @gradient_baseline: if True, use average reward as baseline for gradient based bandit
    @true_reward:       adds to the mean of the true reward distribution
    '''
    def __init__(self, k_arms = 10, epsilon = 0, linear_decay = False, split = False,
                 stationary = True, sample_averages = True, alpha = 0,
                 optimistic = 0, UCB = 0, gradient = False, gradient_baseline = False, true_reward = 0):
        # set parameters
        self.epsilon = epsilon
        self.linear_decay = linear_decay
        self.split = split
        self.sigma = 1
        self.k_arms = k_arms
        self.stationary = stationary
        self.sample_averages = sample_averages
        self.alpha = alpha
        self.optimistic = optimistic
        self.UCB = UCB
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.true_reward = true_reward
        
        # seed and init environment
        self.seed = 42
        np.random.seed(self.seed)
        self.init_bandit()        
    
    def init_bandit(self):
        # init reward parameters
        self.average_reward = 0
        self.q_true = np.random.normal(0, self.sigma, size = self.k_arms) + self.true_reward
        self.q_estimates = np.zeros(self.k_arms) + self.optimistic
        self.q_prob = np.zeros(self.k_arms)
        self.best_action = np.argmax(self.q_true)
        
        # init agent parameters
        self.step = 0
        self.action_count = np.zeros(self.k_arms)
        self.action_indices = np.arange(self.k_arms)
    
    def act(self):
        # random action chance starts at 100% and linearly decays
        # to e% after 10% of episodes are played
        if self.linear_decay:
            if self.step < self.episodes//10:
                chance = -1 * ( (1-self.epsilon) / ((self.episodes//10)-1) )*self.step + 1
            else:
                chance = self.epsilon
 
        # random action until 10% of episodes are played
        elif self.split:
            if self.step < self.episodes//10:
                return np.random.choice(self.action_indices)
            else:
                chance = self.epsilon

        # epsilon-greedy
        else:
            chance = self.epsilon

        # random action chance% of times
        # chance is the probability of a random action, depending on a linear decay, split or epsilon-greedy policy
        if np.random.rand() < chance:
            return np.random.choice(self.action_indices)
                
        # upper-confidence-bound action selection
        if self.UCB > 0:
            UCB_estimates = self.q_estimates + self.UCB * np.sqrt(np.log(self.step + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimates)
            return np.random.choice(np.argwhere(UCB_estimates == q_best).flatten())
        
        # gradient bandit alogrithm
        if self.gradient:
            exp_estimates = np.exp(self.q_estimates)
            self.q_prob = exp_estimates / np.sum(exp_estimates)
            return np.random.choice(self.action_indices, p = self.q_prob)
        
        # default: greedy action
        q_best = np.max(self.q_estimates)
        return np.random.choice(np.argwhere(self.q_estimates == q_best).flatten())
        
    def perform_action(self, action):
        # increment action step and action counter for the corresponding action
        self.step += 1
        self.action_count[action] += 1
        
        # generate reward using gaussian (normal) distribution based on real reward
        reward = np.random.normal(self.q_true[action], 1)
        self.average_reward += (reward - self.average_reward) / self.step
        
        # update action estimates
        self.update_estimates(action, reward)
        
        # if non-stationary environment:
        # - shift underlying reward distribution
        # - update best action
        if not self.stationary:
            self.q_true += np.random.normal(0, self.sigma, size = self.k_arms)
            self.best_action = np.argmax(self.q_true)
        
        return reward

    def update_estimates(self, action, reward):        
        # update estimate using sample averages
        if self.sample_averages:
            self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_count[action]
        
        # gradient bandit estimates update
        elif self.gradient:
            one_hot = np.zeros(self.k_arms)
            one_hot[action] = 1
            
            # check for gradient baseline (baseline = average reward)
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            
            self.q_estimates += self.alpha * (reward - baseline) * (one_hot - self.q_prob)
        
        
        # update estimate using constant step size alpha
        else:
            self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
        
    def evaluate(self, epochs = 1e3, episodes = 1e3):
        # set seed and define zero rewards
        np.random.seed(self.seed)
        rewards = np.zeros(shape = (epochs, episodes))
        best_actions = np.zeros(shape = (epochs, episodes))
        
        # update class with epochs and episodes
        self.epochs = epochs
        self.episodes = episodes
        
        for epoch in range(epochs):
            self.init_bandit()
            
            for episode in range(episodes):
                action = self.act()
                rewards[epoch, episode] = self.perform_action(action)
                
                if action == self.best_action:
                    best_actions[epoch, episode] = 1
        
        return np.mean(rewards, axis = 0), np.mean(best_actions, axis = 0)
