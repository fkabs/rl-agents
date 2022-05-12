import gym
import random as rand

class OneStateEnv(gym.Env):

    def __init__(self):
        self.state_space = gym.spaces.Discrete(2) # 0 main state, 1 terminal state
        self.action_space = gym.spaces.Discrete(2)
        self.current_state = None

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action:int):
        # 0 - left, 1 - right
        next_state = 0
        reward = 0.0
        is_done = False

        if action == 1:
            next_state = 1
            is_done = True
        elif action == 0:
            if rand.random() > 0.9:
                next_state = 1
                reward = 1.0
                is_done = True
        return next_state, reward, is_done, None

    def render(self):
        pass
