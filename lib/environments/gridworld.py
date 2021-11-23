import gym
import random
import itertools
import numpy as np
from ipywidgets import Image
from ipycanvas import Canvas, hold_canvas
from IPython.display import clear_output, display


class Gridworld_Lecture():
    
    def __init__(self):
        self.rows = 1
        self.columns = 5
        self.action_space = [np.array([0, -1]), np.array([0, 1])]
        self.terminal_states = [[0, 0], [0, 4]]
    
    # check if current state is a terminal state
    def is_terminal_state(self, s):
        return s in self.terminal_states
    
    # take action a at state s and return next state
    def next_state(self, s, a):
        return (np.array(s) + a).tolist()
    
    # take action a at given state s
    # return next state t and reward r
    def step(self, s, a):        
        # take action
        t = self.next_state(s, a)
        i, j = t

        # check if out of bounds
        if i < 0 or i >= self.rows or j < 0 or j >= self.columns:
            t = s
        
        # update reward depending on next state
        if self.is_terminal_state(t):
            r = 1 if t == self.terminal_states[0] else -1
        else:
            r = 0
        
        # return next state and reward
        return t, r
    
    # calculate probability of next state t with reward r
    # after current state s taking action a
    def p(self, t, r, s, a):
        prob = 0

        # landing in a terminal state only possible if
        if self.is_terminal_state(t):

            # reward is 1 and next state is left terminal state, when current state is right of it and left action was taken
            if r == 1 and s == self.next_state(t, [0, 1]) and np.array_equal(a, [0, -1]):
                prob = 1

            # reward is -1 and next state is right terminal state, when current state is left of it and right action was taken
            elif r == -1 and s == self.next_state(t, [0, -1]) and np.array_equal(a, [0, 1]):
                prob = 1

        # not landing in a terminal state        
        else:

            # current state is not a terminal state
            if not self.is_terminal_state(s):
                
                # step only possible if reward is 0
                if r == 0:
                    prob = 1
        
        return prob


class Gridworld_Book():
    
    def __init__(self):
        self.rows = 4
        self.columns = 4
        self.action_space = [np.array([0, -1]), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]
        self.terminal_states = [[0, 0], [3, 3]]
    
    # check if current state is a terminal state
    def is_terminal_state(self, s):
        return s in self.terminal_states
    
    # take action a at state s and return next state
    def next_state(self, s, a):
        return (np.array(s) + a).tolist()
    
    # take action a at given state s
    # return next state t and reward r
    def step(self, s, a):
        r = -1

        # check if current state is terminal state
        if self.is_terminal_state(s):
            r = 0
            return s, r
        
        # take action
        t = self.next_state(s, a)
        i, j = t

        # check if out of bounds
        if i < 0 or i >= self.rows or j < 0 or j >= self.columns:
            t = s
        
        # return next state and reward
        return t, r
    
    # calculate probability of next state t with reward
    # after current state s taking action a
    def p(self, t, r, s, a):
        prob = 0

        # out of bounds move, current state and next state are the same
        if s == t:
            
            # out of bounds only possible if (next) state is not terminal state and reward is -1
            if not self.is_terminal_state(s):
                if r == -1:
                    prob = 1
        
        # normal move, not out of bounds
        else:

            # only possible if reward is -1
            if r == -1:
                prob = 1
        
        return prob


class GridWorldEnv(gym.Env):

    def __init__(self, width = 5, height = 5, frame_width = 300, frame_height = 300, state_map = False):
        self.width = width
        self.height = height
        self.state_map = state_map
        
        if self.state_map:
            self.state_space = gym.spaces.MultiDiscrete(np.ones((self.height,self.width))*3)
        else:
            self.state_space = gym.spaces.MultiDiscrete([width, height, width, height])
            
        self.action_space = gym.spaces.Discrete(4) # up, down, left, right
        self.agent_position = np.random.randint((0, 0), (self.height, self.width), size = 2)
        self.target = np.random.randint((0, 0), (self.height, self.width), size = 2)
        self.state = self.state_space.sample()
        
        # only for rendering
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.horizontal_margin = 10
        self.vertical_margin = 10
        self.columns = width
        self.rows = height
        self.canvas = None

    def _to_map(self):
        state = np.zeros(shape = (self.height, self.width), dtype = int)
        state[self.target[0], self.target[1]] = 2
        state[self.agent_position[0], self.agent_position[1]] = 1
        assert self.state_space.contains(state)
        
        return state

    def reset(self):
        self.canvas = None
        
        pos = random.sample(list(itertools.product(range(5), repeat = 2)), k = 2)
        
        self.agent_position = np.array(pos[0])
        self.target = np.array(pos[1])
        
        if self.state_map:
            return self._to_map()
        else:
            return np.concatenate((self.agent_position, self.target))

    def _reward(self):
        reward = 0
        
        if np.array_equal(self.agent_position, self.target):
            reward = 10
        
        return reward
    
    def _done(self):
        done = False
        
        if np.array_equal(self.agent_position, self.target):
            done = True
        
        return done

    def step(self, action):        
        add = np.array([0, 0])
        if action == 0: # up
            add = np.array([-1, 0])
        elif action == 1: # down
            add = np.array([1, 0])
        elif action == 2: # left
            add = np.array([0, -1])
        elif action == 3: # right
            add = np.array([0, 1])
        
        old_agent_state = self.agent_position
        next_agent_state = old_agent_state + add
        
        self.agent_position = next_agent_state
        
        if self.agent_position[0] < 0 or self.agent_position[1] < 0 or self.agent_position[0] > self.height-1 or self.agent_position[1] > self.width-1:
            self.agent_position = old_agent_state
        
        if self.state_map:
            state = self._to_map()
        else:
            state = np.concatenate((self.agent_position, self.target))
        
        reward = self._reward()
        done = self._done()

        return state, reward, done, None

    def render(self):
        if self.canvas is None:
            clear_output()
            self.canvas = Canvas(width = self.frame_width, height = self.frame_height)
            display(self.canvas)
        
        cell_width = (self.frame_width - 2*self.horizontal_margin)/self.columns
        cell_height = (self.frame_height - 2*self.vertical_margin)/self.rows
        
        with hold_canvas(self.canvas):
            self.canvas.clear()
            
            for i in range(self.columns+1):
                self.canvas.stroke_line(self.horizontal_margin+(i*cell_width), self.vertical_margin, self.horizontal_margin+(i*cell_width), self.frame_height-self.vertical_margin)
            for i in range(self.rows+1):
                self.canvas.stroke_line(self.horizontal_margin, self.vertical_margin+(i*cell_height), self.frame_width-self.horizontal_margin, self.vertical_margin+(i*cell_height))
            
            self.canvas.fill_style = 'red'
            self.canvas.fill_circle(self.horizontal_margin+(cell_width*self.target[1])+(cell_width/2), self.vertical_margin+(cell_height*self.target[0])+(cell_height/2), min(cell_height, cell_width)/2)
            self.canvas.fill_style = 'black'
            self.canvas.fill_circle(self.horizontal_margin+(cell_width*self.agent_position[1])+(cell_width/2), self.vertical_margin+(cell_height*self.agent_position[0])+(cell_height/2), min(cell_height, cell_width)/2)


class GridWorldEvalEnv(gym.Env):

    def __init__(self, width = 5, height = 5, frame_width = 300, frame_height = 300, state_map = False):
        self.width = width
        self.height = height
        self.state_map = state_map
        
        if self.state_map:
            self.state_space = gym.spaces.MultiDiscrete(np.ones((self.height,self.width))*3)
        else:
            self.state_space = gym.spaces.MultiDiscrete([width, height, width, height])
            
        self.action_space = gym.spaces.Discrete(4) # up, down, left, right
        self.agent_position = np.random.randint((0, 0), (self.height, self.width), size = 2)
        self.target = np.random.randint((0, 0), (self.height, self.width), size = 2)
        self.state = self.state_space.sample()
        
        # only for rendering
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.horizontal_margin = 10
        self.vertical_margin = 10
        self.columns = width
        self.rows = height
        self.canvas = None

    def _to_map(self):
        state = np.zeros(shape = (self.height, self.width), dtype = int)
        state[self.target[0], self.target[1]] = 2
        state[self.agent_position[0], self.agent_position[1]] = 1
        assert self.state_space.contains(state)
        
        return state

    def reset(self):
        self.canvas = None
        
        pos = random.sample(list(itertools.product(range(5), repeat = 2)), k = 2)
        
        self.agent_position = np.array(pos[0])
        self.target = np.array(pos[1])
        
        if self.state_map:
            return self._to_map()
        else:
            return np.concatenate((self.agent_position, self.target))

    def _reward(self):
        reward = -1        
        return reward
    
    def _done(self):
        done = False
        
        if np.array_equal(self.agent_position, self.target):
            done = True
        
        return done

    def step(self, action):        
        add = np.array([0, 0])
        if action == 0: # up
            add = np.array([-1, 0])
        elif action == 1: # down
            add = np.array([1, 0])
        elif action == 2: # left
            add = np.array([0, -1])
        elif action == 3: # right
            add = np.array([0, 1])
        
        old_agent_state = self.agent_position
        next_agent_state = old_agent_state + add
        
        self.agent_position = next_agent_state
        
        if self.agent_position[0] < 0 or self.agent_position[1] < 0 or self.agent_position[0] > self.height-1 or self.agent_position[1] > self.width-1:
            self.agent_position = old_agent_state
        
        if self.state_map:
            state = self._to_map()
        else:
            state = np.concatenate((self.agent_position, self.target))
        
        reward = self._reward()
        done = self._done()

        return state, reward, done, None

    def render(self):
        if self.canvas is None:
            clear_output()
            self.canvas = Canvas(width = self.frame_width, height = self.frame_height)
            display(self.canvas)
        
        cell_width = (self.frame_width - 2*self.horizontal_margin)/self.columns
        cell_height = (self.frame_height - 2*self.vertical_margin)/self.rows
        
        with hold_canvas(self.canvas):
            self.canvas.clear()
            
            for i in range(self.columns+1):
                self.canvas.stroke_line(self.horizontal_margin+(i*cell_width), self.vertical_margin, self.horizontal_margin+(i*cell_width), self.frame_height-self.vertical_margin)
            for i in range(self.rows+1):
                self.canvas.stroke_line(self.horizontal_margin, self.vertical_margin+(i*cell_height), self.frame_width-self.horizontal_margin, self.vertical_margin+(i*cell_height))
            
            self.canvas.fill_style = 'red'
            self.canvas.fill_circle(self.horizontal_margin+(cell_width*self.target[1])+(cell_width/2), self.vertical_margin+(cell_height*self.target[0])+(cell_height/2), min(cell_height, cell_width)/2)
            self.canvas.fill_style = 'black'
            self.canvas.fill_circle(self.horizontal_margin+(cell_width*self.agent_position[1])+(cell_width/2), self.vertical_margin+(cell_height*self.agent_position[0])+(cell_height/2), min(cell_height, cell_width)/2)