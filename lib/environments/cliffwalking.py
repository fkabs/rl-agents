import numpy as np
import gym
from ipycanvas import Canvas, hold_canvas
from IPython.display import clear_output, display


class CliffWalkingEnv(gym.Env):

    def __init__(self, width=12, height=4, frame_width=600, frame_height=200, state_map=False):
        self.width = width
        self.height = height
        self.state_map = state_map
        if self.state_map:
            self.state_space = gym.spaces.MultiDiscrete(np.ones((self.height,self.width))*3)
        else:
            self.state_space = gym.spaces.MultiDiscrete([width, height, width, height])
        self.action_space = gym.spaces.Discrete(4) # up, down, left, right
        self.agent_position = np.random.randint((0,0), (self.height, self.width), size=2)
        self.target = np.random.randint((0,0), (self.height, self.width), size=2)
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
        state = np.zeros(shape=(self.height, self.width), dtype=int)
        state[self.target[0], self.target[1]] = 2
        state[self.agent_position[0], self.agent_position[1]] = 1
        assert self.state_space.contains(state)
        return state

    def reset(self):
        self.canvas = None
        self.agent_position = np.array([3, 0])
        self.target = np.array([3, 11])
        if self.state_map:
            return self._to_map()
        else:
            return np.concatenate((self.agent_position, self.target))

    def _reward(self):
        reward = -1
        return reward

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

        self.agent_position += add
        self.agent_position[0] = max(min(self.agent_position[0], self.height-1), 0)
        self.agent_position[1] = max(min(self.agent_position[1], self.width-1), 0)
        
        reward = self._reward()
        
        if self.agent_position[0] == 3 and self.agent_position[1] not in [0, 11]:
            self.agent_position = np.array([3, 0])
            reward = -100
            
        if self.state_map:
            state = self._to_map()
        else:
            state = np.concatenate((self.agent_position, self.target))
            
        is_done = np.array_equal(self.agent_position, self.target)

        return state, reward, is_done, None

    def render(self):
        if self.canvas is None:
            clear_output()
            self.canvas = Canvas(width=self.frame_width, height=self.frame_height)
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
            