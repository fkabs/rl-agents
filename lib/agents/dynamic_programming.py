class Random_Agent():
    
    def __init__(self, action_space):
        self.action_space = action_space
        
    # random policy
    def policy(self):
        return 1/len(self.action_space)
