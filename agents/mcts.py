from __future__ import annotations

import time
import math
import random
import numpy as np

from typing import Tuple


class Node():
    def __init__(self, state: Tuple, parent: Node, is_terminal: bool) -> None:
        """Initializes a new Node object for MCTS

        Args:
            state (Tuple): State of the TicTacToeEnv game
            parent (Node): Node object of the parent node
            is_terminal (bool): Set to True if the node is a terminal node
        """
        
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.is_fully_expanded = self.is_terminal
        self.player = state[-1]
        self.N = 0
        self.Q = 0
        self.children = {}
    
    def __str__(self) -> str:
        """Generates a string representation of the Node object

        Returns:
            str: String representation of the Node object
        """
        
        return (
            f'NODE {id(self)}\n' +
            f'state: {self.state}\n' +
            f'parent: NODE {id(self.parent)}\n' +
            f'n_children: {len(self.children)}\n' +
            f'is_terminal: {self.is_terminal}\n' +
            f'is_fully_expanded: {self.is_fully_expanded}\n' +
            f'player: {self.player}\n' +
            f'N: {self.N}\n' +
            f'Q: {self.Q}'
        )


class MCTS():
    def __init__(self, model, c_uct: float = 1/math.sqrt(2.0), player: int = 1, constraint: str = 't1') -> None:
        """Initializes a new MCTS

        Args:
            model : Model of the environment.
            c_uct (float, optional): c parameter for the UCT algorithm. Defaults to 1/math.sqrt(2.0).
            player (int, optional): Player ID. Defaults to 1.
            constraint (str, optional): Search constraint (time in seconds or n_searches) to use. Defaults to 's1'.

        Raises:
            RuntimeError: _description_
        """
        
        self.model = model
        self.c_uct = c_uct
        self.player = player
        self.root = None
        
        # set time or n_executes constraint
        if constraint[0] not in ['t', 'r'] or not constraint[1:].isdigit():
            raise RuntimeError('Invalid constraint: Please use "t#" for a time constraint in seconds or "r#" for a number of rollouts')
        else:
            self.constraint = constraint
    
    def _search(self, node: Node) -> Node:
        """Starts a MCTS search from the given node to find the best successor using the set search constraint

        Args:
            node (Node): Root node of the MCTS search tree

        Returns:
            Node: Best successor node using the UCT algorithm
        """
        if self.constraint[0] == 't':
            t_run = time.time()
            t_end = t_run + int(self.constraint[1:])
        else:
            t_run = 0
            t_end = int(self.constraint[1:])
        
        while t_run < t_end:
            self._execute(node)
            
            if self.constraint[0] == 't':
                t_run = time.time()
            else:
                t_run += 1
        
        return self._uct_select(node)
    
    def _execute(self, node: Node) -> None:
        """Executes a single MCTS search iteration (select, expand, evaluate, backup)

        Args:
            node (Node): Root node of the MCTS search tree
        """
        
        # select and/or expand node
        node = self._select(node)
        
        # evaluate node
        reward = self._evaluate(node)
        
        # backup reward
        self._backup(node, reward)
    
    def _uct_select(self, node: Node) -> Node:
        """Selects the best child node using the UCB1 (for trees) algorithm

        Args:
            node (Node): Node to select the best child from

        Returns:
            Node: Best child node
        """
        
        def _uct_value(child: Node) -> float:
            """Calculates the UCB1 value (for trees) for a node

            Args:
                child (Node): Child node

            Returns:
                float: UCB1 value for the child node
            """
            
            # a good value for MCTS means a bad value for the opponent and vice versa
            invert = True if child.parent.player != self.player else False
            
            # upper confidence bound for trees
            value = (child.Q / child.N) + self.c_uct * math.sqrt(2 * math.log(child.parent.N) / child.N)
            
            return -value if invert else value
        
        # define best value and best childs
        best_value = float('-inf')
        best_childs = []
        
        # iterate over all childs nodes
        for child in node.children.values():
            # calculate the UCT value for the child node
            child_value = _uct_value(child)
            
            # check if the child node has the best value
            if child_value > best_value:
                # update best value and best childs
                best_value = child_value
                best_childs = [child]
            elif child_value == best_value:
                # add child to best childs if the value is equal to the best value
                best_childs.append(child)
        
        # should never happen: if no best childs are found, raise an exception
        if len(best_childs) == 0:
            raise Exception(f'UCT_SELECT: No best childs found!\n{str(node)}')
        
        return random.choice(best_childs)
    
    def _select(self, node: Node) -> Node:
        """Selects a leaf node in the current tree that is worth to be evaluated
        
        Args:
            node (Node): Node to select the leaf node from
            
        Returns:
            Node: Leaf node worth to be evaluated
        """
        
        # run until leaf node is found
        while not node.is_terminal:
            if node.is_fully_expanded:
                # UCT select leaf node if fully expanded
                node = self._uct_select(node)
            else:
                # otherwise expand node
                return self._expand(node)
        
        return node
    
    def _expand(self, node: Node) -> Node:
        """Expands the tree by adding a new child node

        Args:
            node (Node): Node to expand

        Raises:
            Exception: Raised if node can not be expanded (should not happen)

        Returns:
            Node: Expanded node
        """
        
        # reset the model and set the state of the node to get the possible actions
        self.model.reset()
        self.model.set_state(node.state)
        
        # select a random action that has not been tried yet
        possible_actions = self.model.possible_actions()
        untried_actions = set(possible_actions) - set(node.children.keys())
        action = random.choice(list(untried_actions))
        
        try:
            # create new child node and add it to the parent node            
            child = Node(state = self.model.step(action)[0], parent = node, is_terminal = self.model.is_over())
            node.children[action] = child
            
            # check if node is fully expanded
            if len(possible_actions) == len(node.children):
                node.is_fully_expanded = True
            
            return child
        except:
            # Node can not be expanded (should not happen)
            raise Exception(f'EXPAND: Not able to expand node:\n{str(node)}')
    
    def _evaluate(self, node: Node) -> float:
        """Evaluate the selected node by performing a rollout

        Args:
            node (Node): Node to evaluate

        Returns:
            float: Reward of the rollout
        """
        
        def _rollout_policy(node: Node) -> int:
            """Rollout policy for the MCTS algorithm

            Args:
                node (Node): Leaf node to select action from

            Raises:
                Exception: Raised if node is a terminal node and has no possible actions (should not happen)

            Returns:
                action (int): Next action to perform
            """
            
            # reset the model and set the state of the node to get the possible actions
            self.model.reset()
            self.model.set_state(node.state)
            
            try:
                # random rollout policy (fast rollout)
                action = random.choice(self.model.possible_actions())
            except:
                # should not happen: raise an exception if a non-terminal state without possible actions is reached
                raise Exception(f'Non-terminal state has no possible actions: {self.model.current_state()}\n{str(node)}')
            
            return action
        
        # rollout until terminal node is reached    
        while not node.is_terminal:
            action = _rollout_policy(node)
            node = Node(state = self.model.step(action)[0], parent = node, is_terminal = self.model.is_over())
        
        # update the model state with the terminal node and get the reward
        self.model.reset()
        self.model.set_state(node.state)
        reward = self.model.reward()
        
        # a good reward for MCTS means a bad reward for the opponent and vice versa
        return reward if self.player == node.player else -reward
    
    def _backup(self, node: Node, reward: float) -> None:
        """Update the values that lead to the selected/evaluated node

        Args:
            node (Node): Leaf node that was selected and evaluated
            reward (float): Observed reward of the rollout
        """
        while node is not None:            
            node.N += 1
            node.Q += reward
            node = node.parent
    
    def reset(self) -> None:
        """Resets the MCTS"""
        
        self.model.reset()
        self.root = None
    
    def save(self, file: str) -> None:
        """Saves the MCTS to a file

        Args:
            file (str): Filename to save the MCTS to
        """
        
        np.savez(file, model = self.model, c_uct = self.c_uct, root = self.root)
    
    def load(self, file: str) -> None:
        """Loads the MCTS from a file
        
        Args:
            file (str): Filename to load the MCTS from
        """
        
        data = np.load(file)
        self.model = data['model'].item()
        self.c_uct = data['c_uct'].item()
        self.root = data['root'].item()
    
    def get_action(self, state: Tuple, reset: bool = True) -> int:
        """Returns the best action for the given state
        
        Args:
            state (Tuple): State to get the best action for
            reset (bool, optional): Resets the search tree before selecting the action. Defaults to True.
            
        Returns:
            int: Best action for the given state
        """
        
        def _search_root(state: Tuple) -> Node:
            """Searches an existing root node for the given state
            
            Args:
                state (Tuple): State of the TicTacToeEnv
                
            Returns:
                Node: Root node for the given state, otherwise a new root node
            """
            
            if self.root is None:
                # No root node exists, create a new one
                return Node(state = state, parent = None, is_terminal = self.model.is_over())
            else:
                # Search for the inputed state in the root
                if self.root.children:
                    child_states = [child.state for child in self.root.children.values()]
                    
                    if state in child_states:
                        # State exists in the roots children, return the corresponding root node
                        return (child for child in self.root.children.values() if child.state == state).__next__()
                    else:
                        # State does not exist in the roots children, keep searching
                        for child in self.root.children.values():
                            return _search_root(child.state)
                else:
                    # No children exist, create a new root node
                    return Node(state = state, parent = None, is_terminal = self.model.is_over())
            
            return Node(state = state, parent = None, is_terminal = self.model.is_over())      
        
        # reset the model and set the state of the node
        self.model.reset()
        self.model.set_state(state)
        
        if reset:
            # reset the MCTS
            self.root = Node(state = self.model.current_state(), parent = None, is_terminal = self.model.is_over())
        else:
            # search for the inputed state in the tree
            # TODO: For some reason, this variant performs much worse than building a new tree
            # --> Maybe the implementation is wrong; use new tree for now
            # self.root = _search_root(state)
            self.root = Node(state = self.model.current_state(), parent = None, is_terminal = self.model.is_over())
        
        # search for the best successor node
        best_node = self._search(self.root)
        
        # return the action that leads to the best node
        return (action for action, node in self.root.children.items() if node is best_node).__next__()
