import gym
import numpy as np
from typing import List, Tuple

CHARS = [" ", "x", "o"]


class TicTacToeEnv(gym.Env):

    def __init__(self):
        self.observation_space = gym.spaces.MultiDiscrete(
            (3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
        )
        self.action_space = gym.spaces.Discrete(9)
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.over = False

    def current_state(self):
        """Returns the current state of the game

        Returns:
            Tuple[int, int, int, int, int, int, int, int, int, int]: the current
                state of the game. First 9 dimensions are the board state and
                the last dimension is the current playing player (1, or 2)
        """
        return tuple(self.board.flatten().tolist() + [self.current_player])

    def possible_actions(self, show = False) -> List[int]:
        """Returns the

        Returns:
            List[int]: list of possible actions in the current state. Integers
                indicate the field on the board the player can set the piece
        """
        actions = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i, j] == 0:
                    actions.append(i*3+j)
        
        if show:
            print('ENV ACTIONS: ', actions)
        return actions

    def set_state(
        self,
        state: Tuple[int, int, int, int, int, int, int, int, int, int]
    ):
        """Sets the internal state of the environment to the one provided as a
        parameter.

        Args:
            state (tuple[int, int, int, int, int, int, int, int, int, int]):
                the state the environment shall be set to
        """
        self.current_player = state[-1]
        array = np.array(state[:-1])
        self.board = array.reshape((3, 3))

    def reset(self):
        """Reset the game"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.over = False
        return self.current_state()

    def who_has_won(self):
        """Determines whether the game is over or still contuing

        -1 game not yet over
        0 no player won
        1 player 1 won
        2 player 2 won

        Returns:
            int: id of the player who won, or 0 if still ongoing
        """
        for i in range(len(self.board)):
            if self.board[i, 0] > 0 and self.board[i, 0] == self.board[i, 1] == self.board[i, 2]:
                return self.board[i, 0]
            if self.board[0, i] > 0 and self.board[0, i] == self.board[1, i] == self.board[2, i]:
                return self.board[0, i]
        if self.board[0, 0] > 0 and self.board[0, 0] == self.board[1, 1] == self.board[2, 2]:
            return self.board[0, 0]
        if self.board[0, 2] > 0 and self.board[0, 2] == self.board[1, 1] == self.board[2, 0]:
            return self.board[0, 2]
        if not (self.board > 0).all():
            return -1
        return 0

    def is_over(self) -> bool:
        return self.who_has_won() > -1

    def reward(self) -> float:
        player_won = self.who_has_won()
        if player_won == 0 or player_won == -1:
            return 0.0
        if player_won == self.current_player:
            return 1.0
        return -1.0

    def step(
        self,
        field_id: int
    ) -> Tuple[Tuple[int, int, int, int, int, int, int, int, int, int], bool, float, str]:
        """Sets the marker of the current player to the specified field

        Args:
            field_id (int): id of the field of which the marker should be set

        Returns:
            Tuple[Tuple[int, int, int, int, int, int, int, int, int, int], bool, float, str]:
                state, done, reward, info
        """

        if not self.over:
            row_idx = int(field_id / 3)
            column_idx = field_id % 3
            if self.board[row_idx, column_idx] != 0:
                raise ValueError("Invalid Parameter")
            self.board[row_idx, column_idx] = self.current_player
        reward = self.reward()
        self.current_player = 1 if self.current_player == 2 else 2
        if not self.over:
            self.over = self.is_over()

        return self.current_state(), self.over, reward, ""

    def render(self):
        print(self.current_state())
        for i in range(len(self.board)):
            print("-"*13)
            for j in range(len(self.board[i])):
                print(f"| {CHARS[self.board[i, j]]} ", end="")
            print("|")
        print("-"*13)
        
        if self.is_over():
            if self.who_has_won() == 0:
                print('Draw!')
            else:
                print(f'Player {self.who_has_won()} won!')
            print("-"*13)
        
        print('')
