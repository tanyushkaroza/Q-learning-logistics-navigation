import numpy as np
from typing import Optional

class Config:
    GRID_SIZE = 5
    START_POSITION = (4, 0)
    GOAL_POSITION = (0, 4)
    DEFAULT_REWARD_GRID = np.array([
        [-1, -1, -1, -1, 100],
        [-1, -10, -10, -1, -1],
        [-1, -1, -1, -10, -1],
        [-10, -10, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ])
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ['up', 'down', 'left', 'right']
    N_ACTIONS = 4
    GAMMA = 0.9
    START_EPSILON = 0.9
    END_EPSILON = 0.01
    EPISODES = 5000
    TEST_EPISODES = 300
    SUCCESS_THRESHOLD = 80
    MAX_STEPS = 1000

    def __init__(self, reward_grid: Optional[np.ndarray] = None):
        if reward_grid is not None:
            self.REWARD_GRID = reward_grid
            self.GRID_SIZE = reward_grid.shape[0]
            self.START_POSITION = (self.GRID_SIZE - 1, 0)
            self.GOAL_POSITION = (0, self.GRID_SIZE - 1)
        else:
            self.REWARD_GRID = self.DEFAULT_REWARD_GRID
    

