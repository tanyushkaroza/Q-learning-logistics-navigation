import numpy as np
from collections import deque
from typing import List, Tuple
import time

class BFS:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = grid.shape[0]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _neighbors(self, r: int, c: int):
        for dr, dc in self.actions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def _reward(self, r: int, c: int) -> float:
        return self.grid[r, c]

    def find_path(self) -> Tuple[List[Tuple[int, int]], float, int, float]:
        start_time = time.time()

        queue = deque([(self.start, [self.start])])
        visited = {self.start}

        while queue:
            (r, c), path = queue.popleft()

            if (r, c) == self.goal:
                # Используем срез path[1:], чтобы пропустить стартовую позицию
                total_reward = sum(self._reward(pr, pc) for pr, pc in path[1:])
                elapsed_time = time.time() - start_time
                return path, total_reward, len(path) - 1, elapsed_time

            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in visited and self.grid[nr, nc] != -10:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

        elapsed_time = time.time() - start_time
        return [], -float('inf'), 0, elapsed_time

    def compare_with_rl(self, rl_path: List[Tuple[int, int]], rl_reward: float,
                        rl_steps: int, rl_time: float) -> dict:
        bfs_path, bfs_reward, bfs_steps, bfs_time = self.find_path()

        return {
            'bfs': {
                'path': bfs_path,
                'total_reward': bfs_reward,
                'steps': bfs_steps,
                'time': bfs_time
            },
            'rl': {
                'path': rl_path,
                'total_reward': rl_reward,
                'steps': rl_steps,
                'time': rl_time
            },
            'comparison': {
                'reward_diff': rl_reward - bfs_reward,
                'steps_diff': rl_steps - bfs_steps,
                'time_diff': rl_time - bfs_time
            }
        }