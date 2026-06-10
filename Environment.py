import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Environment:
    def __init__(self, config):
        self.__config = config
        self.__size = self.__config.GRID_SIZE
        self._goal = self.__config.GOAL_POSITION
        self._reward_grid = self.__config.REWARD_GRID
        self._start = self.__config.START_POSITION
        self._actions = self.__config.ACTIONS

        self.__agent_position = None
        self.__done = False

        self._goal_reward = 100
        self._difficulty_reward = -10

        self.reset()

    def _is_value_in_grid(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.__size and 0 <= y < self.__size

    def reset(self) -> Tuple[int, int]:
        self.__agent_position = self._start
        self.__done = False
        return self.__agent_position

    def is_goal(self, position: Tuple[int, int]) -> bool:
        return position == self._goal

    def get_reward_at_position(self, position: Tuple[int, int]) -> float:
        return self._reward_grid[position[0], position[1]]

    def step(self, action: int) -> Tuple[Tuple[int, int], int, bool]:
        if self.__done:
            raise RuntimeError("Эпизод завершён. Вызовите reset() чтобы начать сначала")
        row, col = self.__agent_position
        dr, dc = self._actions[action]
        new_row, new_col = row + dr, col + dc
        new_position = (new_row, new_col)

        if not self._is_value_in_grid(new_position):
            new_position = self.__agent_position

        reward = self.get_reward_at_position(new_position)
        self.__agent_position = new_position
        self.__done = self.is_goal(new_position)
        return self.__agent_position, reward, self.__done

    @staticmethod
    def draw_comparison_grid(ax, title, grid, start, goal, path, path_color, steps, reward):
        """
        Отрисовка одной сетки с путем.
        """
        size = grid.shape[0]

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='#bdc3c7', linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nШагов: {steps} | Награда: {reward:.1f}", fontsize=12, fontweight='bold', pad=10)

        # Отрисовка клеток
        for i in range(size):
            for j in range(size):
                val = grid[i, j]
                if val == -10:
                    color = '#2c3e50'  # Препятствие
                elif (i, j) == start:
                    color = '#27ae60'  # Старт
                elif (i, j) == goal:
                    color = '#c0392b'  # Цель
                else:
                    color = '#f5f6fa'  # Пустая клетка

                rect = patches.Rectangle((j, size - 1 - i), 1, 1, facecolor=color, edgecolor='#bdc3c7', zorder=1)
                ax.add_patch(rect)

        # Отрисовка траектории
        if path and len(path) > 0:
            x_coords = [c + 0.5 for r, c in path]
            y_coords = [size - 1 - r + 0.5 for r, c in path]
            ax.plot(x_coords, y_coords, color=path_color, linewidth=3.5,
                    linestyle='-', marker='o', markersize=6, zorder=3)

        # Маркеры Start / Goal
        ax.text(start[1] + 0.5, size - 1 - start[0] + 0.5, 'S', color='white',
                ha='center', va='center', fontweight='bold', fontsize=12, zorder=4)
        ax.text(goal[1] + 0.5, size - 1 - goal[0] + 0.5, 'G', color='white',
                ha='center', va='center', fontweight='bold', fontsize=12, zorder=4)