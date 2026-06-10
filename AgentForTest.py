import numpy as np
from typing import Tuple
from Config import Config

class AgentForTest:
    def __init__(self, config: Config = None):
        self.__config = config
        self._grid_size = self.__config.GRID_SIZE
        self._n_actions = self.__config.N_ACTIONS
        self._q_table = None
        self._reward_grid = config.REWARD_GRID if config else None

    def set_q_table(self, q_table: np.ndarray) -> None:
        self._q_table = q_table

    def set_reward_grid(self, reward_grid: np.ndarray) -> None:
        """Устанавливает карту наград для проверки препятствий"""
        self._reward_grid = reward_grid

    def load_q_table(self, filepath: str) -> None:
        loaded_2d = np.loadtxt(filepath, delimiter=',')
        self._q_table = loaded_2d.reshape((self._grid_size, self._grid_size, self._n_actions))

    def get_best_action(self, state: Tuple[int, int]) -> int:
        """Возвращает лучшее действие, избегая препятствий"""
        row, col = state
        q_values = self._q_table[row, col].copy()

        # Проверяем все возможные действия
        for action_idx, (dr, dc) in enumerate(self.__config.ACTIONS):
            nr, nc = row + dr, col + dc

            # Проверяем выход за границы
            if not (0 <= nr < self._grid_size and 0 <= nc < self._grid_size):
                q_values[action_idx] = -np.inf
                continue

            # Проверяем, не ведет ли действие на препятствие
            if self._reward_grid is not None and self._reward_grid[nr, nc] == -10:
                q_values[action_idx] = -np.inf

        # Если все действия ведут в препятствия или за границы, выбираем первое доступное
        if np.all(q_values == -np.inf):
            # Ищем первое безопасное действие
            for action_idx, (dr, dc) in enumerate(self.__config.ACTIONS):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self._grid_size and 0 <= nc < self._grid_size:
                    if self._reward_grid is None or self._reward_grid[nr, nc] != -10:
                        return action_idx
            return 0  # Если всё плохо, возвращаем 0

        return int(np.argmax(q_values))

    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        row, col = state
        return self._q_table[row, col, action]

    def get_max_q(self, state: Tuple[int, int]) -> float:
        row, col = state
        return np.max(self._q_table[row, col])

    def get_best_action_and_q(self, state: Tuple[int, int]) -> Tuple[float, int]:
        return self.get_max_q(state), self.get_best_action(state)

    def get_stats(self) -> dict:
        return {
            'q_table_shape': self._q_table.shape,
            'mean_q': np.mean(self._q_table),
            'max_q': np.max(self._q_table),
            'min_q': np.min(self._q_table)
        }
