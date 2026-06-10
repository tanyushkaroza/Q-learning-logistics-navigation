import numpy as np
import random
from typing import Tuple
from Config import Config

class AgentForTrain:
    def __init__(self, config: Config = None):
        self.__config = config
        self._grid_size = self.__config.GRID_SIZE
        self._n_actions = self.__config.N_ACTIONS
        self._gamma = self.__config.GAMMA
        self._alpha = 0.1
        self._q_table = np.zeros((self._grid_size, self._grid_size, self._n_actions))
        self._epsilon = None
        self.__rewards_history = []

    def get_action_e_greedy(self, state: Tuple[int, int], epsilon: float) -> int:
       # state: текущее состояние (row, col)
       # epsilon: вероятность случайного выбора
       # return индекс выбранного действия
        row, col = state
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self._n_actions - 1)
        else:
            return int(np.argmax(self._q_table[row, col]))


    def update_q_value_Bellman(self, state: Tuple[int, int], action: int,
                           next_state: Tuple[int, int], reward: float, done: bool = False) -> None:
        # state: текущее состояние
        # action: выполненное действие
        # next_state: следующее состояние
        # reward: полученная награда

        row, col = state
        next_row, next_col = next_state

        # Если мы достигли цели, будущей награды нет
        if done:
            target = reward
        else:
            next_max = np.max(self._q_table[next_row, next_col])
            target = reward + self._gamma * next_max

        self._q_table[row, col, action] += self._alpha * (target - self._q_table[row, col, action])

    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        row, col = state
        return self._q_table[row, col, action]

    def get_best_action(self, state: Tuple[int, int]) -> int:
        row, col = state
        return int(np.argmax(self._q_table[row, col]))

    def get_max_q(self, state: Tuple[int, int]) -> float:
        row, col = state
        return np.max(self._q_table[row, col])

    def get_best_action_and_q(self, state: Tuple[int, int]) -> Tuple[float, int]:
        return self.get_max_q(state), self.get_best_action(state)

    def set_epsilon(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def get_epsilon(self) -> float:
        return self._epsilon

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha

    def get_stats(self) -> dict:
        return {
            'q_table_shape': self._q_table.shape,
            'mean_q': np.mean(self._q_table),
            'max_q': np.max(self._q_table),
            'min_q': np.min(self._q_table)
        }

    def add_reward_to_history(self, episode_reward: float) -> None:
        self.__rewards_history.append(episode_reward)

    def get_rewards_history(self) -> list[float]:
        return self.__rewards_history.copy()

    def is_empty(self) -> bool:
        return len(self.__rewards_history) == 0

    def get_best_reward(self) -> float:
        if self.is_empty():
            return float('-inf')
        else:
            return max(self.__rewards_history)

    def get_discounted_reward(self) -> float:
        discounted_sum = 0
        gamma = self._gamma
        if self.is_empty():
            return -float('inf')
        else:
            for t, reward in enumerate(self.__rewards_history):
                discounted_sum += reward * (gamma ** t)
        return discounted_sum

    def calculate_episode_discounted_reward(self, rewards: list[float]) -> float:
        discounted_sum = 0
        for t, reward in enumerate(rewards):
            discounted_sum += reward * (self._gamma ** t)
        return discounted_sum

    def clear_rewards_history(self) -> None:
        self.__rewards_history.clear()

    def load_q_table(self, filepath: str) -> None:
        loaded_2d = np.loadtxt(filepath, delimiter=',')
        self._q_table = loaded_2d.reshape((self._grid_size, self._grid_size, self._n_actions))

    def save_q_table(self, filepath: str) -> None:
        table_2d = self._q_table.reshape((self._grid_size * self._grid_size, self._n_actions))
        np.savetxt(filepath, table_2d, delimiter=',', fmt='%f')