import numpy as np
from typing import List, Tuple
from Config import Config
from Environment import Environment
from AgentForTest import AgentForTest

class Test:
    def __init__(self, agent: AgentForTest, environment: Environment, config: Config):
        self._agent = agent
        self._env = environment
        self._config = config
        self._test_episodes = config.TEST_EPISODES
        self._success_threshold = config.SUCCESS_THRESHOLD
        self._max_steps = config.MAX_STEPS

    def test_episode(self) -> Tuple[float, int]:
        state = self._env.reset()
        total_reward = 0
        steps = 0

        while not self._env.is_goal(state) and steps < self._max_steps:
            action = self._agent.get_best_action(state)
            state, reward, _ = self._env.step(action)
            total_reward += reward
            steps += 1

        return total_reward, steps

    def _calculate_discounted_rewards(self, rewards: List[float]) -> List[float]:
        gamma = self._config.GAMMA
        return [rewards[t] * (gamma ** t) for t in range(len(rewards))]

    def _find_optimal_path(self) -> List[Tuple[int, int]]:
        path = [self._config.START_POSITION]
        row, col = self._config.START_POSITION
        visited = {self._config.START_POSITION}
        actions = self._config.ACTIONS

        while (row, col) != self._config.GOAL_POSITION:
            action = self._agent.get_best_action((row, col))
            dr, dc = actions[action]
            new_row, new_col = row + dr, col + dc

            if not (0 <= new_row < self._config.GRID_SIZE and
                    0 <= new_col < self._config.GRID_SIZE):
                break
            if (new_row, new_col) in visited:
                break

            path.append((new_row, new_col))
            visited.add((new_row, new_col))
            row, col = new_row, new_col

        return path

    def test(self, verbose: bool = True) -> Tuple[List[float], List[int], int, float]:
        if verbose:
            print("Тестирование агента\n" + "=" * 50)

        rewards = []
        steps_list = []
        successes = 0

        for _ in range(self._test_episodes):
            reward, steps = self.test_episode()
            rewards.append(reward)
            steps_list.append(steps)
            if reward >= self._success_threshold:
                successes += 1

        discounted = self._calculate_discounted_rewards(rewards)
        avg_discounted = np.mean(discounted)

        if verbose:
            print(f"Средняя награда: {np.mean(rewards):.2f}")
            print(f"Средняя дисконтированная награда: {avg_discounted:.2f}")
            print(f"Среднее число шагов: {np.mean(steps_list):.2f}")
            print(f"Успешно: {successes}/{self._test_episodes} ({100 * successes / self._test_episodes:.1f}%)")
            path = self._find_optimal_path()
            print(f"Оптимальный путь: {path}")
            print(f"Длина пути: {len(path) - 1} шагов")

        return rewards, steps_list, successes, avg_discounted