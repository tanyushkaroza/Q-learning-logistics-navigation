import numpy as np
from typing import List, Tuple
from Config import Config
from Environment import Environment
from AgentForTrain import AgentForTrain


class Train:
    def __init__(self, agent: AgentForTrain, environment: Environment, config: Config):
        self.__agent = agent
        self.__env = environment
        self.__config = config
        self._episodes = self.__config.EPISODES
        self._max_steps = self.__config.MAX_STEPS
        self._start_epsilon = self.__config.START_EPSILON
        self._end_epsilon = self.__config.END_EPSILON

    def _update_epsilon(self, episode: int) -> None:
        # Линейное затухание от start_epsilon до end_epsilon
        fraction = min(1.0, episode / (self._episodes * 0.7))
        new_epsilon = self._start_epsilon + fraction * (self._end_epsilon - self._start_epsilon)
        self.__agent.set_epsilon(max(self._end_epsilon, new_epsilon))

    def _train_episode(self) -> Tuple[float, int]:
        state = self.__env.reset()
        total_reward = 0
        steps = 0

        while steps < self._max_steps:
            action = self.__agent.get_action_e_greedy(state, epsilon=self.__agent.get_epsilon())
            next_state, reward, done = self.__env.step(action)

            self.__agent.update_q_value_Bellman(state, action, next_state, reward, done)

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        return total_reward, steps

    def train(self, verbose: bool = True) -> List[float]:
        if verbose:
            print("Начнём обучение агента\n" + "=" * 50)

        rewards_history = []
        self.__agent.set_epsilon(self._start_epsilon)

        for episode in range(self._episodes):
            total_reward, steps = self._train_episode()

            rewards_history.append(total_reward)
            self.__agent.add_reward_to_history(total_reward)
            self._update_epsilon(episode)

            if verbose and (episode + 1) % 100 == 0:
                last_100 = rewards_history[-100:]
                avg_reward = np.mean(last_100)
                recent_successes = sum(1 for r in last_100 if r >= self.__config.SUCCESS_THRESHOLD)

                print(f"Эпизод {episode + 1:4d}/{self._episodes} | "
                      f"Средняя награда: {avg_reward:7.2f} | "
                      f"Успешность: {recent_successes:3d}/100 | "
                      f"ε = {self.__agent.get_epsilon():.4f} | "
                      f"Шагов: {steps:3d}")

        return rewards_history