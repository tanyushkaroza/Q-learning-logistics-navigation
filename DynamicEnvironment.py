import numpy as np
import random
import time
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


class DynamicEnvironment:
    """
    Динамическая среда с движущимися препятствиями, использующая
    технологию Local Vision (маскирование опасных действий агента на лету)
    и предотвращение столкновений препятствий.
    """

    def __init__(self, config, agent_test=None):
        self.config = config
        self.agent = agent_test  # Сюда передается AgentForTest с обученной 2D Q-таблицей
        self.size = config.GRID_SIZE
        self.start = config.START_POSITION
        self.goal = config.GOAL_POSITION
        self.obstacle_speed = 1

        # --- СОХРАНЕНИЕ ЭТАЛОННОГО СОСТОЯНИЯ ---
        self.initial_grid = self._generate_initial_grid()
        self.grid = self.initial_grid.copy()
        self.initial_obstacles = self._obstacle_positions()

        # Генерируем векторы направлений строго один раз для чистоты эксперимента
        self.initial_directions = {}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for pos in self.initial_obstacles:
            self.initial_directions[pos] = random.choice(directions)

        # Переменные текущей сессии
        self.obstacles = []
        self.obstacle_directions = {}

    def _reset_environment(self) -> None:
        """Восстанавливает абсолютно идентичные стартовые условия для каждого алгоритма"""
        self.grid = self.initial_grid.copy()
        self.obstacles = self.initial_obstacles.copy()
        self.obstacle_directions = self.initial_directions.copy()

    def _generate_initial_grid(self) -> np.ndarray:
        if hasattr(self.config, 'REWARD_GRID'):
            return self.config.REWARD_GRID.copy()
        return self.config.DEFAULT_REWARD_GRID.copy()

    def _obstacle_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == -10:
                    positions.append((i, j))
        return positions

    def _move_obstacles(self) -> None:
        """Перемещает препятствия, не допуская их наложения друг на друга."""
        new_grid = np.full((self.size, self.size), -1.0)
        new_grid[self.goal[0], self.goal[1]] = 100.0

        next_obstacles = []
        new_directions = {}
        claimed_positions = set()  # Клетки, которые уже заняты на этот ход

        for r, c in self.obstacles:
            dr, dc = self.obstacle_directions.get((r, c), (0, 1))
            nr, nc = r + dr, c + dc

            # 1. Отскок от границ сетки, старта или финиша
            if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) == self.start or (nr, nc) == self.goal:
                dr, dc = -dr, -dc  # Разворот направления
                nr, nc = r + dr, c + dc

                # Если после разворота все равно уперлись (например, зажаты в углу)
                if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) == self.start or (nr, nc) == self.goal:
                    nr, nc = r, c

            # 2. Проверка на столкновение с ДРУГИМ препятствием
            if (nr, nc) in claimed_positions:
                # Если клетка уже кем-то занята на этот ход, отменяем перемещение
                # и разворачиваем вектор, чтобы на следующем шаге поехать обратно
                nr, nc = r, c
                dr, dc = -dr, -dc

                # Защита от экстремально плотных карт, где даже исходная клетка (r,c)
                # уже может быть в claimed_positions
                while (nr, nc) in claimed_positions:
                    break

                    # Бронируем клетку и сохраняем новые координаты
            claimed_positions.add((nr, nc))
            next_obstacles.append((nr, nc))
            new_directions[(nr, nc)] = (dr, dc)

        # Обновляем состояние среды
        self.obstacles = next_obstacles
        self.obstacle_directions = new_directions

        for r, c in self.obstacles:
            new_grid[r, c] = -10.0

        self.grid = new_grid

    def action_with_local_vision(self, position: Tuple[int, int], recent_path: List[Tuple[int, int]]) -> int:
        """
        Умное локальное зрение с краткосрочной памятью для предотвращения зацикливания.
        """
        row, col = position

        if self.agent is None or self.agent._q_table is None:
            q_values = np.zeros(self.config.N_ACTIONS)
        else:
            q_values = self.agent._q_table[row, col].copy()

        for action_idx, (dr, dc) in enumerate(self.config.ACTIONS):
            nr, nc = row + dr, col + dc
            next_pos = (nr, nc)

            # 1. Жесткая блокировка: выход за карту или столкновение
            if not (0 <= nr < self.size and 0 <= nc < self.size) or self.grid[nr, nc] == -10:
                q_values[action_idx] = -np.inf
                continue

            # 2. Штраф за зацикливание
            # Наказываем возвращение назад, чтобы заставить агента искать обход
            if next_pos in recent_path[-4:]:
                q_values[action_idx] -= 50.0

                # Если все пути заблокированы, возвращаем -1 (сигнал к ожиданию)
        if np.max(q_values) == -np.inf:
            return -1

        return int(np.argmax(q_values))

    def simulate_rl(self) -> Tuple[List[Tuple[int, int]], List[np.ndarray], float, int, float]:
        """Симуляция RL агента с использованием локального зрения и псевдо-ожидания"""
        self._reset_environment()

        path = [self.start]
        recent_path = [self.start]  # Память для предотвращения циклов
        grid_history = [self.grid.copy()]
        current_pos = self.start
        total_reward = 0
        steps = 0

        start_time = time.time()

        while current_pos != self.goal and steps < self.config.MAX_STEPS:
            self._move_obstacles()

            action = self.action_with_local_vision(current_pos, recent_path)

            # Если вернулся -1 (зажали), стоим на месте
            if action == -1:
                next_pos = current_pos
            else:
                dr, dc = self.config.ACTIONS[action]
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)

            # Дополнительная проверка безопасности шага
            if 0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size:
                if self.grid[next_pos[0], next_pos[1]] != -10:
                    current_pos = next_pos

            reward = self.grid[current_pos[0], current_pos[1]]
            total_reward += reward
            steps += 1

            path.append(current_pos)
            recent_path.append(current_pos)

            # Ограничиваем историю 10 шагами
            if len(recent_path) > 10:
                recent_path.pop(0)

            grid_history.append(self.grid.copy())

        elapsed_time = time.time() - start_time
        return path, grid_history, total_reward, steps, elapsed_time

    def simulate_bfs(self) -> Tuple[List[Tuple[int, int]], List[np.ndarray], float, int, float]:
        """Симуляция BFS (динамический перерасчет графа на каждом шаге)"""
        self._reset_environment()

        path = [self.start]
        grid_history = [self.grid.copy()]
        current_pos = self.start
        total_reward = 0
        steps = 0
        total_calc_time = 0

        while current_pos != self.goal and steps < self.config.MAX_STEPS:
            self._move_obstacles()

            bfs_start = time.time()
            from BFS import BFS
            bfs_planner = BFS(self.grid, current_pos, self.goal)
            bfs_path, _, _, _ = bfs_planner.find_path()
            total_calc_time += (time.time() - bfs_start)

            if len(bfs_path) > 1:
                next_pos = bfs_path[1]
                if self.grid[next_pos[0], next_pos[1]] != -10:
                    current_pos = next_pos

            reward = self.grid[current_pos[0], current_pos[1]]
            total_reward += reward
            steps += 1
            path.append(current_pos)
            grid_history.append(self.grid.copy())

        return path, grid_history, total_reward, steps, total_calc_time

    def generate_alternative_map(self) -> np.ndarray:
        """Генерирует альтернативную карту с другим расположением препятствий"""
        alt_grid = np.full((self.size, self.size), -1.0)
        alt_grid[self.goal[0], self.goal[1]] = 100.0

        alternative_obstacles = [
            (1, 1), (1, 2), (2, 1),
            (3, 3), (3, 4), (4, 3),
            (2, 4)
        ]

        for r, c in alternative_obstacles:
            if 0 <= r < self.size and 0 <= c < self.size:
                if (r, c) != self.start and (r, c) != self.goal:
                    alt_grid[r, c] = -10.0

        return alt_grid

    def visualize_dynamic_comparison(self, output_file: str = "dynamic_comparison.gif",
                                     show_alternative: bool = True) -> None:
        """Проводит симуляции, выводит метрики и генерирует side-by-side GIF анимацию"""

        rl_path, rl_grids, rl_reward, rl_steps, rl_time = self.simulate_rl()
        bfs_path, bfs_grids, bfs_reward, bfs_steps, bfs_time = self.simulate_bfs()

        print("\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ В ДИНАМИЧЕСКОЙ СРЕДЕ (ОСНОВНАЯ КАРТА):")
        print("=" * 55)
        print(f" RL Агент со зрением (Local Vision):")
        print(f"   - Сделано шагов:      {rl_steps}")
        print(f"   - Время принятия решений: {rl_time:.6f} сек")
        print(f"   - Суммарная награда:  {rl_reward:.1f}")
        print(f"\n Алгоритм BFS (Пошаговый перерасчет):")
        print(f"   - Сделано шагов:      {bfs_steps}")
        print(f"   - Время принятия решений: {bfs_time:.6f} сек")
        print(f"   - Суммарная награда:  {bfs_reward:.1f}")
        print("=" * 55)

        fig, (ax_rl, ax_bfs) = plt.subplots(1, 2, figsize=(14, 7))
        max_frames = max(len(rl_path), len(bfs_path))

        def update(frame):
            ax_rl.clear()
            ax_bfs.clear()

            ax_rl.set_title(f"RL Agent (Local Vision)\nStep: {min(frame, len(rl_path) - 1)}", fontsize=12,
                            fontweight='bold')
            ax_bfs.set_title(f"BFS (Dynamic Replanning)\nStep: {min(frame, len(bfs_path) - 1)}", fontsize=12,
                             fontweight='bold')

            rl_idx = min(frame, len(rl_path) - 1)
            self._draw_grid_to_axis(ax_rl, rl_grids[rl_idx], rl_path[rl_idx], rl_path[:rl_idx + 1], '#e67e22')

            bfs_idx = min(frame, len(bfs_path) - 1)
            self._draw_grid_to_axis(ax_bfs, bfs_grids[bfs_idx], bfs_path[bfs_idx], bfs_path[:bfs_idx + 1], '#3498db')

        anim = animation.FuncAnimation(fig, update, frames=max_frames, interval=400, repeat=False)
        anim.save(f"dynamic_comparison_{output_file}", writer='pillow', fps=2.5)
        print(f"Анимация для основной карты сохранена в: dynamic_comparison_{output_file}")
        plt.show()

        if show_alternative:
            print("\n" + "=" * 55)
            print("АЛЬТЕРНАТИВНАЯ КАРТА (другое расположение препятствий)")
            print("=" * 55)

            alt_grid = self.generate_alternative_map()
            alt_config = self.config
            alt_config.REWARD_GRID = alt_grid

            alt_dynamic_env = DynamicEnvironment(alt_config, self.agent)

            rl_path_alt, rl_grids_alt, rl_reward_alt, rl_steps_alt, rl_time_alt = alt_dynamic_env.simulate_rl()
            bfs_path_alt, bfs_grids_alt, bfs_reward_alt, bfs_steps_alt, bfs_time_alt = alt_dynamic_env.simulate_bfs()

            print(f"\n RL Агент (Local Vision):")
            print(f"   - Сделано шагов:      {rl_steps_alt}")
            print(f"   - Время выполнения:   {rl_time_alt:.6f} сек")
            print(f"   - Суммарная награда:  {rl_reward_alt:.1f}")

            print(f"\n BFS (Dynamic Replanning):")
            print(f"   - Сделано шагов:      {bfs_steps_alt}")
            print(f"   - Время выполнения:   {bfs_time_alt:.6f} сек")
            print(f"   - Суммарная награда:  {bfs_reward_alt:.1f}")

            fig2, (ax_rl_alt, ax_bfs_alt) = plt.subplots(1, 2, figsize=(14, 7))
            max_frames_alt = max(len(rl_path_alt), len(bfs_path_alt))

            def update_alt(frame):
                ax_rl_alt.clear()
                ax_bfs_alt.clear()

                ax_rl_alt.set_title(f"RL Agent - Альтернативная карта\nStep: {min(frame, len(rl_path_alt) - 1)}",
                                    fontsize=12, fontweight='bold')
                ax_bfs_alt.set_title(f"BFS - Альтернативная карта\nStep: {min(frame, len(bfs_path_alt) - 1)}",
                                     fontsize=12, fontweight='bold')

                rl_idx = min(frame, len(rl_path_alt) - 1)
                self._draw_grid_to_axis(ax_rl_alt, rl_grids_alt[rl_idx], rl_path_alt[rl_idx], rl_path_alt[:rl_idx + 1],
                                        '#e67e22')

                bfs_idx = min(frame, len(bfs_path_alt) - 1)
                self._draw_grid_to_axis(ax_bfs_alt, bfs_grids_alt[bfs_idx], bfs_path_alt[bfs_idx],
                                        bfs_path_alt[:bfs_idx + 1], '#3498db')

            anim_alt = animation.FuncAnimation(fig2, update_alt, frames=max_frames_alt, interval=400, repeat=False)
            anim_alt.save(f"dynamic_comparison_alternative_{output_file}", writer='pillow', fps=2.5)
            print(f"Анимация для альтернативной карты сохранена в: dynamic_comparison_alternative_{output_file}")
            plt.show()

    def _draw_grid_to_axis(self, ax, grid, agent_pos, current_path, path_color):
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='#bdc3c7', linestyle='--', linewidth=0.5)

        for i in range(self.size):
            for j in range(self.size):
                val = grid[i, j]
                if val == -10:
                    color = '#2c3e50'  # Препятствие
                elif (i, j) == self.start:
                    color = '#27ae60'  # Старт
                elif (i, j) == self.goal:
                    color = '#c0392b'  # Цель
                else:
                    color = '#f5f6fa'  # Пусто

                rect = patches.Rectangle((j, self.size - 1 - i), 1, 1, facecolor=color, edgecolor='#bdc3c7')
                ax.add_patch(rect)

        for r, c in current_path:
            if (r, c) != self.start and (r, c) != self.goal:
                rect = patches.Rectangle((c + 0.25, self.size - 1 - r + 0.25), 0.5, 0.5, facecolor=path_color,
                                         alpha=0.5)
                ax.add_patch(rect)

        rect = patches.Rectangle((agent_pos[1] + 0.1, self.size - 1 - agent_pos[0] + 0.1), 0.8, 0.8,
                                 facecolor='#f1c40f', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

