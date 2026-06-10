import numpy as np
import time
from Config import Config
from Environment import Environment
from AgentForTrain import AgentForTrain
from AgentForTest import AgentForTest
from Train import Train
from Test import Test
from MatricesForTrain import MatricesForTrain
from MatricesForTest import MatricesForTest
from BFS import BFS
from DynamicEnvironment import DynamicEnvironment
import matplotlib.pyplot as plt


def load_matrix_from_txt(filepath: str) -> np.ndarray:
    """Загрузка матрицы из txt файла"""
    matrix = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = [float(x) for x in line.strip().split()]
                if row:
                    matrix.append(row)
    return np.array(matrix)


def find_rl_path(agent, grid, start, goal, max_steps=1000):
    """Находит путь RL агента на заданной матрице"""
    size = grid.shape[0]
    temp_config = Config(reward_grid=grid)
    temp_config.START_POSITION = start
    temp_config.GOAL_POSITION = goal

    test_agent = AgentForTest(temp_config)
    test_agent.set_q_table(agent._q_table)
    test_agent.set_reward_grid(grid)  # ВАЖНО: устанавливаем карту препятствий

    state = start
    path = [state]
    total_reward = 0
    steps = 0
    visited_states = set()  # Для отслеживания зацикливания

    print(f"\n  Поиск пути RL от {start} до {goal}")
    print(f"  Размер сетки: {size}x{size}")
    print(f"  Препятствий на карте: {np.sum(grid == -10)}")

    while state != goal and steps < max_steps:
        # Получаем лучшее действие с учетом препятствий
        action = test_agent.get_best_action(state)
        dr, dc = temp_config.ACTIONS[action]
        next_state = (state[0] + dr, state[1] + dc)

        # Проверка границ
        if not (0 <= next_state[0] < size and 0 <= next_state[1] < size):
            print(f"    Шаг {steps}: Попытка выйти за границы! Останавливаемся.")
            break

        # Проверка на препятствие (хотя get_best_action уже должен это блокировать)
        if grid[next_state[0], next_state[1]] == -10:
            print(f"    Шаг {steps}: ОШИБКА! Агент пытается шагнуть в препятствие {next_state}")
            break

        state = next_state
        path.append(state)
        total_reward += grid[state[0], state[1]]
        steps += 1

        # Отслеживание зацикливания
        if state in visited_states:
            print(f"    Шаг {steps}: Обнаружено зацикливание на {state}")
            break
        visited_states.add(state)

        if state == goal:
            print(f"    ✅ Цель достигнута за {steps} шагов!")
            break

        if steps % 50 == 0:
            print(f"    Шаг {steps}: позиция {state}, награда {total_reward:.1f}")

    if state != goal:
        print(f"    ❌ Цель не достигнута! Шагов: {steps}, награда: {total_reward:.1f}")

    return path, total_reward, steps


def find_bfs_path(grid, start, goal):
    """Находит путь BFS на заданной матрице"""
    bfs_solver = BFS(grid, start, goal)
    path, reward, steps, _ = bfs_solver.find_path()
    return path, reward, steps


def visualize_matrix_pair(title, grid1, start1, goal1, path_rl1, reward_rl1, steps_rl1,
                          path_bfs1, reward_bfs1, steps_bfs1,
                          grid2=None, start2=None, goal2=None,
                          path_rl2=None, reward_rl2=0, steps_rl2=0,
                          path_bfs2=None, reward_bfs2=0, steps_bfs2=0):
    """
    Визуализирует одну или две пары графиков (RL слева, BFS справа)
    """
    if grid2 is not None:
        # Две матрицы - создаем два окна
        fig1, (ax_rl1, ax_bfs1) = plt.subplots(1, 2, figsize=(14, 7))

        Environment.draw_comparison_grid(
            ax=ax_rl1, title=f"{title}\nRL Агент (Матрица 1)",
            grid=grid1, start=start1, goal=goal1,
            path=path_rl1, path_color='#e67e22',
            steps=steps_rl1, reward=reward_rl1
        )

        Environment.draw_comparison_grid(
            ax=ax_bfs1, title=f"{title}\nBFS Алгоритм (Матрица 1)",
            grid=grid1, start=start1, goal=goal1,
            path=path_bfs1, path_color='#3498db',
            steps=steps_bfs1, reward=reward_bfs1
        )

        plt.tight_layout()
        plt.show()

        # Второе окно для второй матрицы
        fig2, (ax_rl2, ax_bfs2) = plt.subplots(1, 2, figsize=(14, 7))

        Environment.draw_comparison_grid(
            ax=ax_rl2, title=f"{title}\nRL Агент (Матрица 2)",
            grid=grid2, start=start2, goal=goal2,
            path=path_rl2, path_color='#e67e22',
            steps=steps_rl2, reward=reward_rl2
        )

        Environment.draw_comparison_grid(
            ax=ax_bfs2, title=f"{title}\nBFS Алгоритм (Матрица 2)",
            grid=grid2, start=start2, goal=goal2,
            path=path_bfs2, path_color='#3498db',
            steps=steps_bfs2, reward=reward_bfs2
        )

        plt.tight_layout()
        plt.show()

    else:
        # Одна матрица - одно окно
        fig, (ax_rl, ax_bfs) = plt.subplots(1, 2, figsize=(14, 7))

        Environment.draw_comparison_grid(
            ax=ax_rl, title=f"{title}\nRL Агент",
            grid=grid1, start=start1, goal=goal1,
            path=path_rl1, path_color='#e67e22',
            steps=steps_rl1, reward=reward_rl1
        )

        Environment.draw_comparison_grid(
            ax=ax_bfs, title=f"{title}\nBFS Алгоритм",
            grid=grid1, start=start1, goal=goal1,
            path=path_bfs1, path_color='#3498db',
            steps=steps_bfs1, reward=reward_bfs1
        )

        plt.tight_layout()
        plt.show()


def generate_datasets():
    """Генерация матриц (5x5, 7x7, 10x10)"""
    train_generator = MatricesForTrain(random_state=42)
    train_matrices = train_generator.generate_and_save(
        sizes=[5, 7, 10],
        samples_per_size=100,
        verbose=True
    )

    test_generator = MatricesForTest(random_state=42)
    test_matrices = test_generator.generate_and_save(
        samples=50,
        size=10,
        verbose=False
    )
    print(f"\nСгенерировано {len(test_matrices)} тестовых матриц 10x10")
    print(f"\nФайлы сохранены в папке 'matrices_data':")
    print(f"  - train_matrices.csv")
    print(f"  - test_matrices.csv")

    return train_matrices, test_matrices


def run_training(train_matrices):
    """Обучение агента"""
    print(f"  Всего матриц для обучения: {len(train_matrices)}")

    sizes = sorted(set(m.shape[0] for m in train_matrices))
    print(f"  Размеры матриц: {sizes}")

    from collections import Counter
    size_counts = Counter(m.shape[0] for m in train_matrices)
    for size, count in sorted(size_counts.items()):
        print(f"    - {size}x{size}: {count} шт.")

    max_size = max(sizes)
    print(f"\n  Максимальный размер матрицы: {max_size}x{max_size}")

    temp_config = Config()
    temp_config.GRID_SIZE = max_size

    agent_train = AgentForTrain(temp_config)

    total_episodes = 0
    total_time = 0
    all_rewards = []

    print("\n" + "-" * 60)
    print("Обучение на всех матрицах")
    print("-" * 60)

    for idx, matrix in enumerate(train_matrices):
        matrix_size = matrix.shape[0]

        config = Config(reward_grid=matrix)

        if matrix_size <= 5:
            config.EPISODES = 1000
        elif matrix_size <= 7:
            config.EPISODES = 1500
        else:
            config.EPISODES = 3000

        env = Environment(config)

        if agent_train._q_table.shape[0] != matrix_size:
            print(
                f"\n  Изменение размера Q-таблицы: {agent_train._q_table.shape[0]}x{agent_train._q_table.shape[0]} -> {matrix_size}x{matrix_size}")
            old_q_table = agent_train._q_table
            agent_train._q_table = np.zeros((matrix_size, matrix_size, agent_train._n_actions))
            agent_train._grid_size = matrix_size

            min_size = min(old_q_table.shape[0], matrix_size)
            agent_train._q_table[:min_size, :min_size, :] = old_q_table[:min_size, :min_size, :]

        trainer = Train(agent_train, env, config)

        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(train_matrices) - 1:
            verbose = True
            print(f"\nМатрица {idx + 1}/{len(train_matrices)} (размер {matrix_size}x{matrix_size})")
            print(f"   Эпизодов: {config.EPISODES}")
        else:
            verbose = False

        start_time = time.time()
        rewards = trainer.train(verbose=verbose)
        train_time = time.time() - start_time

        total_episodes += config.EPISODES
        total_time += train_time
        all_rewards.extend(rewards)

        if (idx + 1) % 50 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            success_rate = sum(1 for r in rewards[-100:] if r >= config.SUCCESS_THRESHOLD) if len(rewards) >= 100 else 0
            print(f"   Завершено за {train_time:.2f} сек")
            print(f"   Средняя награда: {avg_reward:.2f}")
            print(f"   Успешность: {success_rate}/{min(100, len(rewards))}")
            print(f"   Прогресс: {idx + 1}/{len(train_matrices)} матриц ({100 * (idx + 1) / len(train_matrices):.1f}%)")

    print(f"\n  Всего обработано матриц: {len(train_matrices)}")
    print(f"  Всего эпизодов обучения: {total_episodes}")
    print(f"  Общее время обучения: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
    print(f"  Среднее время на матрицу: {total_time / len(train_matrices):.2f} сек")

    q_stats = agent_train.get_stats()
    print(f"\n  Финальная Q-таблица:")
    print(f"    Размер: {q_stats['q_table_shape']}")
    print(f"    Среднее Q: {q_stats['mean_q']:.4f}")
    print(f"    Макс Q: {q_stats['max_q']:.4f}")
    print(f"    Мин Q: {q_stats['min_q']:.4f}")

    agent_train.save_q_table("q_table_fully_trained.csv")
    print(f"\n  Финальная Q-таблица сохранена в 'q_table_fully_trained.csv'")

    return agent_train, config, env, total_time


def run_testing(agent_train, config, env):
    """Тестирование агента"""
    agent_test = AgentForTest(config)
    agent_test.set_q_table(agent_train._q_table)

    tester = Test(agent_test, env, config)
    start_test_time = time.time()
    test_rewards, test_steps, success_count, avg_test_discounted = tester.test(verbose=True)
    test_time = time.time() - start_test_time

    print("\n" + "-" * 40)
    print("Результаты")
    print("-" * 40)
    print(f"Обучение:")
    print(f"  Количество эпизодов: {config.EPISODES}")
    best_reward = agent_train.get_best_reward()
    if best_reward != float('-inf'):
        print(f"  Лучшая награда: {best_reward:.2f}")

    print(f"\nТестирование:")
    print(f"  Количество эпизодов: {config.TEST_EPISODES}")
    print(f"  Время тестирования: {test_time:.2f} сек")
    print(f"  Средняя награда: {np.mean(test_rewards):.2f}")
    print(f"  Среднее число шагов: {np.mean(test_steps):.2f}")
    print(f"  Успешность: {success_count}/{config.TEST_EPISODES} ({100 * success_count / config.TEST_EPISODES:.1f}%)")
    print(f"  Средняя дисконтированная награда: {avg_test_discounted:.2f}")

    optimal_path = tester._find_optimal_path()
    print(f"\nОптимальный путь (RL):")
    print(f"  Маршрут: {optimal_path}")
    print(f"  Длина: {len(optimal_path) - 1} шагов")

    return optimal_path, np.mean(test_rewards), len(optimal_path) - 1, test_time, agent_train


def view_matrices():
    """Просмотр сохраненных матриц"""
    import os

    train_path = "matrices_data/train_matrices.csv"
    test_path = "matrices_data/test_matrices.csv"

    if os.path.exists(train_path):
        print(f"\nОбучающие матрицы ({train_path})")
    else:
        print(f"\n Файл {train_path} не найден. Сначала сгенерируйте датасеты (пункт 1).")

    if os.path.exists(test_path):
        print(f"\nТестовые матрицы ({test_path})")
    else:
        print(f"\nФайл {test_path} не найден. Сначала сгенерируйте датасеты (пункт 1).")


def main():
    train_matrices = None
    agent_train = None
    config = None
    env = None

    while True:
        print("\n" + "=" * 50)
        print("МЕНЮ")
        print("=" * 50)
        print("1. Сгенерировать датасеты (CSV файлы)")
        print("2. Обучить агента")
        print("3. Протестировать обученного агента")
        print("4. Полный цикл (генерация + обучение + тест)")
        print("5. Просмотреть сохраненные матрицы")
        print("6. Сравнить RL агента с BFS в статической среде")
        print("7. Сохранить текущую Q-таблицу")
        print("8. Сравнить RL и BFS в динамической среде")
        print("0. Выйти")
        print("-" * 50)

        choice = input("\nВаш выбор (0-8): ").strip()

        if choice == '1':
            train_matrices, _ = generate_datasets()

        elif choice == '2':
            if train_matrices is None:
                print("\nСначала сгенерируйте датасеты (пункт 1)!")
                continue
            agent_train, config, env, _ = run_training(train_matrices)

        elif choice == '3':
            if agent_train is None:
                print("\nСначала обучите агента (пункт 2 или 4)!")
                continue
            run_testing(agent_train, config, env)

        elif choice == '4':
            print("\nЗапуск полного цикла обучения")
            print("=" * 50)
            train_matrices, _ = generate_datasets()
            agent_train, config, env, _ = run_training(train_matrices)
            run_testing(agent_train, config, env)
            print("\nПолный цикл завершен")

        elif choice == '5':
            view_matrices()

        elif choice == '6':
            if agent_train is None:
                print("\nНет обученного агента!")
                print("Сначала выполните пункт 4 (полный цикл обучения)!")
                continue

            print("\n" + "=" * 50)
            print("СРАВНЕНИЕ RL vs BFS В СТАТИЧЕСКОЙ СРЕДЕ")
            print("=" * 50)

            # Получаем start и goal для основной матрицы
            size1 = config.GRID_SIZE
            start1 = config.START_POSITION
            goal1 = config.GOAL_POSITION

            # Находим пути для основной матрицы
            print("\n📊 Обработка основной матрицы...")

            # --- ЗАМЕР ВРЕМЕНИ ДЛЯ ОСНОВНОЙ МАТРИЦЫ ---
            start_time_rl1 = time.time()
            rl_path1, rl_reward1, rl_steps1 = find_rl_path(agent_train, config.REWARD_GRID, start1, goal1)
            time_rl1 = time.time() - start_time_rl1

            start_time_bfs1 = time.time()
            bfs_path1, bfs_reward1, bfs_steps1 = find_bfs_path(config.REWARD_GRID, start1, goal1)
            time_bfs1 = time.time() - start_time_bfs1

            print(f"   RL:  шагов={rl_steps1}, награда={rl_reward1:.1f}, время={time_rl1:.6f} сек")
            print(f"   BFS: шагов={bfs_steps1}, награда={bfs_reward1:.1f}, время={time_bfs1:.6f} сек")

            # Запрашиваем кастомную матрицу
            txt_filepath = input("\n📁 Введите путь к txt файлу с матрицей (Enter - показать только основную): ").strip()

            if txt_filepath:
                try:
                    second_matrix = load_matrix_from_txt(txt_filepath)
                    print(f"✅ Загружена матрица из {txt_filepath}")
                    print(f"   Размер: {second_matrix.shape[0]}x{second_matrix.shape[1]}")

                    size2 = second_matrix.shape[0]
                    start2 = (size2 - 1, 0)
                    goal2 = (0, size2 - 1)

                    print("\n📊 Обработка загруженной матрицы...")

                    # --- ЗАМЕР ВРЕМЕНИ ДЛЯ КАСТОМНОЙ МАТРИЦЫ ---
                    start_time_rl2 = time.time()
                    rl_path2, rl_reward2, rl_steps2 = find_rl_path(agent_train, second_matrix, start2, goal2)
                    time_rl2 = time.time() - start_time_rl2

                    start_time_bfs2 = time.time()
                    bfs_path2, bfs_reward2, bfs_steps2 = find_bfs_path(second_matrix, start2, goal2)
                    time_bfs2 = time.time() - start_time_bfs2

                    print(f"   RL:  шагов={rl_steps2}, награда={rl_reward2:.1f}, время={time_rl2:.6f} сек")
                    print(f"   BFS: шагов={bfs_steps2}, награда={bfs_reward2:.1f}, время={time_bfs2:.6f} сек")

                    # Визуализируем обе матрицы
                    visualize_matrix_pair(
                        title="Сравнение RL vs BFS",
                        grid1=config.REWARD_GRID, start1=start1, goal1=goal1,
                        path_rl1=rl_path1, reward_rl1=rl_reward1, steps_rl1=rl_steps1,
                        path_bfs1=bfs_path1, reward_bfs1=bfs_reward1, steps_bfs1=bfs_steps1,
                        grid2=second_matrix, start2=start2, goal2=goal2,
                        path_rl2=rl_path2, reward_rl2=rl_reward2, steps_rl2=rl_steps2,
                        path_bfs2=bfs_path2, reward_bfs2=bfs_reward2, steps_bfs2=bfs_steps2
                    )

                    print("\n" + "=" * 60)
                    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
                    print("=" * 60)
                    print(f"\n📊 Основная карта ({size1}x{size1}):")
                    print(
                        f"   Разница: RL {'лучше' if rl_reward1 > bfs_reward1 else 'хуже' if rl_reward1 < bfs_reward1 else 'равен'} на {abs(rl_reward1 - bfs_reward1):.1f}")
                    print(
                        f"   По скорости: {'BFS' if time_bfs1 < time_rl1 else 'RL'} быстрее на {abs(time_rl1 - time_bfs1):.6f} сек")

                    print(f"\n📊 Загруженная карта ({size2}x{size2}):")
                    print(
                        f"   Разница: RL {'лучше' if rl_reward2 > bfs_reward2 else 'хуже' if rl_reward2 < bfs_reward2 else 'равен'} на {abs(rl_reward2 - bfs_reward2):.1f}")
                    print(
                        f"   По скорости: {'BFS' if time_bfs2 < time_rl2 else 'RL'} быстрее на {abs(time_rl2 - time_bfs2):.6f} сек")
                    print("=" * 60)

                except FileNotFoundError:
                    print(f"❌ Файл {txt_filepath} не найден! Показываю только основную матрицу.")
                    visualize_matrix_pair(
                        title="Сравнение RL vs BFS",
                        grid1=config.REWARD_GRID, start1=start1, goal1=goal1,
                        path_rl1=rl_path1, reward_rl1=rl_reward1, steps_rl1=rl_steps1,
                        path_bfs1=bfs_path1, reward_bfs1=bfs_reward1, steps_bfs1=bfs_steps1
                    )
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                visualize_matrix_pair(
                    title="Сравнение RL vs BFS",
                    grid1=config.REWARD_GRID, start1=start1, goal1=goal1,
                    path_rl1=rl_path1, reward_rl1=rl_reward1, steps_rl1=rl_steps1,
                    path_bfs1=bfs_path1, reward_bfs1=bfs_reward1, steps_bfs1=bfs_steps1
                )

        elif choice == '7':
            if agent_train is None:
                print("\nНет обученного агента для сохранения!")
                continue
            filepath = input("Введите имя файла для сохранения (например 'my_q_table.csv'): ").strip()
            if not filepath:
                filepath = "q_table_trained.csv"
            agent_train.save_q_table(filepath)
            print(f"Q-таблица сохранена в {filepath}")

        elif choice == '8':
            if agent_train is None:
                print("\nСначала обучите агента (пункт 2 или 4)!")
                continue

            print("\n" + "=" * 60)
            print("ТЕСТИРОВАНИЕ В ДИНАМИЧЕСКОЙ СРЕДЕ")
            print("=" * 60)

            txt_filepath = input(
                "\n📁 Введите путь к txt файлу с матрицей (Enter - использовать стандартные карты): ").strip()

            if txt_filepath:
                try:
                    custom_matrix = load_matrix_from_txt(txt_filepath)
                    print(f"✅ Загружена матрица из {txt_filepath}")
                    print(f"   Размер: {custom_matrix.shape[0]}x{custom_matrix.shape[1]}")

                    custom_config = Config(reward_grid=custom_matrix)
                    custom_agent = AgentForTest(custom_config)

                    # Проверяем размеры Q-таблицы и загруженной матрицы для безопасности
                    q_size = agent_train._q_table.shape[0]
                    mat_size = custom_matrix.shape[0]

                    if q_size == mat_size:
                        custom_agent.set_q_table(agent_train._q_table)
                    else:
                        print(f"⚠️ Внимание: размер матрицы ({mat_size}x{mat_size}) отличается от Q-таблицы ({q_size}x{q_size}).")
                        print("Производится безопасная адаптация Q-таблицы...")
                        new_q_table = np.zeros((mat_size, mat_size, agent_train._n_actions))
                        min_size = min(q_size, mat_size)
                        new_q_table[:min_size, :min_size, :] = agent_train._q_table[:min_size, :min_size, :]
                        custom_agent.set_q_table(new_q_table)

                    dynamic_env = DynamicEnvironment(custom_config, custom_agent)

                    print("\nСоздание анимации для загруженной карты...")
                    dynamic_env.visualize_dynamic_comparison(output_file="dynamic_custom_comparison.gif",
                                                             show_alternative=False)

                except FileNotFoundError:
                    print(f"❌ Файл {txt_filepath} не найден!")
                except Exception as e:
                    print(f"❌ Ошибка при обработке файла: {e}")
            else:
                print("Будут созданы две анимации:")
                print("  1. Основная карта (стандартные препятствия)")
                print("  2. Карта с высокой плотностью препятствий (~40%)")

                agent_test = AgentForTest(config)
                agent_test.set_q_table(agent_train._q_table)
                dynamic_env = DynamicEnvironment(config, agent_test)
                dynamic_env.visualize_dynamic_comparison()

        elif choice == '0':
            print("\nЗавершение работы...")
            break

        else:
            print("\nНеверный выбор. Пожалуйста, выберите от 0 до 8.")


if __name__ == "__main__":
    main()
