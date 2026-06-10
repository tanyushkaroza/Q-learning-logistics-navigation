import numpy as np
import random
from collections import deque
import os
import csv
from typing import Tuple, List, Dict

class MatricesForTest:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def get_params(self, size: int) -> Dict[str, float]:
        configs = {5: 0.25, 7: 0.20, 10: 0.15}
        density = configs.get(size, 0.20)
        return {
            'density': density,
            'step': -1,
            'barrier': -10,
            'goal': 100
        }

    def _get_neighbors(self, r: int, c: int, size: int):
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc

    def generate_matrix(self, size: int = 10) -> np.ndarray:
        p = self.get_params(size)
        start = (size - 1, 0)
        goal = (0, size - 1)

        grid = np.full((size, size), p['step'], dtype=float)
        grid[goal] = p['goal']

        n_barriers = int(size * size * p['density'])
        placed = 0
        attempts = 0
        max_attempts = n_barriers * 5

        while placed < n_barriers and attempts < max_attempts:
            r, c = random.randint(0, size - 1), random.randint(0, size - 1)

            dist_to_start = abs(r - start[0]) + abs(c - start[1])
            dist_to_goal = abs(r - goal[0]) + abs(c - goal[1])

            if dist_to_start > 1 and dist_to_goal > 1 and grid[r, c] == p['step']:
                grid[r, c] = p['barrier']
                placed += 1
            attempts += 1

        return self._ensure_path_exists(grid, start, goal, p['step'])

    def _ensure_path_exists(self, grid: np.ndarray, start: Tuple[int, int],
                            goal: Tuple[int, int], step_val: int) -> np.ndarray:
        size = len(grid)
        queue = deque([(start, [start])])
        visited = {start}
        found_path = None

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                found_path = path
                break

            for nr, nc in self._get_neighbors(r, c, size):
                if (nr, nc) not in visited and grid[nr, nc] != -10:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

        if not found_path:
            queue = deque([(start, [start])])
            visited = {start}
            while queue:
                (r, c), path = queue.popleft()
                if (r, c) == goal:
                    for pr, pc in path:
                        if grid[pr, pc] == -10:
                            grid[pr, pc] = step_val
                    break
                for nr, nc in self._get_neighbors(r, c, size):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        return grid

    def generate_test_set(self, size: int = 10, samples: int = 50,
                          verbose: bool = False) -> List[np.ndarray]:
        test_set = []
        if verbose:
            print(f"Генерация тестовых матриц (размер {size}x{size}, {samples} шт.)")
        for i in range(samples):
            test_set.append(self.generate_matrix(size))
        return test_set

    def save_matrices_csv(self, matrices: List[np.ndarray], filename: str = "test_matrices.csv") -> str:
        os.makedirs("matrices_data", exist_ok=True)
        filepath = os.path.join("matrices_data", filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["#", f"Тестовые матрицы {len(matrices)} шт., размер {matrices[0].shape[0]}x{matrices[0].shape[0]}"])
            writer.writerow(["#", "Формат: step=-1 (проход), barrier=-10 (препятствие), goal=100 (цель)"])
            writer.writerow([])

            for idx, matrix in enumerate(matrices):
                writer.writerow([f"Test_Matrix_{idx}", f"size={matrix.shape[0]}x{matrix.shape[0]}"])
                for row in matrix:
                    writer.writerow(row.tolist())
                writer.writerow([])

        print(f"✅ Сохранено {len(matrices)} матриц в {filepath}")
        return filepath

    def generate_and_save(self, samples: int = 50, size: int = 10,
                          verbose: bool = False) -> List[np.ndarray]:
        matrices = self.generate_test_set(size=size, samples=samples, verbose=verbose)
        self.save_matrices_csv(matrices, filename="test_matrices.csv")
        return matrices

    def load_matrices_csv(self, filename: str = "test_matrices.csv") -> List[np.ndarray]:
        filepath = os.path.join("matrices_data", filename)
        matrices = []
        current_matrix = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    if current_matrix:
                        matrices.append(np.array(current_matrix, dtype=float))
                        current_matrix = []
                elif row[0].startswith('Test_Matrix_') or row[0].startswith('#'):
                    continue
                else:
                    try:
                        current_matrix.append([float(x) for x in row])
                    except ValueError:
                        continue

        if current_matrix:
            matrices.append(np.array(current_matrix, dtype=float))

        return matrices