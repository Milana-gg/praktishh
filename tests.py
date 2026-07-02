import unittest
import numpy as np
import sys
import os
from Мухина_Милана_БИС_24_2 import SchellingModel

# Добавляем путь к папке с основным файлом
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем твою модель (ЗАМЕНИ НА СВОЁ ИМЯ ФАЙЛА)
from Мухина_Милана_БИС_24_2 import SchellingModel


class TestSchellingModel(unittest.TestCase):
    """Тесты для модели сегрегации Шеллинга"""

    def test_initialization(self):
        """Тест 1: Проверка инициализации сетки"""
        model = SchellingModel(size=10, n_agents=50, empty_ratio=0.2)

        self.assertEqual(model.grid.shape, (10, 10))
        self.assertEqual(len(model.agents), 50)

        expected_empty = int(10 * 10 * 0.2)
        self.assertEqual(len(model.empty_cells), expected_empty)

        for agent in model.agents:
            self.assertIn(agent['type'], [1, 2])

        print("Тест 1 пройден: инициализация работает")

    def test_is_happy_empty_cell(self):
        """Тест 2: Проверка is_happy на пустой клетке"""
        model = SchellingModel(size=5, n_agents=10, empty_ratio=0.3)
        empty_cell = list(model.empty_cells)[0]
        row, col = empty_cell

        self.assertTrue(model.is_happy(row, col))
        print("Тест 2 пройден: пустая клетка считается удовлетворённой")

    def test_is_happy_no_neighbors(self):
        """Тест 3: Проверка is_happy на агенте без соседей"""
        model = SchellingModel(size=5, n_agents=5, empty_ratio=0.8)

        if model.agents:
            row, col = model.agents[0]['pos']
            self.assertTrue(model.is_happy(row, col))
        print("Тест 3 пройден: агент без соседей удовлетворён")

    def test_get_neighbors(self):
        """Тест 4: Проверка get_neighbors"""
        model = SchellingModel(size=5, n_agents=10, empty_ratio=0.3)

        model.grid = np.zeros((5, 5), dtype=int)
        model.grid[2, 2] = 1
        model.grid[2, 3] = 2
        model.grid[1, 2] = 1

        neighbors = model.get_neighbors(2, 2)

        self.assertIn(2, neighbors)
        self.assertIn(1, neighbors)
        self.assertEqual(len(neighbors), 2)
        print("Тест 4 пройден: соседи определяются корректно")

    def test_segregation_index(self):
        """Тест 5: Проверка расчёта индекса сегрегации"""
        model = SchellingModel(size=5, n_agents=10, empty_ratio=0.3)

        model.grid = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]
        ])

        model.agents = []
        for row in range(5):
            for col in range(5):
                if model.grid[row, col] > 0:
                    model.agents.append({'pos': (row, col), 'type': model.grid[row, col]})

        index = model.calculate_segregation_index()

        self.assertGreater(index, 0.8)
        print(f"Тест 5 пройден: индекс сегрегации = {index:.3f}")

    def test_step(self):
        """Тест 6: Проверка одного шага моделирования"""
        model = SchellingModel(size=10, n_agents=50, empty_ratio=0.2, tolerance=0.3)

        moved = model.step()

        self.assertEqual(model.generation, 1)
        self.assertEqual(len(model.moves_history), 1)
        self.assertEqual(len(model.segregation_history), 1)
        self.assertGreaterEqual(moved, 0)
        print("Тест 6 пройден: шаг моделирования выполнен")

    def test_full_model_run(self):
        """Тест 7: Полный цикл моделирования до стабилизации"""
        model = SchellingModel(size=20, n_agents=100, empty_ratio=0.2, tolerance=0.3)

        for _ in range(100):
            moved = model.step()
            if moved == 0:
                break

        self.assertGreater(model.generation, 0)
        self.assertGreaterEqual(len(model.segregation_history), 1)
        print(f"Тест 7 пройден: модель стабилизировалась на поколении {model.generation}")

    def test_export_results(self):
        """Тест 8: Проверка экспорта данных"""
        model = SchellingModel(size=10, n_agents=30, empty_ratio=0.2, tolerance=0.3)

        for _ in range(10):
            model.step()

        # Импортируем функцию сохранения
        from Мухина_Милана_БИС_24_2 import save_results_to_txt

        test_filename = 'test_results_temp.txt'
        save_results_to_txt(model, test_filename)

        self.assertTrue(os.path.exists(test_filename))
        self.assertGreater(os.path.getsize(test_filename), 0)

        os.remove(test_filename)
        print("Тест 8 пройден: экспорт данных работает")


if __name__ == '__main__':
    unittest.main(verbosity=2)
    # ===================== ТЕСТЫ =====================
    if __name__ == '__main__':
        import unittest
        import numpy as np


        class TestSchellingModel(unittest.TestCase):

            def test_initialization(self):
                """Тест 1: Проверка инициализации"""
                model = SchellingModel(size=10, n_agents=50, empty_ratio=0.2)
                self.assertEqual(model.grid.shape, (10, 10))
                self.assertEqual(len(model.agents), 50)
                print(" Тест 1 пройден")

            def test_is_happy_empty(self):
                """Тест 2: Проверка is_happy на пустой клетке"""
                model = SchellingModel(size=5, n_agents=10, empty_ratio=0.3)
                empty_cell = list(model.empty_cells)[0]
                row, col = empty_cell
                self.assertTrue(model.is_happy(row, col))
                print("Тест 2 пройден")

            def test_get_neighbors(self):
                """Тест 3: Проверка get_neighbors"""
                model = SchellingModel(size=5, n_agents=10, empty_ratio=0.3)
                model.grid = np.zeros((5, 5), dtype=int)
                model.grid[2, 2] = 1
                model.grid[2, 3] = 2
                neighbors = model.get_neighbors(2, 2)
                self.assertIn(2, neighbors)
                print("Тест 3 пройден")

            def test_segregation_index(self):
                """Тест 4: Проверка индекса сегрегации"""
                model = SchellingModel(size=5, n_agents=10, empty_ratio=0.3)
                model.grid = np.array([
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 2, 2],
                    [0, 0, 0, 2, 2]
                ])
                model.agents = []
                for row in range(5):
                    for col in range(5):
                        if model.grid[row, col] > 0:
                            model.agents.append({'pos': (row, col), 'type': model.grid[row, col]})
                index = model.calculate_segregation_index()
                self.assertGreater(index, 0.8)
                print("Тест 4 пройден")

            def test_step(self):
                """Тест 5: Проверка шага моделирования"""
                model = SchellingModel(size=10, n_agents=50, empty_ratio=0.2, tolerance=0.3)
                moved = model.step()
                self.assertEqual(model.generation, 1)
                self.assertGreaterEqual(moved, 0)
                print("Тест 5 пройден")


        # Запускаем тесты
        unittest.main(verbosity=2)