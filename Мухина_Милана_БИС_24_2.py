import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings('ignore')


class SchellingModel:
    def __init__(self, size=40, n_agents=1200, empty_ratio=0.2, tolerance=0.3, n_types=2):
        self.size = size
        self.n_agents = n_agents
        self.empty_ratio = empty_ratio
        self.tolerance = tolerance
        self.n_types = n_types

        self.grid = np.zeros((size, size), dtype=int)
        self.agents = []
        self.empty_cells = set()
        self.moves_history = []
        self.segregation_history = []
        self.generation = 0

        self._initialize_grid()

    def _initialize_grid(self):
        total_cells = self.size * self.size
        n_empty = int(total_cells * self.empty_ratio)
        n_agents_per_type = self.n_agents // self.n_types

        all_cells = list(range(total_cells))
        np.random.shuffle(all_cells)

        empty_cells = all_cells[:n_empty]
        for cell in empty_cells:
            row = cell // self.size
            col = cell % self.size
            self.empty_cells.add((row, col))

        agent_cells = all_cells[n_empty:n_empty + self.n_agents]
        agent_counter = 0

        for cell in agent_cells:
            row = cell // self.size
            col = cell % self.size
            agent_type = (agent_counter // n_agents_per_type) % self.n_types + 1
            self.grid[row, col] = agent_type
            self.agents.append({'pos': (row, col), 'type': agent_type})
            agent_counter += 1

    def get_neighbors(self, row, col, radius=1):
        neighbors = []
        for i in range(max(0, row - radius), min(self.size, row + radius + 1)):
            for j in range(max(0, col - radius), min(self.size, col + radius + 1)):
                if i == row and j == col:
                    continue
                if self.grid[i, j] > 0:
                    neighbors.append(self.grid[i, j])
        return neighbors

    def is_happy(self, row, col):
        if self.grid[row, col] == 0:
            return True

        agent_type = self.grid[row, col]
        neighbors = self.get_neighbors(row, col)

        if len(neighbors) == 0:
            return True

        same_type = sum(1 for n in neighbors if n == agent_type)
        ratio = same_type / len(neighbors)

        return ratio >= self.tolerance

    def find_empty_cell(self):
        if not self.empty_cells:
            return None
        empty_list = list(self.empty_cells)
        idx = np.random.randint(len(empty_list))
        return empty_list[idx]

    def move_agent(self, row, col):
        if self.grid[row, col] == 0:
            return False

        new_pos = self.find_empty_cell()
        if new_pos is None:
            return False

        new_row, new_col = new_pos

        agent_type = self.grid[row, col]
        self.grid[new_row, new_col] = agent_type
        self.grid[row, col] = 0

        self.empty_cells.remove((new_row, new_col))
        self.empty_cells.add((row, col))

        for agent in self.agents:
            if agent['pos'] == (row, col):
                agent['pos'] = (new_row, new_col)
                break

        return True

    def step(self):
        unhappy_agents = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] > 0 and not self.is_happy(row, col):
                    unhappy_agents.append((row, col))

        np.random.shuffle(unhappy_agents)

        moved = 0
        for row, col in unhappy_agents:
            if not self.is_happy(row, col):
                if self.move_agent(row, col):
                    moved += 1

        self.generation += 1
        self.moves_history.append(moved)
        self.segregation_history.append(self.calculate_segregation_index())

        return moved

    def calculate_segregation_index(self):
        total_ratio = 0
        n_agents = 0

        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] > 0:
                    neighbors = self.get_neighbors(row, col)
                    if len(neighbors) > 0:
                        agent_type = self.grid[row, col]
                        same = sum(1 for n in neighbors if n == agent_type)
                        total_ratio += same / len(neighbors)
                        n_agents += 1

        if n_agents == 0:
            return 0
        return total_ratio / n_agents


def save_results_to_txt(model, filename='results.txt'):
    """Сохранение результатов в текстовый файл"""

    with open(filename, 'w', encoding='utf-8') as f:
        # Параметры модели
        f.write("=" * 60 + "\n")
        f.write("МОДЕЛЬ СЕГРЕГАЦИИ ШЕЛЛИНГА - РЕЗУЛЬТАТЫ\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Размер сетки: {model.size}×{model.size}\n")
        f.write(f"Количество агентов: {model.n_agents}\n")
        f.write(f"Доля пустых клеток: {model.empty_ratio}\n")
        f.write(f"Терпимость: {model.tolerance:.2f}\n")
        f.write(f"Типов агентов: {model.n_types}\n\n")

        # Финальная статистика
        f.write("ФИНАЛЬНАЯ СТАТИСТИКА\n")
        f.write("")
        f.write(f"Конечное поколение: {model.generation}\n")
        f.write(f"Финальная сегрегация: {model.calculate_segregation_index():.3f}\n")
        f.write(f"Всего перемещений: {sum(model.moves_history)}\n\n")

        # История по поколениям
        f.write("")
        f.write("ИСТОРИЯ ПО ПОКОЛЕНИЯМ\n")
        f.write("")
        f.write(f"{'Поколение':<12} {'Перемещений':<15} {'Индекс сегрегации':<20}\n")
        f.write("")

        for i in range(len(model.moves_history)):
            f.write(f"{i + 1:<12} {model.moves_history[i]:<15} {model.segregation_history[i]:<20.3f}\n")

        f.write("")

    print(f"Результаты сохранены в {filename}")


def save_benchmark_to_txt(tolerances, segregations, generations, filename='benchmark.txt'):
    """Сохранение результатов бенчмаркинга в текстовый файл"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("")
        f.write("ЗАВИСИМОСТЬ СЕГРЕГАЦИИ ОТ ТЕРПИМОСТИ\n")
        f.write("\n\n")

        f.write(f"{'Терпимость':<12} {'Финальная сегрегация':<22} {'Поколений':<12}\n")
        f.write("")

        for tol, seg, gen in zip(tolerances, segregations, generations):
            f.write(f"{tol:<12.1f} {seg:<22.3f} {gen:<12}\n")

        f.write("")

    print(f"Результаты бенчмаркинга сохранены в {filename}")


def create_animation_with_tolerance(tolerance):
    """Создает анимацию с заданной терпимостью"""

    print(f"\n Запуск модели с терпимостью {tolerance:.2f}...")

    model = SchellingModel(size=40, n_agents=1200, empty_ratio=0.2, tolerance=tolerance)

    initial_grid = model.grid.copy()
    initial_seg = model.calculate_segregation_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['white', 'blue', 'red', 'green', 'orange', 'purple']
    cmap = ListedColormap(colors[:model.n_types + 1])

    im = ax1.imshow(initial_grid, cmap=cmap, vmin=0, vmax=model.n_types,
                    interpolation='nearest', animated=True)
    ax1.set_title(f'Поколение: 0  |  Терпимость: {tolerance:.2f}', fontsize=12)
    ax1.set_xlabel('Строка')
    ax1.set_ylabel('Колонка')

    patches = [mpatches.Patch(color=colors[i], label=f'Тип {i}')
               for i in range(1, model.n_types + 1)]
    patches.append(mpatches.Patch(color='white', label='Пусто'))
    ax1.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # настройка второго графика
    ax2.set_xlabel('Поколение')
    ax2.set_ylabel('Индекс сегрегации')
    ax2.set_title('Динамика сегрегации')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=tolerance, color='r', linestyle='--',
                label=f'Терпимость={tolerance:.2f}', alpha=0.5)
    ax2.legend()
    ax2.set_ylim(0, 1)

    line, = ax2.plot([], [], 'b-', linewidth=2)
    scatter, = ax2.plot([], [], 'ro', markersize=6)

    history = [initial_seg]
    generations = [0]
    current_grid = initial_grid.copy()
    stopped = False

    def init():
        im.set_array(current_grid)
        line.set_data(generations, history)
        scatter.set_data([0], [initial_seg])
        ax1.set_title(f'Поколение: 0  |  Терпимость: {tolerance:.2f}')
        ax2.set_xlim(0, 10)
        return im, line, scatter

    def update(frame):
        nonlocal current_grid, history, generations, stopped

        if stopped:
            return im, line, scatter

        moved = model.step()

        current_grid = model.grid.copy()
        current_seg = model.calculate_segregation_index()
        current_gen = model.generation

        history.append(current_seg)
        generations.append(current_gen)

        im.set_array(current_grid)

        if moved == 0:
            ax1.set_title(f' СТАБИЛИЗАЦИЯ! Поколение: {current_gen}  |  Сегрегация: {current_seg:.3f}',
                          color='green', fontsize=12)
            stopped = True
            ani.event_source.stop()
            print(f" Стабилизация на поколении {current_gen}")
            print(f" Финальная сегрегация: {current_seg:.3f}")
        else:
            ax1.set_title(f'Поколение: {current_gen}  |  Перемещений: {moved}  |  Сегрегация: {current_seg:.3f}')

        line.set_data(generations, history)
        scatter.set_data([current_gen], [current_seg])

        ax2.relim()
        ax2.autoscale_view(scalex=True, scaley=False)
        # Устанавливаем правую границу с запасом
        ax2.set_xlim(0, max(current_gen + 5, 10))
        # принудительная перерисовка
        fig.canvas.draw_idle()

        print(f"  Поколение {current_gen}: Перемещений = {moved}, Сегрегация = {current_seg:.3f}")
        print(f"  Ось X установлена до: {max(current_gen + 5, 10)}")  # ОТЛАДКА

        return im, line, scatter

    ani = FuncAnimation(fig, update,
                        init_func=init, blit=False, interval=200, repeat=False)

    plt.tight_layout()
    return ani, fig, model

def build_tolerance_dependency():
    """Построение графика зависимости сегрегации от терпимости"""

    tolerances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    final_segregations = []
    generations_to_stabilize = []

    print("\n")
    print("ПОСТРОЕНИЕ ЗАВИСИМОСТИ СЕГРЕГАЦИИ ОТ ТЕРПИМОСТИ")
    print("")

    for tolerance in tolerances:
        print(f"\nЗапуск с терпимостью {tolerance:.1f}...")

        model = SchellingModel(size=40, n_agents=1200, empty_ratio=0.2, tolerance=tolerance)

        # Запускаем до стабилизации
        for _ in range(200):
            moved = model.step()
            if moved == 0:
                break

        final_seg = model.calculate_segregation_index()
        final_segregations.append(final_seg)
        generations_to_stabilize.append(model.generation)

        print(f"  Поколений до стабилизации: {model.generation}")
        print(f"  Финальная сегрегация: {final_seg:.3f}")

    #создание фигуры с двумя графиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #график 1: Зависимость сегрегации от терпимости
    ax1.plot(tolerances, final_segregations, 'bo-', linewidth=2, markersize=10)
    ax1.set_xlabel('Терпимость (tolerance)', fontsize=12)
    ax1.set_ylabel('Финальный индекс сегрегации', fontsize=12)
    ax1.set_title('Зависимость сегрегации от уровня терпимости', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    #добавление подписи значений
    for i, (tol, seg) in enumerate(zip(tolerances, final_segregations)):
        ax1.annotate(f'{seg:.3f}', (tol, seg), xytext=(5, 5), textcoords='offset points', fontsize=10)

    #график 2: кол-во поколений до стабилизации
    ax2.plot(tolerances, generations_to_stabilize, 'rs-', linewidth=2, markersize=10)
    ax2.set_xlabel('Терпимость (tolerance)', fontsize=12)
    ax2.set_ylabel('Поколений до стабилизации', fontsize=12)
    ax2.set_title('Скорость стабилизации модели', fontsize=14)
    ax2.grid(True, alpha=0.3)

    #добавление подписи значений
    for i, (tol, gen) in enumerate(zip(tolerances, generations_to_stabilize)):
        ax2.annotate(f'{gen}', (tol, gen), xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    plt.savefig('segregation_vs_tolerance.png', dpi=150)
    plt.show()

    #сохранение результатов бенчмаркинга в текстовый файл
    save_benchmark_to_txt(tolerances, final_segregations, generations_to_stabilize, 'benchmark_results.txt')

    #вывод сводной таблицы
    print("\n")
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("-" * 60)
    print(f"{'Терпимость':<12} {'Финальная сегрегация':<22} {'Поколений':<12}")
    print("-" * 60)
    for tol, seg, gen in zip(tolerances, final_segregations, generations_to_stabilize):
        print(f"{tol:<12.1f} {seg:<22.3f} {gen:<12}")
    print("-" * 60)


#выбор терпимости
print("")
print("Модель сеградации Шеллинга")
print("")

print("\nВыберите режим работы:")
print("  1 - Запустить модель с выбором терпимости")
print("  2 - Построить график зависимости сегрегации от терпимости")
print("  0 - Выход")

mode = input("\nВаш выбор: ")

if mode == '0':
    print("Выход из программы...")
    exit()
elif mode == '2':
    build_tolerance_dependency()
    exit()
elif mode == '1':
    print("\nВыберите уровень терпимости:")
    print("  0.1 - Очень низкая (быстрая сегрегация)")
    print("  0.2 - Низкая")
    print("  0.3 - Средняя (классическая)")
    print("  0.4 - Выше среднего")
    print("  0.5 - Высокая")
    print("  0.6 - Очень высокая (долгая сегрегация)")
    print("  0.7 - Экстремально высокая (может не стабилизироваться)")

    #ввод терпимости
    while True:
        try:
            user_input = input("\nВведите значение терпимости (0.1 - 0.7): ")
            tolerance = float(user_input)

            if 0.0 < tolerance <= 0.7:
                break
            else:
                print(" Ошибка: введите число от 0.1 до 0.7")
        except ValueError:
            print(" Ошибка: введите число (например, 0.3)")

    print(f"\n Выбрана терпимость: {tolerance:.2f}")

    #запуск анимации
    ani, fig, model = create_animation_with_tolerance(tolerance)

    print("\n Запуск анимации (закройте окно для завершения)")
    plt.show()

    #финальная статистика
    print("\n")
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print(f"  Терпимость: {tolerance:.2f}")
    print(f"  Конечное поколение: {model.generation}")
    print(f"  Финальная сегрегация: {model.calculate_segregation_index():.3f}")
    print(f"  Всего перемещений: {sum(model.moves_history)}")
    print("")

    #сохранение результатов в текстовый файл
    save_option = input("\n Сохранить результаты в текстовый файл? (y/n): ")
    if save_option.lower() == 'y':
        filename = f'results_tolerance_{tolerance:.2f}.txt'
        save_results_to_txt(model, filename)

    save_png = input("\nСохранить финальный кадр как PNG? (y/n): ")
    if save_png.lower() == 'y':
        try:
            filename = f'schelling_final_tolerance_{tolerance:.2f}.png'

            fig.savefig(filename, dpi=150, bbox_inches='tight')

            print(f"Кадр сохранён как '{filename}'")
            print("   Открой файл в папке с проектом.")
        except Exception as e:
            print(f"Ошибка: {e}")
            print("Попробуй сохранить до закрытия окна.")