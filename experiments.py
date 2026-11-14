"""Эксперименты с алгоритмом Apriori на наборе данных baskets.csv.

Для каждого порога поддержки измеряется время работы алгоритма и число
найденных частых наборов каждой длины. Результаты сохраняются в папку results:
    experiments.csv      - сводная таблица;
    itemsets_XX.csv      - частые наборы и их поддержка для порога XX %;
    time.png             - диаграмма быстродействия;
    counts.png           - диаграмма количества частых наборов разной длины.

Запуск:  python experiments.py
"""

import csv
import os
import statistics
import time

import matplotlib.pyplot as plt

from apriori import apriori, load_transactions, sort_itemsets

DATASET = "baskets.csv"
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15]   # пороги поддержки
REPEATS = 20                                  # число запусков на каждый порог
RESULTS_DIR = "results"

# цвета диаграмм
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run_experiments(transactions):
    """Запускает Apriori для каждого порога поддержки и собирает результаты.

    Параметры:
        transactions (list[tuple]) - список транзакций.

    Возвращает:
        list[dict] - по одному словарю на каждый порог с ключами:
            min_support - порог поддержки (доля);
            time_ms     - медиана времени работы по REPEATS запускам, мс;
            total       - общее число частых наборов;
            by_length   - {длина набора: число частых наборов этой длины};
            candidates  - {k: число кандидатов C(k)};
            frequent    - {набор: поддержка}.
    """
    rows = []
    for min_support in THRESHOLDS:
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            frequent, candidates = apriori(transactions, min_support)
            times.append(time.perf_counter() - start)

        # количество частых наборов каждой длины: {длина: количество}
        by_length = {}
        for itemset in frequent:
            by_length[len(itemset)] = by_length.get(len(itemset), 0) + 1

        rows.append({
            "min_support": min_support,
            "time_ms": round(statistics.median(times) * 1000, 1),   # медиана, мс
            "total": len(frequent),
            "by_length": by_length,
            "candidates": candidates,
            "frequent": frequent,
        })
    return rows


def save_tables(rows, max_length):
    """Сохраняет в папку результатов сводную таблицу и списки частых наборов.

    Параметры:
        rows (list[dict]) - результаты экспериментов (см. run_experiments);
        max_length (int) - наибольшая длина найденного частого набора.

    Возвращаемого значения нет.
    """
    path = os.path.join(RESULTS_DIR, "experiments.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        header = ["порог, %", "время, мс", "всего наборов"]
        header += [f"наборов длины {k}" for k in range(1, max_length + 1)]
        header += ["кандидатов C2", "кандидатов C3", "кандидатов C4"]
        writer.writerow(header)
        for row in rows:
            line = [f"{row['min_support'] * 100:g}", f"{row['time_ms']:.1f}", row["total"]]
            line += [row["by_length"].get(k, 0) for k in range(1, max_length + 1)]
            line += [row["candidates"].get(k, 0) for k in (2, 3, 4)]
            writer.writerow(line)

    for row in rows:
        percent = round(row["min_support"] * 100)
        path = os.path.join(RESULTS_DIR, f"itemsets_{percent:02d}.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["набор", "длина", "поддержка"])
            for itemset, support in sort_itemsets(row["frequent"], "support"):
                writer.writerow([", ".join(itemset), len(itemset), f"{support:.4f}"])


def style_axes(ax):
    """Общее оформление диаграммы: без рамки, только горизонтальная сетка.

    Параметры:
        ax (matplotlib.axes.Axes) - оси диаграммы, которые нужно оформить.

    Возвращаемого значения нет.
    """
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_time(rows):
    """Диаграмма 1: время работы при разных порогах поддержки (time.png).

    Параметры:
        rows (list[dict]) - результаты экспериментов (см. run_experiments).

    Возвращаемого значения нет, диаграмма сохраняется в файл.
    """
    labels = [f"{row['min_support'] * 100:g} %" for row in rows]
    values = [row["time_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(labels, values, color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    for x, value in enumerate(values):
        ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=11, color=TEXT)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_xlabel("Порог поддержки", fontsize=11, color=MUTED)
    ax.set_ylabel("Время работы, мс", fontsize=11, color=MUTED)
    ax.set_title("Время работы Apriori при разных порогах поддержки",
                 fontsize=13, color=TEXT, loc="left", pad=14)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "time.png"), dpi=200, facecolor="white")
    plt.close(fig)


def plot_counts(rows, max_length):
    """Диаграмма 2: количество частых наборов каждой длины (counts.png).

    Параметры:
        rows (list[dict]) - результаты экспериментов (см. run_experiments);
        max_length (int) - наибольшая длина найденного частого набора.

    Возвращаемого значения нет, диаграмма сохраняется в файл.
    """
    labels = [f"{row['min_support'] * 100:g} %" for row in rows]
    colors = [BLUE, ORANGE, AQUA, "#eda100"]
    width = 0.8 / max_length                     # ширина одного столбца в группе

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in range(1, max_length + 1):
        values = [row["by_length"].get(k, 0) for row in rows]
        # смещение столбца длины k относительно центра группы
        offset = (k - 1 - (max_length - 1) / 2) * width
        positions = [x + offset for x in range(len(rows))]
        ax.bar(positions, values, width=width * 0.92, color=colors[k - 1],
               label=f"{k}-наборы")
        for x, value in zip(positions, values):
            ax.annotate(str(value), (x, value), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=10, color=TEXT if value else MUTED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Порог поддержки", fontsize=11, color=MUTED)
    ax.set_ylabel("Количество частых наборов", fontsize=11, color=MUTED)
    ax.set_title("Количество частых наборов разной длины",
                 fontsize=13, color=TEXT, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=11, labelcolor=TEXT)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "counts.png"), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    """Точка входа: проводит эксперименты, выводит таблицу на экран,
    сохраняет таблицы и диаграммы. Параметров и возвращаемого значения нет.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    transactions = load_transactions(DATASET)
    print(f"Транзакций: {len(transactions)}")

    rows = run_experiments(transactions)
    max_length = max(max(row["by_length"]) for row in rows)

    print(f"{'Порог':>6} {'Время, мс':>10} {'Всего':>6}  По длинам")
    for row in rows:
        lengths = ", ".join(f"{k}: {row['by_length'].get(k, 0)}"
                            for k in range(1, max_length + 1))
        print(f"{row['min_support'] * 100:>5g}% {row['time_ms']:>10.1f} "
              f"{row['total']:>6}  {lengths}")

    save_tables(rows, max_length)
    plot_time(rows)
    plot_counts(rows, max_length)
    print(f"Таблицы и диаграммы сохранены в папку {RESULTS_DIR}")


if __name__ == "__main__":
    main()
