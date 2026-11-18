"""Sorting Algorithms Performance Comparison

Implements Selection, Bubble, Insertion, Merge and Quick sort and compares
their running time on several input sizes and input orderings.

Run:  python sorting_comparison.py
"""

import random
import sys
import time

sys.setrecursionlimit(10_000)


# --------------------------------------------------------------------------
# Algorithms (each returns a NEW sorted list; the input is never modified)
# --------------------------------------------------------------------------

def selection_sort(data):
    """O(n^2) always. Fewest swaps (at most n-1)."""
    a = list(data)
    n = len(a)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
    return a


def bubble_sort(data):
    """O(n^2) average/worst, O(n) best (already sorted) thanks to early exit."""
    a = list(data)
    n = len(a)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def insertion_sort(data):
    """O(n^2) average/worst, O(n) best. Very fast on nearly sorted data."""
    a = list(data)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def merge_sort(data):
    """O(n log n) always. Stable, uses O(n) extra memory."""
    a = list(data)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(data):
    """O(n log n) average, O(n^2) worst. In-place, median-of-three pivot
    and an explicit stack so sorted/reversed input does not blow the
    recursion limit or degrade to quadratic time."""
    a = list(data)
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        # median-of-three pivot selection
        mid = (lo + hi) // 2
        if a[mid] < a[lo]:
            a[mid], a[lo] = a[lo], a[mid]
        if a[hi] < a[lo]:
            a[hi], a[lo] = a[lo], a[hi]
        if a[hi] < a[mid]:
            a[hi], a[mid] = a[mid], a[hi]
        pivot = a[mid]

        # Hoare-style partition
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1

        # push the larger part first so the stack stays O(log n)
        if (j - lo) > (hi - i):
            stack.append((lo, j))
            stack.append((i, hi))
        else:
            stack.append((i, hi))
            stack.append((lo, j))
    return a


ALGORITHMS = {
    "Selection": selection_sort,
    "Bubble": bubble_sort,
    "Insertion": insertion_sort,
    "Merge": merge_sort,
    "Quick": quick_sort,
}


# --------------------------------------------------------------------------
# Input generators
# --------------------------------------------------------------------------

def make_random(n):
    return [random.randint(0, n * 10) for _ in range(n)]


def make_sorted(n):
    return list(range(n))


def make_reversed(n):
    return list(range(n, 0, -1))


def make_nearly_sorted(n):
    a = list(range(n))
    for _ in range(max(1, n // 50)):          # swap ~2% of the elements
        i, j = random.randrange(n), random.randrange(n)
        a[i], a[j] = a[j], a[i]
    return a


SCENARIOS = {
    "Random": make_random,
    "Sorted": make_sorted,
    "Reversed": make_reversed,
    "Nearly sorted": make_nearly_sorted,
}


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def time_once(func, data):
    start = time.perf_counter()
    result = func(data)
    return time.perf_counter() - start, result


def best_of(func, data, repeats):
    best = float("inf")
    for _ in range(repeats):
        elapsed, result = time_once(func, data)
        best = min(best, elapsed)
    return best, result


def verify():
    """Check every algorithm against Python's built-in sorted()."""
    random.seed(0)
    cases = [[], [1], [2, 1], [3, 3, 3], [5, -1, 4, -1, 0]]
    cases += [make_random(random.randint(2, 200)) for _ in range(200)]
    for name, func in ALGORITHMS.items():
        for case in cases:
            assert func(case) == sorted(case), f"{name} sort failed on {case}"
    print("Correctness check passed: all five algorithms match sorted().\n")


def run_benchmark(sizes, repeats=3, quadratic_limit=5000):
    random.seed(42)
    results = {}   # (scenario, n) -> {algo: seconds or None}

    for scenario, gen in SCENARIOS.items():
        for n in sizes:
            data = gen(n)
            expected = sorted(data)
            row = {}
            for name, func in ALGORITHMS.items():
                quadratic = name in ("Selection", "Bubble", "Insertion")
                if quadratic and n > quadratic_limit:
                    row[name] = None          # skipped: would take too long
                    continue
                secs, out = best_of(func, data, repeats)
                assert out == expected, f"{name} gave a wrong result"
                row[name] = secs
            results[(scenario, n)] = row
    return results


def print_report(results, sizes):
    names = list(ALGORITHMS)
    for scenario in SCENARIOS:
        print(f"=== {scenario} input (time in milliseconds, best of 3) ===")
        print(f"{'n':>8} | " + " | ".join(f"{h:>10}" for h in names))
        print("-" * (11 + 13 * len(names)))
        for n in sizes:
            row = results[(scenario, n)]
            cells = []
            for name in names:
                v = row[name]
                cells.append(f"{'--':>10}" if v is None else f"{v * 1000:>10.2f}")
            print(f"{n:>8} | " + " | ".join(cells))
        print()
    print("'--' = skipped (O(n^2) algorithm on a large input).\n")


def print_summary(results, sizes):
    n = max(s for s in sizes if all(
        results[("Random", s)][a] is not None for a in ALGORITHMS))
    row = results[("Random", n)]
    fastest = min(row.values())
    print(f"Relative speed on random data, n = {n} (1.0x = fastest):")
    for name, secs in sorted(row.items(), key=lambda kv: kv[1]):
        print(f"  {name:<10} {secs / fastest:>8.1f}x")
    print()


def plot_results(results, sizes, filename="sorting_chart.png"):
    """Save a 2x2 chart (one panel per input type). Needs matplotlib;
    if it is not installed the program still runs, just without the chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")            # no window needed, just save a file
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping chart "
              "(install it with: pip install matplotlib)")
        return

    styles = {
        "Selection": ("#d95f02", "s"),
        "Bubble": ("#e7298a", "^"),
        "Insertion": ("#7570b3", "D"),
        "Merge": ("#1b9e77", "o"),
        "Quick": ("#1f78b4", "v"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        for name, (color, marker) in styles.items():
            xs, ys = [], []
            for n in sizes:
                v = results[(scenario, n)][name]
                if v is not None:
                    xs.append(n)
                    ys.append(max(v * 1000, 0.001))   # ms; avoid log(0)
            ax.plot(xs, ys, marker=marker, color=color, label=name, linewidth=1.8)
        ax.set_title(f"{scenario} input")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("Number of elements (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time in ms (lower is better)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Sorting algorithm running time vs input size (log-log scale)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {filename}")


if __name__ == "__main__":
    SIZES = [100, 500, 1000, 2000, 5000, 20000]
    verify()
    res = run_benchmark(SIZES)
    print_report(res, SIZES)
    print_summary(res, SIZES)
    plot_results(res, SIZES)
