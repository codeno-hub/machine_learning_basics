# Sorting Algorithms Performance Comparison

A Python benchmarking suite implementing and comparing five foundational sorting algorithms across multiple input distributions and array sizes.

## Overview

This project evaluates the empirical runtime performance of quadratic ($O(n^2)$) and linearithmic ($O(n \log n)$) sorting algorithms against Python's built-in `sorted()` (Timsort) for correctness verification.

### Implemented Algorithms
* **Selection Sort** – $O(n^2)$ time complexity across all cases; minimal swaps.
* **Bubble Sort** – $O(n^2)$ average/worst case; $O(n)$ best case with early-exit optimization.
* **Insertion Sort** – $O(n^2)$ average/worst case; $O(n)$ on sorted/nearly sorted data.
* **Merge Sort** – Guaranteed $O(n \log n)$ time complexity; $O(n)$ auxiliary space.
* **Quick Sort** – $O(n \log n)$ average time complexity; in-place partitioning.

## Project Structure

```text
├── sorting_comparison.py   # Algorithm implementations, test generators, and timer logic
├── sorting_report.txt      # Generated benchmark output
└── README.md
