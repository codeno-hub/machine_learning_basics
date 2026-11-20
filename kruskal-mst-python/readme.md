# Kruskal's Minimum Spanning Tree (MST)

A Python implementation of Kruskal's greedy algorithm to compute the Minimum Spanning Tree (MST) of a weighted, undirected graph. The script sorts all edges in ascending order of weight and iteratively selects non-cyclic edges using a cycle-detection helper to connect all vertices at the lowest possible cumulative path cost.

## Features

- **Ascending Edge Sorting:** Orders all graph edges by weight magnitude prior to evaluating connections.
- **Cycle Detection Helper:** Prevents closed path cycles (such as rejecting edge `B — E`) when building the spanning tree.
- **Step-by-Step Execution Log:** Outputs the chronological sequence of accepted edges, rejected cyclic edges, and the final cumulative MST weight.
- **Visual Diagram Included:** Accompanied by a node-edge layout PDF (`mst_structure.pdf`) illustrating the final spanning tree topology.

## Graph Representation

The default implementation runs on the following weighted, undirected edge list:

| Edge Connection | Edge Weight (Magnitude) |
| :--- | :--- |
| `D — E` | `1` |
| `A — D` | `2` |
| `C — F` | `2` |
| `A — B` | `3` |
| `B — E` | `4` *(Rejected — forms a cycle)* |
| `B — C` | `5` |
| `E — F` | `6` *(Rejected — forms a cycle)* |

## Repository Structure

```text
├── kruskal_mst.py      # Edge sorting, cycle detection, and MST computation logic
├── kruskal_output.txt  # Terminal output showing accepted/rejected edges and total cost
├── mst_structure.pdf   # Visual node-edge diagram of the resulting Minimum Spanning Tree
└── README.md           # Project documentation
