# Breadth-First Search (BFS) Graph Traversal

A Python implementation of the Breadth-First Search (BFS) graph traversal algorithm using a First-In, First-Out (FIFO) queue. The script traverses an unweighted graph layer by layer, tracking vertex discovery states via a three-color coding mechanism (`WHITE`, `GRAY`, `BLACK`) while computing shortest-path distances and predecessor trees from a given source node.

## Features

- **FIFO Queue Progression:** Ensures strict level-by-level (breadth-first) traversal across unweighted graph layers.
- **Three-Color State Tracking:**
  - `WHITE`: Unvisited vertex.
  - `GRAY`: Discovered and currently enqueued vertex.
  - `BLACK`: Fully processed vertex with all adjacent neighbors explored.
- **Shortest-Path Tree Metrics:** Calculates the minimum edge distance ($u.d + 1$) and predecessor node ($\pi$) for every reachable vertex relative to the source.
- **Execution Trace:** Includes a manual dry-run log detailing queue states across the initial iterations.

## Graph Representation

The default implementation runs on the following unweighted adjacency list:

| Source Vertex | Neighbor Vertices |
| :--- | :--- |
| `1` | `2, 3` |
| `2` | `1, 4, 5` |
| `3` | `1, 6` |
| `4` | `2` |
| `5` | `2` |
| `6` | `3` |

## Repository Structure

```text
├── bfs_traversal.py   # Core graph representation and BFS algorithm implementation
├── bfs_dry_run.txt    # Step-by-step manual queue state trace for the first three iterations
├── bfs_output.txt     # Console output showing traversal order, distances, and predecessors
└── README.md          # Project documentation
