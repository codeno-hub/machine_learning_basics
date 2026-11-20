"""
Breadth First Search (BFS) graph traversal.

Uses a FIFO queue and white/gray/black vertex colouring (CLRS style).
For every vertex it records:
    d[v]  - discovery distance (number of edges) from the source
    pi[v] - predecessor of v in the breadth-first (shortest-path) tree

Colour meaning:
    WHITE - undiscovered
    GRAY  - discovered and enqueued, neighbours not fully processed yet
    BLACK - dequeued and all neighbours examined
"""

from collections import deque

WHITE, GRAY, BLACK = "white", "gray", "black"

# Default sample graph (adjacency list of an unweighted, undirected graph)
SAMPLE_GRAPH = {
    1: [2, 3],
    2: [1, 4, 5],
    3: [1, 6],
    4: [2],
    5: [2],
    6: [3],
}


def bfs(graph, source, verbose=False):
    """Run BFS from `source`.

    Returns (order, color, d, pi) where
        order - vertices in the order they were dequeued/processed
        color - final colour of every vertex
        d     - distance from source (None if unreachable)
        pi    - predecessor (None for the source and unreachable vertices)
    """
    if source not in graph:
        raise ValueError(f"Source vertex {source} is not in the graph")

    # Initialise every vertex: white, infinite (None) distance, no predecessor
    color = {v: WHITE for v in graph}
    d = {v: None for v in graph}
    pi = {v: None for v in graph}

    # Discover the source
    color[source] = GRAY
    d[source] = 0
    queue = deque([source])  # FIFO queue
    order = []
    iteration = 0

    while queue:
        iteration += 1
        u = queue.popleft()  # dequeue from the front (FIFO)
        order.append(u)

        if verbose:
            print(f"Iteration {iteration}: dequeue {u}")

        for v in graph[u]:
            if color[v] == WHITE:  # undiscovered neighbour
                color[v] = GRAY
                d[v] = d[u] + 1  # v.d = u.d + 1
                pi[v] = u  # v.pi = u
                queue.append(v)  # enqueue at the back
                if verbose:
                    print(f"    discover {v}: d={d[v]}, pi={u}, color=gray")

        color[u] = BLACK  # u is fully processed
        if verbose:
            print(f"    color[{u}] = black, queue = {list(queue)}")

    return order, color, d, pi


def print_results(source, order, color, d, pi):
    print("=" * 52)
    print(f"BFS RESULTS (source vertex = {source})")
    print("=" * 52)
    print("Traversal order :", " -> ".join(map(str, order)))
    print()
    print(f"{'Vertex':<8}{'Color':<8}{'Distance':<10}{'Predecessor'}")
    print("-" * 44)
    for v in sorted(color):
        dist = "inf" if d[v] is None else d[v]
        pred = "NIL" if pi[v] is None else pi[v]
        print(f"{v:<8}{color[v]:<8}{dist!s:<10}{pred}")


def main():
    source = 1
    print("Graph (adjacency list):")
    for v, nbrs in SAMPLE_GRAPH.items():
        print(f"  {v}: {', '.join(map(str, nbrs))}")
    print()

    order, color, d, pi = bfs(SAMPLE_GRAPH, source, verbose=True)
    print()
    print_results(source, order, color, d, pi)


if __name__ == "__main__":
    main()
