"""
Kruskal's Minimum Spanning Tree algorithm.

Steps:
    1. Sort all edges in ascending order of weight.
    2. Walk through the sorted edges. For each edge (u, v):
         - if u and v are in different components -> accept the edge
           and merge the components (union)
         - otherwise the edge would form a cycle -> reject it
    3. Stop when V - 1 edges have been accepted.

Cycle detection uses a Disjoint Set Union (union-find) helper with
path compression and union by rank.
"""


class DisjointSet:
    """Union-find structure used to detect cycles."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        # Path compression: point nodes directly at the root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """Merge the sets of a and b.
        Returns False if they were already in the same set (cycle)."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        # Union by rank
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


# Default sample dataset: (u, v, weight)
SAMPLE_EDGES = [
    ("D", "E", 1),
    ("A", "D", 2),
    ("C", "F", 2),
    ("A", "B", 3),
    ("B", "E", 4),
    ("B", "C", 5),
    ("E", "F", 6),
]


def kruskal(edges):
    """Return (mst_edges, rejected_edges, total_weight, sorted_edges, log)."""
    vertices = sorted({v for u, w, _ in edges for v in (u, w)})

    # Step 1: ascending sort by weight (sorted() is stable, so equal-weight
    # edges keep their original input order)
    sorted_edges = sorted(edges, key=lambda e: e[2])

    ds = DisjointSet(vertices)
    mst, rejected, log = [], [], []
    total = 0

    # Step 2: pick edges that do not create a cycle
    for u, v, w in sorted_edges:
        if ds.union(u, v):
            mst.append((u, v, w))
            total += w
            log.append(("ACCEPT", u, v, w))
        else:
            rejected.append((u, v, w))
            log.append(("REJECT", u, v, w))
        if len(mst) == len(vertices) - 1:
            # Remaining edges (if any) are not needed; record them as skipped
            idx = sorted_edges.index((u, v, w)) + 1
            for ru, rv, rw in sorted_edges[idx:]:
                rejected.append((ru, rv, rw))
                log.append(("REJECT", ru, rv, rw))
            break

    return mst, rejected, total, sorted_edges, log


def main():
    mst, rejected, total, sorted_edges, log = kruskal(SAMPLE_EDGES)

    print("=" * 56)
    print("KRUSKAL'S MINIMUM SPANNING TREE")
    print("=" * 56)

    print("\nEdges sorted by ascending weight:")
    for u, v, w in sorted_edges:
        print(f"  {u} - {v}   weight {w}")

    print("\nChronological processing:")
    for step, (status, u, v, w) in enumerate(log, 1):
        note = "" if status == "ACCEPT" else "  (forms a cycle)"
        print(f"  {step}. {status:<7} {u} - {v}  (w = {w}){note}")

    print("\nAccepted edges (MST):")
    for u, v, w in mst:
        print(f"  {u} - {v}  weight {w}")

    print("\nRejected edges (cycle-forming):")
    for u, v, w in rejected:
        print(f"  {u} - {v}  weight {w}")

    expr = " + ".join(str(w) for _, _, w in mst)
    print(f"\nMST total weight = {expr} = {total}")


if __name__ == "__main__":
    main()
