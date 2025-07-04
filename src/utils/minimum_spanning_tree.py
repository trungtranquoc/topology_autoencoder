import torch

def euclidean_distance(x: torch.Tensor, y: torch.Tensor):
    """
        Return euclidean distance between two data points
    """
    return (x - y).norm()

def minimum_spanning_tree(points: torch.Tensor, distance = euclidean_distance) -> list:
    """
        Computing the minimum spanning tree of a set of points. Return list of edges to be added.
    """
    n = len(points)

    # Step 1: Generate all possible edges with their distances
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(points[i], points[j])
            edges.append((dist, i, j))

    # Step 2: Sort edges by distance
    edges.sort()

    # Union-Find data structure for Kruskal's algorithm
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
            return True
        return False

    # Step 3: Kruskal's algorithm to build MST
    mst = []
    for dist, u, v in edges:
        if union(u, v):
            mst.append((u, v))
        if len(mst) == n - 1:
            break

    return mst