import heapq

def create_kd_tree(points, dim, depth=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[depth])
        depth = (depth + 1) % dim
        half = len(points) / 2
        return (
            create_kd_tree(points[: half], dim, depth),
            create_kd_tree(points[half + 1:], dim, depth),
            points[half])
    elif len(points) == 1:
        return (None, None, points[0])

# K nearest neighbors. The heap is a bounded priority queue.
def naive_knn(kd_node, point, k, dim, dist_func, return_distances=True, depth=0, heap=None):
    root_or_not = not heap
    if root_or_not:
        heap = list()
    if kd_node:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][depth] - point[depth]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        depth = (depth + 1) % dim
        # traverse all node in k-d tree
        naive_knn(kd_node[0], point, k, dim, dist_func, return_distances, depth, heap)
        naive_knn(kd_node[1], point, k, dim, dist_func, return_distances, depth, heap)
    # After traverse all the candidates, back to root
    if root_or_not:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors





