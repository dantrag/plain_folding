import random
from math import pi, sin, cos
from collections import deque

def points_closer_than(point1, point2, distance):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 < distance ** 2

class Line:
    def __init__(self, p1, p2 = None):
        """Constructor from two points or a point and a random angle
        """
        if p2 is None:
            # construct the second point from a random angle
            angle = random.random() * pi
            p2 = (p1[0] + sin(angle),
                  p1[1] + cos(angle))

        if p1[1] == p2[1]:
            self.a = 0
            self.b = 1
            self.c = -p1[1]
        else:
            self.a = 1
            self.b = (p1[0] - p2[0]) / (p2[1] - p1[1])
            self.c = -p1[0] - self.b * p1[1]

    def eval(self, point):
        return self.a * point[0] + self.b * point[1] + self.c

    def reflect(self, point, rounded=True):
        b2 = self.b * self.b
        a2 = self.a * self.a
        x = point[0]
        y = point[1]
        reflected =  ((x * (b2 - a2) - 2 * self.a * (self.b * y + self.c)) / (a2 + b2),
                      (y * (a2 - b2) - 2 * self.b * (self.a * x + self.c)) / (a2 + b2))
        if rounded:
            reflected = (round(reflected[0]),
                         round(reflected[1]))
        return reflected

    def closer_than(self, point, distance):
        return (self.a * point[0] + self.b * point[1] + self.c) ** 2 / \
               (self.a ** 2 + self.b ** 2) < distance ** 2

# Cubic Bezier curve wrapper class
class Bezier:
    def __init__(self, P0, P1, P2, P3):
        """Constructor from four control points
        """
        self.p0 = P0
        self.p1 = P1
        self.p2 = P2
        self.p3 = P3

    def evaluate(self, t: float, rounded=True):
        """Evaluates the curve at a parameter t in [0, 1]
        
        point = self.p0 * (1 - t) ** 3 + \
                self.p1 * (1 - t) ** 2 * 3 * t + \
                self.p2 * (1 - t) * 3 * t ** 2 + \
                self.p3 * t ** 3
        """
        point = (self.p0[0] * (1 - t) ** 3 + \
                 self.p1[0] * (1 - t) ** 2 * 3 * t + \
                 self.p2[0] * (1 - t) * 3 * t ** 2 + \
                 self.p3[0] * t ** 3,
                 self.p0[1] * (1 - t) ** 3 + \
                 self.p1[1] * (1 - t) ** 2 * 3 * t + \
                 self.p2[1] * (1 - t) * 3 * t ** 2 + \
                 self.p3[1] * t ** 3)
        if rounded:
            point = (round(point[0]),
                     round(point[1]))
        return point

def graph_from_points(points: set):
    graph = {}
    for p in points:
        graph[p] = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = (p[0] + dx, p[1] + dy)
            if neighbour in points:
                graph[p].append(neighbour)
    return graph

def dfs(graph: dict, start, shuffle_neigbours=False):
    depths = {}

    def dfs_recursion(current, depth=0, previous=None):
        depths[current] = depth
        neighbours = graph[start]
        if shuffle_neigbours:
            neighbours = random.shuffle(neighbours)
            
        for neighbour in neighbours:
            if neighbour != previous:
                dfs_recursion(neighbour, depth=depth + 1, previous=neighbour)
    
    dfs_recursion(start)

    return depths
    

def move_points_connected(points: set, mapping):
    original_graph = graph_from_points(points)
    graph = {}

    for p in points:
        graph[mapping(p)] = [mapping(x) for x in original_graph[p]]

    for p in graph.keys():
        for neighbour in graph[p]:
            distance = max(abs(neighbour[0] - p[0]), abs(neighbour[1] - p[1]))
            if distance > 1:
                # need to fill the gap
                dx = (neighbour[0] - p[0]) / distance
                dy = (neighbour[1] - p[1]) / distance
                for i in range(1, distance):
                    new_x = int(round(p[0] + dx * i))
                    new_y = int(round(p[1] + dy * i))
                    new_p = (new_x, new_y)
                    if not new_p in graph:
                        graph[new_p] = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbour = (new_p[0] + dx, new_p[1] + dy)
                        if neighbour in graph:
                            graph[new_p].append(neighbour)
    return graph.keys()

def bfs_unless(starts, unless_what):
    q = deque()
    for start in starts:
        q.append(start)
    visited = set(starts)

    while q:
        point = q.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = (point[0] + dx, point[1] + dy)
            if not neighbour in visited:
                if not unless_what(neighbour):
                    visited.add(neighbour)
                    q.append(neighbour)

    return visited

def bfs_distances(graph, starts):
    q = deque()
    distances = {}
    for start in starts:
        q.append(start)
        distances[start] = 0

    while q:
        point = q.popleft()
        for neighbour in graph[point]:
            if not neighbour in distances:
                distances[neighbour] = distances[point] + 1
                q.append(neighbour)

    return distances

def split_into_components(points: set, sort_by_size=True):
    graph = graph_from_points(points)

    components = []
    visited = set()
    for p in points:
        if not p in visited:
            component = []
            stack = [p]
            visited.add(p)

            while stack:
                v = stack.pop()
                component.append(v)
                for neighbour in graph[v]:
                    if not neighbour in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)

            components.append(component)

    if sort_by_size:
        components.sort(key=lambda x: -len(x))
    return components


def extract_contours(points: set):
    if not points:
        return []

    border_points = set()
    for p in points:
        for dx in {-1, 0, 1}:
            for dy in {-1, 0, 1}:
                neighbour = (p[0] + dx, p[1] + dy)
                if not neighbour in points:
                    # this is a borderline point
                    border_points.add(p)

    return split_into_components(border_points)

