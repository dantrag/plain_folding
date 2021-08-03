import random
from math import pi, sin, cos
from collections import deque

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __copy__(self):
        return Point(self.x, self.y)

    def __deepcopy__(self):
        return self.copy()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str((self.x, self.y))

    def round(self):
        self.x = int(round(self.x))
        self.y = int(round(self.y))

class Line:
    def __init__(self, p1, p2 = None):
        """Constructor from two points or a point and a random angle
        """
        if p2 is None:
            # construct the second point from a random angle
            angle = random.random() * pi
            p2 = Point(p1.x + sin(angle),
                       p1.y + cos(angle))

        self.a = 1
        if p1.x == p2.x:
            self.b = 0
            self.c = -p1.x
        else:
            self.b = (p1.x - p2.x) / (p2.y - p1.y)
            self.c = -p1.x - self.b * p1.y

    def eval(self, point):
        return self.a * point.x + self.b * point.y + self.c

    def reflect(self, point, rounded=True):
        b2 = self.b * self.b
        a2 = self.a * self.a
        x = point.x
        y = point.y
        reflected =  Point((x * (b2 - a2) - 2 * self.a * (self.b * y + self.c)) / (a2 + b2),
                           (y * (a2 - b2) - 2 * self.b * (self.a * x + self.c)) / (a2 + b2))
        if rounded:
            reflected.round()
        return reflected

def graph_from_points(points: set):
    graph = {}
    for p in points:
        graph[p] = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = Point(p.x + dx, p.y + dy)
            if neighbour in points:
                graph[p].append(neighbour)
    return graph

def move_points_connected(points: set, mapping):
    original_graph = graph_from_points(points)
    graph = {}

    for p in points:
        graph[mapping(p)] = [mapping(x) for x in original_graph[p]]

    for p in graph.keys():
        for neighbour in graph[p]:
            distance = max(abs(neighbour.x - p.x), abs(neighbour.y - p.y))
            if distance > 1:
                # need to fill the gap
                dx = (neighbour.x - p.x) / distance
                dy = (neighbour.y - p.y) / distance
                for i in range(1, distance):
                    new_x = int(round(p.x + dx * i))
                    new_y = int(round(p.y + dy * i))
                    new_p = Point(new_x, new_y)
                    if not new_p in graph:
                        graph[new_p] = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbour = Point(new_p.x + dx, new_p.y + dy)
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
            neighbour = Point(point.x + dx, point.y + dy)
            if not neighbour in visited:
                if not unless_what(neighbour):
                    visited.add(neighbour)
                    q.append(neighbour)

    return visited

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
                neighbour = Point(p.x + dx, p.y + dy)
                if not neighbour in points:
                    # this is a borderline point
                    border_points.add(p)

    return split_into_components(border_points)

