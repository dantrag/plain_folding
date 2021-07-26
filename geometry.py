import random
from math import pi, sin, cos

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

