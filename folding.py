import random
import copy
from math import pi, sin, cos

from PIL import Image
import numpy as np

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

    def reflect(self, point):
        b2 = self.b * self.b
        a2 = self.a * self.a
        x = point.x
        y = point.y
        return Point((x * (b2 - a2) - 2 * self.a * (self.b * y + self.c)) / (a2 + b2),
                     (y * (a2 - b2) - 2 * self.b * (self.a * x + self.c)) / (a2 + b2))

class Bild:
    def __init__(self, image):
        """Constructor from a PIL.Image in L (grayscale) format
        """
        self.foldcount = 0
        self.w, self.h = image.size
        self.pixels = np.zeros((self.h, self.w), dtype = np.uint8)
        self.nonzero = []
        for i in range(self.h):
            for j in range(self.w):
                self.pixels[i, j] = image.getpixel((j, i))
                if self.pixels[i, j] != 0:
                    self.nonzero.append(Point(i, j))

    def copy(self):
        result = Bild(Image.new('L', (0, 0)))
        result.foldcount = self.foldcount
        result.w = self.w
        result.h = self.h
        result.nonzero = self.nonzero.copy()
        result.pixels = self.pixels.copy()
        return result

    def fold(self, threshold):
        """Folds along a randomly generated axis

        Args:
            threshold: threshold for the ratio between two folding parts
        """
        attempts = 100
        while attempts:
            center = random.choice(self.nonzero)
            axis = Line(center)
            part1 = []
            part2 = []
            for point in self.nonzero:
                if axis.eval(point) < 0:
                    part1.append(point)
                else:
                    part2.append(point)
            if len(part1) > len(part2):
                part1, part2 = part2, part1
            if len(part1) / len(part2) >= threshold:
                # perform folding
                folded = []
                self.nonzero.clear()
                for point in part1:
                    reflected = axis.reflect(point)
                    reflected.round()
                    if 0 <= reflected.x < self.h and 0 <= reflected.y < self.w:
                        folded.append(reflected)
                        value = self.pixels[point.x, point.y]    
                        self.pixels[reflected.x, reflected.y] += value
                        self.pixels[point.x, point.y] -= value
                    else:
                        print("Warning! Folded out of the frame")
                for point in list(set(part1 + part2 + folded)):
                    if self.pixels[point.x, point.y] > 0:
                        self.nonzero.append(point)

                self.foldcount += 1
                return True
            attempts -= 1

        return False

    def center(self):
        deltax = (self.h - self.x_min() - self.x_max() - 1) // 2
        deltay = (self.w - self.y_min() - self.y_max() - 1) // 2

        self.pixels = np.roll(self.pixels, deltax, axis=0)
        self.pixels = np.roll(self.pixels, deltay, axis=1)

        self.nonzero = [Point(p.x + deltax, p.y + deltay) for p in self.nonzero]

    def x_min(self):
        return min(self.nonzero, key=lambda p: p.x).x

    def y_min(self):
        return min(self.nonzero, key=lambda p: p.y).y

    def x_max(self):
        return max(self.nonzero, key=lambda p: p.x).x

    def y_max(self):
        return max(self.nonzero, key=lambda p: p.y).y


seed = Bild(Image.open("input/unfolded.png")
                 .resize((100, 100), resample=Image.NEAREST)
                 .convert('L'))

seed.center()
h = seed.x_max() - seed.x_min() + 1
w = seed.y_max() - seed.y_min() + 1

data = [seed]

np.set_printoptions(threshold=np.inf)

for i in range(100):
    while True:
        seed = random.choice(data)
        if seed.foldcount < 4:
            break
    image = seed.copy()
    image.fold(0.2)
    image.center()
    data.append(image)

    h = max(h, image.x_max() - image.x_min() + 1)
    w = max(w, image.y_max() - image.y_min() + 1)

for i in range(len(data)):
    image = data[i]
    saved = Image.fromarray(image.pixels, 'L').crop(((image.w - w) // 2,
                                                     (image.h - h) // 2,
                                                     image.w - (image.w - w + 1) // 2,
                                                     image.h - (image.h - h + 1) // 2))
    saved.save("output/image%d.png" % i)

print(len(data))
