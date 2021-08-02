import random
from math import pi, sin, cos

from PIL import Image
import numpy as np

from geometry import Point, Line, extract_contours, split_into_components, move_points_connected, bfs_unless

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

    def fold(self, threshold, xy_axis_bias):
        """Folds along a randomly generated axis

        Args:
            threshold: threshold for the ratio between two folding parts
            xy_axis_bias: probability of a fold being along X or Y axis
        """
        attempts = 100
        while attempts:
            center = random.choice(self.nonzero)
            xy_aligned = random.random() <= xy_axis_bias
            if xy_aligned:
                if random.random() <= 0.5:
                    axis = Line(center, Point(center.x + 1, center.y))
                else:
                    axis = Line(center, Point(center.x, center.y + 1))
            else:
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
            if len(part1) / len(self.nonzero) >= threshold:
                # fold each connected component separately
                pieces = split_into_components(set(part1))
                folded = []
                updates = {}
                stationary_points = set()

                for piece in pieces:
                    points = set(piece)
                    extracted_contours = extract_contours(points)
                    if not extracted_contours:
                        continue
                    contour = set(extracted_contours[0])
                    reflected_contour = set(move_points_connected(contour,
                                                                  mapping=lambda p: axis.reflect(p)))
                    reflected_interior = set()

                    forward_map = {}
                    backward_map = {}
                    for point in piece:
                        forward_map[point] = axis.reflect(point)
                        if not forward_map[point] in backward_map:
                            backward_map[forward_map[point]] = []
                        backward_map[forward_map[point]].append(point)


                    if len(backward_map) > len(reflected_contour):
                        # start from points inside reflected contour to fill it
                        seeds = []
                        for point in piece:
                            reflected = axis.reflect(point)
                            if not reflected in reflected_contour:
                                seeds.append(reflected)
                       
                        # find pixels inside reflected contour
                        reflected_interior = bfs_unless(seeds, lambda p: p in reflected_contour or \
                                                                         p.x < 0 or p.x >= self.h or \
                                                                         p.y < 0 or p.y >= self.w)

                    for point in list(reflected_interior.union(reflected_contour)):
                        value = 0
                        folded.append(point)
                        if not point in backward_map:
                            original_values = []
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                neighbour = Point(point.x + dx, point.y + dy)
                                if neighbour in backward_map:
                                    for p in backward_map[neighbour]:
                                        original_values.append(self.pixels[p.x, p.y])
                            if not original_values:
                                continue
                            else:
                                value = max(set(original_values), key=original_values.count)
                        else:
                            original_values = [self.pixels[p.x, p.y] for p in backward_map[point]]
                            value = max(set(original_values), key=original_values.count)

                        updates[point] = value
                        if point in backward_map:
                            if point in backward_map[point]:
                                stationary_points.add(point)
                            else:
                                for p in backward_map[point]:
                                    updates[p] = -int(self.pixels[p.x, p.y])

                self.nonzero.clear()
                for point in list(set(part1 + part2 + folded)):
                    if point in updates:
                        if not point in stationary_points:
                            self.pixels[point.x, point.y] += updates[point]
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


def performe_folding(input_img, num_examples, max_fold_count, min_fold_area, xy_folding_bias=0.0):
    seed = Bild(Image.open(input_img)
                     .resize((200, 200), resample=Image.NEAREST)
                     .convert('L'))

    seed.center()
    h = seed.x_max() - seed.x_min() + 1
    w = seed.y_max() - seed.y_min() + 1

    data = [seed]

    for i in range(num_examples):
        while True:
            seed = random.choice(data)
            if seed.foldcount < max_fold_count:
                break
        image = seed.copy()
        image.fold(threshold=min_fold_area, xy_axis_bias=xy_folding_bias)
        image.center()
        data.append(image)

        h = max(h, image.x_max() - image.x_min() + 1)
        w = max(w, image.y_max() - image.y_min() + 1)
        print(".", end='', flush=True)

    size = max(h, w)
    h = 1 << (size - 1).bit_length()
    w = h

    fin_data=[]
    for i in range(len(data)):
        image = data[i]
        saved = Image.fromarray(image.pixels, 'L').crop(((image.w - w) // 2,
                                                         (image.h - h) // 2,
                                                         image.w - (image.w - w + 1) // 2,
                                                         image.h - (image.h - h + 1) // 2))
        saved.save("output/image%d.png" % i)
        fin_data.append(np.array(saved))

    print(len(data))
    return fin_data



def main():

    folded_imgs=performe_folding("input/unfolded.png",100)

    seed = Bild(Image.open("input/unfolded.png")
                     .resize((200, 200), resample=Image.NEAREST)
                     .convert('L'))

    seed.center()
    h = seed.x_max() - seed.x_min() + 1
    w = seed.y_max() - seed.y_min() + 1

    data = [seed]

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
        print(".", end='', flush=True)

    size = max(h, w)
    h = 1 << (size - 1).bit_length()
    w = h

    for i in range(len(data)):
        image = data[i]
        saved = Image.fromarray(image.pixels, 'L').crop(((image.w - w) // 2,
                                                         (image.h - h) // 2,
                                                         image.w - (image.w - w + 1) // 2,
                                                         image.h - (image.h - h + 1) // 2))
        saved.save("output/image%d.png" % i)

    print(len(data))



if __name__== "__main__":
  main()
