import random
from math import pi, sin, cos

from PIL import Image
import numpy as np

from geometry import *

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
                    self.nonzero.append((i, j))

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
            attempts -= 1
            center = random.choice(self.nonzero)
            center = (center[0] + 0.5, center[1] + 0.5)
            xy_aligned = random.random() <= xy_axis_bias
            if xy_aligned:
                if random.random() <= 0.5:
                    axis = Line(center, (center[0] + 1, center[1]))
                else:
                    axis = Line(center, (center[0], center[1] + 1))
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
                                                                         p[0] < 0 or p[0] >= self.h or \
                                                                         p[1] < 0 or p[1] >= self.w)

                    for point in list(reflected_interior.union(reflected_contour)):
                        value = 0
                        folded.append(point)
                        if not point in backward_map:
                            original_values = []
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                neighbour = (point[0] + dx, point[1] + dy)
                                if neighbour in backward_map:
                                    for p in backward_map[neighbour]:
                                        original_values.append(self.pixels[p[0], p[1]])
                            if not original_values:
                                continue
                            else:
                                value = max(set(original_values), key=original_values.count)
                        else:
                            original_values = [self.pixels[p[0], p[1]] for p in backward_map[point]]
                            value = max(set(original_values), key=original_values.count)

                        updates[point] = value
                        if point in backward_map:
                            if point in backward_map[point]:
                                stationary_points.add(point)
                                updates[point] = self.pixels[point[0], point[1]]
                            else:
                                for p in backward_map[point]:
                                    updates[p] = -int(self.pixels[p[0], p[1]])

                    part2_set = set(part2)
                    for point in contour:
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                neighbour = (point[0] + dx, point[1] + dy)
                                if neighbour in part2_set and \
                                   not neighbour in updates:
                                    stationary_points.add(neighbour)
                                    updates[neighbour] = 0
                                    

                self.nonzero.clear()
                for point in list(set(part1 + part2 + folded)):
                    if point in updates:
                        if not point in stationary_points:
                            self.pixels[point[0], point[1]] += updates[point]
                        else:
                            self.pixels[point[0], point[1]] *= 2
                    if self.pixels[point[0], point[1]] > 0:
                        self.nonzero.append(point)

                self.foldcount += 1
                return True

        return False

    def perturb(self, num_samples=10, resolution=20, integral_tolerance=1):
        # resolution = width of a perturbation window
        resolution = resolution // 2 * 2 + 1

        # sample seed points for perturbations from a contour
        contours = extract_contours(set(self.nonzero))
        if not contours:
            return

        # contour should be one, just in case use the longest one
        contour = contours[0]
        seeds = []

        for _ in range(num_samples):
            # make sure all seed points are separated, so that
            # perturbation windows do not overlap
            tries = 50
            seed = None
            while tries:
                tries -= 1
                seed = random.choice(contour)
                is_separated_well = True
                for previous_seed in seeds:
                    if points_closer_than(seed, previous_seed, resolution):
                        is_separated_well = False
                        break
                if is_separated_well:
                    break
            if seed:
                seeds.append(seed)

        for seed in seeds:
            # shrink the perturbation window until non-zero pixels inside have
            # a single value only; split the window into square slices;
            # perturbation window is a square with side (size * 2 - 1)
            border = []
            nonzero_value = 0
            size = 0
            for i in range(resolution // 2 + 1):
                slice = [(x, seed[1] - i) for x in range(seed[0] - i,
                                                        seed[0] + i)] +\
                        [(x, seed[1] + i) for x in range(seed[0] - i + 1,
                                                        seed[0] + i + 1)] +\
                        [(seed[0] - i, y) for y in range(seed[1] - i + 1,
                                                        seed[1] + i + 1)] +\
                        [(seed[0] + i, y) for y in range(seed[1] - i,
                                                        seed[1] + i)]
                min_value = 256
                max_value = 0
                for point in slice:
                    value = self.pixels[point[0], point[1]]
                    if value > 0:
                        min_value = min(value, min_value)
                        max_value = max(value, max_value)
                        if min_value != max_value:
                            break

                if min_value != max_value and max_value != 0:
                    break
                else:
                    if not nonzero_value and max_value != 0:
                        nonzero_value = max_value
                    if max_value != nonzero_value:
                        break
                    border = slice
                    size += 1            
            
            if size < 3:
                print("Too small window size")
                continue
            
            # find the part of a contour that falls into the window
            border = set(border)
            original_contour = set()
            original_filled = set()

            for x in range(seed[0] - size + 1, seed[0] + size):
                for y in range(seed[1] - size + 1, seed[1] + size):
                    point = (x, y)
                    if point in contour:
                        original_contour.add(point)
                    if self.pixels[x, y] > 0:
                        original_filled.add(point)

            # only allow windows in which a single piece of contour is captured
            if len(split_into_components(original_contour)) > 1:
                print("Multiple contours in the window")
                continue

            # find two ends of the contour inside the window lying on its border
            graph = graph_from_points(original_contour)
            depths = bfs_distances(graph, [seed])
            start = None
            for depth, point in enumerate(depths):
                if point in border:
                    if not start or depths[start] < depth:
                        start = point
            if not start:
                print("Not found a start")
                continue

            depths = bfs_distances(graph, [start])
            finish = None
            for depth, point in enumerate(depths):
                if point in border:
                    if not finish or depths[finish] < depth:
                        finish = point
            if not finish or start == finish:
                print("Not found a finish")
                continue

            # find a point on the window border that is inside the filled region

            inside_points = []
            for point in border:
                if point in original_filled and not point in original_contour:
                    inside_points.append(point)
            if not inside_points:
                print("No seed points for BFS")
                continue

            # now we make a new contour Bezier segment between start and finish
            sampling_area = list(border)
            sampling_area.remove(start)
            sampling_area.remove(finish)
            
            # first try several random curves and evaluate filled integral

            def apply_new_curve(curve: Bezier):
                def in_bounds(point):
                    return seed[0] - size + 1 <= point[0] < seed[0] + size and \
                           seed[1] - size + 1 <= point[1] < seed[1] + size

                curve_pixels = set()
                for t in np.linspace(0, 1, (size * 2 + 1) * 5):
                    curve_pixels.add(curve.evaluate(t))

                # fill the correct area bounded by the curve and the window
                starts = set(inside_points)
                for point in curve_pixels:
                    if point in starts:
                        starts.remove(point)
                if not starts:
                    return []

                #boundary = curve_pixels.union(border)
                filled = bfs_unless(starts,
                                    lambda point: point in curve_pixels or \
                                                  not in_bounds(point))
                return filled.union(curve_pixels)

            samples = []
            tries = 30
            best_curve = None
            best_integral = None
            while tries:
                tries -= 1
                P1 = random.choice(sampling_area)
                P2 = random.choice(sampling_area)
                if size > min(self.h, self.w) / 20:
                    P1 = (P1[0] / 2 + start[0] / 2,
                        P1[1] / 2 + start[1] / 2)
                    P2 = (P2[0] / 2 + finish[0] / 2,
                        P2[1] / 2 + finish[1] / 2)
                curve = Bezier(start, P1, P2, finish)
                integral = len(apply_new_curve(curve))
                if integral >= len(original_filled):
                    if not best_integral or integral < best_integral:
                        best_curve, best_integral = curve, integral
            if not best_curve:
                print("Not a single curve with big enough integral")
                continue

            # if the filled area is too large, shrink it by scaling the curve
            # but first check if the smallest possible curve gives smaller area
            
            integral = len(original_filled)
            min_curve = Bezier(start, start, finish, finish)
            if len(apply_new_curve(min_curve)) > integral + integral_tolerance:
                print("Even straight line gives too big integral")
                continue

            min_factor = 0
            max_factor = 1
            result = []
            while max_factor - min_factor > 0.01:
                factor = (min_factor + max_factor) / 2
                curve = Bezier(
                      best_curve.p0,
                      (best_curve.p0[0] + (best_curve.p1[0] - best_curve.p0[0]) * factor,
                       best_curve.p0[1] + (best_curve.p1[1] - best_curve.p0[1]) * factor),
                      (best_curve.p3[0] + (best_curve.p2[0] - best_curve.p3[0]) * factor,
                       best_curve.p3[1] + (best_curve.p2[1] - best_curve.p3[1]) * factor),
                      best_curve.p3)
                filled = apply_new_curve(curve)
                if abs(len(filled) - integral) <= integral_tolerance:
                    result = filled
                    break
                if len(filled) > integral + integral_tolerance:
                    max_factor = factor
                else:
                    min_factor = factor
            if not result:
                print("Binary search failed")
                continue
            
            for x in range(seed[0] - size + 1, seed[0] + size):
                for y in range(seed[1] - size + 1, seed[1] + size):
                    if (x, y) in result:
                        self.pixels[x, y] = 200
                    else:
                        self.pixels[x, y] = 0

    def center(self):
        deltax = (self.h - self.x_min() - self.x_max() - 1) // 2
        deltay = (self.w - self.y_min() - self.y_max() - 1) // 2

        self.pixels = np.roll(self.pixels, deltax, axis=0)
        self.pixels = np.roll(self.pixels, deltay, axis=1)

        self.nonzero = [(p[0] + deltax, p[1] + deltay) for p in self.nonzero]

    def x_min(self):
        return min(self.nonzero, key=lambda p: p[0])[0]

    def y_min(self):
        return min(self.nonzero, key=lambda p: p[1])[1]

    def x_max(self):
        return max(self.nonzero, key=lambda p: p[0])[0]

    def y_max(self):
        return max(self.nonzero, key=lambda p: p[1])[1]


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

    #folded_imgs=performe_folding("input/unfolded.png", 100)

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
            if seed.foldcount < 3:
                break
        image = seed.copy()
        image.fold(0.2, xy_axis_bias=True)
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
        image.perturb(20, 20, 5)
        saved = Image.fromarray(image.pixels, 'L').crop(((image.w - w) // 2,
                                                         (image.h - h) // 2,
                                                         image.w - (image.w - w + 1) // 2,
                                                         image.h - (image.h - h + 1) // 2))
        saved.save("output/image%d.png" % i)

    print(len(data))



if __name__== "__main__":
  main()
