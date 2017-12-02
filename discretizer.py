import math
class PointDiscretizer:
    def __init__(self, axis_div):
        self.ubound = 1.5
        self.lbound = -1.5
        self.bound_len = self.ubound - self.lbound
        self.box_div = self.bound_len / axis_div # width and height of a box
        self.num_axis_subdivisions = axis_div
        self.num_states = self.num_axis_subdivisions * self.num_axis_subdivisions
    def discretize(self, x, y):
        # first, add lbound to each
        x -= self.lbound
        y -= self.lbound
        # second, determine the row and column
        row = int(y // self.box_div)
        col = int(x // self.box_div)
        return row, col
    def un_discretize(self, row, col):
        # inverse steps from discretize
        x = col * self.box_div
        y = row * self.box_div
        x += self.lbound
        y += self.lbound
        # finally, move to 'center' of discretized box:
        x += .5 * self.box_div
        y += .5 * self.box_div
        return (x,y,)
    def states(self):
        ret = []
        for r in range(self.num_axis_subdivisions):
            for c in range(self.num_axis_subdivisions):
                ret.append( (r, c,) )
        return ret
    def adj(self, point, rad):
        ret = []
        for r in range(-rad, rad+1, 1):
            if not self.in_bounds(point[0] + r):
                continue
            for c in range(-rad, rad+1, 1):
                if not self.in_bounds(point[1] + c):
                    continue
                ret.append( (point[0] + r, point[1] + c,) )
        return ret
    def in_bounds(self, row): # works for row or col values
        return row >= 0 and row < self.num_axis_subdivisions

class AngleDiscretizer:
    def __init__(self, divisions):
        self.divisions = divisions
        self.ubound = math.pi/2
        self.increment = self.ubound / self.divisions
    def discretize(self, angle):
        return int(angle // self.increment)
    @property
    def num_states(self):
        return self.divisions