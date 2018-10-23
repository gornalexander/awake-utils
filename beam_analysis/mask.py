import numpy as np


class Condition(object):
    def __init__(self, cond):
        self._cond = cond

    def __call__(self, x, y):
        return self._cond(x, y)

    def __add__(self, other):
        return Condition(lambda x, y: self(x, y) | other(x, y))

    def __sub__(self, other):
        return Condition(lambda x, y: self(x, y) & (~other(x, y)))


class Box(Condition):
    def __init__(self, a, b, c=None, d=None):  # clockwise, starting bottom left
        if c is None and d is None:
            c = b
            b = (a[0], c[1])
            d = (c[0], a[1])
        self.ml = (a[0] - b[0]) / (a[1] - b[1])
        self.bl = a[0] - self.ml * a[1]
        # l = lambda x, y: x > ml * y + bl

        self.mr = (c[0] - d[0]) / (c[1] - d[1])
        self.br = c[0] - self.mr * c[1]
        # r = lambda x, y: x < mr * y + br

        self.mt = (b[1] - c[1]) / (b[0] - c[0])
        self.bt = b[1] - self.mt * b[0]
        # t = lambda x, y: y > mt * x + bt

        self.mb = (a[1] - d[1]) / (a[0] - d[0])
        self.bb = a[1] - self.mb * a[0]
        # b = lambda x, y: y > mb * x + bb

        super().__init__(self.condition)

    def condition(self, x, y):
        return (x > self.ml * y + self.bl) & (x < self.mr * y + self.br) & (
                    y < self.mt * x + self.bt) & (y > self.mb * x + self.bb)


class Circle(Condition):
    def __init__(self, cx, cy, r):
        self.x = cx
        self.y = cy
        self.r2 = r ** 2
        super().__init__(self.condition)

    def condition(self, x, y):
        return (self.x - x) ** 2 + (self.y - y) ** 2 < self.r2


def apply_mask(cam, condition):
    y, x = np.indices(cam.shape)
    cam.data = np.ma.array(cam.data, copy=False, mask=condition(x, y))
    return cam
