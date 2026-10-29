import numpy as np

def make_oval(a = 40.0, b = 20.0, n = 200):
    t = np.linspace(0, 2 * np.pi, n, endpoint = False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis = 1)

class Track:
    def __init__(self, centerline, width = 8.0):
        self.center = np.asarray(centerline, dtype = float)
        self.width = width

        d = np.roll(self.center, -1, axis = 0) - self.center
        tangent = d / np.linalg.norm(d, axis = 1, keepdims = True)

        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis = 1)

        self.left = self.center + normal * width / 2
        self.right = self.center - normal * width / 2
        