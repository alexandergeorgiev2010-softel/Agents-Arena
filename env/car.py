import numpy as np


class Car:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0

    def step(self, throttle, steer, dt=0.1):
        self.angle += steer * 1.5 * dt
        self.speed += throttle * 5.0 * dt
        self.speed = np.clip(self.speed, 0.0, 10.0)
        self.x += np.cos(self.angle) * self.speed * dt
        self.y += np.sin(self.angle) * self.speed * dt