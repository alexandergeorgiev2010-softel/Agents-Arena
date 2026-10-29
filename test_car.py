import numpy as np
import matplotlib.pyplot as plt
from env.track import Track, make_oval
from env.car import Car

track = Track(make_oval())

start = track.center[0]
car = Car(start[0], start[1], angle=np.pi / 2)  # гледа нагоре

path = []
for i in range(200):
    car.step(throttle=0.5, steer=0.1)
    path.append([car.x, car.y])
path = np.array(path)

plt.plot(*track.left.T)
plt.plot(*track.right.T)
plt.plot(*path.T, "r")
plt.axis("equal")
plt.show()