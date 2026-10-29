import matplotlib.pyplot as plt
from env.track import Track, make_oval

track = Track(make_oval())

print(type(track.left), track.left.shape)
print("център[0]:", track.center[0])
print("лява[0]:  ", track.left[0])

plt.plot(*track.center.T, "--", label="център")
plt.plot(*track.left.T, label="лява стена")
plt.plot(*track.right.T, label="дясна стена")
plt.axis("equal")
plt.legend()
plt.show()