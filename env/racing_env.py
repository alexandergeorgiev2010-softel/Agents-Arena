import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.track import Track, make_oval
from env.car import Car


class RacingEnv(gym.Env):
    def __init__(self, track=None, max_steps=1000, n_rays=7, max_range=30.0):
        super().__init__()
        self.track = track if track is not None else Track(make_oval())
        self.max_steps = max_steps
        self.max_range = max_range
        self.dt = 0.1

        # Sensors: rays from -90 deg (right) to +90 deg (left) relative to the car heading
        self.ray_angles = np.linspace(-np.pi / 2, np.pi / 2, n_rays)

        # Walls as closed line segments: point i -> point i+1 (both left and right wall)
        self.seg_a = np.concatenate([self.track.left, self.track.right])
        self.seg_b = np.concatenate([np.roll(self.track.left, -1, axis=0),
                                     np.roll(self.track.right, -1, axis=0)])

        # Average length of one centerline segment (converts progress to meters)
        c = self.track.center
        self.seg_len = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
        self.n_center = len(c)

        # Action: [throttle, steer], observation: 7 sensors + speed (all in 0..1)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_rays + 1,), dtype=np.float32)

    # ---------- helpers ----------
    def _nearest_idx(self):
        # Index of the closest centerline point and the distance to it
        d = np.linalg.norm(self.track.center - [self.car.x, self.car.y], axis=1)
        i = int(np.argmin(d))
        return i, d[i]

    def _sensors(self):
        # Ray casting: intersect every ray with every wall segment at once,
        # keep the nearest hit per ray, normalise by max_range.
        origin = np.array([self.car.x, self.car.y])
        ang = self.car.angle + self.ray_angles
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (R, 2)
        e = self.seg_b - self.seg_a                                   # (S, 2)
        ap = self.seg_a - origin                                      # (S, 2)

        denom = dirs[:, None, 0] * e[None, :, 1] - dirs[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ap[None, :, 0] * e[None, :, 1] - ap[None, :, 1] * e[None, :, 0]) / denom
            u = (ap[None, :, 0] * dirs[:, None, 1] - ap[None, :, 1] * dirs[:, None, 0]) / denom
        valid = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, self.max_range)
        return np.clip(t.min(axis=1), 0, self.max_range) / self.max_range

    def _obs(self):
        speed = self.car.speed / 10.0
        return np.append(self._sensors(), speed).astype(np.float32)

    # ---------- Gymnasium interface ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.track.center
        start, nxt = c[0], c[1]
        angle = np.arctan2(nxt[1] - start[1], nxt[0] - start[0])
        self.car = Car(start[0], start[1], angle)
        self.prev_idx = 0
        self.cum_idx = 0          # accumulated progress in centerline points (signed)
        self.steps = 0
        self.trajectory = [[self.car.x, self.car.y, self.car.angle]]
        return self._obs(), {}

    def step(self, action):
        throttle, steer = float(action[0]), float(action[1])
        self.car.step(throttle, steer, self.dt)
        self.steps += 1
        self.trajectory.append([self.car.x, self.car.y, self.car.angle])

        # Progress = change of the nearest centerline index, wrapped around the loop
        # so that crossing the start line does not produce a huge jump.
        idx, dist = self._nearest_idx()
        delta = idx - self.prev_idx
        if delta > self.n_center / 2:
            delta -= self.n_center
        elif delta < -self.n_center / 2:
            delta += self.n_center
        self.prev_idx = idx
        self.cum_idx += delta
        progress = delta * self.seg_len.mean()

        # Reward: progress (m) - small time penalty; -10 on crash, +50 on lap
        reward = progress - 0.01
        terminated = False
        truncated = False
        crashed = dist > self.track.width / 2      # outside the corridor
        lap = self.cum_idx >= self.n_center        # full loop completed

        if crashed:
            reward -= 10.0
            terminated = True
        elif lap:
            reward += 50.0
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        info = {"crashed": crashed, "lap": lap, "steps": self.steps}
        if terminated or truncated:
            info["trajectory"] = np.array(self.trajectory)   # for the 3D viewer
        return self._obs(), reward, terminated, truncated, info