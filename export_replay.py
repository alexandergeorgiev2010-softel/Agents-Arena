import json
import numpy as np
from stable_baselines3 import PPO
from env.racing_env import RacingEnv

env = RacingEnv()
model = PPO.load("results/ppo_oval", device="cpu")


def wall_follow(obs):
    steer = 2.0 * (obs[5] + obs[6] - obs[0] - obs[1])
    return np.array([0.3, np.clip(steer, -1, 1)], dtype=np.float32)


policies = {
    "random": lambda obs: env.action_space.sample(),
    "wall_follow": wall_follow,
    "PPO": lambda obs: model.predict(obs, deterministic=True)[0],
}


def run(policy, seed=0):
    obs, _ = env.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        if term or trunc:
            return info


t = env.track
data = {
    "track": {
        "center": np.round(t.center, 3).tolist(),
        "left": np.round(t.left, 3).tolist(),
        "right": np.round(t.right, 3).tolist(),
    },
    "agents": {},
}

for name, pol in policies.items():
    info = run(pol)
    data["agents"][name] = {
        "steps": int(info["steps"]),
        "lap": bool(info["lap"]),
        "crashed": bool(info["crashed"]),
        "traj": np.round(info["trajectory"], 3).tolist(),
    }
    print(name, info["steps"], "lap" if info["lap"] else "no lap")

with open("web/data.js", "w") as f:
    f.write("window.REPLAY = " + json.dumps(data) + ";")
print("Saved web/data.js")