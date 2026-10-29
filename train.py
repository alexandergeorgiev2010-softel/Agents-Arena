import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.racing_env import RacingEnv

STEPS = 100_000
SEED = 0

env = RacingEnv()
model = PPO("MlpPolicy", env, seed=SEED, device="cpu", verbose=1)
model.learn(total_timesteps=STEPS)
model.save("results/ppo_oval")


def run(policy):
    obs, _ = env.reset(seed=SEED)
    total = 0.0
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        if term or trunc:
            return total, info


ppo_policy = lambda obs: model.predict(obs, deterministic=True)[0]
random_policy = lambda obs: env.action_space.sample()

def wall_follow(obs):
    steer = 2.0 * (obs[5] + obs[6] - obs[0] - obs[1])
    return np.array([0.3, np.clip(steer, -1, 1)], dtype=np.float32)

fig, ax = plt.subplots()
ax.plot(*env.track.left.T, "k")
ax.plot(*env.track.right.T, "k")

for name, pol, col in [("random", random_policy, "r"),
                       ("wall_follow", wall_follow, "g"),
                       ("PPO", ppo_policy, "b")]:
    total, info = run(pol)
    print(f"{name}: награда={total:.1f} стъпки={info['steps']} "
          f"удар={info['crashed']} обиколка={info['lap']}")
    ax.plot(*info["trajectory"][:, :2].T, col, label=name)

ax.axis("equal")
ax.legend()
plt.show()