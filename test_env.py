import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from env.racing_env import RacingEnv


def run(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        if term or trunc:
            return total, info


env = RacingEnv()
check_env(env)   # проверка, че интерфейсът е валиден за gym/SB3
print("check_env: OK")

random_policy = lambda obs: env.action_space.sample()

# ръчен алгоритъм: газ 0.3, завива към страната с повече място
def wall_follow(obs):
    steer = 2.0 * (obs[5] + obs[6] - obs[0] - obs[1])
    return np.array([0.3, np.clip(steer, -1, 1)], dtype=np.float32)

fig, ax = plt.subplots()
ax.plot(*env.track.left.T, "k")
ax.plot(*env.track.right.T, "k")

for name, pol, col in [("random", random_policy, "r"), ("wall_follow", wall_follow, "g")]:
    total, info = run(env, pol)
    print(f"{name}: награда={total:.1f} стъпки={info['steps']} "
          f"удар={info['crashed']} обиколка={info['lap']}")
    ax.plot(*info["trajectory"][:, :2].T, col, label=name)

ax.axis("equal")
ax.legend()
plt.show()