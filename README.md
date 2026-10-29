# Agents Arena

A benchmark for comparing reinforcement learning agents and rule-based baselines
on racing tracks. Training runs in Python (Gymnasium + Stable-Baselines3);
results are meant to be replayed in 3D in the browser (Three.js).

## 🎥 Demo


https://github.com/user-attachments/assets/dbe226fe-243a-4f8e-8d9b-53ed038035ae



## How it works

- **Environment:** a car drives around a closed track. It sees 7 distance
  sensors and its speed (8 numbers) and outputs throttle and steering.
- **Reward:** progress along the track, a small time penalty, -10 for a crash,
  +50 for completing a lap.
- **Agents:** PPO (learned) compared with a random agent and a rule-based
  wall-following controller.

## First results (oval track, 1 seed)

| Agent | Reward | Steps per lap | Outcome |
|---|---|---|---|
| random | 19.7 | 228 | crash |
| wall_follow | 241.5 | 231 | lap |
| PPO (100k steps) | 241.8 | 198 | lap |

Single seed, so this is a first look and not yet a statistical result.

## Run it

```
python -m venv .venv
source .venv/bin/activate
pip install gymnasium stable-baselines3 numpy matplotlib
python test_env.py   # baselines
python train.py      # train PPO and compare
```

## Roadmap

- Multiple seeds with mean and spread
- More algorithms (SAC, TD3, DQN) and tracks
- 3D browser viewer with free camera

See [DECISIONS.md](DECISIONS.md) for the design decisions and their rationale.
