# Decision Log

Research project: comparing reinforcement learning algorithms (and hand-written baselines) on a simulated racing task, with results visualised in a 3D browser viewer.

This log records what was decided at each step and why, so that every result is reproducible and every design choice can be defended.

---

## 0. Project goal and scope

**Goal.** Train and compare different AI agents (PPO first, then SAC, DQN and others) that drive a robot car around tracks. Measure which is fastest, most reliable, and best at generalising to unseen tracks.

**Scope decisions**

| Decision | Choice | Rationale |
|---|---|---|
| Training language | Python (Gymnasium + Stable-Baselines3 + PyTorch) | Standard RL ecosystem; fast headless simulation. |
| Visualisation | Browser, Three.js (planned) | No installation for the audience, shareable by link, free-fly camera over a 3D terrain. |
| Training/visualisation coupling | Python writes trajectories to JSON; the browser only replays them | No server or language bridge; robust for a live conference demo. |
| Simulation dimensionality | 2D physics, 3D rendering | 5-50x faster training on a laptop without a GPU. Real 3D physics (e.g. parkour) is deferred to a later stage. |
| Previous C++/OpenGL engine | Not reused | Refactoring cost is high and it is irrelevant to the RL research question. Mentioned only as background experience. |

---

## 1. Development environment

- Machine: MacBook Pro (Intel, x86_64), no dedicated GPU, VS Code.
- Python 3.10.4 managed by pyenv; activated per-project with `pyenv local 3.10.4` (creates `.python-version`).
- Isolated virtual environment in `.venv` so that project packages never affect other projects.
- Training runs on CPU only (`device="cpu"`).

**Pinned versions (resolved by pip)**

| Package | Version |
|---|---|
| gymnasium | 1.0.0 |
| stable-baselines3 | 2.4.1 |
| torch | 2.2.2 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |

Why older versions: the newest PyTorch no longer ships wheels for Intel Macs, so pip selected the latest compatible releases. They are sufficient for this project. Versions are recorded for reproducibility.

**Project structure**

```
AI project/
├── .venv/              virtual environment
├── .python-version     pyenv pin (3.10.4)
├── env/
│   ├── __init__.py
│   ├── track.py        track geometry
│   ├── car.py          vehicle dynamics
│   └── racing_env.py   Gymnasium environment
├── agents/             (planned) training scripts, baselines
├── results/            saved models, logs, JSON trajectories
├── web/                (planned) Three.js viewer
├── test_track.py       visual check of track geometry
├── test_car.py         visual check of car motion
├── test_env.py         environment check + baseline agents
├── train.py            PPO training and comparison
└── DECISIONS.md
```

---

## 2. Track geometry (`env/track.py`)

- A track is a **centerline** (an array of N points) plus a constant **width**. Walls are the centerline offset by `width / 2` along the normal on each side.
- Construction: tangent = (next point - current point), normalised; normal = tangent rotated 90° left, `(x, y) -> (-y, x)`; left wall = center + normal * width/2; right wall = center - normal * width/2.
- First track: an ellipse, `x = a*cos(t)`, `y = b*sin(t)`, with `a=40`, `b=20`, 200 points, `endpoint=False` to avoid a duplicated start point. Width 8.
- **Why this design:** a new track is only a new list of points; no other code changes. Geometry is kept separate from vehicle physics.
- **Known limit:** if the width exceeds the minimum turning radius of the centerline (about 10 units at the ends of this oval), the inner wall self-intersects.

---

## 3. Vehicle model (`env/car.py`)

- State: position `(x, y)`, heading `angle`, `speed`.
- Actions: `throttle` in [-1, 1], `steer` in [-1, 1].
- Per step (`dt = 0.1 s`): `angle += steer * 1.5 * dt`; `speed += throttle * 5.0 * dt`; speed clipped to [0, 10]; position advances along the heading by `speed * dt`.
- **Why:** a minimal kinematic model with continuous actions, compatible with PPO and SAC. No reverse gear and a speed cap keep the task well-posed. A discrete wrapper will be added for DQN.

---

## 4. Environment design (`env/racing_env.py`)

Implements the Gymnasium interface (`reset`, `step`), validated with Stable-Baselines3 `check_env`.

**Observation (8 values, all in [0, 1])**
- 7 distance sensors (rays) spread from -90° (right) to +90° (left) relative to the heading, max range 30, normalised.
- Current speed / 10.

The agent never sees a map, only what its sensors report. This is more realistic and makes generalisation to unseen tracks possible. Ray-segment intersection is computed against all wall segments at once with numpy.

**Action:** continuous `[throttle, steer]`.

**Progress measurement.** The index of the nearest centerline point is tracked; its change per step (wrapped around the loop) gives progress along the track. Moving backwards yields negative progress, so circling or reversing cannot be exploited. Progress is scaled by the mean centerline segment length.

**Reward**
- `+ progress` along the centerline (metres)
- `- 0.01` per step (time pressure)
- `- 10` on crash (episode terminates)
- `+ 50` on completing a lap (episode terminates)

*Why progress and not distance from start:* a distance-based reward can be farmed by driving in circles; progress along the centerline only rewards forward movement in the right direction.

**Termination**
- Crash: distance to the nearest centerline point exceeds `width / 2` (approximation of "outside the corridor").
- Lap: accumulated progress reaches the number of centerline points.
- Truncation: 1000 steps.

**Logging:** the full trajectory `(x, y, angle)` per step is returned in `info["trajectory"]` at episode end, for replay in Three.js.

**Observation about the reward.** The time penalty is small, so the total reward barely distinguishes a fast lap from a slow one (maximum about 245). Therefore the **primary metric is lap time in steps**, and reward is logged as a secondary metric. A larger time penalty will be tested as a separate experiment.

---

## 5. Baselines (`test_env.py`)

Baselines are essential: without them an RL result has no reference point.

1. **Random agent:** uniformly random actions.
2. **wall_follow (hand-written):** constant throttle 0.3; steer = `2.0 * (right sensors - left sensors)`, i.e. turn towards the side with more free space.

**Baseline results (seed 0)**

| Agent | Reward | Steps | Outcome |
|---|---|---|---|
| random | 19.7 | 228 | crash |
| wall_follow | 241.5 | 231 | lap completed |

The random agent does not stand still: its speed accumulates by chance and it crashes after about 228 steps.

---

## 6. First RL experiment: PPO (`train.py`)

- Algorithm: PPO (Stable-Baselines3), `MlpPolicy`, default hyperparameters.
- 100,000 timesteps, seed 0, CPU, about 650 steps/s (a few minutes in total).
- The training code is the library's; the project contribution is the environment, the reward design and the evaluation.

**Training behaviour.** Mean episode reward (`ep_rew_mean`) reached about 241 by roughly 40,000 steps and then plateaued near the theoretical maximum (about 245). Mean episode length kept decreasing (212 -> 206), meaning laps became faster rather than merely safe. The oval is an easy task.

**Result (seed 0)**

| Agent | Reward | Steps | Outcome |
|---|---|---|---|
| random | 19.7 | 228 | crash |
| wall_follow | 241.5 | 231 | lap completed |
| **PPO** | **241.8** | **198** | **lap completed** |

PPO completed the lap in 198 steps (19.8 s of simulated time) versus 231 for the hand-written controller, about **14% faster**.

**Observed behaviour.** The PPO trajectory follows the track closely but drifts towards the outer boundary at the two tight ends of the oval. The agent has learned to cut corners and carry speed, staying near the edge of the allowed corridor. This is expected: it optimises exactly what is measured.

**Limitation.** This is a single seed and a single run. It does not yet support the claim that PPO outperforms the baseline; this could be luck.

---

## 7. Limitations and threats to validity

- Single seed; no confidence intervals yet.
- Single, easy track; no generalisation test.
- 2D physics with a simple kinematic car model.
- The crash test is approximate (nearest centerline point).
- Default PPO hyperparameters, not tuned.
- Reward design strongly influences behaviour; conclusions are specific to this reward.

---

## 8. Next steps

1. `compare.py`: train PPO with 5 seeds; record lap time, completion rate and training steps to convergence; report mean and spread; save trajectories as JSON.
2. Add SAC (continuous actions) and DQN (discrete wrapper) with equal training budgets and the same seeds.
3. Add more tracks (curves, narrow sections, obstacles) and an unseen-track generalisation test.
4. Reward experiment: larger per-step time penalty; compare effect on lap time.
5. Three.js viewer: 3D terrain, lighting, multiple robots replaying JSON trajectories, free-fly / orbit / follow cameras, live leaderboard.
6. Optional: export trained policies to ONNX and run them live in the browser (`onnxruntime-web`).
7. Optional stage 3: true 3D physics (slopes, jumps, parkour) as an additional experiment on increasing environment complexity.

---

## 9. Use of AI in this project

An AI assistant (Claude) was used to generate most of the code and to explain design decisions. The author set the research direction, chose the experimental design (algorithms, baselines, metrics), ran and modified the code, diagnosed integration errors, and evaluated the results. Where the AI made mistakes (for example, misdiagnosing a file-import error and mis-describing random-agent behaviour), they were identified and corrected during the work. This is intended to be documented as part of the comparison of AI-assisted and manual development.
