# Introduction to Deep Reinforcement Learning — Simulator

An interactive browser-based simulator for the **PPO (Proximal Policy Optimisation)** algorithm applied to the classic **MountainCar-v0** environment.

The agent learns entirely in your browser — no server, no GPU, no installation required.
A reference Python implementation (matching OpenAI Gym) is included under `docs/example/`.

**Live demo:** https://tvy14.github.io/Introduction-DRL-Simulator/

---

## Repository Structure

```
Introduction-DRL-Simulator/
│
├── index.html                    # App entry point
├── package.json                  # Node dependencies (Vite + TF.js)
├── tsconfig.json                 # TypeScript config
├── vite.config.ts                # Vite build config (base: './')
│
├── src/
│   ├── env.ts                    # MountainCar-v0 physics (matches Gym exactly)
│   ├── ppo.ts                    # PPO Actor-Critic + GAE + clipped surrogate (TF.js)
│   ├── renderer.ts               # Canvas renderer — mountain, car, reward chart
│   ├── main.ts                   # Training loop, UI controls, stats panel
│   └── style.css                 # Dark-theme stylesheet
│
├── docs/
│   └── example/
│       ├── PPO_MountainCar-v0.py     # Reference Python implementation (PyTorch)
│       ├── PPO_MountainCar-v0.ipynb  # Jupyter notebook version
│       ├── README.md                 # Python exercise description
│       └── Step.md                   # Python setup & run guide
│
└── .github/
    └── workflows/
        └── deploy.yml            # GitHub Actions — build & deploy to gh-pages
```

---

## How It Works

| Component | Details |
|-----------|---------|
| **Environment** | MountainCar-v0 — a car must build momentum to reach the hilltop. State: `[position, velocity]`. Actions: push left / none / push right. Reward: −1 per step. |
| **Algorithm** | PPO with Actor-Critic. Shared 2×64 tanh network → softmax actor head + linear critic head. GAE (λ=0.95) for advantage estimation, clipped surrogate objective (ε=0.2), entropy bonus. |
| **Runtime** | TensorFlow.js runs inference and backprop in-browser via WebGL. No server needed. |
| **Deployment** | Vite bundles TypeScript → static files. GitHub Actions pushes `dist/` to the `gh-pages` branch on every push to `main`. |

---

## Tutorial

### Option A — Live Demo (no setup)

Open **https://tvy14.github.io/Introduction-DRL-Simulator/** in any modern browser.

1. Press **▶ Start** to begin training.
2. Watch the car on the canvas — it starts by oscillating randomly, then learns to swing up.
3. Drag the **Speed** slider to train faster (more steps per frame) or slower for clearer visualisation.
4. Press **⏸ Pause** at any time, then **▶ Resume** to continue.
5. Press **↺ Reset** to start training from scratch.

The **reward chart** below the canvas shows per-episode reward over the last 200 episodes.
A successful run converges to rewards around **−100 to −150** (vs −200 for random).

---

### Option B — Run Locally (TypeScript / browser)

**Prerequisites:** Node.js ≥ 18

**Step 1 — Clone the repository**
```bash
git clone https://github.com/tvy14/Introduction-DRL-Simulator.git
cd Introduction-DRL-Simulator
```

**Step 2 — Install dependencies**
```bash
npm install
```

**Step 3 — Start the dev server**
```bash
npm run dev
```
Open the URL printed in the terminal (usually `http://localhost:5173`).

**Step 4 — Build for production**
```bash
npm run build
```
Output goes to `dist/`. Serve it with any static file server:
```bash
npm run preview
```

---

### Option C — Python Reference Implementation

The `docs/example/` folder contains the original PyTorch implementation for offline / GPU training.

**Step 1 — Install Python dependencies**
```bash
pip3 install --user "gym[classic_control]==0.21.0"
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user tensorboardX
```

> `gym==0.21.0` is required. The code uses the legacy Gym API (`env.seed()`,
> 4-value `env.step()` return) which was removed in gym ≥ 0.26.

**Step 2 — Run training**
```bash
python3 docs/example/PPO_MountainCar-v0.py
```

Expected output (prints every 1000 gradient steps):
```
I_ep 0 ，train 0 times
I_ep 0 ，train 1000 times
...
```

**Step 3 — Monitor with TensorBoard**
```bash
pip3 install tensorboard
tensorboard --logdir exp/ --port 6006
```
Open `http://localhost:6006`. Watch **`Steptime/steptime`** — it should decrease as the agent learns.

**Step 4 — Jupyter notebook**

Open `docs/example/PPO_MountainCar-v0.ipynb` in JupyterLab or VS Code to run the same code cell-by-cell.

---

### Option D — Deploy Your Own Fork to GitHub Pages

**Step 1 — Fork this repository** on GitHub.

**Step 2 — Enable GitHub Pages**
Go to your fork → **Settings → Pages → Source: Deploy from branch → `gh-pages` / `/ (root)`** → Save.

**Step 3 — Trigger a deploy**
Push any change to `main` (or go to **Actions → Deploy to GitHub Pages → Run workflow**).
The workflow installs dependencies, builds with Vite, and pushes `dist/` to the `gh-pages` branch automatically.

**Step 4 — Visit your site**
```
https://<your-username>.github.io/Introduction-DRL-Simulator/
```

---

## PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor |
| `lam` (λ) | 0.95 | GAE smoothing |
| `clipEps` (ε) | 0.2 | PPO clipping range |
| `valueCoef` | 0.5 | Critic loss weight |
| `entropyCoef` | 0.01 | Entropy bonus weight |
| `nSteps` | 2048 | Rollout length before update |
| `nEpochs` | 4 | PPO update epochs per rollout |
| `batchSize` | 64 | Mini-batch size |
| `learningRate` | 3×10⁻⁴ | Adam optimiser LR |
| Network | 2 → 64 → 64 → (3 + 1) | tanh hidden, softmax actor, linear critic |

---

## Reference

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Brockman et al. (2016). *OpenAI Gym.* [arXiv:1606.01540](https://arxiv.org/abs/1606.01540)
- Book exercise: *Whole-Course Machine Learning*, Chapter 1, Exercise 1.12
