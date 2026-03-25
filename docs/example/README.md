# The code of the PPO algorithm for MountainCar-v0.

The code for Exercise 1.12. Please fill this code to implement the PPO algorithm for MountainCar-vo.
You may need to install the gym library at first.


# How to Run PPO_MountainCar-v0.py

## Prerequisites

Install the required packages (one-time setup):

```bash
pip3 install --user "gym[classic_control]==0.21.0"
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user tensorboardX
```

> **Note:** `gym==0.21.0` is required. The code uses the old Gym API (`env.seed()`,
> 4-value `env.step()` return) which was removed in gym >= 0.26.

## Run the Script

Navigate to the exercise directory and run:

```bash
cd /home/iccl813/tvy14/wcmlbook/ch1/Exercise_1.12
python3 PPO_MountainCar-v0.py
```

## Expected Output

Training prints every 1000 gradient steps:

```
I_ep 0 ，train 0 times
I_ep 0 ，train 1000 times
I_ep 0 ，train 2000 times
...
```

The script runs for 1000 epochs. Each epoch is one episode of MountainCar-v0
(max 200 steps), so it will run for a long time.

## Monitor Training with TensorBoard

Training metrics are logged to `../exp/`. To visualise them:

**Step 1 — Install TensorBoard (one-time):**
```bash
pip3 install tensorboard
```

**Step 2 — Launch TensorBoard:**
```bash
~/.local/bin/tensorboard --logdir /home/iccl813/tvy14/wcmlbook/ch1/exp --port 6006
```

Then open `http://localhost:6006` in your browser.

> If port 6006 is busy, use `--port 6007` (or any free port).

**Key metric to watch:** `Steptime/steptime` — the number of steps per episode.
A successful agent will show this value **decreasing** over time as it learns to reach the goal faster.

## Stop the Script

Press `Ctrl+C` to stop at any time.
