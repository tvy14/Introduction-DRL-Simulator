/**
 * Main training loop.
 *
 * Architecture:
 *   - MountainCar env  (env.ts)
 *   - PPO agent        (ppo.ts)
 *   - Canvas renderer  (renderer.ts)
 *
 * The loop runs entirely in the browser; no server is needed.
 * Training speed is controlled by the "steps/frame" slider.
 */

import { MountainCar } from './env';
import { PPOAgent, type Transition } from './ppo';
import { renderEnv, renderChart } from './renderer';

// ─── DOM refs ────────────────────────────────────────────────────────────────

const envCanvas    = document.getElementById('env-canvas')    as HTMLCanvasElement;
const chartCanvas  = document.getElementById('chart-canvas')  as HTMLCanvasElement;
const btnStart     = document.getElementById('btn-start')     as HTMLButtonElement;
const btnReset     = document.getElementById('btn-reset')     as HTMLButtonElement;
const speedSlider  = document.getElementById('speed-slider')  as HTMLInputElement;
const speedLabel   = document.getElementById('speed-label')   as HTMLSpanElement;
const statusEl     = document.getElementById('status')        as HTMLDivElement;

const statEpisode  = document.getElementById('stat-episode')  as HTMLSpanElement;
const statSteps    = document.getElementById('stat-steps')    as HTMLSpanElement;
const statReward   = document.getElementById('stat-reward')   as HTMLSpanElement;
const statAvg      = document.getElementById('stat-avg')      as HTMLSpanElement;
const statUpdates  = document.getElementById('stat-updates')  as HTMLSpanElement;
const statSuccess  = document.getElementById('stat-success')  as HTMLSpanElement;

// ─── state ───────────────────────────────────────────────────────────────────

let env:      MountainCar;
let agent:    PPOAgent;
let buffer:   Transition[] = [];

let episodeCount  = 0;
let successCount  = 0;
let updateCount   = 0;
let epSteps       = 0;
let epReward      = 0;
let lastEpReward  = 0;
let lastSuccess   = false;
let rewardHistory: number[] = [];   // per-episode totals (capped at 200)

let running       = false;
let rafHandle     = 0;

// ─── init ────────────────────────────────────────────────────────────────────

function init() {
  env   = new MountainCar();
  agent = new PPOAgent();
  buffer = [];

  episodeCount = 0;
  successCount = 0;
  updateCount  = 0;
  epSteps      = 0;
  epReward     = 0;
  lastSuccess  = false;
  rewardHistory = [];

  env.reset();
  updateUI();
  renderEnv(envCanvas, env.position, 0, false, false);
  renderChart(chartCanvas, []);
  setStatus('Ready. Press Start to begin training.');
}

// ─── training loop ───────────────────────────────────────────────────────────

function trainFrame() {
  if (!running) return;

  const stepsPerFrame = parseInt(speedSlider.value, 10);

  for (let s = 0; s < stepsPerFrame; s++) {
    const rawState = env.state;
    const normState = MountainCar.normalise(rawState);

    // Agent picks action
    const { action, logProb, value } = agent.act(normState);

    // Step env
    const { state: nextRaw, reward, done } = env.step(action as 0 | 1 | 2);

    buffer.push({ state: normState, action, reward, value, logProb, done });
    epSteps++;
    epReward += reward;

    if (done) {
      const success = env.position >= 0.5;
      if (success) successCount++;

      episodeCount++;
      lastEpReward = epReward;
      lastSuccess  = success;
      rewardHistory.push(epReward);
      if (rewardHistory.length > 200) rewardHistory.shift();

      epSteps  = 0;
      epReward = 0;
      env.reset();
    }

    // PPO update when buffer is full
    if (buffer.length >= agent.nSteps) {
      const lastVal = env.position >= 0.5 ? 0 :
        agent.getValue(MountainCar.normalise(env.state));

      const loss = agent.update(buffer, lastVal);
      buffer = [];
      updateCount++;
      setStatus(
        `Update #${updateCount} — loss ${loss.toFixed(4)} ` +
        `| success rate ${(successCount / Math.max(episodeCount, 1) * 100).toFixed(1)} %`
      );
    }
  }

  // Re-render once per frame
  renderEnv(envCanvas, env.position, lastEpReward, lastSuccess, running);
  if (rewardHistory.length > 0) renderChart(chartCanvas, rewardHistory);
  updateUI();

  rafHandle = requestAnimationFrame(trainFrame);
}

// ─── UI helpers ──────────────────────────────────────────────────────────────

function updateUI() {
  statEpisode.textContent = String(episodeCount);
  statSteps.textContent   = String(epSteps);
  statReward.textContent  = episodeCount > 0 ? lastEpReward.toFixed(0) : '—';
  statUpdates.textContent = String(updateCount);
  statSuccess.textContent = String(successCount);

  const n = Math.min(rewardHistory.length, 50);
  if (n > 0) {
    const avg = rewardHistory.slice(-n).reduce((a, b) => a + b, 0) / n;
    statAvg.textContent = avg.toFixed(1);
  }
}

function setStatus(msg: string) {
  statusEl.textContent = msg;
}

// ─── button handlers ─────────────────────────────────────────────────────────

btnStart.addEventListener('click', () => {
  if (running) {
    // Pause
    running = false;
    cancelAnimationFrame(rafHandle);
    btnStart.textContent = '▶ Resume';
    setStatus('Paused.');
  } else {
    // Start / resume
    running = true;
    btnStart.textContent = '⏸ Pause';
    setStatus('Training…');
    rafHandle = requestAnimationFrame(trainFrame);
  }
});

btnReset.addEventListener('click', () => {
  running = false;
  cancelAnimationFrame(rafHandle);
  btnStart.textContent = '▶ Start';
  init();
});

speedSlider.addEventListener('input', () => {
  speedLabel.textContent = `${speedSlider.value} steps/frame`;
});

// ─── boot ────────────────────────────────────────────────────────────────────

init();
