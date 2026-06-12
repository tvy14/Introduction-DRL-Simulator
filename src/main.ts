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

// ─── DOM refs (queried lazily) ────────────────────────────────────────────────

let envCanvas: HTMLCanvasElement | null = null;
let chartCanvas: HTMLCanvasElement | null = null;
let btnStart: HTMLButtonElement | null = null;
let btnReset: HTMLButtonElement | null = null;
let speedSlider: HTMLInputElement | null = null;
let speedLabel: HTMLSpanElement | null = null;
let statusEl: HTMLDivElement | null = null;
let statEpisode: HTMLSpanElement | null = null;
let statSteps: HTMLSpanElement | null = null;
let statReward: HTMLSpanElement | null = null;
let statAvg: HTMLSpanElement | null = null;
let statUpdates: HTMLSpanElement | null = null;
let statSuccess: HTMLSpanElement | null = null;

function getElements(): boolean {
  envCanvas = document.getElementById('env-canvas') as HTMLCanvasElement;
  chartCanvas = document.getElementById('chart-canvas') as HTMLCanvasElement;
  btnStart = document.getElementById('btn-start') as HTMLButtonElement;
  btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
  speedLabel = document.getElementById('speed-label') as HTMLSpanElement;
  statusEl = document.getElementById('status') as HTMLDivElement;
  statEpisode = document.getElementById('stat-episode') as HTMLSpanElement;
  statSteps = document.getElementById('stat-steps') as HTMLSpanElement;
  statReward = document.getElementById('stat-reward') as HTMLSpanElement;
  statAvg = document.getElementById('stat-avg') as HTMLSpanElement;
  statUpdates = document.getElementById('stat-updates') as HTMLSpanElement;
  statSuccess = document.getElementById('stat-success') as HTMLSpanElement;

  const allFound = !!envCanvas && !!chartCanvas && !!btnStart && !!btnReset &&
    !!speedSlider && !!speedLabel && !!statusEl && !!statEpisode && !!statSteps &&
    !!statReward && !!statAvg && !!statUpdates && !!statSuccess;

  if (!allFound) {
    console.error('Some DOM elements not found:', {
      envCanvas: !!envCanvas, chartCanvas: !!chartCanvas,
      btnStart: !!btnStart, btnReset: !!btnReset,
      speedSlider: !!speedSlider, speedLabel: !!speedLabel,
      statusEl: !!statusEl, statEpisode: !!statEpisode,
      statSteps: !!statSteps, statReward: !!statReward,
      statAvg: !!statAvg, statUpdates: !!statUpdates,
      statSuccess: !!statSuccess
    });
  }
  return allFound;
}

// ─── state ───────────────────────────────────────────────────────────────────

let env: MountainCar;
let agent: PPOAgent;
let buffer: Transition[] = [];

let episodeCount = 0;
let successCount = 0;
let updateCount = 0;
let epSteps = 0;
let epReward = 0;
let lastEpReward = 0;
let lastSuccess = false;
let rewardHistory: number[] = [];

let running = false;
let rafHandle = 0;

// ─── init ────────────────────────────────────────────────────────────────────

function init() {
  if (!getElements()) {
    setTimeout(init, 50);
    return;
  }

  env = new MountainCar();
  agent = new PPOAgent();
  buffer = [];

  episodeCount = 0;
  successCount = 0;
  updateCount = 0;
  epSteps = 0;
  epReward = 0;
  lastEpReward = 0;
  lastSuccess = false;
  rewardHistory = [];

  env.reset();
  updateUI();
  if (envCanvas) renderEnv(envCanvas, env.position, 0, false, false);
  if (chartCanvas) renderChart(chartCanvas!, []);
  setStatus('Ready. Press Start to begin training.');

  attachEventListeners();
}

function attachEventListeners() {
  btnStart!.addEventListener('click', () => {
    if (running) {
      running = false;
      cancelAnimationFrame(rafHandle);
      btnStart!.textContent = '▶ Resume';
      setStatus('Paused.');
    } else {
      running = true;
      btnStart!.textContent = '⏸ Pause';
      setStatus('Training…');
      rafHandle = requestAnimationFrame(trainFrame);
    }
  });

  btnReset!.addEventListener('click', () => {
    running = false;
    cancelAnimationFrame(rafHandle);
    btnStart!.textContent = '▶ Start';
    init();
  });

  speedSlider!.addEventListener('input', () => {
    speedLabel!.textContent = `${speedSlider!.value} steps/frame`;
  });

  // Initial render
  if (envCanvas) {
    env.reset();
    renderEnv(envCanvas, env.position, 0, false, false);
  }
}

// ─── training loop ───────────────────────────────────────────────────────────

function trainFrame() {
  if (!running) return;

  const stepsPerFrame = parseInt(speedSlider!.value, 10);

  for (let s = 0; s < stepsPerFrame; s++) {
    const rawState = env.state;
    const normState = MountainCar.normalise(rawState);

    const { action, logProb, value } = agent.act(normState);
    const { state: nextRaw, reward, done } = env.step(action as 0 | 1 | 2);

    buffer.push({ state: normState, action, reward, value, logProb, done });
    epSteps++;
    epReward += reward;

    if (done) {
      const success = env.position >= 0.5;
      if (success) successCount++;

      episodeCount++;
      lastEpReward = epReward;
      lastSuccess = success;
      rewardHistory.push(epReward);
      if (rewardHistory.length > 200) rewardHistory.shift();

      epSteps = 0;
      epReward = 0;
      env.reset();
    }

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
  if (envCanvas) renderEnv(envCanvas, env.position, lastEpReward, lastSuccess, running);
  if (chartCanvas && rewardHistory.length > 0) renderChart(chartCanvas, rewardHistory);
  updateUI();

  rafHandle = requestAnimationFrame(trainFrame);
}

// ─── UI helpers ──────────────────────────────────────────────────────────────

function updateUI() {
  if (!statEpisode) return;
  statEpisode.textContent = String(episodeCount);
  statSteps!.textContent = String(epSteps);
  statReward!.textContent = episodeCount > 0 ? lastEpReward.toFixed(0) : '—';
  statUpdates!.textContent = String(updateCount);
  statSuccess!.textContent = String(successCount);

  const n = Math.min(rewardHistory.length, 50);
  if (n > 0) {
    const avg = rewardHistory.slice(-n).reduce((a, b) => a + b, 0) / n;
    statAvg!.textContent = avg.toFixed(1);
  }
}

function setStatus(msg: string) {
  if (statusEl) statusEl.textContent = msg;
}

// ─── boot ────────────────────────────────────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
