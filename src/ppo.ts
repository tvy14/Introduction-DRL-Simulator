/**
 * PPO (Proximal Policy Optimisation) — pure TypeScript, no external deps.
 * Uses ActorCritic from net.ts for the policy/value network.
 */

import { ActorCritic } from './net';
import type { State } from './env';

// ─── helpers ─────────────────────────────────────────────────────────────────

function sampleCategorical(probs: Float32Array): number {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function shuffleIndices(n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

// ─── transition ───────────────────────────────────────────────────────────────

export interface Transition {
  state:   State;
  action:  number;
  reward:  number;
  value:   number;
  logProb: number;
  done:    boolean;
}

// ─── PPO agent ────────────────────────────────────────────────────────────────

export class PPOAgent {
  readonly gamma       = 0.99;
  readonly lam         = 0.95;
  readonly clipEps     = 0.2;
  readonly valueCoef   = 0.5;
  readonly entropyCoef = 0.01;
  readonly nEpochs     = 4;
  readonly batchSize   = 64;
  readonly nSteps      = 1024;
  readonly lr          = 3e-4;

  private net: ActorCritic;

  constructor() { this.net = new ActorCritic(); }

  act(state: State): { action: number; logProb: number; value: number } {
    const { probs, value } = this.net.forward(state);
    const action = sampleCategorical(probs);
    return { action, logProb: Math.log(probs[action] + 1e-8), value };
  }

  getValue(state: State): number {
    return this.net.forward(state).value;
  }

  update(transitions: Transition[], lastValue: number): number {
    const n = transitions.length;

    // ── GAE advantage estimation ──────────────────────────────────────────
    const advantages = new Float32Array(n);
    const returns    = new Float32Array(n);
    let gae = 0, nextVal = lastValue;

    for (let t = n - 1; t >= 0; t--) {
      const { reward, value, done } = transitions[t];
      const mask = done ? 0 : 1;
      const delta = reward + this.gamma * nextVal * mask - value;
      gae = delta + this.gamma * this.lam * mask * gae;
      advantages[t] = gae;
      returns[t]    = gae + value;
      nextVal       = value;
    }

    // ── Normalise advantages ──────────────────────────────────────────────
    let mean = 0;
    for (let i = 0; i < n; i++) mean += advantages[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (advantages[i] - mean) ** 2;
    const std = Math.sqrt(variance / n) + 1e-8;
    for (let i = 0; i < n; i++) advantages[i] = (advantages[i] - mean) / std;

    // ── PPO epochs ────────────────────────────────────────────────────────
    let totalLoss = 0;
    const nBatches = Math.max(1, Math.floor(n / this.batchSize));

    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      const idx = shuffleIndices(n);
      for (let b = 0; b < nBatches; b++) {
        const batch = idx
          .slice(b * this.batchSize, (b + 1) * this.batchSize)
          .map(i => ({
            state:     transitions[i].state,
            action:    transitions[i].action,
            oldLogP:   transitions[i].logProb,
            advantage: advantages[i],
            return_:   returns[i],
          }));
        totalLoss += this.net.update(
          batch, this.clipEps, this.valueCoef, this.entropyCoef, this.lr,
        );
      }
    }

    return totalLoss / (this.nEpochs * nBatches);
  }
}
