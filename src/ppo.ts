/**
 * PPO (Proximal Policy Optimisation) — Actor-Critic variant.
 *
 * Architecture:  state(2) → Dense64(tanh) → Dense64(tanh)
 *                         ↳ actor head  → Softmax(3)   (action probabilities)
 *                         ↳ critic head → Linear(1)    (state value)
 *
 * Hyper-parameters are kept close to the Stable-Baselines3 defaults for
 * MountainCar so results are comparable with the Python version.
 */

import * as tf from '@tensorflow/tfjs';
import type { State } from './env';

// ─── helpers ────────────────────────────────────────────────────────────────

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

// ─── transition buffer ───────────────────────────────────────────────────────

export interface Transition {
  state:   State;
  action:  number;
  reward:  number;
  value:   number;
  logProb: number;
  done:    boolean;
}

// ─── agent ───────────────────────────────────────────────────────────────────

export class PPOAgent {
  // Hyper-parameters
  readonly gamma      = 0.99;
  readonly lam        = 0.95;   // GAE λ
  readonly clipEps    = 0.2;
  readonly valueCoef  = 0.5;
  readonly entropyCoef = 0.01;
  readonly nEpochs    = 4;
  readonly batchSize  = 64;
  readonly nSteps     = 2048;   // rollout length before an update

  private model:     tf.LayersModel;
  private optimizer: tf.Optimizer;

  constructor() {
    this.model     = this._buildModel();
    this.optimizer = tf.train.adam(3e-4);
  }

  private _buildModel(): tf.LayersModel {
    const inp   = tf.input({ shape: [2] });
    let   x: tf.SymbolicTensor = tf.layers.dense({ units: 64, activation: 'tanh' }).apply(inp) as tf.SymbolicTensor;
    x = tf.layers.dense({ units: 64, activation: 'tanh' }).apply(x) as tf.SymbolicTensor;

    const actor  = tf.layers.dense({ units: 3, activation: 'softmax', name: 'actor'  }).apply(x) as tf.SymbolicTensor;
    const critic = tf.layers.dense({ units: 1,                        name: 'critic' }).apply(x) as tf.SymbolicTensor;

    return tf.model({ inputs: inp, outputs: [actor, critic] });
  }

  /** Sample an action and return log-prob + value estimate. */
  act(state: State): { action: number; logProb: number; value: number } {
    return tf.tidy(() => {
      const s = tf.tensor2d([state]);
      const [probs, val] = this.model.predict(s) as [tf.Tensor2D, tf.Tensor2D];
      const probsArr = probs.dataSync() as Float32Array;
      const action   = sampleCategorical(probsArr);
      return {
        action,
        logProb: Math.log(probsArr[action] + 1e-8),
        value:   val.dataSync()[0],
      };
    });
  }

  /** Critic-only forward pass (used for bootstrap at rollout end). */
  getValue(state: State): number {
    return tf.tidy(() => {
      const s = tf.tensor2d([state]);
      const [, val] = this.model.predict(s) as [tf.Tensor2D, tf.Tensor2D];
      return val.dataSync()[0];
    });
  }

  /**
   * Run one PPO update on the provided rollout buffer.
   * Returns average loss across all mini-batches.
   */
  update(transitions: Transition[], lastValue: number): number {
    const n = transitions.length;

    // ── 1. Generalised Advantage Estimation ──────────────────────────────
    const advantages = new Float32Array(n);
    const returns    = new Float32Array(n);
    let gae     = 0;
    let nextVal = lastValue;

    for (let t = n - 1; t >= 0; t--) {
      const { reward, value, done } = transitions[t];
      const mask  = done ? 0 : 1;
      const delta = reward + this.gamma * nextVal * mask - value;
      gae         = delta + this.gamma * this.lam * mask * gae;
      advantages[t] = gae;
      returns[t]    = gae + value;
      nextVal       = value;
    }

    // ── 2. Normalise advantages ───────────────────────────────────────────
    let mean = 0;
    for (let i = 0; i < n; i++) mean += advantages[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (advantages[i] - mean) ** 2;
    const std = Math.sqrt(variance / n) + 1e-8;
    for (let i = 0; i < n; i++) advantages[i] = (advantages[i] - mean) / std;

    // ── 3. Pre-extract arrays ─────────────────────────────────────────────
    const states      = transitions.map(t => t.state);
    const actions     = transitions.map(t => t.action);
    const oldLogProbs = transitions.map(t => t.logProb);
    const vars        = this.model.trainableWeights as tf.Variable[];

    let totalLoss = 0;
    const nBatches = Math.max(1, Math.floor(n / this.batchSize));

    // ── 4. PPO epochs ─────────────────────────────────────────────────────
    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      const idx = shuffleIndices(n);

      for (let b = 0; b < nBatches; b++) {
        const bIdx = idx.slice(b * this.batchSize, (b + 1) * this.batchSize);
        const bs   = bIdx.length;

        const loss = this.optimizer.minimize(() => {
          const sTensor = tf.tensor2d(bIdx.map(i => states[i]));
          const [probs, vals] = this.model.predict(sTensor) as [tf.Tensor2D, tf.Tensor2D];

          // Actor loss (clipped surrogate objective)
          const aMask      = tf.oneHot(tf.tensor1d(bIdx.map(i => actions[i]),    'int32'), 3) as tf.Tensor2D;
          const selProbs   = tf.sum(tf.mul(probs, aMask), 1);
          const newLogP    = tf.log(tf.add(selProbs, 1e-8));
          const oldLogP    = tf.tensor1d(bIdx.map(i => oldLogProbs[i]));
          const ratio      = tf.exp(tf.sub(newLogP, oldLogP));
          const adv        = tf.tensor1d(Array.from(bIdx.map(i => advantages[i])));
          const surr1      = tf.mul(ratio, adv);
          const surr2      = tf.mul(tf.clipByValue(ratio, 1 - this.clipEps, 1 + this.clipEps), adv);
          const actorLoss  = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

          // Critic loss
          const ret        = tf.tensor2d(bIdx.map(i => [returns[i]]), [bs, 1]);
          const criticLoss = tf.mul(tf.scalar(this.valueCoef),
                               tf.mean(tf.square(tf.sub(vals, ret))));

          // Entropy bonus (maximise → subtract from loss)
          const entropy      = tf.neg(tf.mean(tf.sum(tf.mul(probs, tf.log(tf.add(probs, 1e-8))), 1)));
          const entropyLoss  = tf.mul(tf.scalar(-this.entropyCoef), entropy);

          return tf.add(tf.add(actorLoss, criticLoss), entropyLoss) as tf.Scalar;
        }, /* returnCost */ true, vars);

        if (loss) {
          totalLoss += loss.dataSync()[0];
          loss.dispose();
        }
      }
    }

    return totalLoss / (this.nEpochs * nBatches);
  }
}
