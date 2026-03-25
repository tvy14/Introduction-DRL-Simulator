/**
 * Tiny Actor-Critic MLP — pure TypeScript, zero dependencies.
 *
 * Architecture: state(2) → Dense64(tanh) → Dense64(tanh)
 *                        ↳ actor  → Softmax(3)  (action probabilities)
 *                        ↳ critic → Linear(1)   (state value)
 *
 * Parameters are updated with mini-batch gradient descent + Adam.
 * Gradients are computed analytically (manual backprop).
 */

const H = 64; // hidden units (both layers)

// ─── math helpers ────────────────────────────────────────────────────────────

function zeros(n: number): Float32Array { return new Float32Array(n); }

/** Box-Muller normal samples, scaled by `s`. */
function randn(n: number, s: number): Float32Array {
  const v = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.random() + 1e-9, u2 = Math.random();
    const r   = Math.sqrt(-2 * Math.log(u1));
    v[i]           = r * Math.cos(2 * Math.PI * u2) * s;
    if (i + 1 < n) v[i + 1] = r * Math.sin(2 * Math.PI * u2) * s;
  }
  return v;
}

/** M[rows×cols] @ v[cols] → out[rows] */
function mv(M: Float32Array, v: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0, base = i * cols;
    for (let j = 0; j < cols; j++) s += M[base + j] * v[j];
    out[i] = s;
  }
  return out;
}

/** M.T @ v (M is rows×cols stored row-major) → out[cols] */
function mvT(M: Float32Array, v: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(cols);
  for (let i = 0; i < rows; i++) {
    const vi = v[i], base = i * cols;
    for (let j = 0; j < cols; j++) out[j] += M[base + j] * vi;
  }
  return out;
}

function softmax(v: Float32Array): Float32Array {
  const max = Math.max(v[0], v[1], v[2]);
  let sum = 0;
  const out = new Float32Array(3);
  for (let i = 0; i < 3; i++) { out[i] = Math.exp(v[i] - max); sum += out[i]; }
  for (let i = 0; i < 3; i++) out[i] /= sum;
  return out;
}

/** Adam in-place update. */
function adamStep(
  p: Float32Array, g: Float32Array,
  m: Float32Array, v: Float32Array, t: number,
  lr: number,
) {
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const bc1 = 1 - b1 ** t, bc2 = 1 - b2 ** t;
  for (let i = 0; i < p.length; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
  }
}

// ─── gradient accumulator ────────────────────────────────────────────────────

interface Grads {
  dW1: Float32Array; db1: Float32Array;   // 64×2, 64
  dW2: Float32Array; db2: Float32Array;   // 64×64, 64
  dWa: Float32Array; dba: Float32Array;   // 3×64, 3
  dWc: Float32Array; dbc: Float32Array;   // 64, 1
}

// ─── Actor-Critic network ─────────────────────────────────────────────────────

export class ActorCritic {
  W1: Float32Array; b1: Float32Array;
  W2: Float32Array; b2: Float32Array;
  Wa: Float32Array; ba: Float32Array;
  Wc: Float32Array; bc: Float32Array;   // Wc is a flat [H] vector (1×H critic)

  // Adam moments
  private mW1: Float32Array; private vW1: Float32Array;
  private mb1: Float32Array; private vb1: Float32Array;
  private mW2: Float32Array; private vW2: Float32Array;
  private mb2: Float32Array; private vb2: Float32Array;
  private mWa: Float32Array; private vWa: Float32Array;
  private mba: Float32Array; private vba: Float32Array;
  private mWc: Float32Array; private vWc: Float32Array;
  private mbc: Float32Array; private vbc: Float32Array;
  private _t = 0;

  constructor() {
    // Xavier init (fan-in scale)
    this.W1 = randn(H * 2,  Math.sqrt(1 / 2));   this.b1 = zeros(H);
    this.W2 = randn(H * H,  Math.sqrt(1 / H));   this.b2 = zeros(H);
    this.Wa = randn(3 * H,  Math.sqrt(1 / H));   this.ba = zeros(3);
    this.Wc = randn(H,      Math.sqrt(1 / H));   this.bc = zeros(1);

    const mk = (n: number): [Float32Array, Float32Array] => [zeros(n), zeros(n)];
    [this.mW1, this.vW1] = mk(H * 2);  [this.mb1, this.vb1] = mk(H);
    [this.mW2, this.vW2] = mk(H * H);  [this.mb2, this.vb2] = mk(H);
    [this.mWa, this.vWa] = mk(3 * H);  [this.mba, this.vba] = mk(3);
    [this.mWc, this.vWc] = mk(H);      [this.mbc, this.vbc] = mk(1);
  }

  /** Forward pass. Returns probs, value, and hidden activations for backprop. */
  forward(state: [number, number]): {
    probs: Float32Array; value: number; h1: Float32Array; h2: Float32Array;
  } {
    const x = new Float32Array(state);

    const h1 = mv(this.W1, x, H, 2);
    for (let i = 0; i < H; i++) h1[i] = Math.tanh(h1[i] + this.b1[i]);

    const h2 = mv(this.W2, h1, H, H);
    for (let i = 0; i < H; i++) h2[i] = Math.tanh(h2[i] + this.b2[i]);

    const logits = mv(this.Wa, h2, 3, H);
    for (let i = 0; i < 3; i++) logits[i] += this.ba[i];
    const probs = softmax(logits);

    let value = this.bc[0];
    for (let j = 0; j < H; j++) value += this.Wc[j] * h2[j];

    return { probs, value, h1, h2 };
  }

  /**
   * Run one PPO mini-batch update.
   * Gradients are accumulated over the batch then averaged before the Adam step.
   * Returns mean loss.
   */
  update(
    batch: Array<{
      state:     [number, number];
      action:    number;
      oldLogP:   number;
      advantage: number;
      return_:   number;
    }>,
    clipEps:     number,
    valueCoef:   number,
    entropyCoef: number,
    lr:          number,
  ): number {
    const B = batch.length;

    // Gradient accumulators (zeroed each call)
    const dW1 = zeros(H * 2); const db1 = zeros(H);
    const dW2 = zeros(H * H); const db2 = zeros(H);
    const dWa = zeros(3 * H); const dba = zeros(3);
    const dWc = zeros(H);     const dbc = zeros(1);

    let totalLoss = 0;

    for (const { state, action, oldLogP, advantage, return_ } of batch) {
      const { probs, value, h1, h2 } = this.forward(state);
      const x = new Float32Array(state);

      // ── PPO actor loss ─────────────────────────────────────────────────
      const newLogP = Math.log(probs[action] + 1e-8);
      const ratio   = Math.exp(newLogP - oldLogP);
      const surr1   = ratio * advantage;
      const surr2   = Math.min(Math.max(ratio, 1 - clipEps), 1 + clipEps) * advantage;
      const actorL  = -Math.min(surr1, surr2);

      // ── Critic loss ────────────────────────────────────────────────────
      const criticL = (value - return_) ** 2;

      // ── Entropy ───────────────────────────────────────────────────────
      let H_ent = 0;
      for (let k = 0; k < 3; k++) H_ent -= probs[k] * Math.log(probs[k] + 1e-8);

      totalLoss += actorL + valueCoef * criticL - entropyCoef * H_ent;

      // ── Gradient w.r.t. actor logits ──────────────────────────────────
      // Actor: d(-min(s1,s2))/d(new_logp) * d(new_logp)/d(logits)
      const g_logp = surr1 <= surr2 ? -ratio * advantage : 0;

      const dLogits = new Float32Array(3);
      for (let k = 0; k < 3; k++) {
        const ind = k === action ? 1 : 0;
        // Actor gradient
        dLogits[k] = g_logp * (ind - probs[k]);
        // Entropy gradient: d(-coef*H)/d(logits_k) = coef * p_k * (log(p_k) + H)
        dLogits[k] += entropyCoef * probs[k] * (Math.log(probs[k] + 1e-8) + H_ent);
      }

      // ── Gradient w.r.t. value ─────────────────────────────────────────
      const dValue = 2 * valueCoef * (value - return_);

      // ── Backprop through actor head ───────────────────────────────────
      for (let k = 0; k < 3; k++) {
        dba[k] += dLogits[k];
        const base = k * H;
        for (let j = 0; j < H; j++) dWa[base + j] += dLogits[k] * h2[j];
      }
      const dh2 = mvT(this.Wa, dLogits, 3, H);

      // ── Backprop through critic head ──────────────────────────────────
      dbc[0] += dValue;
      for (let j = 0; j < H; j++) {
        dWc[j]  += dValue * h2[j];
        dh2[j]  += dValue * this.Wc[j];
      }

      // ── Layer 2 ───────────────────────────────────────────────────────
      const dpre2 = new Float32Array(H);
      for (let i = 0; i < H; i++) dpre2[i] = dh2[i] * (1 - h2[i] * h2[i]);
      for (let i = 0; i < H; i++) {
        db2[i] += dpre2[i];
        const base = i * H;
        for (let j = 0; j < H; j++) dW2[base + j] += dpre2[i] * h1[j];
      }
      const dh1 = mvT(this.W2, dpre2, H, H);

      // ── Layer 1 ───────────────────────────────────────────────────────
      const dpre1 = new Float32Array(H);
      for (let i = 0; i < H; i++) dpre1[i] = dh1[i] * (1 - h1[i] * h1[i]);
      for (let i = 0; i < H; i++) {
        db1[i] += dpre1[i];
        for (let j = 0; j < 2; j++) dW1[i * 2 + j] += dpre1[i] * x[j];
      }
    }

    // Average gradients over batch
    const inv = 1 / B;
    for (let i = 0; i < dW1.length; i++) dW1[i] *= inv;
    for (let i = 0; i < dW2.length; i++) dW2[i] *= inv;
    for (let i = 0; i < dWa.length; i++) dWa[i] *= inv;
    for (let i = 0; i < dWc.length; i++) dWc[i] *= inv;
    for (let i = 0; i < db1.length; i++) db1[i] *= inv;
    for (let i = 0; i < db2.length; i++) db2[i] *= inv;
    for (let i = 0; i < dba.length; i++) dba[i] *= inv;
    dbc[0] *= inv;

    // Adam step
    this._t++;
    adamStep(this.W1, dW1, this.mW1, this.vW1, this._t, lr);
    adamStep(this.b1, db1, this.mb1, this.vb1, this._t, lr);
    adamStep(this.W2, dW2, this.mW2, this.vW2, this._t, lr);
    adamStep(this.b2, db2, this.mb2, this.vb2, this._t, lr);
    adamStep(this.Wa, dWa, this.mWa, this.vWa, this._t, lr);
    adamStep(this.ba, dba, this.mba, this.vba, this._t, lr);
    adamStep(this.Wc, dWc, this.mWc, this.vWc, this._t, lr);
    adamStep(this.bc, dbc, this.mbc, this.vbc, this._t, lr);

    return totalLoss / B;
  }
}
