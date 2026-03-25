// MountainCar-v0 — matches OpenAI Gym dynamics exactly.
// Physics reference: gym/envs/classic_control/mountain_car.py

export const ENV = {
  MIN_POS:       -1.2,
  MAX_POS:        0.6,
  MIN_VEL:       -0.07,
  MAX_VEL:        0.07,
  GOAL_POS:       0.5,
  GOAL_VEL:       0.0,   // v0 has no velocity requirement
  GRAVITY:        0.0025,
  POWER:          0.001,
  MAX_STEPS:      200,
} as const;

export type State = [number, number]; // [position, velocity]

export class MountainCar {
  position = 0;
  velocity = 0;
  steps    = 0;

  /** Returns initial state. */
  reset(): State {
    this.position = Math.random() * 0.2 - 0.6; // uniform in [-0.6, -0.4]
    this.velocity = 0;
    this.steps    = 0;
    return [this.position, this.velocity];
  }

  /**
   * Actions: 0 = push left, 1 = no push, 2 = push right.
   * Returns next state, reward (-1 per step), and done flag.
   */
  step(action: 0 | 1 | 2): { state: State; reward: number; done: boolean } {
    const force = action - 1; // maps 0,1,2 → -1,0,+1

    this.velocity += force * ENV.POWER
      - Math.cos(3 * this.position) * ENV.GRAVITY;
    this.velocity = Math.max(ENV.MIN_VEL, Math.min(ENV.MAX_VEL, this.velocity));

    this.position += this.velocity;
    this.position = Math.max(ENV.MIN_POS, Math.min(ENV.MAX_POS, this.position));

    if (this.position === ENV.MIN_POS && this.velocity < 0) {
      this.velocity = 0;
    }

    this.steps++;
    const done = (this.position >= ENV.GOAL_POS) || (this.steps >= ENV.MAX_STEPS);

    return {
      state:  [this.position, this.velocity],
      reward: -1.0,
      done,
    };
  }

  get state(): State { return [this.position, this.velocity]; }

  /** Normalise state to roughly [-1, 1] for the network input. */
  static normalise([pos, vel]: State): State {
    return [
      (pos - (ENV.MIN_POS + ENV.MAX_POS) / 2) / ((ENV.MAX_POS - ENV.MIN_POS) / 2),
      vel / ENV.MAX_VEL,
    ];
  }

  /** Mountain height at position x (matches Gym renderer). */
  static height(x: number): number {
    return Math.sin(3 * x) * 0.45 + 0.55;
  }
}
