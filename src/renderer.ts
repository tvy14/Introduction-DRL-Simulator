/**
 * Canvas renderer for MountainCar-v0.
 * Draws the mountain profile, goal flag, and the car.
 */

import { ENV, MountainCar } from './env';

const SAMPLES = 200; // mountain polyline resolution

/** Map environment x-position to canvas pixel x. */
function envToCanvasX(pos: number, W: number): number {
  return (pos - ENV.MIN_POS) / (ENV.MAX_POS - ENV.MIN_POS) * W;
}

/** Map mountain height [0..1] to canvas pixel y (y-axis is inverted). */
function heightToCanvasY(h: number, H: number): number {
  // Reserve 30 px margin at top and 10 px at bottom.
  return H - 10 - h * (H - 40);
}

function posToCanvas(pos: number, W: number, H: number): { x: number; y: number } {
  const h = MountainCar.height(pos);
  return { x: envToCanvasX(pos, W), y: heightToCanvasY(h, H) };
}

export function renderEnv(
  canvas: HTMLCanvasElement,
  position: number,
  episodeReward: number,
  success: boolean,
  training: boolean,
): void {
  const ctx = canvas.getContext('2d')!;
  const W   = canvas.width;
  const H   = canvas.height;

  // ── background ──────────────────────────────────────────────────────────
  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, '#0d1b2a');
  sky.addColorStop(1, '#1a2d42');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  // ── mountain ────────────────────────────────────────────────────────────
  ctx.beginPath();
  for (let i = 0; i <= SAMPLES; i++) {
    const pos = ENV.MIN_POS + (ENV.MAX_POS - ENV.MIN_POS) * (i / SAMPLES);
    const { x, y } = posToCanvas(pos, W, H);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();

  const ground = ctx.createLinearGradient(0, 0, 0, H);
  ground.addColorStop(0, '#4a7c59');
  ground.addColorStop(1, '#2d4a34');
  ctx.fillStyle = ground;
  ctx.fill();
  ctx.strokeStyle = '#6aaf7e';
  ctx.lineWidth   = 2;
  ctx.stroke();

  // ── goal flag ────────────────────────────────────────────────────────────
  const gp = posToCanvas(ENV.GOAL_POS, W, H);
  ctx.strokeStyle = '#ffffffcc';
  ctx.lineWidth   = 2;
  ctx.beginPath();
  ctx.moveTo(gp.x, gp.y);
  ctx.lineTo(gp.x, gp.y - 36);
  ctx.stroke();
  ctx.fillStyle = '#ff4444';
  ctx.beginPath();
  ctx.moveTo(gp.x,      gp.y - 36);
  ctx.lineTo(gp.x + 16, gp.y - 28);
  ctx.lineTo(gp.x,      gp.y - 20);
  ctx.closePath();
  ctx.fill();

  // ── car ──────────────────────────────────────────────────────────────────
  const cp   = posToCanvas(position, W, H);
  const cW   = 22;
  const cH   = 11;
  const tilt = Math.atan(3 * Math.cos(3 * position) * 0.45); // slope angle

  ctx.save();
  ctx.translate(cp.x, cp.y);
  ctx.rotate(-tilt);

  // Body
  const bodyColor = success ? '#44ff88' : (training ? '#7ecfff' : '#aaaaaa');
  ctx.fillStyle = bodyColor;
  ctx.beginPath();
  ctx.roundRect(-cW / 2, -cH - 6, cW, cH, 3);
  ctx.fill();

  // Wheels
  ctx.fillStyle = '#222';
  for (const wx of [-7, 7]) {
    ctx.beginPath();
    ctx.arc(wx, -3, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.restore();

  // ── overlay text ─────────────────────────────────────────────────────────
  ctx.font      = '12px monospace';
  ctx.fillStyle = '#ffffff88';
  ctx.fillText(`pos ${position.toFixed(3)}  vel ${(MountainCar.height(position) * 0).toFixed(0)}`, 8, 18);
}

// ─── Reward history chart ────────────────────────────────────────────────────

export function renderChart(
  canvas: HTMLCanvasElement,
  rewards: number[],
): void {
  const ctx = canvas.getContext('2d')!;
  const W   = canvas.width;
  const H   = canvas.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#12151e';
  ctx.fillRect(0, 0, W, H);

  if (rewards.length < 2) return;

  const pad  = { top: 16, bottom: 24, left: 48, right: 12 };
  const iW   = W - pad.left - pad.right;
  const iH   = H - pad.top  - pad.bottom;

  const min  = Math.min(...rewards);
  const max  = Math.max(...rewards);
  const span = max - min || 1;

  const toX  = (i: number) => pad.left + (i / (rewards.length - 1)) * iW;
  const toY  = (v: number) => pad.top  + (1 - (v - min) / span) * iH;

  // Axis lines
  ctx.strokeStyle = '#2a2d3a';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + iH);
  ctx.lineTo(pad.left + iW, pad.top + iH);
  ctx.stroke();

  // Y labels
  ctx.fillStyle = '#666';
  ctx.font      = '10px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(max.toFixed(0), pad.left - 4, pad.top + 4);
  ctx.fillText(min.toFixed(0), pad.left - 4, pad.top + iH);

  // Reward line
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + iH);
  grad.addColorStop(0, '#7ecfff');
  grad.addColorStop(1, '#2563eb');

  ctx.beginPath();
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 1.5;
  rewards.forEach((r, i) => {
    i === 0 ? ctx.moveTo(toX(i), toY(r)) : ctx.lineTo(toX(i), toY(r));
  });
  ctx.stroke();

  // X label
  ctx.fillStyle = '#555';
  ctx.textAlign = 'center';
  ctx.fillText(`episode reward (last ${rewards.length})`, pad.left + iW / 2, H - 4);
}
