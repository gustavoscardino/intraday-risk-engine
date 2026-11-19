// Forecast sources and the replay player.
//
// A source delivers forecast records in time order:
//   { i, t, y, evaluated, R, RV, q: [quantile of each model] }
// t is the forecast time (exchange clock, seconds), y the realized return of the target bar
// (null while pending), q the log-return quantiles (negative = loss). Every view of the replay tab
// reads records through this interface only.
//
// ReplaySource (level B) serves a run the WebAssembly engine has just computed. A LiveSource
// (level C, see LEVEL_C.md) would implement the same `size` / `record(i)` / `subscribe(fn)` and
// push a record whenever a bar closes on a live market-data feed.

export class ReplaySource {
  constructor(run) {
    this.run = run;
    this.models = run.models;
    this.size = run.forecasts.t.length;
    this.listeners = new Set();
  }

  record(i) {
    const f = this.run.forecasts;
    return { i, t: f.t[i], y: f.y[i], evaluated: f.eval[i] === 1, R: f.R[i], RV: f.RV[i], q: f.q.map((c) => c[i]) };
  }

  // A replay has every record from the start; subscribers are told when playback moves.
  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(i) { this.listeners.forEach((fn) => fn(i)); }
}

// Moves a playhead over a source at `speed` records per second (requestAnimationFrame).
export class Player {
  constructor(source, { onFrame }) {
    this.source = source;
    this.onFrame = onFrame;
    this.pos = 0;
    this.speed = 20;
    this.playing = false;
    this.last = 0;
    this.tick = this.tick.bind(this);
  }

  play() {
    if (this.playing) return;
    if (this.pos >= this.source.size - 1) this.pos = 0;
    this.playing = true;
    this.last = performance.now();
    requestAnimationFrame(this.tick);
  }

  pause() { this.playing = false; this.onFrame(Math.floor(this.pos), false); }
  setSpeed(recordsPerSecond) { this.speed = recordsPerSecond; }

  seek(i) {
    this.pos = Math.max(0, Math.min(this.source.size - 1, i));
    this.source.emit(Math.floor(this.pos));
    this.onFrame(Math.floor(this.pos), this.playing);
  }

  tick(now) {
    if (!this.playing) return;
    const before = Math.floor(this.pos);
    this.pos = Math.min(this.source.size - 1, this.pos + ((now - this.last) / 1000) * this.speed);
    this.last = now;
    const at = Math.floor(this.pos);
    if (at !== before) { this.source.emit(at); this.onFrame(at, true); }
    if (this.pos >= this.source.size - 1) { this.playing = false; this.onFrame(at, false); return; }
    requestAnimationFrame(this.tick);
  }
}
