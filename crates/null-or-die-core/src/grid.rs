//! Constant-tempo grid detection from raw audio.
//!
//! Pipeline:
//! 1. Spectral-flux onset envelope at a coarse hop.
//! 2. Whole-song tempo spectrum (zero-padded FFT of the envelope) to pick a
//!    BPM candidate inside the requested range.
//! 3. Per-block phase regression on the envelope to refine BPM and phase.
//! 4. Refinement with the bias kernel itself: a wide fingerprint walks the
//!    phase in, a per-block bias tilt fit corrects the BPM, and a final walk
//!    drives the global bias to ~0 ms (null sync).
//!
//! Variable tempo, stops, and downbeat/measure placement are out of scope.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use serde::Serialize;

use crate::bias::{BiasCfg, BiasEstimate, BiasRuntime, estimate_bias_with_beat_fn_reuse};

const ENV_HOP_SEC: f64 = 0.005;
const ENV_WINDOW_SEC: f64 = 0.046;
const ENV_DETREND_SEC: f64 = 1.0;
const LOW_BAND_HZ: f64 = 150.0;
const FOLD_BIN_SEC: f64 = 0.0025;
const BANDS: usize = 24;
const PITCH_WINDOW_SEC: f64 = 0.186;
const PITCH_HOP_SEC: f64 = 0.0116;
const MACRO_BANDS: usize = 4;
const TEMPO_POOL: usize = 5;
const TEMPO_MATCH_REL: f64 = 2e-3;
const MAX_PHASE_CANDIDATES: usize = 6;
const TEMPO_FFT_LEN: usize = 1 << 21;
const WIDE_FINGERPRINT_MS: f64 = 160.0;
const TILT_BLOCK_BEATS: usize = 32;
const WALK_DONE_MS: f64 = 0.2;
const WALK_MAX_STEPS: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct GridCfg {
    pub min_bpm: f64,
    pub max_bpm: f64,
    /// Skip tempo search and start refinement from this BPM.
    pub bpm_hint: Option<f64>,
    /// Snap to an integer BPM when that moves the last beat by at most this much.
    pub snap_ms: f64,
}

impl Default for GridCfg {
    fn default() -> Self {
        Self {
            min_bpm: 100.0,
            max_bpm: 200.0,
            bpm_hint: None,
            snap_ms: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TempoCandidate {
    pub bpm: f64,
    pub salience: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct GridStage {
    pub stage: &'static str,
    pub bpm: f64,
    pub first_beat_sec: f64,
    pub bias_ms: Option<f64>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GridEstimate {
    pub bpm: f64,
    /// Time of the first beat at or after t=0 (seconds).
    pub first_beat_sec: f64,
    /// Time of the first detected measure start (4/4 assumed), within the
    /// first four beats.
    pub first_downbeat_sec: f64,
    /// How far the chosen measure rotation beats the runner-up (summed
    /// z-scores of the harmonic-change and kick cues; ~0 means a toss-up).
    pub downbeat_margin: f64,
    /// Bias of the final grid as measured by the bias kernel.
    pub bias_ms: f64,
    pub confidence: f64,
    pub snapped: bool,
    pub duration_sec: f64,
    pub candidates: Vec<TempoCandidate>,
    /// Tempo candidates from the low-band (kick) envelope alone.
    pub kick_candidates: Vec<TempoCandidate>,
    pub stages: Vec<GridStage>,
}

impl GridEstimate {
    pub fn period_sec(&self) -> f64 {
        60.0 / self.bpm
    }
}

#[derive(Clone, Copy)]
struct Grid {
    period: f64,
    phase: f64,
}

impl Grid {
    fn bpm(self) -> f64 {
        60.0 / self.period
    }

    fn normalized(self) -> Self {
        Self {
            period: self.period,
            phase: self.phase.rem_euclid(self.period),
        }
    }

    fn beat_time(self, i: usize) -> f64 {
        self.phase + i as f64 * self.period
    }
}

pub fn detect_grid(
    audio_mono: &[f32],
    sample_rate_hz: u32,
    cfg: &GridCfg,
    bias_cfg: &BiasCfg,
) -> Result<GridEstimate, String> {
    if !(cfg.min_bpm > 0.0 && cfg.max_bpm > cfg.min_bpm) {
        return Err("invalid BPM range".to_string());
    }
    let duration_sec = audio_mono.len() as f64 / f64::from(sample_rate_hz);
    let env = onset_envelope(audio_mono, sample_rate_hz)?;
    let candidates = tempo_candidates(&env.values, env.hop_sec, cfg);
    let kick_candidates = tempo_candidates(&env.low, env.hop_sec, cfg);
    let start_bpm = match cfg.bpm_hint {
        Some(b) => b,
        None => choose_tempo(&candidates, &kick_candidates)
            .ok_or_else(|| "no tempo candidates found".to_string())?,
    };
    let mut stages = Vec::new();
    let mut grid = Grid {
        period: 60.0 / start_bpm,
        phase: 0.0,
    };
    grid.phase = choose_beat_phase(&env, grid.period);
    push_stage(&mut stages, "envelope", grid, None);

    let mut runtime = BiasRuntime::default();
    let wide = BiasCfg {
        fingerprint_ms: WIDE_FINGERPRINT_MS,
        ..*bias_cfg
    };
    let (g, est) = walk_bias(audio_mono, sample_rate_hz, &wide, &mut runtime, grid)?;
    grid = g;
    push_stage(&mut stages, "wide-walk", grid, Some(est));

    for _ in 0..4 {
        let Some(next) = tilt_fit(audio_mono, sample_rate_hz, bias_cfg, &mut runtime, grid)? else {
            break;
        };
        let moved = beats_in(duration_sec, next) as f64 * (next.period - grid.period).abs();
        grid = next.normalized();
        if moved < 0.5e-3 {
            break;
        }
    }
    let (g, mut est) = walk_bias(audio_mono, sample_rate_hz, bias_cfg, &mut runtime, grid)?;
    grid = g;
    push_stage(&mut stages, "tilt-fit", grid, Some(est));

    let mut snapped = false;
    let int_bpm = grid.bpm().round();
    let int_period = 60.0 / int_bpm;
    let drift_ms = beats_in(duration_sec, grid) as f64 * (int_period - grid.period).abs() * 1e3;
    if drift_ms <= cfg.snap_ms && (grid.bpm() - int_bpm).abs() > 1e-9 {
        let snap = Grid {
            period: int_period,
            phase: grid.phase,
        };
        let (g, e) = walk_bias(audio_mono, sample_rate_hz, bias_cfg, &mut runtime, snap)?;
        if e.confidence >= est.confidence - 0.03 {
            grid = g;
            est = e;
            snapped = true;
            push_stage(&mut stages, "snap", grid, Some(est));
        }
    }

    let (rotation, downbeat_margin) = choose_downbeat(audio_mono, sample_rate_hz, &env, grid);
    Ok(GridEstimate {
        bpm: grid.bpm(),
        first_beat_sec: grid.phase,
        first_downbeat_sec: grid.phase + rotation as f64 * grid.period,
        downbeat_margin,
        bias_ms: est.bias_ms,
        confidence: est.confidence,
        snapped,
        duration_sec,
        candidates,
        kick_candidates,
        stages,
    })
}

fn push_stage(
    stages: &mut Vec<GridStage>,
    stage: &'static str,
    g: Grid,
    est: Option<BiasEstimate>,
) {
    stages.push(GridStage {
        stage,
        bpm: g.bpm(),
        first_beat_sec: g.phase,
        bias_ms: est.map(|e| e.bias_ms),
        confidence: est.map(|e| e.confidence),
    });
}

fn beats_in(duration_sec: f64, g: Grid) -> usize {
    ((duration_sec - g.phase) / g.period).max(0.0) as usize
}

// ---------------------------------------------------------------------------
// Onset envelope

struct Envelope {
    values: Vec<f64>,
    /// Low-band (kick) flux.
    low: Vec<f64>,
    /// Detrended onset flux in four macro bands (low to high).
    macro_flux: Vec<Vec<f64>>,
    hop_sec: f64,
    /// Time of `values[0]`.
    t0_sec: f64,
}

impl Envelope {
    fn time(&self, i: usize) -> f64 {
        self.t0_sec + i as f64 * self.hop_sec
    }
}

fn onset_envelope(audio: &[f32], sample_rate_hz: u32) -> Result<Envelope, String> {
    let sr = f64::from(sample_rate_hz);
    let hop = (sr * ENV_HOP_SEC).round().max(1.0) as usize;
    let n = ((sr * ENV_WINDOW_SEC) as usize).next_power_of_two();
    if audio.len() < n + hop * 16 {
        return Err("audio too short for grid detection".to_string());
    }
    let frames = (audio.len() - n) / hop + 1;
    let bins = n / 2 + 1;
    let max_bin = ((11_000.0 / sr) * n as f64).min(bins as f64) as usize;
    let low_bin = ((LOW_BAND_HZ / sr) * n as f64).ceil().max(2.0) as usize;
    let window: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n);
    let mut buf = vec![Complex::new(0.0, 0.0); n];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let mut prev = vec![0.0; max_bin];
    let mut cur = vec![0.0; max_bin];
    let mut flux = Vec::with_capacity(frames);
    let mut low_flux = Vec::with_capacity(frames);
    let band_edges = band_edges(sr, n, max_bin);
    let mut bands = Vec::with_capacity(frames * BANDS);
    for f in 0..frames {
        let start = f * hop;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(f64::from(audio[start + i]) * window[i], 0.0);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        for (k, c) in cur.iter_mut().enumerate() {
            *c = (1.0 + 100.0 * buf[k].norm()).ln();
        }
        let (v, lv) = if f == 0 {
            (0.0, 0.0)
        } else {
            let rise = |k: usize| (cur[k] - prev[k]).max(0.0);
            ((0..max_bin).map(rise).sum(), (1..low_bin).map(rise).sum())
        };
        flux.push(v);
        low_flux.push(lv);
        for w in band_edges.windows(2) {
            let (a, b) = (w[0], w[1].max(w[0] + 1));
            bands.push((cur[a..b].iter().sum::<f64>() / (b - a) as f64) as f32);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let hop_sec = hop as f64 / sr;
    Ok(Envelope {
        values: detrend(&flux, (ENV_DETREND_SEC / hop_sec * 0.5) as usize),
        low: detrend(&low_flux, (ENV_DETREND_SEC / hop_sec * 0.5) as usize),
        macro_flux: macro_flux(&bands, (ENV_DETREND_SEC / hop_sec * 0.5) as usize),
        hop_sec,
        // Flux at frame f compares frames f-1 and f; attribute it to the
        // midpoint of their window centers.
        t0_sec: (n as f64 * 0.5 - hop as f64 * 0.5) / sr,
    })
}

fn macro_flux(bands: &[f32], half: usize) -> Vec<Vec<f64>> {
    let frames = bands.len() / BANDS;
    let per = BANDS / MACRO_BANDS;
    (0..MACRO_BANDS)
        .map(|m| {
            let raw: Vec<f64> = (0..frames)
                .map(|f| {
                    if f == 0 {
                        return 0.0;
                    }
                    (m * per..(m + 1) * per)
                        .map(|b| {
                            let d = bands[f * BANDS + b] - bands[(f - 1) * BANDS + b];
                            f64::from(d.max(0.0))
                        })
                        .sum()
                })
                .collect();
            detrend(&raw, half)
        })
        .collect()
}

/// `BANDS + 1` log-spaced bin edges from 40 Hz to `max_bin`.
fn band_edges(sr: f64, n: usize, max_bin: usize) -> Vec<usize> {
    let lo = (40.0 / sr * n as f64).max(1.0);
    let hi = max_bin as f64;
    (0..=BANDS)
        .map(|i| (lo * (hi / lo).powf(i as f64 / BANDS as f64)).round() as usize)
        .map(|b| b.min(max_bin - 1))
        .collect()
}

/// Subtract a moving average and half-wave rectify, removing slow loudness
/// trends so the tempo spectrum isn't dominated by DC.
fn detrend(flux: &[f64], half: usize) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(flux.len() + 1);
    prefix.push(0.0);
    for v in flux {
        prefix.push(prefix.last().unwrap() + v);
    }
    (0..flux.len())
        .map(|i| {
            let a = i.saturating_sub(half);
            let b = (i + half + 1).min(flux.len());
            (flux[i] - (prefix[b] - prefix[a]) / (b - a) as f64).max(0.0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tempo search

fn tempo_candidates(values: &[f64], hop_sec: f64, cfg: &GridCfg) -> Vec<TempoCandidate> {
    let len = TEMPO_FFT_LEN.max(values.len().next_power_of_two());
    let mut buf = vec![Complex::new(0.0, 0.0); len];
    for (b, v) in buf.iter_mut().zip(values) {
        *b = Complex::new(*v, 0.0);
    }
    FftPlanner::<f64>::new()
        .plan_fft_forward(len)
        .process(&mut buf);
    let mag: Vec<f64> = buf[..len / 2].iter().map(|c| c.norm()).collect();
    let bpm_per_bin = 60.0 / (len as f64 * hop_sec);
    let at = |bpm: f64| -> f64 {
        let x = bpm / bpm_per_bin;
        let i = x.floor() as usize;
        if i + 1 >= mag.len() {
            return 0.0;
        }
        let t = x - i as f64;
        mag[i] * (1.0 - t) + mag[i + 1] * t
    };
    // Harmonic salience: a beat-rate pulse train also has energy at 2x and 4x.
    let salience = |bpm: f64| at(bpm) + 0.5 * at(2.0 * bpm) + 0.25 * at(4.0 * bpm);
    let lo = (cfg.min_bpm / bpm_per_bin).floor() as usize;
    let hi = (cfg.max_bpm / bpm_per_bin).ceil() as usize;
    let scores: Vec<f64> = (lo..=hi + 1)
        .map(|i| salience(i as f64 * bpm_per_bin))
        .collect();
    let mut peaks = Vec::new();
    for j in 1..scores.len() - 1 {
        if scores[j] > scores[j - 1] && scores[j] >= scores[j + 1] {
            let (a, b, c) = (scores[j - 1], scores[j], scores[j + 1]);
            let denom = a - 2.0 * b + c;
            let delta = if denom.abs() > 1e-12 {
                (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let bpm = (lo + j) as f64 * bpm_per_bin + delta * bpm_per_bin;
            if bpm >= cfg.min_bpm && bpm <= cfg.max_bpm {
                peaks.push(TempoCandidate { bpm, salience: b });
            }
        }
    }
    peaks.sort_by(|a, b| b.salience.total_cmp(&a.salience));
    // Drop near-duplicates (sidelobes of a stronger peak).
    let mut out: Vec<TempoCandidate> = Vec::new();
    for p in peaks {
        if out.iter().all(|q| (q.bpm - p.bpm).abs() > 0.5) {
            out.push(p);
        }
        if out.len() == 8 {
            break;
        }
    }
    if let Some(top) = out.first().map(|c| c.salience) {
        for c in &mut out {
            c.salience /= top;
        }
    }
    out
}

/// Prefer a tempo both the full band and the kick band support: each band
/// alone falls into its own traps (full band: dotted 3/2 rhythms; kick band:
/// syncopated kick patterns at 5/4 or 3/2), but rarely the same one.
fn choose_tempo(full: &[TempoCandidate], kick: &[TempoCandidate]) -> Option<f64> {
    let salience = |list: &[TempoCandidate], bpm: f64| {
        list.iter()
            .filter(|c| (c.bpm / bpm - 1.0).abs() < TEMPO_MATCH_REL)
            .map(|c| c.salience)
            .fold(0.0, f64::max)
    };
    if kick.is_empty() {
        return full.first().map(|c| c.bpm);
    }
    full.iter()
        .take(TEMPO_POOL)
        .chain(kick.iter().take(TEMPO_POOL))
        .map(|c| {
            let (f, k) = (salience(full, c.bpm), salience(kick, c.bpm));
            // Full-band candidates carry the precise BPM; break ties toward them.
            (c.bpm, f.min(k) + 0.01 * f)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(bpm, _)| {
            full.iter()
                .find(|c| (c.bpm / bpm - 1.0).abs() < TEMPO_MATCH_REL)
                .map_or(bpm, |c| c.bpm)
        })
}

// ---------------------------------------------------------------------------
// Bias-kernel refinement

fn grid_bias(
    audio: &[f32],
    sr: u32,
    cfg: &BiasCfg,
    rt: &mut BiasRuntime,
    g: Grid,
    beats: Option<(usize, usize)>,
) -> Result<BiasEstimate, String> {
    let g = g.normalized();
    estimate_bias_with_beat_fn_reuse(audio, sr, cfg, rt, |i| match beats {
        Some((a, _)) if i < a => -1.0,
        Some((_, b)) if i >= b => f64::INFINITY,
        _ => g.beat_time(i),
    })
}

/// Shift the phase by the measured bias until it reads ~0.
fn walk_bias(
    audio: &[f32],
    sr: u32,
    cfg: &BiasCfg,
    rt: &mut BiasRuntime,
    mut g: Grid,
) -> Result<(Grid, BiasEstimate), String> {
    let mut est = grid_bias(audio, sr, cfg, rt, g, None)?;
    for _ in 0..WALK_MAX_STEPS {
        if est.bias_ms.abs() <= WALK_DONE_MS {
            break;
        }
        // Positive bias means the audio's onsets land later than the grid.
        g.phase += est.bias_ms * 1e-3;
        g = g.normalized();
        est = grid_bias(audio, sr, cfg, rt, g, None)?;
    }
    Ok((g, est))
}

/// Candidate beat phases: peaks of the onset envelope folded over one beat
/// period, from both the full band and the kick band.
fn phase_candidates(env: &Envelope, period: f64) -> Vec<f64> {
    let bins = ((period / FOLD_BIN_SEC).round() as usize).max(8);
    let fold = |values: &[f64]| {
        let mut acc = vec![0.0; bins];
        for (i, v) in values.iter().enumerate() {
            let pos = (env.time(i) / period).rem_euclid(1.0);
            acc[((pos * bins as f64) as usize).min(bins - 1)] += v;
        }
        // Circular smoothing over roughly +-7.5 ms.
        let half = ((0.0075 / FOLD_BIN_SEC) as usize).max(1);
        (0..bins)
            .map(|i| {
                (0..=2 * half)
                    .map(|k| acc[(i + bins + k - half) % bins])
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>()
    };
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for values in [&env.values, &env.low] {
        let folded = fold(values);
        let max = folded.iter().copied().fold(0.0, f64::max);
        if max <= 0.0 {
            continue;
        }
        for i in 0..bins {
            let (l, c, r) = (
                folded[(i + bins - 1) % bins],
                folded[i],
                folded[(i + 1) % bins],
            );
            if c > l && c >= r && c >= 0.25 * max {
                peaks.push(((i as f64 + 0.5) / bins as f64 * period, c / max));
            }
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut out: Vec<f64> = Vec::new();
    for (phase, _) in peaks {
        let near = out.iter().any(|q| {
            let d = (phase - q).rem_euclid(period);
            d.min(period - d) < 0.02
        });
        if !near {
            out.push(phase);
        }
        if out.len() == MAX_PHASE_CANDIDATES {
            break;
        }
    }
    if out.is_empty() {
        out.push(0.0);
    }
    out
}

/// Score each candidate by onset strength at its beat positions in four
/// frequency bands, combined as a geometric mean of each band's lift over its
/// average. A real beat tends to carry onsets across the spectrum; off-beat
/// hats or bass and syncopated leads usually light up only some bands.
fn choose_beat_phase(env: &Envelope, period: f64) -> f64 {
    let score = |phase: f64| -> f64 {
        env.macro_flux
            .iter()
            .map(|x| {
                let mean = x.iter().sum::<f64>() / x.len().max(1) as f64;
                (beat_mean(env, x, phase, period) / (mean + 1e-9) + 1e-9).ln()
            })
            .sum()
    };
    phase_candidates(env, period)
        .into_iter()
        .map(|p| (p, score(p)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(p, _)| p)
        .unwrap_or(0.0)
}

/// Mean over beats of the local maximum of `x` within +-12 ms of each beat.
fn beat_mean(env: &Envelope, x: &[f64], phase: f64, period: f64) -> f64 {
    let half = (0.012 / env.hop_sec).round() as isize;
    let (mut total, mut count) = (0.0, 0usize);
    let mut t = phase;
    loop {
        let c = ((t - env.t0_sec) / env.hop_sec).round() as isize;
        if c + half >= x.len() as isize {
            break;
        }
        if c - half >= 0 {
            total += x[(c - half) as usize..=(c + half) as usize]
                .iter()
                .copied()
                .fold(0.0, f64::max);
            count += 1;
        }
        t += period;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

// ---------------------------------------------------------------------------
// Measure placement

/// Pick which of the 4 beats in a measure starts it (4/4 assumed). Chords
/// and bass notes tend to change on downbeats and the kick tends to accent
/// them, so each rotation is scored by harmonic change (chroma, 4-beat
/// windows), bass-note change (2-beat windows) and kick-band onset strength,
/// each z-scored across the four rotations. Returns the rotation and its
/// margin over the runner-up (in summed z units).
fn choose_downbeat(audio: &[f32], sr: u32, env: &Envelope, g: Grid) -> (usize, f64) {
    let beats = beat_pitch_profiles(audio, sr, g);
    let n = beats.chroma.len();
    if n < 16 {
        return (0, 0.0);
    }
    let chroma = window_change(&beats.chroma, 4);
    let bass = window_change(&beats.bass, 2);
    let low = &env.macro_flux[0];
    let low_mean = low.iter().sum::<f64>() / low.len().max(1) as f64 + 1e-9;
    let half = (0.025 / env.hop_sec).round() as isize;
    let kick: Vec<f64> = (0..n)
        .map(|k| {
            let c = ((g.beat_time(k) - env.t0_sec) / env.hop_sec).round() as isize;
            let a = (c - half).max(0) as usize;
            let b = ((c + half + 1).max(0) as usize).min(low.len());
            low[a.min(b)..b].iter().copied().fold(0.0, f64::max) / low_mean
        })
        .collect();
    let mut total = [0.0; 4];
    for cue in [&chroma, &bass, &kick] {
        let mut by_rot = [0.0; 4];
        let mut counts = [0usize; 4];
        for (k, v) in cue.iter().enumerate() {
            by_rot[k % 4] += v;
            counts[k % 4] += 1;
        }
        for r in 0..4 {
            by_rot[r] /= counts[r].max(1) as f64;
        }
        let mean = by_rot.iter().sum::<f64>() / 4.0;
        let sd = (by_rot.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0).sqrt() + 1e-9;
        for r in 0..4 {
            total[r] += (by_rot[r] - mean) / sd;
        }
    }
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|a, b| total[*b].total_cmp(&total[*a]));
    (order[0], total[order[0]] - total[order[1]])
}

struct BeatProfiles {
    /// Per-beat mean chroma (80 Hz - 4 kHz).
    chroma: Vec<[f64; 12]>,
    /// Per-beat mean bass chroma (30 - 200 Hz).
    bass: Vec<[f64; 12]>,
}

/// Long-window STFT folded to pitch classes, averaged over each beat span.
fn beat_pitch_profiles(audio: &[f32], sr: u32, g: Grid) -> BeatProfiles {
    let srf = f64::from(sr);
    let n = 1usize << (srf * PITCH_WINDOW_SEC).log2().round() as u32;
    let hop = ((srf * PITCH_HOP_SEC).round() as usize).max(1);
    let empty = BeatProfiles {
        chroma: Vec::new(),
        bass: Vec::new(),
    };
    if audio.len() < n + hop {
        return empty;
    }
    let pitch_class = |k: usize, lo: f64, hi: f64| -> Option<usize> {
        let f = k as f64 * srf / n as f64;
        (f >= lo && f <= hi)
            .then(|| ((12.0 * (f / 440.0).log2()).round() as i64 + 9).rem_euclid(12) as usize)
    };
    let bins = n / 2 + 1;
    let chroma_pc: Vec<Option<usize>> = (0..bins).map(|k| pitch_class(k, 80.0, 4000.0)).collect();
    let bass_pc: Vec<Option<usize>> = (0..bins).map(|k| pitch_class(k, 30.0, 200.0)).collect();
    let last_bin = chroma_pc.iter().rposition(Option::is_some).unwrap_or(0) + 1;
    let window: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n);
    let mut buf = vec![Complex::new(0.0, 0.0); n];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

    let beats = ((audio.len() as f64 / srf - g.phase) / g.period)
        .floor()
        .max(0.0) as usize;
    let beats = beats.saturating_sub(1);
    let mut chroma = vec![[0.0; 12]; beats];
    let mut bass = vec![[0.0; 12]; beats];
    let mut counts = vec![0usize; beats];
    let frames = (audio.len() - n) / hop + 1;
    for f in 0..frames {
        let center = (f * hop) as f64 / srf + n as f64 * 0.5 / srf;
        let k = (center - g.phase) / g.period;
        if k < 0.0 || k as usize >= beats {
            continue;
        }
        let k = k as usize;
        let start = f * hop;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(f64::from(audio[start + i]) * window[i], 0.0);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        for bin in 0..last_bin {
            let m = buf[bin].norm().sqrt();
            if let Some(pc) = chroma_pc[bin] {
                chroma[k][pc] += m;
            }
            if let Some(pc) = bass_pc[bin] {
                bass[k][pc] += m;
            }
        }
        counts[k] += 1;
    }
    for k in 0..beats {
        let c = counts[k].max(1) as f64;
        for pc in 0..12 {
            chroma[k][pc] /= c;
            bass[k][pc] /= c;
        }
    }
    BeatProfiles { chroma, bass }
}

/// Cosine distance between the mean profile of the `w` beats before beat k
/// and the `w` beats from k on (0 where the windows don't fit).
fn window_change(profiles: &[[f64; 12]], w: usize) -> Vec<f64> {
    let n = profiles.len();
    let mut prefix = vec![[0.0; 12]; n + 1];
    for (k, p) in profiles.iter().enumerate() {
        for pc in 0..12 {
            prefix[k + 1][pc] = prefix[k][pc] + p[pc];
        }
    }
    (0..n)
        .map(|k| {
            if k < w || k + w > n {
                return 0.0;
            }
            let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
            let (lo, mid, hi) = (&prefix[k - w], &prefix[k], &prefix[k + w]);
            for ((l, m), h) in lo.iter().zip(mid).zip(hi) {
                let (a, b) = (m - l, h - m);
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            1.0 - dot / (na.sqrt() * nb.sqrt() + 1e-9)
        })
        .collect()
}

/// Per-block bias vs beat index; the slope is the per-beat period error.
fn tilt_fit(
    audio: &[f32],
    sr: u32,
    cfg: &BiasCfg,
    rt: &mut BiasRuntime,
    g: Grid,
) -> Result<Option<Grid>, String> {
    let duration = audio.len() as f64 / f64::from(sr);
    let total = beats_in(duration, g);
    if total < TILT_BLOCK_BEATS * 3 {
        return Ok(None);
    }
    let mut pts = Vec::new();
    let mut a = 0;
    while a + TILT_BLOCK_BEATS / 2 <= total {
        let b = (a + TILT_BLOCK_BEATS).min(total + 1);
        if let Ok(e) = grid_bias(audio, sr, cfg, rt, g, Some((a, b))) {
            let center = 0.5 * (a + b - 1) as f64;
            pts.push((center, e.bias_ms, e.confidence.powi(2) * e.conv_quint));
        }
        a = b;
    }
    if pts.len() < 3 {
        return Ok(None);
    }
    let (mut slope, mut intercept) = weighted_line(&pts);
    // One robust pass: drop blocks far from the fit (half-time or triplet
    // sections lock onto the wrong onsets).
    let resid: Vec<f64> = pts
        .iter()
        .map(|(x, y, _)| (y - (intercept + slope * x)).abs())
        .collect();
    let mad = median(&resid).max(0.5);
    let kept: Vec<_> = pts
        .iter()
        .zip(&resid)
        .filter(|(_, r)| **r <= 3.0 * mad)
        .map(|(p, _)| *p)
        .collect();
    if kept.len() >= 3 {
        (slope, intercept) = weighted_line(&kept);
    }
    Ok(Some(Grid {
        period: g.period + slope * 1e-3,
        phase: g.phase + intercept * 1e-3,
    }))
}

fn weighted_line(pts: &[(f64, f64, f64)]) -> (f64, f64) {
    let sw: f64 = pts.iter().map(|p| p.2).sum();
    if sw <= 0.0 {
        return (0.0, 0.0);
    }
    let mx = pts.iter().map(|p| p.0 * p.2).sum::<f64>() / sw;
    let my = pts.iter().map(|p| p.1 * p.2).sum::<f64>() / sw;
    let sxx: f64 = pts.iter().map(|p| p.2 * (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| p.2 * (p.0 - mx) * (p.1 - my)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (slope, my - slope * mx)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() { 0.0 } else { s[s.len() / 2] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BiasKernel, KernelTarget};

    fn bias_cfg() -> BiasCfg {
        BiasCfg {
            fingerprint_ms: 50.0,
            window_ms: 10.0,
            step_ms: 0.2,
            magic_offset_ms: 0.0,
            kernel_target: KernelTarget::Digest,
            kernel_type: BiasKernel::Rising,
            _full_spectrogram: false,
        }
    }

    /// Decaying noise bursts on a fixed grid, with weaker off-beat hats.
    fn click_track(sr: u32, bpm: f64, first: f64, secs: f64) -> Vec<f32> {
        let n = (f64::from(sr) * secs) as usize;
        let mut out = vec![0.0f32; n];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        let period = 60.0 / bpm;
        let mut add = |t: f64, amp: f64, decay: f64| {
            let s = (t * f64::from(sr)).round() as usize;
            for k in 0..(f64::from(sr) * 0.08) as usize {
                if s + k < n {
                    let env = (-(k as f64) / (decay * f64::from(sr))).exp();
                    out[s + k] += (amp * env * noise()) as f32;
                }
            }
        };
        let mut t = first;
        while t < secs {
            add(t, 0.6, 0.012);
            add(t + period * 0.5, 0.15, 0.004);
            t += period;
        }
        out
    }

    #[test]
    fn recovers_synthetic_grid() {
        let sr = 22_050;
        let bpm = 137.3;
        let first = 0.2345;
        let audio = click_track(sr, bpm, first, 60.0);
        let cfg = GridCfg {
            snap_ms: 0.0,
            ..GridCfg::default()
        };
        let est = detect_grid(&audio, sr, &cfg, &bias_cfg()).expect("grid");
        assert!((est.bpm - bpm).abs() < 0.01, "bpm {}", est.bpm);
        let period = 60.0 / bpm;
        let mut d = (est.first_beat_sec - first).rem_euclid(period);
        if d > period * 0.5 {
            d -= period;
        }
        // The bias kernel locks onto onset rise, a few ms after the nominal
        // click time; just check we landed on the beat, not the off-beat.
        assert!(d.abs() < 0.015, "phase error {d}");
    }
}
