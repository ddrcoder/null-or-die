# null-or-die

Terminal-first Rust reimplementation scaffold for `nine-or-null` parity work.

## Why This Project Exists

`null-or-die` is a rewrite of [`nine-or-null`](https://github.com/telperion/nine-or-null) by [@telperion](https://github.com/telperion).

Why:

1. Performance advantages from implementing the sync-analysis pipeline in a lower-level language (Rust).
2. Acting as a submodule/integration target for [`deadsync`](https://github.com/pnn64/deadsync), the Rust rewrite of ITGmania/StepMania.

## Current status

- `null-or-die analyze <path> [--plot]`: scans simfiles, parses chart metadata through `rssp`, decodes OGG audio, and computes native bias metrics per chart. With `--plot`, also writes nine-or-null-style PNGs (`bias-freqdomain-*`, `bias-beatdigest-*`, `bias-postkernel-*`) into the report directory.
- `null-or-die parity <path> --baseline <dir>`: validates MD5-sharded fixtures and checks native bias outputs against baseline chart rows (including split `#MUSIC` rows).
- `null-or-die harness <path> --baseline <dir>`: runs Python `nine-or-null` reference analysis and writes canonical `json.zst` fixtures.
- `null-or-die bench <simfile>`: runs repeated analyze-style passes on one simfile and reports phase timings (read, parse, decode, bias, total).
- `null-or-die plot <input.json> <out.png>`: draws bias markers from JSON (`bias_ms`, `bias_result`, or `bias`).
- `null-or-die grid <audio>`: detects a constant-BPM grid (BPM, beat phase, measure start) from raw audio with no chart, and optionally writes a minimal `.ssc`. See [Grid detection](#grid-detection).

`analyze` and `parity` now share the native bias math path.

For `harness`, `--source-root` should point to the Python package root containing `nine_or_null/` (for example `nine-or-null-0.8.0/nine-or-null`). If omitted, `null-or-die` auto-detects that sibling path from the current working directory.

## Baseline layout

MD5-sharded baseline lookup matches the existing `rssp` corpus style:

`<baseline>/<md5[0..2]>/<md5>.json` or `<baseline>/<md5[0..2]>/<md5>.json.zst`

MD5 is computed from raw simfile bytes.

Baseline chart rows include a `music` field (chart `#MUSIC` if present, else simfile `#MUSIC`) so split-audio parity can target the correct OGG per chart.

## Examples

```bash
cargo run -- analyze /path/to/Songs --output /tmp/null-or-die-scan.json
cargo run -- analyze /path/to/song.sm --plot --report-path /tmp/null-or-die-plots
# legacy-style invocation is also supported:
cargo run -- --analyze /path/to/song.sm --plot --report-path /tmp/null-or-die-plots
cargo run -- parity /path/to/Songs --baseline /path/to/baseline --fail-on-missing --fail-on-mismatch
cargo run -- harness /path/to/Songs --baseline /path/to/baseline --source-root /path/to/nine-or-null-0.8.0/nine-or-null
cargo run --release -- bench "/path/to/PEMDMonium.sm" --warmup 3 --iterations 20
cargo run -- plot /tmp/null-or-die-scan.json /tmp/bias.png --span-ms 20
cargo run --release -- grid /path/to/song.ogg --ssc /path/to/song.ssc --beat-chart
```

`analyze` creates its report directory (default `<path>/__bias-check`) even without `--plot`; pass `--report-path` to keep song folders clean.

## Grid detection

`grid` estimates a constant-tempo grid from audio alone. Any format ffmpeg can decode is accepted. OGG uses the parity decoder, with ffmpeg as a fallback.

```bash
cargo run --release -- grid song.ogg                          # JSON report to stdout
cargo run --release -- grid song.ogg --ssc song.ssc           # also write a minimal .ssc
cargo run --release -- grid song.ogg --ssc song.ssc --beat-chart --paradigm itg
```

Options:

- `--ssc <path>`: write `#TITLE/#ARTIST/#MUSIC/#OFFSET/#BPMS`; `#MUSIC` is the audio file name, so put the `.ssc` next to the audio. Refuses to overwrite without `--force`.
- `--beat-chart`: add a Beginner dance-single chart with a tap on every beat (playable, and usable by `analyze`/the SYNC screen).
- `--min-bpm` / `--max-bpm` (default 100-200): tempo search range. Half/double tempo is a charting convention, so set this (or `--bpm`) for slow or fast songs.
- `--bpm <hint>`: skip tempo search and refine from this BPM.
- `--snap-ms <ms>` (default 3): snap to an integer BPM if that moves the last beat by at most this much.
- `--paradigm null|itg`: offset paradigm for the written `#OFFSET` (`itg` adds 9 ms).
- `--title`, `--artist`, `-o <report.json>`.

The report includes `bpm`, `offset_sec`, `first_beat_sec`, `first_downbeat_sec`, `downbeat_margin`, the final `bias_ms`/`confidence` from the bias kernel, tempo candidates, and a per-stage trace.

Pipeline (all signal processing, no learned models):

1. Spectral-flux onset envelope (5 ms hop), full band and kick band.
2. Whole-song tempo spectrum; pick a tempo both bands support.
3. Beat phase: among peaks of the envelope folded over one beat, take the one with the strongest onsets across four frequency bands.
4. Refinement with the bias kernel: a wide-fingerprint walk, a per-block bias tilt fit to correct BPM, then a walk to |bias| <= 0.2 ms (null sync).
5. Measure start: score the four beat rotations by chroma change, bass-chroma change and kick strength.

Accuracy on 271 constant-BPM songs (232 ITG official charts, 39 Geometry Dash):

- Tempo: correct for nearly all songs whose BPM is inside the search range. Dotted rhythms can still lock to 3/2 tempo.
- Fine sync: at least as good as official ITG timing by null-or-die's own confidence.
- Measure start: ~91% on ITG when the beat phase is right. Treat `downbeat_margin` below ~1.5 as uncertain.
- Main weakness: on roughly a quarter of songs the beat lands on a subdivision (1/2, 1/4 or 1/3 beat off). The bias kernel's confidence can't detect this, so check by ear or cross-check with another beat tracker.

Out of scope: BPM changes, stops, warps, and non-4/4 meters. Expect ~15-20 s per song.

## Library API (for deadsync integration)

`null-or-die` now exposes a direct Rust API for in-engine sync tooling:

- `null_or_die::api::inspect_simfile(path)` -> chart metadata for selection/UI
- `null_or_die::api::analyze_chart(path, chart_index, &cfg)` -> full bias estimate + full graph matrices
- `null_or_die::api::analyze_chart_stream(path, chart_index, &cfg, stream_cfg, on_event)` -> incremental events while processing (`Init`, `Beat`, `Convolution`, `Done`) plus final result

For streamed rendering layout, use `stream_cfg.orientation` (`Vertical` or `Horizontal`) as display intent in your actor/UI layer.

Grid detection is exposed from `null-or-die-core` as `detect_grid(audio_mono, sample_rate_hz, &GridCfg, &BiasCfg) -> GridEstimate`, and decoding from `null-or-die-audio` as `decode_audio_mono(path)`.
