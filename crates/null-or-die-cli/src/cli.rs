use std::path::PathBuf;
use std::{env, ffi::OsString};

use clap::{ArgAction, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "null-or-die")]
#[command(about = "Rust reverse-engineering helper for nine-or-null parity", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn parse_with_compat() -> Result<Self, String> {
        let argv = rewrite_legacy_args(env::args_os().collect())?;
        Self::try_parse_from(argv).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Analyze(AnalyzeCmd),
    Parity(ParityCmd),
    Harness(HarnessCmd),
    Bench(BenchCmd),
    Plot(PlotCmd),
    Grid(GridCmd),
}

#[derive(Debug, Parser)]
pub struct AnalyzeCmd {
    pub root_path: PathBuf,

    #[arg(long)]
    pub plot: bool,

    #[arg(short, long)]
    pub report_path: Option<PathBuf>,

    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    #[arg(long)]
    pub to_paradigm: Option<String>,

    #[arg(long = "consider-null", action = ArgAction::SetFalse, default_value_t = true)]
    pub consider_null: bool,

    #[arg(long = "consider-p9ms", action = ArgAction::SetFalse, default_value_t = true)]
    pub consider_p9ms: bool,

    #[arg(short = 't', long, default_value_t = 4.0)]
    pub tolerance: f64,

    #[arg(short = 'c', long = "confidence", default_value_t = 0.80)]
    pub confidence_limit: f64,

    #[arg(short = 'f', long = "fingerprint", default_value_t = 50.0)]
    pub fingerprint_ms: f64,

    #[arg(short = 'w', long = "window", default_value_t = 10.0)]
    pub window_ms: f64,

    #[arg(short = 's', long = "step", default_value_t = 0.2)]
    pub step_ms: f64,

    #[arg(long = "magic-offset", default_value_t = 0.0)]
    pub magic_offset_ms: f64,

    #[arg(long = "kernel-target", default_value = "digest")]
    pub kernel_target: String,

    #[arg(long = "kernel-type", default_value = "rising")]
    pub kernel_type: String,

    #[arg(long = "full-spectrogram")]
    pub full_spectrogram: bool,
}

#[derive(Debug, Parser)]
pub struct ParityCmd {
    pub root_path: PathBuf,

    #[arg(short = 'b', long = "baseline", alias = "baseline-path")]
    pub baseline_path: PathBuf,

    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    #[arg(long = "fail-on-missing")]
    pub fail_on_missing: bool,

    #[arg(long = "fail-on-mismatch")]
    pub fail_on_mismatch: bool,

    #[arg(long = "bias-only")]
    pub bias_only: bool,
}

#[derive(Debug, Parser)]
pub struct HarnessCmd {
    pub root_path: PathBuf,

    #[arg(short = 'b', long = "baseline", alias = "baseline-path")]
    pub baseline_path: PathBuf,

    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    #[arg(long = "python", default_value = "python3")]
    pub python_bin: String,

    #[arg(long = "source-root")]
    pub source_root: Option<PathBuf>,

    #[arg(long = "scratch")]
    pub scratch_path: Option<PathBuf>,

    #[arg(long = "keep-scratch")]
    pub keep_scratch: bool,

    #[arg(long = "overwrite")]
    pub overwrite: bool,

    #[arg(long = "zstd-level", default_value_t = 19)]
    pub zstd_level: i32,

    #[arg(long = "consider-null", action = ArgAction::SetFalse, default_value_t = true)]
    pub consider_null: bool,

    #[arg(long = "consider-p9ms", action = ArgAction::SetFalse, default_value_t = true)]
    pub consider_p9ms: bool,

    #[arg(short = 't', long, default_value_t = 4.0)]
    pub tolerance: f64,

    #[arg(short = 'f', long = "fingerprint", default_value_t = 50.0)]
    pub fingerprint_ms: f64,

    #[arg(short = 'w', long = "window", default_value_t = 10.0)]
    pub window_ms: f64,

    #[arg(short = 's', long = "step", default_value_t = 0.2)]
    pub step_ms: f64,

    #[arg(long = "magic-offset", default_value_t = 0.0)]
    pub magic_offset_ms: f64,

    #[arg(long = "kernel-target", default_value = "digest")]
    pub kernel_target: String,

    #[arg(long = "kernel-type", default_value = "rising")]
    pub kernel_type: String,

    #[arg(long = "full-spectrogram")]
    pub full_spectrogram: bool,
}

/// Detect a constant-BPM grid from raw audio and optionally write a minimal .ssc.
#[derive(Debug, Parser)]
pub struct GridCmd {
    /// Audio file (ogg, mp3, m4a, flac, wav, ... anything ffmpeg decodes).
    pub audio_path: PathBuf,

    /// Write a minimal .ssc here (MUSIC is the audio file name, so put it next to the audio).
    #[arg(long)]
    pub ssc: Option<PathBuf>,

    /// Overwrite an existing .ssc.
    #[arg(long)]
    pub force: bool,

    /// Write the JSON report here instead of stdout.
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    #[arg(long, default_value_t = 100.0)]
    pub min_bpm: f64,

    /// Upper bound of the BPM search range (inclusive).
    #[arg(long, default_value_t = 200.0)]
    pub max_bpm: f64,

    /// Skip tempo search and refine from this BPM.
    #[arg(long = "bpm")]
    pub bpm_hint: Option<f64>,

    /// Snap to an integer BPM when that moves the last beat by at most this many ms.
    #[arg(long, default_value_t = 3.0)]
    pub snap_ms: f64,

    /// Sync paradigm for the written offset: "null" or "itg" (+9 ms).
    #[arg(long, default_value = "null")]
    pub paradigm: String,

    /// Include a Beginner dance-single chart with a tap on every beat.
    #[arg(long)]
    pub beat_chart: bool,

    #[arg(long)]
    pub title: Option<String>,

    #[arg(long)]
    pub artist: Option<String>,
}

#[derive(Debug, Parser)]
pub struct PlotCmd {
    pub input_json: PathBuf,
    pub output_png: PathBuf,

    #[arg(long, default_value_t = 1024)]
    pub width: u32,

    #[arg(long, default_value_t = 256)]
    pub height: u32,

    #[arg(long, default_value_t = 50.0)]
    pub span_ms: f64,
}

#[derive(Debug, Parser)]
pub struct BenchCmd {
    pub simfile_path: PathBuf,

    #[arg(short = 'n', long = "iterations", default_value_t = 20)]
    pub iterations: usize,

    #[arg(long = "warmup", default_value_t = 3)]
    pub warmup: usize,

    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    #[arg(short = 'f', long = "fingerprint", default_value_t = 50.0)]
    pub fingerprint_ms: f64,

    #[arg(short = 'w', long = "window", default_value_t = 10.0)]
    pub window_ms: f64,

    #[arg(short = 's', long = "step", default_value_t = 0.2)]
    pub step_ms: f64,

    #[arg(long = "magic-offset", default_value_t = 0.0)]
    pub magic_offset_ms: f64,

    #[arg(long = "kernel-target", default_value = "digest")]
    pub kernel_target: String,

    #[arg(long = "kernel-type", default_value = "rising")]
    pub kernel_type: String,

    #[arg(long = "full-spectrogram")]
    pub full_spectrogram: bool,
}

fn rewrite_legacy_args(argv: Vec<OsString>) -> Result<Vec<OsString>, String> {
    if argv.len() < 2 || has_subcommand(argv.get(1)) {
        return Ok(argv);
    }
    let Some((flag_idx, subcmd)) = legacy_cmd_flag(&argv) else {
        return Ok(argv);
    };
    rewrite_cmd_flag(argv, flag_idx, subcmd)
}

fn has_subcommand(arg: Option<&OsString>) -> bool {
    matches!(
        arg.map(|v| v.to_string_lossy().to_string()),
        Some(cmd) if matches!(cmd.as_str(), "analyze" | "parity" | "harness" | "bench" | "plot" | "grid")
    )
}

fn legacy_cmd_flag(argv: &[OsString]) -> Option<(usize, &'static str)> {
    argv.iter().enumerate().skip(1).find_map(|(i, arg)| {
        let flag = arg.to_string_lossy();
        if flag == "--analyze" || flag == "-a" {
            Some((i, "analyze"))
        } else if flag == "--parity" {
            Some((i, "parity"))
        } else if flag == "--harness" {
            Some((i, "harness"))
        } else if flag == "--bench" {
            Some((i, "bench"))
        } else {
            None
        }
    })
}

fn rewrite_cmd_flag(
    argv: Vec<OsString>,
    flag_idx: usize,
    subcmd: &'static str,
) -> Result<Vec<OsString>, String> {
    let path_idx = flag_idx + 1;
    if path_idx >= argv.len() {
        let flag = argv[flag_idx].to_string_lossy();
        return Err(format!("{flag} requires a path argument"));
    }
    let mut out = Vec::with_capacity(argv.len() + 1);
    out.push(argv[0].clone());
    out.push(OsString::from(subcmd));
    out.push(argv[path_idx].clone());
    for (i, arg) in argv.into_iter().enumerate().skip(1) {
        if i != flag_idx && i != path_idx {
            out.push(arg);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::rewrite_legacy_args;
    use std::ffi::OsString;

    fn as_vec(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    fn as_text(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|v| v.to_string_lossy().to_string())
            .collect()
    }

    #[test]
    fn rewrite_analyze_flag_to_subcommand() {
        let out = rewrite_legacy_args(as_vec(&["null-or-die", "--analyze", "file.sm", "--plot"]))
            .expect("legacy analyze rewrite should succeed");
        assert_eq!(
            as_text(&out),
            ["null-or-die", "analyze", "file.sm", "--plot"]
        );
    }

    #[test]
    fn keep_subcommand_argv_unchanged() {
        let out = rewrite_legacy_args(as_vec(&["null-or-die", "analyze", "file.sm", "--plot"]))
            .expect("subcommand argv should parse");
        assert_eq!(
            as_text(&out),
            ["null-or-die", "analyze", "file.sm", "--plot"]
        );
    }

    #[test]
    fn error_when_legacy_flag_lacks_path() {
        let err = rewrite_legacy_args(as_vec(&["null-or-die", "--analyze"]))
            .expect_err("missing path should error");
        assert!(err.contains("requires a path"));
    }
}
