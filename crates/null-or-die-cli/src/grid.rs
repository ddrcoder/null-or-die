use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::cli::GridCmd;
use null_or_die_core::{GridCfg, GridEstimate, detect_grid};

const ITG_OFFSET_SEC: f64 = 0.009;

#[derive(Debug, Serialize)]
pub struct GridReport {
    pub audio_path: String,
    pub sample_rate_hz: u32,
    pub paradigm: String,
    pub bpm: f64,
    /// Value written to #BPMS (trimmed decimals).
    pub bpm_text: String,
    /// Value written to #OFFSET.
    pub offset_sec: f64,
    pub estimate: GridEstimate,
    pub ssc_path: Option<String>,
    pub elapsed_ms: f64,
}

pub fn run(args: &GridCmd) -> Result<GridReport, String> {
    let started = Instant::now();
    let itg = match args.paradigm.to_ascii_lowercase().as_str() {
        "null" => false,
        "itg" | "+9ms" | "p9ms" => true,
        other => return Err(format!("unknown paradigm {other:?} (use null or itg)")),
    };
    if let Some(ssc) = &args.ssc
        && ssc.exists()
        && !args.force
    {
        return Err(format!(
            "{} exists; pass --force to overwrite",
            ssc.display()
        ));
    }
    let decoded = null_or_die_audio::decode_audio_mono(&args.audio_path)?;
    let cfg = GridCfg {
        min_bpm: args.min_bpm,
        max_bpm: args.max_bpm,
        bpm_hint: args.bpm_hint,
        snap_ms: args.snap_ms,
    };
    let bias_cfg = null_or_die_rssp::default_bias_cfg();
    let estimate = detect_grid(&decoded.mono, decoded.sample_rate_hz, &cfg, &bias_cfg)?;

    let beats = (estimate.duration_sec / estimate.period_sec()).max(1.0);
    let bpm_text = format_bpm(estimate.bpm, beats);
    // #OFFSET is minus the time of beat 0, placed on the detected downbeat. ITG sync puts the grid 9 ms early.
    let offset_sec = -estimate.first_downbeat_sec + if itg { ITG_OFFSET_SEC } else { 0.0 };

    let ssc_path = if let Some(path) = &args.ssc {
        let text = render_ssc(args, &bpm_text, offset_sec, beats as usize)?;
        fs::write(path, text).map_err(|e| format!("write {} failed: {e}", path.display()))?;
        Some(path.display().to_string())
    } else {
        None
    };

    Ok(GridReport {
        audio_path: args.audio_path.display().to_string(),
        sample_rate_hz: decoded.sample_rate_hz,
        paradigm: if itg { "+9ms" } else { "null" }.to_string(),
        bpm: estimate.bpm,
        bpm_text,
        offset_sec,
        estimate,
        ssc_path,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    })
}

/// Fewest decimals (3..=6) whose rounding moves the last beat by < 0.5 ms.
fn format_bpm(bpm: f64, beats: f64) -> String {
    let period = 60.0 / bpm;
    for decimals in 3..=6 {
        let text = format!("{bpm:.decimals$}");
        let rounded: f64 = text.parse().unwrap_or(bpm);
        if beats * (60.0 / rounded - period).abs() < 0.5e-3 {
            return text;
        }
    }
    format!("{bpm:.6}")
}

fn render_ssc(
    args: &GridCmd,
    bpm_text: &str,
    offset_sec: f64,
    beats: usize,
) -> Result<String, String> {
    let music = args
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("bad audio file name {}", args.audio_path.display()))?;
    let title = args
        .title
        .clone()
        .unwrap_or_else(|| file_stem(&args.audio_path));
    let artist = args.artist.clone().unwrap_or_default();
    let mut out = String::new();
    let _ = writeln!(out, "#VERSION:0.83;");
    let _ = writeln!(out, "#TITLE:{};", escape(&title));
    let _ = writeln!(out, "#ARTIST:{};", escape(&artist));
    let _ = writeln!(out, "#MUSIC:{};", escape(music));
    let _ = writeln!(out, "#OFFSET:{offset_sec:.6};");
    let _ = writeln!(out, "#BPMS:0.000={bpm_text};");
    let _ = writeln!(out, "#STOPS:;");
    let _ = writeln!(out, "#SELECTABLE:YES;");
    if args.beat_chart {
        let _ = writeln!(out, "//---------------dance-single - grid----------------");
        let _ = writeln!(out, "#NOTEDATA:;");
        let _ = writeln!(out, "#STEPSTYPE:dance-single;");
        let _ = writeln!(out, "#DIFFICULTY:Beginner;");
        let _ = writeln!(out, "#METER:1;");
        let _ = writeln!(out, "#CREDIT:null-or-die grid;");
        let _ = writeln!(out, "#NOTES:");
        let measures = beats.div_ceil(4).max(1);
        for m in 0..measures {
            if m > 0 {
                let _ = writeln!(out, ",");
            }
            for row in ["1000", "0100", "0010", "0001"] {
                let _ = writeln!(out, "{row}");
            }
        }
        let _ = writeln!(out, ";");
    }
    Ok(out)
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string()
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace(';', "\\;")
        .replace(':', "\\:")
}

#[cfg(test)]
mod tests {
    use super::format_bpm;

    #[test]
    fn bpm_text_trims_when_precise_enough() {
        assert_eq!(format_bpm(128.0, 400.0), "128.000");
        assert_eq!(format_bpm(128.01312345, 640.0), "128.013");
        assert_eq!(format_bpm(128.0136, 640.0), "128.0136");
    }
}
