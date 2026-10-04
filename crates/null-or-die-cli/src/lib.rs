mod analyze;
mod bench;
mod cli;
mod fs_scan;
mod grid;
mod harness;
mod model;
mod parity;
mod plot;

use std::fs;
use std::path::Path;

use serde::Serialize;

pub const TOOL_NAME: &str = "null-or-die";

mod audio {
    pub use null_or_die_audio::*;
}

mod bias {
    pub use null_or_die_core::*;
    pub use null_or_die_rssp::{
        estimate_bias_reuse, estimate_bias_reuse_with_plot, estimate_bias_reuse_with_trace,
    };
}

mod compat {
    pub use null_or_die_core::{guess_paradigm, slot_abbreviation};
}

pub fn run() -> Result<(), String> {
    let cli = cli::Cli::parse_with_compat()?;
    match cli.command {
        cli::Command::Analyze(args) => {
            let report = analyze::run(&args)?;
            write_json(&report, args.output.as_deref())
        }
        cli::Command::Parity(args) => {
            let report = parity::run(&args)?;
            write_json(&report, args.output.as_deref())?;
            if args.fail_on_missing && report.missing_baseline > 0 {
                Err(format!(
                    "missing {} baseline fixture(s)",
                    report.missing_baseline
                ))
            } else if args.fail_on_mismatch && report.mismatched > 0 {
                Err(format!(
                    "found {} parity mismatch case(s)",
                    report.mismatched
                ))
            } else {
                Ok(())
            }
        }
        cli::Command::Harness(args) => {
            let report = harness::run(&args)?;
            write_json(&report, args.output.as_deref())
        }
        cli::Command::Bench(args) => {
            let report = bench::run(&args)?;
            write_json(&report, args.output.as_deref())
        }
        cli::Command::Grid(args) => {
            let report = grid::run(&args)?;
            write_json(&report, args.output.as_deref())
        }
        cli::Command::Plot(args) => {
            let report = plot::run(&args)?;
            write_json(&report, None)
        }
    }
}

fn write_json<T: Serialize>(value: &T, out_path: Option<&Path>) -> Result<(), String> {
    let json =
        serde_json::to_string_pretty(value).map_err(|e| format!("serialize json failed: {e}"))?;
    if let Some(path) = out_path {
        fs::write(path, json).map_err(|e| format!("write {} failed: {e}", path.display()))
    } else {
        println!("{json}");
        Ok(())
    }
}
