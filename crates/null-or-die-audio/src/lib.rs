use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use lewton::inside_ogg::OggStreamReader;

const PCM_INV_SCALE: f32 = 1.0 / 32768.0;
const OGG_BUF_CAP: usize = 256 * 1024;

#[derive(Debug, Clone)]
pub struct OggDecode {
    pub sample_rate_hz: u32,
    pub mono: Vec<f32>,
}

pub fn decode_ogg_mono_like_python(path: &Path) -> Result<OggDecode, String> {
    let (sample_rate_hz, source_channels) = probe_ogg_header(path)?;
    match decoder_pref().as_deref() {
        Some("lewton") | None => return decode_ogg_lewton(path),
        Some("ffmpeg") => {
            return decode_ogg_ffmpeg(path, sample_rate_hz, source_channels);
        }
        Some("auto") => {}
        _ => {}
    }
    match decode_ogg_ffmpeg(path, sample_rate_hz, source_channels) {
        Ok(decoded) => Ok(decoded),
        Err(ffmpeg_err) => match decode_ogg_lewton(path) {
            Ok(decoded) => Ok(decoded),
            Err(lewton_err) => Err(format!(
                "{ffmpeg_err}; fallback lewton decode failed: {lewton_err}"
            )),
        },
    }
}

/// Decode any audio file to mono. OGG goes through the parity-matched path;
/// everything else (mp3, m4a, flac, wav, ...) is decoded with ffmpeg.
pub fn decode_audio_mono(path: &Path) -> Result<OggDecode, String> {
    let is_ogg = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("ogg"));
    let ogg_err = if is_ogg {
        match decode_ogg_mono_like_python(path) {
            Ok(decoded) => return Ok(decoded),
            Err(e) => Some(e),
        }
    } else {
        None
    };
    let (sample_rate_hz, channels) = probe_ffmpeg_stream(path)?;
    decode_ogg_ffmpeg(path, sample_rate_hz, channels).map_err(|e| match ogg_err {
        Some(ogg) => format!("{ogg}; ffmpeg fallback failed: {e}"),
        None => e,
    })
}

fn probe_ffmpeg_stream(path: &Path) -> Result<(u32, usize), String> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args([
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe {} failed: {e}", path.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffprobe {} failed: {}",
            path.display(),
            stderr.trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut parts = text.trim().split(',');
    let parse_err = || {
        format!(
            "ffprobe {}: unexpected output {:?}",
            path.display(),
            text.trim()
        )
    };
    let sample_rate_hz = parts
        .next()
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(parse_err)?;
    let channels = parts
        .next()
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(parse_err)?;
    Ok((sample_rate_hz, channels))
}

fn decoder_pref() -> Option<String> {
    std::env::var("NOD_AUDIO_DECODER")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

fn probe_ogg_header(path: &Path) -> Result<(u32, usize), String> {
    let file = File::open(path).map_err(|e| format!("open {} failed: {e}", path.display()))?;
    let reader = OggStreamReader::new(BufReader::with_capacity(OGG_BUF_CAP, file))
        .map_err(|e| format!("ogg header parse {} failed: {e}", path.display()))?;
    Ok((
        reader.ident_hdr.audio_sample_rate,
        usize::from(reader.ident_hdr.audio_channels),
    ))
}

fn decode_ogg_ffmpeg(
    path: &Path,
    sample_rate_hz: u32,
    source_channels: usize,
) -> Result<OggDecode, String> {
    let output = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-map")
        .arg("a:0")
        .arg("-f")
        .arg("s16le")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-")
        .output()
        .map_err(|e| format!("ffmpeg decode {} failed: {e}", path.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg decode {} failed: {}",
            path.display(),
            stderr.trim()
        ));
    }
    Ok(OggDecode {
        sample_rate_hz,
        mono: mono_from_interleaved_pcm_i16(&output.stdout, source_channels),
    })
}

fn decode_ogg_lewton(path: &Path) -> Result<OggDecode, String> {
    let file = File::open(path).map_err(|e| format!("open {} failed: {e}", path.display()))?;
    let mut reader = OggStreamReader::new(BufReader::with_capacity(OGG_BUF_CAP, file))
        .map_err(|e| format!("ogg header parse {} failed: {e}", path.display()))?;
    let sample_rate_hz = reader.ident_hdr.audio_sample_rate;
    let source_channels = usize::from(reader.ident_hdr.audio_channels);
    let mut mono = Vec::new();
    while let Some(packet) = reader
        .read_dec_packet()
        .map_err(|e| format!("ogg decode {} failed: {e}", path.display()))?
    {
        append_python_mono_like(&packet, source_channels, &mut mono);
    }
    Ok(OggDecode {
        sample_rate_hz,
        mono,
    })
}

fn mono_from_interleaved_pcm_i16(bytes: &[u8], channels: usize) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 2 {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for frame in bytes.chunks_exact(4) {
            let left = i16::from_le_bytes([frame[0], frame[1]]);
            let right = i16::from_le_bytes([frame[2], frame[3]]);
            out.push(f32::from(left.max(right)) * PCM_INV_SCALE);
        }
        return out;
    }
    if channels == 1 {
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for sample in bytes.chunks_exact(2) {
            out.push(f32::from(i16::from_le_bytes([sample[0], sample[1]])) * PCM_INV_SCALE);
        }
        return out;
    }
    let frame_bytes = channels * 2;
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for frame in bytes.chunks_exact(frame_bytes) {
        for sample in frame.chunks_exact(2) {
            out.push(f32::from(i16::from_le_bytes([sample[0], sample[1]])) * PCM_INV_SCALE);
        }
    }
    out
}

fn append_python_mono_like(packet: &[Vec<i16>], channels: usize, out: &mut Vec<f32>) {
    if channels == 2 && packet.len() >= 2 {
        append_stereo_max(&packet[0], &packet[1], out);
    } else if packet.len() == 1 {
        append_passthrough(&packet[0], out);
    } else {
        append_interleaved_passthrough(packet, out);
    }
}

fn append_stereo_max(left: &[i16], right: &[i16], out: &mut Vec<f32>) {
    let len = left.len().min(right.len());
    out.reserve(len);
    let mut i = 0usize;
    while i < len {
        out.push(f32::from(left[i].max(right[i])) * PCM_INV_SCALE);
        i += 1;
    }
}

fn append_passthrough(packet: &[i16], out: &mut Vec<f32>) {
    out.reserve(packet.len());
    for s in packet {
        out.push(f32::from(*s) * PCM_INV_SCALE);
    }
}

fn append_interleaved_passthrough(packet: &[Vec<i16>], out: &mut Vec<f32>) {
    if packet.is_empty() {
        return;
    }
    let channels = packet.len();
    let frames = packet[0].len();
    out.reserve(channels * frames);
    let mut i = 0usize;
    while i < frames {
        for ch in packet {
            out.push(f32::from(ch[i]) * PCM_INV_SCALE);
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{append_python_mono_like, mono_from_interleaved_pcm_i16};

    #[test]
    fn stereo_collapse_uses_channel_max() {
        let mut out = Vec::new();
        append_python_mono_like(&[vec![100, -3200], vec![200, -6400]], 2, &mut out);
        assert_eq!(out.len(), 2);
        assert!((out[0] - (200.0 / 32768.0)).abs() < 1e-7);
        assert!((out[1] - (-3200.0 / 32768.0)).abs() < 1e-7);
    }

    #[test]
    fn mono_passthrough_is_normalized() {
        let mut out = Vec::new();
        append_python_mono_like(&[vec![32767, 0, -32768]], 1, &mut out);
        assert_eq!(out.len(), 3);
        assert!((out[0] - (32767.0 / 32768.0)).abs() < 1e-7);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], -1.0);
    }

    #[test]
    fn multichannel_passthrough_stays_interleaved() {
        let mut out = Vec::new();
        append_python_mono_like(&[vec![1, 2], vec![10, 20], vec![100, 200]], 1, &mut out);
        assert_eq!(out.len(), 6);
        assert!((out[0] - (1.0 / 32768.0)).abs() < 1e-7);
        assert!((out[1] - (10.0 / 32768.0)).abs() < 1e-7);
        assert!((out[2] - (100.0 / 32768.0)).abs() < 1e-7);
        assert!((out[3] - (2.0 / 32768.0)).abs() < 1e-7);
        assert!((out[4] - (20.0 / 32768.0)).abs() < 1e-7);
        assert!((out[5] - (200.0 / 32768.0)).abs() < 1e-7);
    }

    #[test]
    fn ffmpeg_pcm_stereo_uses_channel_max() {
        let bytes = [100u8, 0, 200, 0, 0x80, 0xff, 0x00, 0x80];
        let out = mono_from_interleaved_pcm_i16(&bytes, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - (200.0 / 32768.0)).abs() < 1e-7);
        assert!((out[1] - (-128.0 / 32768.0)).abs() < 1e-7);
    }
}
