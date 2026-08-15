//! `raw_image` alone, so the same measurement can be taken either side of a change to how it
//! decodes. `cargo run --release --example frame_cost -- <raw>`.

use rawler::decoders::RawDecodeParams;
use rawler::rawsource::RawSource;
use rawler::{RawImageData, get_decoder};

fn main() -> anyhow::Result<()> {
  let path = std::env::args().nth(1).expect("usage: frame_cost <raw>");
  let source = RawSource::new(std::path::Path::new(&path))?;
  let decoder = get_decoder(&source)?;
  let start = std::time::Instant::now();
  let image = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;
  let sum: u64 = match &image.data {
    RawImageData::Integer(samples) => samples.iter().map(|s| *s as u64).sum(),
    RawImageData::Float(_) => 0,
  };
  println!("frame {}x{} sum={} {}ms", image.width, image.height, sum, start.elapsed().as_millis());
  println!("peak rss {} MB", peak_rss_mb());
  Ok(())
}

/// Read rather than sampled: a decode is over in a few hundred milliseconds, and a sampler that
/// misses the one read where the frame is resident reports a saving that was never there.
fn peak_rss_mb() -> usize {
  let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
  status
    .lines()
    .find_map(|line| line.strip_prefix("VmHWM:"))
    .and_then(|v| v.split_whitespace().next())
    .and_then(|kb| kb.parse::<usize>().ok())
    .map_or(0, |kb| kb / 1024)
}
