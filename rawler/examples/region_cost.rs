//! What a decode costs in memory, one mode per process so `/usr/bin/time -v` can be believed.
//!
//! `cargo run --release --example region_cost -- <raw> <mode>`, modes below. Each prints a sum over
//! the samples it produced, so two modes that claim the same picture can be held against each other
//! without either keeping it.

use rawler::decoders::RawDecodeParams;
use rawler::imgop::{Dim2, Point, Rect};
use rawler::rawsource::RawSource;
use rawler::{RawImageData, get_decoder};

fn sum(data: &RawImageData) -> u64 {
  match data {
    RawImageData::Integer(samples) => samples.iter().map(|s| *s as u64).sum(),
    RawImageData::Float(samples) => samples.iter().map(|s| *s as u64).sum(),
  }
}

fn main() -> anyhow::Result<()> {
  let mut args = std::env::args().skip(1);
  let path = args.next().expect("usage: region_cost <raw> <mode> [x y w h]");
  let mode = args.next().unwrap_or_else(|| "info".into());
  let rect = |args: &mut dyn Iterator<Item = String>| {
    let mut n = || args.next().map(|a| a.parse::<usize>().unwrap());
    match (n(), n(), n(), n()) {
      (Some(x), Some(y), Some(w), Some(h)) => Rect::new(Point::new(x, y), Dim2::new(w, h)),
      _ => Rect::new(Point::new(3060, 2254), Dim2::new(480, 480)),
    }
  };

  let source = RawSource::new(std::path::Path::new(&path))?;
  let decoder = get_decoder(&source)?;
  let params = RawDecodeParams::default();
  let start = std::time::Instant::now();

  match mode.as_str() {
    "info" => {
      let shape = decoder.raw_image(&source, &params, true)?;
      println!(
        "{}x{} cpp={} band={:?} crop={:?} active={:?}",
        shape.width,
        shape.height,
        shape.cpp,
        decoder.raw_image_band_height(&source, &params)?,
        shape.crop_area,
        shape.active_area
      );
    }
    // The whole mosaic, held at once.
    "frame" => {
      let image = decoder.raw_image(&source, &params, false)?;
      println!("frame {}x{} sum={}", image.width, image.height, sum(&image.data));
    }
    // The same mosaic, one band at a time and never more than one.
    "strips" => {
      let shape = decoder.raw_image(&source, &params, true)?;
      let bands_at_once = args.next().and_then(|a| a.parse::<usize>().ok()).unwrap_or(1);
      let band = decoder.raw_image_band_height(&source, &params)?.unwrap_or(shape.height) * bands_at_once;
      let mut total = 0u64;
      let mut rows = 0;
      for top in (0..shape.height).step_by(band) {
        let want = Rect::new(Point::new(0, top), Dim2::new(shape.width, band.min(shape.height - top)));
        let (image, got) = decoder.raw_image_region_tight(&source, &params, want, false)?;
        total += sum(&image.data);
        rows += got.d.h;
      }
      println!("strips {}x{} of {} sum={}", shape.width, rows, band, total);
    }
    // A loupe tile, the way a frame-sized region decode makes you take it.
    "region" => {
      let want = rect(&mut args);
      let image = decoder.raw_image_region(&source, &params, want, false)?;
      let RawImageData::Integer(samples) = &image.data else { unreachable!() };
      let mut window = vec![0u16; want.d.w * want.d.h];
      for row in 0..want.d.h {
        let from = (want.p.y + row) * image.width + want.p.x;
        window[row * want.d.w..(row + 1) * want.d.w].copy_from_slice(&samples[from..from + want.d.w]);
      }
      println!("region {:?} sum={}", want, window.iter().map(|s| *s as u64).sum::<u64>());
    }
    // The same tile, from a buffer that is only ever the tiles it touches.
    "region-tight" => {
      let want = rect(&mut args);
      let (image, got) = decoder.raw_image_region_tight(&source, &params, want, false)?;
      let RawImageData::Integer(samples) = &image.data else { unreachable!() };
      let mut window = vec![0u16; want.d.w * want.d.h];
      for row in 0..want.d.h {
        let from = (want.p.y - got.p.y + row) * image.width + (want.p.x - got.p.x);
        window[row * want.d.w..(row + 1) * want.d.w].copy_from_slice(&samples[from..from + want.d.w]);
      }
      println!(
        "region-tight {:?} decoded {:?} sum={}",
        want,
        got,
        window.iter().map(|s| *s as u64).sum::<u64>()
      );
    }
    other => anyhow::bail!("unknown mode {other}"),
  }

  println!("{}ms", start.elapsed().as_millis());
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
