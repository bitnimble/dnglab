//! The tight region decode and the strip decode against the frame decode, sample for sample.
//!
//! Both exist to allocate less, and a decode that allocates less and answers differently is not the
//! same decode. Set `RAWLER_TEST_RAW` to a raw file to run them; there is no sample in the tree.

use rawler::decoders::RawDecodeParams;
use rawler::imgop::{Dim2, Point, Rect};
use rawler::rawsource::RawSource;
use rawler::{RawImage, RawImageData, get_decoder};

fn samples(image: &RawImage) -> &[u16] {
  match &image.data {
    RawImageData::Integer(samples) => samples,
    RawImageData::Float(_) => panic!("float raw"),
  }
}

/// `want` lifted out of an image covering `have`, which must contain it.
fn window(image: &RawImage, have: Rect, want: Rect) -> Vec<u16> {
  assert!(
    have.p.x <= want.p.x && have.p.y <= want.p.y && have.p.x + have.d.w >= want.p.x + want.d.w && have.p.y + have.d.h >= want.p.y + want.d.h,
    "{:?} does not contain {:?}",
    have,
    want
  );
  let samples = samples(image);
  let mut out = Vec::with_capacity(want.d.w * want.d.h);
  for row in 0..want.d.h {
    let from = (want.p.y - have.p.y + row) * image.width + (want.p.x - have.p.x);
    out.extend_from_slice(&samples[from..from + want.d.w]);
  }
  out
}

fn source() -> Option<RawSource> {
  let Ok(path) = std::env::var("RAWLER_TEST_RAW") else {
    eprintln!("SKIP: set RAWLER_TEST_RAW to a raw file to run this");
    return None;
  };
  Some(RawSource::new(std::path::Path::new(&path)).expect("RAWLER_TEST_RAW unreadable"))
}

#[test]
fn tight_region_holds_the_same_samples_as_a_frame_sized_one() {
  let Some(source) = source() else { return };
  let decoder = get_decoder(&source).unwrap();
  let params = RawDecodeParams::default();
  let shape = decoder.raw_image(&source, &params, true).unwrap();
  let whole = Rect::new(Point::zero(), Dim2::new(shape.width, shape.height));

  // An interior crop, both corners, a crop straddling a tile seam, and one whose size is not a
  // multiple of anything - the alignment the tight path applies is exactly what these can expose.
  let wants = [
    Rect::new(Point::new(3060, 2254), Dim2::new(480, 480)),
    Rect::new(Point::zero(), Dim2::new(64, 64)),
    Rect::new(Point::new(shape.width - 300, shape.height - 300), Dim2::new(300, 300)),
    Rect::new(Point::new(510, 510), Dim2::new(4, 4)),
    Rect::new(Point::new(1021, 2043), Dim2::new(517, 129)),
  ];

  for want in wants {
    let frame = decoder.raw_image_region(&source, &params, want, false).unwrap();
    let (tight, got) = decoder.raw_image_region_tight(&source, &params, want, false).unwrap();

    assert_eq!(tight.width, got.d.w, "{:?}: image is not the rectangle it reports", want);
    assert_eq!(tight.height, got.d.h, "{:?}: image is not the rectangle it reports", want);
    assert_eq!(samples(&tight).len(), got.d.w * got.d.h, "{:?}", want);
    // Over everything it claims, not just what was asked for: both decodes cover the same tiles, so
    // a comparison limited to `want` would pass on a buffer that is wrong everywhere else in `got`.
    assert_eq!(window(&frame, whole, got), samples(&tight), "{:?} differs over {:?}", want, got);

    assert_eq!(tight.active_area, None, "{:?}: a rectangle in the frame's coordinates survived", want);
    assert_eq!(tight.crop_area, None, "{:?}: a rectangle in the frame's coordinates survived", want);
    assert!(tight.blackareas.is_empty(), "{:?}: a rectangle in the frame's coordinates survived", want);

    assert_eq!(frame.whitelevel, tight.whitelevel, "{:?}", want);
    assert_eq!(frame.blacklevel, tight.blacklevel, "{:?}", want);
    // The fourth coefficient is NaN on a three-colour sensor, and NaN is not equal to itself.
    assert!(
      frame.wb_coeffs.iter().zip(tight.wb_coeffs).all(|(a, b)| a == &b || (a.is_nan() && b.is_nan())),
      "{:?}: {:?} vs {:?}",
      want,
      frame.wb_coeffs,
      tight.wb_coeffs
    );
    assert_eq!(frame.cpp, tight.cpp, "{:?}", want);
  }
}

#[test]
fn strips_reassemble_into_the_frame() {
  let Some(source) = source() else { return };
  let decoder = get_decoder(&source).unwrap();
  let params = RawDecodeParams::default();
  let frame = decoder.raw_image(&source, &params, false).unwrap();
  let band = decoder.raw_image_band_height(&source, &params).unwrap().unwrap_or(frame.height);

  let mut assembled = vec![0u16; frame.width * frame.height];
  let mut covered = 0;
  for top in (0..frame.height).step_by(band) {
    let want = Rect::new(Point::new(0, top), Dim2::new(frame.width, band.min(frame.height - top)));
    let (strip, got) = decoder.raw_image_region_tight(&source, &params, want, false).unwrap();
    assert_eq!(got.p.x, 0, "a full-width band should not be split sideways");
    assert_eq!(got.d.w, frame.width, "a full-width band should not be split sideways");
    for row in 0..got.d.h.min(frame.height - got.p.y) {
      let to = (got.p.y + row) * frame.width;
      assembled[to..to + frame.width].copy_from_slice(&samples(&strip)[row * got.d.w..(row + 1) * got.d.w]);
      covered += 1;
    }
  }

  assert_eq!(covered, frame.height, "the bands did not cover the frame");
  assert!(assembled == samples(&frame), "reassembled strips differ from the frame decode");
}
