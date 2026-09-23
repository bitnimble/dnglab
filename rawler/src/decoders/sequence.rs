//! LOCAL PATCH (bowerbird). Which multi-shot capture a frame belongs to, as the body recorded it.
//!
//! A file of ours rather than an edit to the decoders, so a rebase onto upstream replays it
//! without touching their code: each decoder reaches it through one line in
//! `Decoder::capture_sequence`.

use crate::bits::LEu32;
use crate::formats::tiff::{IFD, Value};

/// A multi-shot capture the camera ran as one: the frames are meant to be merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceKind {
  /// Sensor-shift multi shot: every frame the same exposure, a photosite apart.
  PixelShift,
  /// Auto exposure bracketing: every frame a different exposure of the same framing.
  ExposureBracket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureSequence {
  pub kind: SequenceKind,
  /// Shared by every frame of one capture, where the body writes such a key. Only pixel shift does.
  pub group: Option<u32>,
  /// 1-based position within the capture.
  pub index: u32,
  /// How many shots the capture fires, where the body says.
  pub count: Option<u32>,
}

/// Sony maker note `0x202f`, exiv2's `PixelShiftInfo`: a group id, then the shot's index and the
/// group's size, one byte each. Plaintext, unlike the `0x94xx` blocks.
const SONY_PIXEL_SHIFT: u16 = 0x202f;
/// Sony maker note `0x9400`, enciphered, three layouts told apart by the first (enciphered) byte.
const SONY_SEQUENCE: u16 = 0x9400;

/// `ReleaseMode2` values that are an exposure bracket: continuous (`2`) and single-frame (`23`),
/// the latter being what the ILCE-7 series writes. `3`, DRO or WB bracketing, is one exposure
/// rendered three ways and so never a bracket to merge.
const SONY_EXPOSURE_BRACKETS: [u8; 2] = [2, 23];

pub(crate) fn sony(makernote: &IFD, decipher: impl Fn(&[u8]) -> Vec<u8>) -> Option<CaptureSequence> {
  sony_pixel_shift(makernote).or_else(|| sony_exposure_bracket(makernote, decipher))
}

fn sony_pixel_shift(makernote: &IFD) -> Option<CaptureSequence> {
  let bytes = bytes_of(makernote, SONY_PIXEL_SHIFT)?;
  if bytes.len() < 6 {
    return None;
  }
  let (index, count) = (bytes[4] as u32, bytes[5] as u32);
  // Index 0 is Imaging Edge's merged ARQ carrying the tag, not a frame of the burst.
  if index == 0 || count < 2 || index > count {
    return None;
  }
  Some(CaptureSequence {
    kind: SequenceKind::PixelShift,
    group: Some(LEu32(bytes, 0)),
    index,
    count: Some(count),
  })
}

fn sony_exposure_bracket(makernote: &IFD, decipher: impl Fn(&[u8]) -> Vec<u8>) -> Option<CaptureSequence> {
  let enciphered = bytes_of(makernote, SONY_SEQUENCE)?;
  let layout = SonySequenceLayout::of(*enciphered.first()?)?;
  let plain = decipher(enciphered);
  if plain.len() <= layout.shots.max(layout.image_number + 3) {
    return None;
  }
  if !SONY_EXPOSURE_BRACKETS.contains(&plain[layout.release_mode]) {
    return None;
  }
  let shots = plain[layout.shots] as u32;
  Some(CaptureSequence {
    kind: SequenceKind::ExposureBracket,
    group: None,
    // Stored 0-based.
    index: LEu32(&plain, layout.image_number).saturating_add(1),
    count: (shots > 1).then_some(shots),
  })
}

/// Where Tag9400a/b/c keep the three fields, per exiftool's `Sony.pm`.
struct SonySequenceLayout {
  release_mode: usize,
  image_number: usize,
  /// `SequenceLength`, counted in shots fired rather than files written.
  shots: usize,
}

impl SonySequenceLayout {
  fn of(first: u8) -> Option<SonySequenceLayout> {
    match first {
      0x07 | 0x09 | 0x0a => Some(SonySequenceLayout {
        release_mode: 0x10,
        image_number: 0x08,
        shots: 0x22,
      }),
      0x0c => Some(SonySequenceLayout {
        release_mode: 0x10,
        image_number: 0x08,
        shots: 0x1e,
      }),
      0x23 | 0x24 | 0x26 | 0x28 | 0x31 | 0x32 | 0x33 | 0x41 | 0x43 => Some(SonySequenceLayout {
        release_mode: 0x09,
        image_number: 0x12,
        shots: 0x16,
      }),
      _ => None,
    }
  }
}

/// Canon `CanonFileInfo` indices: `BracketMode` (1 is AEB) and `BracketShotNumber`, 1-based.
const CANON_BRACKET_MODE: usize = 3;
const CANON_BRACKET_SHOT: usize = 5;
const CANON_AEB: u16 = 1;

/// Canon writes no count and no group; the index restarting is what ends a bracket.
pub(crate) fn canon(file_info: &IFD, tag: u16) -> Option<CaptureSequence> {
  let Value::Short(fields) = &file_info.get_entry(tag)?.value else {
    return None;
  };
  if *fields.get(CANON_BRACKET_MODE)? != CANON_AEB {
    return None;
  }
  let index = *fields.get(CANON_BRACKET_SHOT)? as u32;
  (index > 0).then_some(CaptureSequence {
    kind: SequenceKind::ExposureBracket,
    group: None,
    index,
    count: None,
  })
}

fn bytes_of(ifd: &IFD, tag: u16) -> Option<&[u8]> {
  match &ifd.get_entry(tag)?.value {
    Value::Undefined(bytes) | Value::Byte(bytes) => Some(bytes),
    _ => None,
  }
}
