//! LOCAL PATCH (bowerbird). Which multi-shot capture a frame belongs to, as the body recorded it.
//!
//! A file of ours rather than an edit to the decoders, so a rebase onto upstream replays it
//! without touching their code: each decoder reaches it through one line in
//! `Decoder::capture_sequence`.

use crate::bits::{Endian, LEu32};
use crate::formats::tiff::{IFD, Value};

/// A multi-shot capture the camera ran as one: the frames are meant to be merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceKind {
  /// Sensor-shift multi shot: every frame the same exposure, a photosite apart.
  PixelShift,
  /// Auto exposure bracketing: every frame a different exposure of the same framing.
  ExposureBracket,
  /// Focus bracketing: every frame the same exposure, focused a step further than the last.
  FocusBracket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureSequence {
  pub kind: SequenceKind,
  /// Shared by every frame of one capture, where the body writes such a key. Only pixel shift does.
  pub group: Option<u32>,
  /// 1-based position within the capture, where the body says. Canon's focus bracketing does not,
  /// and its frames are in the order they were taken.
  pub index: Option<u32>,
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
    index: Some(index),
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
    index: Some(LEu32(&plain, layout.image_number).saturating_add(1)),
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

/// Canon `CameraSettings` index of `FocusBracketing`, 1 when the body was set to it.
const CANON_FOCUS_BRACKETING: usize = 50;
/// Canon `FocusBracketingInfo` (`0x4053`), 32-bit words: on or off at 1, `ImageCount` at 2.
const CANON_FOCUS_BRACKETING_INFO: u16 = 0x4053;
const CANON_FOCUS_IMAGE_COUNT: usize = 2;

/// Canon writes no count and no group for an exposure bracket; the index restarting is what ends
/// one. A focus bracket says it is one and how many shots it fires, and nothing of which shot this
/// is.
pub(crate) fn canon(makernote: &IFD, file_info_tag: u16, camera_settings_tag: u16) -> Option<CaptureSequence> {
  canon_exposure_bracket(makernote, file_info_tag).or_else(|| canon_focus_bracket(makernote, camera_settings_tag))
}

fn canon_exposure_bracket(makernote: &IFD, file_info_tag: u16) -> Option<CaptureSequence> {
  let Value::Short(fields) = &makernote.get_entry(file_info_tag)?.value else {
    return None;
  };
  if *fields.get(CANON_BRACKET_MODE)? != CANON_AEB {
    return None;
  }
  let index = *fields.get(CANON_BRACKET_SHOT)? as u32;
  (index > 0).then_some(CaptureSequence {
    kind: SequenceKind::ExposureBracket,
    group: None,
    index: Some(index),
    count: None,
  })
}

fn canon_focus_bracket(makernote: &IFD, camera_settings_tag: u16) -> Option<CaptureSequence> {
  let Value::Short(settings) = &makernote.get_entry(camera_settings_tag)?.value else {
    return None;
  };
  if *settings.get(CANON_FOCUS_BRACKETING)? != 1 {
    return None;
  }
  let count = match &makernote.get_entry(CANON_FOCUS_BRACKETING_INFO).map(|entry| &entry.value) {
    Some(Value::Long(words)) => words.get(CANON_FOCUS_IMAGE_COUNT).copied(),
    Some(Value::SLong(words)) => words.get(CANON_FOCUS_IMAGE_COUNT).and_then(|count| u32::try_from(*count).ok()),
    Some(Value::Undefined(bytes)) if bytes.len() >= (CANON_FOCUS_IMAGE_COUNT + 1) * 4 => Some(LEu32(bytes, CANON_FOCUS_IMAGE_COUNT * 4)),
    _ => None,
  };
  Some(CaptureSequence {
    kind: SequenceKind::FocusBracket,
    group: None,
    index: None,
    count: count.filter(|count| *count > 1),
  })
}

/// Nikon `ShotInfo` (`0x0091`) versions whose focus shift shooting is documented: Z 9 and Z 8.
const NIKON_SEQUENCED_SHOT_INFO: [&[u8; 4]; 2] = [b"0805", b"0806"];
/// Where, in the decrypted `ShotInfo`, the offset of `SeqInfoZ9` is, and where in that block
/// `FocusShiftShooting` is: 0 when off, the 1-based shot otherwise.
const NIKON_SEQUENCE_OFFSET: usize = 0x30;
const NIKON_FOCUS_SHIFT_SHOT: usize = 0x20;

/// Nikon's focus shift shooting, off a `ShotInfo` block the caller has already decrypted from
/// byte 4. It writes the shot and not the count.
pub(crate) fn nikon(shot_info: &[u8], endian: Endian) -> Option<CaptureSequence> {
  let version: &[u8; 4] = shot_info.get(..4)?.try_into().ok()?;
  if !NIKON_SEQUENCED_SHOT_INFO.contains(&version) || shot_info.len() < NIKON_SEQUENCE_OFFSET + 4 {
    return None;
  }
  let sequence = endian.read_u32(shot_info, NIKON_SEQUENCE_OFFSET) as usize;
  let shot = *shot_info.get(sequence.checked_add(NIKON_FOCUS_SHIFT_SHOT)?)?;
  (shot > 0).then_some(CaptureSequence {
    kind: SequenceKind::FocusBracket,
    group: None,
    index: Some(shot as u32),
    count: None,
  })
}

/// Olympus and OM System `CameraSettings` `DriveMode` (`0x0600`): a mode, the shot, and a word of
/// flags. Mode 5 is bracketing, and flag `0x40` says of focus.
const OLYMPUS_DRIVE_MODE: u16 = 0x0600;
const OLYMPUS_BRACKETING: u16 = 5;
const OLYMPUS_FOCUS: u16 = 0x40;

/// Off the `CameraSettings` IFD. It writes the shot and not the count.
pub(crate) fn olympus(camera_settings: &IFD) -> Option<CaptureSequence> {
  let Value::Short(drive) = &camera_settings.get_entry(OLYMPUS_DRIVE_MODE)?.value else {
    return None;
  };
  if *drive.first()? != OLYMPUS_BRACKETING || drive.get(2)? & OLYMPUS_FOCUS == 0 {
    return None;
  }
  let shot = *drive.get(1)? as u32;
  (shot > 0).then_some(CaptureSequence {
    kind: SequenceKind::FocusBracket,
    group: None,
    index: Some(shot),
    count: None,
  })
}

fn bytes_of(ifd: &IFD, tag: u16) -> Option<&[u8]> {
  match &ifd.get_entry(tag)?.value {
    Value::Undefined(bytes) | Value::Byte(bytes) => Some(bytes),
    _ => None,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A decrypted Z 8 `ShotInfo` whose `SeqInfoZ9` sits at `at`, with `shot` where it keeps the
  /// focus shift frame.
  fn shot_info(version: &[u8; 4], at: u32, shot: u8, endian: Endian) -> Vec<u8> {
    let mut block = vec![0u8; 0x200];
    block[..4].copy_from_slice(version);
    let offset = match endian {
      Endian::Big => at.to_be_bytes(),
      Endian::Little => at.to_le_bytes(),
    };
    block[NIKON_SEQUENCE_OFFSET..NIKON_SEQUENCE_OFFSET + 4].copy_from_slice(&offset);
    block[at as usize + NIKON_FOCUS_SHIFT_SHOT] = shot;
    block
  }

  #[test]
  fn a_nikon_focus_shift_frame_is_read_in_either_byte_order() {
    for endian in [Endian::Big, Endian::Little] {
      let read = nikon(&shot_info(b"0806", 0x100, 3, endian), endian).expect("a focus shift frame");
      assert_eq!((read.kind, read.index, read.count), (SequenceKind::FocusBracket, Some(3), None));
    }
  }

  #[test]
  fn a_nikon_frame_not_shot_in_focus_shift_is_no_sequence() {
    assert_eq!(nikon(&shot_info(b"0806", 0x100, 0, Endian::Big), Endian::Big), None);
    assert_eq!(nikon(&shot_info(b"0801", 0x100, 3, Endian::Big), Endian::Big), None, "a body whose layout is not the Z 9's");
  }

  #[test]
  fn a_sequence_offset_past_the_block_is_no_sequence() {
    let mut block = shot_info(b"0805", 0x100, 3, Endian::Big);
    block[NIKON_SEQUENCE_OFFSET..NIKON_SEQUENCE_OFFSET + 4].copy_from_slice(&0x1f0u32.to_be_bytes());
    assert_eq!(nikon(&block, Endian::Big), None);
  }
}
