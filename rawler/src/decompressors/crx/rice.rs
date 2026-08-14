// SPDX-License-Identifier: LGPL-2.1
// Copyright 2021 Daniel Vogelbacher <daniel@chaospixel.com>

// Original Crx decoder crx.cpp was written by Alexey Danilchenko for libraw.
// Rewritten in Rust by Daniel Vogelbacher, based on logic found in
// crx.cpp and documentation done by Laurent Clévy (https://github.com/lclevy/canon_cr3).

use super::BitPump;
use super::Result;

/// Adaptive Golomb-Rice decoder
///
/// `Copy` so a decode loop can lift the whole decoder into a local for the length of a line and
/// put it back once. Behind a `&mut` every read of the bit cache and the K parameter is a memory
/// access the compiler cannot promote; as a local they live in registers, which is the shape
/// LibRaw's `crx.cpp` gets by keeping `bitData`/`bitsLeft` in the decode function itself.
#[derive(Clone, Copy)]
pub(super) struct RiceDecoder<'mdat> {
  /// Bitstream from MDAT
  bitpump: BitPump<'mdat>,
  k_param: u32,
}

impl<'mdat> RiceDecoder<'mdat> {
  /// Create new decoder for given bit pump
  pub(super) fn new(bitpump: BitPump<'mdat>) -> Self {
    Self { bitpump, k_param: 0 }
  }

  /// Get current K parameter
  #[inline(always)]
  pub(super) fn k(&self) -> u32 {
    self.k_param
  }

  /// Set K parameter
  #[inline(always)]
  pub(super) fn set_k(&mut self, k: u32) {
    self.k_param = k;
  }

  /// Return the requested bits
  // All bits are consumed.
  // The maximum number of bits are 32
  #[inline(always)]
  pub(super) fn bitstream_get_bits(&mut self, bits: u32) -> Result<u32> {
    debug_assert!(bits <= 32);
    self.bitpump.read_bits(bits)
  }

  /// Adaptive Golomb-Rice decoding, by adapting k value
  /// Sometimes adapting is based on the next coefficent (n) instead
  /// of current (x) coefficent. So you can disable it with `adapt_k`
  /// and update k later.
  #[inline(always)]
  pub(super) fn adaptive_rice_decode(&mut self, adapt_k: bool, escape: u32, esc_bits: u32, k_max: u32) -> Result<u32> {
    adaptive_rice_decode(&mut self.bitpump, &mut self.k_param, adapt_k, escape, esc_bits, k_max)
  }

  /// The pump and the K parameter, for a caller that will hold them in locals.
  pub(super) fn split(&self) -> (BitPump<'mdat>, u32) {
    (self.bitpump, self.k_param)
  }

  /// Both back, after such a caller is done with them.
  pub(super) fn rejoin(&mut self, bitpump: BitPump<'mdat>, k: u32) {
    self.bitpump = bitpump;
    self.k_param = k;
  }

  /// Update current K parameter
  #[inline(always)]
  pub(super) fn update_k_param(&mut self, bit_code: u32, k_max: u32) {
    self.k_param = Self::predict_k_param_max(self.k_param, bit_code, k_max);
  }

  /// Predict K parameter with maximum constraint
  /// Golomb-Rice becomes more efficient when used with an adaptive
  /// K parameter. This is done by predicting the next K value for the
  /// next sample value.
  #[inline(always)]
  fn predict_k_param_max(prev_k: u32, value: u32, k_max: u32) -> u32 {
    // Branchless, and the shift taken once. These three conditions are decided by entropy-coded
    // data, so they are the least predictable branches in the decoder and they run once per
    // sample - `perf` put the two surviving ones at 8.9% between them. The subtraction cannot
    // underflow: at `prev_k == 0` its condition is `value < 0`, which no `u32` satisfies.
    let shifted = value >> prev_k;
    let new_k = prev_k + u32::from(shifted > 2) + u32::from(shifted > 5) - u32::from(value < ((1 << prev_k) >> 1));

    if k_max > 0 { std::cmp::min(new_k, k_max) } else { new_k }
  }
}

/// The same decode, over a pump and a K the caller is holding in locals.
///
/// The hot line loop keeps both in registers for the length of a line and puts them back once
/// (`RiceDecoder::split` / `rejoin`); behind `&mut self` every read of the bit cache is a memory
/// access the compiler cannot promote. The method above is this function, so there is one
/// implementation rather than two that have to agree.
#[inline(always)]
pub(super) fn adaptive_rice_decode(
  pump: &mut BitPump<'_>,
  k: &mut u32,
  adapt_k: bool,
  escape: u32,
  esc_bits: u32,
  k_max: u32,
) -> Result<u32> {
  let prefix = pump.read_zeros()?;
  let val = if prefix >= escape {
    pump.read_bits(esc_bits)?
  } else if *k > 0 {
    (prefix << *k) | pump.read_bits(*k)?
  } else {
    prefix
  };
  if adapt_k {
    *k = RiceDecoder::predict_k_param_max(*k, val, k_max);
  }
  Ok(val)
}

