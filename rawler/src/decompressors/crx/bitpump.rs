// SPDX-License-Identifier: LGPL-2.1

//! Big-endian bit reader for the CRX entropy stream.
//!
//! This was `bitstream_io::BitReader<Cursor<&[u8]>, BigEndian>`, which is a fine general reader
//! and the wrong shape for this one loop. Its bit queue is a single `u8`, so a zero run longer
//! than eight bits costs one `leading_ones` per byte, and every byte it crosses is an
//! `io::Read::read` call through a `Cursor` returning an `io::Result` that the caller has to
//! branch on. CRX is almost entirely Golomb-Rice prefixes, so that per-byte cost *is* the decode:
//! measured against LibRaw's `crx.cpp`, which keeps a 32-bit cache in a register and takes the
//! whole run with one `__builtin_clzl`, the difference was 2.2x single-threaded.
//!
//! So: a 64-bit cache over the slice itself, refilled eight bytes at a time, with the unconsumed
//! bits held at the **top** of the cache. Top-aligned is what makes `leading_zeros` the answer
//! rather than an input to it.

use super::{CrxError, Result};

/// Bits the cache is guaranteed to hold after a refill that had bytes to give.
///
/// Not 64: a refill tops up in whole bytes, so it stops as soon as fewer than eight bits of room
/// are left. Every read below wants at most 32, which this clears comfortably.
const REFILL_TO: u32 = 56;

/// `Copy` so a decode loop can lift the whole reader into locals for the length of a line and
/// write it back once. Behind a `&mut` every read and write of `cache`/`bits`/`pos` is a memory
/// access the compiler cannot promote; as locals they live in registers, which is the shape
/// LibRaw's `crx.cpp` gets for free by keeping `bitData`/`bitsLeft` in the decode function.
#[derive(Clone, Copy)]
pub(super) struct BitPump<'a> {
  data: &'a [u8],
  /// Next byte to pull into `cache`.
  pos: usize,
  /// Unconsumed bits, most-significant first. Bits below `bits` are zero padding.
  cache: u64,
  bits: u32,
}

impl<'a> BitPump<'a> {
  pub(super) fn new(data: &'a [u8]) -> Self {
    Self { data, pos: 0, cache: 0, bits: 0 }
  }

  /// Tops the cache up to at least `REFILL_TO` bits, or to whatever the input has left.
  #[inline(always)]
  fn refill(&mut self) {
    if self.bits > REFILL_TO {
      return;
    }
    // Eight bytes in one load, which is the whole point: the byte-at-a-time version below costs
    // a bounds check and a shift per byte, and this runs between every pair of symbols.
    if self.pos + 8 <= self.data.len() {
      let chunk = u64::from_be_bytes(self.data[self.pos..self.pos + 8].try_into().expect("eight bytes"));
      // Whole bytes only, so `pos` stays a byte index. Everything below the bits actually
      // claimed must stay zero - `read_zeros` reads the padding as "cache exhausted", so a
      // stray 1 down there would end a run early.
      let take = (64 - self.bits) & !7;
      let mask = (u64::MAX << (64 - take)) >> self.bits;
      self.cache |= (chunk >> self.bits) & mask;
      self.pos += (take / 8) as usize;
      self.bits += take;
      return;
    }
    while self.bits <= REFILL_TO {
      let Some(byte) = self.data.get(self.pos) else { return };
      self.cache |= u64::from(*byte) << (REFILL_TO - self.bits);
      self.bits += 8;
      self.pos += 1;
    }
  }

  /// Drops the top `n` bits, which the caller has already read.
  ///
  /// `n` is always less than 64 here; a 64-bit shift is undefined and the callers below are
  /// bounded at 32 and at `bits` respectively.
  #[inline(always)]
  fn consume(&mut self, n: u32) {
    self.cache <<= n;
    self.bits -= n;
  }

  /// The number of 0-bits before the next 1-bit. Both the zeros and the 1 are consumed.
  ///
  /// The whole reason this file exists: `leading_zeros` takes up to 64 bits of run in one
  /// instruction, where a byte-wide queue takes one call per eight.
  #[inline(always)]
  pub(super) fn read_zeros(&mut self) -> Result<u32> {
    let mut found = 0;
    loop {
      self.refill();
      if self.bits == 0 {
        return Err(CrxError::Overflow("bitstream ended inside a Golomb-Rice prefix".into()));
      }
      // Padding below `bits` is zero, so a run that reaches it is not a run - it has only run
      // out of cache, and the rest of it is in the next refill.
      let zeros = self.cache.leading_zeros();
      if zeros < self.bits {
        self.consume(zeros + 1);
        return Ok(found + zeros);
      }
      found += self.bits;
      self.cache = 0;
      self.bits = 0;
    }
  }

  /// The next `n` bits, most-significant first. At most 32.
  #[inline(always)]
  pub(super) fn read_bits(&mut self, n: u32) -> Result<u32> {
    debug_assert!(n <= 32);
    if n == 0 {
      return Ok(0);
    }
    if self.bits < n {
      self.refill();
      if self.bits < n {
        return Err(CrxError::Overflow("bitstream ended inside a value".into()));
      }
    }
    let value = (self.cache >> (64 - n)) as u32;
    self.consume(n);
    Ok(value)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The two readers have to agree bit for bit, so this is the old one's behaviour written down:
  /// big-endian, MSB first, the stop bit consumed along with the zeros before it.
  #[test]
  fn reads_big_endian_bits_most_significant_first() {
    let mut pump = BitPump::new(&[0b1010_0110, 0b1111_0000]);
    assert_eq!(pump.read_bits(1).unwrap(), 1);
    assert_eq!(pump.read_bits(3).unwrap(), 0b010);
    assert_eq!(pump.read_bits(4).unwrap(), 0b0110);
    assert_eq!(pump.read_bits(8).unwrap(), 0b1111_0000);
  }

  #[test]
  fn a_zero_run_consumes_its_stop_bit() {
    let mut pump = BitPump::new(&[0b0001_0100]);
    assert_eq!(pump.read_zeros().unwrap(), 3);
    assert_eq!(pump.read_bits(1).unwrap(), 0);
    assert_eq!(pump.read_zeros().unwrap(), 0);
  }

  /// The case a byte-wide queue handles by looping and this one has to stitch across refills.
  #[test]
  fn a_zero_run_crosses_the_cache() {
    let mut data = vec![0u8; 16];
    data.push(0b0000_0001);
    let mut pump = BitPump::new(&data);
    assert_eq!(pump.read_zeros().unwrap(), 16 * 8 + 7);
  }

  #[test]
  fn zero_bits_is_zero_and_reads_nothing() {
    let mut pump = BitPump::new(&[0b1111_1111]);
    assert_eq!(pump.read_bits(0).unwrap(), 0);
    assert_eq!(pump.read_bits(8).unwrap(), 0xFF);
  }

  #[test]
  fn running_out_is_an_error_rather_than_a_panic() {
    let mut pump = BitPump::new(&[0b0000_0000]);
    assert!(pump.read_zeros().is_err());
    let mut pump = BitPump::new(&[0xFF]);
    assert!(pump.read_bits(9).is_err());
  }
}
