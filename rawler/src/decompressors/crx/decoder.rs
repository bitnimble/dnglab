// SPDX-License-Identifier: LGPL-2.1
// Copyright 2021 Daniel Vogelbacher <daniel@chaospixel.com>

// Original Crx decoder crx.cpp was written by Alexey Danilchenko for libraw.
// Rewritten in Rust by Daniel Vogelbacher, based on logic found in
// crx.cpp and documentation done by Laurent Clévy (https://github.com/lclevy/canon_cr3).

use super::{
  BandParam, CodecParams, CrxError, Result,
  mdat::{Plane, Tile},
};
use crate::decompressors::crx::{idwt::WaveletTransform, mdat::parse_header, rice::RiceDecoder};
use super::BitPump;
use itertools::izip;
use log::debug;
use rayon::prelude::*;
use std::convert::TryInto;

// LOCAL PATCH (bowerbird). Upstream times this decode with `std::time::Instant`, which
// wasm32-unknown-unknown has no clock for: `Instant::now()` panics with "time not
// implemented on this platform", so every CR3 decode in a browser trapped. The timing is
// a debug log line, so on wasm it reports zero rather than bringing the decode down.
//
// The whole of this vendored crate exists for these few lines; upstream's fix is to gate
// the timing on `#[cfg(not(target_arch = "wasm32"))]`.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
  fn now() -> Self {
    Self
  }

  fn elapsed(&self) -> std::time::Duration {
    std::time::Duration::ZERO
  }
}

/// Maximum value for K during Adaptive Golomb-Rice for K prediction
pub(super) const PREDICT_K_MAX: u32 = 15;
pub(super) const PREDICT_K_ESCAPE: u32 = 41;
pub(super) const PREDICT_K_ESCBITS: u32 = 21;

struct PlaneLineIter<'a> {
  tile: &'a Tile,
  plane: &'a Plane,
  codec: CodecParams,
  params: Vec<BandParam<'a>>,
  iwt_transforms: Vec<WaveletTransform>,
  //plane_buf: Vec<i32>,
  next_row: usize,
}

impl<'a> PlaneLineIter<'a> {
  /// Create a new PlaneLine iterator for decoding
  fn new(codec: CodecParams, tile: &'a Tile, plane: &'a Plane, mdat: &'a [u8]) -> Result<Self> {
    // Some checks for correct input
    assert!(tile.plane_height > 0);
    assert!(tile.plane_width > 0);

    // Reference to data section in MDAT
    // All calculated offsets are relative to the data section.
    let data = codec.get_data(mdat);

    let plane_mdat_offset =
      tile.data_offset + tile.qp_data.as_ref().map(|qp| qp.mdat_qp_data_size + qp.mdat_extra_size as u32).unwrap_or(0) as usize + plane.data_offset;

    let mut params = Vec::with_capacity(plane.subbands.len());
    for (band_id, band) in plane.subbands.iter().enumerate() {
      let band_mdat_offset = plane_mdat_offset + band.data_offset;
      debug!("Band {} has MDAT offset: {}", band_id, band_mdat_offset);
      let band_buf = &data[band_mdat_offset..band_mdat_offset + band.data_size];
      // Line length is subband + one additional pixel at start and end
      let line_len = 1 + band.width + 1;
      let bitpump = BitPump::new(band_buf);

      let param = BandParam {
        subband_width: band.width,
        subband_height: band.height,
        rounded_bits_mask: if plane.support_partial && band_id == 0 { plane.rounded_bits_mask } else { 0 },
        rounded_bits: 0,
        cur_line: 0,
        line_buf: [vec![0; line_len], vec![0; line_len]],
        line_k: vec![0; line_len],
        line_pos: 0,
        line_len,
        s_param: 0,
        q_param: band.q_param,
        supports_partial: plane.support_partial && band_id == 0,
        rice: RiceDecoder::new(bitpump),
        predecoded: Vec::new(),
        predecoded_row: 0,
      };
      params.push(param);
    }

    let mut iwt_transforms = Vec::with_capacity(codec.levels);

    if codec.levels > 0 {
      // create Wavelet transforms
      for level in 0..codec.levels {
        let band = 3 * level + 1;
        let (height, width) = if level >= codec.levels - 1 {
          (tile.plane_height, tile.plane_width)
        } else {
          (plane.subbands[band + 3].height, plane.subbands[band + 4].width)
        };
        iwt_transforms.push(WaveletTransform::new(height, width));
      }
      codec.idwt_53_filter_init(tile, plane, &mut params, &mut iwt_transforms, codec.levels)?;
    }

    Ok(Self {
      params,
      tile,
      plane,
      codec,
      iwt_transforms,
      next_row: 0,
    })
  }

  /// Decode a single line from plane
  fn decode_plane_line(&mut self) -> Result<&[i32]> {
    if self.next_row < self.tile.plane_height {
      self.next_row += 1;
      if self.codec.levels > 0 {
        self
          .codec
          .idwt_53_filter_decode(self.tile, self.plane, &mut self.params, &mut self.iwt_transforms, self.codec.levels - 1)?;
        self
          .codec
          .idwt_53_filter_transform(self.tile, self.plane, &mut self.params, &mut self.iwt_transforms, self.codec.levels - 1)?;
        let line_data = self.iwt_transforms[self.codec.levels - 1].getline();
        debug_assert_eq!(line_data.len(), self.tile.plane_width);
        Ok(line_data)
      } else {
        debug_assert_eq!(self.plane.subbands.len(), 1);
        let param = &mut self.params[0];
        self.codec.decode_line(param)?;
        let line_data = param.decoded_buf();
        debug_assert_eq!(line_data.len(), param.subband_width as usize);
        debug_assert_eq!(line_data.len(), self.tile.plane_width);
        Ok(line_data)
      }
    } else {
      Err(CrxError::General("All rows processed, can't decode more".to_string()))
    }
  }
}

/// Decodes `rows` lines of a plane into one contiguous buffer, row-major.
///
/// Contiguous rather than a vector of rows, and taken straight from `decode_plane_line` rather
/// than through the `Iterator` impl this replaces: that handed back an owned `Vec` per line, some
/// eight thousand of them for a 25MP frame, each a fresh allocation and a copy of a buffer the
/// decoder had just filled. Worth about 5% of a single-threaded decode.
fn decode_full_plane(codec: &CodecParams, tile: &Tile, plane: &Plane, mdat: &[u8], rows: usize) -> Result<Vec<i32>> {
  let mut lines = PlaneLineIter::new(*codec, tile, plane, mdat)?;
  let rows = rows.min(tile.plane_height);

  // Only for a wavelet plane, only for a whole one, and only where there are threads to put it on.
  //
  // **`levels == 0` must not pre-decode**: that plane has one band and no wavelet, and
  // `decode_plane_line` takes its line straight from `decode_line` rather than through
  // `decode_line_with_iquantization`, which is the only reader of the cache. Filling it there
  // consumes the band's bitstream and hands back nothing, so the line decode that follows reads a
  // stream already at its end - a wrong picture where it does not fail outright.
  //
  // A row limit is asking to stop early, and a band decoded up front is decoded in full, so
  // pre-decoding would spend exactly what the limit exists to save. The thread count is the other
  // half: buffering every band costs a pass over the whole plane again, which pays for itself
  // only once there are more cores than the four planes can occupy. Measured on a 24MP CR3, this
  // path is 77ms against 61ms at twelve threads and 90ms against 114ms at four.
  if codec.levels > 0 && rows == tile.plane_height && rayon::current_num_threads() > usize::from(codec.plane_count) {
    let q_step = tile.q_step.as_ref();
    lines
      .params
      .par_iter_mut()
      .zip(plane.subbands.par_iter())
      .enumerate()
      .try_for_each(|(band_id, (param, band))| {
        // Which wavelet level a band belongs to, and so which quantiser table it takes:
        // `idwt_53_filter_decode` reads bands `3 * level + 1 ..= 3 * level + 3`, with the LL
        // band 0 sitting at level 0 beside them.
        let level = if band_id == 0 { 0 } else { (band_id - 1) / 3 };
        codec.predecode_band(band, param, q_step.map(|steps| &steps[level]))
      })?;
  }
  let mut out = Vec::with_capacity(rows * tile.plane_width);
  for _ in 0..rows {
    out.extend_from_slice(lines.decode_plane_line()?);
  }
  Ok(out)
}

impl CodecParams {
  /// Decode MDAT section into a single CFA image
  ///
  /// Decoding processes all planes in all tiles and assembles the
  /// decoded planes into proper tile output position and CFA pattern.
  pub fn decode(self, mdat: &[u8]) -> Result<Vec<u16>> {
    self.decode_rows(mdat, usize::MAX)
  }

  /// Decode the image down to `last_row`, leaving everything below it zero.
  ///
  /// **The only partial decode CRX allows, and it is one-dimensional.** A crop cannot be seeked
  /// to: prediction reads the line above and the 5/3 wavelet spans lines, so row N costs every
  /// row before it. Nor can tiles be skipped - Canon writes one tile for the whole frame, which
  /// is also why both this decoder and LibRaw's stop scaling at four threads (the four colour
  /// planes, not tiles). What is left is stopping early, which is worth having for a magnifier:
  /// a crop halfway down the frame costs half the decode, and one near the top costs almost none.
  ///
  /// `last_row` is in image rows. Rounded up to a whole plane row, since each plane carries every
  /// other line of the frame.
  pub fn decode_rows(mut self, mdat: &[u8], last_row: usize) -> Result<Vec<u16>> {
    let instant = Instant::now();
    debug!("Tile configuration: rows: {}, columns: {}", self.tile_rows, self.tile_cols);
    let plane_rows = last_row.saturating_div(2).saturating_add(1);
    // Build nested Tiles/Planes/Bands
    let mut tiles = parse_header(self.get_header(mdat))?;
    self.process_tiles(&mut tiles);
    for tile in tiles.iter_mut() {
      tile.generate_qstep_table(&self, self.get_data(mdat))?;
    }

    // cfa output is of final resolution
    let mut cfa: Vec<u16> = vec![0; self.resolution()];

    // Combine all tiles and planes into parallel iterators
    // and decode the full planes.
    let plane_bufs: Result<Vec<Vec<Vec<i32>>>> = tiles
      .par_iter()
      .map(|tile| {
        tile
          .planes
          .par_iter()
          .map(move |plane| decode_full_plane(&self, tile, plane, mdat, plane_rows))
          .collect()
      })
      .collect();

    // Now we have a list of tiles->planes->plane-lines
    // and can combine them to the final CFA
    // One tile, which is what Canon actually writes: every plane row then owns two CFA rows
    // outright, so the conversion can be split across threads by row with no overlap to
    // reconcile. It is worth the special case - the serial version of this stage is about half
    // the wall time of a threaded decode, because the four-way plane parallelism above finishes
    // and then everything queues behind one core doing the colour conversion. LibRaw splits the
    // same stage over `planeHeight` for the same reason.
    if let Ok(bufs) = plane_bufs.as_ref()
      && bufs.len() == 1
      && bufs[0].len() == 4
    {
      let planes = &bufs[0];
      let width = self.image_width;
      let plane_width = tiles[0].plane_width;
      let (p0, p1, p2, p3) = (&planes[0], &planes[1], &planes[2], &planes[3]);
      let decoded_rows = p0.len() / plane_width;
      cfa
        .par_chunks_mut(2 * width)
        .enumerate()
        .take(decoded_rows)
        .try_for_each(|(plane_row, rows)| {
          let at = plane_row * plane_width;
          let to = at + plane_width;
          convert_rows_into_cfa(&self, rows, width, &p0[at..to], &p1[at..to], &p2[at..to], &p3[at..to])
        })?;
      debug!("MDAT decoding and CFA build: {} s", instant.elapsed().as_secs_f32());
      return Ok(cfa);
    }

    match plane_bufs {
      Ok(bufs) => {
        for (tile_id, tile) in bufs.into_iter().enumerate() {
          let plane_count = tile.len();
          debug_assert_eq!(plane_count, 4);
          // Convert vector of planes to excact count of 4 planes - or fail
          let planes: [Vec<i32>; 4] = tile
            .try_into()
            .map_err(|_| CrxError::General(format!("Invalid plane count {} (expected 4) for tile {}", plane_count, tile_id)))?;
          // Each plane is one contiguous buffer, so its rows are chunks of the tile's plane width
          let plane_width = tiles[tile_id].plane_width;
          let (p0, p1, p2, p3) = (
            planes[0].chunks_exact(plane_width),
            planes[1].chunks_exact(plane_width),
            planes[2].chunks_exact(plane_width),
            planes[3].chunks_exact(plane_width),
          );
          for (plane_row, (l0, l1, l2, l3)) in izip!(p0, p1, p2, p3).enumerate() {
            let (c0, c1, c2, c3) = convert_plane_line(&self, l0, l1, l2, l3)?;
            integrate_cfa(&self, &tiles, &mut cfa, tile_id, 0, plane_row, &c0)?;
            integrate_cfa(&self, &tiles, &mut cfa, tile_id, 1, plane_row, &c1)?;
            integrate_cfa(&self, &tiles, &mut cfa, tile_id, 2, plane_row, &c2)?;
            integrate_cfa(&self, &tiles, &mut cfa, tile_id, 3, plane_row, &c3)?;
          }
        }
      }
      Err(e) => {
        return Err(e);
      }
    }
    debug!("MDAT decoding and CFA build: {} s", instant.elapsed().as_secs_f32());
    Ok(cfa)
  }

  /// Decode top line without a previous K buffer
  fn decode_top_line_no_ref_prev_line(&self, p: &mut BandParam) -> Result<()> {
    debug_assert_eq!(p.line_pos, 1);
    let mut remaining = p.subband_width as u32;
    // Init coef a and c (real image pixel starts at 1)
    p.line_buf[0][p.line_pos - 1] = 0; // is [0] because at start line_pos is 1
    p.line_buf[1][p.line_pos - 1] = 0; // is [0] because at start line_pos is 1
    while remaining > 1 {
      //println!("remaining: {}", remaining);
      // Loop over full width of line (backwards)
      if p.coeff_a() != 0 {
        //println!("coeff {} is != 0", p.coeff_a());
        let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
        p.line_buf[1][p.line_pos] = error_code_signed(bit_code);
      } else {
        //println!("coeff {} = 0", p.coeff_a());
        if p.rice.bitstream_get_bits(1)? == 1 {
          let n_syms = self.symbol_run_count(p, remaining)?;
          //println!("found {} syms", n_syms);
          remaining = remaining.saturating_sub(n_syms);
          // copy symbol n_syms times
          for _ in 0..n_syms {
            // For the first line, run-length coding uses only the symbol
            // value 0, so we can fill the line buffer and K buffer with 0.
            p.line_buf[1][p.line_pos] = 0;
            p.line_k[p.line_pos - 1] = 0;
            p.line_pos += 1;
          }

          if remaining == 0 {
            break;
          }
        } // if bitstream == 1

        let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
        p.line_buf[1][p.line_pos] = error_code_signed(bit_code + 1); // Caution: + 1
        //println!("code: {}", p.line_buf[1][p.line_pos]);
      }
      p.line_k[p.line_pos - 1] = p.rice.k();
      p.line_pos += 1;
      remaining = remaining.saturating_sub(1);
    }
    // Remaining pixel?
    if remaining == 1 {
      let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      p.line_buf[1][p.line_pos] = error_code_signed(bit_code);
      p.line_k[p.line_pos - 1] = p.rice.k();
      p.line_pos += 1;
    }
    debug_assert!(p.line_pos < p.line_buf[1].len());
    p.line_buf[1][p.line_pos] = 0;
    Ok(())
  }

  /// Decode nontop line with a previous K buffer
  fn decode_nontop_line_no_ref_prev_line(&self, p: &mut BandParam) -> Result<()> {
    //println!("Decode nontop {}", p.cur_line);
    debug_assert_eq!(p.line_pos, 1);
    let mut remaining = p.subband_width as u32;
    // Borrowed field by field so the two line buffers are plain slices for the loop below. Read
    // through `p`, every one of the seven accesses per sample is a fresh walk from the struct to
    // a `Vec`'s heap block, and the bounds check that goes with it cannot be hoisted out of the
    // loop because the compiler cannot know the `Vec` did not move.
    let BandParam { line_buf, line_k, rice, s_param, line_pos, .. } = p;
    let [above, current] = line_buf;
    // Cut to one common length so a single check can stand for all of them. Every access below is
    // `pos - 1`, `pos` or `pos + 1` into one of these three, and left as separate slices of
    // separately-known lengths that is seven bounds checks a sample - which is where `perf` puts
    // `core::slice::index`, second only to the decode itself.
    let span = above.len().min(current.len()).min(line_k.len());
    let above = &above.as_slice()[..span];
    let current = &mut current.as_mut_slice()[..span];
    let line_k = &mut line_k[..span];
    // The decoder as a local for the length of the line, put back below. An error leaves it
    // behind, which costs nothing: a band that failed to decode is not decoded further.
    let (mut pump, mut k) = rice.split();
    let mut pos = *line_pos;
    while remaining > 1 {
      // The one check the rest of the iteration rides on: `pos + 1` is the furthest any access
      // below reaches, and `pos` is at least 1 on entry and only ever grows.
      if pos + 1 >= span {
        return Err(CrxError::Overflow(format!("line position {pos} is past the {span} the line has")));
      }
      // Loop over full width of line (backwards)
      if (above[pos + 1] | above[pos] | current[pos - 1]) != 0 {
        let bit_code = crate::decompressors::crx::rice::adaptive_rice_decode(&mut pump, &mut k, true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, 0)?;
        current[pos] = error_code_signed(bit_code);
        if line_k[pos].saturating_sub(k) <= 1 {
          if k >= 15 {
            k = 15;
          }
        } else {
          k += 1;
        }
      } else {
        if pump.read_bits(1)? == 1 {
          debug_assert!(remaining != 1);
          let n_syms = self.symbol_run_count_at(&mut pump, s_param, remaining)?;

          remaining = remaining.saturating_sub(n_syms);
          // copy symbol n_syms times
          for _ in 0..n_syms {
            // For the first line, run-length coding uses only the symbol
            // value 0, so we can fill the line buffer and K buffer with 0.
            current[pos] = 0;
            line_k[pos - 1] = 0;
            pos += 1;
          }
        } // if bitstream == 1

        if remaining <= 1 {
          if remaining == 1 {
            let bit_code = crate::decompressors::crx::rice::adaptive_rice_decode(&mut pump, &mut k, true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
            current[pos] = error_code_signed(bit_code + 1);
            line_k[pos - 1] = k;
            pos += 1;
            remaining = remaining.saturating_sub(1); // skip remaining check at end of function
          }
          break;
        } else {
          let bit_code = crate::decompressors::crx::rice::adaptive_rice_decode(&mut pump, &mut k, true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, 0)?;
          current[pos] = error_code_signed(bit_code + 1); // Caution: + 1
          if line_k[pos].saturating_sub(k) <= 1 {
            if k >= 15 {
              k = 15;
            }
          } else {
            k += 1;
          }
        }
      }
      line_k[pos - 1] = k;
      pos += 1;
      remaining = remaining.saturating_sub(1);
    }
    // Remaining pixel?
    if remaining == 1 {
      let bit_code = crate::decompressors::crx::rice::adaptive_rice_decode(&mut pump, &mut k, true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      current[pos] = error_code_signed(bit_code);
      line_k[pos - 1] = k;
      pos += 1;
    }
    debug_assert!(pos < current.len());
    *line_pos = pos;
    rice.rejoin(pump, k);
    Ok(())
  }

  /// Decode top line
  /// For the first line (top) in a plane, no MED is used because
  /// there is no previous line for coeffs b, c and d.
  /// So this decoding is a simplified version from decode_nontop_line().
  fn decode_top_line(&self, p: &mut BandParam) -> Result<()> {
    debug_assert_eq!(p.line_pos, 1);
    let mut remaining = p.subband_width as u32;
    // Init coeff a (real image pixel starts at 1)
    p.line_buf[1][p.line_pos - 1] = 0; // is is [0] because at start line_pos is 1
    while remaining > 1 {
      // Loop over full width of line (backwards)
      if p.coeff_a() != 0 {
        p.line_buf[1][p.line_pos] = p.coeff_a();
      } else {
        if p.rice.bitstream_get_bits(1)? == 1 {
          let n_syms = self.symbol_run_count(p, remaining)?;
          remaining = remaining.saturating_sub(n_syms);
          // copy symbol n_syms times
          for _ in 0..n_syms {
            p.line_buf[1][p.line_pos] = p.coeff_a();
            p.line_pos += 1;
          }
          if remaining == 0 {
            break;
          }
        } // if bitstream == 1
        p.line_buf[1][p.line_pos] = 0;
      }
      let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      p.line_buf[1][p.line_pos] += error_code_signed(bit_code);
      p.line_pos += 1;
      remaining = remaining.saturating_sub(1);
    }
    // Remaining pixel?
    if remaining == 1 {
      let x = p.coeff_a(); // no MED, just use coeff a
      let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      p.line_buf[1][p.line_pos] = x + error_code_signed(bit_code);
      p.line_pos += 1;
    }
    debug_assert!(p.line_pos < p.line_buf[1].len());
    p.line_buf[1][p.line_pos] = p.coeff_a() + 1;
    Ok(())
  }

  /// Decode a line which is not a top line
  /// This used run length coding, Median Edge Detection (MED) and
  /// adaptive Golomb-Rice entropy encoding.
  /// Golomb-Rice becomes more efficient when using an adaptive K value
  /// instead of a fixed one.
  /// The K parameter is used as q = n >> k where n is the sample to encode.
  fn decode_nontop_line(&self, p: &mut BandParam) -> Result<()> {
    debug_assert_eq!(p.line_pos, 1);
    let mut remaining = p.subband_width as u32;
    // Init coeff a: a = b
    p.line_buf[1][p.line_pos - 1] = p.coeff_b();
    // Loop over full width of line (backwards)
    while remaining > 1 {
      let mut x = 0;
      //  c b d
      //  a x n
      // Median Edge Detection to predict pixel x. Described in patent US2016/0323602 and T.87
      if p.coeff_a() == p.coeff_b() && p.coeff_a() == p.coeff_d() {
        // different than step [0104], where Condition: "a=c and c=b and b=d", c not used
        if p.rice.bitstream_get_bits(1)? == 1 {
          let n_syms = self.symbol_run_count(p, remaining)?;
          remaining = remaining.saturating_sub(n_syms);
          // copy symbol n_syms times
          for _ in 0..n_syms {
            p.line_buf[1][p.line_pos] = p.coeff_a();
            p.line_pos += 1;
          }
        } // if bitstream == 1
        if remaining > 0 {
          x = p.coeff_b(); // use new coeff b because we moved line_pos!
        }
      } else {
        // no run length coding, use MED instead
        x = med(p.coeff_a(), p.coeff_b(), p.coeff_c());
      }
      if remaining > 0 {
        let mut bit_code = p.rice.adaptive_rice_decode(false, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
        // add converted (+/-) error code to predicted value
        p.line_buf[1][p.line_pos] = x + error_code_signed(bit_code);
        // for not end of the line - use one symbol ahead to estimate next K
        if remaining > 1 {
          let delta: i32 = (p.coeff_d() - p.coeff_b()) << 1;
          bit_code = (bit_code + delta.unsigned_abs()) >> 1;
        }
        p.rice.update_k_param(bit_code, PREDICT_K_MAX);
        p.line_pos += 1;
      }
      remaining = remaining.saturating_sub(1);
    } // end while length > 1
    // Remaining pixel?
    if remaining == 1 {
      let x = med(p.coeff_a(), p.coeff_b(), p.coeff_c());
      let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      // add converted (+/-) error code to predicted value
      p.line_buf[1][p.line_pos] = x + error_code_signed(bit_code);
      p.line_pos += 1;
    }
    debug_assert!(p.line_pos < p.line_buf[1].len());
    p.line_buf[1][p.line_pos] = p.coeff_a() + 1;
    Ok(())
  }

  /// Decode a symbol x in rounded mode.
  /// Used only when levels==0 (lossless mode)
  fn decode_symbol_rounded(&self, p: &mut BandParam, use_med: bool, not_eol: bool) -> Result<()> {
    let sym = if use_med { med(p.coeff_a(), p.coeff_b(), p.coeff_c()) } else { p.coeff_b() };
    let bit_code = p.rice.adaptive_rice_decode(false, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
    let mut code = error_code_signed(bit_code);
    let x = p.rounded_bits_mask * 2 * code + (code >> 31);
    p.line_buf[1][p.line_pos] = x + sym;

    if not_eol {
      if p.coeff_d() > p.coeff_b() {
        code = (p.coeff_d() - p.coeff_b() + p.rounded_bits_mask - 1) >> p.rounded_bits;
      } else {
        code = -((p.coeff_b() - p.coeff_d() + p.rounded_bits_mask) >> p.rounded_bits);
      }
      p.rice.update_k_param((bit_code + 2 * code.unsigned_abs()) >> 1, PREDICT_K_MAX);
    } else {
      p.rice.update_k_param(bit_code, PREDICT_K_MAX);
    }

    p.line_pos += 1;
    Ok(())
  }

  /// Decode a rounded line which is not a top line
  fn decode_top_line_rounded(&self, p: &mut BandParam) -> Result<()> {
    debug_assert_eq!(p.line_pos, 1);
    let mut remaining = p.subband_width as u32;
    // Init coeff a (real image pixel starts at 1)
    p.line_buf[1][p.line_pos - 1] = 0; // is is [0] because at start line_pos is 1
    while remaining > 1 {
      // Loop over full width of line (backwards)
      if p.coeff_a().abs() > p.rounded_bits_mask {
        p.line_buf[1][p.line_pos] = p.coeff_a();
      } else {
        if p.rice.bitstream_get_bits(1)? == 1 {
          let n_syms = self.symbol_run_count(p, remaining)?;
          remaining = remaining.saturating_sub(n_syms);
          // copy symbol n_syms times
          for _ in 0..n_syms {
            p.line_buf[1][p.line_pos] = p.coeff_a();
            p.line_pos += 1;
          }
          if remaining == 0 {
            break;
          }
        } // if bitstream == 1
        p.line_buf[1][p.line_pos] = 0;
      }
      let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      let code = error_code_signed(bit_code);
      p.line_buf[1][p.line_pos] += p.rounded_bits_mask * 2 * code + (code >> 31);
      p.line_pos += 1;
      remaining = remaining.saturating_sub(1);
    }
    // Remaining pixel?
    if remaining == 1 {
      let bit_code = p.rice.adaptive_rice_decode(true, PREDICT_K_ESCAPE, PREDICT_K_ESCBITS, PREDICT_K_MAX)?;
      let code = error_code_signed(bit_code);
      p.line_buf[1][p.line_pos] += p.rounded_bits_mask * 2 * code + (code >> 31);
      p.line_pos += 1;
    }
    debug_assert!(p.line_pos < p.line_buf[1].len());
    p.line_buf[1][p.line_pos] = p.coeff_a() + 1;
    Ok(())
  }

  /// Decode a line which is not a top line
  /// This used run length coding, Median Edge Detection (MED) and
  /// adaptive Golomb-Rice entropy encoding.
  /// Golomb-Rice becomes more efficient when using an adaptive K value
  /// instead of a fixed one.
  /// The K parameter is used as q = n >> k where n is the sample to encode.
  #[allow(clippy::comparison_chain)]
  fn decode_nontop_line_rounded(&self, p: &mut BandParam) -> Result<()> {
    debug_assert_eq!(p.line_pos, 1);
    let mut remaining = p.subband_width as u32;
    let mut value_reached = false;
    p.line_buf[0][p.line_pos - 1] = p.coeff_b();
    p.line_buf[1][p.line_pos - 1] = p.coeff_b();
    // Loop over full width of line (backwards)
    while remaining > 1 {
      if (p.coeff_d() - p.coeff_b()).abs() > p.rounded_bits_mask {
        self.decode_symbol_rounded(p, true, true)?;
        value_reached = true;
      } else if value_reached || (p.coeff_c() - p.coeff_a()).abs() > p.rounded_bits_mask {
        self.decode_symbol_rounded(p, true, true)?;
        value_reached = false;
      } else {
        if p.rice.bitstream_get_bits(1)? == 1 {
          let n_syms = self.symbol_run_count(p, remaining)?;
          remaining = remaining.saturating_sub(n_syms);
          // copy symbol n_syms times
          for _ in 0..n_syms {
            p.line_buf[1][p.line_pos] = p.coeff_a();
            p.line_pos += 1;
          }
        } // if bitstream == 1
        if remaining > 1 {
          self.decode_symbol_rounded(p, false, true)?;
          value_reached = (p.coeff_b() - p.coeff_c()).abs() > p.rounded_bits_mask;
        } else if remaining == 1 {
          self.decode_symbol_rounded(p, false, false)?;
        }
      }
      remaining = remaining.saturating_sub(1);
    } // end while length > 1
    // Remaining pixel?
    if remaining == 1 {
      self.decode_symbol_rounded(p, true, false)?;
    }
    debug_assert!(p.line_pos < p.line_buf[1].len());
    p.line_buf[1][p.line_pos] = p.coeff_a() + 1;
    Ok(())
  }

  /// Decode a single line from input band
  /// For decoding, two line buffers are required (except for the first line).
  /// After each decoding line, the two buffers are swapped, so the previous one
  /// is always in line_buf[0] (containing coefficents c, b, d) and the current
  /// line is in line_buf[1] (containing coefficents a, x, n).
  ///
  /// The line buffers has an extra sample on both ends. So the buffer layout is:
  ///
  /// |E|Samples........................|E|
  /// |c|bd                           cb|d|
  /// |a|xn                           ax|n|
  ///  ^ ^                               ^
  ///  | |                               |-- Extra sample to provide fake d coefficent
  ///  | |---- First sample value
  ///  |------ Extra sample to provide a fake a/c coefficent
  ///
  /// After line is decoded, the E samples are ignored when
  /// copied into the final plane buffer.
  ///
  /// For non-LL bands, decoding process differs a little bit
  /// because some value rounding is added.
  pub(super) fn decode_line(&self, param: &mut BandParam) -> Result<()> {
    debug_assert!(param.cur_line < param.subband_height);
    // We start at first real pixel value
    param.line_pos = 1;
    if param.cur_line == 0 {
      param.s_param = 0;
      param.rice.set_k(0); // TODO: required?
      if param.supports_partial {
        if param.rounded_bits_mask <= 0 {
          self.decode_top_line(param)?;
        } else {
          param.rounded_bits = 1;
          if (param.rounded_bits_mask & !1) != 0 {
            while param.rounded_bits_mask >> param.rounded_bits != 0 {
              param.rounded_bits += 1;
            }
          }
          self.decode_top_line_rounded(param)?;
        }
      } else {
        self.decode_top_line_no_ref_prev_line(param)?;
      }
    } else if !param.supports_partial {
      // Swap line buffers so previous decoded (1) is now above (0)
      param.line_buf.swap(0, 1);
      self.decode_nontop_line_no_ref_prev_line(param)?;
    } else if param.rounded_bits_mask <= 0 {
      // Swap line buffers so previous decoded (1) is now above (0)
      param.line_buf.swap(0, 1);
      self.decode_nontop_line(param)?;
    } else {
      // Swap line buffers so previous decoded (1) is now above (0)
      param.line_buf.swap(0, 1);
      self.decode_nontop_line_rounded(param)?;
    }
    param.cur_line += 1;
    Ok(())
  }
}

/// Constrain a given value into min/max
#[inline(always)]
pub(super) fn constrain(value: i32, min: i32, max: i32) -> i32 {
  std::cmp::min(std::cmp::max(value, min), max)
  /*
  let res = if value < min {
    min
  } else if value > max {
    max
  } else {
    value
  };
  debug_assert!(res <= u16::MAX as i32);
  res
   */
}

/// The error code contains a sign bit at bit 0.
/// Example: 10010 1 -> negative value, 10010 0 -> positive value
/// This routine converts an unsigned bit_code to the correct
/// signed integer value.
/// For this, the sign bit is inverted and XOR with
/// the shifted integer value.
#[inline(always)]
pub(super) fn error_code_signed(bit_code: u32) -> i32 {
  -((bit_code & 1) as i32) ^ (bit_code >> 1) as i32
}

/// Median Edge Detection
/// [0053] Obtains a predictive value p of the coefficient by using
/// MED prediction, thereby performing predictive coding.
#[inline(always)]
pub(super) fn med(a: i32, b: i32, c: i32) -> i32 {
  if c >= std::cmp::max(a, b) {
    std::cmp::min(a, b)
  } else if c <= std::cmp::min(a, b) {
    std::cmp::max(a, b)
  } else {
    a + b - c // no edge detected
  }
}

/// Convert a decoded line to plane output
/// Results from decode_line() are signed 32 bit integers.
/// By using a median and max value, these are converted
/// to unsigned 16 bit integers.
#[allow(clippy::type_complexity)]
fn convert_plane_line(codec: &CodecParams, l0: &[i32], l1: &[i32], l2: &[i32], l3: &[i32]) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>)> {
  let mut p0 = vec![0; l0.len()];
  let mut p1 = vec![0; l1.len()];
  let mut p2 = vec![0; l2.len()];
  let mut p3 = vec![0; l3.len()];

  match codec.enc_type {
    0 => {
      let median: i32 = 1 << (codec.median_bits - 1);
      let max_val: i32 = (1 << codec.median_bits) - 1;

      izip!(l0, l1, l2, l3).enumerate().for_each(|(i, (v0, v1, v2, v3))| {
        p0[i] = constrain(median + v0, 0, max_val) as u16;
        p1[i] = constrain(median + v1, 0, max_val) as u16;
        p2[i] = constrain(median + v2, 0, max_val) as u16;
        p3[i] = constrain(median + v3, 0, max_val) as u16;
      });
    }
    3 => {
      let median: i32 = 1 << (codec.median_bits - 1) << 10;
      let max_val: i32 = (1 << codec.median_bits) - 1;

      izip!(l0, l1, l2, l3).enumerate().for_each(|(i, (v0, v1, v2, v3))| {
        let mut gr: i32 = median + (v0 << 10) - 168 * v1 - 585 * v3;
        if gr < 0 {
          gr = -(((gr.abs() + 512) >> 9) & !1);
        } else {
          gr = ((gr.abs() + 512) >> 9) & !1;
        }
        p0[i] = constrain((median + (v0 << 10) + 1510 * v3 + 512) >> 10, 0, max_val) as u16;
        p1[i] = constrain((v2 + gr + 1) >> 1, 0, max_val) as u16;
        p2[i] = constrain((gr - v2 + 1) >> 1, 0, max_val) as u16;
        p3[i] = constrain((median + (v0 << 10) + 1927 * v1 + 512) >> 10, 0, max_val) as u16;
      });
    }
    enc_type => {
      return Err(CrxError::General(format!("Unsupported encoding type {}", enc_type)));
    }
  }

  Ok((p0, p1, p2, p3))
}

/// One plane row of all four planes, converted and written straight into the two CFA rows it
/// occupies.
///
/// **This is the whole assembly stage for a single-tile image, and it exists to be run per row
/// on its own thread.** `convert_plane_line` + `integrate_cfa` do the same work with four `Vec`
/// allocations per row and four stride-2 passes over the same pair of output rows; this takes
/// one pass, allocates nothing, and writes each row front to back. The four samples of a Bayer
/// quad are adjacent in the output, so computing them together is also what lets them be stored
/// together.
#[inline]
fn convert_rows_into_cfa(codec: &CodecParams, rows: &mut [u16], width: usize, l0: &[i32], l1: &[i32], l2: &[i32], l3: &[i32]) -> Result<()> {
  let (top, bottom) = rows.split_at_mut(width);
  let median_bits = codec.median_bits;
  let max_val: i32 = (1 << median_bits) - 1;

  match codec.enc_type {
    0 => {
      let median: i32 = 1 << (median_bits - 1);
      for (i, (v0, v1, v2, v3)) in izip!(l0, l1, l2, l3).enumerate() {
        top[2 * i] = constrain(median + v0, 0, max_val) as u16;
        top[2 * i + 1] = constrain(median + v1, 0, max_val) as u16;
        bottom[2 * i] = constrain(median + v2, 0, max_val) as u16;
        bottom[2 * i + 1] = constrain(median + v3, 0, max_val) as u16;
      }
    }
    3 => {
      let median: i32 = 1 << (median_bits - 1) << 10;
      for (i, (v0, v1, v2, v3)) in izip!(l0, l1, l2, l3).enumerate() {
        let mut gr: i32 = median + (v0 << 10) - 168 * v1 - 585 * v3;
        if gr < 0 {
          gr = -(((gr.abs() + 512) >> 9) & !1);
        } else {
          gr = ((gr.abs() + 512) >> 9) & !1;
        }
        top[2 * i] = constrain((median + (v0 << 10) + 1510 * v3 + 512) >> 10, 0, max_val) as u16;
        top[2 * i + 1] = constrain((v2 + gr + 1) >> 1, 0, max_val) as u16;
        bottom[2 * i] = constrain((gr - v2 + 1) >> 1, 0, max_val) as u16;
        bottom[2 * i + 1] = constrain((median + (v0 << 10) + 1927 * v1 + 512) >> 10, 0, max_val) as u16;
      }
    }
    enc_type => {
      return Err(CrxError::General(format!("Unsupported encoding type {}", enc_type)));
    }
  }
  Ok(())
}

/// Integrate a plane buffer into CFA output image
///
/// A plane is a single monochrome image for one of the four CFA colors.
/// `plane_id` is 0, 1, 2 or 3 for R, G1, G2, B
fn integrate_cfa(codec: &CodecParams, tiles: &[Tile], cfa_buf: &mut [u16], tile_id: usize, plane_id: usize, plane_row: usize, plane_buf: &[u16]) -> Result<()> {
  // 2x2 pixel for RGGB
  const CFA_DIM: usize = 2;

  debug_assert_ne!(plane_buf.len(), 0);
  debug_assert_ne!(cfa_buf.len(), 0);
  debug_assert!(codec.tile_cols > 0);
  debug_assert!(codec.tile_rows > 0);

  if plane_id > 3 {
    return Err(CrxError::Overflow(format!(
      "More then 4 planes detected, unable to process plane_id {}",
      plane_id
    )));
  }

  let tile_row_idx = tile_id / codec.tile_cols; // round down
  let tile_col_idx = tile_id % codec.tile_cols; // round down

  // Offset from top
  let row_offset = tile_row_idx * codec.tile_width;
  // Offset from left
  let col_offset = tile_col_idx * codec.tile_width;
  let (row_shift, col_shift) = match plane_id {
    0 => (0, 0),
    1 => (0, 1),
    2 => (1, 0),
    3 => (1, 1),
    _ => {
      return Err(CrxError::General("Invalid plane id".to_string()));
    }
  };
  //println!("plane_width: {}, buf_size: {}", tiles[tile_id].plane_width, plane_buf.len());
  let row_idx = row_offset + (plane_row * CFA_DIM) + row_shift;
  for plane_col in 0..tiles[tile_id].plane_width {
    // Row index into CFA for untiled full area
    let col_idx = col_offset + (plane_col * CFA_DIM) + col_shift;

    // Copy from plane to CFA
    cfa_buf[(row_idx * codec.image_width) + col_idx] = plane_buf[plane_col];
  }
  Ok(())
}
