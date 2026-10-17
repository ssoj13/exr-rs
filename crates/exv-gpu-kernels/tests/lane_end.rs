//! Lane-end bounds of the rANS kernels (GPU-M1): every byte read is clamped to the lane's own `[start, end)`
//! span and reads 0 past it, exactly like the CPU `byte_coder::Decoder`. Without the clamp a truncated lane
//! would read its NEIGHBOUR's bytes (the lane streams are concatenated in one buffer) and the GPU output would
//! depend on the layout of unrelated lanes.
//!
//! The CPU decoder now REJECTS such streams (`done()`), so the reference here is a tiny zero-feeding decoder
//! that mirrors its arithmetic. Needs a real adapter: `cargo test -p exv-gpu-kernels -- --include-ignored`.

use exv_gpu_kernels::{GpuContext, decode_block_gpu};
use rans::{Prepared, RansConfig, prepare};

const RANS_L: u32 = 1 << 23;
/// `rans::byte_coder::RENORM_GUARD` (the single CPU<->WGSL renorm bound).
const GUARD: u32 = 4;

/// Zero-feeding reference decoder (byte-for-byte the CPU `Decoder` arithmetic, minus the integrity verdict).
fn ref_decode(p: &Prepared) -> Vec<u8> {
    let lanes = p.cfg.lanes as usize;
    let tl = p.cfg.table_log as u32;
    let mask = (1u32 << tl) - 1;
    let mut out = vec![0u8; p.n];
    for (k, st) in p.lane_streams.iter().enumerate() {
        let byte = |i: usize| st.get(i).copied().unwrap_or(0) as u32;
        let mut state = byte(0) | (byte(1) << 8) | (byte(2) << 16) | (byte(3) << 24);
        let mut pos = 4;
        let cnt = if p.n > k { (p.n - k).div_ceil(lanes) } else { 0 };
        for j in 0..cnt {
            let s = p.slot_to_sym[(state & mask) as usize] as usize;
            state = (p.freq[s] as u32)
                .wrapping_mul(state >> tl)
                .wrapping_add(state & mask)
                .wrapping_sub(p.cum[s] as u32);
            let mut g = 0;
            while state < RANS_L && g < GUARD {
                state = (state << 8) | byte(pos);
                pos += 1;
                g += 1;
            }
            out[k + j * lanes] = s as u8;
        }
    }
    out
}

/// A skewed pseudo-random symbol stream (so lanes have different, non-trivial lengths).
fn symbols(n: usize) -> Vec<u8> {
    let mut x = 0x2545_f491u32;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 7) as u8 * (x >> 28) as u8 % 61
        })
        .collect()
}

#[test]
#[ignore = "requires a GPU"]
fn valid_stream_matches_reference_and_input() {
    let ctx = GpuContext::new().expect("gpu ctx");
    let syms = symbols(5000);
    let prep = prepare(&syms, RansConfig::new(8, 12)).expect("encode");
    let gpu = decode_block_gpu(&ctx, &prep).expect("gpu decode");
    assert_eq!(gpu, syms, "GPU != original on a valid stream");
    assert_eq!(ref_decode(&prep), syms, "reference decoder != original (test bug)");
}

#[test]
#[ignore = "requires a GPU"]
fn truncated_lanes_read_zero_not_the_neighbour() {
    let ctx = GpuContext::new().expect("gpu ctx");
    let syms = symbols(5000);
    let mut prep = prepare(&syms, RansConfig::new(8, 12)).expect("encode");
    // Cut the tails of lanes 0, 2 and 5 (keep the 4-byte state): an unbounded GPU read would now run into the
    // next lane's first bytes instead of zeros.
    for k in [0usize, 2, 5] {
        let keep = (prep.lane_streams[k].len() / 2).max(4);
        prep.lane_streams[k].truncate(keep);
    }
    let want = ref_decode(&prep);
    assert_ne!(want, syms, "truncation must actually damage the stream (test bug)");
    let gpu = decode_block_gpu(&ctx, &prep).expect("gpu decode");
    assert_eq!(gpu, want, "GPU over-read must be zero-fed per lane like the CPU, never the neighbour's bytes");
}
