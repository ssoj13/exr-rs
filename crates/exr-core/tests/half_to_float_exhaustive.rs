//! P2-READ P-8: the unpack that turns a file's HALF samples into a FLOAT destination
//! (`copy_into_frame_buffer`, the one conversion every scanline / tiled read goes through) is EXACT for
//! ALL 65536 half bit patterns: zero, subnormals, normals, infinities and every NaN payload. The
//! result is compared bit for bit against `f32::from(Half)` (IEEE binary16 -> binary32 widening),
//! so a planar `gbrpf32le` decode of a half EXR carries exactly the values the half file holds.

use exr_core::attr::PixelType;
use exr_core::misc::{copy_into_frame_buffer, BufferFormat};
use imath_rs::Half;

#[test]
fn every_half_bit_pattern_unpacks_to_the_exact_float() {
    // The file side is XDR (little-endian) HALF; the destination FLOAT is host-native.
    let mut src = Vec::with_capacity(65536 * 2);
    for bits in 0..=u16::MAX {
        src.extend_from_slice(&bits.to_le_bytes());
    }
    let mut dst = vec![0u8; 65536 * 4];
    copy_into_frame_buffer(
        &src,
        0,
        &mut dst,
        0,
        65536,
        4,
        false,
        0.0,
        BufferFormat::Xdr,
        PixelType::Float,
        PixelType::Half,
    )
    .expect("unpack");
    for bits in 0..=u16::MAX {
        let got = f32::from_ne_bytes(dst[bits as usize * 4..][..4].try_into().unwrap());
        let want = f32::from(Half::from_bits(bits));
        assert_eq!(got.to_bits(), want.to_bits(), "half {bits:#06x}: {got:?} vs {want:?}");
    }
}
