//! A container whose header names a codec this build cannot open must be REJECTED AT OPEN with the typed
//! reason (`DecodeError::Codec(OpenError)`), never decoded wrong or reported as generic corruption.

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, CodecId, DecodeError, Dtype, Header, OpenError};
use exv_video::{VideoIndex, write_still};

/// Byte offset of the header's `codec_id` in a still container: prelude (magic u32, version u8, gop u32,
/// frame_count u32) + header (version u8, flags u8, width/height/frame_count/tile_w/tile_h u32).
const CODEC_ID_OFF: usize = 13 + 2 + 20;

/// A 16x16 single-channel F16 still.
fn still() -> Image {
    let header = Header {
        width: 16,
        height: 16,
        frame_count: 1,
        tile_w: 16,
        tile_h: 16,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![Channel { name: "Y".into(), dtype: Dtype::F16 }],
    };
    Image::new(header, vec![vec![0u8; 16 * 16 * 2]]).expect("image")
}

fn forged(id: u32) -> Vec<u8> {
    let mut bytes = write_still(&still(), &Profile::lossless_fast()).expect("write");
    let at = CODEC_ID_OFF..CODEC_ID_OFF + 4;
    assert_eq!(bytes[at.clone()], CodecId::LOSSLESS_FAST.0.to_le_bytes(), "offset points at codec_id");
    bytes[at].copy_from_slice(&id.to_le_bytes());
    bytes
}

#[test]
fn unopenable_codec_is_a_typed_open_error() {
    let id = CodecId;
    for (raw, want) in [
        (0x8000_0001, OpenError::Vendor(id(0x8000_0001))),
        (5, OpenError::Unknown(id(5))),
        (0, OpenError::NoProfile(CodecId::RAW)),
    ] {
        assert_eq!(VideoIndex::parse(&forged(raw)).err(), Some(DecodeError::Codec(want)), "codec_id {raw:#x}");
    }
    // a valid file still opens
    assert!(VideoIndex::parse(&forged(CodecId::LOSSLESS_FAST.0)).is_ok());
}
