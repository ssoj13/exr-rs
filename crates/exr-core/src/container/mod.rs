//! Port of OpenEXR's on-disk container layer — Xdr encoding and version/magic constants.
//!
//! This module is named `container` (not `core`) to avoid colliding with Rust's
//! built-in `::core` crate.  It groups the two lowest-level IlmImf modules that
//! define the on-disk wire format:
//!
//! * [`xdr`] — port of `ImfXdr.h`: little-endian read/write for every scalar type.
//! * [`version`] — port of `ImfVersion.h`/`.cpp`: magic number, version word, flag bits.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfXdr.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfVersion.h>

pub mod test_file;
pub mod version;
pub mod xdr;

pub use version::{
    ALL_FLAGS, EXR_VERSION, LIBRARY_VERSION, LONG_NAMES_FLAG, MAGIC, MULTI_PART_FILE_FLAG,
    NON_IMAGE_FLAG, TILED_FLAG, VERSION_FLAGS_FIELD, VERSION_NUMBER_FIELD,
    get_flags, get_library_version, get_version, is_imf_magic, is_multi_part, is_non_image,
    is_single_part_general, is_tiled, make_not_tiled, make_tiled, supports_flags,
};

pub use test_file::{
    ExrFileProbe, is_deep_open_exr_file, is_deep_open_exr_file_stream, is_multi_part_open_exr_file,
    is_multi_part_open_exr_file_stream, is_open_exr_file, is_open_exr_file_stream,
    is_tiled_open_exr_file, is_tiled_open_exr_file_stream,
};

pub use xdr::{
    SIZE_BOOL, SIZE_F32, SIZE_F64, SIZE_HALF, SIZE_I16, SIZE_I32, SIZE_I64, SIZE_I8,
    pad, read_bool, read_char, read_chars, read_f32, read_f64, read_half, read_i16, read_i32,
    read_i64, read_i8, read_string, read_u16, read_u32, read_u64, read_u8_val, skip, write_bool,
    write_char, write_chars, write_f32, write_f64, write_half, write_i16, write_i32, write_i64,
    write_i8, write_string, write_u16, write_u32, write_u64, write_u8_val,
};
