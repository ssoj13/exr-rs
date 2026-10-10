//! Port of OpenEXR `ImfIO.h` — low-level I/O layer.
//!
//! Provides the abstract stream traits ([`IStream`], [`OStream`]) and their
//! standard-library-backed concrete implementations.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfIO.h>

pub mod std_io;
pub mod stream;

pub use std_io::{write_file, MemIStream, MemOStream, StdIfStream, StdOfStream};
pub use stream::{IStream, OStream};
