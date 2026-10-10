//! Scanline image I/O — port of OpenEXR `ImfOutputFile.*` / `ImfScanLineInputFile.*`
//! / `ImfInputFile.*` (the generic single-part reader front door).
//!
//! `output.rs` is the single-part scanline writer (all 12 codecs, `copyPixels`,
//! preview-image update, `breakScanLine`); `input.rs` is the single-part
//! scanline reader; `input_file.rs` is `InputFile`, the storage-dispatching
//! front door (scanline / tiled-as-scanlines / deep-scanline-via-
//! `CompositeDeepScanLine`) plus its multipart-cache mirror
//! (`InputFilePartReader`/`PartReaderState`).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfOutputFile.cpp>

pub mod input;
pub mod input_file;
pub mod output;

pub use input::ScanLineInputFile;
pub use input_file::{GenericInputReader, InputFile, InputFilePartReader};
pub use output::{OutputFile, OutputFileOwned, OutputPartWriter, ScanLineCopySource};
pub(crate) use input::ScanLineInputFileState;
pub(crate) use input_file::{DeepScanLineReaderState, PartReaderState, TiledReaderState};
pub(crate) use output::OutputFileState;
