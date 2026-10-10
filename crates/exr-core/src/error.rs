//! Port of OpenEXR's Iex exception hierarchy as used by IlmImf.
//!
//! OpenEXR / IlmImf reports errors by throwing C++ exceptions derived from
//! `Iex::BaseExc`.  This module maps those categories to a Rust `enum` so that
//! every fallible function in `exr-core` returns `Result<T, ExrError>` rather
//! than panicking.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/Iex/IexBaseExc.h>
//! and <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/Iex/IexMathExc.h>.
//!
//! **Hierarchy** (`IexBaseExc.h`, `IexMathExc.h`): `ArgExc`, `LogicExc`,
//! `InputExc`, `IoExc`, `MathExc`, `ErrnoExc`, `NoImplExc`, `NullExc` and
//! `TypeExc` are all direct children of `BaseExc`. `OverflowExc`,
//! `UnderflowExc`, `DivzeroExc`, `InexactExc` and `InvalidFpOpExc` derive
//! from `MathExc`. `ErrnoExc` is a **sibling** of `IoExc` (both children of
//! `BaseExc`), not a subclass of it.
//!
//! **`NoImplExc` and `NullExc` have zero throw sites** in `OpenEXR/` and
//! `OpenEXRUtil/` (verified by grep against the v3.4.15 tree), so this port
//! does not carry variants for them — they would be permanently dead code.
//! `InexactExc` and `InvalidFpOpExc` are likewise unused by any
//! IlmImf/OpenEXRCore call site this crate reaches; add them if and when
//! that changes (YAGNI). `UnderflowExc` and `DivzeroExc` **are** reachable —
//! [`crate::checked_arithmetic`] (a 1:1 port of `ImfCheckedArithmetic.h`,
//! whose `uiSub`/`uiDiv` throw exactly these) is used at the ported
//! equivalents of `ImfCompressor.cpp` (tiled RLE buffer size),
//! `ImfPreviewImage.cpp` (`checkArraySize(uiMult(w,h))`) and `ImfZip.cpp`
//! (`uiMult(maxScanLineSize, numScanLines)`) — so both have variants below.

use std::fmt;

/// Crate-wide error type mirroring the Iex exception categories used by IlmImf.
///
/// Each variant corresponds to one of the C++ Iex exception classes that
/// IlmImf raises.  The `msg` string carries the human-readable message that
/// would appear in `e.what()`.
///
/// `#[non_exhaustive]`: new Iex classes may be added as more of IlmImf's
/// call-site behaviour is ported; matches outside this crate should end in
/// a wildcard arm (or use [`ExrError::iex_class`] / [`ExrError::is_a`]
/// instead of matching variants directly when only the class matters).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExrError {
    /// `Iex::BaseExc` — general / unclassified error.
    Base(String),

    /// `Iex::InputExc` — malformed or truncated input data (most read errors
    /// land here, e.g. "early end of file").
    Input(String),

    /// `Iex::ArgExc` — invalid argument supplied by the caller.
    Arg(String),

    /// `Iex::IoExc` — I/O operation failed (e.g. the decode/encode pipeline;
    /// see `ImfScanLineInputFile.cpp`, `ImfOutputFile.cpp`).
    Io(String),

    /// `Iex::ErrnoExc` — a C `errno`-style OS failure (open/read/write on the
    /// underlying file). A *sibling* of `Io`, not a subclass of it — see the
    /// module doc.
    Errno(String),

    /// `Iex::LogicExc` — internal logic error (invariant violated).
    Logic(String),

    /// `Iex::TypeExc` — an object is the wrong type, e.g. a `Header` attribute
    /// re-inserted with a different type, or a failed `TypedAttribute::cast`.
    Type(String),

    /// `Iex::OverflowExc` (a `MathExc`) — arithmetic or buffer overflow.
    /// Thrown by [`crate::checked_arithmetic`]'s `ui_mult`/`ui_add`/
    /// `check_array_size` (`uiMult`/`uiAdd`/`checkArraySize`).
    Overflow(String),

    /// `Iex::UnderflowExc` (a `MathExc`) — thrown by
    /// [`crate::checked_arithmetic::ui_sub`] (`uiSub`) when `a < b`.
    Underflow(String),

    /// `Iex::DivzeroExc` (a `MathExc`) — thrown by
    /// [`crate::checked_arithmetic::ui_div`] (`uiDiv`) on division by zero.
    DivZero(String),

    /// A file shape the idiomatic [`crate::Image`] facade deliberately does not
    /// cover (deep / multipart / subsampled) — see [`Unsupported`].
    ///
    /// **Why this is the one intentional exception to "`ExrError` is a 1:1 Iex
    /// port".** Every other variant mirrors a C++ `Iex` exception class. This
    /// one does not: it exists to replace a fragile *cross-crate* contract. The
    /// facade previously reported these shapes as `ExrError::Arg(String)` and
    /// downstream crates (`exr-gpu`, `exr-view`) recovered by matching English
    /// substrings (`msg.contains("… not supported by the Image facade …")`).
    /// Any reword silently broke GPU/viewer routing, and the viewer even mapped
    /// *every* `Arg` to "unsupported", mislabeling unrelated argument errors.
    ///
    /// Making the kind a structured, `Copy` variant means consumers branch on
    /// the variant — stable and exhaustive — and never on wording. The
    /// human-readable pointer to the right low-level API lives only in
    /// [`Display`](fmt::Display), so the message can change freely without
    /// touching a single consumer.
    Unsupported(Unsupported),
}

/// Which Iex exception class an [`ExrError`] corresponds to, modeling the
/// `BaseExc` hierarchy that a flat Rust enum can't express directly.
///
/// `is_a` walks this hierarchy so e.g. `Overflow.is_a(Math)` and
/// `Overflow.is_a(Base)` are both `true`, matching `dynamic_cast<MathExc*>`
/// / `dynamic_cast<BaseExc*>` succeeding on a thrown `OverflowExc` in C++.
/// `Underflow` and `DivZero` are `Math` subclasses the same way (`IexMathExc.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IexClass {
    Base,
    Arg,
    Logic,
    Input,
    Io,
    Errno,
    Type,
    Math,
    Overflow,
    Underflow,
    DivZero,
}

impl IexClass {
    /// Mirrors C++ `dynamic_cast<Target*>(&thrown)` succeeding: every class
    /// `is_a(Base)`, and every `Math` subclass (`Overflow`, `Underflow`,
    /// `DivZero`) additionally `is_a(Math)`.
    pub fn is_a(self, other: IexClass) -> bool {
        if self == other {
            return true;
        }
        match self {
            IexClass::Overflow | IexClass::Underflow | IexClass::DivZero => {
                other == IexClass::Math || other == IexClass::Base
            }
            IexClass::Base => false,
            _ => other == IexClass::Base,
        }
    }
}

impl ExrError {
    /// Convenience constructor for [`ExrError::Errno`] from a
    /// [`std::io::Error`], mirroring `IEX_NAMESPACE::throwErrnoExc()` (formats
    /// the C `errno` message and throws an `ErrnoExc`, a direct `BaseExc`
    /// subclass — see the module doc). An `UnexpectedEof` maps to
    /// [`ExrError::Input`] instead, matching `Imf::StdIFStream::read`'s
    /// `checkError` throwing `InputExc` "Early end of file" rather than an
    /// errno exception when a short read is not an OS error
    /// (`ImfStdIO.cpp`).
    #[inline]
    pub fn from_io(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            ExrError::Input(e.to_string())
        } else {
            ExrError::Errno(e.to_string())
        }
    }

    /// The Iex exception class this variant corresponds to.
    pub fn iex_class(&self) -> IexClass {
        match self {
            ExrError::Base(_) => IexClass::Base,
            ExrError::Input(_) => IexClass::Input,
            ExrError::Arg(_) => IexClass::Arg,
            ExrError::Io(_) => IexClass::Io,
            ExrError::Errno(_) => IexClass::Errno,
            ExrError::Logic(_) => IexClass::Logic,
            ExrError::Type(_) => IexClass::Type,
            ExrError::Overflow(_) => IexClass::Overflow,
            ExrError::Underflow(_) => IexClass::Underflow,
            ExrError::DivZero(_) => IexClass::DivZero,
            // Not a real Iex class; closest ancestor for classification.
            ExrError::Unsupported(_) => IexClass::Arg,
        }
    }

    /// `self.iex_class().is_a(class)` — convenience for callers that only
    /// care about the class (e.g. `err.is_a(IexClass::Math)`).
    pub fn is_a(&self, class: IexClass) -> bool {
        self.iex_class().is_a(class)
    }
}

impl fmt::Display for ExrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExrError::Base(m) => write!(f, "OpenEXR error: {m}"),
            ExrError::Input(m) => write!(f, "OpenEXR input error: {m}"),
            ExrError::Arg(m) => write!(f, "OpenEXR argument error: {m}"),
            ExrError::Io(m) => write!(f, "OpenEXR I/O error: {m}"),
            ExrError::Errno(m) => write!(f, "OpenEXR I/O error: {m}"),
            ExrError::Logic(m) => write!(f, "OpenEXR logic error: {m}"),
            ExrError::Type(m) => write!(f, "OpenEXR type error: {m}"),
            ExrError::Overflow(m) => write!(f, "OpenEXR overflow error: {m}"),
            ExrError::Underflow(m) => write!(f, "OpenEXR underflow error: {m}"),
            ExrError::DivZero(m) => write!(f, "OpenEXR division-by-zero error: {m}"),
            // The helpful "use <low-level API>" pointer lives in `Unsupported`'s
            // own `Display`, so consumers never scrape wording out of here.
            ExrError::Unsupported(u) => write!(f, "OpenEXR unsupported by the Image facade: {u}"),
        }
    }
}

/// Which shape the idiomatic [`crate::Image`] facade rejected.
///
/// The facade covers only the common case (single-part, flat, full-resolution).
/// Deep / multipart / subsampled files are out of scope by design — the flat
/// `Image` model cannot represent N parts, per-pixel sample counts, or
/// mixed-resolution channels — so it returns [`ExrError::Unsupported`] carrying
/// one of these kinds.
///
/// **Why a structured kind instead of a string.** Consumers that fall back per
/// shape (`exr-gpu` opens multipart via `MultiPartInputFile` and subsampled via
/// `RgbaInputFile`; `exr-view` labels the failure in its status bar) branch on
/// this `Copy` variant — never on the diagnostic wording. That replaces the old
/// fragile `String::contains` cross-crate contract, where a reworded message
/// silently broke routing. The human-readable "use the low-level X" hint is
/// this type's [`Display`](fmt::Display); the routing key is the variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// Multipart file — use [`crate::MultiPartInputFile`].
    Multipart,
    /// Deep file — use [`crate::DeepScanLineInputFile`] /
    /// [`crate::DeepTiledInputFile`].
    Deep,
    /// A subsampled (chroma) channel — use [`crate::api::Image::read_all`] (keeps
    /// the native sampling) or the low-level [`crate::frame_buffer::FrameBuffer`] /
    /// [`crate::RgbaInputFile`].
    Subsampled,
}

impl fmt::Display for Unsupported {
    /// The human-readable pointer at the correct low-level API. This is the
    /// *only* place the wording lives; routing consumers match the variant, not
    /// this text, so it can be reworded freely.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::Multipart => write!(
                f,
                "multipart file; use MultiPartInputFile (the low-level multipart API)"
            ),
            Unsupported::Deep => write!(
                f,
                "deep file; use DeepScanLineInputFile / DeepTiledInputFile (the low-level deep API)"
            ),
            Unsupported::Subsampled => write!(
                f,
                "subsampled (chroma) channel; use Image::read_all (native sampling) or the low-level FrameBuffer / RgbaInputFile"
            ),
        }
    }
}

impl std::error::Error for ExrError {}

impl From<std::io::Error> for ExrError {
    fn from(e: std::io::Error) -> Self {
        ExrError::from_io(e)
    }
}

/// Crate-wide `Result` alias.
///
/// Every fallible public function in `exr-core` returns `Result<T>` (i.e.
/// `Result<T, ExrError>`).
pub type Result<T> = core::result::Result<T, ExrError>;

/// Reclassify a codec failure at the **decode** seam
/// as [`ExrError::Io`] "Unable to run decoder: …", matching every reader's
/// decode pipeline in C++ 3.4.15 (`ImfScanLineInputFile.cpp:562-613`,
/// `ImfTiledInputFile.cpp:878-914`, `ImfDeepScanLineInputFile.cpp:743-868`,
/// `ImfDeepTiledInputFile.cpp:1066`, threaded via `IlmThreadProcessGroup.h:138`
/// — all `IoExc`). Call as `c.uncompress(..).map_err(decode_err)` at every
/// read seam that runs a [`crate::compressor::Compressor`]. This is also
/// the seam the rayon workers call, so its error class is stable across
/// thread counts.
pub fn decode_err(e: ExrError) -> ExrError {
    ExrError::Io(format!("Unable to run decoder: {e}"))
}

/// Reclassify a codec failure at the **encode** seam
/// as [`ExrError::Io`] "Failed to write pixel data to image file. …",
/// matching every writer's encode pipeline in C++ 3.4.15
/// (`ImfOutputFile.cpp:1197-1225`, `ImfTiledOutputFile.cpp:1383`,
/// `ImfDeepScanLineOutputFile.cpp:1393`, `ImfDeepTiledOutputFile.cpp:1661` —
/// all `IoExc`, `REPLACE_EXC`'d from whatever the codec threw). Call as
/// `c.compress(..).map_err(encode_err)` at every write seam. Rust streams
/// have no file name at this layer (unlike C++'s `"…to image file \"NAME\""`),
/// so the message omits it.
pub fn encode_err(e: ExrError) -> ExrError {
    ExrError::Io(format!("Failed to write pixel data to image file. {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_is_a_math_and_base() {
        let e = ExrError::Overflow("x".into());
        assert!(e.is_a(IexClass::Overflow));
        assert!(e.is_a(IexClass::Math));
        assert!(e.is_a(IexClass::Base));
        assert!(!e.is_a(IexClass::Input));
    }

    #[test]
    fn every_class_is_a_base() {
        for c in [
            IexClass::Base,
            IexClass::Arg,
            IexClass::Logic,
            IexClass::Input,
            IexClass::Io,
            IexClass::Errno,
            IexClass::Type,
            IexClass::Math,
            IexClass::Overflow,
            IexClass::Underflow,
            IexClass::DivZero,
        ] {
            assert!(c.is_a(IexClass::Base));
        }
    }

    #[test]
    fn underflow_and_divzero_are_math_and_base() {
        for c in [IexClass::Underflow, IexClass::DivZero] {
            assert!(c.is_a(IexClass::Math));
            assert!(c.is_a(IexClass::Base));
            assert!(!c.is_a(IexClass::Input));
        }
    }

    #[test]
    fn from_io_unexpected_eof_is_input() {
        let e = ExrError::from_io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "early end of file",
        ));
        assert!(matches!(e, ExrError::Input(_)));
    }

    #[test]
    fn from_io_other_kind_is_errno() {
        let e = ExrError::from_io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ));
        assert!(matches!(e, ExrError::Errno(_)));
    }
}
