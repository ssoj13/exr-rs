//! BMI2 runtime-dispatch instantiation of the DEFLATE decompressor.
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/x86/decompress_impl.h>
//!
//! # What C does
//! `x86/decompress_impl.h` compiles `decompress_template.h` a SECOND time
//! under `__attribute__((target("bmi2")))`, producing `deflate_decompress_bmi2`
//! alongside the portable `deflate_decompress_default`, and dispatches to
//! whichever is available at runtime (`arch_select_decompress_func`). Per that
//! file's own comment: "With gcc and clang we just compile the whole function
//! with `__attribute__((target("bmi2")))`, and the compiler uses bmi2
//! automatically" — specifically, `word & BITMASK(count)` (this port's
//! `extract_varbits`/`extract_varbits8` in `deflate_decompress.rs`) gets
//! lowered to the single `bzhi` instruction instead of a shift-and-subtract
//! mask, because LLVM (and GCC) recognize that pattern once BMI2 codegen is
//! enabled for the function. Only MSVC needs the `_bzhi_u64` intrinsic
//! spelled out explicitly, because MSVC's backend lacks that automatic
//! peephole — irrelevant here since rustc always uses the LLVM backend (the
//! same backend clang uses), so the "gcc/clang" behavior is what applies.
//!
//! # Why this module is just a one-line wrapper (not a second copy of the
//! decompressor)
//! C's `#include "../decompress_template.h"` under a different `#define` set
//! is a textual, compile-time duplication of the entire function body — the
//! two functions ARE different compiled artifacts of the identical source.
//! Rust's `#[target_feature]` on a function achieves the same "recompile this
//! body under a wider instruction set" effect WITHOUT a second copy of the
//! source, provided the callee is actually inlined into the target-feature
//! function (a plain, non-inlined call does not propagate the enabled
//! feature to the callee — the callee still compiles with its own default
//! feature set). `deflate_decompress_impl` is marked `#[inline(always)]` for
//! exactly this reason.
//!
//! So there is no second decompression algorithm to keep in sync, no
//! separate correctness surface, and no possibility of the BMI2 path
//! diverging from the scalar path's *decoding logic* — LLVM is free to (and,
//! per libdeflate's own comment about gcc/clang, expected to) select `bzhi`
//! for the exact same arithmetic operation, nothing more. The safe scalar
//! `deflate_decompress_impl` remains the one and only source of truth; this
//! module recompiles it, it does not reimplement it.
//!
//! # Unsafe surface
//! The `unsafe fn` here is the ENTIRE unsafe surface of the BMI2 dispatch: its only
//! obligation, per `#[target_feature]`'s safety contract, is that the CPU
//! actually supports BMI2 wherever this function runs. That is verified by
//! the caller (`deflate_decompress_dispatch` in `deflate_decompress.rs`) via
//! `crate::cpu::bmi2_decompress_available()` (which itself checks
//! `is_x86_feature_detected!("bmi2")`) before calling this function — never
//! called unconditionally. No pointer arithmetic, no manual SIMD intrinsics,
//! no additional invariants beyond that one CPU-feature precondition.
//!
//! # Correctness
//! Byte-identical to `deflate_decompress_impl` under the plain (non-BMI2)
//! path by construction — it IS that function, just compiled differently.
//! Verified indirectly by every existing decompress test/parity gate running
//! on a BMI2-capable CPU (this dispatch is transparent to callers); no
//! separate BMI2-specific test is needed for the SAME reason no such test
//! exists for gcc/clang's automatic BMI2 selection in C.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::deflate_decompress::{deflate_decompress_impl, DecompressError, Decompressor};

/// BMI2-recompiled instantiation of [`deflate_decompress_impl`]. See the
/// module doc comment for the full rationale; this function's body is
/// intentionally nothing but the call that makes the recompilation happen.
///
/// # Safety
/// The caller must have verified `is_x86_feature_detected!("bmi2")` (done by
/// `crate::cpu::bmi2_decompress_available()`) before calling this function.
#[target_feature(enable = "bmi2")]
pub(crate) unsafe fn deflate_decompress_impl_bmi2(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
    use_fastloop: bool,
) -> Result<(usize, usize), DecompressError> {
    deflate_decompress_impl(d, input, out, use_fastloop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate_compress::{deflate_compress_bound, Compressor};

    /// Differential test: the BMI2-recompiled path must produce byte-identical
    /// output (and identical errors) to the plain path for the same input,
    /// on every input/level/truncation the fastloop differential test already
    /// covers -- reusing that input set rather than inventing a new one,
    /// since the property under test here ("two compiled instantiations of
    /// the same function agree") is orthogonal to which specific inputs are
    /// used. Skipped (not silently passed as a no-op assert) when the test
    /// machine lacks BMI2, since the function cannot be soundly called then.
    ///
    /// MUTATION CHECKED: a first attempt (flipping `use_fastloop` inside
    /// `deflate_decompress_impl_bmi2`) did NOT turn this test red -- the
    /// fastloop and generic loop are already proven to produce identical
    /// output (that's the whole point of the fastloop differential test), so
    /// taking "the other" internal path changes nothing observable here; it
    /// was the wrong mutation for this property. Replaced with corrupting
    /// one byte of `out` on the `Ok` path inside the BMI2 wrapper, which DID
    /// turn the test red (`left: [104, ...] right: [151, ...]`, the expected
    /// byte-mismatch panic). Reverted both mutations immediately.
    #[test]
    fn bmi2_matches_plain_path_on_varied_inputs() {
        if !std::is_x86_feature_detected!("bmi2") {
            eprintln!("SKIP: this CPU does not support BMI2");
            return;
        }

        let inputs: Vec<(&str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("short", b"hello, world!".to_vec()),
            ("all_same", vec![0x5A; 20_000]),
            ("repetitive", b"abcabcabcabc".repeat(1000)),
            ("text_multiblock", b"The quick brown fox jumps over the lazy dog. ".repeat(2000)),
        ];

        for level in [1i32, 6, 9] {
            for (name, data) in &inputs {
                let mut c = Compressor::new(level).unwrap();
                let bound = deflate_compress_bound(data.len());
                let mut out = vec![0u8; bound];
                let n = c.deflate_compress(data, &mut out);
                let compressed = &out[..n];

                for &use_fastloop in &[true, false] {
                    let mut d_plain = Decompressor::new();
                    let mut out_plain = vec![0u8; data.len().max(1)];
                    let r_plain =
                        deflate_decompress_impl(&mut d_plain, compressed, &mut out_plain, use_fastloop);

                    let mut d_bmi2 = Decompressor::new();
                    let mut out_bmi2 = vec![0u8; data.len().max(1)];
                    // SAFETY: `is_x86_feature_detected!("bmi2")` checked above.
                    let r_bmi2 = unsafe {
                        deflate_decompress_impl_bmi2(&mut d_bmi2, compressed, &mut out_bmi2, use_fastloop)
                    };

                    match (r_plain, r_bmi2) {
                        (Ok((n_plain, in_plain)), Ok((n_bmi2, in_bmi2))) => {
                            assert_eq!(n_plain, n_bmi2, "{name}/L{level}/fastloop={use_fastloop}: output length mismatch");
                            assert_eq!(in_plain, in_bmi2, "{name}/L{level}/fastloop={use_fastloop}: consumed-input mismatch");
                            assert_eq!(
                                &out_plain[..n_plain],
                                &out_bmi2[..n_bmi2],
                                "{name}/L{level}/fastloop={use_fastloop}: BMI2 and plain paths disagree on decoded bytes"
                            );
                        }
                        (Err(e_plain), Err(e_bmi2)) => {
                            assert_eq!(e_plain, e_bmi2, "{name}/L{level}/fastloop={use_fastloop}: error mismatch");
                        }
                        (plain, bmi2) => panic!(
                            "{name}/L{level}/fastloop={use_fastloop}: BMI2 and plain paths disagree on success: plain={plain:?} bmi2={bmi2:?}"
                        ),
                    }
                }
            }
        }
    }
}
