//! Scalar float ops with x86 SSE NaN propagation, for code whose C++ original is compiled by
//! MSVC x64 (our parity oracle) and whose NaN payloads are therefore observable in the output.
//!
//! `addss`/`mulss`/`divss dst, src`: when `dst` is a NaN the result is `dst` quieted, else when
//! `src` is a NaN it is `src` quieted, else the IEEE result. Which NaN wins when both operands
//! are NaN thus depends on which operand the compiler put in the destination register, and
//! LLVM is free to swap the operands of a commutative `+`/`*`; spelling the order out here keeps
//! a payload independent of either optimizer. Callers give the operands in the order the C++
//! binary has them, established by an oracle run (see `rgba::aces`).
//!
//! Only NaN *selection* is modelled; a NaN *created* by an operation (`inf - inf`, `0 * inf`) is
//! whatever the host's FPU makes of it, exactly as in the C++.

macro_rules! sse_ops {
    ($m:ident, $f:ty, $quiet:expr) => {
        /// The ops for one float width (`ss` = f32, `sd` = f64, after the SSE mnemonics); `dst` is
        /// the operand that wins a NaN tie.
        pub mod $m {
            /// `x` with its quiet bit set, as every SSE arithmetic op returns a NaN operand.
            #[inline]
            pub fn quiet(x: $f) -> $f {
                <$f>::from_bits(x.to_bits() | $quiet)
            }

            #[inline]
            fn op(dst: $f, src: $f, f: impl FnOnce($f, $f) -> $f) -> $f {
                if dst.is_nan() {
                    quiet(dst)
                } else if src.is_nan() {
                    quiet(src)
                } else {
                    f(dst, src)
                }
            }

            /// `addss dst, src`.
            #[inline]
            pub fn add(dst: $f, src: $f) -> $f {
                op(dst, src, |a, b| a + b)
            }

            /// `mulss dst, src`.
            #[inline]
            pub fn mul(dst: $f, src: $f) -> $f {
                op(dst, src, |a, b| a * b)
            }

            /// `divss dst, src`.
            #[inline]
            pub fn div(dst: $f, src: $f) -> $f {
                op(dst, src, |a, b| a / b)
            }
        }
    };
}

sse_ops!(ss, f32, 0x0040_0000u32);
sse_ops!(sd, f64, 0x0008_0000_0000_0000u64);

#[cfg(test)]
mod tests {
    use super::*;

    fn nan32(payload: u32) -> f32 {
        f32::from_bits(0x7fc0_0000 | payload)
    }

    /// The destination NaN wins a tie, a lone NaN wins regardless of position, and a signalling
    /// NaN comes out quiet with its payload kept.
    #[test]
    fn nan_selection_follows_the_destination_operand() {
        let (a, b) = (nan32(1), nan32(2));
        assert_eq!(ss::add(a, b).to_bits(), a.to_bits());
        assert_eq!(ss::add(b, a).to_bits(), b.to_bits());
        assert_eq!(ss::mul(a, b).to_bits(), a.to_bits());
        assert_eq!(ss::mul(b, a).to_bits(), b.to_bits());
        assert_eq!(ss::div(b, a).to_bits(), b.to_bits());
        assert_eq!(ss::mul(1.0, b).to_bits(), b.to_bits());
        let (c, d) = (f64::from_bits(0x7ff8_0000_0000_0001), f64::from_bits(0x7ff8_0000_0000_0002));
        assert_eq!(sd::add(c, d).to_bits(), c.to_bits());
        assert_eq!(sd::add(d, c).to_bits(), d.to_bits());
        assert_eq!(sd::mul(d, c).to_bits(), d.to_bits());
        assert_eq!(ss::div(a, 2.0).to_bits(), a.to_bits());
        assert_eq!(ss::add(f32::from_bits(0x7f80_0005), 1.0).to_bits(), 0x7fc0_0005);
        assert_eq!(sd::add(f64::NAN, 1.0).to_bits(), f64::NAN.to_bits() | 0x0008_0000_0000_0000);
    }

    #[test]
    fn plain_operands_are_ordinary_arithmetic() {
        assert_eq!(ss::add(1.5, 2.25), 3.75);
        assert_eq!(ss::mul(1.5, 2.0), 3.0);
        assert_eq!(sd::div(1.0, 4.0), 0.25);
        assert_eq!(ss::div(1.0, 0.0), f32::INFINITY);
    }
}
