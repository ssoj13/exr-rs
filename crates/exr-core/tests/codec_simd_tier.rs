//! Engagement check: the ZIP/RLE decode-side predictor
//! (`compressor::predictor`) runs through `codec-simd`, whose dispatcher picks
//! a SIMD body per CPU. Byte identity against the scalar reference cannot tell
//! whether a SIMD body was selected at all, so this checks the dispatch table
//! itself: on a CPU with the feature, the entries exr-core uses must not be
//! the scalar bodies.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn predictor_primitives_dispatch_to_simd_when_the_cpu_has_it() {
    let active = codec_simd::Dispatch::detect();
    let scalar = codec_simd::Dispatch::scalar();
    if std::is_x86_feature_detected!("sse4.1") {
        assert_ne!(
            active.prefix_sum_bias_u8_inplace as usize,
            scalar.prefix_sum_bias_u8_inplace as usize,
            "SSE4.1 CPU, but prefix_sum_bias_u8_inplace dispatches to the scalar body"
        );
    }
    if std::is_x86_feature_detected!("sse2") {
        assert_ne!(
            active.interleave_halves_u8 as usize,
            scalar.interleave_halves_u8 as usize,
            "SSE2 CPU, but interleave_halves_u8 dispatches to the scalar body"
        );
    }
}
