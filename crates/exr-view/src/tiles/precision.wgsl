// Exact bounded area mean. CPU profile guarantees finite normal-or-zero
// inputs, abs(value)<=65504, occupied width<=45, and block count<=256.
// The signed fixed-point sum has <=53 magnitude bits. No floating arithmetic
// occurs until the already rounded finite half value is reinterpreted for store.
struct Params { size: vec2<u32>, scale: u32, pad: u32, low: vec4<i32> }
@group(0) @binding(0) var original: texture_2d<f32>;
@group(0) @binding(1) var reduced: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p: Params;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn magnitude64(a: vec2<u32>) -> vec2<u32> {
    return add64(~a, vec2(1u, 0u));
}
fn insert_value(sum: vec2<u32>, bits: u32, low: i32) -> vec2<u32> {
    let magnitude = bits & 0x7fffffffu;
    if magnitude == 0u { return sum; }
    let mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    let trailing = countTrailingZeros(mantissa);
    let integer = mantissa >> trailing;
    let signed_shift = i32(magnitude >> 23u) - 150 + i32(trailing) - low;
    var value: vec2<u32>;
    // A negative shift only occurs in `main_wide`, whose per-cell `low` may lie
    // above a small sample's lowest bits: those bits are truncated (toward zero).
    // The exact kernel's `low` is the global minimum, so it never takes it.
    if signed_shift < 0 {
        let drop = u32(-signed_shift);
        value = vec2(select(integer >> min(drop, 31u), 0u, drop >= 32u), 0u);
        if (bits & 0x80000000u) != 0u { return sub64(sum, value); }
        return add64(sum, value);
    }
    let shift = u32(signed_shift);
    // WGSL shifts are modulo 32; zero and the split boundary need branches.
    if shift == 0u {
        value = vec2(integer, 0u);
    } else if shift < 32u {
        value = vec2(integer << shift, integer >> (32u - shift));
    } else {
        value = vec2(0u, integer << (shift - 32u));
    }
    if (bits & 0x80000000u) != 0u { return sub64(sum, value); }
    return add64(sum, value);
}
struct Division { quotient: vec2<u32>, remainder: u32 }
fn divide64_small(value: vec2<u32>, n: u32) -> Division {
    var q = vec2(0u);
    var r = 0u;
    // Base 65536 long division: r<n<=256, so the temporary fits 24 bits.
    for (var remaining = 4u; remaining > 0u; remaining -= 1u) {
        let i = remaining - 1u;
        let shift = (i & 1u) * 16u;
        var limb = value.x;
        if i >= 2u { limb = value.y; }
        let current = (r << 16u) | ((limb >> shift) & 0xffffu);
        let digit = current / n;
        r = current % n;
        if i >= 2u { q.y |= digit << shift; }
        else { q.x |= digit << shift; }
    }
    return Division(q, r);
}
fn round_shift(value: u32, shift: u32) -> u32 {
    let q = value >> shift;
    let tail = value & ((1u << shift) - 1u);
    let midpoint = 1u << (shift - 1u);
    return q + select(0u, 1u, tail > midpoint || (tail == midpoint && (q & 1u) != 0u));
}
// Convert an already rounded normal f32 significand/exponent to half,
// then expand those half bits to an exactly representable normal f32 or zero.
fn half_as_f32_bits(sign: u32, significand: u32, exponent: i32) -> u32 {
    if exponent < -25 { return sign; }
    var half: u32;
    if exponent < -14 {
        half = round_shift(significand, u32(-exponent - 1));
    } else if exponent > 15 {
        half = 0x7c00u;
    } else {
        var fraction = round_shift(significand, 13u);
        var e = exponent + 15;
        if fraction == 0x800u { fraction >>= 1u; e += 1; }
        half = (u32(e) << 10u) | (fraction & 0x3ffu);
    }
    if half == 0u { return sign; }
    // Beyond half range (only `main_wide` can get here: the exact domain caps
    // |mean| at 65504) the half is infinity, like the CPU's `Half::from`.
    if half >= 0x7c00u { return sign | 0x7f800000u; }
    let e = half >> 10u;
    let fraction = half & 0x3ffu;
    if e != 0u { return sign | ((e + 112u) << 23u) | (fraction << 13u); }
    let high = 31u - countLeadingZeros(fraction);
    return sign | (u32(i32(high) + 103) << 23u)
        | ((fraction << (23u - high)) & 0x7fffffu);
}
fn mean_half_bits(sum: vec2<u32>, n: u32, low: i32) -> u32 {
    var sign = 0u;
    var magnitude = sum;
    if (sum.y & 0x80000000u) != 0u {
        sign = 0x80000000u;
        magnitude = magnitude64(sum);
    }
    if all(magnitude == vec2(0u)) { return 0u; }
    let division = divide64_small(magnitude, n);
    let q = division.quotient;
    var r = division.remainder;
    var significand: u32;
    var exponent = low + 23;
    if q.y != 0u || q.x >= 0x1000000u {
        var high = 31u - countLeadingZeros(q.x);
        if q.y != 0u { high = 63u - countLeadingZeros(q.y); }
        let shift = high - 23u; // 1..29: the exact absolute sum has <=53 bits.
        significand = (q.x >> shift) | (q.y << (32u - shift));
        let tail = q.x & ((1u << shift) - 1u);
        let midpoint = 1u << (shift - 1u);
        // A nonzero division remainder makes an integer midpoint strictly above.
        let up = tail > midpoint || (tail == midpoint
            && (r != 0u || (significand & 1u) != 0u));
        significand += select(0u, 1u, up);
        exponent = low + i32(high);
    } else {
        significand = q.x;
        // Generate the remaining significand bits from the rational remainder.
        // The nonzero numerator and n<=256 bound this loop by 31 iterations.
        loop {
            if significand >= 0x800000u { break; }
            significand <<= 1u;
            r <<= 1u;
            if r >= n { r -= n; significand += 1u; }
            exponent -= 1;
        }
        let twice = r << 1u;
        significand += select(0u, 1u, twice > n
            || (twice == n && (significand & 1u) != 0u));
    }
    if significand == 0x1000000u { significand >>= 1u; exponent += 1; }
    // Below this exponent both staged f32->half rounding and exact half
    // rounding yield zero; f32 subnormal construction is therefore unnecessary.
    return half_as_f32_bits(sign, significand, exponent);
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let begin = id.xy * p.scale;
    if any(begin >= p.size) { return; }
    let end = min(begin + vec2(p.scale), p.size);
    let n = (end.x - begin.x) * (end.y - begin.y);
    // A single sample is already the staged f32 mean. Convert its bits with
    // the same explicit ties-even half helper, avoiding accumulation/division.
    if n == 1u {
        let bits = bitcast<vec4<u32>>(textureLoad(original, vec2<i32>(begin), 0));
        var out_bits = vec4(0u);
        for (var c = 0u; c < 4u; c += 1u) {
            let magnitude = bits[c] & 0x7fffffffu;
            // The legacy f64 accumulator starts at +0, including for -0 input.
            if magnitude != 0u {
                out_bits[c] = half_as_f32_bits(bits[c] & 0x80000000u,
                    (magnitude & 0x7fffffu) | 0x800000u, i32(magnitude >> 23u) - 127);
            }
        }
        textureStore(reduced, vec2<i32>(id.xy), bitcast<vec4<f32>>(out_bits));
        return;
    }
    var sum: array<vec2<u32>, 4>;
    for (var y = begin.y; y < end.y; y += 1u) {
        for (var x = begin.x; x < end.x; x += 1u) {
            let bits = bitcast<vec4<u32>>(textureLoad(original, vec2<i32>(i32(x), i32(y)), 0));
            for (var c = 0u; c < 4u; c += 1u) {
                sum[c] = insert_value(sum[c], bits[c], p.low[c]);
            }
        }
    }
    var out_bits = vec4(0u);
    for (var c = 0u; c < 4u; c += 1u) {
        out_bits[c] = mean_half_bits(sum[c], n, p.low[c]);
    }
    textureStore(reduced, vec2<i32>(id.xy), bitcast<vec4<f32>>(out_bits));
}

// Area mean for sources OUTSIDE the checked fixed-point domain (typical float
// renders: a wide exponent spread across the frame, values beyond half range,
// non-finite samples). The same exact 64-bit fixed-point sum as `main`, but the
// window is chosen PER CELL: its lowest bit `low` sits 53 bits below the top of
// the cell's largest sample (incl. the log2(n) carry bits), so the sum never
// overflows and `mean_half_bits` keeps its <=53-bit contract. Sample bits below
// `low` are truncated: at most n * 2^low in total, i.e. <= 2^-36 of the cell's
// largest sample (n <= 256), far below what a half can show unless the cell
// cancels to ~1e-11 of its largest value. Float arithmetic is NOT used for the
// sum: a compiler may reassociate it (a compensated f32 sum came back as 0 for
// 1e6 + 0.01 - 1e6 on a real driver). Non-finite samples are classified by bits
// (WGSL lets a compiler fold x != x) and decide the result like f64 addition.
// f32 subnormals are skipped: they lie far below half's smallest subnormal.
const F32_INF: u32 = 0x7f800000u;
const F32_NAN: u32 = 0x7fc00000u;
@compute @workgroup_size(8, 8)
fn main_wide(@builtin(global_invocation_id) id: vec3<u32>) {
    let begin = id.xy * p.scale;
    if any(begin >= p.size) { return; }
    let end = min(begin + vec2(p.scale), p.size);
    // n <= 256: the direct path holds <= 8192 px and the top level fits one
    // 512 px tile, so the coarsest cell is 16 x 16 (`divide64_small` needs it).
    let n = (end.x - begin.x) * (end.y - begin.y);
    let carry = i32(32u - countLeadingZeros(n - 1u));
    // Pass 1: each channel's largest binary exponent and its non-finite samples.
    var high = vec4<i32>(-1000);
    var pos_inf = vec4<bool>(false);
    var neg_inf = vec4<bool>(false);
    var nan = vec4<bool>(false);
    for (var y = begin.y; y < end.y; y += 1u) {
        for (var x = begin.x; x < end.x; x += 1u) {
            let bits = bitcast<vec4<u32>>(textureLoad(original, vec2<i32>(i32(x), i32(y)), 0));
            for (var c = 0u; c < 4u; c += 1u) {
                let magnitude = bits[c] & 0x7fffffffu;
                if magnitude > F32_INF { nan[c] = true; continue; }
                if magnitude == F32_INF {
                    if (bits[c] & 0x80000000u) != 0u { neg_inf[c] = true; } else { pos_inf[c] = true; }
                    continue;
                }
                if (magnitude >> 23u) == 0u { continue; }
                high[c] = max(high[c], i32(magnitude >> 23u) - 127);
            }
        }
    }
    // The window: top bit of an n-sample sum at high + 1 + carry, 53 bits wide.
    let low = high + vec4<i32>(1 + carry - 53);
    // Pass 2: the exact (truncated below `low`) signed sum of the finite normals.
    var sum: array<vec2<u32>, 4>;
    for (var y = begin.y; y < end.y; y += 1u) {
        for (var x = begin.x; x < end.x; x += 1u) {
            let bits = bitcast<vec4<u32>>(textureLoad(original, vec2<i32>(i32(x), i32(y)), 0));
            for (var c = 0u; c < 4u; c += 1u) {
                let exponent = (bits[c] >> 23u) & 255u;
                if exponent == 0u || exponent == 255u { continue; }
                sum[c] = insert_value(sum[c], bits[c], low[c]);
            }
        }
    }
    var out_bits = vec4(0u);
    for (var c = 0u; c < 4u; c += 1u) {
        if nan[c] || (pos_inf[c] && neg_inf[c]) {
            out_bits[c] = F32_NAN;
        } else if pos_inf[c] {
            out_bits[c] = F32_INF;
        } else if neg_inf[c] {
            out_bits[c] = 0x80000000u | F32_INF;
        } else if high[c] > -1000 {
            out_bits[c] = mean_half_bits(sum[c], n, low[c]);
        }
    }
    textureStore(reduced, vec2<i32>(id.xy), bitcast<vec4<f32>>(out_bits));
}
