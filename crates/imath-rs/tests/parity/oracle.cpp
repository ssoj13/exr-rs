// C++ Imath golden-parity oracle for exr-rs.
//
// Computes reference values for representative ops across every Imath module
// using the *real* C++ Imath library (vcpkg, Imath 3.2.2) and prints them as
// machine-readable `KEY=VALUE` lines on stdout. The companion Rust test
// `crates/imath-rs/tests/parity_golden.rs` recomputes the SAME ops with
// `imath-rs` and asserts bit-exact / tight-ULP agreement.
//
// Output discipline (so values are EXACT and parseable):
//   * floats  -> printed with "%a" (C99 hex-float; cast to double first), which
//                round-trips the exact IEEE value with no decimal-rounding loss.
//   * half/u16 bits, u32 -> printed as "%04x" / "%08x".
//   * ints    -> printed with "%d" / "%lld".
//
// The Rust side parses hex-floats via the same exact representation, so a
// "%a" double parsed back to f64 (then narrowed to f32 where the op is f32)
// reproduces the bit pattern the C++ computation produced.
//
// Build (driven by `cargo run -p xtask -- parity`, MSVC env already applied):
//   cl /nologo /EHsc /std:c++17 /I <vcpkg>/include oracle.cpp /Fe:oracle.exe \
//      /link /LIBPATH:<vcpkg>/lib Imath-3_2.lib
//
// NEVER edit imath-rs/src to make parity pass; a real mismatch is a port bug.

#include <Imath/half.h>
#include <Imath/ImathVec.h>
#include <Imath/ImathMatrix.h>
#include <Imath/ImathMatrixAlgo.h>
#include <Imath/ImathQuat.h>
#include <Imath/ImathEuler.h>
#include <Imath/ImathColor.h>
#include <Imath/ImathColorAlgo.h>
#include <Imath/ImathBox.h>
#include <Imath/ImathBoxAlgo.h>
#include <Imath/ImathLine.h>
#include <Imath/ImathPlane.h>
#include <Imath/ImathSphere.h>
#include <Imath/ImathFrustum.h>
#include <Imath/ImathRandom.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace IMATH_INTERNAL_NAMESPACE;

// ---------------------------------------------------------------------------
// Print helpers — every numeric value is emitted in an EXACT, parseable form.
// ---------------------------------------------------------------------------

// Exact hex-float for an f32 value, tagged with its raw bits for cross-check.
static void pf(const char* key, float v)
{
    // "%a" on a promoted double is the canonical lossless text form; the trailing
    // "bits=" lets the Rust side double-check the f32 bit pattern directly.
    uint32_t fb;
    std::memcpy(&fb, &v, sizeof(fb));
    printf("%s=%a bits=%08x\n", key, (double) v, fb);
}

// Exact hex-float for an f64 value, tagged with its raw bits.
static void pd(const char* key, double v)
{
    uint64_t b;
    std::memcpy(&b, &v, sizeof(b));
    printf("%s=%a bits=%016llx\n", key, v, (unsigned long long) b);
}

static void pu16(const char* key, uint16_t v) { printf("%s=%04x\n", key, v); }
static void pu64(const char* key, uint64_t v)
{
    printf("%s=%016llx\n", key, (unsigned long long) v);
}
static void pu32(const char* key, uint32_t v) { printf("%s=%08x\n", key, v); }
static void pi64(const char* key, long long v) { printf("%s=%lld\n", key, v); }

// ---------------------------------------------------------------------------
// half
// ---------------------------------------------------------------------------

static void emit_half()
{
    // ~30 fixed u16 bit patterns -> float bits. Covers zero, denormals, the
    // smallest/largest normals, +/-1, +/-inf, qNaN/sNaN, and assorted values.
    const uint16_t hbits[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00,
        0x3555, 0x4000, 0xc000, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00,
        0x7c01, 0xfe00, 0x3800, 0xb800, 0x4900, 0x4248, 0x5640, 0x6400,
        0x0200, 0x8200, 0x1234, 0x9abc, 0x5bff, 0x7800,
    };
    for (size_t i = 0; i < sizeof(hbits) / sizeof(hbits[0]); ++i)
    {
        half h;
        h.setBits(hbits[i]);
        char k[64];
        snprintf(k, sizeof(k), "half.u16_to_f32.%04x", hbits[i]);
        pf(k, (float) h);
    }

    // ~30 fixed f32 values -> half bits (round-to-nearest-even path).
    const float fvals[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 3.1415927f,
        65504.0f, -65504.0f, 65520.0f, 1e-08f, 6.1035156e-05f, 5.9604645e-08f,
        100000.0f, -100000.0f, 0.333333343f, 1234.5f, -0.0001f, 12345.0f,
        0.099609375f, 7.0f, 0.000061988831f, 2.9802322e-08f, 0.6666667f,
        1024.5f, -2048.25f, 1e+30f, -1e+30f, 4.7683716e-07f,
    };
    for (size_t i = 0; i < sizeof(fvals) / sizeof(fvals[0]); ++i)
    {
        half h(fvals[i]);
        char k[64];
        uint32_t fb;
        std::memcpy(&fb, &fvals[i], sizeof(fb));
        snprintf(k, sizeof(k), "half.f32_to_u16.%08x", fb);
        pu16(k, h.bits());
    }

    // EXHAUSTIVE half->float: all 65536 bit patterns, folded into one FNV-1a-64
    // hash of the little-endian f32 bit patterns (a 65536-line golden would be
    // unreviewable; the explicit samples above stay for triage). The hash is
    // recomputed identically on the Rust side, so ANY differing conversion --
    // NaN payload included -- changes it.
    {
        uint64_t hsum = 0xcbf29ce484222325ULL;
        for (uint32_t i = 0; i < 65536u; ++i)
        {
            half h;
            h.setBits((uint16_t) i);
            float f = (float) h;
            uint32_t fb;
            std::memcpy(&fb, &f, sizeof(fb));
            for (int b = 0; b < 4; ++b)
            {
                hsum ^= (uint8_t) ((fb >> (8 * b)) & 0xFFu);
                hsum *= 0x100000001b3ULL;
            }
        }
        pu64("half.u16_to_f32.all.fnv1a64", hsum);
    }

    // EXHAUSTIVE float->half over the 65536 exactly-representable halves, same
    // hash scheme over the resulting u16 bit patterns (little-endian).
    {
        uint64_t hsum = 0xcbf29ce484222325ULL;
        for (uint32_t i = 0; i < 65536u; ++i)
        {
            half h;
            h.setBits((uint16_t) i);
            half back((float) h);
            uint16_t hb = back.bits();
            hsum ^= (uint8_t) (hb & 0xFFu);
            hsum *= 0x100000001b3ULL;
            hsum ^= (uint8_t) ((hb >> 8) & 0xFFu);
            hsum *= 0x100000001b3ULL;
        }
        pu64("half.f32_to_u16.roundtrip.fnv1a64", hsum);
    }

    // NaN / infinity payloads for float->half. The port claims to be
    // bit-exact "including NaN payloads"; these are the C++ values backing it.
    {
        const uint32_t nanbits[] = {
            0x7f800000u, 0xff800000u, // +/-inf
            0x7fc00000u, 0xffc00000u, // quiet NaN (default)
            0x7f800001u, 0xff800001u, // signalling NaN, smallest payload
            0x7fbfffffu, 0xffbfffffu, // signalling NaN, largest payload
            0x7fc00001u, 0x7fffffffu, // quiet NaN payloads
            0x7f802000u, 0x7f801000u, // payloads that survive the >>13 shift
            0x7f7fffffu, 0xff7fffffu, // largest finite float (overflows to inf)
        };
        for (size_t i = 0; i < sizeof(nanbits) / sizeof(nanbits[0]); ++i)
        {
            float f;
            std::memcpy(&f, &nanbits[i], sizeof(f));
            half h(f);
            char k[64];
            snprintf(k, sizeof(k), "half.f32_to_u16.%08x", nanbits[i]);
            pu16(k, h.bits());
        }
    }

    // NaN / infinity payloads for half->float.
    {
        const uint16_t nanh[] = {0x7c01, 0x7dff, 0x7e00, 0x7fff, 0xfc01, 0xfe00, 0xffff};
        for (size_t i = 0; i < sizeof(nanh) / sizeof(nanh[0]); ++i)
        {
            half h;
            h.setBits(nanh[i]);
            char k[64];
            snprintf(k, sizeof(k), "half.u16_to_f32.%04x", nanh[i]);
            pf(k, (float) h);
        }
    }

    // Classification flags for a few representative bit patterns.
    const uint16_t cbits[] = {0x0000, 0x0001, 0x3c00, 0x7c00, 0xfc00, 0x7e00, 0xbc00};
    for (size_t i = 0; i < sizeof(cbits) / sizeof(cbits[0]); ++i)
    {
        half h;
        h.setBits(cbits[i]);
        unsigned flags = 0;
        flags |= (h.isFinite() ? 1u : 0u) << 0;
        flags |= (h.isNormalized() ? 1u : 0u) << 1;
        flags |= (h.isDenormalized() ? 1u : 0u) << 2;
        flags |= (h.isZero() ? 1u : 0u) << 3;
        flags |= (h.isNan() ? 1u : 0u) << 4;
        flags |= (h.isInfinity() ? 1u : 0u) << 5;
        flags |= (h.isNegative() ? 1u : 0u) << 6;
        char k[64];
        snprintf(k, sizeof(k), "half.flags.%04x", cbits[i]);
        pu32(k, flags);
    }
}

// ---------------------------------------------------------------------------
// Vec3 / Vec4 (f32 & f64)
// ---------------------------------------------------------------------------

static void emit_vec()
{
    {
        V3f a(1.5f, -2.25f, 3.75f), b(-0.5f, 4.0f, 2.0f);
        pf("v3f.dot", a.dot(b));
        V3f c = a.cross(b);
        pf("v3f.cross.x", c.x); pf("v3f.cross.y", c.y); pf("v3f.cross.z", c.z);
        pf("v3f.length2", a.length2());
        pf("v3f.length", a.length());
        V3f n = a.normalized();
        pf("v3f.norm.x", n.x); pf("v3f.norm.y", n.y); pf("v3f.norm.z", n.z);
    }
    {
        V3d a(1.5, -2.25, 3.75), b(-0.5, 4.0, 2.0);
        pd("v3d.dot", a.dot(b));
        V3d c = a.cross(b);
        pd("v3d.cross.x", c.x); pd("v3d.cross.y", c.y); pd("v3d.cross.z", c.z);
        pd("v3d.length2", a.length2());
        pd("v3d.length", a.length());
        V3d n = a.normalized();
        pd("v3d.norm.x", n.x); pd("v3d.norm.y", n.y); pd("v3d.norm.z", n.z);
    }
    {
        V4f a(1.0f, 2.0f, -3.0f, 0.5f), b(2.0f, -1.0f, 0.25f, 4.0f);
        pf("v4f.dot", a.dot(b));
        pf("v4f.length2", a.length2());
        pf("v4f.length", a.length());
        V4f n = a.normalized();
        pf("v4f.norm.x", n.x); pf("v4f.norm.y", n.y);
        pf("v4f.norm.z", n.z); pf("v4f.norm.w", n.w);
    }
    {
        V4d a(1.0, 2.0, -3.0, 0.5), b(2.0, -1.0, 0.25, 4.0);
        pd("v4d.dot", a.dot(b));
        pd("v4d.length2", a.length2());
        pd("v4d.length", a.length());
        V4d n = a.normalized();
        pd("v4d.norm.x", n.x); pd("v4d.norm.y", n.y);
        pd("v4d.norm.z", n.z); pd("v4d.norm.w", n.w);
    }
}

// ---------------------------------------------------------------------------
// Matrix33 / Matrix44 — all exact (multiply, det, inverse, transpose, multVec/Dir)
// ---------------------------------------------------------------------------

static void emit_matrix()
{
    // Fixed, well-conditioned matrices with exact binary fractions.
    M33f a3(2.0f, -1.0f, 0.5f, 0.25f, 3.0f, -2.0f, 1.0f, 0.5f, 4.0f);
    M33f b3(1.0f, 2.0f, -0.5f, -1.0f, 0.25f, 3.0f, 0.5f, -2.0f, 1.0f);
    M33f p3 = a3 * b3;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            char k[48]; snprintf(k, sizeof(k), "m33f.mul.%d%d", i, j);
            pf(k, p3[i][j]);
        }
    pf("m33f.det", a3.determinant());
    M33f inv3 = a3.inverse();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            char k[48]; snprintf(k, sizeof(k), "m33f.inv.%d%d", i, j);
            pf(k, inv3[i][j]);
        }
    M33f t3 = a3.transposed();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            char k[48]; snprintf(k, sizeof(k), "m33f.transpose.%d%d", i, j);
            pf(k, t3[i][j]);
        }
    {
        V2f src(1.5f, -2.0f), dst;
        a3.multVecMatrix(src, dst);
        pf("m33f.multVec.x", dst.x); pf("m33f.multVec.y", dst.y);
        a3.multDirMatrix(src, dst);
        pf("m33f.multDir.x", dst.x); pf("m33f.multDir.y", dst.y);
    }

    // Matrix44 (double) for the higher-precision path.
    M44d a4(2.0, -1.0, 0.5, 0.25,
            0.25, 3.0, -2.0, 1.0,
            1.0, 0.5, 4.0, -0.5,
            -0.5, 0.25, 1.0, 2.0);
    M44d b4(1.0, 2.0, -0.5, 0.0,
            -1.0, 0.25, 3.0, 0.5,
            0.5, -2.0, 1.0, -1.0,
            0.25, 0.5, -0.25, 1.0);
    M44d p4 = a4 * b4;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            char k[48]; snprintf(k, sizeof(k), "m44d.mul.%d%d", i, j);
            pd(k, p4[i][j]);
        }
    pd("m44d.det", a4.determinant());
    M44d inv4 = a4.inverse();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            char k[48]; snprintf(k, sizeof(k), "m44d.inv.%d%d", i, j);
            pd(k, inv4[i][j]);
        }
    M44d t4 = a4.transposed();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            char k[48]; snprintf(k, sizeof(k), "m44d.transpose.%d%d", i, j);
            pd(k, t4[i][j]);
        }
    {
        V3d src(1.5, -2.0, 3.0), dst;
        a4.multVecMatrix(src, dst);
        pd("m44d.multVec.x", dst.x); pd("m44d.multVec.y", dst.y); pd("m44d.multVec.z", dst.z);
        a4.multDirMatrix(src, dst);
        pd("m44d.multDir.x", dst.x); pd("m44d.multDir.y", dst.y); pd("m44d.multDir.z", dst.z);
    }
}

// ---------------------------------------------------------------------------
// matrix_algo: extractSHRT, jacobiEigenSolver, procrustes
// ---------------------------------------------------------------------------

static void emit_matrix_algo()
{
    // extractSHRT on a fixed affine M44d built from literal element values so the
    // INPUT is bit-identical on both sides (no trig-built matrix -> no ULP at the
    // input). This matrix encodes a scale, a small shear, a rotation, and a
    // translation. The extraction itself uses transcendentals, so the Rust side
    // compares these with a tight ULP tolerance.
    M44d m(0.96 * 2.0,   0.10,        -0.20,       0.0,
           -0.15,        0.97 * 1.5,   0.05,       0.0,
           0.20,        -0.05,         0.98 * 0.5, 0.0,
           5.0,          -3.0,         2.0,        1.0);
    V3d s, h, r, t;
    bool ok = extractSHRT(m, s, h, r, t);
    pi64("shrt.ok", ok ? 1 : 0);
    pd("shrt.s.x", s.x); pd("shrt.s.y", s.y); pd("shrt.s.z", s.z);
    pd("shrt.h.x", h.x); pd("shrt.h.y", h.y); pd("shrt.h.z", h.z);
    pd("shrt.r.x", r.x); pd("shrt.r.y", r.y); pd("shrt.r.z", r.z);
    pd("shrt.t.x", t.x); pd("shrt.t.y", t.y); pd("shrt.t.z", t.z);

    // extractSHRT(Euler) on a SINGULAR matrix with exc=false. Row 0 is all zero,
    // so extractAndRemoveScalingAndShear fails at
    // checkForZeroScaleInRow(scl.x=0, row[0]) (ImathMatrixAlgo.h:587-588) and
    // extractSHRT returns false BEFORE touching `r` (:840). The pre-seeded Euler
    // angles must therefore come back unchanged.
    {
        M44d sing (0.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 1.0);
        V3d  s2, h2, t2;
        Eulerd r2 (V3d (0.25, -0.5, 0.75), Eulerd::XYZ);
        bool ok2 = extractSHRT (sing, s2, h2, r2, t2, false);
        pi64("shrt.fail.ok", ok2 ? 1 : 0);
        pd("shrt.fail.r.x", r2.x); pd("shrt.fail.r.y", r2.y); pd("shrt.fail.r.z", r2.z);
    }

    // jacobiEigenSolver eigenvalues of a fixed symmetric M33d.
    {
        M33d A(4.0, 1.0, 2.0,
               1.0, 5.0, 3.0,
               2.0, 3.0, 6.0);
        V3d S;
        M33d Vv;
        jacobiEigenSolver(A, S, Vv);
        pd("jacobi33.s.x", S.x); pd("jacobi33.s.y", S.y); pd("jacobi33.s.z", S.z);
    }
    // jacobiEigenSolver eigenvalues of a fixed symmetric M44d.
    {
        M44d A(4.0, 1.0, 2.0, 0.5,
               1.0, 5.0, 3.0, 1.0,
               2.0, 3.0, 6.0, 1.5,
               0.5, 1.0, 1.5, 7.0);
        V4d S;
        M44d Vv;
        jacobiEigenSolver(A, S, Vv);
        pd("jacobi44.s.x", S.x); pd("jacobi44.s.y", S.y);
        pd("jacobi44.s.z", S.z); pd("jacobi44.s.w", S.w);
    }

    // procrustesRotationAndTranslation on a fixed point set (rigid, no scale).
    {
        V3d from[4] = {
            V3d(0.0, 0.0, 0.0), V3d(1.0, 0.0, 0.0),
            V3d(0.0, 1.0, 0.0), V3d(0.0, 0.0, 1.0),
        };
        V3d to[4] = {
            V3d(1.0, 2.0, 3.0), V3d(1.0, 2.70710678118654752, 3.70710678118654752),
            V3d(2.0, 2.0, 3.0), V3d(1.0, 2.70710678118654752, 2.29289321881345248),
        };
        M44d res = procrustesRotationAndTranslation(from, to, (size_t) 4);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                char k[48]; snprintf(k, sizeof(k), "procrustes.%d%d", i, j);
                pd(k, res[i][j]);
            }
    }
}

// ---------------------------------------------------------------------------
// Quat
// ---------------------------------------------------------------------------

static void emit_quat()
{
    // setAxisAngle -> toMatrix44.
    {
        Quatd q;
        V3d axis(0.0, 0.0, 1.0);
        axis.normalize();
        q.setAxisAngle(axis, 0.7853981633974483); // 45 deg
        M44d m = q.toMatrix44();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                char k[48]; snprintf(k, sizeof(k), "quat.axisangle.m%d%d", i, j);
                pd(k, m[i][j]);
            }
    }
    // slerp at t = 0.25 / 0.5 / 0.75 between two unit quats.
    {
        Quatd q1; q1.setAxisAngle(V3d(0.0, 0.0, 1.0), 0.0);
        Quatd q2; q2.setAxisAngle(V3d(0.0, 0.0, 1.0), 1.5707963267948966); // 90 deg
        const double ts[3] = {0.25, 0.5, 0.75};
        for (int i = 0; i < 3; ++i)
        {
            Quatd q = slerp(q1, q2, ts[i]);
            char k[48];
            snprintf(k, sizeof(k), "quat.slerp.%d.r", i); pd(k, q.r);
            snprintf(k, sizeof(k), "quat.slerp.%d.x", i); pd(k, q.v.x);
            snprintf(k, sizeof(k), "quat.slerp.%d.y", i); pd(k, q.v.y);
            snprintf(k, sizeof(k), "quat.slerp.%d.z", i); pd(k, q.v.z);
        }
    }
    // setRotation(from, to).
    {
        Quatd q;
        V3d from(1.0, 0.0, 0.0), to(0.0, 1.0, 0.0);
        q.setRotation(from, to);
        pd("quat.setrot.r", q.r);
        pd("quat.setrot.x", q.v.x); pd("quat.setrot.y", q.v.y); pd("quat.setrot.z", q.v.z);
    }
}

// ---------------------------------------------------------------------------
// Euler
// ---------------------------------------------------------------------------

static void emit_euler()
{
    struct { const char* name; Eulerd::Order order; } cases[] = {
        {"XYZ", Eulerd::XYZ},
        {"ZYX", Eulerd::ZYX},
        {"XZX", Eulerd::XZX},
    };
    const double ang[3] = {0.3, -0.5, 0.8};
    for (int c = 0; c < 3; ++c)
    {
        Eulerd e(ang[0], ang[1], ang[2], cases[c].order);
        M44d m = e.toMatrix44();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                char k[64];
                snprintf(k, sizeof(k), "euler.%s.m%d%d", cases[c].name, i, j);
                pd(k, m[i][j]);
            }
        // Extract back from the matrix.
        Eulerd ex(cases[c].order);
        ex.extract(m);
        char k[64];
        snprintf(k, sizeof(k), "euler.%s.extract.x", cases[c].name); pd(k, ex.x);
        snprintf(k, sizeof(k), "euler.%s.extract.y", cases[c].name); pd(k, ex.y);
        snprintf(k, sizeof(k), "euler.%s.extract.z", cases[c].name); pd(k, ex.z);
    }
}

// ---------------------------------------------------------------------------
// Color: hsv2rgb / rgb2hsv (no trig -> bit-exact), rgb2packed / packed2rgb
// ---------------------------------------------------------------------------

static void emit_color()
{
    {
        V3f hsv(0.3333333f, 0.5f, 0.75f); // hue ~120 deg
        V3f rgb = hsv2rgb(hsv);
        pf("color.hsv2rgb.r", rgb.x); pf("color.hsv2rgb.g", rgb.y); pf("color.hsv2rgb.b", rgb.z);
        V3f hsv2 = rgb2hsv(rgb);
        pf("color.rgb2hsv.h", hsv2.x); pf("color.rgb2hsv.s", hsv2.y); pf("color.rgb2hsv.v", hsv2.z);
    }
    {
        V3f rgb(0.25f, 0.5f, 0.75f);
        PackedColor p = rgb2packed(rgb);
        pu32("color.rgb2packed", (uint32_t) p);
        V3f back;
        packed2rgb(p, back);
        pf("color.packed2rgb.r", back.x); pf("color.packed2rgb.g", back.y); pf("color.packed2rgb.b", back.z);
    }
    {
        Color4f c(0.1f, 0.4f, 0.7f, 1.0f);
        PackedColor p = rgb2packed(c);
        pu32("color.rgb2packed.c4", (uint32_t) p);
    }
    // packed2rgb INTEGER branch (ImathColorAlgo.h:216-232 / :240-256):
    //   T f = numeric_limits<T>::max() / (PackedColor) 0xFF;
    // The divide happens in T, so for int64_t f is ~3.6e16, not ~1.7e7. The
    // Rust port used to compute f in u32 and was wrong by a factor of ~2^31.
    {
        const PackedColor packs[] = {0x00000000u, 0xFFFFFFFFu, 0x12345678u, 0x0080FF7Fu};
        for (size_t i = 0; i < sizeof(packs) / sizeof(packs[0]); ++i)
        {
            char k[80];
            Vec3<unsigned char> vu8;   packed2rgb(packs[i], vu8);
            Vec3<short>         vi16;  packed2rgb(packs[i], vi16);
            Vec3<int>           vi32;  packed2rgb(packs[i], vi32);
            Vec3<long long>     vi64;  packed2rgb(packs[i], vi64);
            snprintf(k, sizeof(k), "color.packed2rgb.u8.%08x.x", packs[i]);  pi64(k, (long long) vu8.x);
            snprintf(k, sizeof(k), "color.packed2rgb.u8.%08x.y", packs[i]);  pi64(k, (long long) vu8.y);
            snprintf(k, sizeof(k), "color.packed2rgb.u8.%08x.z", packs[i]);  pi64(k, (long long) vu8.z);
            snprintf(k, sizeof(k), "color.packed2rgb.i16.%08x.x", packs[i]); pi64(k, (long long) vi16.x);
            snprintf(k, sizeof(k), "color.packed2rgb.i16.%08x.y", packs[i]); pi64(k, (long long) vi16.y);
            snprintf(k, sizeof(k), "color.packed2rgb.i16.%08x.z", packs[i]); pi64(k, (long long) vi16.z);
            snprintf(k, sizeof(k), "color.packed2rgb.i32.%08x.x", packs[i]); pi64(k, (long long) vi32.x);
            snprintf(k, sizeof(k), "color.packed2rgb.i32.%08x.y", packs[i]); pi64(k, (long long) vi32.y);
            snprintf(k, sizeof(k), "color.packed2rgb.i32.%08x.z", packs[i]); pi64(k, (long long) vi32.z);
            snprintf(k, sizeof(k), "color.packed2rgb.i64.%08x.x", packs[i]); pi64(k, (long long) vi64.x);
            snprintf(k, sizeof(k), "color.packed2rgb.i64.%08x.y", packs[i]); pi64(k, (long long) vi64.y);
            snprintf(k, sizeof(k), "color.packed2rgb.i64.%08x.z", packs[i]); pi64(k, (long long) vi64.z);

            Color4<long long> ci64; packed2rgb(packs[i], ci64);
            snprintf(k, sizeof(k), "color.packed2rgb.c4i64.%08x.r", packs[i]); pi64(k, (long long) ci64.r);
            snprintf(k, sizeof(k), "color.packed2rgb.c4i64.%08x.g", packs[i]); pi64(k, (long long) ci64.g);
            snprintf(k, sizeof(k), "color.packed2rgb.c4i64.%08x.b", packs[i]); pi64(k, (long long) ci64.b);
            snprintf(k, sizeof(k), "color.packed2rgb.c4i64.%08x.a", packs[i]); pi64(k, (long long) ci64.a);
        }
    }
}

// ---------------------------------------------------------------------------
// Box / Line / Plane / Sphere / Frustum
// ---------------------------------------------------------------------------

static void emit_geometry()
{
    // Box transform by a fixed (literal) affine matrix.
    {
        Box3f b(V3f(-1.0f, -2.0f, -0.5f), V3f(2.0f, 1.0f, 3.0f));
        M44f m(1.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 2.0f, 0.0f, 0.0f,
               0.0f, 0.0f, 0.5f, 0.0f,
               1.0f, -1.0f, 2.0f, 1.0f);
        Box3f r = transform(b, m);
        pf("box.transform.min.x", r.min.x); pf("box.transform.min.y", r.min.y); pf("box.transform.min.z", r.min.z);
        pf("box.transform.max.x", r.max.x); pf("box.transform.max.y", r.max.y); pf("box.transform.max.z", r.max.z);
    }
    // Line closestPointTo(point).
    {
        Line3f l(V3f(0.0f, 0.0f, 0.0f), V3f(1.0f, 1.0f, 0.0f));
        V3f cp = l.closestPointTo(V3f(1.0f, 0.0f, 0.0f));
        pf("line.closest.x", cp.x); pf("line.closest.y", cp.y); pf("line.closest.z", cp.z);
    }
    // Plane intersect(line).
    {
        Plane3f pl(V3f(0.0f, 0.0f, 1.0f), 2.0f); // z = 2 plane
        Line3f l(V3f(0.0f, 0.0f, 0.0f), V3f(0.5f, 0.25f, 1.0f));
        V3f ip;
        bool hit = pl.intersect(l, ip);
        pi64("plane.intersect.hit", hit ? 1 : 0);
        pf("plane.intersect.x", ip.x); pf("plane.intersect.y", ip.y); pf("plane.intersect.z", ip.z);
    }
    // Sphere intersect(line).
    {
        Sphere3f sp(V3f(0.0f, 0.0f, 5.0f), 2.0f);
        Line3f l(V3f(0.0f, 0.0f, 0.0f), V3f(0.0f, 0.0f, 1.0f));
        V3f ip;
        bool hit = sp.intersect(l, ip);
        pi64("sphere.intersect.hit", hit ? 1 : 0);
        pf("sphere.intersect.x", ip.x); pf("sphere.intersect.y", ip.y); pf("sphere.intersect.z", ip.z);
    }
    // Frustum projectionMatrix + planes(M). Built from literal edges (no fov/trig)
    // so the projection matrix is bit-exact.
    {
        Frustumf f(1.0f, 100.0f, -0.5f, 0.5f, 0.4f, -0.4f, false);
        M44f pm = f.projectionMatrix();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                char k[48]; snprintf(k, sizeof(k), "frustum.proj.%d%d", i, j);
                pf(k, pm[i][j]);
            }
        // planes() in world space (identity transform passed via a literal matrix).
        Plane3f planes[6];
        M44f id; // identity
        f.planes(planes, id);
        for (int p = 0; p < 6; ++p)
        {
            char k[48];
            snprintf(k, sizeof(k), "frustum.plane%d.n.x", p); pf(k, planes[p].normal.x);
            snprintf(k, sizeof(k), "frustum.plane%d.n.y", p); pf(k, planes[p].normal.y);
            snprintf(k, sizeof(k), "frustum.plane%d.n.z", p); pf(k, planes[p].normal.z);
            snprintf(k, sizeof(k), "frustum.plane%d.d", p);   pf(k, planes[p].distance);
        }
    }
    // Frustum ZToDepth / DepthToZ over REAL depth-buffer ranges. The Rust port
    // used to build T from a long by adding 1 in a loop (2^31 iterations, and
    // wrong above 2^24 for float), so these had no golden and the bug survived.
    //
    //   ZToDepth : ImathFrustum.h:757-767   (fzval computed in T)
    //   DepthToZ : ImathFrustum.h:871-893   (long(0.5 * (Zp + 1) * zdiff) + zmin)
    //
    // NOTE: `long` is 32 bits with MSVC, and `zmax + 1` / `0.5*(Zp+1)*zdiff`
    // must stay in range or C++ is UB, so every case below is inside int range
    // and inside [zmin, zmax].
    {
        Frustumf fp(1.0f, 100.0f, -0.5f, 0.5f, 0.4f, -0.4f, false);
        Frustumf fo(1.0f, 100.0f, -0.5f, 0.5f, 0.4f, -0.4f, true);
        Frustumd dp(1.0, 1000.0, -1.0, 1.0, 1.0, -1.0, false);

        struct { const char* name; long zmin; long zmax; } ranges[] = {
            {"u16",  0L,     65535L},
            {"u24",  0L,     16777215L},
            {"i31",  0L,     2000000000L},
            {"sym", -1000L,  1000L},
        };
        const double frac[] = {0.0, 0.25, 0.5, 0.75, 1.0};

        for (int rg = 0; rg < 4; ++rg)
        {
            long zmin = ranges[rg].zmin, zmax = ranges[rg].zmax;
            for (int i = 0; i < 5; ++i)
            {
                long zval = zmin + (long) (frac[i] * (double) (zmax - zmin));
                char k[96];
                snprintf(k, sizeof(k), "frustum.ztodepth.persp.%s.%d", ranges[rg].name, i);
                pf(k, fp.ZToDepth(zval, zmin, zmax));
                snprintf(k, sizeof(k), "frustum.ztodepth.ortho.%s.%d", ranges[rg].name, i);
                pf(k, fo.ZToDepth(zval, zmin, zmax));
                snprintf(k, sizeof(k), "frustum.ztodepth.d.%s.%d", ranges[rg].name, i);
                pd(k, dp.ZToDepth(zval, zmin, zmax));
            }
        }

        // DepthToZ: depths spanning near..far (world Z is negative here).
        const float  fdepths[] = {-1.0f, -2.0f, -10.0f, -50.0f, -100.0f};
        const double ddepths[] = {-1.0,  -2.0,  -10.0,  -50.0,  -1000.0};
        for (int rg = 0; rg < 4; ++rg)
        {
            long zmin = ranges[rg].zmin, zmax = ranges[rg].zmax;
            for (int i = 0; i < 5; ++i)
            {
                char k[96];
                snprintf(k, sizeof(k), "frustum.depthtoz.persp.%s.%d", ranges[rg].name, i);
                pi64(k, (long long) fp.DepthToZ(fdepths[i], zmin, zmax));
                snprintf(k, sizeof(k), "frustum.depthtoz.ortho.%s.%d", ranges[rg].name, i);
                pi64(k, (long long) fo.DepthToZ(fdepths[i], zmin, zmax));
                snprintf(k, sizeof(k), "frustum.depthtoz.d.%s.%d", ranges[rg].name, i);
                pi64(k, (long long) dp.DepthToZ(ddepths[i], zmin, zmax));
            }
        }
    }

    // Frustum fov values (depend on atan -> tight-ULP on the Rust side).
    {
        Frustumf f(1.0f, 100.0f, -0.5f, 0.5f, 0.4f, -0.4f, false);
        pf("frustum.fovx", f.fovx());
        pf("frustum.fovy", f.fovy());
    }
}

// ---------------------------------------------------------------------------
// Random: Rand32 & Rand48 — first 8 nexti() and nextf() for a fixed seed.
// ---------------------------------------------------------------------------

static void emit_random()
{
    {
        Rand32 r(12345u);
        for (int i = 0; i < 8; ++i)
        {
            char k[48]; snprintf(k, sizeof(k), "rand32.nexti.%d", i);
            pu32(k, (uint32_t) r.nexti());
        }
        Rand32 rf(12345u);
        for (int i = 0; i < 8; ++i)
        {
            char k[48]; snprintf(k, sizeof(k), "rand32.nextf.%d", i);
            pf(k, rf.nextf());
        }
    }
    {
        Rand48 r(12345u);
        for (int i = 0; i < 8; ++i)
        {
            char k[48]; snprintf(k, sizeof(k), "rand48.nexti.%d", i);
            pi64(k, (long long) r.nexti());
        }
        Rand48 rf(12345u);
        for (int i = 0; i < 8; ++i)
        {
            char k[48]; snprintf(k, sizeof(k), "rand48.nextf.%d", i);
            pd(k, rf.nextf());
        }
    }
}

int main()
{
    printf("# exr-rs Imath golden parity oracle (C++ Imath %d.%d.%d)\n",
           IMATH_VERSION_MAJOR, IMATH_VERSION_MINOR, IMATH_VERSION_PATCH);
    emit_half();
    emit_vec();
    emit_matrix();
    emit_matrix_algo();
    emit_quat();
    emit_euler();
    emit_color();
    emit_geometry();
    emit_random();
    return 0;
}
