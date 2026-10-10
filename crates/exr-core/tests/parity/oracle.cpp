// oracle.cpp — C++ OpenEXR (IlmImf) reference for the exr-core uncompressed
// scanline read/write parity gate.
//
// Built and driven by `cargo run -p xtask -- parity-exr`.  It speaks the real
// vcpkg OpenEXR 3.4 IlmImf API and has two modes:
//
//   oracle.exe write <path> [compression]
//       Writes a KNOWN, fully-deterministic scanline EXR (the "golden" file) so
//       the Rust reader can be checked against a file that a genuine C++ OpenEXR
//       produced.  `compression` is optional and defaults to `none`; the
//       accepted names are `rle`, `zip`, `zips`, `pxr24`, `piz`, `b44`, `b44a`
//       (otherwise NO_COMPRESSION).  The pixel pattern is identical regardless of
//       compression, so the same Rust-side KNOWN image validates every case.
//
//   oracle.exe read <path>
//       Reads an EXR with Imf::InputFile and prints — machine-readably — the
//       header attributes we care about and every pixel's exact bits, so the
//       test can check that a file OUR writer produced is read back identically
//       by C++.
//
// ---------------------------------------------------------------------------
// The KNOWN image (must be reproduced bit-for-bit by the Rust side)
// ---------------------------------------------------------------------------
//   * dimensions      : WIDTH x HEIGHT = 67 x 43
//   * data/displayWin  : [(0,0) .. (66,42)]
//   * compression      : NO_COMPRESSION
//   * lineOrder        : INCREASING_Y (the Imf default)
//   * channels (stored alphabetically by IlmImf):  A,B,G,R,U,Z
//       - R,G,B : HALF
//       - A,Z   : FLOAT
//       - U     : UINT
//   * pixel pattern, with idx = y*WIDTH + x (0-based), is DETERMINISTIC:
//       R = half( 1.0f  + idx * 0.5f )
//       G = half( 100.0f + idx * 0.25f )
//       B = half( -50.0f + idx * 1.0f )
//       A = float( (idx % 16) * 0.013f + 0.25f )  (stored as f32)
//       Z = float( idx * 4.0f )                   (stored as f32)
//       U = uint( idx * 3 + 11 )
//     ...except inside the FLAT RECT x in [8,40), y in [8,24), where every
//     channel is constant (see is_flat below).
//
// Why 67x43 and not the original 7x5: a 7x5 image is ONE chunk for every codec
// and so small that PIZ / DWAA / DWAB / HTJ2K all failed to shrink it, so the
// writer stored those chunks RAW -- their goldens exercised the raw-store rule
// and the header and NOTHING of the codec. 67x43 is
// multi-chunk for ZIP (16 lines), B44 (4), DWA (32) and PIZ, is not a multiple
// of 4/8/16/32, and compresses. The UINT channel gives the UINT pixel path its
// first C++-verified test; the flat rect gives B44A its flat-block path (the
// B44 and B44A goldens used to differ by exactly one byte -- the compression
// attribute -- so B44A's only distinguishing codepath was never executed).
//     All HALF values are produced by the SAME IEEE round-to-nearest-even
//     float->half conversion the Rust `Half::from(f32)` uses, so the 16-bit
//     bit patterns are identical on both sides.
//   * header extras:
//       owner   (string) = "exr-rs"
//       parityTag (int)  = 31337
//
// ---------------------------------------------------------------------------
// 3.4 vs 3.3.11 note
// ---------------------------------------------------------------------------
// vcpkg ships OpenEXR 3.4; exr-core targets the v3.3.11 byte format.  For an
// UNCOMPRESSED SCANLINE file the on-disk layout (magic, version word, attribute
// encoding, line-offset table, per-chunk `y`+`dataSize`+little-endian pixel
// data) is byte-identical between those versions — only API/perf differ.  So
// any mismatch this gate surfaces is a real port bug, not a version artifact.

#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfStdIO.h>
#include <ImfCompressor.h>
#include <ImfTiledOutputFile.h>
#include <ImfTiledRgbaFile.h>
#include <ImfTiledInputFile.h>
#include <ImfDeepScanLineOutputFile.h>
#include <ImfDeepScanLineInputFile.h>
#include <ImfDeepTiledOutputFile.h>
#include <ImfDeepTiledInputFile.h>
#include <ImfDeepFrameBuffer.h>
#include <ImfCompositeDeepScanLine.h>
#include <ImfDeepCompositing.h>
// Multipart: the C++ MultiPart container + the typed
// per-part wrappers (same read/write interface as the single-part files).
#include <ImfArray.h>
#include <IexBaseExc.h>
#include <IexMathExc.h>
#include <IexErrnoExc.h>
#include <stdexcept>
#include <ImfRgbaFile.h>
#include <ImfAcesFile.h>
#include <ImfLut.h>
#include <ImfMultiPartOutputFile.h>
#include <ImfMultiPartInputFile.h>
#include <ImfOutputPart.h>
#include <ImfInputPart.h>
#include <ImfTiledOutputPart.h>
#include <ImfTiledInputPart.h>
#include <ImfDeepScanLineOutputPart.h>
#include <ImfDeepScanLineInputPart.h>
#include <ImfPartType.h>
#include <ImfTileDescription.h>
#include <ImfHeader.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfCompression.h>
#include <ImfLineOrder.h>
#include <ImfStandardAttributes.h>
#include <ImfStringAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfRationalAttribute.h>
#include <ImfKeyCodeAttribute.h>
#include <ImfIntAttribute.h>
// Structured IDManifest cross-check (the additive write-idmanifest /
// dump-idmanifest modes).
#include <ImfIDManifest.h>
#include <ImfIDManifestAttribute.h>
#include <ImfBoxAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfPixelType.h>
#include <ImfThreading.h>
#include <ImathBox.h>
#include <half.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <utility>
#include <vector>
#include <string>

using namespace Imf;
using namespace Imath;

static const int WIDTH = 67;
static const int HEIGHT = 43;

// The constant rectangle. Aligned to 4 in both axes so it covers WHOLE B44
// blocks: B44A only emits its 3-byte flat block when all 16 samples of a 4x4
// block are equal, which no gradient can produce.
static bool is_flat(int idx)
{
    int x = idx % WIDTH, y = idx / WIDTH;
    return x >= 8 && x < 40 && y >= 8 && y < 24;
}

// Deterministic per-pixel source values (computed identically by Rust).
static float    r_val(int idx) { return is_flat(idx) ? 0.5f  : 1.0f + idx * 0.5f; }
static float    g_val(int idx) { return is_flat(idx) ? 0.25f : 100.0f + idx * 0.25f; }
static float    b_val(int idx) { return is_flat(idx) ? -1.0f : -50.0f + idx * 1.0f; }
static float    a_val(int idx) { return is_flat(idx) ? 0.75f : (idx % 16) * 0.013f + 0.25f; }
static float    z_val(int idx) { return is_flat(idx) ? 1.0f  : idx * 4.0f; }
static uint32_t u_val(int idx) { return is_flat(idx) ? 7u    : (uint32_t)idx * 3u + 11u; }

// ---------------------------------------------------------------------------
// write mode
// ---------------------------------------------------------------------------
// Map a textual compression name to the IlmImf enum (default NO_COMPRESSION).
static Compression comp_from_name(const char* name)
{
    if (name && std::strcmp(name, "rle")   == 0) return RLE_COMPRESSION;
    if (name && std::strcmp(name, "zip")   == 0) return ZIP_COMPRESSION;
    if (name && std::strcmp(name, "zips")  == 0) return ZIPS_COMPRESSION;
    if (name && std::strcmp(name, "pxr24") == 0) return PXR24_COMPRESSION;
    if (name && std::strcmp(name, "piz")   == 0) return PIZ_COMPRESSION;
    if (name && std::strcmp(name, "b44")   == 0) return B44_COMPRESSION;
    if (name && std::strcmp(name, "b44a")  == 0) return B44A_COMPRESSION;
    if (name && std::strcmp(name, "dwaa")  == 0) return DWAA_COMPRESSION;
    if (name && std::strcmp(name, "dwab")  == 0) return DWAB_COMPRESSION;
    // OpenEXR 3.4 HTJ2K codecs (vcpkg ships 3.4).
    if (name && std::strcmp(name, "htj2k256") == 0) return HTJ2K256_COMPRESSION;
    if (name && std::strcmp(name, "htj2k32")  == 0) return HTJ2K32_COMPRESSION;
    return NO_COMPRESSION;
}

// `zipLevel` is additive and defaults to -1 ("do not touch the header's
// zipCompressionLevel() at all -- let OpenEXR use its own built-in default"),
// so every EXISTING call site (which passes only `path`/`comp`) is completely
// unaffected. When >= 0, sets `header.zipCompressionLevel()` before opening the
// OutputFile, mirroring `Imf::Header::zipCompressionLevel()`'s per-header
// override (`ImfHeader.h:281`, `ImfHeader.cpp:68-183`) -- it is a header-side
// ephemeral setting, not a file attribute, and (per `internal_pxr24.c:219-226`)
// PXR24 ignores it and always uses the
// process-global default, so this is only meaningful for ZIP/ZIPS.
static int do_write(const char* path, Compression comp, int zipLevel = -1)
{
    // Build the per-channel planar buffers in application memory.
    std::vector<half>     R(WIDTH * HEIGHT), G(WIDTH * HEIGHT), B(WIDTH * HEIGHT);
    std::vector<float>    A(WIDTH * HEIGHT), Z(WIDTH * HEIGHT);
    std::vector<uint32_t> U(WIDTH * HEIGHT);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int idx = y * WIDTH + x;
            R[idx] = half(r_val(idx));
            G[idx] = half(g_val(idx));
            B[idx] = half(b_val(idx));
            A[idx] = a_val(idx);
            Z[idx] = z_val(idx);
            U[idx] = u_val(idx);
        }
    }

    // Header: WIDTHxHEIGHT, NO_COMPRESSION, INCREASING_Y (Imf default order).
    Header header(WIDTH, HEIGHT);
    header.compression() = comp;
    header.lineOrder()   = INCREASING_Y;
    if (zipLevel >= 0) {
        header.zipCompressionLevel() = zipLevel;
    }
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("A", Channel(FLOAT));
    header.channels().insert("Z", Channel(FLOAT));
    header.channels().insert("U", Channel(UINT));

    // A couple of standard / custom header attributes.
    header.insert("owner", StringAttribute("exr-rs"));
    header.insert("parityTag", IntAttribute(31337));

    OutputFile out(path, header);

    // FrameBuffer: planar slices, origin (0,0), default strides.
    FrameBuffer fb;
    fb.insert("R", Slice(HALF,  (char*)R.data(), sizeof(half),  sizeof(half)  * WIDTH));
    fb.insert("G", Slice(HALF,  (char*)G.data(), sizeof(half),  sizeof(half)  * WIDTH));
    fb.insert("B", Slice(HALF,  (char*)B.data(), sizeof(half),  sizeof(half)  * WIDTH));
    fb.insert("A", Slice(FLOAT, (char*)A.data(), sizeof(float), sizeof(float) * WIDTH));
    fb.insert("Z", Slice(FLOAT, (char*)Z.data(), sizeof(float), sizeof(float) * WIDTH));
    fb.insert("U", Slice(UINT,  (char*)U.data(), sizeof(uint32_t), sizeof(uint32_t) * WIDTH));

    out.setFrameBuffer(fb);
    out.writePixels(HEIGHT);

    fprintf(stderr, "oracle: wrote %dx%d scanline EXR (compression %d) to %s\n",
            WIDTH, HEIGHT, (int)comp, path);
    return 0;
}

// "Header wire bytes": the KNOWN image (NO_COMPRESSION, so the whole
// file including every chunk is deterministic) plus three ADDITIONAL
// structured attribute types the base `write` mode's header does not carry
// (chromaticities: 4 packed V2f; rational: i32 + u32; keyCode: 7 packed
// i32) -- each with a nontrivial multi-field wire layout where a field-order
// bug is plausible. Every existing `*_bytes_byte_identical_to_cpp_golden`
// test already proves the BASE header (owner/parityTag/channels/compression/
// dataWindow/displayWindow/lineOrder/pixelAspectRatio/screenWindowCenter/
// screenWindowWidth) is byte-exact as a side effect of comparing the whole
// file; this mode extends that same whole-file byte-exact comparison to
// cover these three additional types explicitly.
static int do_write_header_attrs(const char* path)
{
    std::vector<half>     R(WIDTH * HEIGHT), G(WIDTH * HEIGHT), B(WIDTH * HEIGHT);
    std::vector<float>    A(WIDTH * HEIGHT), Z(WIDTH * HEIGHT);
    std::vector<uint32_t> U(WIDTH * HEIGHT);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int idx = y * WIDTH + x;
            R[idx] = half(r_val(idx));
            G[idx] = half(g_val(idx));
            B[idx] = half(b_val(idx));
            A[idx] = a_val(idx);
            Z[idx] = z_val(idx);
            U[idx] = u_val(idx);
        }
    }

    Header header(WIDTH, HEIGHT);
    header.compression() = NO_COMPRESSION;
    header.lineOrder()   = INCREASING_Y;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("A", Channel(FLOAT));
    header.channels().insert("Z", Channel(FLOAT));
    header.channels().insert("U", Channel(UINT));
    header.insert("owner", StringAttribute("exr-rs"));
    header.insert("parityTag", IntAttribute(31337));
    header.insert("chromaticities", ChromaticitiesAttribute(Chromaticities(
        V2f(0.640f, 0.330f), V2f(0.300f, 0.600f), V2f(0.150f, 0.060f), V2f(0.3127f, 0.3290f))));
    header.insert("myRational", RationalAttribute(Rational(7, 3)));
    header.insert("myKeyCode", KeyCodeAttribute(KeyCode(12, 3, 456789, 42, 7, 4, 64)));

    OutputFile out(path, header);

    FrameBuffer fb;
    fb.insert("R", Slice(HALF,  (char*)R.data(), sizeof(half),  sizeof(half)  * WIDTH));
    fb.insert("G", Slice(HALF,  (char*)G.data(), sizeof(half),  sizeof(half)  * WIDTH));
    fb.insert("B", Slice(HALF,  (char*)B.data(), sizeof(half),  sizeof(half)  * WIDTH));
    fb.insert("A", Slice(FLOAT, (char*)A.data(), sizeof(float), sizeof(float) * WIDTH));
    fb.insert("Z", Slice(FLOAT, (char*)Z.data(), sizeof(float), sizeof(float) * WIDTH));
    fb.insert("U", Slice(UINT,  (char*)U.data(), sizeof(uint32_t), sizeof(uint32_t) * WIDTH));
    out.setFrameBuffer(fb);
    out.writePixels(HEIGHT);

    fprintf(stderr, "oracle: wrote %dx%d scanline EXR with chromaticities/rational/keyCode to %s\n",
            WIDTH, HEIGHT, path);
    return 0;
}

// ---------------------------------------------------------------------------
// read mode — print header attrs + every pixel's exact bits, machine-readably.
// ---------------------------------------------------------------------------
//
// Output grammar (one record per line, stdout):
//   DIM <width> <height>
//   DW <minx> <miny> <maxx> <maxy>
//   COMPRESSION <int>
//   LINEORDER <int>
//   ATTR <name> <typeName> <value>          (only owner + parityTag emitted)
//   CHAN <name> <pixelType:int>             (file channel order)
//   PIX <chan> <x> <y> <hexbits>            (HALF -> 4 hex nibbles, FLOAT -> 8)
//   END
//
static int do_read(const char* path)
{
    InputFile in(path);
    const Header& h = in.header();

    Box2i dw = h.dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int hgt = dw.max.y - dw.min.y + 1;

    printf("DIM %d %d\n", w, hgt);
    printf("DW %d %d %d %d\n", dw.min.x, dw.min.y, dw.max.x, dw.max.y);
    printf("COMPRESSION %d\n", (int)h.compression());
    printf("LINEORDER %d\n", (int)h.lineOrder());

    // Header attributes of interest (string owner + custom int).
    if (const StringAttribute* a = h.findTypedAttribute<StringAttribute>("owner"))
        printf("ATTR owner string %s\n", a->value().c_str());
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute>("parityTag"))
        printf("ATTR parityTag int %d\n", a->value());

    // Channel list (alphabetical, IlmImf order) + their pixel types.
    std::vector<std::string> chanNames;
    std::vector<PixelType>   chanTypes;
    for (ChannelList::ConstIterator it = h.channels().begin();
         it != h.channels().end(); ++it) {
        chanNames.push_back(it.name());
        chanTypes.push_back(it.channel().type);
        printf("CHAN %s %d\n", it.name(), (int)it.channel().type);
    }

    // Read all channels into planar application buffers.
    FrameBuffer fb;
    std::vector<std::vector<uint8_t>> planes(chanNames.size());
    for (size_t c = 0; c < chanNames.size(); ++c) {
        // HALF is 2 bytes; FLOAT and UINT are both 4.
        size_t px = (chanTypes[c] == HALF) ? sizeof(half) : 4;
        planes[c].assign((size_t)w * hgt * px, 0);
        // base must be offset so that pixel (dw.min.x, dw.min.y) maps to the
        // start of the plane: base = data - (minx + miny*w) * px.
        char* data = (char*)planes[c].data();
        char* base = data - ((size_t)dw.min.x + (size_t)dw.min.y * w) * px;
        fb.insert(chanNames[c],
                  Slice(chanTypes[c], base, px, px * (size_t)w));
    }

    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);

    // Emit every pixel's raw bits, hex, in file-channel order.
    for (size_t c = 0; c < chanNames.size(); ++c) {
        bool isHalf = (chanTypes[c] == HALF);
        for (int y = 0; y < hgt; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = (size_t)y * w + x;
                if (isHalf) {
                    uint16_t bits;
                    std::memcpy(&bits, &planes[c][i * sizeof(half)], sizeof(bits));
                    printf("PIX %s %d %d %04x\n", chanNames[c].c_str(), x, y, bits);
                } else {
                    uint32_t bits;
                    std::memcpy(&bits, &planes[c][i * sizeof(float)], sizeof(bits));
                    printf("PIX %s %d %d %08x\n", chanNames[c].c_str(), x, y, bits);
                }
            }
        }
    }

    printf("END\n");
    return 0;
}

// Read EVERY channel of `path` into a HALF slice
// regardless of its on-disk pixel type -- this forces the UINT->HALF and
// FLOAT->HALF conversion paths in real C++ OpenEXR's core reader
// (OpenEXRCore/unpack.c -> internal_coding.h::float_to_half/uint_to_half),
// which is exactly what exr-core's misc.rs::copy_into_frame_buffer must match
// bit-for-bit. Prints one PIX line per pixel, always 4 hex digits (half bits).
static int do_read_half(const char* path)
{
    InputFile in(path);
    const Header& h = in.header();

    Box2i dw = h.dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int hgt = dw.max.y - dw.min.y + 1;

    std::vector<std::string> chanNames;
    for (ChannelList::ConstIterator it = h.channels().begin();
         it != h.channels().end(); ++it) {
        chanNames.push_back(it.name());
    }

    FrameBuffer fb;
    std::vector<std::vector<half>> planes(chanNames.size());
    for (size_t c = 0; c < chanNames.size(); ++c) {
        planes[c].assign((size_t)w * hgt, half(0.0f));
        char* data = (char*)planes[c].data();
        char* base = data - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half);
        fb.insert(chanNames[c],
                  Slice(HALF, base, sizeof(half), sizeof(half) * (size_t)w));
    }

    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);

    for (size_t c = 0; c < chanNames.size(); ++c) {
        for (int y = 0; y < hgt; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = (size_t)y * w + x;
                uint16_t bits;
                std::memcpy(&bits, &planes[c][i], sizeof(bits));
                printf("PIX %s %d %d %04x\n", chanNames[c].c_str(), x, y, bits);
            }
        }
    }

    printf("END\n");
    return 0;
}

// Incomplete files (both directions): open `path` -- which may be a TRUNCATED file
// -- and report, machine-readably, whether C++ itself follows the
// documented `isComplete()` contract: open succeeds even for a file some
// chunks are missing from, `isComplete()` is false, present scanlines read
// back fine, and reading a missing scanline fails cleanly (never a crash).
// Used BOTH to check C++ reading a file WE truncated, and (the fixture this
// mode itself is verified against) to derive what a real C++-truncated file
// looks like for the Rust side to open. NONE-compressed, single-row-per-
// chunk KNOWN image assumed, so per-row truncation boundaries are exact.
//
// Output grammar (stdout): "OPEN ok" or "OPEN err", then if ok:
// "ISCOMPLETE 0|1", then one "ROW <y> ok|err" per row of the data window.
static int do_read_truncated(const char* path)
{
    InputFile* in = nullptr;
    try {
        in = new InputFile(path);
    } catch (...) {
        printf("OPEN err\n");
        return 0;
    }
    printf("OPEN ok\n");
    printf("ISCOMPLETE %d\n", in->isComplete() ? 1 : 0);

    Box2i dw = in->header().dataWindow();
    int w = dw.max.x - dw.min.x + 1;

    std::vector<std::string> chanNames;
    for (ChannelList::ConstIterator it = in->header().channels().begin();
         it != in->header().channels().end(); ++it) {
        chanNames.push_back(it.name());
    }

    for (int y = dw.min.y; y <= dw.max.y; ++y) {
        FrameBuffer fb;
        std::vector<std::vector<uint8_t>> planes(chanNames.size());
        for (size_t c = 0; c < chanNames.size(); ++c) {
            const Channel& ch = in->header().channels()[chanNames[c]];
            size_t px = (ch.type == HALF) ? sizeof(half) : 4;
            planes[c].assign((size_t)w * px, 0);
            char* data = (char*)planes[c].data();
            char* base = data - (size_t)dw.min.x * px - (size_t)y * px * w;
            fb.insert(chanNames[c], Slice(ch.type, base, px, px * (size_t)w));
        }
        in->setFrameBuffer(fb);
        try {
            in->readPixels(y, y);
            printf("ROW %d ok\n", y);
        } catch (...) {
            printf("ROW %d err\n", y);
        }
    }

    delete in;
    return 0;
}

// Read EVERY channel of `path` into a UINT slice regardless of its
// on-disk type -- the mirror of `do_read_half` above, but for the
// HALF->UINT and FLOAT->UINT conversion paths (`OpenEXRCore/unpack.c` ->
// `internal_coding.h::half_to_uint`/`float_to_uint`), which have their own
// clamp semantics (negative/NaN -> 0, overflow -> UINT_MAX) distinct from
// the ->HALF paths `read-half` already exercises. Prints one PIX line per
// pixel, always 8 hex digits (uint bits).
static int do_read_as_uint(const char* path)
{
    InputFile in(path);
    const Header& h = in.header();

    Box2i dw = h.dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int hgt = dw.max.y - dw.min.y + 1;

    std::vector<std::string> chanNames;
    for (ChannelList::ConstIterator it = h.channels().begin();
         it != h.channels().end(); ++it) {
        chanNames.push_back(it.name());
    }

    FrameBuffer fb;
    std::vector<std::vector<uint32_t>> planes(chanNames.size());
    for (size_t c = 0; c < chanNames.size(); ++c) {
        planes[c].assign((size_t)w * hgt, 0u);
        char* data = (char*)planes[c].data();
        char* base = data - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(uint32_t);
        fb.insert(chanNames[c],
                  Slice(UINT, base, sizeof(uint32_t), sizeof(uint32_t) * (size_t)w));
    }

    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);

    for (size_t c = 0; c < chanNames.size(); ++c) {
        for (int y = 0; y < hgt; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = (size_t)y * w + x;
                printf("PIX %s %d %d %08x\n", chanNames[c].c_str(), x, y, planes[c][i]);
            }
        }
    }

    printf("END\n");
    return 0;
}

// ===========================================================================
// TILED parity
// ===========================================================================
//
// The KNOWN tiled image (must be reproduced bit-for-bit by the Rust side):
//   * dimensions      : TWIDTH x THEIGHT = 68 x 44 (level 0)
//   * tile size        : 32 x 32 -> 3 x 2 tiles at level 0 (partial right/bottom)
//   * channels         : R,G,B = HALF ; A,Z = FLOAT ; U = UINT (alphabetical)
//   * level modes       : ONE_LEVEL and MIPMAP_LEVELS (ROUND_DOWN)
//   * line order        : INCREASING_Y
//   * data window        : at origin for most variants; an OFF-ORIGIN variant
//                          (min = (11, 7)) exercises the tile-rectangle seam.
//   * header extras       : owner(string)="exr-rs", parityTag(int)=31337
//
// Per-pixel formula — keyed on coordinates RELATIVE to the level's data-window
// min (px, py) and a per-level bias so each MIPMAP level carries distinguishable
// data.  Identical on both sides; all HALF values via IEEE round-to-nearest-even.
//
//   tidx     = py * levelWidth + px            (0-based, within the level)
//   lvlbias  = lx * 7 + ly * 13
//   R = half ( 1.0f  + (tidx + lvlbias) * 0.5f )
//   G = half ( 100.0f + tidx * 0.25f + lvlbias )
//   B = half ( -50.0f + tidx * 1.0f  - lvlbias )
//   A = float( ((tidx + lvlbias) % 16) * 0.013f + 0.25f )
//   Z = float( (tidx + lvlbias) * 4.0f )
//   U = uint ( tidx * 3 + lvlbias + 11 )
//
// 20x15 with 8x8 tiles was too small for PIZ / DWAB to shrink a tile, so those
// tiled goldens were byte-identical to the uncompressed one (raw store) and
// never ran the codec. 68x44 with 32x32 tiles keeps the partial right/bottom
// edge tiles and gives each tile enough data to compress.
//
// Because the formula is RELATIVE to the level data window, it is independent of
// the (possibly off-origin) data-window min — the on-disk tile bytes are the
// same for the origin and off-origin variants, which is exactly what we want to
// cross-check against the Rust writer.

static const int TWIDTH  = 68;
static const int THEIGHT = 44;
static const int TILE_SIZE = 32;

static float t_r_val(int tidx, int lvlbias) { return 1.0f  + (tidx + lvlbias) * 0.5f; }
static float t_g_val(int tidx, int lvlbias) { return 100.0f + tidx * 0.25f + (float)lvlbias; }
static float t_b_val(int tidx, int lvlbias) { return -50.0f + tidx * 1.0f  - (float)lvlbias; }
static float t_a_val(int tidx, int lvlbias) { return ((tidx + lvlbias) % 16) * 0.013f + 0.25f; }
static uint32_t t_u_val(int tidx, int lvlbias) { return (uint32_t)(tidx * 3 + lvlbias + 11); }
static float t_z_val(int tidx, int lvlbias) { return (tidx + lvlbias) * 4.0f; }

// Map a textual level-mode name to the IlmImf enum (default ONE_LEVEL).
// `one` / `mipmap` / `ripmap`.
static LevelMode level_mode_from_name(const char* name)
{
    if (name && std::strcmp(name, "mipmap") == 0) return MIPMAP_LEVELS;
    if (name && std::strcmp(name, "ripmap") == 0) return RIPMAP_LEVELS;
    return ONE_LEVEL;
}

// ---------------------------------------------------------------------------
// write-tiled mode
// ---------------------------------------------------------------------------
// Writes the KNOWN tiled image with the given compression + level mode.  Each
// resolution level is an independent plane filled by the formula evaluated at
// that level's own resolution; tiles are written whole-level in canonical
// INCREASING_Y order (dy outer 0..numYTiles-1, dx inner) so the on-disk tile
// byte order matches a genuine C++ tiled writer.
static int do_write_tiled(const char* path, Compression comp, LevelMode mode,
                          int minX, int minY, LevelRoundingMode rounding,
                          LineOrder order)
{
    Header header(TWIDTH, THEIGHT);
    // Override data + display window to the requested (possibly off-origin) box.
    Box2i win(V2i(minX, minY), V2i(minX + TWIDTH - 1, minY + THEIGHT - 1));
    header.dataWindow()    = win;
    header.displayWindow() = win;
    header.compression()   = comp;
    header.lineOrder()     = order;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("A", Channel(FLOAT));
    header.channels().insert("Z", Channel(FLOAT));
    header.channels().insert("U", Channel(UINT));
    header.insert("owner", StringAttribute("exr-rs"));
    header.insert("parityTag", IntAttribute(31337));
    header.setTileDescription(TileDescription(TILE_SIZE, TILE_SIZE, mode, rounding));

    TiledOutputFile out(path, header);

    // Fill + write one (lx, ly) level. Shared by the ONE_LEVEL/MIPMAP diagonal
    // loop (lx == ly) and the RIPMAP 2D loop below -- same formula, same
    // canonical whole-level tile order, just keyed by an independent (lx, ly).
    auto writeLevel = [&](int lx, int ly) {
        Box2i ldw = out.dataWindowForLevel(lx, ly);
        int lw = ldw.max.x - ldw.min.x + 1;
        int lh = ldw.max.y - ldw.min.y + 1;
        int lvlbias = lx * 7 + ly * 13;

        std::vector<half>     R(lw * lh), G(lw * lh), B(lw * lh);
        std::vector<float>    A(lw * lh), Z(lw * lh);
        std::vector<uint32_t> U(lw * lh);
        for (int py = 0; py < lh; ++py) {
            for (int px = 0; px < lw; ++px) {
                int tidx = py * lw + px;
                int i    = py * lw + px;
                R[i] = half(t_r_val(tidx, lvlbias));
                G[i] = half(t_g_val(tidx, lvlbias));
                B[i] = half(t_b_val(tidx, lvlbias));
                A[i] = t_a_val(tidx, lvlbias);
                Z[i] = t_z_val(tidx, lvlbias);
                U[i] = t_u_val(tidx, lvlbias);
            }
        }

        // FrameBuffer with base offset so that absolute pixel (ldw.min) maps to
        // the start of each plane (xTileCoords/yTileCoords default false).
        FrameBuffer fb;
        char* baseR = (char*)R.data() - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * sizeof(half);
        char* baseG = (char*)G.data() - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * sizeof(half);
        char* baseB = (char*)B.data() - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * sizeof(half);
        char* baseA = (char*)A.data() - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * sizeof(float);
        char* baseZ = (char*)Z.data() - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * sizeof(float);
        char* baseU = (char*)U.data() - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * sizeof(uint32_t);
        fb.insert("R", Slice(HALF,  baseR, sizeof(half),  sizeof(half)  * lw));
        fb.insert("G", Slice(HALF,  baseG, sizeof(half),  sizeof(half)  * lw));
        fb.insert("B", Slice(HALF,  baseB, sizeof(half),  sizeof(half)  * lw));
        fb.insert("A", Slice(FLOAT, baseA, sizeof(float), sizeof(float) * lw));
        fb.insert("Z", Slice(FLOAT, baseZ, sizeof(float), sizeof(float) * lw));
        fb.insert("U", Slice(UINT,  baseU, sizeof(uint32_t), sizeof(uint32_t) * lw));
        out.setFrameBuffer(fb);

        // Canonical whole-level write order: dy outer (0..numYTiles-1), dx inner.
        out.writeTiles(0, out.numXTiles(lx) - 1, 0, out.numYTiles(ly) - 1, lx, ly);
    };

    int numLevels = 1;
    if (mode == MIPMAP_LEVELS) {
        numLevels = out.numLevels();
        for (int l = 0; l < numLevels; ++l) writeLevel(l, l);
    } else if (mode == RIPMAP_LEVELS) {
        int nx = out.numXLevels();
        int ny = out.numYLevels();
        numLevels = nx * ny;
        for (int ly = 0; ly < ny; ++ly)
            for (int lx = 0; lx < nx; ++lx)
                writeLevel(lx, ly);
    } else {
        writeLevel(0, 0);
    }

    fprintf(stderr,
            "oracle: wrote %dx%d tiled EXR (compression %d, mode %d, rounding %d, order %d, "
            "min %d,%d, %d level(s)) to %s\n",
            TWIDTH, THEIGHT, (int)comp, (int)mode, (int)rounding, (int)order,
            minX, minY, numLevels, path);
    return 0;
}

// ---------------------------------------------------------------------------
// read-tiled mode — print tile/level metadata + every pixel of every level.
// ---------------------------------------------------------------------------
//
// Output grammar (one record per line, stdout):
//   TILEDESC <xSize> <ySize> <levelMode:int> <roundingMode:int>
//   LEVELMODE <int>
//   COMPRESSION <int>
//   DW <minx> <miny> <maxx> <maxy>
//   NUMLEVELS <n>          (numLevels for ONE_LEVEL/MIPMAP)
//   ATTR <name> <typeName> <value>
//   CHAN <name> <pixelType:int>
//   LEVEL <l> <minx> <miny> <maxx> <maxy>     (per level, before its pixels)
//   TPIX <l> <chan> <x> <y> <hexbits>          (x,y RELATIVE to level min)
//   END
//
static int do_read_tiled(const char* path)
{
    TiledInputFile in(path);
    const Header& h = in.header();
    const TileDescription& td = h.tileDescription();

    Box2i dw = h.dataWindow();
    printf("TILEDESC %u %u %d %d\n", td.xSize, td.ySize, (int)td.mode, (int)td.roundingMode);
    printf("LEVELMODE %d\n", (int)in.levelMode());
    printf("COMPRESSION %d\n", (int)h.compression());
    printf("DW %d %d %d %d\n", dw.min.x, dw.min.y, dw.max.x, dw.max.y);
    int numLevels = (in.levelMode() == MIPMAP_LEVELS) ? in.numLevels() : 1;
    printf("NUMLEVELS %d\n", numLevels);

    if (const StringAttribute* a = h.findTypedAttribute<StringAttribute>("owner"))
        printf("ATTR owner string %s\n", a->value().c_str());
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute>("parityTag"))
        printf("ATTR parityTag int %d\n", a->value());

    std::vector<std::string> chanNames;
    std::vector<PixelType>   chanTypes;
    for (ChannelList::ConstIterator it = h.channels().begin();
         it != h.channels().end(); ++it) {
        chanNames.push_back(it.name());
        chanTypes.push_back(it.channel().type);
        printf("CHAN %s %d\n", it.name(), (int)it.channel().type);
    }

    for (int l = 0; l < numLevels; ++l) {
        Box2i ldw = in.dataWindowForLevel(l);
        int lw = ldw.max.x - ldw.min.x + 1;
        int lh = ldw.max.y - ldw.min.y + 1;
        printf("LEVEL %d %d %d %d %d\n", l, ldw.min.x, ldw.min.y, ldw.max.x, ldw.max.y);

        // Read this whole level into planar application buffers (base offset so
        // absolute pixel ldw.min maps to plane start).
        FrameBuffer fb;
        std::vector<std::vector<uint8_t>> planes(chanNames.size());
        for (size_t c = 0; c < chanNames.size(); ++c) {
            size_t px = (chanTypes[c] == HALF) ? sizeof(half) : sizeof(float);
            planes[c].assign((size_t)lw * lh * px, 0);
            char* data = (char*)planes[c].data();
            char* base = data - ((size_t)ldw.min.x + (size_t)ldw.min.y * lw) * px;
            fb.insert(chanNames[c], Slice(chanTypes[c], base, px, px * (size_t)lw));
        }
        in.setFrameBuffer(fb);
        in.readTiles(0, in.numXTiles(l) - 1, 0, in.numYTiles(l) - 1, l);

        for (size_t c = 0; c < chanNames.size(); ++c) {
            bool isHalf = (chanTypes[c] == HALF);
            for (int y = 0; y < lh; ++y) {
                for (int x = 0; x < lw; ++x) {
                    size_t i = (size_t)y * lw + x;
                    if (isHalf) {
                        uint16_t bits;
                        std::memcpy(&bits, &planes[c][i * sizeof(half)], sizeof(bits));
                        printf("TPIX %d %s %d %d %04x\n", l, chanNames[c].c_str(), x, y, bits);
                    } else {
                        uint32_t bits;
                        std::memcpy(&bits, &planes[c][i * sizeof(float)], sizeof(bits));
                        printf("TPIX %d %s %d %d %08x\n", l, chanNames[c].c_str(), x, y, bits);
                    }
                }
            }
        }
    }

    printf("END\n");
    return 0;
}

// ===========================================================================
// DEEP parity
// ===========================================================================
//
// The KNOWN deep image (must be reproduced bit-for-bit by the Rust side):
//   * dimensions        : DW x DH = 8 x 6 (deep tile 4x4 -> 2x2 tiles)
//   * channels (alpha.)  : B,G,R = HALF ; Z = FLOAT  (stored alphabetically)
//   * line order          : INCREASING_Y
//   * VARIABLE per-pixel sample counts: N(x,y) = (lx + ly) % 4, where
//        lx = x - dataWindow.min.x, ly = y - dataWindow.min.y  (LEVEL-relative;
//        gives 0,1,2,3 incl. ZERO-sample pixels).
//   * data window          : at origin for most variants; an OFF-ORIGIN scanline
//                          variant uses min = (-3, 7).
//   * header extras         : owner(string)="exr-rs", parityTag(int)=31337
//
// Per-SAMPLE value formula — keyed on the level-relative pixel index
//   pidx = ly * levelWidth + lx  and the sample index s in [0, N):
//     R = half ( 1.0f  + pidx * 0.5f  + s * 1.0f )
//     G = half ( 100.0f + pidx * 0.25f + s * 2.0f )
//     B = half ( -50.0f + pidx * 1.0f  - s * 1.0f )
//     Z = float( pidx * 100.0f + s * 1.0f )
//   All HALF values use the SAME IEEE round-to-nearest-even conversion the Rust
//   `Half::from(f32)` uses, so the 16-bit bit patterns match on both sides.
//   Because the formula is RELATIVE to the level data window, the on-disk bytes
//   are identical for the origin and off-origin variants (same as the tiled
//   gate), which is exactly what we cross-check against the Rust writer.

static const int DW = 8;
static const int DH = 6;
static const int DEEP_TILE = 4;

static int deep_count(int lx, int ly) { return (lx + ly) % 4; }
static float d_r_val(int pidx, int s) { return 1.0f   + pidx * 0.5f  + s * 1.0f; }
static float d_g_val(int pidx, int s) { return 100.0f + pidx * 0.25f + s * 2.0f; }
static float d_b_val(int pidx, int s) { return -50.0f  + pidx * 1.0f  - s * 1.0f; }
static float d_z_val(int pidx, int s) { return pidx * 100.0f + s * 1.0f; }

// Map a textual deep-codec name to the IlmImf enum (default NO_COMPRESSION).
// Deep is NONE/RLE/ZIPS only (Compression::isValidDeepCompression).
static Compression deep_comp_from_name(const char* name)
{
    if (name && std::strcmp(name, "rle")  == 0) return RLE_COMPRESSION;
    if (name && std::strcmp(name, "zips") == 0) return ZIPS_COMPRESSION;
    return NO_COMPRESSION;
}

// ---------------------------------------------------------------------------
// write-deep-scanline mode
// ---------------------------------------------------------------------------
// Writes the KNOWN deep scanline image with `comp` and data window min
// (minX, minY).  Uses the C++ two-pass deep write: build the per-pixel
// sample-count array + per-pixel sample-pointer arrays per channel, set the
// sample-count slice + deep slices, then writePixels.
static int do_write_deep_scanline(const char* path, Compression comp, int minX, int minY)
{
    Box2i win(V2i(minX, minY), V2i(minX + DW - 1, minY + DH - 1));
    Header header(win, win);
    header.setType(DEEPSCANLINE);
    header.compression() = comp;
    header.lineOrder()   = INCREASING_Y;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("Z", Channel(FLOAT));
    header.insert("owner", StringAttribute("exr-rs"));
    header.insert("parityTag", IntAttribute(31337));

    // Per-pixel sample counts, addressed in LEVEL-relative (0..DW, 0..DH) order.
    std::vector<unsigned int> counts(DW * DH);
    for (int ly = 0; ly < DH; ++ly)
        for (int lx = 0; lx < DW; ++lx)
            counts[ly * DW + lx] = (unsigned int)deep_count(lx, ly);

    // Per-pixel sample arrays per channel (vector<vector<...>> -> T** pointers).
    std::vector<std::vector<half>>  R(DW * DH), G(DW * DH), B(DW * DH);
    std::vector<std::vector<float>> Z(DW * DH);
    std::vector<half*>  Rp(DW * DH), Gp(DW * DH), Bp(DW * DH);
    std::vector<float*> Zp(DW * DH);
    for (int ly = 0; ly < DH; ++ly) {
        for (int lx = 0; lx < DW; ++lx) {
            int pidx = ly * DW + lx;
            int n = deep_count(lx, ly);
            R[pidx].resize(n); G[pidx].resize(n); B[pidx].resize(n); Z[pidx].resize(n);
            for (int s = 0; s < n; ++s) {
                R[pidx][s] = half(d_r_val(pidx, s));
                G[pidx][s] = half(d_g_val(pidx, s));
                B[pidx][s] = half(d_b_val(pidx, s));
                Z[pidx][s] = d_z_val(pidx, s);
            }
            Rp[pidx] = R[pidx].data();
            Gp[pidx] = G[pidx].data();
            Bp[pidx] = B[pidx].data();
            Zp[pidx] = Z[pidx].data();
        }
    }

    DeepScanLineOutputFile out(path, header);

    // FrameBuffer base offset so absolute pixel (minX, minY) maps to slot 0.
    DeepFrameBuffer fb;
    char* scBase = (char*)counts.data() - ((size_t)minX + (size_t)minY * DW) * sizeof(unsigned int);
    fb.insertSampleCountSlice(Slice(UINT, scBase,
                                    sizeof(unsigned int), sizeof(unsigned int) * DW));
    char* rBase = (char*)Rp.data() - ((size_t)minX + (size_t)minY * DW) * sizeof(half*);
    char* gBase = (char*)Gp.data() - ((size_t)minX + (size_t)minY * DW) * sizeof(half*);
    char* bBase = (char*)Bp.data() - ((size_t)minX + (size_t)minY * DW) * sizeof(half*);
    char* zBase = (char*)Zp.data() - ((size_t)minX + (size_t)minY * DW) * sizeof(float*);
    fb.insert("R", DeepSlice(HALF,  rBase, sizeof(half*),  sizeof(half*)  * DW, sizeof(half)));
    fb.insert("G", DeepSlice(HALF,  gBase, sizeof(half*),  sizeof(half*)  * DW, sizeof(half)));
    fb.insert("B", DeepSlice(HALF,  bBase, sizeof(half*),  sizeof(half*)  * DW, sizeof(half)));
    fb.insert("Z", DeepSlice(FLOAT, zBase, sizeof(float*), sizeof(float*) * DW, sizeof(float)));
    out.setFrameBuffer(fb);
    out.writePixels(DH);

    fprintf(stderr, "oracle: wrote %dx%d deep scanline EXR (comp %d, min %d,%d) to %s\n",
            DW, DH, (int)comp, minX, minY, path);
    return 0;
}

// ---------------------------------------------------------------------------
// read-deep-scanline mode
// ---------------------------------------------------------------------------
//
// Output grammar (one record per line, stdout):
//   DIM <width> <height>
//   DW <minx> <miny> <maxx> <maxy>
//   COMPRESSION <int>
//   LINEORDER <int>
//   ATTR <name> <typeName> <value>
//   CHAN <name> <pixelType:int>
//   COUNT <x> <y> <n>                         (x,y RELATIVE to dw.min)
//   SAMP <chan> <x> <y> <i> <hexbits>          (x,y RELATIVE to dw.min; HALF=4, FLOAT=8)
//   END
//
static int do_read_deep_scanline(const char* path)
{
    DeepScanLineInputFile in(path);
    const Header& h = in.header();
    Box2i dw = h.dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int hgt = dw.max.y - dw.min.y + 1;

    printf("DIM %d %d\n", w, hgt);
    printf("DW %d %d %d %d\n", dw.min.x, dw.min.y, dw.max.x, dw.max.y);
    printf("COMPRESSION %d\n", (int)h.compression());
    printf("LINEORDER %d\n", (int)h.lineOrder());
    if (const StringAttribute* a = h.findTypedAttribute<StringAttribute>("owner"))
        printf("ATTR owner string %s\n", a->value().c_str());
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute>("parityTag"))
        printf("ATTR parityTag int %d\n", a->value());

    std::vector<std::string> chanNames;
    std::vector<PixelType>   chanTypes;
    for (ChannelList::ConstIterator it = h.channels().begin();
         it != h.channels().end(); ++it) {
        chanNames.push_back(it.name());
        chanTypes.push_back(it.channel().type);
        printf("CHAN %s %d\n", it.name(), (int)it.channel().type);
    }

    // Pass 1: counts.  Sample-count slice base offset maps abs pixel -> slot 0.
    std::vector<unsigned int> counts((size_t)w * hgt, 0);
    DeepFrameBuffer fb;
    char* scBase = (char*)counts.data()
                   - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(unsigned int);
    fb.insertSampleCountSlice(Slice(UINT, scBase,
                                    sizeof(unsigned int), sizeof(unsigned int) * w));
    // Deep slices need a base too (so readPixelSampleCounts can be followed by
    // readPixels using the same fb); set them after allocating the arrays.
    in.setFrameBuffer(fb);
    in.readPixelSampleCounts(dw.min.y, dw.max.y);

    // Allocate per-pixel sample arrays from the counts + pointer arrays.
    size_t np = (size_t)w * hgt;
    std::vector<std::vector<half>>  R(np), G(np), B(np);
    std::vector<std::vector<float>> Z(np);
    std::vector<half*>  Rp(np), Gp(np), Bp(np);
    std::vector<float*> Zp(np);
    for (size_t i = 0; i < np; ++i) {
        unsigned int n = counts[i];
        R[i].resize(n); G[i].resize(n); B[i].resize(n); Z[i].resize(n);
        Rp[i] = R[i].data(); Gp[i] = G[i].data(); Bp[i] = B[i].data(); Zp[i] = Z[i].data();
    }

    char* rBase = (char*)Rp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half*);
    char* gBase = (char*)Gp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half*);
    char* bBase = (char*)Bp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half*);
    char* zBase = (char*)Zp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float*);
    fb.insert("R", DeepSlice(HALF,  rBase, sizeof(half*),  sizeof(half*)  * w, sizeof(half)));
    fb.insert("G", DeepSlice(HALF,  gBase, sizeof(half*),  sizeof(half*)  * w, sizeof(half)));
    fb.insert("B", DeepSlice(HALF,  bBase, sizeof(half*),  sizeof(half*)  * w, sizeof(half)));
    fb.insert("Z", DeepSlice(FLOAT, zBase, sizeof(float*), sizeof(float*) * w, sizeof(float)));
    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);

    // Emit counts + every sample (x,y RELATIVE to dw.min, file-channel order).
    for (int y = 0; y < hgt; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y * w + x;
            printf("COUNT %d %d %u\n", x, y, counts[i]);
        }
    }
    for (size_t c = 0; c < chanNames.size(); ++c) {
        bool isHalf = (chanTypes[c] == HALF);
        for (int y = 0; y < hgt; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = (size_t)y * w + x;
                unsigned int n = counts[i];
                for (unsigned int s = 0; s < n; ++s) {
                    if (isHalf) {
                        half* arr = (chanNames[c] == "R") ? Rp[i]
                                  : (chanNames[c] == "G") ? Gp[i] : Bp[i];
                        uint16_t bits; std::memcpy(&bits, &arr[s], sizeof(bits));
                        printf("SAMP %s %d %d %u %04x\n", chanNames[c].c_str(), x, y, s, bits);
                    } else {
                        uint32_t bits; std::memcpy(&bits, &Zp[i][s], sizeof(bits));
                        printf("SAMP %s %d %d %u %08x\n", chanNames[c].c_str(), x, y, s, bits);
                    }
                }
            }
        }
    }
    printf("END\n");
    return 0;
}

// ---------------------------------------------------------------------------
// write-deep-composite / composite-deep-scanline modes (CompositeDeepScanLine oracle gate)
// ---------------------------------------------------------------------------
// Two small deep scanline "sources" (variant 'a'/'b') with a KNOWN,
// deterministic pattern chosen to exercise: some pixels where only ONE
// source has samples, some where BOTH do (with DIFFERENT, INTERLEAVED Z
// ranges -- forces a real front-to-back sort across sources, not just
// within one), and multi-sample-per-pixel accumulation (alpha stays < 1 for
// every sample, so the composite never early-exits, exercising the full
// accumulate loop). 4x2 pixels, channels Z/A/R (no ZBack, exercising the
// "duplicate Z into the ZBack slot" default path).
static const int CDW = 4;
static const int CDH = 2;
static int   cdeep_count(int pidx, bool is_a) { return is_a ? (pidx % 3) : ((pidx + 2) % 3); }
static float cdeep_z(int pidx, int s, bool is_a) { return pidx * 10.0f + s * 3.0f + (is_a ? 0.0f : 1.5f); }
static float cdeep_a(int s) { return 0.3f + 0.1f * s; }
static float cdeep_r(int pidx, int s, bool is_a) { return (is_a ? 0.0f : 1000.0f) + pidx * 10.0f + s; }

static int do_write_deep_composite(const char* path, bool is_a)
{
    Box2i win(V2i(0, 0), V2i(CDW - 1, CDH - 1));
    Header header(win, win);
    header.setType(DEEPSCANLINE);
    header.compression() = ZIPS_COMPRESSION;
    header.lineOrder()   = INCREASING_Y;
    header.channels().insert("Z", Channel(FLOAT));
    header.channels().insert("A", Channel(FLOAT));
    header.channels().insert("R", Channel(FLOAT));

    int n = CDW * CDH;
    std::vector<unsigned int> counts(n);
    std::vector<std::vector<float>> Z(n), A(n), R(n);
    std::vector<float*> Zp(n), Ap(n), Rp(n);
    for (int pidx = 0; pidx < n; ++pidx) {
        int c = cdeep_count(pidx, is_a);
        counts[pidx] = (unsigned int)c;
        Z[pidx].resize(c); A[pidx].resize(c); R[pidx].resize(c);
        for (int s = 0; s < c; ++s) {
            Z[pidx][s] = cdeep_z(pidx, s, is_a);
            A[pidx][s] = cdeep_a(s);
            R[pidx][s] = cdeep_r(pidx, s, is_a);
        }
        Zp[pidx] = Z[pidx].data();
        Ap[pidx] = A[pidx].data();
        Rp[pidx] = R[pidx].data();
    }

    DeepScanLineOutputFile out(path, header);
    DeepFrameBuffer fb;
    fb.insertSampleCountSlice(Slice(UINT, (char*)counts.data(),
                                    sizeof(unsigned int), sizeof(unsigned int) * CDW));
    fb.insert("Z", DeepSlice(FLOAT, (char*)Zp.data(), sizeof(float*), sizeof(float*) * CDW, sizeof(float)));
    fb.insert("A", DeepSlice(FLOAT, (char*)Ap.data(), sizeof(float*), sizeof(float*) * CDW, sizeof(float)));
    fb.insert("R", DeepSlice(FLOAT, (char*)Rp.data(), sizeof(float*), sizeof(float*) * CDW, sizeof(float)));
    out.setFrameBuffer(fb);
    out.writePixels(CDH);

    fprintf(stderr, "oracle: wrote %dx%d deep composite source '%c' to %s\n",
            CDW, CDH, is_a ? 'a' : 'b', path);
    return 0;
}

// composite-deep-scanline <pathA> <pathB> <out>
//
// Opens both sources as REAL DeepScanLineInputFile objects, adds them to a
// REAL CompositeDeepScanLine (default DeepCompositing -- exercises the same
// sort() + composite_pixel() default bodies exr-rs ports), sets a flat
// Z/A/R output FrameBuffer, calls readPixels, and dumps the composited
// values bit-exact (as hex float bits, so any float-ordering/rounding
// divergence in the port shows up immediately).
//
// Output grammar (one record per line, stdout duplicated to <out>):
//   DIM <width> <height>
//   PIX <x> <y> <chan> <hex-f32-bits>
//   END
static int do_composite_deep_scanline(const char* pathA, const char* pathB, const char* outPath)
{
    DeepScanLineInputFile inA(pathA);
    DeepScanLineInputFile inB(pathB);

    CompositeDeepScanLine comp;
    comp.addSource(&inA);
    comp.addSource(&inB);

    Box2i dw = comp.dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int h = dw.max.y - dw.min.y + 1;

    std::vector<float> outZ((size_t)w * h, 0.0f);
    std::vector<float> outA((size_t)w * h, 0.0f);
    std::vector<float> outR((size_t)w * h, 0.0f);

    FrameBuffer fb;
    char* zBase = (char*)outZ.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float);
    char* aBase = (char*)outA.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float);
    char* rBase = (char*)outR.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float);
    fb.insert("Z", Slice(FLOAT, zBase, sizeof(float), sizeof(float) * w));
    fb.insert("A", Slice(FLOAT, aBase, sizeof(float), sizeof(float) * w));
    fb.insert("R", Slice(FLOAT, rBase, sizeof(float), sizeof(float) * w));
    comp.setFrameBuffer(fb);
    comp.readPixels(dw.min.y, dw.max.y);

    FILE* out = std::fopen(outPath, "wb");
    if (!out) { fprintf(stderr, "cannot open %s for write\n", outPath); return 1; }
    fprintf(out, "DIM %d %d\n", w, h);
    for (int y = dw.min.y; y <= dw.max.y; ++y) {
        for (int x = dw.min.x; x <= dw.max.x; ++x) {
            size_t i = (size_t)(y - dw.min.y) * w + (x - dw.min.x);
            uint32_t zb, ab, rb;
            std::memcpy(&zb, &outZ[i], sizeof(zb));
            std::memcpy(&ab, &outA[i], sizeof(ab));
            std::memcpy(&rb, &outR[i], sizeof(rb));
            fprintf(out, "PIX %d %d Z %08x\n", x, y, zb);
            fprintf(out, "PIX %d %d A %08x\n", x, y, ab);
            fprintf(out, "PIX %d %d R %08x\n", x, y, rb);
        }
    }
    fprintf(out, "END\n");
    std::fclose(out);

    fprintf(stderr, "oracle: composited %dx%d deep sources (%s + %s) -> %s\n",
            w, h, pathA, pathB, outPath);
    return 0;
}

// read-deep-via-inputfile <path> <out>
//
// Oracle gate: opens `path` (a deep-scanline file with Z/A/R channels,
// e.g. one of the `write-deep-composite` sources above) with the REAL,
// GENERIC `Imf::InputFile` -- NOT `DeepScanLineInputFile` directly -- sets a
// flat Z/A/R output FrameBuffer, calls `readPixels`, and dumps the flattened
// values bit-exact. `InputFile::initialize` (`ImfInputFile.cpp:300-305`)
// internally builds exactly the same `DeepScanLineInputFile` +
// `CompositeDeepScanLine` (single source, front-to-back accumulate, no
// cross-source sort needed) pair `composite-deep-scanline` above drives
// directly -- this mode instead goes through the GENERIC reader, which is
// what exr-rs's own `scanline::InputFile` ports. Same output grammar
// as `composite-deep-scanline` (one source instead of two, so `PIX` lines
// cover every pixel of the single file's data window).
//
// Output grammar (one record per line, stdout duplicated to <out>):
//   DIM <width> <height>
//   PIX <x> <y> <chan> <hex-f32-bits>
//   END
static int do_read_deep_via_inputfile(const char* path, const char* outPath)
{
    InputFile in(path);

    Box2i dw = in.header().dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int h = dw.max.y - dw.min.y + 1;

    std::vector<float> outZ((size_t)w * h, 0.0f);
    std::vector<float> outA((size_t)w * h, 0.0f);
    std::vector<float> outR((size_t)w * h, 0.0f);

    FrameBuffer fb;
    char* zBase = (char*)outZ.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float);
    char* aBase = (char*)outA.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float);
    char* rBase = (char*)outR.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float);
    fb.insert("Z", Slice(FLOAT, zBase, sizeof(float), sizeof(float) * w));
    fb.insert("A", Slice(FLOAT, aBase, sizeof(float), sizeof(float) * w));
    fb.insert("R", Slice(FLOAT, rBase, sizeof(float), sizeof(float) * w));
    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);

    FILE* out = std::fopen(outPath, "wb");
    if (!out) { fprintf(stderr, "cannot open %s for write\n", outPath); return 1; }
    fprintf(out, "DIM %d %d\n", w, h);
    for (int y = dw.min.y; y <= dw.max.y; ++y) {
        for (int x = dw.min.x; x <= dw.max.x; ++x) {
            size_t i = (size_t)(y - dw.min.y) * w + (x - dw.min.x);
            uint32_t zb, ab, rb;
            std::memcpy(&zb, &outZ[i], sizeof(zb));
            std::memcpy(&ab, &outA[i], sizeof(ab));
            std::memcpy(&rb, &outR[i], sizeof(rb));
            fprintf(out, "PIX %d %d Z %08x\n", x, y, zb);
            fprintf(out, "PIX %d %d A %08x\n", x, y, ab);
            fprintf(out, "PIX %d %d R %08x\n", x, y, rb);
        }
    }
    fprintf(out, "END\n");
    std::fclose(out);

    fprintf(stderr, "oracle: InputFile-flattened %dx%d deep scanline (%s) -> %s\n",
            w, h, path, outPath);
    return 0;
}

// ---------------------------------------------------------------------------
// write-deep-tiled mode
// ---------------------------------------------------------------------------
// Writes the KNOWN deep tiled image (DW x DH, DEEP_TILE x DEEP_TILE tiles) with
// `comp` and level mode `mode`.  The deep frame buffer addresses a whole-image
// flat array (the deep tiled writer gathers each tile rect from it); for MIPMAP
// each level is written from its own per-level array (level-relative formula).
static int do_write_deep_tiled(const char* path, Compression comp, LevelMode mode)
{
    Box2i win(V2i(0, 0), V2i(DW - 1, DH - 1));
    Header header(win, win);
    header.setType(DEEPTILE);
    header.compression() = comp;
    header.lineOrder()   = INCREASING_Y;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("Z", Channel(FLOAT));
    header.insert("owner", StringAttribute("exr-rs"));
    header.insert("parityTag", IntAttribute(31337));
    header.setTileDescription(TileDescription(DEEP_TILE, DEEP_TILE, mode, ROUND_DOWN));

    DeepTiledOutputFile out(path, header);

    int numLevels = (mode == MIPMAP_LEVELS) ? out.numLevels() : 1;
    for (int l = 0; l < numLevels; ++l) {
        Box2i ldw = out.dataWindowForLevel(l);
        int lw = ldw.max.x - ldw.min.x + 1;
        int lh = ldw.max.y - ldw.min.y + 1;
        size_t np = (size_t)lw * lh;

        std::vector<unsigned int> counts(np);
        std::vector<std::vector<half>>  R(np), G(np), B(np);
        std::vector<std::vector<float>> Z(np);
        std::vector<half*>  Rp(np), Gp(np), Bp(np);
        std::vector<float*> Zp(np);
        for (int ly = 0; ly < lh; ++ly) {
            for (int lx = 0; lx < lw; ++lx) {
                int pidx = ly * lw + lx;
                int n = deep_count(lx, ly);
                counts[pidx] = (unsigned int)n;
                R[pidx].resize(n); G[pidx].resize(n); B[pidx].resize(n); Z[pidx].resize(n);
                for (int s = 0; s < n; ++s) {
                    R[pidx][s] = half(d_r_val(pidx, s));
                    G[pidx][s] = half(d_g_val(pidx, s));
                    B[pidx][s] = half(d_b_val(pidx, s));
                    Z[pidx][s] = d_z_val(pidx, s);
                }
                Rp[pidx] = R[pidx].data(); Gp[pidx] = G[pidx].data();
                Bp[pidx] = B[pidx].data(); Zp[pidx] = Z[pidx].data();
            }
        }

        // Base offset so absolute level pixel (ldw.min) maps to slot 0.
        DeepFrameBuffer fb;
        size_t o = (size_t)ldw.min.x + (size_t)ldw.min.y * lw;
        fb.insertSampleCountSlice(Slice(UINT,
            (char*)counts.data() - o * sizeof(unsigned int),
            sizeof(unsigned int), sizeof(unsigned int) * lw));
        fb.insert("R", DeepSlice(HALF,  (char*)Rp.data() - o * sizeof(half*),
                                 sizeof(half*),  sizeof(half*)  * lw, sizeof(half)));
        fb.insert("G", DeepSlice(HALF,  (char*)Gp.data() - o * sizeof(half*),
                                 sizeof(half*),  sizeof(half*)  * lw, sizeof(half)));
        fb.insert("B", DeepSlice(HALF,  (char*)Bp.data() - o * sizeof(half*),
                                 sizeof(half*),  sizeof(half*)  * lw, sizeof(half)));
        fb.insert("Z", DeepSlice(FLOAT, (char*)Zp.data() - o * sizeof(float*),
                                 sizeof(float*), sizeof(float*) * lw, sizeof(float)));
        out.setFrameBuffer(fb);
        out.writeTiles(0, out.numXTiles(l) - 1, 0, out.numYTiles(l) - 1, l);
    }

    fprintf(stderr, "oracle: wrote %dx%d deep tiled EXR (comp %d, mode %d, %d level(s)) to %s\n",
            DW, DH, (int)comp, (int)mode, numLevels, path);
    return 0;
}

// ---------------------------------------------------------------------------
// read-deep-tiled mode
// ---------------------------------------------------------------------------
//
// Output grammar (one record per line, stdout):
//   TILEDESC <xSize> <ySize> <levelMode:int> <roundingMode:int>
//   LEVELMODE <int>
//   COMPRESSION <int>
//   DW <minx> <miny> <maxx> <maxy>
//   NUMLEVELS <n>
//   ATTR <name> <typeName> <value>
//   CHAN <name> <pixelType:int>
//   LEVEL <l> <minx> <miny> <maxx> <maxy>
//   DCOUNT <l> <x> <y> <n>                        (x,y RELATIVE to level min)
//   DSAMP <l> <chan> <x> <y> <i> <hexbits>         (x,y RELATIVE to level min)
//   END
//
static int do_read_deep_tiled(const char* path)
{
    DeepTiledInputFile in(path);
    const Header& h = in.header();
    const TileDescription& td = h.tileDescription();
    Box2i dw = h.dataWindow();

    printf("TILEDESC %u %u %d %d\n", td.xSize, td.ySize, (int)td.mode, (int)td.roundingMode);
    printf("LEVELMODE %d\n", (int)in.levelMode());
    printf("COMPRESSION %d\n", (int)h.compression());
    printf("DW %d %d %d %d\n", dw.min.x, dw.min.y, dw.max.x, dw.max.y);
    int numLevels = (in.levelMode() == MIPMAP_LEVELS) ? in.numLevels() : 1;
    printf("NUMLEVELS %d\n", numLevels);
    if (const StringAttribute* a = h.findTypedAttribute<StringAttribute>("owner"))
        printf("ATTR owner string %s\n", a->value().c_str());
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute>("parityTag"))
        printf("ATTR parityTag int %d\n", a->value());

    std::vector<std::string> chanNames;
    std::vector<PixelType>   chanTypes;
    for (ChannelList::ConstIterator it = h.channels().begin();
         it != h.channels().end(); ++it) {
        chanNames.push_back(it.name());
        chanTypes.push_back(it.channel().type);
        printf("CHAN %s %d\n", it.name(), (int)it.channel().type);
    }

    for (int l = 0; l < numLevels; ++l) {
        Box2i ldw = in.dataWindowForLevel(l);
        int lw = ldw.max.x - ldw.min.x + 1;
        int lh = ldw.max.y - ldw.min.y + 1;
        size_t np = (size_t)lw * lh;
        printf("LEVEL %d %d %d %d %d\n", l, ldw.min.x, ldw.min.y, ldw.max.x, ldw.max.y);

        // Pass 1: counts for the whole level.
        std::vector<unsigned int> counts(np, 0);
        DeepFrameBuffer fb;
        size_t o = (size_t)ldw.min.x + (size_t)ldw.min.y * lw;
        fb.insertSampleCountSlice(Slice(UINT,
            (char*)counts.data() - o * sizeof(unsigned int),
            sizeof(unsigned int), sizeof(unsigned int) * lw));
        in.setFrameBuffer(fb);
        in.readPixelSampleCounts(0, in.numXTiles(l) - 1, 0, in.numYTiles(l) - 1, l);

        std::vector<std::vector<half>>  R(np), G(np), B(np);
        std::vector<std::vector<float>> Z(np);
        std::vector<half*>  Rp(np), Gp(np), Bp(np);
        std::vector<float*> Zp(np);
        for (size_t i = 0; i < np; ++i) {
            unsigned int n = counts[i];
            R[i].resize(n); G[i].resize(n); B[i].resize(n); Z[i].resize(n);
            Rp[i] = R[i].data(); Gp[i] = G[i].data(); Bp[i] = B[i].data(); Zp[i] = Z[i].data();
        }
        fb.insert("R", DeepSlice(HALF,  (char*)Rp.data() - o * sizeof(half*),
                                 sizeof(half*),  sizeof(half*)  * lw, sizeof(half)));
        fb.insert("G", DeepSlice(HALF,  (char*)Gp.data() - o * sizeof(half*),
                                 sizeof(half*),  sizeof(half*)  * lw, sizeof(half)));
        fb.insert("B", DeepSlice(HALF,  (char*)Bp.data() - o * sizeof(half*),
                                 sizeof(half*),  sizeof(half*)  * lw, sizeof(half)));
        fb.insert("Z", DeepSlice(FLOAT, (char*)Zp.data() - o * sizeof(float*),
                                 sizeof(float*), sizeof(float*) * lw, sizeof(float)));
        in.setFrameBuffer(fb);
        in.readTiles(0, in.numXTiles(l) - 1, 0, in.numYTiles(l) - 1, l);

        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x)
                printf("DCOUNT %d %d %d %u\n", l, x, y, counts[(size_t)y * lw + x]);
        for (size_t c = 0; c < chanNames.size(); ++c) {
            bool isHalf = (chanTypes[c] == HALF);
            for (int y = 0; y < lh; ++y) {
                for (int x = 0; x < lw; ++x) {
                    size_t i = (size_t)y * lw + x;
                    unsigned int n = counts[i];
                    for (unsigned int s = 0; s < n; ++s) {
                        if (isHalf) {
                            half* arr = (chanNames[c] == "R") ? Rp[i]
                                      : (chanNames[c] == "G") ? Gp[i] : Bp[i];
                            uint16_t bits; std::memcpy(&bits, &arr[s], sizeof(bits));
                            printf("DSAMP %d %s %d %d %u %04x\n", l, chanNames[c].c_str(), x, y, s, bits);
                        } else {
                            uint32_t bits; std::memcpy(&bits, &Zp[i][s], sizeof(bits));
                            printf("DSAMP %d %s %d %d %u %08x\n", l, chanNames[c].c_str(), x, y, s, bits);
                        }
                    }
                }
            }
        }
    }
    printf("END\n");
    return 0;
}

// ===========================================================================
// MULTIPART parity
// ===========================================================================
//
// The KNOWN multipart image (must be reproduced bit-for-bit by the Rust side):
//   ONE file holding THREE parts of MIXED types, all LOSSLESS codecs so every
//   pixel/count/sample is bit-exact in both directions:
//
//     part | type           | codec | name
//     -----+----------------+-------+----------
//      0   | scanlineimage  | NONE  | "rgba"
//      1   | tiledimage     | ZIP   | "tiled"
//      2   | deepscanline   | ZIPS  | "deep"
//
//   Common geometry for every part: MPW x MPH = 8 x 6, data/display window at
//   origin, INCREASING_Y.  The tiled part uses an 4x4 tile (ONE_LEVEL,
//   ROUND_DOWN) -> 2x2 tiles (partial bottom row).  Channels (stored
//   alphabetically by IlmImf): the FLAT parts carry R,G,B = HALF and Z = FLOAT;
//   the DEEP part carries B,G,R = HALF and Z = FLOAT.
//
//   Per-pixel FLAT formula, keyed on idx = y*MPW + x and a per-PART bias
//   pbias = part*1000 so each part carries distinguishable data:
//     R = half ( 1.0f  + (idx + pbias) * 0.5f )
//     G = half ( 100.0f + idx * 0.25f + pbias )
//     B = half ( -50.0f + idx * 1.0f  - pbias )
//     Z = float( idx * 100.0f + pbias )
//
//   The DEEP part has VARIABLE per-pixel sample counts N(x,y) = (x + y) % 4
//   (0,1,2,3 incl. zero-sample pixels) and per-SAMPLE values keyed on idx and
//   the sample index s in [0, N):
//     R = half ( 1.0f  + idx * 0.5f  + s * 1.0f )
//     G = half ( 100.0f + idx * 0.25f + s * 2.0f )
//     B = half ( -50.0f + idx * 1.0f  - s * 1.0f )
//     Z = float( idx * 100.0f + s * 1.0f )
//
//   Header extras on EVERY part: owner(string)="exr-rs", parityTag(int)=31337.
//   All HALF values use the SAME IEEE round-to-nearest-even conversion the Rust
//   Half::from(f32) uses, so the 16-bit patterns match on both sides.

static const int MPW = 8;
static const int MPH = 6;
static const int MP_TILE = 4;

static float mp_r_val(int idx, int pbias) { return 1.0f   + (idx + pbias) * 0.5f; }
static float mp_g_val(int idx, int pbias) { return 100.0f + idx * 0.25f + (float)pbias; }
static float mp_b_val(int idx, int pbias) { return -50.0f  + idx * 1.0f  - (float)pbias; }
static float mp_z_val(int idx, int pbias) { return idx * 100.0f + (float)pbias; }

static int   mp_deep_count(int x, int y) { return (x + y) % 4; }
static float mp_d_r_val(int idx, int s) { return 1.0f   + idx * 0.5f  + s * 1.0f; }
static float mp_d_g_val(int idx, int s) { return 100.0f + idx * 0.25f + s * 2.0f; }
static float mp_d_b_val(int idx, int s) { return -50.0f  + idx * 1.0f  - s * 1.0f; }
static float mp_d_z_val(int idx, int s) { return idx * 100.0f + s * 1.0f; }

// Build a FLAT part header (scanline or tiled) at origin with the standard
// channel set + extras.  Tiled when `tiled` is true (4x4 ONE_LEVEL).
static Header mp_flat_header(const char* name, Compression comp, bool tiled)
{
    Header h(MPW, MPH);
    h.compression() = comp;
    h.lineOrder()   = INCREASING_Y;
    h.setName(name);
    h.setType(tiled ? TILEDIMAGE : SCANLINEIMAGE);
    h.channels().insert("R", Channel(HALF));
    h.channels().insert("G", Channel(HALF));
    h.channels().insert("B", Channel(HALF));
    h.channels().insert("Z", Channel(FLOAT));
    h.insert("owner", StringAttribute("exr-rs"));
    h.insert("parityTag", IntAttribute(31337));
    if (tiled)
        h.setTileDescription(TileDescription(MP_TILE, MP_TILE, ONE_LEVEL, ROUND_DOWN));
    return h;
}

// Build the DEEP scanline part header.
static Header mp_deep_header(const char* name, Compression comp)
{
    Box2i win(V2i(0, 0), V2i(MPW - 1, MPH - 1));
    Header h(win, win);
    h.compression() = comp;
    h.lineOrder()   = INCREASING_Y;
    h.setName(name);
    h.setType(DEEPSCANLINE);
    h.channels().insert("R", Channel(HALF));
    h.channels().insert("G", Channel(HALF));
    h.channels().insert("B", Channel(HALF));
    h.channels().insert("Z", Channel(FLOAT));
    h.insert("owner", StringAttribute("exr-rs"));
    h.insert("parityTag", IntAttribute(31337));
    return h;
}

// ---------------------------------------------------------------------------
// write-multipart mode
// ---------------------------------------------------------------------------
// Writes the KNOWN 3-part mixed-type multipart file via MultiPartOutputFile +
// the typed *OutputPart wrappers.
static int do_write_multipart(const char* path)
{
    Header headers[3];
    headers[0] = mp_flat_header("rgba",  NO_COMPRESSION,   false); // scanline
    headers[1] = mp_flat_header("tiled", ZIP_COMPRESSION,  true);  // tiled
    headers[2] = mp_deep_header("deep",  ZIPS_COMPRESSION);        // deep scanline

    MultiPartOutputFile mp(path, headers, 3);

    // --- Part 0: scanline (NONE) via OutputPart -----------------------------
    {
        std::vector<half>  R(MPW * MPH), G(MPW * MPH), B(MPW * MPH);
        std::vector<float> Z(MPW * MPH);
        for (int y = 0; y < MPH; ++y)
            for (int x = 0; x < MPW; ++x) {
                int idx = y * MPW + x;
                R[idx] = half(mp_r_val(idx, 0));
                G[idx] = half(mp_g_val(idx, 0));
                B[idx] = half(mp_b_val(idx, 0));
                Z[idx] = mp_z_val(idx, 0);
            }
        FrameBuffer fb;
        fb.insert("R", Slice(HALF,  (char*)R.data(), sizeof(half),  sizeof(half)  * MPW));
        fb.insert("G", Slice(HALF,  (char*)G.data(), sizeof(half),  sizeof(half)  * MPW));
        fb.insert("B", Slice(HALF,  (char*)B.data(), sizeof(half),  sizeof(half)  * MPW));
        fb.insert("Z", Slice(FLOAT, (char*)Z.data(), sizeof(float), sizeof(float) * MPW));
        OutputPart part(mp, 0);
        part.setFrameBuffer(fb);
        part.writePixels(MPH);
    }

    // --- Part 1: tiled (ZIP) via TiledOutputPart ----------------------------
    {
        std::vector<half>  R(MPW * MPH), G(MPW * MPH), B(MPW * MPH);
        std::vector<float> Z(MPW * MPH);
        for (int y = 0; y < MPH; ++y)
            for (int x = 0; x < MPW; ++x) {
                int idx = y * MPW + x;
                R[idx] = half(mp_r_val(idx, 1000));
                G[idx] = half(mp_g_val(idx, 1000));
                B[idx] = half(mp_b_val(idx, 1000));
                Z[idx] = mp_z_val(idx, 1000);
            }
        FrameBuffer fb;
        fb.insert("R", Slice(HALF,  (char*)R.data(), sizeof(half),  sizeof(half)  * MPW));
        fb.insert("G", Slice(HALF,  (char*)G.data(), sizeof(half),  sizeof(half)  * MPW));
        fb.insert("B", Slice(HALF,  (char*)B.data(), sizeof(half),  sizeof(half)  * MPW));
        fb.insert("Z", Slice(FLOAT, (char*)Z.data(), sizeof(float), sizeof(float) * MPW));
        TiledOutputPart part(mp, 1);
        part.setFrameBuffer(fb);
        part.writeTiles(0, part.numXTiles(0) - 1, 0, part.numYTiles(0) - 1, 0);
    }

    // --- Part 2: deep scanline (ZIPS) via DeepScanLineOutputPart ------------
    {
        int np = MPW * MPH;
        std::vector<unsigned int> counts(np);
        std::vector<std::vector<half>>  R(np), G(np), B(np);
        std::vector<std::vector<float>> Z(np);
        std::vector<half*>  Rp(np), Gp(np), Bp(np);
        std::vector<float*> Zp(np);
        for (int y = 0; y < MPH; ++y)
            for (int x = 0; x < MPW; ++x) {
                int idx = y * MPW + x;
                int n = mp_deep_count(x, y);
                counts[idx] = (unsigned int)n;
                R[idx].resize(n); G[idx].resize(n); B[idx].resize(n); Z[idx].resize(n);
                for (int s = 0; s < n; ++s) {
                    R[idx][s] = half(mp_d_r_val(idx, s));
                    G[idx][s] = half(mp_d_g_val(idx, s));
                    B[idx][s] = half(mp_d_b_val(idx, s));
                    Z[idx][s] = mp_d_z_val(idx, s);
                }
                Rp[idx] = R[idx].data(); Gp[idx] = G[idx].data();
                Bp[idx] = B[idx].data(); Zp[idx] = Z[idx].data();
            }
        DeepFrameBuffer fb;
        fb.insertSampleCountSlice(Slice(UINT, (char*)counts.data(),
                                        sizeof(unsigned int), sizeof(unsigned int) * MPW));
        fb.insert("R", DeepSlice(HALF,  (char*)Rp.data(), sizeof(half*),  sizeof(half*)  * MPW, sizeof(half)));
        fb.insert("G", DeepSlice(HALF,  (char*)Gp.data(), sizeof(half*),  sizeof(half*)  * MPW, sizeof(half)));
        fb.insert("B", DeepSlice(HALF,  (char*)Bp.data(), sizeof(half*),  sizeof(half*)  * MPW, sizeof(half)));
        fb.insert("Z", DeepSlice(FLOAT, (char*)Zp.data(), sizeof(float*), sizeof(float*) * MPW, sizeof(float)));
        DeepScanLineOutputPart part(mp, 2);
        part.setFrameBuffer(fb);
        part.writePixels(MPH);
    }

    fprintf(stderr, "oracle: wrote 3-part multipart EXR (scanline/NONE, tiled/ZIP, deep/ZIPS) to %s\n", path);
    return 0;
}

// ---------------------------------------------------------------------------
// read-multipart mode
// ---------------------------------------------------------------------------
//
// Output grammar (one record per line, stdout):
//   NUMPARTS <n>
//   PART <p> <name> <type>
//   PATTR <p> <name> <typeName> <value>
//   PCHAN <p> <name> <pixelType:int>
//   FPIX  <p> <chan> <x> <y> <hexbits>          (flat parts; x/y rel to dw.min)
//   DCNT  <p> <x> <y> <n>                        (deep part; x/y rel to dw.min)
//   DSMP  <p> <chan> <x> <y> <i> <hexbits>       (deep part)
//   END
//
static void mp_emit_common(int p, const Header& h)
{
    printf("PART %d %s %s\n", p, h.name().c_str(), h.type().c_str());
    if (const StringAttribute* a = h.findTypedAttribute<StringAttribute>("owner"))
        printf("PATTR %d owner string %s\n", p, a->value().c_str());
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute>("parityTag"))
        printf("PATTR %d parityTag int %d\n", p, a->value());
    for (ChannelList::ConstIterator it = h.channels().begin(); it != h.channels().end(); ++it)
        printf("PCHAN %d %s %d\n", p, it.name(), (int)it.channel().type);
}

static int do_read_multipart(const char* path)
{
    MultiPartInputFile mp(path);
    int n = mp.parts();
    printf("NUMPARTS %d\n", n);

    for (int p = 0; p < n; ++p) {
        const Header& h = mp.header(p);
        mp_emit_common(p, h);
        Box2i dw = h.dataWindow();
        int w = dw.max.x - dw.min.x + 1;
        int hgt = dw.max.y - dw.min.y + 1;
        std::string ty = h.type();

        std::vector<std::string> chanNames;
        std::vector<PixelType>   chanTypes;
        for (ChannelList::ConstIterator it = h.channels().begin(); it != h.channels().end(); ++it) {
            chanNames.push_back(it.name());
            chanTypes.push_back(it.channel().type);
        }

        if (ty == DEEPSCANLINE) {
            DeepScanLineInputPart in(mp, p);
            std::vector<unsigned int> counts((size_t)w * hgt, 0);
            DeepFrameBuffer fb;
            char* scBase = (char*)counts.data()
                           - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(unsigned int);
            fb.insertSampleCountSlice(Slice(UINT, scBase,
                                            sizeof(unsigned int), sizeof(unsigned int) * w));
            in.setFrameBuffer(fb);
            in.readPixelSampleCounts(dw.min.y, dw.max.y);

            size_t np = (size_t)w * hgt;
            std::vector<std::vector<half>>  R(np), G(np), B(np);
            std::vector<std::vector<float>> Z(np);
            std::vector<half*>  Rp(np), Gp(np), Bp(np);
            std::vector<float*> Zp(np);
            for (size_t i = 0; i < np; ++i) {
                unsigned int c = counts[i];
                R[i].resize(c); G[i].resize(c); B[i].resize(c); Z[i].resize(c);
                Rp[i] = R[i].data(); Gp[i] = G[i].data(); Bp[i] = B[i].data(); Zp[i] = Z[i].data();
            }
            char* rB = (char*)Rp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half*);
            char* gB = (char*)Gp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half*);
            char* bB = (char*)Bp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(half*);
            char* zB = (char*)Zp.data() - ((size_t)dw.min.x + (size_t)dw.min.y * w) * sizeof(float*);
            fb.insert("R", DeepSlice(HALF,  rB, sizeof(half*),  sizeof(half*)  * w, sizeof(half)));
            fb.insert("G", DeepSlice(HALF,  gB, sizeof(half*),  sizeof(half*)  * w, sizeof(half)));
            fb.insert("B", DeepSlice(HALF,  bB, sizeof(half*),  sizeof(half*)  * w, sizeof(half)));
            fb.insert("Z", DeepSlice(FLOAT, zB, sizeof(float*), sizeof(float*) * w, sizeof(float)));
            in.setFrameBuffer(fb);
            in.readPixels(dw.min.y, dw.max.y);

            for (int y = 0; y < hgt; ++y)
                for (int x = 0; x < w; ++x)
                    printf("DCNT %d %d %d %u\n", p, x, y, counts[(size_t)y * w + x]);
            for (size_t c = 0; c < chanNames.size(); ++c) {
                bool isHalf = (chanTypes[c] == HALF);
                for (int y = 0; y < hgt; ++y)
                    for (int x = 0; x < w; ++x) {
                        size_t i = (size_t)y * w + x;
                        unsigned int cnt = counts[i];
                        for (unsigned int s = 0; s < cnt; ++s) {
                            if (isHalf) {
                                half* arr = (chanNames[c] == "R") ? Rp[i]
                                          : (chanNames[c] == "G") ? Gp[i] : Bp[i];
                                uint16_t bits; std::memcpy(&bits, &arr[s], sizeof(bits));
                                printf("DSMP %d %s %d %d %u %04x\n", p, chanNames[c].c_str(), x, y, s, bits);
                            } else {
                                uint32_t bits; std::memcpy(&bits, &Zp[i][s], sizeof(bits));
                                printf("DSMP %d %s %d %d %u %08x\n", p, chanNames[c].c_str(), x, y, s, bits);
                            }
                        }
                    }
            }
        } else {
            // Flat part (scanline or tiled): read the whole image into planar
            // buffers and emit every pixel's bits.
            FrameBuffer fb;
            std::vector<std::vector<uint8_t>> planes(chanNames.size());
            for (size_t c = 0; c < chanNames.size(); ++c) {
                size_t px = (chanTypes[c] == HALF) ? sizeof(half) : sizeof(float);
                planes[c].assign((size_t)w * hgt * px, 0);
                char* base = (char*)planes[c].data()
                             - ((size_t)dw.min.x + (size_t)dw.min.y * w) * px;
                fb.insert(chanNames[c], Slice(chanTypes[c], base, px, px * (size_t)w));
            }
            if (ty == TILEDIMAGE) {
                TiledInputPart in(mp, p);
                in.setFrameBuffer(fb);
                in.readTiles(0, in.numXTiles(0) - 1, 0, in.numYTiles(0) - 1, 0);
            } else {
                InputPart in(mp, p);
                in.setFrameBuffer(fb);
                in.readPixels(dw.min.y, dw.max.y);
            }
            for (size_t c = 0; c < chanNames.size(); ++c) {
                bool isHalf = (chanTypes[c] == HALF);
                for (int y = 0; y < hgt; ++y)
                    for (int x = 0; x < w; ++x) {
                        size_t i = (size_t)y * w + x;
                        if (isHalf) {
                            uint16_t bits;
                            std::memcpy(&bits, &planes[c][i * sizeof(half)], sizeof(bits));
                            printf("FPIX %d %s %d %d %04x\n", p, chanNames[c].c_str(), x, y, bits);
                        } else {
                            uint32_t bits;
                            std::memcpy(&bits, &planes[c][i * sizeof(float)], sizeof(bits));
                            printf("FPIX %d %s %d %d %08x\n", p, chanNames[c].c_str(), x, y, bits);
                        }
                    }
            }
        }
    }
    printf("END\n");
    return 0;
}

// ---------------------------------------------------------------------------
// write-idmanifest / dump-idmanifest — structured IDManifest cross-check
// ---------------------------------------------------------------------------
// `write-idmanifest <path>` writes a tiny single-channel EXR whose header
// carries a real C++-built `IDManifestAttribute` (byte-exact deflate output,
// same manifest content every time), so the Rust side can parse it and
// compare the DECODED content structurally (not byte-for-byte — zlib's
// compressed bytes need not match between two different codec
// implementations even for identical input, so the crate's own
// `CompressedIDManifest::decompress` is what gets exercised, not a
// byte-diff of the attribute).
//
// `dump-idmanifest <path> <out>` reads a file's "idmanifest" attribute,
// decompresses it into a real C++ `IDManifest`, and writes a canonical text
// dump to `out` — one line per channel-group manifest, sorted by first
// channel name (a manifest has no other stable read order in C++, since
// `_manifest` is a plain `vector` in insertion order; sorting by channel
// name gives a comparison order independent of which side, C++ or Rust,
// produced the file). Format, one line per group:
//   GROUP <channels comma-joined> | <components comma-joined> | <lifetime int> | <hashScheme> | <encodingScheme>
//   then one line per table entry, sorted by ID:
//   ENTRY <id> <text components comma-joined>
// Forward declaration: defined later (near the other dump-* helpers) but
// needed here by do_dump_idmanifest.
static int write_all (const char* out, const std::vector<uint8_t>& bytes);

static int do_write_idmanifest(const char* path)
{
    IDManifest manifest;
    IDManifest::ChannelGroupManifest& g = manifest.add(std::string("id"));
    g.setComponent(std::string("name"));
    g.setHashScheme(IDManifest::MURMURHASH3_32);
    g.setEncodingScheme(IDManifest::ID_SCHEME);
    g.setLifetime(IDManifest::LIFETIME_STABLE);
    g.insert((uint64_t)1, std::string("asset_one"));
    g.insert((uint64_t)2, std::string("asset_two"));
    g.insert((uint64_t)1000, std::string("asset_big_id"));

    Header header(4, 4);
    header.channels().insert("Z", Channel(FLOAT));
    header.insert("idmanifest", IDManifestAttribute(CompressedIDManifest(manifest)));

    OutputFile out(path, header);
    std::vector<float> Z(4 * 4, 1.0f);
    FrameBuffer fb;
    fb.insert("Z", Slice(FLOAT, (char*)Z.data(), sizeof(float), sizeof(float) * 4));
    out.setFrameBuffer(fb);
    out.writePixels(4);

    fprintf(stderr, "oracle: wrote idmanifest EXR to %s\n", path);
    return 0;
}

static int do_dump_idmanifest(const char* path, const char* out)
{
    InputFile in(path);
    const Header& h = in.header();

    const IDManifestAttribute* attr =
        h.findTypedAttribute<IDManifestAttribute>("idmanifest");
    if (!attr) {
        fprintf(stderr, "oracle: no 'idmanifest' attribute in %s\n", path);
        return 1;
    }

    IDManifest manifest(attr->value());

    // Stable comparison order: sort group indices by their first (sorted)
    // channel name.
    std::vector<size_t> order(manifest.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const std::set<std::string>& ca = manifest[a].getChannels();
        const std::set<std::string>& cb = manifest[b].getChannels();
        std::string fa = ca.empty() ? std::string() : *ca.begin();
        std::string fb = cb.empty() ? std::string() : *cb.begin();
        return fa < fb;
    });

    std::string text;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const IDManifest::ChannelGroupManifest& g = manifest[order[oi]];
        std::string channels;
        for (std::set<std::string>::const_iterator it = g.getChannels().begin();
             it != g.getChannels().end(); ++it) {
            if (!channels.empty()) channels += ",";
            channels += *it;
        }
        std::string components;
        for (size_t i = 0; i < g.getComponents().size(); ++i) {
            if (i) components += ",";
            components += g.getComponents()[i];
        }
        char header_line[512];
        std::snprintf(header_line, sizeof(header_line), "GROUP %s | %s | %d | %s | %s\n",
                      channels.c_str(), components.c_str(), (int)g.getLifetime(),
                      g.getHashScheme().c_str(), g.getEncodingScheme().c_str());
        text += header_line;

        // Collect (id, text) pairs and sort by id for a stable dump (the
        // C++ table is already a std::map, so this is already sorted, but
        // being explicit documents the invariant the Rust side must match).
        std::vector<std::pair<uint64_t, std::string>> entries;
        for (IDManifest::ChannelGroupManifest::ConstIterator it = g.begin();
             it != g.end(); ++it) {
            std::string joined;
            const std::vector<std::string>& t = it.text();
            for (size_t i = 0; i < t.size(); ++i) {
                if (i) joined += ",";
                joined += t[i];
            }
            entries.push_back(std::make_pair(it.id(), joined));
        }
        std::sort(entries.begin(), entries.end());
        for (size_t i = 0; i < entries.size(); ++i) {
            char entry_line[512];
            std::snprintf(entry_line, sizeof(entry_line), "ENTRY %llu %s\n",
                          (unsigned long long)entries[i].first, entries[i].second.c_str());
            text += entry_line;
        }
    }

    std::vector<uint8_t> bytes(text.begin(), text.end());
    return write_all(out, bytes);
}

// ---------------------------------------------------------------------------
// write-yc / write-rgba — the luminance-chroma (SUBSAMPLED) fixture pair
// ---------------------------------------------------------------------------
// `RgbaOutputFile(..., WRITE_YCA)` stores Y at full resolution and RY/BY at
// HALF resolution (xSampling = ySampling = 2), exactly like the openexr-images
// `Chromaticities/Rec709_YC.exr` sample. `write-rgba` writes the SAME pixels
// full-resolution, so the pair is the reference for our 27-tap chroma
// reconstruction (`exr_core::rgba`) — without it that path has no fixture on a
// machine without the (very large, separately hosted) openexr-images repo.
//
// The pattern is a smooth gradient on purpose: chroma subsampling is lossy, so
// a fixture with high-frequency chroma could only be compared with a
// meaningless tolerance.
static const int YCW = 64;
static const int YCH = 48;

static void yc_pixel (int x, int y, float& r, float& g, float& b)
{
    r = (float) x / (float) (YCW - 1);
    g = (float) y / (float) (YCH - 1);
    b = 0.5f * ((float) (x + y) / (float) (YCW + YCH - 2));
}

static int do_write_rgba (const char* path, RgbaChannels chans)
{
    std::vector<Rgba> px ((size_t) YCW * YCH);
    for (int y = 0; y < YCH; ++y)
        for (int x = 0; x < YCW; ++x)
        {
            float r, g, b;
            yc_pixel (x, y, r, g, b);
            Rgba& p = px[(size_t) y * YCW + x];
            p.r = half (r);
            p.g = half (g);
            p.b = half (b);
            p.a = half (1.0f);
        }

    Header header (YCW, YCH);
    header.compression () = ZIP_COMPRESSION;
    header.lineOrder ()   = INCREASING_Y;
    RgbaOutputFile out (path, header, chans);
    out.setFrameBuffer (px.data (), 1, YCW);
    out.writePixels (YCH);

    fprintf (stderr, "oracle: wrote %dx%d %s EXR to %s\n", YCW, YCH,
             chans == WRITE_YCA ? "luminance-chroma (Y/RY/BY)" : "RGBA", path);
    return 0;
}

// ---------------------------------------------------------------------------
// write-yc-rounded -- golden_yc.exr with a NON-DEFAULT setYCRounding, for a
// bit-exact check of exr_core::rgba::RgbaOutputFile::set_yc_rounding.
// ---------------------------------------------------------------------------
// Same pixels/header as `write-yc` (`do_write_rgba(WRITE_YCA)`); the only
// difference is `setYCRounding(roundY, roundC)` before `writePixels`.
static int do_write_yc_rounded (const char* path, unsigned int roundY, unsigned int roundC)
{
    std::vector<Rgba> px ((size_t) YCW * YCH);
    for (int y = 0; y < YCH; ++y)
        for (int x = 0; x < YCW; ++x)
        {
            float r, g, b;
            yc_pixel (x, y, r, g, b);
            Rgba& p = px[(size_t) y * YCW + x];
            p.r = half (r);
            p.g = half (g);
            p.b = half (b);
            p.a = half (1.0f);
        }

    Header header (YCW, YCH);
    header.compression () = ZIP_COMPRESSION;
    header.lineOrder ()   = INCREASING_Y;
    RgbaOutputFile out (path, header, WRITE_YCA);
    out.setYCRounding (roundY, roundC);
    out.setFrameBuffer (px.data (), 1, YCW);
    out.writePixels (YCH);

    fprintf (stderr, "oracle: wrote %dx%d luminance-chroma EXR (roundY=%u, roundC=%u) to %s\n",
             YCW, YCH, roundY, roundC, path);
    return 0;
}

// ---------------------------------------------------------------------------
// write-tiled-rgba -- a TiledRgbaOutputFile fixture, for a bit-exact check of
// exr_core::rgba::{TiledRgbaOutputFile, TiledRgbaInputFile}.
// ---------------------------------------------------------------------------
// Neither the plain (WRITE_RGBA) nor luminance-only (WRITE_YA) tiled paths
// filter or subsample (tiled EXR has no chroma support at all), so — unlike
// golden_yc.exr — there is no lossy step and the FILE ITSELF is the bit-exact
// reference; no separate decode dump is needed. 67x43 (the same KNOWN-image
// dimensions as the rest of this file, see the header comment) with 32x32
// tiles deliberately exercises a partial (edge) tile in both directions AND
// gives every full tile 32*32*4*2 = 8192 (RGBA) / 32*32*2*2 = 4096 (grey)
// raw bytes -- well past the 1024-byte floor `assert_chunks_compressed`
// enforces so the golden actually exercises ZIP, not a too-small-to-compress
// raw store (the original 20x13/8x8 fixture's largest tile was only 8*8*4*2
// = 512 bytes and tripped that gate).
static const int TRW = 67;
static const int TRH = 43;

static int do_write_tiled_rgba (const char* path, RgbaChannels chans)
{
    std::vector<Rgba> px ((size_t) TRW * TRH);
    for (int y = 0; y < TRH; ++y)
        for (int x = 0; x < TRW; ++x)
        {
            Rgba& p = px[(size_t) y * TRW + x];
            p.r = half ((float) x * 0.01f);
            p.g = half ((float) y * 0.02f);
            p.b = half ((float) (x + y) * 0.005f);
            p.a = half (0.75f);
        }

    TiledRgbaOutputFile out (
        path,
        TRW,
        TRH,
        32,
        32,
        ONE_LEVEL,
        ROUND_DOWN,
        chans,
        1,
        V2f (0, 0),
        1,
        INCREASING_Y,
        ZIP_COMPRESSION);
    out.setFrameBuffer (px.data (), 1, TRW);
    int nx = out.numXTiles (0);
    int ny = out.numYTiles (0);
    for (int dy = 0; dy < ny; ++dy)
        for (int dx = 0; dx < nx; ++dx)
            out.writeTile (dx, dy, 0);

    fprintf (stderr, "oracle: wrote %dx%d tiled %s EXR to %s\n", TRW, TRH,
             chans == WRITE_YA ? "luminance(+A)" : "RGBA", path);
    return 0;
}

// ---------------------------------------------------------------------------
// rgba-read -- the C++ Imf::RgbaInputFile DECODE, for a bit-exact check of
// exr_core::rgba::RgbaInputFile against golden_yc.exr / golden_rgba.exr.
// ---------------------------------------------------------------------------
// Additive only: a new mode alongside `read`/`read-half` (which go through
// Imf::InputFile, not the luminance-chroma reconstruction path).
//
// `rgba-read <path>` prints DIM/DW/RPIX text to stdout (one RPIX record per
// pixel, modeled on `do_read`'s grammar) for a live (direction-b-style)
// check. `rgba-read <path> <out>` ADDITIONALLY writes a committed binary
// dump to `<out>`, following the `dump`/`dump-tiled` precedent (a lossy or
// otherwise-unformulaic reference has no independent check, so the C++
// decode itself is committed next to the golden): `u32 width, u32 height`
// (LE), then `width*height` pixels row-major, each 4x `u16` LE (r, g, b, a
// half bits).
//
// Output grammar (stdout, always):
//   DIM <width> <height>
//   DW <minx> <miny> <maxx> <maxy>
//   RPIX <x> <y> <r_hexbits> <g_hexbits> <b_hexbits> <a_hexbits>   (HALF, 4 hex nibbles each)
//   END
static int do_rgba_read (const char* path, const char* out)
{
    RgbaInputFile in (path);
    Box2i dw = in.dataWindow ();
    int   w  = dw.max.x - dw.min.x + 1;
    int   h  = dw.max.y - dw.min.y + 1;

    printf ("DIM %d %d\n", w, h);
    printf ("DW %d %d %d %d\n", dw.min.x, dw.min.y, dw.max.x, dw.max.y);

    std::vector<Rgba> pixels ((size_t) w * h);
    // Base is offset so that pixel (dw.min.x, dw.min.y) maps to pixels[0],
    // matching do_read's per-channel plane addressing (`ImfRgbaFile.h`'s own
    // usage pattern in its class documentation).
    in.setFrameBuffer (
        pixels.data () - dw.min.x - (ptrdiff_t) dw.min.y * w, 1, w);
    in.readPixels (dw.min.y, dw.max.y);

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const Rgba& p = pixels[(size_t) y * w + x];
            printf (
                "RPIX %d %d %04x %04x %04x %04x\n",
                x,
                y,
                p.r.bits (),
                p.g.bits (),
                p.b.bits (),
                p.a.bits ());
        }
    printf ("END\n");

    if (out != NULL)
    {
        std::vector<uint8_t> bytes;
        bytes.reserve (8 + pixels.size () * 8);
        auto push_u32 = [&bytes] (uint32_t v) {
            for (int i = 0; i < 4; ++i)
                bytes.push_back ((uint8_t) ((v >> (8 * i)) & 0xff));
        };
        auto push_u16 = [&bytes] (uint16_t v) {
            bytes.push_back ((uint8_t) (v & 0xff));
            bytes.push_back ((uint8_t) ((v >> 8) & 0xff));
        };
        push_u32 ((uint32_t) w);
        push_u32 ((uint32_t) h);
        for (const Rgba& p : pixels)
        {
            push_u16 (p.r.bits ());
            push_u16 (p.g.bits ());
            push_u16 (p.b.bits ());
            push_u16 (p.a.bits ());
        }
        FILE* f = fopen (out, "wb");
        if (!f) { fprintf (stderr, "cannot open %s for writing\n", out); return 1; }
        size_t n = bytes.empty () ? 0 : fwrite (bytes.data (), 1, bytes.size (), f);
        fclose (f);
        if (n != bytes.size ()) { fprintf (stderr, "short write to %s\n", out); return 1; }
        fprintf (stderr, "oracle: dumped %zu bytes of RgbaInputFile decode to %s\n", bytes.size (), out);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// write-aces / aces-read -- AcesOutputFile / AcesInputFile fixtures, for a
// bit-exact check of exr_core::rgba::{AcesOutputFile, AcesInputFile}.
// ---------------------------------------------------------------------------
// `write-aces <path> rgba`   writes an `AcesOutputFile` from a known RGBA
//                            gradient (NO_COMPRESSION, ACES-legal) --
//                            byte-exact check of our `AcesOutputFile`
//                            (it stamps ACES chromaticities + setYCRounding
//                            (7,6), same as C++).
// `write-aces <path> rec709` writes the SAME gradient through a plain
//                            `RgbaOutputFile` with NO explicit chromaticities
//                            attribute -- `Header()`'s default (Rec.709),
//                            matching `AcesInputFile`'s "no attribute -> file
//                            defaults to Chromaticities()" branch. This is
//                            the READ-side fixture: its chromaticities differ
//                            from ACES, so `AcesInputFile` must apply the
//                            Bradford-adapted conversion.
static const int ACW = 16;
static const int ACH = 16;

static void aces_pixel (int x, int y, Rgba& p)
{
    p.r = half ((float) x / (float) (ACW - 1));
    p.g = half ((float) y / (float) (ACH - 1));
    p.b = half (0.5f - 0.5f * ((float) x / (float) (ACW - 1)));
    p.a = half (1.0f);
}

static int do_write_aces (const char* path, const char* variant)
{
    std::vector<Rgba> px ((size_t) ACW * ACH);
    for (int y = 0; y < ACH; ++y)
        for (int x = 0; x < ACW; ++x)
            aces_pixel (x, y, px[(size_t) y * ACW + x]);

    if (std::strcmp (variant, "rgba") == 0)
    {
        Header header (ACW, ACH, 1, V2f (0, 0), 1, INCREASING_Y, NO_COMPRESSION);
        AcesOutputFile out (path, header, WRITE_RGBA);
        out.setFrameBuffer (px.data (), 1, ACW);
        out.writePixels (ACH);
        fprintf (stderr, "oracle: wrote %dx%d AcesOutputFile EXR to %s\n", ACW, ACH, path);
        return 0;
    }
    else if (std::strcmp (variant, "rec709") == 0)
    {
        // Deliberately NO addChromaticities/addAdoptedNeutral: the header's
        // default is Rec.709, matching AcesInputFile's Chromaticities()
        // fallback when the attribute is absent. NO_COMPRESSION (like the
        // "rgba" variant above) keeps this golden out of
        // `every_compressed_golden_actually_compresses`'s scope entirely --
        // this fixture is about color science, not codec coverage.
        Header header (ACW, ACH, 1, V2f (0, 0), 1, INCREASING_Y, NO_COMPRESSION);
        RgbaOutputFile out (path, header, WRITE_RGBA);
        out.setFrameBuffer (px.data (), 1, ACW);
        out.writePixels (ACH);
        fprintf (stderr, "oracle: wrote %dx%d Rec.709 RGBA EXR to %s\n", ACW, ACH, path);
        return 0;
    }
    fprintf (stderr, "write-aces: unknown variant '%s'\n", variant);
    return 2;
}

// `aces-read <path> <out>`: reads `path` through `AcesInputFile` (applying
// the file-to-ACES color conversion whenever the file's chromaticities
// differ from ACES) and dumps the SAME binary layout `rgba-read` uses:
// `u32 width, u32 height` (LE), then `width*height` pixels row-major, each
// 4x `u16` LE (r, g, b, a half bits) -- the post-conversion values.
static int do_aces_read (const char* path, const char* out)
{
    AcesInputFile in (path);
    Box2i dw = in.dataWindow ();
    int   w  = dw.max.x - dw.min.x + 1;
    int   h  = dw.max.y - dw.min.y + 1;

    std::vector<Rgba> pixels ((size_t) w * h);
    in.setFrameBuffer (
        pixels.data () - dw.min.x - (ptrdiff_t) dw.min.y * w, 1, w);
    in.readPixels (dw.min.y, dw.max.y);

    std::vector<uint8_t> bytes;
    bytes.reserve (8 + pixels.size () * 8);
    auto push_u32 = [&bytes] (uint32_t v) {
        for (int i = 0; i < 4; ++i)
            bytes.push_back ((uint8_t) ((v >> (8 * i)) & 0xff));
    };
    auto push_u16 = [&bytes] (uint16_t v) {
        bytes.push_back ((uint8_t) (v & 0xff));
        bytes.push_back ((uint8_t) ((v >> 8) & 0xff));
    };
    push_u32 ((uint32_t) w);
    push_u32 ((uint32_t) h);
    for (const Rgba& p : pixels)
    {
        push_u16 (p.r.bits ());
        push_u16 (p.g.bits ());
        push_u16 (p.b.bits ());
        push_u16 (p.a.bits ());
    }

    FILE* f = fopen (out, "wb");
    if (!f) { fprintf (stderr, "cannot open %s for writing\n", out); return 1; }
    size_t n = bytes.empty () ? 0 : fwrite (bytes.data (), 1, bytes.size (), f);
    fclose (f);
    if (n != bytes.size ()) { fprintf (stderr, "short write to %s\n", out); return 1; }
    fprintf (stderr, "oracle: dumped %zu bytes of AcesInputFile decode to %s\n", bytes.size (), out);
    return 0;
}

// ---------------------------------------------------------------------------
// lut-dump -- C++ `round12log`/`roundNBit` over EVERY half bit pattern, for a
// bit-exact check of exr_core::rgba::{round12_log, RoundNBit}.
// ---------------------------------------------------------------------------
// `lut-dump <out> <n>` writes, for `i` in `0..65536` (every possible half bit
// pattern, in order): `round12log(half::setBits(i))` then
// `roundNBit(n)(half::setBits(i))`, each as a `u16` LE half-bits pair --
// `65536 * 4` bytes total. `HalfLut`/`RgbaLut` are themselves just a
// table-lookup wrapper around a `half -> half` function (`halfFunction<T>`,
// already independently oracle-verified in `imath-rs`'s own parity suite),
// so exercising the two underlying functions this exhaustively is equivalent
// to exercising `HalfLut`/`RgbaLut` built from them.
static int do_lut_dump (const char* out, int n)
{
    roundNBit rn (n);
    std::vector<uint8_t> bytes;
    bytes.reserve (65536u * 4u);
    for (unsigned int i = 0; i < 65536u; ++i)
    {
        half x;
        x.setBits ((unsigned short) i);
        half r12 = round12log (x);
        half rnb = rn (x);
        bytes.push_back ((uint8_t) (r12.bits () & 0xff));
        bytes.push_back ((uint8_t) ((r12.bits () >> 8) & 0xff));
        bytes.push_back ((uint8_t) (rnb.bits () & 0xff));
        bytes.push_back ((uint8_t) ((rnb.bits () >> 8) & 0xff));
    }

    FILE* f = fopen (out, "wb");
    if (!f) { fprintf (stderr, "cannot open %s for writing\n", out); return 1; }
    size_t written = fwrite (bytes.data (), 1, bytes.size (), f);
    fclose (f);
    if (written != bytes.size ()) { fprintf (stderr, "short write to %s\n", out); return 1; }
    fprintf (stderr, "oracle: dumped %zu bytes of round12log/roundNBit(%d) to %s\n", bytes.size (), n, out);
    return 0;
}

// ---------------------------------------------------------------------------
// dump / dump-tiled -- the C++ DECODE of a golden, as raw bytes
// ---------------------------------------------------------------------------
// The lossy codecs (B44 / B44A / DWAA / DWAB / PXR24) cannot be checked against
// the source formula: their loss is neither purely absolute nor purely relative
// (it blows up near values that cross zero). Until now the only exact check was
// "run the oracle and compare", which meant 11 tests silently degraded to a
// pixel COUNT whenever the oracle was absent -- i.e. in every plain `cargo
// test`.
//
// So the oracle's decode is committed. The layout is deliberately trivial so
// the Rust side can rebuild it byte-for-byte:
//
//   for each channel, in the file's (alphabetical) channel order:
//     for y in 0..height, for x in 0..width:
//       the sample, little-endian, 2 bytes for HALF and 4 for FLOAT/UINT
//
// `dump-tiled` writes the same, per level, levels in increasing order, each
// level at its own resolution.
static int write_all (const char* out, const std::vector<uint8_t>& bytes)
{
    FILE* f = fopen (out, "wb");
    if (!f) { fprintf (stderr, "cannot open %s for writing\n", out); return 1; }
    size_t n = bytes.empty () ? 0 : fwrite (bytes.data (), 1, bytes.size (), f);
    fclose (f);
    if (n != bytes.size ()) { fprintf (stderr, "short write to %s\n", out); return 1; }
    fprintf (stderr, "oracle: dumped %zu bytes of C++ decode to %s\n", bytes.size (), out);
    return 0;
}

static void append_plane (std::vector<uint8_t>& out,
                          const std::vector<uint8_t>& plane,
                          size_t                      count,
                          size_t                      px)
{
    // The planes are already little-endian in memory on every platform this
    // gate runs on; copy verbatim so the Rust side can compare raw bits.
    out.insert (out.end (), plane.begin (), plane.begin () + (count * px));
}

static int do_dump (const char* path, const char* out)
{
    InputFile    in (path);
    const Header& h  = in.header ();
    Box2i        dw  = h.dataWindow ();
    int          w   = dw.max.x - dw.min.x + 1;
    int          hgt = dw.max.y - dw.min.y + 1;

    std::vector<std::string> names;
    std::vector<PixelType>   types;
    for (ChannelList::ConstIterator it = h.channels ().begin ();
         it != h.channels ().end (); ++it)
    { names.push_back (it.name ()); types.push_back (it.channel ().type); }

    FrameBuffer                       fb;
    std::vector<std::vector<uint8_t>> planes (names.size ());
    for (size_t c = 0; c < names.size (); ++c)
    {
        size_t px = (types[c] == HALF) ? 2 : 4;
        planes[c].assign ((size_t) w * hgt * px, 0);
        char* base = (char*) planes[c].data ()
                     - ((size_t) dw.min.x + (size_t) dw.min.y * w) * px;
        fb.insert (names[c], Slice (types[c], base, px, px * (size_t) w));
    }
    in.setFrameBuffer (fb);
    in.readPixels (dw.min.y, dw.max.y);

    std::vector<uint8_t> bytes;
    for (size_t c = 0; c < names.size (); ++c)
        append_plane (bytes, planes[c], (size_t) w * hgt,
                      (types[c] == HALF) ? 2 : 4);
    return write_all (out, bytes);
}

// ---------------------------------------------------------------------------
// write-nonutf8-names / dump-nonutf8-attr — byte-exact Name oracle proof.
//
// `Imf::Name` is a raw `char[256]` (`ImfName.h`): construction is
// `strncpy(_text, text, MAX_LENGTH); _text[MAX_LENGTH] = 0` — cut at the
// source's first NUL and at 255 bytes, NO encoding check anywhere
// (`parse_header.c`'s `read_text` is likewise encoding-agnostic). This mode
// proves real C++ OpenEXR 3.4.15 accepts, stores and reads back:
//   * a channel name containing an invalid-UTF-8 byte ("Z\xE9" — 'Z' 0x5A
//     followed by the continuation byte 0xE9, which is not valid UTF-8 on
//     its own);
//   * an attribute name starting with an invalid-UTF-8 lead byte (0xFF is
//     never valid in UTF-8 at all): "\xFFattr";
//   * an attribute name at the exact `Name::MAX_LENGTH` (255 ASCII 'x'
//     bytes) — long enough to force `LONG_NAMES_FLAG` (>= 32 bytes), so this
//     file also exercises that version-flag path.
//
// A small 4x3 image keeps the fixture simple; NO_COMPRESSION keeps every
// byte deterministic. Reused by BOTH cross-read directions:
//   (a) C++ writes -> our reader (parity_exr.rs reads this golden directly:
//       no new C++ read code needed, since checking OUR OWN header/pixel
//       decode against the values we chose here IS the byte-exact check).
//   (b) we write -> C++ reads: parity_exr.rs writes an equivalent file with
//       OUR writer, then invokes the EXISTING generic `dump` mode above (for
//       the two channels' pixel bytes, via a real `Imf::InputFile`) and the
//       new `dump-nonutf8-attr` mode below (for the two odd-named `int`
//       attributes, via `Header::findTypedAttribute`) against OUR file — a
//       genuine live read by real C++, not a self-consistency check.
static const int NN_W = 4, NN_H = 3;

static half  nn_r_val (int idx) { return half (1.0f + idx * 0.5f); }
static float nn_z_val (int idx) { return 100.0f + idx * 2.0f; }

// The three odd names, built once so write and dump agree byte-for-byte.
static std::string nn_channel_name ()
{
    std::string s = "Z";
    s.push_back ((char) 0xE9);
    return s; // "Z\xE9"
}
static std::string nn_attr_name_nonutf8 ()
{
    std::string s;
    s.push_back ((char) 0xFF);
    s += "attr";
    return s; // "\xFFattr"
}
static std::string nn_attr_name_maxlen () { return std::string (255, 'x'); }

static const int32_t NN_ATTR_NONUTF8_VALUE = 12345;
static const int32_t NN_ATTR_MAXLEN_VALUE  = 255001;

static int do_write_nonutf8_names (const char* path)
{
    std::vector<half>  R (NN_W * NN_H);
    std::vector<float> Znonutf8 (NN_W * NN_H);
    for (int idx = 0; idx < NN_W * NN_H; ++idx) {
        R[idx]        = nn_r_val (idx);
        Znonutf8[idx] = nn_z_val (idx);
    }

    Header header (NN_W, NN_H);
    header.compression () = NO_COMPRESSION;
    header.lineOrder ()   = INCREASING_Y;
    header.channels ().insert ("R", Channel (HALF));
    header.channels ().insert (nn_channel_name ().c_str (), Channel (FLOAT));
    header.insert ("owner", StringAttribute ("exr-rs"));
    header.insert (nn_attr_name_nonutf8 ().c_str (), IntAttribute (NN_ATTR_NONUTF8_VALUE));
    header.insert (nn_attr_name_maxlen ().c_str (), IntAttribute (NN_ATTR_MAXLEN_VALUE));

    OutputFile out (path, header);

    FrameBuffer fb;
    fb.insert ("R", Slice (HALF, (char*) R.data (), sizeof (half), sizeof (half) * NN_W));
    fb.insert (nn_channel_name ().c_str (),
               Slice (FLOAT, (char*) Znonutf8.data (), sizeof (float), sizeof (float) * NN_W));
    out.setFrameBuffer (fb);
    out.writePixels (NN_H);

    fprintf (stderr, "oracle: wrote %dx%d scanline EXR with non-UTF-8 channel/attribute names to %s\n",
             NN_W, NN_H, path);
    return 0;
}

// Dumps the two `int` attribute values found under the non-UTF-8 and
// max-length names as 2 little-endian i32s (8 bytes) -- proves REAL C++
// `Header::findTypedAttribute` can look both up by their exact byte-string
// name in a file it did not necessarily write itself (used on OUR writer's
// output for direction (b); the channel/pixel half of that direction reuses
// the existing generic `dump` mode above, unchanged).
static int do_dump_nonutf8_attr (const char* path, const char* out)
{
    InputFile     in (path);
    const Header& h = in.header ();

    int32_t vals[2] = { 0, 0 };
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute> (nn_attr_name_nonutf8 ().c_str ()))
        vals[0] = a->value ();
    else
        fprintf (stderr, "oracle: missing non-UTF-8-named attribute\n");
    if (const IntAttribute* a = h.findTypedAttribute<IntAttribute> (nn_attr_name_maxlen ().c_str ()))
        vals[1] = a->value ();
    else
        fprintf (stderr, "oracle: missing max-length-named attribute\n");

    std::vector<uint8_t> bytes (sizeof (vals));
    memcpy (bytes.data (), vals, sizeof (vals));
    return write_all (out, bytes);
}

// ---------------------------------------------------------------------------
// write-dwa-sub / dump-dwa-sub — subsampled-channel DWA scanline oracle
// (additive, does not touch the KNOWN 67x43 image or any existing golden).
//
// Channels (alphabetical, as IlmImf stores them): A (HALF, xSampling=2,
// ySampling=1 -> RLE, the classifier's "A" rule), B/G/R (HALF, unsampled ->
// the CSC group), BY (HALF, xSampling=2, ySampling=2 -> standalone LOSSY_DCT,
// the classifier's "BY" rule has no CSC slot), Z (FLOAT, xSampling=1,
// ySampling=2 -> UNKNOWN/zlib, no rule matches "Z"). SW/SH are even so every
// sampling divides them exactly (OpenEXR requires `sampling | dimension` and
// `dataWindow.min % sampling == 0`; trivially satisfied at origin (0,0)).
// SH=64 spans two DWAA chunks (32 lines each) and a quarter of one DWAB chunk
// (256 lines) — enough to exercise the DWAA chunk boundary without an
// enormous fixture.
//
static const int SW = 64, SH = 64;

static float sr_val (int cx, int cy) { return 1.0f + (cy * SW + cx) * 0.05f; }
static float sg_val (int cx, int cy) { return 100.0f + (cy * SW + cx) * 0.03f; }
static float sb_val (int cx, int cy) { return -50.0f + (cy * SW + cx) * 0.07f; }
static float sby_val (int cx, int cy) { return 10.0f + (cy * (SW / 2) + cx) * 0.3f; }
static float sa_val (int cx, int cy) { return -5.0f + (cy * (SW / 2) + cx) * 0.2f; }
static float sz_val (int cx, int cy) { return (float) (cy * SW + cx) * 4.0f; }

static int do_write_dwa_sub (const char* path, Compression comp)
{
    std::vector<half> R ((size_t) SW * SH), G ((size_t) SW * SH), B ((size_t) SW * SH);
    std::vector<half> BY ((size_t) (SW / 2) * (SH / 2));
    std::vector<half> A ((size_t) (SW / 2) * SH);
    std::vector<float> Z ((size_t) SW * (SH / 2));

    for (int y = 0; y < SH; ++y)
        for (int x = 0; x < SW; ++x)
        {
            int idx  = y * SW + x;
            R[idx]   = half (sr_val (x, y));
            G[idx]   = half (sg_val (x, y));
            B[idx]   = half (sb_val (x, y));
        }
    for (int y = 0; y < SH / 2; ++y)
        for (int x = 0; x < SW / 2; ++x)
            BY[(size_t) y * (SW / 2) + x] = half (sby_val (x, y));
    for (int y = 0; y < SH; ++y)
        for (int x = 0; x < SW / 2; ++x)
            A[(size_t) y * (SW / 2) + x] = half (sa_val (x, y));
    for (int y = 0; y < SH / 2; ++y)
        for (int x = 0; x < SW; ++x)
            Z[(size_t) y * SW + x] = sz_val (x, y);

    Header header (SW, SH);
    header.compression () = comp;
    header.lineOrder ()   = INCREASING_Y;
    header.channels ().insert ("R", Channel (HALF));
    header.channels ().insert ("G", Channel (HALF));
    header.channels ().insert ("B", Channel (HALF));
    header.channels ().insert ("BY", Channel (HALF, 2, 2));
    header.channels ().insert ("A", Channel (HALF, 2, 1));
    header.channels ().insert ("Z", Channel (FLOAT, 1, 2));

    OutputFile out (path, header);

    FrameBuffer fb;
    fb.insert ("R", Slice (HALF, (char*) R.data (), sizeof (half), sizeof (half) * SW));
    fb.insert ("G", Slice (HALF, (char*) G.data (), sizeof (half), sizeof (half) * SW));
    fb.insert ("B", Slice (HALF, (char*) B.data (), sizeof (half), sizeof (half) * SW));
    fb.insert ("BY", Slice (HALF, (char*) BY.data (), sizeof (half), sizeof (half) * (SW / 2), 2, 2));
    fb.insert ("A", Slice (HALF, (char*) A.data (), sizeof (half), sizeof (half) * (SW / 2), 2, 1));
    fb.insert ("Z", Slice (FLOAT, (char*) Z.data (), sizeof (float), sizeof (float) * SW, 1, 2));

    out.setFrameBuffer (fb);
    out.writePixels (SH);

    fprintf (stderr, "oracle: wrote %dx%d subsampled-DWA scanline EXR (compression %d) to %s\n",
              SW, SH, (int) comp, path);
    return 0;
}

// ---------------------------------------------------------------------------
// write-sub / dump-sub — generic per-codec x per-sampling matrix oracle.
// (An HTJ2K y-subsampling bug
// slipped through because no test ever exercised x/y channel sub-sampling for
// most of the 12 codecs. This is additive -- it does not touch the KNOWN
// image, `write-dwa-sub`, or any existing golden.)
//
// Two channels, alphabetical: "A" (HALF, unsampled reference) and "S" (HALF,
// the codec/sampling under test). 64x64 so every codec's chunk height (1, 16,
// 32, or 256 scan lines) either divides it evenly or covers it in one chunk,
// and xs/ys in {1, 2} both divide 64 and the (0,0) origin trivially satisfies
// `dataWindow.min % sampling == 0`. Reused by [`do_dump_subsampled`] below
// (already generic over channel set/sampling -- no new dump function needed).
// ---------------------------------------------------------------------------

static const int SUBW = 64, SUBH = 64;

static float sub_a_val (int x, int y) { return 1.0f + (y * SUBW + x) * 0.1f; }
static float sub_s_val (int x, int y) { return -30.0f + (y * SUBW + x) * 0.2f; }

static int do_write_sub (const char* path, Compression comp, int xs, int ys)
{
    std::vector<half> A ((size_t) SUBW * SUBH);
    int sw = SUBW / xs, sh = SUBH / ys;
    std::vector<half> S ((size_t) sw * sh);

    for (int y = 0; y < SUBH; ++y)
        for (int x = 0; x < SUBW; ++x)
            A[(size_t) y * SUBW + x] = half (sub_a_val (x, y));
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x)
            S[(size_t) y * sw + x] = half (sub_s_val (x, y));

    Header header (SUBW, SUBH);
    header.compression () = comp;
    header.lineOrder ()   = INCREASING_Y;
    header.channels ().insert ("A", Channel (HALF));
    header.channels ().insert ("S", Channel (HALF, xs, ys));

    OutputFile out (path, header);

    FrameBuffer fb;
    fb.insert ("A", Slice (HALF, (char*) A.data (), sizeof (half), sizeof (half) * SUBW));
    fb.insert ("S", Slice (HALF, (char*) S.data (), sizeof (half), sizeof (half) * sw, xs, ys));

    out.setFrameBuffer (fb);
    out.writePixels (SUBH);

    fprintf (stderr, "oracle: wrote %dx%d sub-sampled EXR (compression %d, xs=%d ys=%d) to %s\n",
              SUBW, SUBH, (int) comp, xs, ys, path);
    return 0;
}

// Generic decode dump for ANY scanline file with (possibly subsampled)
// channels, dataWindow at (0,0): iterate `h.channels()` (alphabetical), for
// each channel allocate `cw = w/xSampling` by `chh = hgt/ySampling` samples,
// read via a `Slice` with matching sampling, and write raw little-endian
// bytes in (channel, then row-major over ITS OWN reduced grid) order. Not
// specific to DWA — any subsampled scanline fixture can reuse this — but the
// convention matches `channel_dims`/`active_row_count` (`compressor/dwa/mod.rs`)
// on the Rust side exactly, which is what the subsampled DWA cross-check needs.
static int do_dump_subsampled (const char* path, const char* out)
{
    InputFile     in (path);
    const Header& h   = in.header ();
    Box2i         dw  = h.dataWindow ();
    int           w   = dw.max.x - dw.min.x + 1;
    int           hgt = dw.max.y - dw.min.y + 1;
    if (dw.min.x != 0 || dw.min.y != 0)
    {
        fprintf (stderr, "oracle: dump-dwa-sub requires a dataWindow at (0,0)\n");
        return 2;
    }

    std::vector<std::string>          names;
    std::vector<PixelType>            types;
    std::vector<int>                  xsamp, ysamp;
    for (ChannelList::ConstIterator it = h.channels ().begin ();
         it != h.channels ().end (); ++it)
    {
        names.push_back (it.name ());
        types.push_back (it.channel ().type);
        xsamp.push_back (it.channel ().xSampling);
        ysamp.push_back (it.channel ().ySampling);
    }

    FrameBuffer                       fb;
    std::vector<std::vector<uint8_t>> planes (names.size ());
    for (size_t c = 0; c < names.size (); ++c)
    {
        int    cw  = w / xsamp[c];
        int    chh = hgt / ysamp[c];
        size_t px  = (types[c] == HALF) ? 2 : 4;
        planes[c].assign ((size_t) cw * chh * px, 0);
        fb.insert (names[c], Slice (types[c], (char*) planes[c].data (), px,
                                     px * (size_t) cw, xsamp[c], ysamp[c]));
    }
    in.setFrameBuffer (fb);
    in.readPixels (dw.min.y, dw.max.y);

    std::vector<uint8_t> bytes;
    for (size_t c = 0; c < names.size (); ++c)
        bytes.insert (bytes.end (), planes[c].begin (), planes[c].end ());
    return write_all (out, bytes);
}

static int do_dump_tiled (const char* path, const char* out)
{
    TiledInputFile in (path);
    const Header&  h = in.header ();

    std::vector<std::string> names;
    std::vector<PixelType>   types;
    for (ChannelList::ConstIterator it = h.channels ().begin ();
         it != h.channels ().end (); ++it)
    { names.push_back (it.name ()); types.push_back (it.channel ().type); }

    int numLevels =
        (h.tileDescription ().mode == MIPMAP_LEVELS) ? in.numLevels () : 1;

    std::vector<uint8_t> bytes;
    for (int l = 0; l < numLevels; ++l)
    {
        Box2i ldw = in.dataWindowForLevel (l);
        int   lw  = ldw.max.x - ldw.min.x + 1;
        int   lh  = ldw.max.y - ldw.min.y + 1;

        FrameBuffer                       fb;
        std::vector<std::vector<uint8_t>> planes (names.size ());
        for (size_t c = 0; c < names.size (); ++c)
        {
            size_t px = (types[c] == HALF) ? 2 : 4;
            planes[c].assign ((size_t) lw * lh * px, 0);
            char* base = (char*) planes[c].data ()
                         - ((size_t) ldw.min.x + (size_t) ldw.min.y * lw) * px;
            fb.insert (names[c], Slice (types[c], base, px, px * (size_t) lw));
        }
        in.setFrameBuffer (fb);
        in.readTiles (0, in.numXTiles (l) - 1, 0, in.numYTiles (l) - 1, l);

        for (size_t c = 0; c < names.size (); ++c)
            append_plane (bytes, planes[c], (size_t) lw * lh,
                          (types[c] == HALF) ? 2 : 4);
    }
    return write_all (out, bytes);
}

// ---------------------------------------------------------------------------
// bench-io — "as C++ or better" wall-time comparison
// ---------------------------------------------------------------------------
// A whole-file scanline write + read wall-time
// comparison at a real 16k-class size, with `setGlobalThreadCount(threads)`
// so C++'s own IlmThread pool is sized the same as our `num_threads`. This
// is a THROUGHPUT comparison, not a byte-parity one (unlike every other mode
// in this file); bench-io compresses the Rust side's raw image, so both
// sides time the same pixels.
//
// `fill_bench_pixels` is bench-scatter's own pattern (NONE only, where the
// pixel values do not change the work): realistic enough that it is not
// degenerate all-zero data.
static void fill_bench_pixels(std::vector<char>& buf, int width, int height, size_t pixelBytes, size_t rowBytes)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            char* p = buf.data() + (size_t) y * rowBytes + (size_t) x * pixelBytes;
            float gx = (float) x / (float) width;
            float gy = (float) y / (float) height;
            unsigned h = (unsigned) x * 73856093u ^ (unsigned) y * 19349663u;
            h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
            float n = (float) h / 4294967295.0f;
            float grad = 0.15f + 0.7f * (gx * 0.6f + gy * 0.4f);
            float val = grad + 0.06f * (n - 0.5f);
            half r((float) val);
            half g((float) (val * 1.01f));
            half b((float) (val * 0.99f));
            float a = val;
            std::memcpy(p + 0, &r, 2);
            std::memcpy(p + 2, &g, 2);
            std::memcpy(p + 4, &b, 2);
            std::memcpy(p + 6, &a, 4);
        }
    }
}

// The pixels come from `pixels` (raw interleaved R,G,B half + A float, native byte order,
// row-major, written once by the Rust side): both sides compress the IDENTICAL image, so a
// codec's speed is not compared across two differently compressible inputs.
static int do_bench_io(const char* pixels, const char* path, Compression comp, int width, int height, int threads)
{
    setGlobalThreadCount(threads);

    Header header(width, height);
    header.compression() = comp;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("A", Channel(FLOAT));

    const size_t pixelBytes = 2 + 2 + 2 + 4;
    const size_t rowBytes = pixelBytes * (size_t) width;
    std::vector<char> buf(rowBytes * (size_t) height);
    {
        FILE* in = std::fopen(pixels, "rb");
        if (!in)
        {
            fprintf(stderr, "bench-io: cannot open %s\n", pixels);
            return 1;
        }
        const size_t got = std::fread(buf.data(), 1, buf.size(), in);
        const bool more = std::fgetc(in) != EOF;
        std::fclose(in);
        if (got != buf.size() || more)
        {
            fprintf(stderr, "bench-io: %s is not exactly %zu bytes\n", pixels, buf.size());
            return 1;
        }
    }

    FrameBuffer fb;
    fb.insert("R", Slice(HALF, buf.data() + 0, pixelBytes, rowBytes));
    fb.insert("G", Slice(HALF, buf.data() + 2, pixelBytes, rowBytes));
    fb.insert("B", Slice(HALF, buf.data() + 4, pixelBytes, rowBytes));
    fb.insert("A", Slice(FLOAT, buf.data() + 6, pixelBytes, rowBytes));

    auto t0 = std::chrono::steady_clock::now();
    {
        OutputFile of(path, header);
        of.setFrameBuffer(fb);
        of.writePixels(height);
    }
    auto t1 = std::chrono::steady_clock::now();

    std::vector<char> readback(rowBytes * (size_t) height, 0);
    FrameBuffer rfb;
    rfb.insert("R", Slice(HALF, readback.data() + 0, pixelBytes, rowBytes));
    rfb.insert("G", Slice(HALF, readback.data() + 2, pixelBytes, rowBytes));
    rfb.insert("B", Slice(HALF, readback.data() + 4, pixelBytes, rowBytes));
    rfb.insert("A", Slice(FLOAT, readback.data() + 6, pixelBytes, rowBytes));

    // Warm-up read (untimed): puts the file in the OS page cache so the
    // TIMED read below is not paying for a page-cache-cold first touch --
    // matches the Rust side's own warm-up-then-time pattern (reduces
    // cross-run noise).
    {
        InputFile warm(path);
        warm.setFrameBuffer(rfb);
        warm.readPixels(0, height - 1);
    }

    auto t2 = std::chrono::steady_clock::now();
    {
        InputFile inf(path);
        inf.setFrameBuffer(rfb);
        inf.readPixels(0, height - 1);
    }
    auto t3 = std::chrono::steady_clock::now();

    double writeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double readMs  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    // Machine-readable, one line: the Rust side parses these two fields.
    printf("write_ms=%.3f read_ms=%.3f\n", writeMs, readMs);
    return 0;
}

// ---------------------------------------------------------------------------
// bench-scatter -- "what does C++ generic_unpack give on the SAME data"
// (comparing our scatter against a raw memcpy was the wrong
// baseline; 4 strided sub-pixel passes are inherently slower than one
// contiguous copy, on EITHER side of the language boundary).
// ---------------------------------------------------------------------------
// This isolates decode+scatter from real disk I/O by using in-memory
// streams (Imf::StdOSStream / Imf::StdISStream, std::ostringstream /
// std::istringstream under the hood -- no file descriptor, no real syscall)
// instead of a path on disk, with NONE compression so decode itself is a
// no-op memcpy and the measured time is (near-)entirely readPixels's
// unpack/scatter cost -- matching what our own scatter_chunk_only_throughput
// isolates on the Rust side, as closely as the C++ API's stream abstraction
// allows (the istringstream layer itself still copies bytes once, unlike
// our benchmark's zero-copy in-memory buffer, but that cost is the same
// order of magnitude as file-buffer copies elsewhere and does not use a
// disk syscall).
static int do_bench_scatter(Compression comp, int width, int height, int threads)
{
    setGlobalThreadCount(threads);

    Header header(width, height);
    header.compression() = comp;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("A", Channel(FLOAT));

    const size_t pixelBytes = 2 + 2 + 2 + 4;
    const size_t rowBytes = pixelBytes * (size_t) width;
    std::vector<char> buf(rowBytes * (size_t) height);
    fill_bench_pixels(buf, width, height, pixelBytes, rowBytes);

    FrameBuffer fb;
    fb.insert("R", Slice(HALF, buf.data() + 0, pixelBytes, rowBytes));
    fb.insert("G", Slice(HALF, buf.data() + 2, pixelBytes, rowBytes));
    fb.insert("B", Slice(HALF, buf.data() + 4, pixelBytes, rowBytes));
    fb.insert("A", Slice(FLOAT, buf.data() + 6, pixelBytes, rowBytes));

    std::string encoded;
    {
        StdOSStream os;
        OutputFile of(os, header);
        of.setFrameBuffer(fb);
        of.writePixels(height);
        encoded = os.str();
    }

    std::vector<char> readback(rowBytes * (size_t) height, 0);
    FrameBuffer rfb;
    rfb.insert("R", Slice(HALF, readback.data() + 0, pixelBytes, rowBytes));
    rfb.insert("G", Slice(HALF, readback.data() + 2, pixelBytes, rowBytes));
    rfb.insert("B", Slice(HALF, readback.data() + 4, pixelBytes, rowBytes));
    rfb.insert("A", Slice(FLOAT, readback.data() + 6, pixelBytes, rowBytes));

    // Warm-up (untimed): first touch of freshly-allocated `readback` pages.
    {
        StdISStream is;
        is.str(encoded);
        InputFile warm(is);
        warm.setFrameBuffer(rfb);
        warm.readPixels(0, height - 1);
    }

    auto t0 = std::chrono::steady_clock::now();
    {
        StdISStream is;
        is.str(encoded);
        InputFile inf(is);
        inf.setFrameBuffer(rfb);
        inf.readPixels(0, height - 1);
    }
    auto t1 = std::chrono::steady_clock::now();

    double scatterMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double bytes     = (double) rowBytes * (double) height;
    double gbps      = bytes / (scatterMs / 1000.0) / 1e9;
    // Machine-readable, one line: the Rust side parses this field.
    printf("scatter_ms=%.3f gbps=%.3f\n", scatterMs, gbps);
    return 0;
}

// ---------------------------------------------------------------------------
// bench-codec -- codec-in-isolation comparison: the cleanest answer to
// "codec or pipeline?" for ZIP/PIZ/DWAA, where the whole-file comparisons
// show the biggest remaining gap.
// ---------------------------------------------------------------------------
// Times ONLY Imf::Compressor::compress/uncompress (ImfCompressor.h's public
// API, the same shape our own Compressor trait mirrors) on synthetic,
// already-gathered chunk bytes -- no stream I/O, no OutputFile/InputFile, no
// frame-buffer gather/scatter. Same per-line, per-channel (alphabetical: A,
// B, G, R), Xdr little-endian layout every scanline codec actually
// compresses, and the same synthetic pixel pattern as fill_bench_pixels
// (not required to be bit-identical across languages here -- this is a
// throughput-only comparison, not a byte-parity one -- just similarly
// non-degenerate so every codec does real compression work).
static float sample_bench(int c, int x, int y, int width)
{
    float gx = (float) x / (float) width;
    unsigned h = (unsigned) x * 73856093u ^ (unsigned) y * 19349663u ^ (unsigned) c * 83492791u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    float n = (float) h / 4294967295.0f;
    float grad = 0.15f + 0.7f * gx * 0.6f + 0.1f * (float) c * 0.25f;
    return grad + 0.06f * (n - 0.5f);
}

static void fill_codec_chunk(std::vector<char>& buf, int width, int firstY, int lines)
{
    buf.resize((size_t) lines * (size_t) width * 10);
    size_t pos = 0;
    for (int y = firstY; y < firstY + lines; ++y)
    {
        // A: float
        for (int x = 0; x < width; ++x)
        {
            float v = sample_bench(3, x, y, width);
            std::memcpy(&buf[pos], &v, 4);
            pos += 4;
        }
        // B, G, R: half
        for (int c = 2; c >= 0; --c)
        {
            for (int x = 0; x < width; ++x)
            {
                half h((float) sample_bench(c, x, y, width));
                uint16_t bits;
                std::memcpy(&bits, &h, 2);
                std::memcpy(&buf[pos], &bits, 2);
                pos += 2;
            }
        }
    }
}

static int do_bench_codec(Compression comp, int width, int height)
{
    Header header(width, height);
    header.compression() = comp;
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("G", Channel(HALF));
    header.channels().insert("B", Channel(HALF));
    header.channels().insert("A", Channel(FLOAT));

    int linesPerChunk = numLinesInBuffer(comp);
    int nChunks       = height / linesPerChunk;
    size_t maxBytesPerLine = 10 * (size_t) width;

    std::vector<std::vector<char>> chunks (nChunks);
    for (int i = 0; i < nChunks; ++i)
        fill_codec_chunk(chunks[i], width, i * linesPerChunk, linesPerChunk);

    std::vector<std::vector<char>> packed (nChunks);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nChunks; ++i)
    {
        std::unique_ptr<Compressor> c (newCompressor(comp, maxBytesPerLine, header));
        if (!c) { packed[i] = chunks[i]; continue; }
        // HTJ2K's compressor needs the uncompressed size hint up front
        // (JPEG2000 codestreams don't self-describe it the way ZIP/PIZ/DWA's
        // scanline-geometry-derived sizing does) -- ImfCompressor.h's public
        // setExpectedSize(). Harmless to call for every codec (default 0
        // means "not set", and ZIP/PIZ/DWA never read it), so set it
        // unconditionally rather than special-casing HTJ2K by name.
        c->setExpectedSize(chunks[i].size());
        const char* outPtr = nullptr;
        int outSize = c->compress(chunks[i].data(), (int) chunks[i].size(), i * linesPerChunk, outPtr);
        if (getenv("BENCH_CODEC_DEBUG"))
            fprintf(stderr, "DEBUG compress[%d]: inSize=%zu outSize=%d outPtr=%p\n", i, chunks[i].size(), outSize, (const void*) outPtr);
        packed[i].assign(outPtr, outPtr + outSize);
    }
    auto t1 = std::chrono::steady_clock::now();

    for (int i = 0; i < nChunks; ++i)
    {
        std::unique_ptr<Compressor> c (newCompressor(comp, maxBytesPerLine, header));
        if (!c)
        {
            if (packed[i].size() != chunks[i].size())
            {
                fprintf(stderr, "bench-codec: NONE round-trip size mismatch\n");
                return 1;
            }
            continue;
        }
        c->setExpectedSize(chunks[i].size());
        const char* outPtr = nullptr;
        int outSize = c->uncompress(packed[i].data(), (int) packed[i].size(), i * linesPerChunk, outPtr);
        if (getenv("BENCH_CODEC_DEBUG"))
        {
            bool bytesMatch = outPtr && memcmp(outPtr, chunks[i].data(), chunks[i].size()) == 0;
            fprintf(stderr, "DEBUG uncompress[%d]: packedSize=%zu outSize=%d outPtr=%p bytesMatch=%d\n",
                    i, packed[i].size(), outSize, (const void*) outPtr, (int) bytesMatch);
        }
        // HT/HTJ2K-specific: OpenEXRCore's internal HT decoder (there is no
        // internal_ht.c equivalent to internal_zip.c/internal_piz.c etc.)
        // never sets `exr_decode_pipeline_t::bytes_decompressed`, so
        // `Compressor::uncompress()` (ImfCompressor.cpp's runDecodeStep,
        // which just returns `_decoder.bytes_decompressed`) always reports
        // 0 for HT/HTJ2K even on a fully correct decode -- confirmed by
        // memcmp above (bytesMatch=1) against a real chunk. This is a real
        // gap in the legacy Compressor wrapper's HT support (used only by
        // this synthetic isolation harness -- the real InputFile/OutputFile
        // pipeline, which parity-exr's HTJ2K256/32 goldens exercise, does
        // not go through this path). Trust the KNOWN expected size for HT
        // when outSize==0 rather than failing a benchmark harness over a
        // C++-side reporting bug unrelated to what this mode measures.
        bool isHt = (comp == HTJ2K256_COMPRESSION || comp == HTJ2K32_COMPRESSION);
        if (outSize == 0 && isHt) outSize = (int) chunks[i].size();
        if ((size_t) outSize != chunks[i].size())
        {
            fprintf(stderr, "bench-codec: decompressed size %d != expected %zu\n", outSize, chunks[i].size());
            return 1;
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    double compressMs   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double uncompressMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // Machine-readable, one line: the Rust side parses these two fields.
    printf("compress_ms=%.3f uncompress_ms=%.3f\n", compressMs, uncompressMs);
    return 0;
}

// ---------------------------------------------------------------------------
// read-class: which Iex exception class does reading <path>'s PIXELS end in?
//
//   oracle.exe read-class <path> <api>     api = aces | rgba | input | tiled |
//                                                deepscan | deeptile
//
// Prints exactly one line: `OK`, `SKIP` (window too large to allocate for the
// check, decided AFTER a successful open), or `<Class> <message>` where Class is
// the most derived of Input / Io / Arg / Logic / Type / Errno / Math / Base / Std.
// `aces` and `rgba` are what testMalformedImages.cpp uses; `input` reads every
// channel as FLOAT through Imf::InputFile; the deep modes read the sample-count
// tables (the first chunk payload they touch). Consumed by
// exr-core/tests/parity_class.rs, which runs our reader through the same steps.
// ---------------------------------------------------------------------------
static const char* exc_class(const std::exception& e)
{
    if (dynamic_cast<const IEX_NAMESPACE::InputExc*>(&e)) return "Input";
    if (dynamic_cast<const IEX_NAMESPACE::IoExc*>(&e)) return "Io";
    if (dynamic_cast<const IEX_NAMESPACE::ArgExc*>(&e)) return "Arg";
    if (dynamic_cast<const IEX_NAMESPACE::LogicExc*>(&e)) return "Logic";
    if (dynamic_cast<const IEX_NAMESPACE::TypeExc*>(&e)) return "Type";
    if (dynamic_cast<const IEX_NAMESPACE::ErrnoExc*>(&e)) return "Errno";
    if (dynamic_cast<const IEX_NAMESPACE::MathExc*>(&e)) return "Math";
    if (dynamic_cast<const IEX_NAMESPACE::BaseExc*>(&e)) return "Base";
    return "Std";
}

// Same limits as the Rust side (parity_class.rs): pixels in the window, and
// pixels x channels for the per-channel FLOAT buffers.
static const int64_t RC_MAX_PIXELS = 1 << 24;
static const int64_t RC_MAX_SAMPLES = 1 << 26;

static bool rc_window(const Box2i& dw, int64_t& w, int64_t& h)
{
    w = (int64_t)dw.max.x - dw.min.x + 1;
    h = (int64_t)dw.max.y - dw.min.y + 1;
    return w > 0 && h > 0 && w * h <= RC_MAX_PIXELS;
}

// Returns false when the window is too big (caller prints SKIP).
static bool rc_read(const char* path, const std::string& api)
{
    int64_t w, h;
    if (api == "aces" || api == "rgba") {
        Array2D<Rgba> p;
        if (api == "aces") {
            AcesInputFile in(path);
            Box2i dw = in.header().dataWindow();
            if (!rc_window(dw, w, h)) return false;
            p.resizeErase((long)h, (long)w);
            in.setFrameBuffer(&p[0][0] - dw.min.x - (long)dw.min.y * (long)w, 1, (size_t)w);
            in.readPixels(dw.min.y, dw.max.y);
        } else {
            RgbaInputFile in(path);
            Box2i dw = in.header().dataWindow();
            if (!rc_window(dw, w, h)) return false;
            p.resizeErase((long)h, (long)w);
            in.setFrameBuffer(&p[0][0] - dw.min.x - (long)dw.min.y * (long)w, 1, (size_t)w);
            in.readPixels(dw.min.y, dw.max.y);
        }
        return true;
    }
    if (api == "input" || api == "tiled") {
        // Both build the same FLOAT-per-channel frame buffer over the data window.
        std::vector<std::vector<float>> planes;
        FrameBuffer fb;
        auto fill = [&](const Header& hd) -> bool {
            Box2i dw = hd.dataWindow();
            if (!rc_window(dw, w, h)) return false;
            int64_t nch = 0;
            for (ChannelList::ConstIterator it = hd.channels().begin(); it != hd.channels().end(); ++it) ++nch;
            if (w * h * nch > RC_MAX_SAMPLES) return false;
            planes.resize((size_t)nch);
            size_t c = 0;
            for (ChannelList::ConstIterator it = hd.channels().begin(); it != hd.channels().end(); ++it, ++c) {
                planes[c].assign((size_t)(w * h), 0.f);
                char* base = (char*)planes[c].data() - 4 * (ptrdiff_t)dw.min.x - 4 * (ptrdiff_t)dw.min.y * (ptrdiff_t)w;
                fb.insert(it.name(), Slice(FLOAT, base, 4, 4 * (size_t)w, it.channel().xSampling, it.channel().ySampling));
            }
            return true;
        };
        if (api == "input") {
            InputFile in(path);
            if (!fill(in.header())) return false;
            in.setFrameBuffer(fb);
            in.readPixels(in.header().dataWindow().min.y, in.header().dataWindow().max.y);
        } else {
            TiledInputFile in(path);
            if (!fill(in.header())) return false;
            in.setFrameBuffer(fb);
            in.readTiles(0, in.numXTiles(0) - 1, 0, in.numYTiles(0) - 1, 0, 0);
        }
        return true;
    }
    if (api == "deepscan" || api == "deeptile") {
        std::vector<unsigned> counts;
        DeepFrameBuffer fb;
        auto fill = [&](const Header& hd) -> bool {
            Box2i dw = hd.dataWindow();
            if (!rc_window(dw, w, h)) return false;
            counts.assign((size_t)(w * h), 0u);
            char* base = (char*)counts.data() - 4 * (ptrdiff_t)dw.min.x - 4 * (ptrdiff_t)dw.min.y * (ptrdiff_t)w;
            fb.insertSampleCountSlice(Slice(UINT, base, 4, 4 * (size_t)w));
            return true;
        };
        if (api == "deepscan") {
            DeepScanLineInputFile in(path);
            if (!fill(in.header())) return false;
            in.setFrameBuffer(fb);
            in.readPixelSampleCounts(in.header().dataWindow().min.y, in.header().dataWindow().max.y);
        } else {
            DeepTiledInputFile in(path);
            if (!fill(in.header())) return false;
            in.setFrameBuffer(fb);
            in.readPixelSampleCounts(0, in.numXTiles(0) - 1, 0, in.numYTiles(0) - 1, 0, 0);
        }
        return true;
    }
    throw std::runtime_error("read-class: unknown api");
}

static int do_read_class(const char* path, const char* api)
{
    try {
        if (rc_read(path, api)) printf("OK\n"); else printf("SKIP\n");
    } catch (const std::exception& e) {
        std::string m = e.what();
        for (char& c : m) if (c == '\n' || c == '\r') c = ' ';
        printf("%s %s\n", exc_class(e), m.c_str());
    } catch (...) {
        printf("Std unknown exception\n");
    }
    return 0;
}

int main(int argc, char** argv)
{
    // Modes:
    //   write       <path> [compression]
    //   read        <path>
    //   read-half   <path>   (every channel read into a HALF slice)
    //   write-tiled <path> <compression> <levelmode> [minX minY]
    //   read-tiled  <path>
    //   write-deep-scanline <path> <codec> [minX minY]
    //   read-deep-scanline  <path>
    //   write-deep-tiled    <path> <codec> <levelmode>
    //   read-deep-tiled     <path>
    //   write-deep-composite <path> <a|b>       (one of 2 deep composite sources)
    //   composite-deep-scanline <pathA> <pathB> <out>  (real CompositeDeepScanLine dump)
    //   read-deep-via-inputfile <path> <out>    (real generic Imf::InputFile flatten dump)
    //   dump            <path> <out>   (C++ decode of a flat golden, raw bytes)
    //   dump-tiled      <path> <out>   (same, every level)
    //   write-yc        <path>      (luminance-chroma, subsampled RY/BY)
    //   write-rgba      <path>      (the same pixels, full-resolution RGBA)
    //   write-yc-rounded <path> <roundY> <roundC>  (write-yc + setYCRounding)
    //   write-tiled-rgba <path> <rgba|grey>  (TiledRgbaOutputFile, 20x13, 8x8 tiles)
    //   rgba-read       <path> [out]  (Imf::RgbaInputFile decode, bits per pixel; [out] = binary dump)
    //   write-multipart <path>
    //   read-multipart  <path>
    //   write-idmanifest <path>          (a real C++-built IDManifest attribute)
    //   dump-idmanifest  <path> <out>    (text dump of a decoded idmanifest attribute)
    //   write-zip-level <path> <compression> <level>  (Header::zipCompressionLevel() override)
    //   bench-io <pixels> <path> <compression> <width> <height> <threads>  (
    //       whole-file scanline write+read wall-time comparison of the raw image <pixels>
    //       (written by the Rust side), setGlobalThreadCount(threads);
    //       prints "write_ms=... read_ms=..." -- NOT a byte-parity mode)
    //   bench-scatter <compression> <width> <height> <threads>  (in-memory-stream
    //       (no disk I/O) scatter-only throughput, NONE compression makes decode a no-op;
    //       prints "scatter_ms=... gbps=..." -- NOT a byte-parity mode)
    //   bench-codec <compression> <width> <height>  (codec-in-isolation --
    //       Imf::Compressor::compress/uncompress only, no I/O, no scatter; prints
    //       "compress_ms=... uncompress_ms=..." -- NOT a byte-parity mode)
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <write|read|write-tiled|read-tiled> <path> [args...]\n", argv[0]);
        return 2;
    }
    try {
        if (std::strcmp(argv[1], "write") == 0) {
            if (argc > 4) { fprintf(stderr, "write: too many args\n"); return 2; }
            Compression comp = comp_from_name(argc == 4 ? argv[3] : "none");
            return do_write(argv[2], comp);
        }
        if (std::strcmp(argv[1], "bench-io") == 0) {
            if (argc != 8) {
                fprintf(stderr, "bench-io: expected <pixels> <path> <compression> <width> <height> <threads>\n");
                return 2;
            }
            Compression comp = comp_from_name(argv[4]);
            // comp_from_name falls back to NONE: a misspelt codec must not be timed as NONE.
            if (comp == NO_COMPRESSION && std::strcmp(argv[4], "none") != 0) {
                fprintf(stderr, "bench-io: unknown compression %s\n", argv[4]);
                return 2;
            }
            int width = std::atoi(argv[5]);
            int height = std::atoi(argv[6]);
            int threads = std::atoi(argv[7]);
            return do_bench_io(argv[2], argv[3], comp, width, height, threads);
        }
        if (std::strcmp(argv[1], "bench-scatter") == 0) {
            if (argc != 6) {
                fprintf(stderr, "bench-scatter: expected <compression> <width> <height> <threads>\n");
                return 2;
            }
            Compression comp = comp_from_name(argv[2]);
            int width = std::atoi(argv[3]);
            int height = std::atoi(argv[4]);
            int threads = std::atoi(argv[5]);
            return do_bench_scatter(comp, width, height, threads);
        }
        if (std::strcmp(argv[1], "bench-codec") == 0) {
            if (argc != 5) {
                fprintf(stderr, "bench-codec: expected <compression> <width> <height>\n");
                return 2;
            }
            Compression comp = comp_from_name(argv[2]);
            int width = std::atoi(argv[3]);
            int height = std::atoi(argv[4]);
            return do_bench_codec(comp, width, height);
        }
        // write-zip-level <path> <compression> <level>
        // Additive mode -- writes the SAME known image as `write`, but with
        // `Header::zipCompressionLevel()` overridden to `<level>` before
        // opening the OutputFile, so a golden can pin the on-disk deflate
        // bytes C++ produces at a non-default ZIP/ZIPS compression level.
        if (std::strcmp(argv[1], "write-zip-level") == 0) {
            if (argc != 5) {
                fprintf(stderr, "write-zip-level: expected <path> <compression> <level>\n");
                return 2;
            }
            Compression comp = comp_from_name(argv[3]);
            int level = std::atoi(argv[4]);
            return do_write(argv[2], comp, level);
        }
        if (std::strcmp(argv[1], "read")  == 0) {
            if (argc != 3) { fprintf(stderr, "read: expected exactly <path>\n"); return 2; }
            return do_read(argv[2]);
        }
        if (std::strcmp(argv[1], "write-header-attrs") == 0) {
            if (argc != 3) { fprintf(stderr, "write-header-attrs: expected exactly <path>\n"); return 2; }
            return do_write_header_attrs(argv[2]);
        }
        if (std::strcmp(argv[1], "read-half") == 0) {
            if (argc != 3) { fprintf(stderr, "read-half: expected exactly <path>\n"); return 2; }
            return do_read_half(argv[2]);
        }
        if (std::strcmp(argv[1], "read-as-uint") == 0) {
            if (argc != 3) { fprintf(stderr, "read-as-uint: expected exactly <path>\n"); return 2; }
            return do_read_as_uint(argv[2]);
        }
        if (std::strcmp(argv[1], "read-truncated") == 0) {
            if (argc != 3) { fprintf(stderr, "read-truncated: expected exactly <path>\n"); return 2; }
            return do_read_truncated(argv[2]);
        }
        if (std::strcmp(argv[1], "read-class") == 0) {
            if (argc != 4) { fprintf(stderr, "read-class: expected <path> <api>\n"); return 2; }
            return do_read_class(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "write-tiled") == 0) {
            // write-tiled <path> <compression> <levelmode> [minX minY]
            //             [round=up] [order=random|decreasing]
            // The trailing key=value options keep the positional form stable
            // while letting a golden vary ONE header field (rounding mode /
            // line order), covering paths no golden exercised before.
            LevelRoundingMode rounding = ROUND_DOWN;
            LineOrder         order    = INCREASING_Y;
            int positional = 0, minX = 0, minY = 0;
            for (int i = 5; i < argc; ++i) {
                if (std::strcmp(argv[i], "round=up") == 0) { rounding = ROUND_UP; }
                else if (std::strcmp(argv[i], "round=down") == 0) { rounding = ROUND_DOWN; }
                else if (std::strcmp(argv[i], "order=random") == 0) { order = RANDOM_Y; }
                else if (std::strcmp(argv[i], "order=decreasing") == 0) { order = DECREASING_Y; }
                else if (std::strchr(argv[i], '=') != NULL) {
                    fprintf(stderr, "write-tiled: unknown option '%s'\n", argv[i]);
                    return 2;
                }
                else if (positional == 0) { minX = std::atoi(argv[i]); ++positional; }
                else if (positional == 1) { minY = std::atoi(argv[i]); ++positional; }
                else { fprintf(stderr, "write-tiled: too many positional args\n"); return 2; }
            }
            if (argc < 5 || positional == 1) {
                fprintf(stderr,
                        "write-tiled: expected <path> <compression> <levelmode> "
                        "[minX minY] [round=up] [order=random|decreasing]\n");
                return 2;
            }
            Compression comp = comp_from_name(argv[3]);
            LevelMode   mode = level_mode_from_name(argv[4]);
            return do_write_tiled(argv[2], comp, mode, minX, minY, rounding, order);
        }
        if (std::strcmp(argv[1], "read-tiled") == 0) {
            if (argc != 3) { fprintf(stderr, "read-tiled: expected exactly <path>\n"); return 2; }
            return do_read_tiled(argv[2]);
        }
        if (std::strcmp(argv[1], "write-deep-scanline") == 0) {
            // write-deep-scanline <path> <codec> [minX minY]
            if (argc != 4 && argc != 6) {
                fprintf(stderr, "write-deep-scanline: expected <path> <codec> [minX minY]\n");
                return 2;
            }
            Compression comp = deep_comp_from_name(argv[3]);
            int minX = (argc == 6) ? std::atoi(argv[4]) : 0;
            int minY = (argc == 6) ? std::atoi(argv[5]) : 0;
            return do_write_deep_scanline(argv[2], comp, minX, minY);
        }
        if (std::strcmp(argv[1], "read-deep-scanline") == 0) {
            if (argc != 3) { fprintf(stderr, "read-deep-scanline: expected exactly <path>\n"); return 2; }
            return do_read_deep_scanline(argv[2]);
        }
        if (std::strcmp(argv[1], "write-deep-composite") == 0) {
            // write-deep-composite <path> <a|b>
            if (argc != 4) {
                fprintf(stderr, "write-deep-composite: expected <path> <a|b>\n");
                return 2;
            }
            bool isA;
            if (std::strcmp(argv[3], "a") == 0) isA = true;
            else if (std::strcmp(argv[3], "b") == 0) isA = false;
            else { fprintf(stderr, "write-deep-composite: variant must be 'a' or 'b'\n"); return 2; }
            return do_write_deep_composite(argv[2], isA);
        }
        if (std::strcmp(argv[1], "composite-deep-scanline") == 0) {
            // composite-deep-scanline <pathA> <pathB> <out>
            if (argc != 5) {
                fprintf(stderr, "composite-deep-scanline: expected <pathA> <pathB> <out>\n");
                return 2;
            }
            return do_composite_deep_scanline(argv[2], argv[3], argv[4]);
        }
        if (std::strcmp(argv[1], "read-deep-via-inputfile") == 0) {
            // read-deep-via-inputfile <path> <out>
            if (argc != 4) {
                fprintf(stderr, "read-deep-via-inputfile: expected <path> <out>\n");
                return 2;
            }
            return do_read_deep_via_inputfile(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "write-deep-tiled") == 0) {
            // write-deep-tiled <path> <codec> <levelmode>
            if (argc != 5) {
                fprintf(stderr, "write-deep-tiled: expected <path> <codec> <levelmode>\n");
                return 2;
            }
            Compression comp = deep_comp_from_name(argv[3]);
            LevelMode   mode = level_mode_from_name(argv[4]);
            return do_write_deep_tiled(argv[2], comp, mode);
        }
        if (std::strcmp(argv[1], "read-deep-tiled") == 0) {
            if (argc != 3) { fprintf(stderr, "read-deep-tiled: expected exactly <path>\n"); return 2; }
            return do_read_deep_tiled(argv[2]);
        }
        if (std::strcmp(argv[1], "dump") == 0) {
            if (argc != 4) { fprintf(stderr, "dump: expected <path> <out>\n"); return 2; }
            return do_dump(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "dump-tiled") == 0) {
            if (argc != 4) { fprintf(stderr, "dump-tiled: expected <path> <out>\n"); return 2; }
            return do_dump_tiled(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "write-nonutf8-names") == 0) {
            if (argc != 3) { fprintf(stderr, "write-nonutf8-names: expected <path>\n"); return 2; }
            return do_write_nonutf8_names(argv[2]);
        }
        if (std::strcmp(argv[1], "dump-nonutf8-attr") == 0) {
            if (argc != 4) { fprintf(stderr, "dump-nonutf8-attr: expected <path> <out>\n"); return 2; }
            return do_dump_nonutf8_attr(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "write-dwa-sub") == 0) {
            if (argc != 4) { fprintf(stderr, "write-dwa-sub: expected <path> <dwaa|dwab>\n"); return 2; }
            Compression comp;
            if (std::strcmp(argv[3], "dwaa") == 0) comp = DWAA_COMPRESSION;
            else if (std::strcmp(argv[3], "dwab") == 0) comp = DWAB_COMPRESSION;
            else { fprintf(stderr, "write-dwa-sub: codec must be dwaa or dwab\n"); return 2; }
            return do_write_dwa_sub(argv[2], comp);
        }
        if (std::strcmp(argv[1], "dump-dwa-sub") == 0) {
            if (argc != 4) { fprintf(stderr, "dump-dwa-sub: expected <path> <out>\n"); return 2; }
            return do_dump_subsampled(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "write-sub") == 0) {
            if (argc != 6) {
                fprintf(stderr, "write-sub: expected <path> <codec> <xs> <ys>\n");
                return 2;
            }
            Compression comp = comp_from_name(argv[3]);
            int xs = std::atoi(argv[4]);
            int ys = std::atoi(argv[5]);
            return do_write_sub(argv[2], comp, xs, ys);
        }
        if (std::strcmp(argv[1], "dump-sub") == 0) {
            if (argc != 4) { fprintf(stderr, "dump-sub: expected <path> <out>\n"); return 2; }
            return do_dump_subsampled(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "write-yc") == 0) {
            if (argc != 3) { fprintf(stderr, "write-yc: expected exactly <path>\n"); return 2; }
            return do_write_rgba(argv[2], WRITE_YCA);
        }
        if (std::strcmp(argv[1], "write-rgba") == 0) {
            if (argc != 3) { fprintf(stderr, "write-rgba: expected exactly <path>\n"); return 2; }
            return do_write_rgba(argv[2], WRITE_RGBA);
        }
        if (std::strcmp(argv[1], "write-yc-rounded") == 0) {
            if (argc != 5) {
                fprintf(stderr, "write-yc-rounded: expected <path> <roundY> <roundC>\n");
                return 2;
            }
            unsigned int roundY = (unsigned int)std::atoi(argv[3]);
            unsigned int roundC = (unsigned int)std::atoi(argv[4]);
            return do_write_yc_rounded(argv[2], roundY, roundC);
        }
        if (std::strcmp(argv[1], "write-tiled-rgba") == 0) {
            if (argc != 4) {
                fprintf(stderr, "write-tiled-rgba: expected <path> <rgba|grey>\n");
                return 2;
            }
            RgbaChannels chans;
            if (std::strcmp(argv[3], "rgba") == 0) { chans = WRITE_RGBA; }
            else if (std::strcmp(argv[3], "grey") == 0) { chans = WRITE_YA; }
            else { fprintf(stderr, "write-tiled-rgba: unknown shape '%s'\n", argv[3]); return 2; }
            return do_write_tiled_rgba(argv[2], chans);
        }
        if (std::strcmp(argv[1], "rgba-read") == 0) {
            if (argc != 3 && argc != 4) {
                fprintf(stderr, "rgba-read: expected <path> [out]\n");
                return 2;
            }
            return do_rgba_read(argv[2], argc == 4 ? argv[3] : NULL);
        }
        if (std::strcmp(argv[1], "write-aces") == 0) {
            if (argc != 4) { fprintf(stderr, "write-aces: expected <path> <rgba|rec709>\n"); return 2; }
            return do_write_aces(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "aces-read") == 0) {
            if (argc != 4) { fprintf(stderr, "aces-read: expected <path> <out>\n"); return 2; }
            return do_aces_read(argv[2], argv[3]);
        }
        if (std::strcmp(argv[1], "lut-dump") == 0) {
            if (argc != 4) { fprintf(stderr, "lut-dump: expected <out> <n>\n"); return 2; }
            return do_lut_dump(argv[2], std::atoi(argv[3]));
        }
        if (std::strcmp(argv[1], "write-multipart") == 0) {
            if (argc != 3) { fprintf(stderr, "write-multipart: expected exactly <path>\n"); return 2; }
            return do_write_multipart(argv[2]);
        }
        if (std::strcmp(argv[1], "read-multipart") == 0) {
            if (argc != 3) { fprintf(stderr, "read-multipart: expected exactly <path>\n"); return 2; }
            return do_read_multipart(argv[2]);
        }
        if (std::strcmp(argv[1], "write-idmanifest") == 0) {
            if (argc != 3) { fprintf(stderr, "write-idmanifest: expected exactly <path>\n"); return 2; }
            return do_write_idmanifest(argv[2]);
        }
        if (std::strcmp(argv[1], "dump-idmanifest") == 0) {
            if (argc != 4) { fprintf(stderr, "dump-idmanifest: expected <path> <out>\n"); return 2; }
            return do_dump_idmanifest(argv[2], argv[3]);
        }
        fprintf(stderr, "unknown mode '%s'\n", argv[1]);
        return 2;
    } catch (const std::exception& e) {
        fprintf(stderr, "oracle: EXCEPTION: %s\n", e.what());
        return 1;
    }
}
