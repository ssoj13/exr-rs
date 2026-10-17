#pragma once
// Export options of the AE plugin: the ONE table that maps a user-facing export mode to the exv-capi
// encode parameters (codec id + codec_mode + q) and says which sample types / quality range each mode allows.
//
// Header-only and AE-free, so `tests/enc_opts_test.cpp` covers it without the SDK. exv-capi stays the
// authority: it re-validates every ExvEncParams at venc_open (hard error). This table only keeps the dialog
// from offering combinations the library would refuse, and its limits come from the generated header
// (EXV_Q_*_MAX, EXV_CODEC_*), so a changed ABI breaks THIS build instead of drifting.

#include <cstdint>
#include <string>
#include <utility>
#include "exv-capi.h"

namespace exvae {

// The table below encodes the codec/q/dtype rules of ABI v5. A bumped ABI must be reviewed against
// `profile_of` in crates/exv-capi (then this number follows), never just recompiled.
static_assert(EXV_ABI_VERSION == 5, "exv_enc_opts.h: re-check kModes against the new ABI");

// Persisted in the output-options handle as a number: append only, never renumber (bump EXV_OPT_VERSION if you do).
enum class ExportMode : uint32_t {
    Htj2kLossless = 0,  // default
    Htj2kHdr,           // HTJ2K lossy on float channels
    Htj2kLdr97,         // HTJ2K irreversible 9/7 on 8/16-bit
    Fast,               // reorder+delta -> rANS
    Wave,               // 5/3 wavelet -> rANS
    LossyWave,          // 5/3 wavelet + quantisation
    Rt,                 // EXV-RT realtime block codec
    Bc6h,               // Hardware BC6H_SF16 4x4 block codec (q=0 lossless residual, q=1 lossy)
    Count
};

constexpr uint32_t DtypeBit(uint32_t dtype) { return 1u << dtype; }
constexpr uint32_t kDtypesFloat = DtypeBit(EXV_DTYPE_F16) | DtypeBit(EXV_DTYPE_F32) | DtypeBit(EXV_DTYPE_BF16);
constexpr uint32_t kDtypesInt   = DtypeBit(EXV_DTYPE_U8) | DtypeBit(EXV_DTYPE_U16);
constexpr uint32_t kDtypesAny   = kDtypesFloat | kDtypesInt;

struct ModeInfo {
    ExportMode  mode;
    const char* label;       // combo entry
    const char* hint;        // one-paragraph explanation shown under the combo
    uint32_t    codec_id;    // ExvCodec
    uint32_t    codec_mode;  // ExvCodecMode
    uint32_t    q_min, q_max, q_def;  // q_max == 0: the mode has no quality knob (q is sent as 0)
    uint32_t    dtypes;      // allowed ExvDtype set (DtypeBit mask)
    uint32_t    dtype_def;
};

constexpr ModeInfo kModes[] = {
    { ExportMode::Htj2kLossless, "HTJ2K lossless (default)",
      "Bit-exact. Best lossless ratio for HDR float; decodes on the CPU (GPU path pending).",
      EXV_CODEC_HTJ2K, EXV_CODEC_MODE_DEFAULT, 0, 0, 0, kDtypesAny, EXV_DTYPE_F16 },
    { ExportMode::Htj2kHdr, "HTJ2K lossy, HDR float",
      "Perceptual log-domain quantisation of float channels (relative error). Quality 16 = near-lossless; "
      "lower = smaller. Higher quality than 16 does not exist.",
      EXV_CODEC_HTJ2K, EXV_CODEC_MODE_DEFAULT, 1, EXV_Q_HDR_MAX, 8, kDtypesFloat, EXV_DTYPE_F16 },
    { ExportMode::Htj2kLdr97, "HTJ2K lossy, 8/16-bit (9/7)",
      "Irreversible 9/7 wavelet for 8/16-bit images, far smaller than lossless. Quality 16 is about one 8-bit step. "
      "Different decoders may differ by +-1 LSB (rounding of the 9/7 reconstruction); the same decoder is deterministic.",
      EXV_CODEC_HTJ2K, EXV_CODEC_MODE_HTJ2K_IRREVERSIBLE97, 1, EXV_Q_LDR_MAX, 16, kDtypesInt, EXV_DTYPE_U16 },
    { ExportMode::Fast, "Fast lossless (rANS)",
      "Bit-exact, simple reorder+delta+rANS; decodes on the GPU today. Larger files than HTJ2K.",
      EXV_CODEC_LOSSLESS_FAST, EXV_CODEC_MODE_DEFAULT, 0, 0, 0, kDtypesAny, EXV_DTYPE_F16 },
    { ExportMode::Wave, "Wavelet lossless (5/3 + rANS)",
      "Bit-exact 5/3 wavelet + rANS.",
      EXV_CODEC_WAVELET53, EXV_CODEC_MODE_DEFAULT, 0, 0, 0, kDtypesAny, EXV_DTYPE_F16 },
    { ExportMode::LossyWave, "Wavelet lossy (5/3 + rANS)",
      "Quantises the wavelet details of float channels (q = quantisation bits, lower = smaller).",
      EXV_CODEC_WAVELET53_LOSSY, EXV_CODEC_MODE_DEFAULT, 1, EXV_Q_WAVE_MAX, 8, kDtypesFloat, EXV_DTYPE_F16 },
    { ExportMode::Rt, "EXV-RT realtime (lossy, GPU)",
      "Fixed 16x16 blocks for realtime GPU playback. Half-float only. Quality = detail quantisation, 0 = lossless detail.",
      EXV_CODEC_EXV_RT, EXV_CODEC_MODE_DEFAULT, 0, EXV_Q_RT_MAX, 4, DtypeBit(EXV_DTYPE_F16), EXV_DTYPE_F16 },
    { ExportMode::Bc6h, "BC6H hardware (GPU 4x4 block)",
      "Hardware BC6H_SF16 4x4 blocks for instant GPU texture upload. Half-float only. "
      "Quality 0 = 100% bit-exact lossless (BC6H base + rANS residual); 1 = lossy BC6H blocks only (6:1 fixed ratio).",
      EXV_CODEC_BC6H, EXV_CODEC_MODE_DEFAULT, 0, 1, 0, DtypeBit(EXV_DTYPE_F16), EXV_DTYPE_F16 },
};
static_assert(sizeof(kModes) / sizeof(kModes[0]) == static_cast<size_t>(ExportMode::Count), "one table row per ExportMode");

struct DtypeInfo { uint32_t code; const char* label; };
constexpr DtypeInfo kDtypes[] = {
    { EXV_DTYPE_F16,  "16-bit float (half)" },
    { EXV_DTYPE_F32,  "32-bit float" },
    { EXV_DTYPE_BF16, "bfloat16" },
    { EXV_DTYPE_U8,   "8-bit integer" },
    { EXV_DTYPE_U16,  "16-bit integer" },
};

// Raw handle value -> table row (nullptr when out of range, i.e. a corrupt/foreign handle).
inline const ModeInfo* FindMode(uint32_t raw)
{
    return raw < static_cast<uint32_t>(ExportMode::Count) ? &kModes[raw] : nullptr;
}
inline const char* DtypeLabel(uint32_t code)
{
    for (const DtypeInfo& d : kDtypes) if (d.code == code) return d.label;
    return nullptr;
}
inline bool ModeHasQ(const ModeInfo& m) { return m.q_max != 0; }

// What the output-module dialog edits and the options handle stores.
struct EncOpts {
    uint32_t mode  = static_cast<uint32_t>(ExportMode::Htj2kLossless);
    uint32_t dtype = EXV_DTYPE_F16;
    uint32_t q     = 0;
};

// The mode's own defaults (what picking it in the dialog resets dtype/q to).
inline EncOpts DefaultsFor(const ModeInfo& m)
{
    EncOpts o;
    o.mode  = static_cast<uint32_t>(m.mode);
    o.dtype = m.dtype_def;
    o.q     = m.q_def;
    return o;
}
inline EncOpts DefaultOpts() { return DefaultsFor(kModes[0]); }

// Hard validation: `why` (optional) gets a sentence for the user/log. Nothing is clamped here.
inline bool Validate(const EncOpts& o, std::string* why = nullptr)
{
    auto no = [&](std::string s) { if (why) *why = std::move(s); return false; };
    const ModeInfo* m = FindMode(o.mode);
    if (!m) return no("unknown export mode " + std::to_string(o.mode));
    if (o.dtype >= 32 || !(m->dtypes & DtypeBit(o.dtype)) || !DtypeLabel(o.dtype))
        return no(std::string(m->label) + " does not support sample type " + std::to_string(o.dtype));
    if (ModeHasQ(*m)) {
        if (o.q < m->q_min || o.q > m->q_max)
            return no(std::string(m->label) + ": quality must be " + std::to_string(m->q_min) + ".." + std::to_string(m->q_max));
    } else if (o.q != 0) {
        return no(std::string(m->label) + " has no quality setting");
    }
    return true;
}

// THE mapping options -> ExvEncParams (zero-initialised first, so every unused/new field means "default").
// Returns false (and `why`) for invalid options or an empty frame; `out` is only written on success.
inline bool MakeParams(const EncOpts& o, uint32_t w, uint32_t h, ExvEncParams* out, std::string* why = nullptr)
{
    if (!Validate(o, why)) return false;
    if (w == 0 || h == 0) { if (why) *why = "empty frame"; return false; }
    const ModeInfo& m = kModes[o.mode];
    ExvEncParams p{};
    p.width      = w;
    p.height     = h;
    p.dtype_code = o.dtype;
    p.codec_id   = m.codec_id;
    p.codec_mode = m.codec_mode;
    p.q          = o.q;
    *out = p;  // tile_w/tile_h stay 0 = whole frame
    return true;
}

// "HTJ2K lossy, 8/16-bit (9/7), 16-bit integer, q16" for the Output Module summary.
inline std::string Summary(const EncOpts& o)
{
    const ModeInfo* m = FindMode(o.mode);
    const char* d = DtypeLabel(o.dtype);
    if (!m || !d) return "invalid options";
    std::string s = std::string(m->label) + ", " + d;
    if (ModeHasQ(*m)) s += ", q" + std::to_string(o.q);
    return s;
}

} // namespace exvae
