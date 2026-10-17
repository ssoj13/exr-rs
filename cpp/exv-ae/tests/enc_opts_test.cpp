// Stand-alone test of the export-options table / ExvEncParams mapping (no AE SDK needed):
//   cl /std:c++17 /EHsc /W3 /I..\src /I..\..\..\docs enc_opts_test.cpp && enc_opts_test.exe
#include "exv_enc_opts.h"
#include <cstdio>
#include <cstdlib>

using namespace exvae;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static EncOpts Opts(ExportMode m, uint32_t dtype, uint32_t q) { return EncOpts{ static_cast<uint32_t>(m), dtype, q }; }

int main()
{
    // Row i of the table is ExportMode i (the handle stores the number).
    for (uint32_t i = 0; i < static_cast<uint32_t>(ExportMode::Count); ++i) {
        const ModeInfo* m = FindMode(i);
        CHECK(m && static_cast<uint32_t>(m->mode) == i);
        // Every mode's own defaults are valid options, and its dtype set contains its default dtype.
        CHECK(Validate(DefaultsFor(*m)));
        CHECK(m->dtypes & DtypeBit(m->dtype_def));
        // The quality range is inside the ABI limits and starts at 0 or 1.
        if (ModeHasQ(*m)) CHECK(m->q_min <= m->q_def && m->q_def <= m->q_max);
    }
    CHECK(FindMode(static_cast<uint32_t>(ExportMode::Count)) == nullptr);
    CHECK(Validate(DefaultOpts()));

    // The default export = HTJ2K lossless F16 (id 6, mode 0, q 0, whole frame).
    ExvEncParams p{};
    CHECK(MakeParams(DefaultOpts(), 64, 32, &p));
    CHECK(p.width == 64 && p.height == 32 && p.dtype_code == EXV_DTYPE_F16);
    CHECK(p.codec_id == EXV_CODEC_HTJ2K && p.codec_mode == EXV_CODEC_MODE_DEFAULT && p.q == 0);
    CHECK(p.tile_w == 0 && p.tile_h == 0);

    // 9/7: codec_mode 1, integers only, q 1..=EXV_Q_LDR_MAX.
    CHECK(MakeParams(Opts(ExportMode::Htj2kLdr97, EXV_DTYPE_U8, 16), 8, 8, &p));
    CHECK(p.codec_id == EXV_CODEC_HTJ2K && p.codec_mode == EXV_CODEC_MODE_HTJ2K_IRREVERSIBLE97 && p.q == 16);
    CHECK(!Validate(Opts(ExportMode::Htj2kLdr97, EXV_DTYPE_F16, 16)));
    CHECK(!Validate(Opts(ExportMode::Htj2kLdr97, EXV_DTYPE_U16, 0)));
    CHECK(!Validate(Opts(ExportMode::Htj2kLdr97, EXV_DTYPE_U16, EXV_Q_LDR_MAX + 1)));
    CHECK(Validate(Opts(ExportMode::Htj2kLdr97, EXV_DTYPE_U16, EXV_Q_LDR_MAX)));

    // HDR lossy: floats only, q 1..=16 (no silent saturation above).
    CHECK(MakeParams(Opts(ExportMode::Htj2kHdr, EXV_DTYPE_F32, 12), 8, 8, &p) && p.codec_mode == 0 && p.q == 12);
    CHECK(!Validate(Opts(ExportMode::Htj2kHdr, EXV_DTYPE_F32, 0)));
    CHECK(!Validate(Opts(ExportMode::Htj2kHdr, EXV_DTYPE_F32, EXV_Q_HDR_MAX + 1)));
    CHECK(!Validate(Opts(ExportMode::Htj2kHdr, EXV_DTYPE_U8, 8)));

    // Lossless modes take no q; RT is half-only with q 0..=15; lossy wavelet q 1..=62.
    CHECK(MakeParams(Opts(ExportMode::Fast, EXV_DTYPE_U16, 0), 8, 8, &p) && p.codec_id == EXV_CODEC_LOSSLESS_FAST);
    CHECK(MakeParams(Opts(ExportMode::Wave, EXV_DTYPE_F32, 0), 8, 8, &p) && p.codec_id == EXV_CODEC_WAVELET53);
    CHECK(!Validate(Opts(ExportMode::Wave, EXV_DTYPE_F32, 3)));
    CHECK(MakeParams(Opts(ExportMode::Rt, EXV_DTYPE_F16, 0), 8, 8, &p) && p.codec_id == EXV_CODEC_EXV_RT && p.q == 0);
    CHECK(!Validate(Opts(ExportMode::Rt, EXV_DTYPE_F32, 4)));
    CHECK(!Validate(Opts(ExportMode::Rt, EXV_DTYPE_F16, EXV_Q_RT_MAX + 1)));
    CHECK(MakeParams(Opts(ExportMode::LossyWave, EXV_DTYPE_BF16, EXV_Q_WAVE_MAX), 8, 8, &p) && p.codec_id == EXV_CODEC_WAVELET53_LOSSY);
    CHECK(!Validate(Opts(ExportMode::LossyWave, EXV_DTYPE_F16, 0)));
    CHECK(MakeParams(Opts(ExportMode::Bc6h, EXV_DTYPE_F16, 0), 8, 8, &p) && p.codec_id == EXV_CODEC_BC6H && p.q == 0);
    CHECK(MakeParams(Opts(ExportMode::Bc6h, EXV_DTYPE_F16, 1), 8, 8, &p) && p.codec_id == EXV_CODEC_BC6H && p.q == 1);
    CHECK(!Validate(Opts(ExportMode::Bc6h, EXV_DTYPE_F16, 2)));
    CHECK(!Validate(Opts(ExportMode::Bc6h, EXV_DTYPE_F32, 0)));

    // Garbage handle contents are rejected, never mapped; `out` stays untouched on failure.
    ExvEncParams keep{}; keep.width = 77;
    CHECK(!MakeParams(EncOpts{ 99, 0, 0 }, 8, 8, &keep) && keep.width == 77);
    CHECK(!MakeParams(EncOpts{ 0, 77, 0 }, 8, 8, &keep));
    CHECK(!MakeParams(DefaultOpts(), 0, 8, &keep) && keep.width == 77);
    CHECK(Summary(Opts(ExportMode::Htj2kLdr97, EXV_DTYPE_U16, 16)).find("q16") != std::string::npos);

    if (g_fail == 0) std::puts("enc_opts_test: OK");
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
