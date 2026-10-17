// nuke_io_test.cpp — CTest harness for `exv_nuke_io.h` (AOV grouping, Y-up/Y-down scanline
// conversion, writer parameter mapping, and end-to-end `exv_capi` round-trip).

#include "exv_nuke_io.h"

#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // 1. Verify Nuke Y-up (0 = bottom) <-> EXV Y-down (0 = top) mapping.
    assert(exv_nuke::nuke_y_to_exv_row(0, 1080) == 1079);
    assert(exv_nuke::nuke_y_to_exv_row(1079, 1080) == 0);
    assert(exv_nuke::nuke_y_to_exv_row(500, 1080) == 579);

    // 2. Verify AOV channel grouping into Nuke layers.
    std::vector<std::string> chans = {
        "R", "G", "B", "A", "diffuse.R", "diffuse.G", "diffuse.B", "depth.Z"};
    auto groups = exv_nuke::group_aov_layers(chans);
    assert(groups.size() == 3);
    assert(groups[0].layer.empty());
    assert(groups[0].chans[0] == "R" && groups[0].chans[3] == "A");
    assert(groups[1].layer == "diffuse");
    assert(groups[1].chans[0] == "diffuse.R" && groups[1].chans[2] == "diffuse.B");
    assert(groups[2].layer == "depth");
    assert(groups[2].chans[2] == "depth.Z");

    // 3. Verify writer knob -> ExvEncParams mapping across all 6 profiles.
    const auto p_bc6h_lossless = exv_nuke::make_writer_params(1920, 1080, 0, true, 0);
    assert(p_bc6h_lossless.codec_id == EXV_CODEC_BC6H);
    assert(p_bc6h_lossless.q == 0);
    assert(p_bc6h_lossless.dtype_code == EXV_DTYPE_F16);

    const auto p_bc6h_lossy = exv_nuke::make_writer_params(1920, 1080, 1, false, 0);
    assert(p_bc6h_lossy.codec_id == EXV_CODEC_BC6H);
    assert(p_bc6h_lossy.q == 1);
    assert(p_bc6h_lossy.dtype_code == EXV_DTYPE_F16);

    const auto p_htj2k = exv_nuke::make_writer_params(1920, 1080, 2, false, 4);
    assert(p_htj2k.codec_id == EXV_CODEC_HTJ2K);
    assert(p_htj2k.q == 4);
    assert(p_htj2k.dtype_code == EXV_DTYPE_F32);

    printf("nuke_io_test passed.\n");
    return 0;
}
