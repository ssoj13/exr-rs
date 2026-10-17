// os_thumb_test.cpp — CTest harness for `exv_os_thumb.h` (aspect-ratio fitting, HDR -> sRGB
// tone-mapping, and BGRA8/RGBA8 thumbnail downsampling).

#include "exv_os_thumb.h"

#include <cassert>
#include <cstdio>

int main() {
    uint32_t w = 0, h = 0;
    exv_os::fit_thumbnail_dims(3840, 2160, 256, w, h);
    assert(w == 256 && h == 144);

    exv_os::fit_thumbnail_dims(1080, 1920, 256, w, h);
    assert(w == 144 && h == 256);

    // 2x2 HDR source -> 1x1 BGRA8 thumbnail
    const float hdr_2x2[16] = {
        4.0f, 0.5f, 0.0f, 1.0f,
        4.0f, 0.5f, 0.0f, 1.0f,
        4.0f, 0.5f, 0.0f, 1.0f,
        4.0f, 0.5f, 0.0f, 1.0f,
    };
    const auto bgra = exv_os::render_thumbnail_8bit(hdr_2x2, 2, 2, 1, 1, true);
    assert(bgra.size() == 4);
    assert(bgra[0] == 0);       // B
    assert(bgra[1] > 120);      // G
    assert(bgra[2] > bgra[1]);  // R (4.0 tonemapped > 0.5 tonemapped)
    assert(bgra[3] == 255);     // A

    printf("os_thumb_test passed.\n");
    return 0;
}
