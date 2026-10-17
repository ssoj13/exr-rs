// exv_os_thumb.h — OS Thumbnail & System Codec frame renderer over `exv_capi`.
//
// Shared by:
//   - Windows Shell `IThumbnailProvider` COM handler (`ExvOsThumb.dll`)
//   - Cross-platform CTest verification (`os_thumb_test`)
//
// Converts linear HDR `RGBA_F32` frames from `exv_capi` into aspect-fitted 8-bit `BGRA8` (Windows
// DIB) or `RGBA8` thumbnails using Reinhard highlight roll-off + IEC 61966-2-1 sRGB OETF.

#pragma once

#include "exv-capi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace exv_os {

/// Fit `(src_w, src_h)` inside `max_side x max_side` while preserving aspect ratio (minimum 1x1).
inline void fit_thumbnail_dims(
    uint32_t src_w,
    uint32_t src_h,
    uint32_t max_side,
    uint32_t& out_w,
    uint32_t& out_h) {
    if (src_w == 0 || src_h == 0 || max_side == 0) {
        out_w = 1;
        out_h = 1;
        return;
    }
    if (src_w <= max_side && src_h <= max_side) {
        out_w = src_w;
        out_h = src_h;
        return;
    }
    if (src_w >= src_h) {
        out_w = max_side;
        out_h = std::max<uint32_t>(
            1u, static_cast<uint32_t>((static_cast<uint64_t>(src_h) * max_side + src_w / 2) / src_w));
    } else {
        out_h = max_side;
        out_w = std::max<uint32_t>(
            1u, static_cast<uint32_t>((static_cast<uint64_t>(src_w) * max_side + src_h / 2) / src_h));
    }
}

/// Linear scene-referred float -> Reinhard tonemap -> sRGB 8-bit byte.
inline uint8_t linear_to_srgb_u8(float v) {
    if (!(v > 0.0f)) return 0;
    // Soft highlight shoulder for HDR values > 1.0
    const float mapped = v / (1.0f + v);
    const float srgb = mapped <= 0.0031308f
        ? (12.92f * mapped)
        : (1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f);
    const int code = static_cast<int>(srgb * 255.0f + 0.5f);
    return static_cast<uint8_t>(std::clamp(code, 0, 255));
}

/// Downsample + tonemap a linear `RGBA_F32` image (`src_w x src_h`) into `dst_w x dst_h` 8-bit
/// pixels (`bgraOrder = true` emits `B, G, R, A` for Windows 32-bpp DIBs; `false` emits `R, G, B, A`).
inline std::vector<uint8_t> render_thumbnail_8bit(
    const float* rgba_f32,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    bool bgra_order) {
    std::vector<uint8_t> out(static_cast<size_t>(dst_w) * dst_h * 4, 0);
    if (!rgba_f32 || src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) {
        return out;
    }

    for (uint32_t dy = 0; dy < dst_h; ++dy) {
        const uint32_t y0 = static_cast<uint32_t>((static_cast<uint64_t>(dy) * src_h) / dst_h);
        const uint32_t y1 = std::max<uint32_t>(
            y0 + 1,
            std::min<uint32_t>(src_h, static_cast<uint32_t>(((static_cast<uint64_t>(dy) + 1) * src_h) / dst_h)));
        for (uint32_t dx = 0; dx < dst_w; ++dx) {
            const uint32_t x0 = static_cast<uint32_t>((static_cast<uint64_t>(dx) * src_w) / dst_w);
            const uint32_t x1 = std::max<uint32_t>(
                x0 + 1,
                std::min<uint32_t>(src_w, static_cast<uint32_t>(((static_cast<uint64_t>(dx) + 1) * src_w) / dst_w)));

            float acc_r = 0.0f;
            float acc_g = 0.0f;
            float acc_b = 0.0f;
            float acc_a = 0.0f;
            uint32_t count = 0;
            for (uint32_t sy = y0; sy < y1; ++sy) {
                const float* row = rgba_f32 + static_cast<size_t>(sy) * src_w * 4;
                for (uint32_t sx = x0; sx < x1; ++sx) {
                    const float* px = row + static_cast<size_t>(sx) * 4;
                    acc_r += px[0];
                    acc_g += px[1];
                    acc_b += px[2];
                    acc_a += px[3];
                    ++count;
                }
            }
            const float inv = count > 0 ? (1.0f / static_cast<float>(count)) : 1.0f;
            const uint8_t r8 = linear_to_srgb_u8(acc_r * inv);
            const uint8_t g8 = linear_to_srgb_u8(acc_g * inv);
            const uint8_t b8 = linear_to_srgb_u8(acc_b * inv);
            const int a_code = static_cast<int>(std::clamp(acc_a * inv, 0.0f, 1.0f) * 255.0f + 0.5f);
            const uint8_t a8 = static_cast<uint8_t>(std::clamp(a_code, 0, 255));

            uint8_t* dst = out.data() + (static_cast<size_t>(dy) * dst_w + dx) * 4;
            if (bgra_order) {
                dst[0] = b8;
                dst[1] = g8;
                dst[2] = r8;
                dst[3] = a8;
            } else {
                dst[0] = r8;
                dst[1] = g8;
                dst[2] = b8;
                dst[3] = a8;
            }
        }
    }
    return out;
}

}  // namespace exv_os
