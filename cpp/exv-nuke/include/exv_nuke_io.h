// exv_nuke_io.h — Nuke NDK <-> exv_capi bridge (SDK-independent core + dynamic loader).
//
// Handles:
//   1. Cross-platform dynamic loading of `exv_capi.dll` / `libexv_capi.so` / `libexv_capi.dylib`
//      with `EXV_ABI_VERSION` verification.
//   2. AOV channel discovery & Nuke layer.channel mapping (`R`/`G`/`B`/`A` -> `rgba.*`,
//      `diffuse.R` -> `diffuse.red`, etc.).
//   3. Nuke Y-up scanline addressing (`y = 0` at bottom) <-> EXV Y-down row addressing (`row = 0` at top).
//   4. Codec / quality / dtype knob mapping for `exvWriter`.

#pragma once

#include "exv-capi.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace exv_nuke {

struct ApiTable {
    decltype(&exv_abi_version) abi_version = nullptr;
    decltype(&exv_last_error_message) last_error_message = nullptr;
    decltype(&exv_session_open) session_open = nullptr;
    decltype(&exv_session_info) session_info = nullptr;
    decltype(&exv_session_meta) session_meta = nullptr;
    decltype(&exv_session_decode) session_decode = nullptr;
    decltype(&exv_session_channel_count) session_channel_count = nullptr;
    decltype(&exv_session_channel_name) session_channel_name = nullptr;
    decltype(&exv_session_decode_mapped_rgbaf32) session_decode_mapped = nullptr;
    decltype(&exv_session_close) session_close = nullptr;
    decltype(&exv_encode_rgba_f32) encode_rgba_f32 = nullptr;
    decltype(&exv_write_file) write_file = nullptr;
    decltype(&exv_buffer_data) buffer_data = nullptr;
    decltype(&exv_buffer_len) buffer_len = nullptr;
    decltype(&exv_buffer_free) buffer_free = nullptr;
    decltype(&exv_venc_open) venc_open = nullptr;
    decltype(&exv_venc_add_rgba_f32) venc_add_rgba_f32 = nullptr;
    decltype(&exv_venc_finish) venc_finish = nullptr;
    decltype(&exv_venc_free) venc_free = nullptr;

    bool valid() const {
        return abi_version && session_open && session_info && session_meta && session_decode &&
               session_channel_count && session_channel_name && session_decode_mapped &&
               session_close && encode_rgba_f32 && write_file && buffer_data && buffer_len &&
               buffer_free && venc_open && venc_add_rgba_f32 && venc_finish && venc_free;
    }
};

inline void* load_sym(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

inline bool load_api(const char* explicit_path, ApiTable& out, std::string& err) {
    void* lib = nullptr;
#if defined(_WIN32)
    if (explicit_path && explicit_path[0]) {
        lib = static_cast<void*>(LoadLibraryA(explicit_path));
    }
    if (!lib) {
        lib = static_cast<void*>(LoadLibraryA("exv_capi.dll"));
    }
#elif defined(__APPLE__)
    if (explicit_path && explicit_path[0]) {
        lib = dlopen(explicit_path, RTLD_NOW | RTLD_LOCAL);
    }
    if (!lib) {
        lib = dlopen("libexv_capi.dylib", RTLD_NOW | RTLD_LOCAL);
    }
#else
    if (explicit_path && explicit_path[0]) {
        lib = dlopen(explicit_path, RTLD_NOW | RTLD_LOCAL);
    }
    if (!lib) {
        lib = dlopen("libexv_capi.so", RTLD_NOW | RTLD_LOCAL);
    }
#endif
    if (!lib) {
        err = "failed to load exv_capi shared library";
        return false;
    }

#define EXV_BIND(field, sym) out.field = reinterpret_cast<decltype(out.field)>(load_sym(lib, #sym))
    EXV_BIND(abi_version, exv_abi_version);
    EXV_BIND(last_error_message, exv_last_error_message);
    EXV_BIND(session_open, exv_session_open);
    EXV_BIND(session_info, exv_session_info);
    EXV_BIND(session_meta, exv_session_meta);
    EXV_BIND(session_decode, exv_session_decode);
    EXV_BIND(session_channel_count, exv_session_channel_count);
    EXV_BIND(session_channel_name, exv_session_channel_name);
    EXV_BIND(session_decode_mapped, exv_session_decode_mapped_rgbaf32);
    EXV_BIND(session_close, exv_session_close);
    EXV_BIND(encode_rgba_f32, exv_encode_rgba_f32);
    EXV_BIND(write_file, exv_write_file);
    EXV_BIND(buffer_data, exv_buffer_data);
    EXV_BIND(buffer_len, exv_buffer_len);
    EXV_BIND(buffer_free, exv_buffer_free);
    EXV_BIND(venc_open, exv_venc_open);
    EXV_BIND(venc_add_rgba_f32, exv_venc_add_rgba_f32);
    EXV_BIND(venc_finish, exv_venc_finish);
    EXV_BIND(venc_free, exv_venc_free);
#undef EXV_BIND

    if (!out.valid()) {
        err = "exv_capi is missing required symbols";
        return false;
    }
    const uint32_t got_abi = out.abi_version();
    if (got_abi != EXV_ABI_VERSION) {
        err = "exv_capi ABI mismatch (expected " + std::to_string(EXV_ABI_VERSION) + ", got " +
              std::to_string(got_abi) + ")";
        return false;
    }
    return true;
}

/// Convert Nuke Y-up row index (`0` = bottom scanline) to EXV Y-down row index (`0` = top row).
inline uint32_t nuke_y_to_exv_row(int nuke_y, uint32_t height) {
    if (height == 0) return 0;
    const int clamped = nuke_y < 0 ? 0 : (nuke_y >= static_cast<int>(height) ? static_cast<int>(height) - 1 : nuke_y);
    return static_cast<uint32_t>(static_cast<int>(height) - 1 - clamped);
}

/// Group of up to 4 EXV channel names belonging to one Nuke layer (e.g. `rgba`, `diffuse`, `depth`).
struct LayerGroup {
    std::string layer;      // empty for default rgba
    std::string chans[4];   // EXV channel names for R, G, B, A slots ("" = unmapped)
};

/// Query all channel names from an open `ExvSession`.
inline std::vector<std::string> query_channels(const ApiTable& api, const ExvSession* s) {
    uint32_t count = 0;
    if (api.session_channel_count(s, &count) != 0) {
        return {};
    }
    std::vector<std::string> names;
    names.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        char buf[256];
        size_t written = 0;
        if (api.session_channel_name(s, i, buf, sizeof(buf), &written) == 0 && written > 0) {
            names.emplace_back(buf, written - 1);
        }
    }
    return names;
}

/// Partition EXV channel names into Nuke layer groups of up to 4 channels (`R`/`G`/`B`/`A` first).
inline std::vector<LayerGroup> group_aov_layers(const std::vector<std::string>& names) {
    std::vector<LayerGroup> groups;
    auto find_or_add = [&](const std::string& layer) -> LayerGroup& {
        for (auto& g : groups) {
            if (g.layer == layer) return g;
        }
        groups.push_back(LayerGroup{layer, {"", "", "", ""}});
        return groups.back();
    };

    for (const auto& full : names) {
        std::string layer;
        std::string suffix = full;
        const auto dot = full.rfind('.');
        if (dot != std::string::npos) {
            layer = full.substr(0, dot);
            suffix = full.substr(dot + 1);
        }
        LayerGroup& g = find_or_add(layer);
        int slot = -1;
        if (suffix == "R" || suffix == "r" || suffix == "red" || suffix == "x" || suffix == "X") {
            slot = 0;
        } else if (suffix == "G" || suffix == "g" || suffix == "green" || suffix == "y" || suffix == "Y") {
            slot = 1;
        } else if (suffix == "B" || suffix == "b" || suffix == "blue" || suffix == "z" || suffix == "Z") {
            slot = 2;
        } else if (suffix == "A" || suffix == "a" || suffix == "alpha" || suffix == "w" || suffix == "W") {
            slot = 3;
        } else {
            for (int k = 0; k < 4; ++k) {
                if (g.chans[k].empty()) {
                    slot = k;
                    break;
                }
            }
        }
        if (slot >= 0 && slot < 4 && g.chans[slot].empty()) {
            g.chans[slot] = full;
        }
    }
    return groups;
}

/// Map Nuke writer codec menu index (0..5) to `ExvEncParams` (`codec_id`, `codec_mode`, `q`).
/// Menu order:
///   0 = BC6H Lossless Residual (default)
///   1 = BC6H Lossy (GPU direct)
///   2 = HTJ2K Lossless
///   3 = Fast rANS Lossless
///   4 = Wavelet 5/3 Lossless
///   5 = EXV-RT Realtime
inline ExvEncParams make_writer_params(
    uint32_t width,
    uint32_t height,
    int codec_knob,
    bool half_float,
    int quality_knob) {
    ExvEncParams p{};
    p.width = width;
    p.height = height;
    p.dtype_code = half_float ? EXV_DTYPE_F16 : EXV_DTYPE_F32;
    p.tile_w = 256;
    p.tile_h = 256;
    p.codec_mode = EXV_CODEC_MODE_DEFAULT;
    p.q = 0;

    switch (codec_knob) {
        case 0:
            p.codec_id = EXV_CODEC_BC6H;
            p.dtype_code = EXV_DTYPE_F16;
            p.q = 0;
            break;
        case 1:
            p.codec_id = EXV_CODEC_BC6H;
            p.dtype_code = EXV_DTYPE_F16;
            p.q = 1;
            break;
        case 2:
            p.codec_id = EXV_CODEC_HTJ2K;
            p.q = quality_knob > 0 ? static_cast<uint32_t>(quality_knob) : 0;
            if (p.q > EXV_Q_HDR_MAX) p.q = EXV_Q_HDR_MAX;
            break;
        case 3:
            p.codec_id = EXV_CODEC_LOSSLESS_FAST;
            p.q = 0;
            break;
        case 4:
            p.codec_id = EXV_CODEC_WAVELET53;
            p.q = 0;
            break;
        case 5:
        default:
            p.codec_id = EXV_CODEC_EXV_RT;
            p.dtype_code = EXV_DTYPE_F16;
            p.q = quality_knob > 0 ? static_cast<uint32_t>(quality_knob) : 0;
            if (p.q > EXV_Q_RT_MAX) p.q = EXV_Q_RT_MAX;
            break;
    }
    return p;
}

}  // namespace exv_nuke
