// exvWriter.cpp — Foundry Nuke NDK `FileWriter` plugin for `.exv` / `.exs` still & video clips.
//
// Supports BC6H Lossless Residual, BC6H Lossy, HTJ2K, Fast rANS, Wavelet 5/3, and EXV-RT profiles
// via `exv_capi`. Converts Nuke bottom-up scanlines (`y = 0` at bottom) into EXV top-down rows.

#include "exv_nuke_io.h"

#include "DDImage/Enumeration_KnobI.h"
#include "DDImage/FileWriter.h"
#include "DDImage/Knobs.h"
#include "DDImage/Row.h"

#include <mutex>

using namespace DD::Image;

namespace {

static const char* const kCodecNames[] = {
    "BC6H Lossless Residual",
    "BC6H Lossy (GPU Direct)",
    "HTJ2K",
    "Fast rANS Lossless",
    "Wavelet 5/3 Lossless",
    "EXV-RT Realtime",
    nullptr,
};

static const char* const kDatatypeNames[] = {
    "16-bit half",
    "32-bit float",
    nullptr,
};

const exv_nuke::ApiTable& get_api(std::string& err) {
    static exv_nuke::ApiTable api;
    static std::string init_err;
    static std::once_flag once;
    std::call_once(once, []() { exv_nuke::load_api(nullptr, api, init_err); });
    err = init_err;
    return api;
}

class ExvWriter final : public FileWriter {
    int codec_knob_ = 0;
    int datatype_knob_ = 0;
    int quality_knob_ = 0;

public:
    explicit ExvWriter(Write* iop) : FileWriter(iop) {}

    void execute() override {
        std::string err;
        const auto& api = get_api(err);
        if (!api.valid()) {
            iop->error("exvWriter: %s", err.c_str());
            return;
        }

        const int w = width();
        const int h = height();
        if (w <= 0 || h <= 0) {
            iop->error("exvWriter: invalid image dimensions %dx%d", w, h);
            return;
        }

        ChannelSet mask = Mask_RGBA;
        input0().request(0, 0, w, h, mask, 1);
        if (aborted()) return;

        std::vector<float> rgba(static_cast<size_t>(w) * h * 4, 0.0f);
        Row row(0, w);
        for (int y = 0; y < h; ++y) {
            if (aborted()) return;
            get(y, 0, w, mask, row);
            const uint32_t exv_row = exv_nuke::nuke_y_to_exv_row(y, static_cast<uint32_t>(h));
            float* dst = rgba.data() + static_cast<size_t>(exv_row) * w * 4;
            const float* r = row[Chan_Red];
            const float* g = row[Chan_Green];
            const float* b = row[Chan_Blue];
            const float* a = (mask & Mask_Alpha) ? row[Chan_Alpha] : nullptr;
            for (int x = 0; x < w; ++x) {
                dst[x * 4 + 0] = r ? r[x] : 0.0f;
                dst[x * 4 + 1] = g ? g[x] : 0.0f;
                dst[x * 4 + 2] = b ? b[x] : 0.0f;
                dst[x * 4 + 3] = a ? a[x] : 1.0f;
            }
        }

        const ExvEncParams params = exv_nuke::make_writer_params(
            static_cast<uint32_t>(w),
            static_cast<uint32_t>(h),
            codec_knob_,
            datatype_knob_ == 0,
            quality_knob_);

        ExvBuffer* buf = nullptr;
        size_t len = 0;
        if (api.encode_rgba_f32(&params, rgba.data(), rgba.size(), &buf, &len) != 0 || !buf) {
            iop->error("exvWriter encode failed: %s", api.last_error_message());
            return;
        }

        const int32_t wr = api.write_file(filename(), api.buffer_data(buf), api.buffer_len(buf));
        api.buffer_free(buf);
        if (wr != 0) {
            iop->error("exvWriter write failed: %s", api.last_error_message());
        }
    }

    void knobs(Knob_Callback f) override {
        Enumeration_knob(f, &codec_knob_, kCodecNames, "compression", "compression");
        Enumeration_knob(f, &datatype_knob_, kDatatypeNames, "datatype", "datatype");
        Int_knob(f, &quality_knob_, "quality", "quality");
    }

    const char* help() override {
        return "EXV HDR Still/Video format (BC6H / HTJ2K / rANS / Wavelet / EXV-RT)";
    }
};

static Writer* build(Write* iop) {
    return new ExvWriter(iop);
}

}  // namespace

const Writer::Description d("exv\0exs\0", build);
