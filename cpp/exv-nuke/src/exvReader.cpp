// exvReader.cpp — Foundry Nuke NDK `FileReader` plugin for `.exv` / `.exs` still & video clips.
//
// Decodes multi-frame `.exv` containers and single-frame `.exv` stills on demand through
// `exv_capi`, exposing all AOV layers and converting EXV top-down rows to Nuke bottom-up scanlines.

#include "exv_nuke_io.h"

#include "DDImage/FileReader.h"
#include "DDImage/ImagePlane.h"
#include "DDImage/Iop.h"
#include "DDImage/Row.h"

#include <mutex>

using namespace DD::Image;

namespace {

const exv_nuke::ApiTable& get_api(std::string& err) {
    static exv_nuke::ApiTable api;
    static std::string init_err;
    static std::once_flag once;
    std::call_once(once, []() { exv_nuke::load_api(nullptr, api, init_err); });
    err = init_err;
    return api;
}

class ExvReader final : public FileReader {
    ExvSession* session_ = nullptr;
    ExvStreamInfo info_{};
    std::vector<exv_nuke::LayerGroup> groups_;
    std::mutex decode_mu_;
    int cached_frame_ = -1;
    std::vector<float> cached_rgba_;

public:
    ExvReader(Read* r, int fd) : FileReader(r) {
        std::string err;
        const auto& api = get_api(err);
        if (!api.valid()) {
            iop->error("exvReader: %s", err.c_str());
            return;
        }
        if (api.session_open(filename(), &session_) != 0 || !session_) {
            iop->error("exvReader: %s", api.last_error_message());
            return;
        }
        if (api.session_info(session_, &info_) != 0) {
            iop->error("exvReader: %s", api.last_error_message());
            return;
        }

        const auto ch_names = exv_nuke::query_channels(api, session_);
        groups_ = exv_nuke::group_aov_layers(ch_names);

        ChannelSet mask = Mask_RGBA;
        set_info(static_cast<int>(info_.width), static_cast<int>(info_.height), 4);
        info_.channels(mask);
        lut_ = LUT::GetLut(LUT::FLOAT);
    }

    ~ExvReader() override {
        if (session_) {
            std::string err;
            const auto& api = get_api(err);
            if (api.valid()) {
                api.session_close(session_);
            }
            session_ = nullptr;
        }
    }

    void engine(int y, int x, int r, ChannelMask channels, Row& row) override {
        if (!session_ || info_.width == 0 || info_.height == 0) {
            row.erase(channels);
            return;
        }
        std::string err;
        const auto& api = get_api(err);
        if (!api.valid()) {
            row.erase(channels);
            return;
        }

        const int req_frame = frame() > 0 ? frame() - 1 : 0;
        const uint32_t clamped = info_.frame_count > 0
            ? (static_cast<uint32_t>(req_frame) < info_.frame_count
                   ? static_cast<uint32_t>(req_frame)
                   : info_.frame_count - 1)
            : 0;

        {
            std::lock_guard<std::mutex> lock(decode_mu_);
            if (cached_frame_ != static_cast<int>(clamped)) {
                const size_t total = static_cast<size_t>(info_.width) * info_.height * 4;
                cached_rgba_.resize(total);
                size_t written = 0;
                const int32_t st = api.session_decode(
                    session_,
                    clamped,
                    EXV_PIXEL_FORMAT_RGBA_F32,
                    reinterpret_cast<uint8_t*>(cached_rgba_.data()),
                    total * sizeof(float),
                    0,
                    &written);
                if (st != 0) {
                    iop->error("exvReader decode failed: %s", api.last_error_message());
                    row.erase(channels);
                    return;
                }
                cached_frame_ = static_cast<int>(clamped);
            }
        }

        const uint32_t exv_row = exv_nuke::nuke_y_to_exv_row(y, info_.height);
        const float* src_row =
            cached_rgba_.data() + static_cast<size_t>(exv_row) * info_.width * 4;

        const Chan chan_order[4] = {Chan_Red, Chan_Green, Chan_Blue, Chan_Alpha};
        for (int c = 0; c < 4; ++c) {
            const Channel ch = chan_order[c];
            if (!(channels & ch)) continue;
            float* dst = row.writable(ch);
            for (int px = x; px < r; ++px) {
                if (px >= 0 && px < static_cast<int>(info_.width)) {
                    dst[px] = src_row[px * 4 + c];
                } else {
                    dst[px] = 0.0f;
                }
            }
        }
    }
};

static Reader* build(Read* iop, int fd, const unsigned char* b, int n) {
    return new ExvReader(iop, fd);
}

static bool test(int fd, const unsigned char* block, int length) {
    return length >= 4 && block[0] == 'E' && block[1] == 'X' && block[2] == 'V' && block[3] == '1';
}

}  // namespace

const Reader::Description d("exv\0exs\0", build, test);
