// Cache keys for every per-file cache of the AE plugin (channel routing, open sessions, decoded frame).
// Header-only and free of AE/Win32 types so `tests/cache_key_test.cpp` can compile and run it stand-alone.
//
// Why ONE helper: the routing cache used to strip a trailing frame number (`take_01.exv` and `take_02.exv`
// collapsed into one entry) while the session/frame caches used the raw path. EXV is a one-file container
// (frame index comes from the comp time, not from numbered files), so a numbered name is NOT a sequence:
// every cache is keyed from the same normalised path.
#pragma once

#include <cstdint>
#include <string>

namespace exvae {

// Identity of a file's CONTENT on disk (last-write time + size). A cached session/frame is reusable only
// while the file still has the identity it was opened with.
struct FileId {
    uint64_t mtime = 0, size = 0;
    bool operator==(const FileId& o) const { return mtime == o.mtime && size == o.size; }
    bool operator!=(const FileId& o) const { return !(*this == o); }
};

// Normalised path used as the footage identity: '\' -> '/' and ASCII case folded (Windows paths are
// case-insensitive and AE may hand the same item over with either spelling). Non-ASCII bytes are kept.
inline std::string NormPath(const std::string& p)
{
    std::string k = p;
    for (char& c : k) {
        if (c == '\\') c = '/';
        else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return k;
}

// Key of the user's channel routing: the footage item (path only). It must survive the file being
// re-rendered/replaced, so it carries no FileId; the routing is validated against the file's channels at use.
inline std::string RouteKey(const std::string& path) { return NormPath(path); }

// Key of content caches (open session, decoded frame): path AND file identity. Together with the global
// cache generation (bumped on invalidate, checked by the publisher) this is "path + mtime + size + gen".
struct FileKey {
    std::string path;   // NormPath()-ed
    FileId id;
    bool operator==(const FileKey& o) const { return path == o.path && id == o.id; }
    bool operator!=(const FileKey& o) const { return !(*this == o); }
};

inline FileKey MakeKey(const std::string& path, const FileId& id) { return FileKey{NormPath(path), id}; }

}  // namespace exvae
