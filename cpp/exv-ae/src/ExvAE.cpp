/*******************************************************************/
/*  ExvAE — EXV (.exv) AEIO plugin via exv_capi.dll.             */
/*                                                                  */
/*  Imports/exports EXV through the exv_capi.dll C ABI. A frame is  */
/*  linear scene-referred RGBA float; named channels (AOVs) route to */
/*  R/G/B/A via the Channel Matrix options dialog.                   */
/*  ONE format: .exv (a still is a 1-frame .exv). Temporal-delta, so */
/*  frames are accumulated by a video encoder and written at the end.*/
/*******************************************************************/

#ifdef AE_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "ExvAE.h"
#include "exv_capi_bridge.h"
#include "exv_cache_key.h"
#include "exv_enc_opts.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <memory>
#include <new>
#include <mutex>
#include <numeric>
#include <vector>
#ifdef AE_OS_WIN
#include <profileapi.h>
#include <commctrl.h>    // ListView channel matrix + InitCommonControlsEx
#pragma comment(lib, "comctl32.lib")
#endif

static AEGP_PluginID S_mem_id = 0;
static exvae::RustBridge g_bridge;

// ---------------------------------------------------------------------------
// Encode session state.
//
// ONE format: a still is a 1-frame .exv, so the movie path and the image-sequence path share everything.
// Streaming encoder (ABI v3+): exv_venc_open (first AddFrame, once the geometry is known) creates a temp file
// next to the target and every AddFrame encodes+appends straight to it (memory = one frame, not the clip);
// EndAdding commits it atomically with exv_venc_finish. Freeing the encoder unfinished deletes the temp (abort).
//
// The movie module (VIDEO) drives ONE EncState (`g_enc`): AE serializes a single output module's
// StartAdding/AddFrame/EndAdding sequence, so it needs no lock. The sequence module (STILL) calls OutputFrame
// once per frame; that path uses a LOCAL EncState, so it never touches `g_enc` (a movie render in flight is
// unaffected) and never shares state between frames.
// ---------------------------------------------------------------------------

struct EncState {
    std::string      path;                 // target .exv (UTF-8; goes to Rust RAW)
    ExvVideoEncoder* venc = nullptr;       // opened on the first frame; owns the temp file
    ExvMeta          meta{};               // fps / PAR read from the OutSpec in StartAdding
    A_u_long         frames = 0;           // frames appended so far
    bool             failed = false;       // an AddFrame failed: EndAdding aborts instead of committing
};
static EncState g_enc;

// Abort and reset: a still-open encoder is freed UNFINISHED (temp deleted, an existing target untouched).
static void EncReset(EncState& s)
{
    if (s.venc && g_bridge.venc_free) g_bridge.venc_free(s.venc);
    s = EncState{};
}

// ---------------------------------------------------------------------------
// Decode state shared by AE's render threads
// ---------------------------------------------------------------------------
// exv_session_decode* is safe to call concurrently on ONE session, so `g_cache_mutex` guards only the
// slots below and is NEVER held across a decode or a blit: a render thread copies the shared_ptrs it needs
// under the lock and works on them outside it. Replacing/invalidating a slot just drops the slot's
// reference; the session is closed by the LAST user, so an in-flight decode can never lose its session.

// Reference-counted owner of an open session (closed when the last reference goes away).
using SessionPtr = std::shared_ptr<ExvSession>;

// Channel routing for displayed R/G/B/A: a channel name, or the sentinels "<none>" / "<white>".
using ExvMap = std::array<std::string, 4>;

static SessionPtr WrapSession(ExvSession* raw)
{
    // `module` guards a reference outliving DeathHook's Unload (the function pointers would dangle).
    return SessionPtr(raw, [](ExvSession* p) {
        if (g_bridge.module && g_bridge.session_close) g_bridge.session_close(p);
    });
}

// Cache keys live in exv_cache_key.h (ONE helper for routing, session and frame caches). A cached
// session/frame is reusable only while the file still has the identity (mtime + size) it was opened with:
// another process (or a re-render over the same path) replacing the file must not keep serving old pixels.
using exvae::FileId;
using exvae::FileKey;
using exvae::MakeKey;
using exvae::RouteKey;

static FileId StatFile(const std::string& utf8)
{
    FileId id;
#ifdef AE_OS_WIN
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (n <= 0) return id;
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, w.data(), n);
    WIN32_FILE_ATTRIBUTE_DATA fa{};
    if (GetFileAttributesExW(w.c_str(), GetFileExInfoStandard, &fa)) {
        id.mtime = (static_cast<uint64_t>(fa.ftLastWriteTime.dwHighDateTime) << 32) | fa.ftLastWriteTime.dwLowDateTime;
        id.size  = (static_cast<uint64_t>(fa.nFileSizeHigh) << 32) | fa.nFileSizeLow;
    }
#else
    (void)utf8;
#endif
    return id;
}

// One decoded frame (tight RGBA f32) plus everything that makes it reusable.
struct DecodedFrame {
    FileKey key;                // footage path + file identity
    uint32_t id = 0, w = 0, h = 0;
    ExvMap map;                 // the routing it was decoded with: a route change must re-decode
    std::vector<float> rgba;
};

static std::mutex g_cache_mutex;
static constexpr size_t kMaxSessions = 4;                              // footage items kept open (no slot thrash)
struct SessionSlot { FileKey key; SessionPtr s; };
static std::vector<SessionSlot> g_sessions;                            // MRU at the back
static std::shared_ptr<const DecodedFrame> g_frame;                    // the last decoded frame
// Bumped by every CacheInvalidate. A render thread that started before the bump must not publish its
// session/frame afterwards (it was built from the file that has just been replaced).
static uint64_t g_gen = 0;

static void CacheInvalidate()
{
    // AE runs DrawSparseFrame on worker threads while DisposeInSpec / DeathHook / EndAdding call this;
    // dropping the slots under the lock is safe because in-flight users hold their own references.
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    ++g_gen;
    g_sessions.clear();
    g_frame.reset();
}

static uint64_t CacheGen()
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    return g_gen;
}

// Session for `path` (whose current key is `key`, drawn at cache generation `gen`): the shared one if
// open and still matching the file on disk, else a freshly opened one (opened OUTSIDE the lock: the open
// reads the file). A session opened across a CacheInvalidate is returned but NOT published.
// nullptr = cannot open (the caller logs the library's error).
static SessionPtr GetSession(const std::string& path, const FileKey& key, uint64_t gen)
{
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = std::find_if(g_sessions.begin(), g_sessions.end(),
                               [&](const auto& e) { return e.key.path == key.path; });
        if (it != g_sessions.end() && it->key == key) {
            std::rotate(it, it + 1, g_sessions.end());   // most recently used -> back
            return g_sessions.back().s;
        }
        if (it != g_sessions.end()) g_sessions.erase(it);   // the file changed under us: drop the stale session
    }
    ExvSession* raw = nullptr;
    const int32_t rc = g_bridge.session_open(path.c_str(), &raw);
    if (rc != 0 || !raw) return nullptr;
    SessionPtr s = WrapSession(raw);
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (gen != g_gen) return s;   // invalidated while opening: usable once, never shared
    // Another render thread may have opened the same file meanwhile: share theirs, ours closes on return.
    for (const auto& e : g_sessions)
        if (e.key == key) return e.s;
    if (g_sessions.size() >= kMaxSessions) g_sessions.erase(g_sessions.begin());
    g_sessions.push_back({key, s});
    return s;
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

#ifdef AE_OS_WIN
static double HiResMs()
{
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart) * 1000.0 / static_cast<double>(freq.QuadPart);
}
#else
static double HiResMs() { return 0.0; }
#endif

// ---------------------------------------------------------------------------
// Logging: OutputDebugString ALWAYS (cheap, debugger-only) + an OPT-IN disk log.
// ---------------------------------------------------------------------------
// When the `EXV_AE_TRACE` env var is set (to anything
// non-empty / not starting with '0'), every line is ALSO appended to `<TEMP>\ExvAE.log`
// (flushed per line) so an After Effects import/export trace survives WITHOUT a DebugView
// capture running — the thing you need to diagnose e.g. a slow/hung open. Off by default: a
// host plugin must not write to disk unasked, especially not per-frame in DrawSparseFrame.

#ifdef AE_OS_WIN
// Enabled once per process (magic-static): `EXV_AE_TRACE` present + not "0"/empty.
static bool ExvTraceEnabled()
{
    static const bool enabled = []() {
        char v[8]{};
        return ::GetEnvironmentVariableA("EXV_AE_TRACE", v, (DWORD)sizeof(v)) > 0 && v[0] != '0' && v[0] != '\0';
    }();
    return enabled;
}

// `<TEMP>\ExvAE.log`, opened for append once (the OS temp dir is always present + writable).
// The FILE* intentionally lives for the process lifetime (reclaimed at exit).
static FILE* ExvLogFile()
{
    static FILE* f = []() -> FILE* {
        char dir[MAX_PATH]{};
        DWORD n = ::GetTempPathA((DWORD)sizeof(dir), dir);
        if (n == 0 || n > sizeof(dir)) return nullptr;
        std::string path = std::string(dir) + "ExvAE.log";
        return std::fopen(path.c_str(), "a");
    }();
    return f;
}

static void ExvLog(const char* msg)
{
    OutputDebugStringA("[ExvAE] ");
    OutputDebugStringA(msg ? msg : "");
    OutputDebugStringA("\n");

    if (!ExvTraceEnabled()) return;
    // AEIO callbacks can fire from multiple threads (playback) — serialize the file writes.
    static std::mutex log_mtx;
    std::lock_guard<std::mutex> lk(log_mtx);
    if (FILE* f = ExvLogFile()) {
        std::fprintf(f, "[ExvAE] %s\n", msg ? msg : "");
        std::fflush(f);
    }
}
#else
static void ExvLog(const char* msg) { (void)msg; }
#endif

// ---------------------------------------------------------------------------
// Channel-matrix helpers + per-footage map cache
// Single-part: EXV's AE path has no multipart, so there is no part index.
//
// The cache guards the user's R/G/B/A selection across AE's footage-reload
// cycle (DisposeInSpec -> InitInSpecFromFile): AE disposes and re-inits the
// InSpec options handle, so without this the chosen map reverts to the
// auto-default. Keyed by `RouteKey` (the normalised footage path, exv_cache_key.h): numbered takes
// (`take_01.exv`, `take_02.exv`) are different footage and never share a routing.
// ---------------------------------------------------------------------------

// Find the file channel that best feeds a displayed component (e.g. R/G/B/A):
// exact name match first, then a "layer.X" suffix match (e.g. "beauty.R").
// Returns the matched channel name or nullptr.
static const char* FindBestChannel(const std::vector<std::string>& channels,
                                   const char* upperZ, const char* lowerZ)
{
    for (const std::string& name : channels) {
        if ((upperZ && name == upperZ) || (lowerZ && name == lowerZ)) return name.c_str();
    }
    for (const std::string& name : channels) {
        size_t dot = name.find_last_of('.');
        if (dot != std::string::npos) {
            const char* suffix = name.c_str() + dot + 1;
            if ((upperZ && std::strcmp(suffix, upperZ) == 0) ||
                (lowerZ && std::strcmp(suffix, lowerZ) == 0)) {
                return name.c_str();
            }
        }
    }
    return nullptr;
}

static std::map<std::string, ExvMap> S_channel_cache;
static std::mutex S_cache_mutex;

// Write the R/G/B/A selection into the cache (past 200 entries one is evicted, never all of them).
static void UpdateChannelCache(const std::string& key, const ExvMap& map)
{
    std::lock_guard<std::mutex> lock(S_cache_mutex);
    if (S_channel_cache.size() > 200 && !S_channel_cache.count(key)) S_channel_cache.erase(S_channel_cache.begin());
    S_channel_cache[key] = map;
}

// The cached routing for `key`, if any.
static bool FindChannelCache(const std::string& key, ExvMap* out)
{
    std::lock_guard<std::mutex> lock(S_cache_mutex);
    auto it = S_channel_cache.find(key);
    if (it == S_channel_cache.end()) return false;
    *out = it->second;
    return true;
}

// In-memory form of the InSpec options handle: the POD header + the strings that follow it in the handle
// (see ExvAE.h). Strings are unbounded; nothing is ever truncated.
struct ExvFileInfo {
    ExvFileHeader h{};
    std::string path;                  // footage path when the header was built (informational)
    ExvMap map;                        // routing to displayed R/G/B/A
    std::vector<std::string> chans;    // every channel name of the container, in order
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Convert AE UTF-16 path to UTF-8.
static std::string Utf16ToUtf8(const A_UTF16Char* src)
{
    if (!src) return {};
#ifdef AE_OS_WIN
    int src_len = 0;
    while (src[src_len]) ++src_len;
    if (src_len == 0) return {};
    int needed = WideCharToMultiByte(CP_UTF8, 0,
        reinterpret_cast<const wchar_t*>(src), src_len, NULL, 0, NULL, NULL);
    if (needed <= 0) return {};
    std::string result(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0,
        reinterpret_cast<const wchar_t*>(src), src_len,
        &result[0], needed, NULL, NULL);
    return result;
#else
    return {};
#endif
}

// True if the UTF-8 path ends with ".exv" (case-insensitive). The ONE extension predicate: import
// verification and export both use it. Paths go to exv_capi.dll RAW — the library percent-decodes
// file:// URLs in exactly one place, so NO scheme stripping happens here (a C++ copy would bypass it).
static bool HasExvExt(const std::string& p)
{
    static const char kExt[] = ".exv";
    const size_t n = sizeof(kExt) - 1;
    if (p.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = p[p.size() - n + i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c != kExt[i]) return false;
    }
    return true;
}

// A_Fixed (16.16) fps -> exact rational. AE rounds NTSC rates to 16.16 (29.97 -> 29.9700012), so try the
// broadcast denominators 1 and 1001 first (accepting a 1e-3 fps match); otherwise use the exact fixed ratio.
// Returns false for a non-positive rate (caller leaves the fps unspecified).
static bool FpsToRational(A_Fixed fixed, uint32_t* num, uint32_t* den)
{
    if (fixed <= 0) return false;
    const double fps = static_cast<double>(fixed) / 65536.0;
    for (uint32_t d : {1u, 1001u}) {
        const double n = std::floor(fps * d + 0.5);
        if (n >= 1.0 && std::fabs(n / d - fps) < 1e-3) { *num = static_cast<uint32_t>(n); *den = d; return true; }
    }
    const uint32_t g = std::gcd(static_cast<uint32_t>(fixed), 65536u);
    *num = static_cast<uint32_t>(fixed) / g;
    *den = 65536u / g;
    return true;
}

// Serialize `fi` into the handle layout: fixed ExvFileHeader, then the NUL-terminated string table
// (path, map_r..map_a, channel names). `h.size`/`h.version`/`h.num_channels` are set here.
static std::vector<char> PackFileInfo(ExvFileInfo fi)
{
    std::string table;
    auto put = [&](const std::string& str) { table.append(str); table.push_back('\0'); };
    put(fi.path);
    for (const std::string& m : fi.map) put(m);
    for (const std::string& c : fi.chans) put(c);
    fi.h.version      = EXV_FILE_OPT_VERSION;
    fi.h.num_channels = static_cast<A_u_long>(fi.chans.size());
    fi.h.size         = static_cast<A_u_long>(sizeof(ExvFileHeader) + table.size());
    std::vector<char> out(sizeof(ExvFileHeader) + table.size());
    std::memcpy(out.data(), &fi.h, sizeof(ExvFileHeader));
    std::memcpy(out.data() + sizeof(ExvFileHeader), table.data(), table.size());
    return out;
}

// Parse `p[0..sz)` (a handle's contents) ONLY if it is exactly the current layout: size == total, version
// matches, and every string is NUL-terminated inside the buffer. A project saved by another build has a
// different layout; reading it as the current one would over-read / feed garbage into channel names.
// `full` false = fixed part only (cheap; for the hot per-frame queries).
static bool UnpackFileInfo(const char* p, size_t sz, bool full, ExvFileInfo* out)
{
    if (sz < sizeof(ExvFileHeader)) return false;
    ExvFileHeader h;
    std::memcpy(&h, p, sizeof(h));
    if (h.version != EXV_FILE_OPT_VERSION || h.size != sz) return false;
    out->h = h;
    if (!full) return true;
    const char* cur = p + sizeof(ExvFileHeader);
    const char* const end = p + sz;
    auto next = [&](std::string* dst) -> bool {
        const void* nul = std::memchr(cur, 0, static_cast<size_t>(end - cur));
        if (!nul) return false;
        dst->assign(cur, static_cast<const char*>(nul));
        cur = static_cast<const char*>(nul) + 1;
        return true;
    };
    if (!next(&out->path)) return false;
    for (std::string& m : out->map) if (!next(&m)) return false;
    // Every name takes >= 1 byte (its NUL): a count above the remaining bytes is a forged header.
    if (h.num_channels > static_cast<size_t>(end - cur)) return false;
    out->chans.assign(h.num_channels, std::string());
    for (std::string& c : out->chans) if (!next(&c)) return false;
    return cur == end;
}

// Copy an InSpec options handle into `out` (`full` = with the string table). false = absent or another
// layout/version (nothing left locked). Copying means no handle stays locked across dialogs / decodes.
static bool ReadFileInfo(AEGP_SuiteHandler& suites, AEIO_Handle h, bool full, ExvFileInfo* out)
{
    if (!h) return false;
    AEGP_MemSize sz = 0;
    if (suites.MemorySuite1()->AEGP_GetMemHandleSize(h, &sz) != A_Err_NONE) return false;
    char* p = nullptr;
    if (suites.MemorySuite1()->AEGP_LockMemHandle(h, reinterpret_cast<void**>(&p)) != A_Err_NONE || !p)
        return false;
    const bool ok = UnpackFileInfo(p, static_cast<size_t>(sz), full, out);
    suites.MemorySuite1()->AEGP_UnlockMemHandle(h);
    return ok;
}

// Allocate a NEW handle holding `fi` (header + string table).
static A_Err NewFileInfoHandle(AEGP_SuiteHandler& suites, const char* what, const ExvFileInfo& fi,
                               AEIO_Handle* outH)
{
    const std::vector<char> bytes = PackFileInfo(fi);
    A_Err err = A_Err_NONE;
    AEIO_Handle newH = nullptr;
    ERR(suites.MemorySuite1()->AEGP_NewMemHandle(S_mem_id, what, static_cast<AEGP_MemSize>(bytes.size()),
        AEGP_MemFlag_CLEAR, &newH));
    if (err) return err;
    if (!newH) return A_Err_ALLOC;
    void* dst = nullptr;
    ERR(suites.MemorySuite1()->AEGP_LockMemHandle(newH, &dst));
    if (!err && dst) std::memcpy(dst, bytes.data(), bytes.size());
    suites.MemorySuite1()->AEGP_UnlockMemHandle(newH);
    if (!err && !dst) err = A_Err_ALLOC;
    if (err) { suites.MemorySuite1()->AEGP_FreeMemHandle(newH); return err; } // never handed out -> don't leak
    *outH = newH;
    return err;
}

// The handle's fields as the pure options struct (the dialog / mapping work on this, not on the A_u_long POD).
static exvae::EncOpts OptsOf(const ExvOutputOptions& o)
{
    return exvae::EncOpts{ static_cast<uint32_t>(o.mode), static_cast<uint32_t>(o.dtype_code), static_cast<uint32_t>(o.q) };
}
static void StoreOpts(ExvOutputOptions* p, const exvae::EncOpts& o)
{
    p->version    = EXV_OPT_VERSION;
    p->size       = sizeof(ExvOutputOptions);
    p->mode       = o.mode;
    p->dtype_code = o.dtype;
    p->q          = o.q;
}

// Lock the output-module options handle and return it ONLY if it is exactly the current layout AND holds a
// valid option set (nullptr = absent or stale, nothing left locked; a non-null result MUST be unlocked by the caller).
static ExvOutputOptions* LockOutOptions(AEGP_SuiteHandler& suites, AEIO_Handle h)
{
    if (!h) return nullptr;
    AEGP_MemSize sz = 0;
    if (suites.MemorySuite1()->AEGP_GetMemHandleSize(h, &sz) != A_Err_NONE || sz != sizeof(ExvOutputOptions))
        return nullptr;
    ExvOutputOptions* p = nullptr;
    if (suites.MemorySuite1()->AEGP_LockMemHandle(h, reinterpret_cast<void**>(&p)) != A_Err_NONE || !p)
        return nullptr;
    if (p->size != sizeof(ExvOutputOptions) || p->version != EXV_OPT_VERSION || !exvae::Validate(OptsOf(*p))) {
        suites.MemorySuite1()->AEGP_UnlockMemHandle(h);
        return nullptr;
    }
    return p;
}

static A_Err GetInSpecPathUtf8(AEIO_BasicData* bd, AEIO_InSpecH specH, std::string* out)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(bd->pica_basicP);
    AEGP_MemHandle pathH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecFilePath(specH, &pathH));
    if (!err && pathH) {
        A_UTF16Char* pathZ = nullptr;
        ERR(suites.MemorySuite1()->AEGP_LockMemHandle(pathH, reinterpret_cast<void**>(&pathZ)));
        if (!err && pathZ)
            *out = Utf16ToUtf8(pathZ);
        suites.MemorySuite1()->AEGP_UnlockMemHandle(pathH);
        suites.MemorySuite1()->AEGP_FreeMemHandle(pathH);
    }
    return err;
}

static A_Err GetWorldPixelFormat(const AEIO_BasicData* bd, const PF_EffectWorld* wP,
                                  PF_PixelFormat* pf)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(bd->pica_basicP);
    const PF_WorldSuite2* ws2 = nullptr;
    ERR(bd->pica_basicP->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2,
                                       reinterpret_cast<const void**>(&ws2)));
    if (!err && ws2) {
        ERR(ws2->PF_GetPixelFormat(wP, pf));
        bd->pica_basicP->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
    }
    return err;
}

// Exception firewall: AE calls every callback we hand it from its own (often render) threads and has no
// handler, so a C++ exception (std::bad_alloc on a huge frame, std::length_error, ...) escaping would
// terminate AE. `Guarded` runs `fn` and maps an exception to an AE error code (+ log); every AEIO callback
// is registered through `SAFE`, and EntryPointFunc / DeathHook / the dialog proc use `Guarded` directly.
// Logging from a catch handler must not itself throw (string building can hit bad_alloc again).
static void LogCaught(const char* what, const char* detail) noexcept
{
    try { ExvLog((std::string("EXCEPTION in ") + what + ": " + detail).c_str()); } catch (...) {}
}

template <typename F>
static A_Err Guarded(const char* what, F&& fn)
{
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        LogCaught(what, "std::bad_alloc (out of memory)");
        return A_Err_ALLOC;
    } catch (const std::exception& e) {
        LogCaught(what, e.what());
        return A_Err_GENERIC;
    } catch (...) {
        LogCaught(what, "unknown exception");
        return A_Err_GENERIC;
    }
}

// `Safe<My_X>::call` has My_X's exact signature, so it can be stored in AE's function block as is.
template <auto Fn> struct Safe;
template <typename... Args, A_Err (*Fn)(Args...)>
struct Safe<Fn> {
    static A_Err call(Args... args) { return Guarded("AEIO callback", [&] { return Fn(args...); }); }
};
#define SAFE(fn) (&Safe<fn>::call)

// ---------------------------------------------------------------------------
// DeathHook
// ---------------------------------------------------------------------------

static A_Err DeathHookImpl(AEGP_GlobalRefcon, AEGP_DeathRefcon)
{
    CacheInvalidate();
    EncReset(g_enc);  // a leaked encoder aborts (temp deleted)
    ExvLog("DeathHook");
    g_bridge.Unload(); // explicit, outside the loader lock (the bridge has no destructor-unload)
    return A_Err_NONE;
}
static A_Err DeathHook(AEGP_GlobalRefcon g, AEGP_DeathRefcon d)
{
    return Guarded("DeathHook", [&] { return DeathHookImpl(g, d); });
}

// ---------------------------------------------------------------------------
// InitInSpecFromFile
// ---------------------------------------------------------------------------

// Open `path` and build the InSpec options header: stream geometry, the authored fps / pixel aspect /
// alpha interpretation (24/1 only when the file records no fps), the channel list, and the default
// (or per-footage cached) R/G/B/A routing. The session is opened and closed here. false = the file
// cannot be read (logged). Shared by InitInSpecFromFile and by InflateOptions' stale-layout rebuild.
// Fetch channel `ci`'s name WITHOUT truncation: a too-small buffer makes the library fail and report the
// size it needs, so retry once with exactly that. false = the library failed for another reason.
static bool GetChannelName(ExvSession* s, uint32_t ci, std::string* out)
{
    std::vector<char> buf(256);
    size_t need = 0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (g_bridge.channel_name(s, ci, buf.data(), buf.size(), &need) == 0) {
            out->assign(buf.data());   // NUL-terminated by the library
            return true;
        }
        if (need <= buf.size()) break;   // not a "buffer too small" failure
        buf.resize(need);
    }
    return false;
}

static std::string LastError()
{
    const char* e = g_bridge.last_error ? g_bridge.last_error() : nullptr;
    return e ? e : "";
}

// Open `path` and build the InSpec options: stream geometry, the authored fps / pixel aspect /
// alpha interpretation (24/1 only when the file records no fps), the channel list, and the default
// (or per-footage cached) R/G/B/A routing. The session is opened and closed here. false = the file
// cannot be read (logged). Shared by InitInSpecFromFile and by InflateOptions' stale-layout rebuild.
static bool BuildFileHeader(const std::string& path, ExvFileInfo& fi)
{
    ExvSession* raw = nullptr;
    const int32_t rc = g_bridge.session_open(path.c_str(), &raw);
    if (rc != 0 || !raw) {
        ExvLog(("ERR: session_open rc=" + std::to_string(rc) + ": " + LastError()).c_str());
        return false;
    }
    const SessionPtr session = WrapSession(raw);   // closed on every return path

    ExvStreamInfo sinfo{};
    ExvMeta meta{};
    meta.size = sizeof(meta);
    if (g_bridge.session_info(session.get(), &sinfo) != 0 || g_bridge.session_meta(session.get(), &meta) != 0) {
        ExvLog(("ERR: session_info/meta failed: " + LastError()).c_str());
        return false;
    }
    { char buf[256]; std::snprintf(buf, sizeof(buf),
        "InitInSpec: %ux%u frames=%u chans=%u",
        sinfo.width, sinfo.height, sinfo.frame_count, sinfo.num_channels); ExvLog(buf); }

    fi = ExvFileInfo{};
    ExvFileHeader& h = fi.h;
    fi.path        = path;
    h.widthLu       = sinfo.width;
    h.heightLu      = sinfo.height;
    h.frame_countLu = sinfo.frame_count > 0 ? sinfo.frame_count : 1;
    h.bitdepthS     = 128;   // ARGB 32bpc float
    h.par_numL      = 1;
    h.par_denL      = 1;
    h.alpha_type    = meta.alpha;

    // Authored playback rate; AE's own default (24/1) only when the file records none.
    h.fpsT.value = 24;
    h.fpsT.scale = 1;
    if ((meta.flags & EXV_META_FLAG_FPS) && meta.fps_num <= INT_MAX && meta.fps_den <= INT_MAX) {
        h.fpsT.value = static_cast<A_long>(meta.fps_num);
        h.fpsT.scale = static_cast<A_u_long>(meta.fps_den);
    } else {
        ExvLog("InitInSpec: file records no fps -> 24/1");
    }
    if ((meta.flags & EXV_META_FLAG_PAR) && meta.par_num <= INT_MAX && meta.par_den <= INT_MAX) {
        h.par_numL = static_cast<A_long>(meta.par_num);
        h.par_denL = static_cast<A_long>(meta.par_den);
    }

    // Duration = frame_count / fps = frame_count * fps_den / fps_num (seconds as A_Time value/scale).
    const int64_t dur = static_cast<int64_t>(h.frame_countLu) * h.fpsT.scale;
    if (dur > INT_MAX) {
        ExvLog("ERR: clip duration overflows A_Time");
        return false;
    }
    h.durationT.value = static_cast<A_long>(dur);
    h.durationT.scale = static_cast<A_u_long>(h.fpsT.value);

    // --- Channel matrix: enumerate the container's channels (full names, never truncated) + compute a
    // default R/G/B/A routing (name match; Y-luma fallback across RGB; A=<white>). The SeqOptions
    // dialog (My_SeqOptionsDlg) lets the user override these; then any cached user selection is
    // restored over the auto-default. (Single-part: no part index.)
    uint32_t chan_count = 0;
    if (g_bridge.channel_count(session.get(), &chan_count) != 0) {
        ExvLog(("ERR: channel_count failed: " + LastError()).c_str());
        return false;
    }
    fi.chans.resize(chan_count);
    std::vector<std::string> names;   // the non-empty ones (candidates for the default routing)
    names.reserve(chan_count);
    for (uint32_t ci = 0; ci < chan_count; ++ci) {
        if (!GetChannelName(session.get(), ci, &fi.chans[ci])) {
            ExvLog(("ERR: channel_name(" + std::to_string(ci) + ") failed: " + LastError()).c_str());
            return false;
        }
        if (!fi.chans[ci].empty()) names.push_back(fi.chans[ci]);
    }

    const char* def_r = FindBestChannel(names, "R", "r");
    const char* def_g = FindBestChannel(names, "G", "g");
    const char* def_b = FindBestChannel(names, "B", "b");
    const char* def_a = FindBestChannel(names, "A", "a");
    if (!def_r && !def_g && !def_b) {
        const char* y = FindBestChannel(names, "Y", "y");
        if (y) { def_r = y; def_g = y; def_b = y; }
    }
    fi.map = { def_r ? def_r : "<none>", def_g ? def_g : "<none>", def_b ? def_b : "<none>",
               def_a ? def_a : "<white>" };

    // Restore the user's cached selection (validated against this file: an absent channel -> <none>).
    ExvMap cached;
    const bool have_cached = FindChannelCache(RouteKey(path), &cached);
    if (have_cached) {
        static const char kSlot[4] = { 'r', 'g', 'b', 'a' };
        for (size_t k = 0; k < cached.size(); ++k) {
            const std::string& nm = cached[k];
            const bool exists = nm.empty() || nm == "<none>" || nm == "<white>" ||
                std::find(fi.chans.begin(), fi.chans.end(), nm) != fi.chans.end();
            if (exists) {
                fi.map[k] = nm;
            } else {
                ExvLog(("InitInSpec: cached map_" + std::string(1, kSlot[k]) + " '" + nm +
                        "' not in file, reset to <none>").c_str());
                fi.map[k] = "<none>";
            }
        }
        ExvLog("InitInSpec: channel map restored from cache");
    }
    ExvLog(("InitInSpec: cache_key='" + RouteKey(path) + "' hit=" + std::to_string((int)have_cached) +
            " final_map R=" + fi.map[0] + " G=" + fi.map[1] + " B=" + fi.map[2] + " A=" + fi.map[3]).c_str());
    return true;
}

// Install `fi` as the InSpec options handle (a fresh allocation) and free the previous handle AE
// returns (else it leaks one handle per InitInSpec/Inflate: project load / duplicate / undo).
static A_Err SetFileHeader(AEGP_SuiteHandler& suites, AEIO_InSpecH specH, const ExvFileInfo& fi)
{
    AEIO_Handle newH = nullptr;
    A_Err err = NewFileInfoHandle(suites, "ExvAE options", fi, &newH);
    if (err) return err;
    AEIO_Handle old_optionsH = nullptr;
    err = suites.IOInSuite4()->AEGP_SetInSpecOptionsHandle(specH, newH, reinterpret_cast<void**>(&old_optionsH));
    if (err) { suites.MemorySuite1()->AEGP_FreeMemHandle(newH); return err; } // never installed -> don't leak
    if (old_optionsH) suites.MemorySuite1()->AEGP_FreeMemHandle(old_optionsH);
    return err;
}

static A_Err My_InitInSpecFromFile(
    AEIO_BasicData* basic_dataP,
    const A_UTF16Char* file_pathZ,
    AEIO_InSpecH specH)
{
    ExvLog("InitInSpecFromFile");
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);

    if (!g_bridge.Load()) {
        ExvLog(("ERR: " + g_bridge.load_error).c_str());
        return AEIO_Err_BAD_FILENAME;
    }

    // RAW path (Rust normalizes file:// URLs): see HasExvExt.
    std::string path = Utf16ToUtf8(file_pathZ);
    if (path.empty()) return AEIO_Err_BAD_FILENAME;
    { char buf[512]; std::snprintf(buf, sizeof(buf), "InitInSpec: path=%s", path.c_str()); ExvLog(buf); }

    ExvFileInfo fileH;
    if (!BuildFileHeader(path, fileH)) return AEIO_Err_BAD_FILENAME;

    // Set InSpec dimensions, fps, depth, duration, PAR, alpha interpretation.
    ERR(suites.IOInSuite4()->AEGP_SetInSpecDimensions(specH,
        static_cast<A_long>(fileH.h.widthLu), static_cast<A_long>(fileH.h.heightLu)));
    ERR(suites.IOInSuite4()->AEGP_SetInSpecDuration(specH, &fileH.h.durationT));

    A_Fixed fps_fixed = static_cast<A_Fixed>(
        static_cast<double>(fileH.h.fpsT.value) * 65536.0 / static_cast<double>(fileH.h.fpsT.scale) + 0.5);
    ERR(suites.IOInSuite4()->AEGP_SetInSpecNativeFPS(specH, fps_fixed));
    ERR(suites.IOInSuite4()->AEGP_SetInSpecDepth(specH, fileH.h.bitdepthS));

    A_Ratio par = { fileH.h.par_numL, static_cast<A_u_long>(fileH.h.par_denL) };
    ERR(suites.IOInSuite4()->AEGP_SetInSpecHSF(specH, &par));

    if (fileH.h.alpha_type != 0) { // authored interpretation (1 straight, 2 premultiplied); 0 = leave AE's default
        AEIO_AlphaLabel alpha;
        AEFX_CLR_STRUCT(alpha);
        alpha.version = AEIO_AlphaLabel_VERSION;
        alpha.alpha   = (fileH.h.alpha_type == 2) ? AEIO_Alpha_PREMUL : AEIO_Alpha_STRAIGHT;
        alpha.flags   = (fileH.h.alpha_type == 2) ? AEIO_AlphaPremul : 0;
        ERR(suites.IOInSuite4()->AEGP_SetInSpecAlphaLabel(specH, &alpha));
    }

    ERR(SetFileHeader(suites, specH, fileH));

    ExvLog("InitInSpecFromFile OK");
    return err;
}

static A_Err My_InitInSpecInteractive(AEIO_BasicData* bd, AEIO_InSpecH specH)
{
    (void)bd; (void)specH;
    return A_Err_NONE;
}

// ---------------------------------------------------------------------------
// DisposeInSpec
// ---------------------------------------------------------------------------

static A_Err My_DisposeInSpec(AEIO_BasicData* basic_dataP, AEIO_InSpecH specH)
{
    CacheInvalidate();
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecOptionsHandle(specH,
        reinterpret_cast<void**>(&optionsH)));
    if (!err && optionsH)
        ERR(suites.MemorySuite1()->AEGP_FreeMemHandle(optionsH));
    return err;
}

// ---------------------------------------------------------------------------
// FlattenOptions / InflateOptions (layout-checked; the string table travels with the header)
// ---------------------------------------------------------------------------

// The options handle is immutable once installed; the user's channel routing lives in the process cache
// (SeqOptionsDlg writes it there). Flatten persists THAT routing, so saving the project keeps it.
static A_Err My_FlattenOptions(AEIO_BasicData* basic_dataP, AEIO_InSpecH specH,
                                AEIO_Handle* flat_optionsPH)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecOptionsHandle(specH,
        reinterpret_cast<void**>(&optionsH)));
    if (err || !optionsH || !flat_optionsPH) return err;

    ExvFileInfo fi;
    if (!ReadFileInfo(suites, optionsH, true, &fi)) {
        ExvLog("FlattenOptions ERR: options handle missing or another layout");
        return A_Err_GENERIC;
    }
    std::string path;
    if (GetInSpecPathUtf8(basic_dataP, specH, &path) != A_Err_NONE || path.empty()) path = fi.path;
    if (!path.empty()) FindChannelCache(RouteKey(path), &fi.map);
    return NewFileInfoHandle(suites, "ExvAE flat", fi, flat_optionsPH);
}

static A_Err My_InflateOptions(AEIO_BasicData* basic_dataP, AEIO_InSpecH specH,
                                AEIO_Handle flat_optionsH)
{
    A_Err err = A_Err_NONE;
    if (!flat_optionsH) return err;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);

    // Take the saved bytes ONLY if they are exactly the current layout (handle size + version + string
    // table). A project saved by another build has a different struct; reinterpreting it would over-read
    // the handle and feed garbage into the channel names — so rebuild from the file instead (the channel
    // map falls back to the cache / auto-default).
    ExvFileInfo fileH;
    std::string path;
    if (!ReadFileInfo(suites, flat_optionsH, true, &fileH)) {
        ExvLog("InflateOptions: saved options have another layout/version; rebuilding from the file");
        if (!g_bridge.Load()) {
            ExvLog(("ERR: " + g_bridge.load_error).c_str());
            return AEIO_Err_BAD_FILENAME;
        }
        ERR(GetInSpecPathUtf8(basic_dataP, specH, &path));
        if (err || path.empty() || !BuildFileHeader(path, fileH)) return AEIO_Err_BAD_FILENAME;
    }

    ERR(SetFileHeader(suites, specH, fileH));
    if (err) return err;

    // Seed the process cache from the inflated (persisted) map so decode + the dialog pick it
    // up on project load (AE does not re-run InitInSpec here). Keyed by the LIVE footage path. ONLY if
    // there is no live cache entry for this footage — never clobber the user's current selection with a
    // (possibly default/stale) inflated map.
    if (path.empty() && GetInSpecPathUtf8(basic_dataP, specH, &path) != A_Err_NONE) path.clear();
    if (path.empty()) path = fileH.path;
    if (!path.empty()) {
        const std::string k = RouteKey(path);
        ExvMap live;
        if (!FindChannelCache(k, &live)) UpdateChannelCache(k, fileH.map);
    }
    return err;
}

static A_Err My_SynchInSpec(AEIO_BasicData* bd, AEIO_InSpecH specH, A_Boolean* changed0)
{
    if (changed0) *changed0 = FALSE;
    (void)bd; (void)specH;
    return A_Err_NONE;
}

static A_Err My_GetActiveExtent(AEIO_BasicData* bd, AEIO_InSpecH specH,
                                 const A_Time* tr, A_LRect* extent)
{
    (void)tr;
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(bd->pica_basicP);
    A_long w = 0, h = 0;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecDimensions(specH, &w, &h));
    if (!err && extent) {
        extent->left = 0; extent->top = 0;
        extent->right = w; extent->bottom = h;
    }
    return err;
}

// ---------------------------------------------------------------------------
// GetInSpecInfo
// ---------------------------------------------------------------------------

static A_Err My_GetInSpecInfo(AEIO_BasicData* basic_dataP, AEIO_InSpecH specH,
                               AEIO_Verbiage* verbiageP)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    if (!verbiageP) return A_Err_STRUCT;

    suites.ANSICallbacksSuite1()->strcpy(verbiageP->name, "");
    suites.ANSICallbacksSuite1()->strcpy(verbiageP->type, "EXV Video");

    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecOptionsHandle(specH,
        reinterpret_cast<void**>(&optionsH)));
    if (!err && optionsH) {
        ExvFileInfo fi;
        if (ReadFileInfo(suites, optionsH, false, &fi)) {
            char info[256];
            std::snprintf(info, sizeof(info), "%lux%lu %lu frames",
                (unsigned long)fi.h.widthLu, (unsigned long)fi.h.heightLu,
                (unsigned long)fi.h.frame_countLu);
            suites.ANSICallbacksSuite1()->strcpy(verbiageP->sub_type, info);
        } else {
            ExvLog("GetInSpecInfo: options handle missing or another layout");
        }
    }
    return err;
}

// ---------------------------------------------------------------------------
// DrawSparseFrame: decode and blit one frame into the PF_EffectWorld.
//
// EXV decode -> interleaved RGBA f32 (linear), row-major, tight.
// AE pixel structs are ARGB order (PF_PixelFloat.red/green/blue/alpha).
// Alpha is opaque (1.0) unless the stream has 4 channels (RGBA), in which case
// the decoded alpha channel is used.
// ---------------------------------------------------------------------------

static A_Err My_DrawSparseFrame(
    AEIO_BasicData* basic_dataP,
    AEIO_InSpecH specH,
    const AEIO_DrawSparseFramePB* sparse_framePPB,
    PF_EffectWorld* wP,
    AEIO_DrawingFlags* draw_flagsP)
{
    double t0 = HiResMs();
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    if (!sparse_framePPB || !wP) { ExvLog("DrawSparseFrame: NULL args"); return A_Err_STRUCT; }
    if (draw_flagsP) *draw_flagsP = 0;

    if (!g_bridge.Load()) { ExvLog(("ERR: " + g_bridge.load_error).c_str()); return AEIO_Err_BAD_FILENAME; }

    std::string path_utf8;
    ERR(GetInSpecPathUtf8(basic_dataP, specH, &path_utf8));
    if (err || path_utf8.empty()) return AEIO_Err_BAD_FILENAME;

    // Read options.
    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecOptionsHandle(specH,
        reinterpret_cast<void**>(&optionsH)));
    ExvFileInfo fileH;
    if (err) return err;
    if (!ReadFileInfo(suites, optionsH, true, &fileH)) {
        ExvLog("DrawSparseFrame ERR: options handle missing or another layout");
        return AEIO_Err_BAD_FILENAME;
    }

    // Apply the user's cached channel routing (keyed by path) OVER the header map. The SeqOptions
    // dialog updates S_channel_cache, but the options handle is immutable and AE never re-runs
    // InitInSpecFromFile after the dialog, so that handle keeps the default map. The cache is the
    // authoritative per-footage routing — without this the decode uses the stale default and the
    // picture never reflects the matrix.
    // The footage may have been replaced by a file with other channels since the routing was chosen: an
    // overlay naming a channel THIS file does not have is ignored (the file's own map stays), instead of
    // resolving to a silent black picture.
    {
        ExvMap cached;
        if (FindChannelCache(RouteKey(path_utf8), &cached)) {
            const auto known = [&](const std::string& n) {
                return n == "<none>" || n == "<white>" ||
                       std::find(fileH.chans.begin(), fileH.chans.end(), n) != fileH.chans.end();
            };
            if (std::all_of(cached.begin(), cached.end(), known)) fileH.map = cached;
            else ExvLog("DrawSparseFrame: cached routing names channels this file lacks -> using the file's own map");
        }
    }
    const FileKey file_key = MakeKey(path_utf8, StatFile(path_utf8));
    const uint64_t gen = CacheGen();

    // Compute frame index from A_Time at the file's fps.
    const A_Time& tr = sparse_framePPB->tr;
    uint32_t frame_id = 0;
    if (tr.scale > 0 && fileH.h.fpsT.value > 0 && fileH.h.fpsT.scale > 0) {
        double time_sec = static_cast<double>(tr.value) / static_cast<double>(tr.scale);
        double fps = static_cast<double>(fileH.h.fpsT.value) / static_cast<double>(fileH.h.fpsT.scale);
        int64_t fid = static_cast<int64_t>(time_sec * fps + 0.5);
        if (fid < 0) fid = 0;
        if (fileH.h.frame_countLu > 0 && fid >= static_cast<int64_t>(fileH.h.frame_countLu))
            fid = static_cast<int64_t>(fileH.h.frame_countLu) - 1;
        frame_id = static_cast<uint32_t>(fid);
    }

    uint32_t w = fileH.h.widthLu;
    uint32_t h = fileH.h.heightLu;
    if (w == 0 || h == 0) return AEIO_Err_BAD_FILENAME;

    { char buf[256]; std::snprintf(buf, sizeof(buf),
        "DrawSparseFrame: frame=%u time=%ld/%lu src=%ux%u world=%ldx%ld",
        frame_id, (long)tr.value, (unsigned long)tr.scale,
        w, h, (long)wP->width, (long)wP->height);
      ExvLog(buf); }
    { char mb[512]; std::snprintf(mb, sizeof(mb),
        "DrawSparseFrame: map R=%s G=%s B=%s A=%s", fileH.map[0].c_str(), fileH.map[1].c_str(),
        fileH.map[2].c_str(), fileH.map[3].c_str());
      ExvLog(mb); }

    double t_open = HiResMs();
    // Shared session (opened outside the cache lock). The lock is never held across decode or blit.
    const SessionPtr sess = GetSession(path_utf8, file_key, gen);
    double t_opened = HiResMs(), t_decoded = t_opened;
    if (!sess) {
        ExvLog(("DrawSparseFrame ERR: session_open path=" + path_utf8 + ": " + LastError()).c_str());
        return AEIO_Err_BAD_FILENAME;
    }

    // Per-frame geometry: a .exv may carry variable-dimension frames, so the
    // stream dims (frame 0, captured in InitInSpecFromFile) can be wrong for a
    // later frame -> wrong buffer size -> black/garbage. Query the actual frame
    // size from the (now-open) session and size the decode buffer + blit to it.
    // Falls back to the stream dims only if the query fails.
    {
        ExvFrameInfo fi{};
        if (g_bridge.session_frame_info(sess.get(), frame_id, &fi) == 0 &&
            fi.width > 0 && fi.height > 0) {
            w = fi.width;
            h = fi.height;
        }
    }

    // Decode or hit the frame cache. The cached frame is shared (shared_ptr): the lock only guards the slot.
    std::shared_ptr<const DecodedFrame> frame;
    bool cache_hit = false;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (g_frame && g_frame->key == file_key && g_frame->id == frame_id &&
            g_frame->w == w && g_frame->h == h && g_frame->map == fileH.map) {
            frame = g_frame;
            cache_hit = true;
        }
    }
    if (!frame) {
        auto f = std::make_shared<DecodedFrame>();
        f->key  = file_key;
        f->id   = frame_id;
        f->w    = w;
        f->h    = h;
        f->map  = fileH.map;
        f->rgba.resize(static_cast<size_t>(w) * h * 4);

        size_t written = 0;
        double t_dec = HiResMs();
        // Route the requested named channels into displayed RGBA via the channel
        // matrix (map entries are channel names or the "<none>"/"<white>" sentinels,
        // resolved inside the FFI). Replaces the fixed session_decode.
        int32_t rc = g_bridge.decode_mapped(
            sess.get(), frame_id,
            EXV_PIXEL_FORMAT_RGBA_F32,
            reinterpret_cast<uint8_t*>(f->rgba.data()),
            f->rgba.size() * sizeof(float), /*stride=*/0,
            f->map[0].c_str(), f->map[1].c_str(), f->map[2].c_str(), f->map[3].c_str(),
            &written);
        t_decoded = HiResMs();

        if (rc != 0) {
            char buf[256]; std::snprintf(buf, sizeof(buf),
                "DrawSparseFrame ERR: decode_mapped rc=%d frame=%u (%.1fms): %s",
                rc, frame_id, t_decoded - t_dec, LastError().c_str());
            ExvLog(buf);
            return AEIO_Err_PARSING;
        }
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            if (gen == g_gen) g_frame = f;   // decoded across a CacheInvalidate -> use once, do not publish
        }
        frame = f;
    }
    const float* rgba_f32 = frame->rgba.data();

    // decode_mapped always fills the alpha slot (a channel, <white>=1.0, or
    // <none>=0.0), so the decoded alpha is authoritative regardless of the
    // container's channel count.
    const bool has_alpha = true;

    // Blit into PF_EffectWorld. AE ARGB order: .alpha .red .green .blue.
    // Our buffer is tight RGBA f32 (R=idx+0, G=+1, B=+2, A=+3).
    PF_PixelFormat pf = PF_PixelFormat_ARGB128;
    // The world's actual bit depth selects the row stride below; guessing ARGB128
    // on an 8/16bpc world would over-read each row by 2-4x and walk off the
    // buffer's last row (OOB). Hard-fail rather than assume a format.
    ERR(GetWorldPixelFormat(basic_dataP, wP, &pf));
    if (err) return err;

    A_long out_w = wP->width;
    A_long out_h = wP->height;
    double t_blit = HiResMs();

    // Nearest-neighbour scale the decoded FILE-resolution frame (w×h) into the world (out_w×out_h).
    // AE hands us the world at the REQUESTED resolution (Full / Half / Quarter / thumbnail); a 1:1
    // copy would fill it with the top-left crop at full scale — e.g. in Half the image looks 2× big.
    for (A_long oy = 0; oy < out_h; ++oy) {
        A_long sy = out_h > 0 ? static_cast<A_long>(static_cast<int64_t>(oy) * static_cast<int64_t>(h) / out_h) : 0;
        if (sy >= static_cast<A_long>(h)) sy = static_cast<A_long>(h) - 1;
        if (pf == PF_PixelFormat_ARGB128) {
            PF_PixelFloat* row = reinterpret_cast<PF_PixelFloat*>(
                reinterpret_cast<char*>(wP->data) + static_cast<size_t>(oy) * wP->rowbytes);
            for (A_long ox = 0; ox < out_w; ++ox) {
                A_long sx = out_w > 0 ? static_cast<A_long>(static_cast<int64_t>(ox) * static_cast<int64_t>(w) / out_w) : 0;
                if (sx >= static_cast<A_long>(w)) sx = static_cast<A_long>(w) - 1;
                size_t i = (static_cast<size_t>(sy) * w + static_cast<size_t>(sx)) * 4;
                row[ox].red   = rgba_f32[i + 0];
                row[ox].green = rgba_f32[i + 1];
                row[ox].blue  = rgba_f32[i + 2];
                row[ox].alpha = has_alpha ? rgba_f32[i + 3] : 1.0f;
            }
        } else if (pf == PF_PixelFormat_ARGB64) {
            PF_Pixel16* row = reinterpret_cast<PF_Pixel16*>(
                reinterpret_cast<char*>(wP->data) + static_cast<size_t>(oy) * wP->rowbytes);
            auto clamp16 = [](float v) -> A_u_short {
                return static_cast<A_u_short>((std::max)(0.0f, (std::min)(1.0f, v)) * 32768.0f + 0.5f);
            };
            for (A_long ox = 0; ox < out_w; ++ox) {
                A_long sx = out_w > 0 ? static_cast<A_long>(static_cast<int64_t>(ox) * static_cast<int64_t>(w) / out_w) : 0;
                if (sx >= static_cast<A_long>(w)) sx = static_cast<A_long>(w) - 1;
                size_t i = (static_cast<size_t>(sy) * w + static_cast<size_t>(sx)) * 4;
                row[ox].red   = clamp16(rgba_f32[i + 0]);
                row[ox].green = clamp16(rgba_f32[i + 1]);
                row[ox].blue  = clamp16(rgba_f32[i + 2]);
                row[ox].alpha = has_alpha ? clamp16(rgba_f32[i + 3]) : 32768;
            }
        } else {
            PF_Pixel8* row = reinterpret_cast<PF_Pixel8*>(
                reinterpret_cast<char*>(wP->data) + static_cast<size_t>(oy) * wP->rowbytes);
            auto clamp8 = [](float v) -> A_u_char {
                return static_cast<A_u_char>((std::max)(0.0f, (std::min)(1.0f, v)) * 255.0f + 0.5f);
            };
            for (A_long ox = 0; ox < out_w; ++ox) {
                A_long sx = out_w > 0 ? static_cast<A_long>(static_cast<int64_t>(ox) * static_cast<int64_t>(w) / out_w) : 0;
                if (sx >= static_cast<A_long>(w)) sx = static_cast<A_long>(w) - 1;
                size_t i = (static_cast<size_t>(sy) * w + static_cast<size_t>(sx)) * 4;
                row[ox].red   = clamp8(rgba_f32[i + 0]);
                row[ox].green = clamp8(rgba_f32[i + 1]);
                row[ox].blue  = clamp8(rgba_f32[i + 2]);
                row[ox].alpha = has_alpha ? clamp8(rgba_f32[i + 3]) : 255;
            }
        }
    }

    double t_done = HiResMs();
    { char buf[512]; std::snprintf(buf, sizeof(buf),
        "DrawSparseFrame OK: frame=%u %s open=%.1fms dec=%.1fms blit=%.1fms total=%.1fms",
        frame_id, cache_hit ? "CACHED" : "decoded",
        t_opened - t_open, t_decoded - t_opened, t_done - t_blit, t_done - t0);
      ExvLog(buf); }

    return err;
}

// ---------------------------------------------------------------------------
// GetDimensions / GetDuration / GetTime / GetSound / InqNextFrameTime
// ---------------------------------------------------------------------------

static A_Err My_GetDimensions(AEIO_BasicData* bd, AEIO_InSpecH specH,
                               const AEIO_RationalScale* rs0, A_long* w0, A_long* h0)
{
    (void)rs0;
    return bd ? AEGP_SuiteHandler(bd->pica_basicP).IOInSuite4()
                    ->AEGP_GetInSpecDimensions(specH, w0, h0) : A_Err_STRUCT;
}

static A_Err My_GetDuration(AEIO_BasicData* bd, AEIO_InSpecH specH, A_Time* tr)
{
    return bd ? AEGP_SuiteHandler(bd->pica_basicP).IOInSuite4()
                    ->AEGP_GetInSpecDuration(specH, tr) : A_Err_STRUCT;
}

static A_Err My_GetTime(AEIO_BasicData* bd, AEIO_InSpecH specH, A_Time* tr)
{
    if (tr) { tr->value = 0; tr->scale = 1; }
    (void)bd; (void)specH;
    return A_Err_NONE;
}

static A_Err My_GetSound(AEIO_BasicData* bd, AEIO_InSpecH specH, AEIO_SndQuality quality,
    const AEIO_InterruptFuncs* inter0, const A_Time* startPT, const A_Time* durPT,
    A_u_long start_sampLu, A_u_long num_samplesLu, void* dataPV)
{
    (void)bd; (void)specH; (void)quality; (void)inter0;
    (void)startPT; (void)durPT; (void)start_sampLu; (void)num_samplesLu; (void)dataPV;
    return A_Err_NONE;
}

static A_Err My_InqNextFrameTime(AEIO_BasicData* bd, AEIO_InSpecH specH,
    const A_Time* base_time_tr, AEIO_TimeDir time_dir,
    A_Boolean* found0, A_Time* key_time_tr0)
{
    A_Err err = A_Err_NONE;
    if (!found0 || !key_time_tr0 || !base_time_tr) return A_Err_STRUCT;
    *found0 = FALSE;

    AEGP_SuiteHandler suites(bd->pica_basicP);
    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecOptionsHandle(specH,
        reinterpret_cast<void**>(&optionsH)));
    if (err || !optionsH) return err;

    ExvFileInfo fi;
    if (!ReadFileInfo(suites, optionsH, false, &fi)) {
        ExvLog("InqNextFrameTime: options handle missing or another layout");
        return A_Err_NONE;
    }

    A_u_long fc = fi.h.frame_countLu;
    A_long fps_num = fi.h.fpsT.value;
    A_u_long fps_den = fi.h.fpsT.scale;

    if (fc == 0 || fps_num <= 0) return A_Err_NONE;

    double base_sec = static_cast<double>(base_time_tr->value) /
                      static_cast<double>(base_time_tr->scale);
    double fps = static_cast<double>(fps_num) / static_cast<double>(fps_den);
    int64_t cur_frame = static_cast<int64_t>(base_sec * fps + 0.5);

    bool forward = !(time_dir & AEIO_TimeDir_BACKWARD);
    bool include_base = !!(time_dir & AEIO_TimeDir_INCLUDE_BASE_TIME);
    int64_t next_frame = include_base ? cur_frame : (forward ? cur_frame + 1 : cur_frame - 1);

    if (next_frame < 0 || next_frame >= static_cast<int64_t>(fc)) return A_Err_NONE;

    key_time_tr0->value = static_cast<A_long>(next_frame * static_cast<int64_t>(fps_den));
    key_time_tr0->scale = static_cast<A_u_long>(fps_num);
    *found0 = TRUE;
    return A_Err_NONE;
}

// ---------------------------------------------------------------------------
// Output spec (encode path)
// ---------------------------------------------------------------------------

static A_Err My_InitOutputSpec(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH,
                                A_Boolean* user_interacted)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    if (user_interacted) *user_interacted = FALSE;

    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOOutSuite4()->AEGP_GetOutSpecOptionsHandle(outH,
        reinterpret_cast<void**>(&optionsH)));
    // A handle saved by another build (other layout/version) can never be read again: replace it with the
    // defaults here, so reopening Output Module Settings actually repairs the module instead of failing forever.
    AEIO_Handle stale = nullptr;
    if (!err && optionsH) {
        if (ExvOutputOptions* ok = LockOutOptions(suites, optionsH)) {
            (void)ok;
            suites.MemorySuite1()->AEGP_UnlockMemHandle(optionsH);
        } else {
            ExvLog("InitOutputSpec: stale output options handle -> rebuilding with defaults");
            stale = optionsH;
            optionsH = nullptr;
            if (user_interacted) *user_interacted = TRUE;
        }
    }
    if (!err && !optionsH) {
        ERR(suites.MemorySuite1()->AEGP_NewMemHandle(S_mem_id, "ExvAE out opts",
            sizeof(ExvOutputOptions), AEGP_MemFlag_CLEAR, &optionsH));
        if (!err && optionsH) {
            ExvOutputOptions* optP = nullptr;
            ERR(suites.MemorySuite1()->AEGP_LockMemHandle(optionsH,
                reinterpret_cast<void**>(&optP)));
            if (!err && optP) StoreOpts(optP, exvae::DefaultOpts());   // HTJ2K lossless, half float
            suites.MemorySuite1()->AEGP_UnlockMemHandle(optionsH);
            AEIO_Handle old_outH = nullptr;
            ERR(suites.IOOutSuite4()->AEGP_SetOutSpecOptionsHandle(outH, optionsH,
                reinterpret_cast<void**>(&old_outH)));
            // AE hands back the handle it replaced; it is ours to free (it is `stale` unless AE kept none).
            if (!err && stale) suites.MemorySuite1()->AEGP_FreeMemHandle(stale);
        }
    }
    return err;
}

static A_Err My_GetFlatOutputOptions(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH,
                                      AEIO_Handle* new_optionsPH)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    AEIO_Handle old_optionsH = nullptr;
    ERR(suites.IOOutSuite4()->AEGP_GetOutSpecOptionsHandle(outH,
        reinterpret_cast<void**>(&old_optionsH)));
    if (!err && old_optionsH && new_optionsPH) {
        ExvOutputOptions *new_p = nullptr, *old_p = LockOutOptions(suites, old_optionsH);
        if (!old_p) {
            ExvLog("GetFlatOutputOptions ERR: output options handle has another layout/size");
            return A_Err_GENERIC;
        }
        ERR(suites.MemorySuite1()->AEGP_NewMemHandle(S_mem_id, "ExvAE flat out",
            sizeof(ExvOutputOptions), AEGP_MemFlag_CLEAR, new_optionsPH));
        if (!err && *new_optionsPH) {
            ERR(suites.MemorySuite1()->AEGP_LockMemHandle(*new_optionsPH,
                reinterpret_cast<void**>(&new_p)));
            if (!err && new_p) std::memcpy(new_p, old_p, sizeof(ExvOutputOptions));
            suites.MemorySuite1()->AEGP_UnlockMemHandle(*new_optionsPH);
        }
        suites.MemorySuite1()->AEGP_UnlockMemHandle(old_optionsH);
    }
    return err;
}

static A_Err My_DisposeOutputOptions(AEIO_BasicData* basic_dataP, void* optionsPV)
{
    A_Err err = A_Err_NONE;
    if (basic_dataP && optionsPV) {
        AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
        AEIO_Handle optionsH = reinterpret_cast<AEIO_Handle>(optionsPV);
        ERR(suites.MemorySuite1()->AEGP_FreeMemHandle(optionsH));
    }
    return err;
}

static A_Err My_GetOutputInfo(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH,
                               AEIO_Verbiage* verbiageP)
{
    if (!verbiageP) return A_Err_STRUCT;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    suites.ANSICallbacksSuite1()->strcpy(verbiageP->name, "");
    A_Boolean still = FALSE;
    if (suites.IOOutSuite4()->AEGP_GetOutSpecIsStill(outH, &still) != A_Err_NONE) still = FALSE;
    suites.ANSICallbacksSuite1()->strcpy(verbiageP->type, still ? "EXV Sequence" : "EXV Video");

    // Sub-type = the chosen export mode ("HTJ2K lossy, 8/16-bit (9/7), 16-bit integer, q16").
    std::string sub = "exv-capi";
    AEIO_Handle optionsH = nullptr;
    if (suites.IOOutSuite4()->AEGP_GetOutSpecOptionsHandle(outH, reinterpret_cast<void**>(&optionsH)) == A_Err_NONE && optionsH) {
        if (ExvOutputOptions* o = LockOutOptions(suites, optionsH)) {
            sub = exvae::Summary(OptsOf(*o));
            suites.MemorySuite1()->AEGP_UnlockMemHandle(optionsH);
        }
    }
    std::snprintf(verbiageP->sub_type, sizeof(verbiageP->sub_type), "%s", sub.c_str());
    return A_Err_NONE;
}

static A_Err My_OutputInfoChanged(AEIO_BasicData* bd, AEIO_OutSpecH outH)
{
    (void)bd; (void)outH;
    return A_Err_NONE;
}

static A_Err My_SetOutputFile(AEIO_BasicData* bd, AEIO_OutSpecH outH, const A_UTF16Char* file_pathZ)
{
    (void)bd; (void)outH; (void)file_pathZ;
    return AEIO_Err_USE_DFLT_CALLBACK;
}

// Modal "EXV Output Options" dialog (defined with the other Win32 dialog code below): edits `*opts`,
// returns true when the user confirmed a VALID option set.
static bool ShowExvOutDlg(exvae::EncOpts* opts);

// Output Module Settings -> Format Options: edit the options handle IN PLACE (a fixed-size POD; swapping the
// handle inside a callback orphans the one AE holds). Nothing is changed on Cancel.
static A_Err My_UserOptionsDialog(AEIO_BasicData* bd, AEIO_OutSpecH outH,
                                   const PF_EffectWorld* sample0, A_Boolean* user_interacted0)
{
    (void)sample0;
    if (user_interacted0) *user_interacted0 = FALSE;
    A_Err err = My_InitOutputSpec(bd, outH, nullptr);   // guarantees a valid handle (stale ones are replaced)
    if (err) return err;

    AEGP_SuiteHandler suites(bd->pica_basicP);
    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOOutSuite4()->AEGP_GetOutSpecOptionsHandle(outH, reinterpret_cast<void**>(&optionsH)));
    if (err || !optionsH) return err ? err : A_Err_GENERIC;

    exvae::EncOpts opts;
    if (ExvOutputOptions* o = LockOutOptions(suites, optionsH)) {
        opts = OptsOf(*o);
        suites.MemorySuite1()->AEGP_UnlockMemHandle(optionsH);   // nothing stays locked during the modal
    } else {
        ExvLog("UserOptionsDialog ERR: output options handle is invalid after InitOutputSpec");
        return A_Err_GENERIC;
    }
    if (!ShowExvOutDlg(&opts)) return A_Err_NONE;

    ExvOutputOptions* o = LockOutOptions(suites, optionsH);
    if (!o) return A_Err_GENERIC;
    StoreOpts(o, opts);
    suites.MemorySuite1()->AEGP_UnlockMemHandle(optionsH);
    ExvLog(("UserOptionsDialog: " + exvae::Summary(opts)).c_str());
    if (user_interacted0) *user_interacted0 = TRUE;
    return A_Err_NONE;
}

// ---------------------------------------------------------------------------
// Encode: StartAdding / AddFrame / EndAdding / OutputFrame
//
// One path: every frame goes to an exv_venc encoder (opened in the first AddFrame, once the geometry is
// known) which appends it to a temp file; EndAdding commits the temp atomically over the target.
//   movie module    (VIDEO): AE calls StartAdding, AddFrame x N, EndAdding -> ONE .exv with N frames, on `g_enc`.
//   sequence module (STILL): AE calls OutputFrame once per frame with that frame's own file path
//                            (name.0001.exv, ...) -> a 1-frame .exv each, on a LOCAL EncState.
// ---------------------------------------------------------------------------

// Read the output path and clip meta (fps / PAR) from the OutSpec into `st` and reset it for a new file.
static A_Err StartImpl(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH, EncState& st)
{
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);

    if (!g_bridge.Load()) { ExvLog(("ERR: " + g_bridge.load_error).c_str()); return A_Err_GENERIC; }
    EncReset(st);   // an encoder leaked by a render that aborted before EndAdding/DeathHook is aborted here

    // Get output path.
    AEGP_MemHandle pathH = nullptr;
    A_Boolean file_reserved = FALSE;
    ERR(suites.IOOutSuite4()->AEGP_GetOutSpecFilePath(outH, &pathH, &file_reserved));
    if (err || !pathH) return A_Err_GENERIC;
    A_UTF16Char* pathZ = nullptr;
    ERR(suites.MemorySuite1()->AEGP_LockMemHandle(pathH, reinterpret_cast<void**>(&pathZ)));
    if (!err && pathZ) st.path = Utf16ToUtf8(pathZ);
    suites.MemorySuite1()->AEGP_UnlockMemHandle(pathH);
    suites.MemorySuite1()->AEGP_FreeMemHandle(pathH);
    if (err) return err;
    if (st.path.empty()) return AEIO_Err_BAD_FILENAME;
    // The same predicate the import side uses: what we write must be something we would import.
    if (!HasExvExt(st.path)) {
        char buf[512]; std::snprintf(buf, sizeof(buf), "StartAdding ERR: output '%s' is not a .exv path", st.path.c_str());
        ExvLog(buf);
        return AEIO_Err_BAD_FILENAME;
    }

    // Clip metadata from the OutSpec: the playback rate and pixel aspect AE renders at. (Alpha
    // interpretation is left unspecified: the OutSpec exposes no alpha label.)
    st.meta = ExvMeta{};
    st.meta.size = sizeof(ExvMeta);
    {
        A_Fixed fps_fixed = 0;
        uint32_t num = 0, den = 1;
        if (suites.IOOutSuite4()->AEGP_GetOutSpecFPS(outH, &fps_fixed) == A_Err_NONE && FpsToRational(fps_fixed, &num, &den)) {
            st.meta.flags |= EXV_META_FLAG_FPS;
            st.meta.fps_num = num;
            st.meta.fps_den = den;
        }
        A_Ratio hsf = { 0, 1 };
        if (suites.IOOutSuite4()->AEGP_GetOutSpecHSF(outH, &hsf) == A_Err_NONE && hsf.num > 0 && hsf.den > 0) {
            st.meta.flags |= EXV_META_FLAG_PAR;
            st.meta.par_num = static_cast<uint32_t>(hsf.num);
            st.meta.par_den = static_cast<uint32_t>(hsf.den);
        }
    }
    { char buf[512]; std::snprintf(buf, sizeof(buf), "StartAdding OK: path=%s fps=%u/%u",
        st.path.c_str(), st.meta.fps_num, st.meta.fps_den); ExvLog(buf); }
    return err;
}

// The options chosen in Output Module Settings. No handle yet (InitOutputSpec never ran) = the documented
// defaults; a handle with another layout or invalid content (a saved template from another build) is an
// error, never silently reinterpreted.
static A_Err ReadEncOpts(AEGP_SuiteHandler& suites, AEIO_OutSpecH outH, exvae::EncOpts* out)
{
    *out = exvae::DefaultOpts();
    AEIO_Handle optionsH = nullptr;
    if (suites.IOOutSuite4()->AEGP_GetOutSpecOptionsHandle(outH, reinterpret_cast<void**>(&optionsH)) != A_Err_NONE) {
        ExvLog("AddFrame ERR: GetOutSpecOptionsHandle failed");
        return A_Err_GENERIC;
    }
    if (!optionsH) return A_Err_NONE;
    ExvOutputOptions* optP = LockOutOptions(suites, optionsH);
    if (!optP) {
        ExvLog("AddFrame ERR: output options are invalid or from another build; reopen Output Module Settings");
        return A_Err_GENERIC;
    }
    *out = OptsOf(*optP);
    suites.MemorySuite1()->AEGP_UnlockMemHandle(optionsH);
    return A_Err_NONE;
}

// AE world -> tight interleaved RGBA f32 (AE's ARGB order -> codec RGBA).
static A_Err WorldToRgbaF32(AEIO_BasicData* basic_dataP, const PF_EffectWorld* wP, std::vector<float>& rgba)
{
    const A_long w = wP->width, h = wP->height;
    // H4/CLASS-F: do not guess the format. On error, defaulting to ARGB128 for an 8/16bpc world would give
    // a 16-byte stride -> OOB read of the last row. Propagate the error.
    PF_PixelFormat pf = PF_PixelFormat_ARGB128;
    if (GetWorldPixelFormat(basic_dataP, wP, &pf) != A_Err_NONE) {
        ExvLog("AddFrame ERR: GetWorldPixelFormat failed");
        return A_Err_GENERIC;
    }
    rgba.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    for (A_long y = 0; y < h; ++y) {
        const char* rowp = reinterpret_cast<const char*>(wP->data) + static_cast<size_t>(y) * wP->rowbytes;
        float* dst = &rgba[static_cast<size_t>(y) * static_cast<size_t>(w) * 4];
        if (pf == PF_PixelFormat_ARGB128) {
            const PF_PixelFloat* row = reinterpret_cast<const PF_PixelFloat*>(rowp);
            for (A_long x = 0; x < w; ++x, dst += 4) { dst[0] = row[x].red; dst[1] = row[x].green; dst[2] = row[x].blue; dst[3] = row[x].alpha; }
        } else if (pf == PF_PixelFormat_ARGB64) {
            const PF_Pixel16* row = reinterpret_cast<const PF_Pixel16*>(rowp);
            for (A_long x = 0; x < w; ++x, dst += 4) {
                dst[0] = row[x].red / 32768.0f; dst[1] = row[x].green / 32768.0f;
                dst[2] = row[x].blue / 32768.0f; dst[3] = row[x].alpha / 32768.0f;
            }
        } else {
            const PF_Pixel8* row = reinterpret_cast<const PF_Pixel8*>(rowp);
            for (A_long x = 0; x < w; ++x, dst += 4) {
                dst[0] = row[x].red / 255.0f; dst[1] = row[x].green / 255.0f;
                dst[2] = row[x].blue / 255.0f; dst[3] = row[x].alpha / 255.0f;
            }
        }
    }
    return A_Err_NONE;
}

static A_Err AddFrameImpl(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH, EncState& st,
                          A_long frame_index, const PF_EffectWorld* wP)
{
    const double t0 = HiResMs();
    if (!wP || wP->width <= 0 || wP->height <= 0) return A_Err_STRUCT;

    std::vector<float> rgba_f32;
    A_Err err = WorldToRgbaF32(basic_dataP, wP, rgba_f32);
    if (err) return err;

    AEGP_SuiteHandler suites(basic_dataP->pica_basicP);
    exvae::EncOpts opts;
    err = ReadEncOpts(suites, outH, &opts);
    if (err) return err;

    // Open the streaming encoder (gop=1 all-intra, search=0 off) with the clip meta from StartAdding. The
    // path goes to Rust RAW. It creates the temp file next to the target now, so an unwritable folder
    // fails on the first frame instead of after the whole render. Params are mapped in ONE place
    // (exvae::MakeParams) and the library re-validates them.
    if (!st.venc) {
        ExvEncParams params{};
        std::string why;
        if (!exvae::MakeParams(opts, static_cast<uint32_t>(wP->width), static_cast<uint32_t>(wP->height), &params, &why)) {
            ExvLog(("AddFrame ERR: export options: " + why).c_str());
            return A_Err_GENERIC;
        }
        const int32_t rc = g_bridge.venc_open(st.path.c_str(), &params, &st.meta, /*gop=*/1, /*search=*/0, &st.venc);
        if (rc != 0 || !st.venc) {
            char buf[512]; std::snprintf(buf, sizeof(buf),
                "AddFrame ERR: venc_open rc=%d frame=%ld: %s", rc, (long)frame_index,
                g_bridge.last_error ? g_bridge.last_error() : "");
            ExvLog(buf);
            st.venc = nullptr;
            return A_Err_GENERIC;
        }
        ExvLog(("AddFrame: encoding " + exvae::Summary(opts)).c_str());
    }
    const double t_enc = HiResMs();
    const int32_t rc = g_bridge.venc_add_rgba(st.venc, rgba_f32.data(), rgba_f32.size());
    const double t_done = HiResMs();
    if (rc != 0) {
        char buf[512]; std::snprintf(buf, sizeof(buf), "AddFrame ERR: venc_add rc=%d frame=%ld: %s",
            rc, (long)frame_index, g_bridge.last_error ? g_bridge.last_error() : "");
        ExvLog(buf);
        return A_Err_GENERIC;
    }
    { char buf[256]; std::snprintf(buf, sizeof(buf),
        "AddFrame OK: frame=%ld %ldx%ld add=%.1fms total=%.1fms",
        (long)frame_index, (long)wP->width, (long)wP->height, t_done - t_enc, t_done - t0);
      ExvLog(buf); }
    st.frames++;
    return A_Err_NONE;
}

// Commit the streamed clip (index + atomic rename), then free the handle. ALL failures here must propagate
// as an A_Err (else AE reports a successful render with a missing/truncated file). `st` is reset.
static A_Err EndImpl(EncState& st)
{
    A_Err err = A_Err_NONE;
    if (st.failed) {
        // The temp holds an incomplete clip: EncReset below aborts it (deletes the temp, target untouched).
        ExvLog("EndAdding ERR: an AddFrame failed -> aborting the clip, NOT committing a partial file");
        err = A_Err_GENERIC;
    } else if (st.venc) {
        // Release OUR decode session first: it may hold the output file (a footage item being
        // re-rendered over itself), and Windows refuses to rename over a still-mapped file.
        CacheInvalidate();
        uint64_t bytes = 0;
        const int32_t rc = g_bridge.venc_finish(st.venc, &bytes);
        if (rc != 0) {
            char buf[512]; std::snprintf(buf, sizeof(buf),
                "EndAdding ERR: venc_finish rc=%d path=%s err=%s", rc, st.path.c_str(),
                g_bridge.last_error ? g_bridge.last_error() : "");
            ExvLog(buf);
            err = A_Err_GENERIC;
        }
    } else {
        // Zero frames added -> nothing written. Surface it.
        ExvLog("EndAdding ERR: output had zero frames");
        err = A_Err_GENERIC;
    }
    char buf[64]; std::snprintf(buf, sizeof(buf),
        "EndAdding: %lu frames %s", (unsigned long)st.frames, err ? "NOT written" : "written");
    ExvLog(buf);
    EncReset(st);
    return err;
}

static A_Err My_StartAdding(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH, A_long flags)
{
    ExvLog("StartAdding");
    (void)flags;
    return StartImpl(basic_dataP, outH, g_enc);
}

// AE may still call EndAdding after a failed AddFrame; remembering the failure lets EndAdding abort the
// encoder (delete its temp) instead of committing a clip that is missing frames.
static A_Err My_AddFrame(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH,
    A_long frame_index, A_long frames, const PF_EffectWorld* wP,
    const A_LPoint* origin0, A_Boolean was_compressedB, AEIO_InterruptFuncs* inter0)
{
    (void)frames; (void)origin0; (void)was_compressedB; (void)inter0;
    const A_Err err = AddFrameImpl(basic_dataP, outH, g_enc, frame_index, wP);
    if (err != A_Err_NONE) g_enc.failed = true;
    return err;
}

static A_Err My_EndAdding(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_long flags)
{
    ExvLog("EndAdding");
    (void)bd; (void)outH; (void)flags;
    return EndImpl(g_enc);
}

// Sequence module (STILL): one call per frame, the OutSpec's file path is that frame's own file. The whole
// Start/Add/End cycle runs on a local EncState, so the file is complete (or absent) when this returns.
static A_Err My_OutputFrame(AEIO_BasicData* basic_dataP, AEIO_OutSpecH outH,
                             const PF_EffectWorld* wP)
{
    ExvLog("OutputFrame");
    if (!wP) return A_Err_STRUCT;
    EncState st;
    A_Err err = StartImpl(basic_dataP, outH, st);
    if (!err) {
        err = AddFrameImpl(basic_dataP, outH, st, 0, wP);
        if (err) st.failed = true;
    }
    const A_Err end = EndImpl(st);   // always runs: commits on success, aborts + frees on failure
    return err ? err : end;
}

// ---------------------------------------------------------------------------
// Remaining stubs (mirror reference)
// ---------------------------------------------------------------------------

static A_Err My_WriteLabels(AEIO_BasicData* bd, AEIO_OutSpecH outH, AEIO_LabelFlags* written)
{
    if (written) *written = 0;
    (void)bd; (void)outH;
    return A_Err_NONE;
}

static A_Err My_GetSizes(AEIO_BasicData* bd, AEIO_OutSpecH outH,
                          A_u_longlong* free_space, A_u_longlong* file_size)
{
    (void)bd; (void)outH; (void)free_space; (void)file_size;
    return A_Err_NONE;
}

static A_Err My_Flush(AEIO_BasicData* bd, AEIO_OutSpecH outH)
{
    (void)bd; (void)outH;
    return A_Err_NONE;
}

static A_Err My_AddSoundChunk(AEIO_BasicData* bd, AEIO_OutSpecH outH, const A_Time* start,
                               A_u_long num_samplesLu, const void* dataPV)
{
    (void)bd; (void)outH; (void)start; (void)num_samplesLu; (void)dataPV;
    return A_Err_NONE;
}

static A_Err My_Idle(AEIO_BasicData* bd, AEIO_ModuleSignature sig, AEIO_IdleFlags* idle_flags0)
{
    (void)bd; (void)sig; (void)idle_flags0;
    return A_Err_NONE;
}

static A_Err My_GetDepths(AEIO_BasicData* bd, AEIO_OutSpecH outH, AEIO_SupportedDepthFlags* which)
{
    if (which)
        // H5: advertise only the depths that have a blit branch (8/16bpc-int + 32bpc-float).
        // DEPTH_96 (RGB 32-bit without alpha) has no branch — removed so AE doesn't send a 96bpc world.
        *which = AEIO_SupportedDepthFlags_DEPTH_32 | AEIO_SupportedDepthFlags_DEPTH_64
               | AEIO_SupportedDepthFlags_DEPTH_128;
    (void)bd; (void)outH;
    return A_Err_NONE;
}

static A_Err My_GetOutputSuffix(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_char* suffix)
{
    if (!suffix) return A_Err_STRUCT;
    AEGP_SuiteHandler suites(bd->pica_basicP);
    // The one container: .exv (a still is a 1-frame .exv).
    suites.ANSICallbacksSuite1()->strcpy(suffix, "exv");
    return A_Err_NONE;
}

// ---------------------------------------------------------------------------
// SeqOptions dialog: "EXV Channel Matrix". Single-part (EXV's AE path has no
// multipart), so no part dropdown. A Win32 ListView routes any named EXV channel
// (or the sentinels <none>->0.0 / <white>->1.0) into each of displayed R/G/B/A.
// Registering AEIO_SeqOptionsDlg + setting AEIO_MFlag_SEQ_OPTIONS_DLG is what
// lights AE's "Options" button for imported .exv footage.
// ---------------------------------------------------------------------------

// Dialog data exchanged with the modal. `new_map` holds the live selections, updated on each cell
// click; `rows` = the matrix row names (so a click reads the full name back from here, never from the
// ListView's text buffer).
struct ExvSeqDlgData {
    const ExvFileInfo* infoP = nullptr;
    ExvMap new_map;
    std::vector<std::string> rows;
};

#ifdef AE_OS_WIN

enum {
    IDC_LABEL_HINT       = 106,
    IDC_LIST_PLACEHOLDER = 114,
    IDC_LIST_MATRIX      = 115,
};
// ListView columns: 0 = channel name, 1..4 = R/G/B/A.
enum { MTX_COL_NAME = 0, MTX_COL_R = 1, MTX_COL_G = 2, MTX_COL_B = 3, MTX_COL_A = 4 };
// Selected-cell marker. CP-1252 bullet (0x95) renders as a dot in AE's fonts.
static const char* const MTX_MARK = "\x95";

struct WinDialogItemSpec {
    DWORD style;
    int x;
    int y;
    int cx;
    int cy;
    WORD id;
    WORD cls;
    const wchar_t* text;
};

// Build an in-memory DLGTEMPLATE (+ DLGITEMTEMPLATEs) for DialogBoxIndirectParamA.
// Each item's class is a predefined control atom (0x0080 button, 0x0082 static,
// 0x0085 combo, 0x0081 edit).
static std::vector<BYTE> BuildWinDialogTemplate(
    int dlg_w, int dlg_h, const wchar_t* title,
    const WinDialogItemSpec* items, size_t item_count, size_t buffer_size)
{
    std::vector<BYTE> buf(buffer_size, 0);
    BYTE* const begin = buf.data();
    BYTE* p = begin;

    auto append_bytes = [&](const void* src, size_t size) -> bool {
        if (static_cast<size_t>(p - begin) + size > buf.size()) return false;
        std::memcpy(p, src, size);
        p += size;
        return true;
    };
    auto append_word = [&](WORD v) -> bool { return append_bytes(&v, sizeof(v)); };
    auto align4 = [&]() -> bool {
        while ((reinterpret_cast<uintptr_t>(p) % sizeof(DWORD)) != 0) {
            if (static_cast<size_t>(p - begin) >= buf.size()) return false;
            *p++ = 0;
        }
        return true;
    };
    auto wstr_bytes = [](const wchar_t* s) -> size_t {
        if (!s) return sizeof(wchar_t);
        size_t len = 0;
        while (s[len]) ++len;
        return (len + 1) * sizeof(wchar_t);
    };

    DLGTEMPLATE dlg = {};
    dlg.style = DS_MODALFRAME | DS_CENTER | WS_POPUP | WS_CAPTION | WS_SYSMENU;
    dlg.cdit = static_cast<WORD>(item_count);
    dlg.cx = static_cast<short>(dlg_w);
    dlg.cy = static_cast<short>(dlg_h);
    if (!append_bytes(&dlg, sizeof(dlg))) return {};
    if (!append_word(0) || !append_word(0)) return {};   // no menu, default class
    if (!append_bytes(title, wstr_bytes(title))) return {};

    for (size_t i = 0; i < item_count; ++i) {
        const WinDialogItemSpec& it = items[i];
        if (!align4()) return {};
        DLGITEMTEMPLATE item = {};
        item.style = it.style | WS_CHILD | WS_VISIBLE;
        item.x = static_cast<short>(it.x);
        item.y = static_cast<short>(it.y);
        item.cx = static_cast<short>(it.cx);
        item.cy = static_cast<short>(it.cy);
        item.id = it.id;
        if (!append_bytes(&item, sizeof(item))) return {};
        if (!append_word(0xFFFF) || !append_word(it.cls)) return {};
        if (!append_bytes(it.text, wstr_bytes(it.text))) return {};
        if (!append_word(0)) return {};   // no creation data
    }
    return buf;
}

// (Re)build the matrix rows: "<none>", "<white>", then the container's channels;
// then mark the selected cell in each R/G/B/A column from the current map.
static void MatrixBuild(HWND lv, ExvSeqDlgData& d)
{
    ListView_DeleteAllItems(lv);
    d.rows.clear();
    d.rows.emplace_back("<none>");
    d.rows.emplace_back("<white>");
    for (const std::string& c : d.infoP->chans)
        if (!c.empty()) d.rows.push_back(c);
    for (size_t i = 0; i < d.rows.size(); ++i) {
        LVITEMA it; ZeroMemory(&it, sizeof(it));
        it.mask = LVIF_TEXT; it.iItem = (int)i; it.iSubItem = MTX_COL_NAME;
        it.pszText = const_cast<LPSTR>(d.rows[i].c_str());
        ListView_InsertItem(lv, &it);
    }
    for (int c = 0; c < 4; ++c)
        for (int i = 0; i < (int)d.rows.size(); ++i) {
            bool sel = _stricmp(d.rows[i].c_str(), d.new_map[c].c_str()) == 0;
            ListView_SetItemText(lv, i, MTX_COL_R + c, const_cast<LPSTR>(sel ? MTX_MARK : ""));
        }
}

// Click in column `col` (0=R..3=A) at `row`: route that channel into the output and move the column's
// single marker there. The chosen name comes from `d.rows` (full length), not from the ListView text.
static void MatrixSetColumn(HWND lv, ExvSeqDlgData& d, int col, int row)
{
    if (row < 0 || static_cast<size_t>(row) >= d.rows.size()) return;
    d.new_map[col] = d.rows[row];
    int n = ListView_GetItemCount(lv);
    for (int i = 0; i < n; ++i)
        ListView_SetItemText(lv, i, MTX_COL_R + col, const_cast<LPSTR>(i == row ? MTX_MARK : ""));
}

static INT_PTR ExvMatrixDlgProcImpl(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_INITDIALOG: {
        SetWindowLongPtrA(hDlg, GWLP_USERDATA, (LONG_PTR)lParam);
        ExvSeqDlgData* d = reinterpret_cast<ExvSeqDlgData*>(lParam);
        if (d && d->infoP) {
            // Create the channel-matrix ListView over the placeholder static.
            HWND ph = GetDlgItem(hDlg, IDC_LIST_PLACEHOLDER);
            RECT rc; GetWindowRect(ph, &rc);
            POINT tl = { rc.left, rc.top }, br = { rc.right, rc.bottom };
            ScreenToClient(hDlg, &tl); ScreenToClient(hDlg, &br);
            DestroyWindow(ph);
            HWND lv = CreateWindowExA(WS_EX_CLIENTEDGE, WC_LISTVIEWA, "",
                WS_CHILD | WS_VISIBLE | WS_TABSTOP | LVS_REPORT | LVS_SINGLESEL,
                tl.x, tl.y, br.x - tl.x, br.y - tl.y, hDlg,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_LIST_MATRIX)),
                reinterpret_cast<HINSTANCE>(GetWindowLongPtrA(hDlg, GWLP_HINSTANCE)), nullptr);
            if (lv) {
                ListView_SetExtendedListViewStyle(lv,
                    LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);
                const struct { const char* t; int w; int fmt; } cols[] = {
                    { "Channel", 150, LVCFMT_LEFT },
                    { "R", 36, LVCFMT_CENTER }, { "G", 36, LVCFMT_CENTER },
                    { "B", 36, LVCFMT_CENTER }, { "A", 36, LVCFMT_CENTER },
                };
                for (int c = 0; c < 5; ++c) {
                    LVCOLUMNA col; ZeroMemory(&col, sizeof(col));
                    col.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM | LVCF_FMT;
                    col.iSubItem = c; col.pszText = const_cast<LPSTR>(cols[c].t);
                    col.cx = cols[c].w; col.fmt = cols[c].fmt;
                    ListView_InsertColumn(lv, c, &col);
                }
                MatrixBuild(lv, *d);
            }
        }
        return TRUE;
    }
    case WM_NOTIFY: {
        LPNMHDR nh = reinterpret_cast<LPNMHDR>(lParam);
        if (nh->idFrom == IDC_LIST_MATRIX && nh->code == NM_CLICK) {
            ExvSeqDlgData* d = reinterpret_cast<ExvSeqDlgData*>(GetWindowLongPtrA(hDlg, GWLP_USERDATA));
            LPNMITEMACTIVATE ia = reinterpret_cast<LPNMITEMACTIVATE>(lParam);
            if (d && ia->iItem >= 0 && ia->iSubItem >= MTX_COL_R && ia->iSubItem <= MTX_COL_A)
                MatrixSetColumn(nh->hwndFrom, *d, ia->iSubItem - MTX_COL_R /* 0=R .. 3=A */, ia->iItem);
        }
        break;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK) {
            // d->new_map already holds the live selections (updated on each click).
            EndDialog(hDlg, IDOK);
            return TRUE;
        }
        if (LOWORD(wParam) == IDCANCEL) {
            EndDialog(hDlg, IDCANCEL);
            return TRUE;
        }
        break;
    }
    return FALSE;
}

// Win32 calls this on the modal's message pump inside AE: an exception must not unwind through user32.
static INT_PTR CALLBACK ExvMatrixDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    INT_PTR r = FALSE;
    const A_Err e = Guarded("ExvMatrixDlgProc", [&] { r = ExvMatrixDlgProcImpl(hDlg, msg, wParam, lParam); return A_Err_NONE; });
    return e == A_Err_NONE ? r : FALSE;
}

// Common-Controls-v6 manifest, written to `%LOCALAPPDATA%\ExvAE\exvae_comctl6.manifest`: a per-user directory
// only that user can write (a shared %TEMP% file could be swapped by another local process between our
// write and CreateActCtx reading it). Written at most once per process, ATOMICALLY (temp file +
// MoveFileEx), so a second AE instance racing us never reads a torn file. Empty path = unavailable.
static const std::wstring& ComCtlManifestPath()
{
    static const std::wstring path = []() -> std::wstring {
        wchar_t base[MAX_PATH] = {0};
        const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", base, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) return {};
        const std::wstring dir = std::wstring(base) + L"\\ExvAE";
        if (!CreateDirectoryW(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) return {};
        const std::wstring file = dir + L"\\exvae_comctl6.manifest";
        const std::wstring tmp  = file + L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
        static const char kManifest[] =
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\">"
            "<dependency><dependentAssembly><assemblyIdentity type=\"win32\" "
            "name=\"Microsoft.Windows.Common-Controls\" version=\"6.0.0.0\" "
            "processorArchitecture=\"*\" publicKeyToken=\"6595b64144ccf1df\" "
            "language=\"*\"/></dependentAssembly></dependency></assembly>";
        HANDLE f = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) return {};
        DWORD wrote = 0;
        const BOOL wok = WriteFile(f, kManifest, (DWORD)(sizeof(kManifest) - 1), &wrote, nullptr);
        CloseHandle(f);
        const bool moved = wok && wrote == sizeof(kManifest) - 1 &&
            MoveFileExW(tmp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        if (moved) return file;
        DeleteFileW(tmp.c_str());
        // Replacing fails while another AE / aerender process has the manifest open; that existing file is a
        // complete one (always written atomically), so use it instead of dropping to the classic look.
        return GetFileAttributesW(file.c_str()) != INVALID_FILE_ATTRIBUTES ? file : std::wstring();
    }();
    return path;
}

// RAII: activate a Common-Controls-v6 activation context so the modal's ListView
// renders themed (modern) instead of the flat classic look. A DLL-created dialog
// does not inherit AE's v6 context unless one is active. Cosmetic only: without the manifest the dialog
// still works with the classic look (logged).
struct ComCtlV6Scope {
    HANDLE hCtx = INVALID_HANDLE_VALUE;
    ULONG_PTR cookie = 0;
    ComCtlV6Scope() {
        const std::wstring& path = ComCtlManifestPath();
        if (path.empty()) {
            ExvLog("ComCtlV6Scope: manifest unavailable (%LOCALAPPDATA%\\ExvAE not writable) -> classic look");
        } else {
            ACTCTXW ac; ZeroMemory(&ac, sizeof(ac)); ac.cbSize = sizeof(ac);
            ac.lpSource = path.c_str();
            hCtx = CreateActCtxW(&ac);
            if (hCtx != INVALID_HANDLE_VALUE) ActivateActCtx(hCtx, &cookie);
        }
        INITCOMMONCONTROLSEX icc; icc.dwSize = sizeof(icc);
        icc.dwICC = ICC_LISTVIEW_CLASSES | ICC_STANDARD_CLASSES;
        InitCommonControlsEx(&icc);
    }
    ~ComCtlV6Scope() {
        if (cookie) DeactivateActCtx(0, cookie);
        if (hCtx != INVALID_HANDLE_VALUE) ReleaseActCtx(hCtx);
    }
};

// Show the modal channel-matrix dialog. Returns true if the user clicked OK.
static bool ShowExvMatrixDlg(ExvSeqDlgData* data)
{
    if (!data || !data->infoP) return false;
    ComCtlV6Scope theme;  // themed (v6) controls for this modal dialog

    // A hint label up top; the channel-matrix ListView fills the rest (created
    // over IDC_LIST_PLACEHOLDER in WM_INITDIALOG). No part combo (single-part).
    const int DLG_W = 300, DLG_H = 260;
    const WinDialogItemSpec items[] = {
        {SS_LEFT, 10, 10, DLG_W - 20, 9, IDC_LABEL_HINT, 0x0082,
            L"Click a cell to route a channel into R / G / B / A:"},
        {SS_LEFT, 10, 24, DLG_W - 20, DLG_H - 52, IDC_LIST_PLACEHOLDER, 0x0082, L""},
        {BS_DEFPUSHBUTTON | WS_TABSTOP, DLG_W - 120, DLG_H - 22, 50, 14, IDOK, 0x0080, L"OK"},
        {BS_PUSHBUTTON | WS_TABSTOP, DLG_W - 60, DLG_H - 22, 50, 14, IDCANCEL, 0x0080, L"Cancel"},
    };
    std::vector<BYTE> buf = BuildWinDialogTemplate(
        DLG_W, DLG_H, L"EXV Channel Matrix",
        items, sizeof(items) / sizeof(items[0]), 8192);
    if (buf.empty()) return false;

    INT_PTR result = DialogBoxIndirectParamA(nullptr, reinterpret_cast<DLGTEMPLATE*>(buf.data()),
        nullptr, ExvMatrixDlgProc, reinterpret_cast<LPARAM>(data));
    return (result == IDOK);
}


// ---------------------------------------------------------------------------
// Output options dialog: "EXV Output Options". Mode / sample type / quality, all from exv_enc_opts.h (the
// dialog holds no codec knowledge of its own). Picking a mode resets quality to the mode's default and
// narrows the sample-type list to what the mode supports; OK refuses an invalid set with the reason.
// ---------------------------------------------------------------------------

enum {
    IDC_OUT_MODE_LABEL  = 201,
    IDC_OUT_MODE        = 202,
    IDC_OUT_HINT        = 203,
    IDC_OUT_DTYPE_LABEL = 204,
    IDC_OUT_DTYPE       = 205,
    IDC_OUT_Q_LABEL     = 206,
    IDC_OUT_Q           = 207,
    IDC_OUT_Q_RANGE     = 208,
    IDC_OUT_NOTE        = 209,
};

struct ExvOutDlgData {
    exvae::EncOpts opts;   // in: current options; out: the confirmed ones
};

// Refill the controls that depend on the selected mode: hint text, allowed sample types, quality field.
static void OutDlgRefresh(HWND hDlg, const exvae::EncOpts& o)
{
    const exvae::ModeInfo* m = exvae::FindMode(o.mode);
    if (!m) return;
    SetDlgItemTextA(hDlg, IDC_OUT_HINT, m->hint);

    HWND dt = GetDlgItem(hDlg, IDC_OUT_DTYPE);
    SendMessageA(dt, CB_RESETCONTENT, 0, 0);
    for (const exvae::DtypeInfo& d : exvae::kDtypes) {
        if (!(m->dtypes & exvae::DtypeBit(d.code))) continue;
        const LRESULT i = SendMessageA(dt, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(d.label));
        SendMessageA(dt, CB_SETITEMDATA, static_cast<WPARAM>(i), static_cast<LPARAM>(d.code));
        if (d.code == o.dtype) SendMessageA(dt, CB_SETCURSEL, static_cast<WPARAM>(i), 0);
    }
    if (SendMessageA(dt, CB_GETCURSEL, 0, 0) == CB_ERR) SendMessageA(dt, CB_SETCURSEL, 0, 0);

    const bool has_q = exvae::ModeHasQ(*m);
    EnableWindow(GetDlgItem(hDlg, IDC_OUT_Q), has_q ? TRUE : FALSE);
    SetDlgItemInt(hDlg, IDC_OUT_Q, has_q ? o.q : 0, FALSE);
    char range[96];
    if (has_q) std::snprintf(range, sizeof(range), "%u - %u (higher = better)", m->q_min, m->q_max);
    else       std::snprintf(range, sizeof(range), "not used by this mode");
    SetDlgItemTextA(hDlg, IDC_OUT_Q_RANGE, range);
}

// Controls -> options (the combo's item data is the ExvDtype code; the mode combo index is the ExportMode
// because it is filled in table order). A missing selection / unparsable quality becomes an INVALID value
// that Validate() rejects, never a silent 0.
static exvae::EncOpts OutDlgRead(HWND hDlg)
{
    exvae::EncOpts o;
    const LRESULT mi = SendDlgItemMessageA(hDlg, IDC_OUT_MODE, CB_GETCURSEL, 0, 0);
    o.mode = mi == CB_ERR ? ~0u : static_cast<uint32_t>(mi);
    const LRESULT di = SendDlgItemMessageA(hDlg, IDC_OUT_DTYPE, CB_GETCURSEL, 0, 0);
    o.dtype = di == CB_ERR ? ~0u
        : static_cast<uint32_t>(SendDlgItemMessageA(hDlg, IDC_OUT_DTYPE, CB_GETITEMDATA, static_cast<WPARAM>(di), 0));
    const exvae::ModeInfo* m = exvae::FindMode(o.mode);
    if (m && exvae::ModeHasQ(*m)) {
        BOOL ok = FALSE;
        const UINT q = GetDlgItemInt(hDlg, IDC_OUT_Q, &ok, FALSE);
        o.q = ok ? q : ~0u;
    } else {
        o.q = 0;
    }
    return o;
}

static INT_PTR ExvOutDlgProcImpl(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_INITDIALOG: {
        SetWindowLongPtrA(hDlg, GWLP_USERDATA, (LONG_PTR)lParam);
        const ExvOutDlgData* d = reinterpret_cast<const ExvOutDlgData*>(lParam);
        if (!d) return TRUE;
        for (const exvae::ModeInfo& m : exvae::kModes)
            SendDlgItemMessageA(hDlg, IDC_OUT_MODE, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(m.label));
        SendDlgItemMessageA(hDlg, IDC_OUT_MODE, CB_SETCURSEL, d->opts.mode, 0);
        OutDlgRefresh(hDlg, d->opts);
        return TRUE;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_OUT_MODE && HIWORD(wParam) == CBN_SELCHANGE) {
            // New mode: its defaults, but keep the current sample type when the mode allows it.
            const exvae::EncOpts cur = OutDlgRead(hDlg);
            if (const exvae::ModeInfo* m = exvae::FindMode(cur.mode)) {
                exvae::EncOpts o = exvae::DefaultsFor(*m);
                if (cur.dtype < 32 && (m->dtypes & exvae::DtypeBit(cur.dtype))) o.dtype = cur.dtype;
                OutDlgRefresh(hDlg, o);
            }
            return TRUE;
        }
        if (LOWORD(wParam) == IDOK) {
            ExvOutDlgData* d = reinterpret_cast<ExvOutDlgData*>(GetWindowLongPtrA(hDlg, GWLP_USERDATA));
            const exvae::EncOpts o = OutDlgRead(hDlg);
            std::string why;
            if (!exvae::Validate(o, &why)) {
                MessageBoxA(hDlg, why.c_str(), "EXV Output Options", MB_OK | MB_ICONWARNING);
                return TRUE;   // stay open
            }
            if (d) d->opts = o;
            EndDialog(hDlg, IDOK);
            return TRUE;
        }
        if (LOWORD(wParam) == IDCANCEL) {
            EndDialog(hDlg, IDCANCEL);
            return TRUE;
        }
        break;
    }
    return FALSE;
}

// Win32 calls this on the modal's message pump inside AE: an exception must not unwind through user32.
static INT_PTR CALLBACK ExvOutDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    INT_PTR r = FALSE;
    const A_Err e = Guarded("ExvOutDlgProc", [&] { r = ExvOutDlgProcImpl(hDlg, msg, wParam, lParam); return A_Err_NONE; });
    return e == A_Err_NONE ? r : FALSE;
}

// Show the modal output-options dialog. Returns true if the user confirmed (then `*opts` is valid).
static bool ShowExvOutDlg(exvae::EncOpts* opts)
{
    if (!opts) return false;
    ComCtlV6Scope theme;
    ExvOutDlgData data{ *opts };

    const int DLG_W = 300, DLG_H = 176;
    const DWORD combo = CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP;
    const WinDialogItemSpec items[] = {
        {SS_LEFT, 10, 10, 280, 9, IDC_OUT_MODE_LABEL, 0x0082, L"Export mode:"},
        {combo, 10, 21, 280, 120, IDC_OUT_MODE, 0x0085, L""},
        {SS_LEFT, 10, 40, 280, 44, IDC_OUT_HINT, 0x0082, L""},
        {SS_LEFT, 10, 90, 135, 9, IDC_OUT_DTYPE_LABEL, 0x0082, L"Sample type:"},
        {combo, 10, 101, 135, 80, IDC_OUT_DTYPE, 0x0085, L""},
        {SS_LEFT, 155, 90, 135, 9, IDC_OUT_Q_LABEL, 0x0082, L"Quality:"},
        {ES_NUMBER | ES_AUTOHSCROLL | WS_BORDER | WS_TABSTOP, 155, 101, 40, 13, IDC_OUT_Q, 0x0081, L""},
        {SS_LEFT, 201, 103, 89, 20, IDC_OUT_Q_RANGE, 0x0082, L""},
        {SS_LEFT, 10, 128, 280, 18, IDC_OUT_NOTE, 0x0082,
            L"Frames are exported as linear RGBA. Integer sample types clip values outside 0..1."},
        {BS_DEFPUSHBUTTON | WS_TABSTOP, DLG_W - 120, DLG_H - 22, 50, 14, IDOK, 0x0080, L"OK"},
        {BS_PUSHBUTTON | WS_TABSTOP, DLG_W - 60, DLG_H - 22, 50, 14, IDCANCEL, 0x0080, L"Cancel"},
    };
    std::vector<BYTE> buf = BuildWinDialogTemplate(
        DLG_W, DLG_H, L"EXV Output Options", items, sizeof(items) / sizeof(items[0]), 8192);
    if (buf.empty()) return false;

    const INT_PTR result = DialogBoxIndirectParamA(nullptr, reinterpret_cast<DLGTEMPLATE*>(buf.data()),
        nullptr, ExvOutDlgProc, reinterpret_cast<LPARAM>(&data));
    if (result != IDOK) return false;
    *opts = data.opts;
    return true;
}

#else  // !AE_OS_WIN
static bool ShowExvMatrixDlg(ExvSeqDlgData*) { return false; }
static bool ShowExvOutDlg(exvae::EncOpts*) { return false; }
#endif // AE_OS_WIN

// SeqOptions dialog entry: opens the matrix on a COPY of the options (no handle stays locked during the
// modal) and, on OK, stores the chosen map in the process cache. DrawSparseFrame and FlattenOptions
// both read that cache, so the routing applies to the next render and is persisted on project save.
static A_Err My_SeqOptionsDlg(AEIO_BasicData* bd, AEIO_InSpecH specH,
                               A_Boolean* user_interactedPB0)
{
    ExvLog("SeqOptionsDlg");
    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(bd->pica_basicP);
    if (user_interactedPB0) *user_interactedPB0 = FALSE;

    AEIO_Handle optionsH = nullptr;
    ERR(suites.IOInSuite4()->AEGP_GetInSpecOptionsHandle(specH, reinterpret_cast<void**>(&optionsH)));
    if (err || !optionsH) return err;

    ExvFileInfo fi;
    if (!ReadFileInfo(suites, optionsH, true, &fi)) {
        ExvLog("SeqOptionsDlg ERR: options handle has another layout/version");
        return A_Err_GENERIC;
    }

    // Cache key = the footage path via GetInSpecPathUtf8 — the SAME source DrawSparseFrame uses.
    // Keying on anything else diverges from decode, so the dialog would reopen showing the default
    // instead of the user's selection. Without a path the choice cannot be persisted: hard error.
    std::string cache_key;
    {
        std::string p;
        if (GetInSpecPathUtf8(bd, specH, &p) == A_Err_NONE && !p.empty())
            cache_key = RouteKey(p);
    }
    if (cache_key.empty()) {
        ExvLog("SeqOptionsDlg ERR: cannot resolve the footage path (routing could not be stored)");
        return A_Err_GENERIC;
    }

    ExvSeqDlgData dlg;
    dlg.infoP = &fi;
    // Seed from the user's cached routing (authoritative) if present, else the header map.
    dlg.new_map = fi.map;
    FindChannelCache(cache_key, &dlg.new_map);
    ExvLog(("SeqOptionsDlg: seeded key='" + cache_key + "' map R=" + dlg.new_map[0] + " G=" + dlg.new_map[1] +
            " B=" + dlg.new_map[2] + " A=" + dlg.new_map[3]).c_str());

    const bool ok = ShowExvMatrixDlg(&dlg);
    ExvLog(("SeqOptionsDlg result: ok=" + std::to_string((int)ok) + " map R=" + dlg.new_map[0] + " G=" +
            dlg.new_map[1] + " B=" + dlg.new_map[2] + " A=" + dlg.new_map[3]).c_str());

    if (ok) {
        // Persist under the SAME key DrawSparseFrame reads. NEVER create a fresh options handle +
        // AEGP_SetInSpecOptionsHandle here: swapping a handle inside this callback orphans the one AE
        // holds and CRASHES After Effects. The cache + `*user_interacted = TRUE` is the whole commit
        // (TRUE is what makes AE re-render with the new map).
        UpdateChannelCache(cache_key, dlg.new_map);
        if (user_interactedPB0) *user_interactedPB0 = TRUE;
        ExvLog(("SeqOptionsDlg: cached under key='" + cache_key + "'").c_str());
    }
    return err;
}

// Verify importable: the path must end in ".exv" (the one extension; case-insensitive).
static A_Err My_VerifyFileImportable(AEIO_BasicData* bd, AEIO_ModuleSignature sig,
                                      const A_UTF16Char* file_pathZ, A_Boolean* importablePB)
{
    (void)bd; (void)sig;
    if (!importablePB) return A_Err_STRUCT;
    *importablePB = (file_pathZ && HasExvExt(Utf16ToUtf8(file_pathZ))) ? TRUE : FALSE;
    return A_Err_NONE;
}

// Minimal stubs for rarely-used entry points.
static A_Err My_NumAuxFiles(AEIO_BasicData* bd, AEIO_InSpecH seqH, A_long* files_per_framePL0)
{
    if (files_per_framePL0) *files_per_framePL0 = 0;
    (void)bd; (void)seqH;
    return A_Err_NONE;
}
static A_Err My_GetNthAuxFileSpec(AEIO_BasicData* bd, AEIO_InSpecH seqH, A_long frame_numL,
                                   A_long n, AEGP_MemHandle* pathPH)
{
    (void)bd; (void)seqH; (void)frame_numL; (void)n; (void)pathPH;
    return A_Err_NONE;
}
static A_Err My_CloseSourceFiles(AEIO_BasicData* bd, AEIO_InSpecH seqH)
{
    (void)bd; (void)seqH;
    return A_Err_NONE;
}
static A_Err My_CountUserData(AEIO_BasicData* bd, AEIO_InSpecH inH, A_u_long typeLu,
                               A_u_long max_sizeLu, A_u_long* num_of_typePLu)
{
    if (num_of_typePLu) *num_of_typePLu = 0;
    (void)bd; (void)inH; (void)typeLu; (void)max_sizeLu;
    return A_Err_NONE;
}
static A_Err My_SetUserData(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_u_long typeLu,
                             A_u_long indexLu, const AEIO_Handle dataH)
{
    (void)bd; (void)outH; (void)typeLu; (void)indexLu; (void)dataH;
    return A_Err_NONE;
}
static A_Err My_GetUserData(AEIO_BasicData* bd, AEIO_InSpecH inH, A_u_long typeLu,
                             A_u_long indexLu, A_u_long max_sizeLu, AEIO_Handle* dataPH)
{
    (void)bd; (void)inH; (void)typeLu; (void)indexLu; (void)max_sizeLu; (void)dataPH;
    return A_Err_NONE;
}
static A_Err My_AddMarker(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_long frame_index,
                           AEIO_MarkerType marker_type, void* marker_dataPV, AEIO_InterruptFuncs* inter0)
{
    (void)bd; (void)outH; (void)frame_index; (void)marker_type; (void)marker_dataPV; (void)inter0;
    return A_Err_NONE;
}
static A_Err My_UserAudioOptionsDialog(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_Boolean* user_interacted0)
{
    if (user_interacted0) *user_interacted0 = FALSE;
    (void)bd; (void)outH;
    return A_Err_NONE;
}
static A_Err My_AddMarker2(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_long frame_index,
                            const AEIO_Marker* markerP, AEIO_InterruptFuncs* inter0)
{
    (void)bd; (void)outH; (void)frame_index; (void)markerP; (void)inter0;
    return A_Err_NONE;
}
static A_Err My_AddMarker3(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_long frame_index,
                            AEGP_ConstMarkerValP marker_valP, AEIO_RenderMarkerFlag marker_flag,
                            AEIO_InterruptFuncs* inter0)
{
    (void)bd; (void)outH; (void)frame_index; (void)marker_valP; (void)marker_flag; (void)inter0;
    return A_Err_NONE;
}
static A_Err My_GetMimeType(AEIO_BasicData* bd, AEIO_OutSpecH outH, A_long mime_type_sizeL, char* mime_typeZ)
{
    if (mime_typeZ && mime_type_sizeL > 0) mime_typeZ[0] = '\0';
    (void)bd; (void)outH;
    return A_Err_NONE;
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

// Fields both registered modules share (name/limits/create kind); the callers add the flags and the sig.
static void FillCommonInfo(AEGP_SuiteHandler& suites, AEIO_ModuleInfo* info, A_long sig, const char* name)
{
    info->sig                        = sig;
    info->max_width                  = 65535;
    info->max_height                 = 65535;
    info->num_clips                  = 0;
    info->create_kind.type           = sig;
    info->create_kind.creator        = 'DTEK';
    info->create_ext.pad             = '.';
    info->create_ext.extension[0]    = 'e';
    info->create_ext.extension[1]    = 'x';
    info->create_ext.extension[2]    = 'v';
    suites.ANSICallbacksSuite1()->strcpy(info->name, name);
    info->num_aux_extensionsS        = 0;
}

// The MOVIE module: imports and exports .exv as a video (many frames in one file). AEIO_MFlag_VIDEO and
// AEIO_MFlag_STILL are mutually exclusive in AE_IO.h ("multiple image support (not STILL)" / "still image
// support (not VIDEO)"), and AE only calls OutputFrame on STILL modules, so a per-frame sequence needs the
// second module below.
static A_Err ConstructModuleInfo(SPBasicSuite* pica_basicP, AEIO_ModuleInfo* info)
{
    if (!info) return A_Err_STRUCT;
    AEGP_SuiteHandler suites(pica_basicP);

    // AE plugin 4CC 'EXVS' (big-endian char literal order); not a file-format magic.
    const A_long sig_4cc = (A_long)('E') << 24 | (A_long)('X') << 16
                         | (A_long)('V') << 8  | (A_long)('S');
    FillCommonInfo(suites, info, sig_4cc, "EXV");
    info->num_filetypes              = 1;   // one signature kind
    info->num_extensions             = 1;   // .exv ONLY (a still is a 1-frame .exv; no separate format)

    // VIDEO: multi-frame sequence; CAN_DRAW_DEEP (16bpc) + CAN_DRAW_FLOAT (32bpc).
    info->flags  = AEIO_MFlag_INPUT | AEIO_MFlag_OUTPUT | AEIO_MFlag_FILE
                 | AEIO_MFlag_VIDEO | AEIO_MFlag_CAN_DRAW_DEEP
                 // AEIO_MFlag_SEQ_OPTIONS_DLG is what ENABLES the "More Options..." button in
                 // Interpret Footage — registering funcs->AEIO_SeqOptionsDlg alone is NOT enough
                 // (AE greys the button unless this flag is set). This opens the EXV Channel Matrix.
                 | AEIO_MFlag_SEQ_OPTIONS_DLG | AEIO_MFlag_HSF_AWARE;
    info->flags2 = AEIO_MFlag2_CAN_DRAW_FLOAT;
    // NOT set: AEIO_MFlag_SEQUENCE_OPTIONS_OK ("adopt the sequence options of its parent when a folder is
    // selected"): AE applies it to numbered-still import, which a VIDEO module never is (see CLAUDE.md #18).

    // read_kinds layout: [0..num_filetypes) = mac type/creator kinds, then
    // [num_filetypes..+num_extensions) = extension kinds. ONE format: `.exv`.
    info->read_kinds[0].mac.type    = sig_4cc;
    info->read_kinds[0].mac.creator = AEIO_ANY_CREATOR;
    info->read_kinds[1].ext.pad          = '.';
    info->read_kinds[1].ext.extension[0] = 'e';
    info->read_kinds[1].ext.extension[1] = 'x';
    info->read_kinds[1].ext.extension[2] = 'v';

    return A_Err_NONE;
}

// The IMAGE-SEQUENCE module: OUTPUT only, a STILL format ("EXV Sequence"). AE renders `name.[####].exv` by
// calling OutputFrame once per frame, each with that frame's own path; every file is a self-contained
// 1-frame .exv. It reuses the movie module's callbacks and output options (same dialog, same mapping).
// Import stays with the movie module (this one claims no extension), so there is no import ambiguity.
static A_Err ConstructSeqModuleInfo(SPBasicSuite* pica_basicP, AEIO_ModuleInfo* info)
{
    if (!info) return A_Err_STRUCT;
    AEGP_SuiteHandler suites(pica_basicP);
    const A_long sig_4cc = (A_long)('E') << 24 | (A_long)('X') << 16
                         | (A_long)('V') << 8  | (A_long)('F');
    FillCommonInfo(suites, info, sig_4cc, "EXV Sequence");
    info->num_filetypes  = 0;
    info->num_extensions = 0;
    info->flags  = AEIO_MFlag_OUTPUT | AEIO_MFlag_FILE | AEIO_MFlag_STILL;
    info->flags2 = AEIO_MFlag2_NONE;
    return A_Err_NONE;
}


A_Err ConstructFunctionBlock(AEIO_FunctionBlock4* funcs)
{
    if (!funcs) return A_Err_STRUCT;
    funcs->AEIO_AddFrame           = SAFE(My_AddFrame);
    funcs->AEIO_AddMarker          = SAFE(My_AddMarker);
    funcs->AEIO_AddSoundChunk      = SAFE(My_AddSoundChunk);
    funcs->AEIO_CloseSourceFiles   = SAFE(My_CloseSourceFiles);
    funcs->AEIO_CountUserData      = SAFE(My_CountUserData);
    funcs->AEIO_DisposeInSpec      = SAFE(My_DisposeInSpec);
    funcs->AEIO_DisposeOutputOptions = SAFE(My_DisposeOutputOptions);
    funcs->AEIO_DrawSparseFrame    = SAFE(My_DrawSparseFrame);
    funcs->AEIO_EndAdding          = SAFE(My_EndAdding);
    funcs->AEIO_FlattenOptions     = SAFE(My_FlattenOptions);
    funcs->AEIO_Flush              = SAFE(My_Flush);
    funcs->AEIO_GetActiveExtent    = SAFE(My_GetActiveExtent);
    funcs->AEIO_GetDepths          = SAFE(My_GetDepths);
    funcs->AEIO_GetDimensions      = SAFE(My_GetDimensions);
    funcs->AEIO_GetDuration        = SAFE(My_GetDuration);
    funcs->AEIO_GetInSpecInfo      = SAFE(My_GetInSpecInfo);
    funcs->AEIO_GetOutputInfo      = SAFE(My_GetOutputInfo);
    funcs->AEIO_GetOutputSuffix    = SAFE(My_GetOutputSuffix);
    funcs->AEIO_GetSizes           = SAFE(My_GetSizes);
    funcs->AEIO_GetSound           = SAFE(My_GetSound);
    funcs->AEIO_GetTime            = SAFE(My_GetTime);
    funcs->AEIO_GetUserData        = SAFE(My_GetUserData);
    funcs->AEIO_Idle               = SAFE(My_Idle);
    funcs->AEIO_InflateOptions     = SAFE(My_InflateOptions);
    funcs->AEIO_InitInSpecFromFile = SAFE(My_InitInSpecFromFile);
    funcs->AEIO_InitInSpecInteractive = SAFE(My_InitInSpecInteractive);
    funcs->AEIO_InqNextFrameTime   = SAFE(My_InqNextFrameTime);
    funcs->AEIO_NumAuxFiles        = SAFE(My_NumAuxFiles);
    funcs->AEIO_OutputFrame        = SAFE(My_OutputFrame);
    funcs->AEIO_SeqOptionsDlg      = SAFE(My_SeqOptionsDlg);
    funcs->AEIO_SetOutputFile      = SAFE(My_SetOutputFile);
    funcs->AEIO_SetUserData        = SAFE(My_SetUserData);
    funcs->AEIO_StartAdding        = SAFE(My_StartAdding);
    funcs->AEIO_SynchInSpec        = SAFE(My_SynchInSpec);
    funcs->AEIO_UserOptionsDialog  = SAFE(My_UserOptionsDialog);
    funcs->AEIO_VerifyFileImportable = SAFE(My_VerifyFileImportable);
    funcs->AEIO_GetNthAuxFileSpec  = SAFE(My_GetNthAuxFileSpec);
    funcs->AEIO_UserAudioOptionsDialog = SAFE(My_UserAudioOptionsDialog);
    funcs->AEIO_AddMarker2         = SAFE(My_AddMarker2);
    funcs->AEIO_AddMarker3         = SAFE(My_AddMarker3);
    funcs->AEIO_GetMimeType        = SAFE(My_GetMimeType);
    funcs->AEIO_WriteLabels        = SAFE(My_WriteLabels);
    funcs->AEIO_InitOutputSpec     = SAFE(My_InitOutputSpec);
    funcs->AEIO_GetFlatOutputOptions = SAFE(My_GetFlatOutputOptions);
    funcs->AEIO_OutputInfoChanged  = SAFE(My_OutputInfoChanged);
    return A_Err_NONE;
}

// ---------------------------------------------------------------------------
// EntryPointFunc — called by AE at plugin load time.
// ---------------------------------------------------------------------------

static A_Err EntryPointImpl(
    struct SPBasicSuite* pica_basicP,
    A_long major_versionL, A_long minor_versionL,
    AEGP_PluginID aegp_plugin_id,
    AEGP_GlobalRefcon* global_refconP)
{
    (void)major_versionL; (void)minor_versionL; (void)global_refconP;
    ExvLog("EntryPointFunc START");

    A_Err err = A_Err_NONE;
    AEGP_SuiteHandler suites(pica_basicP);
    AEIO_FunctionBlock4 funcs;
    AEFX_CLR_STRUCT(funcs);

    ERR(suites.RegisterSuite5()->AEGP_RegisterDeathHook(aegp_plugin_id, DeathHook, 0));
    ERR(ConstructFunctionBlock(&funcs));
    if (err) return err;

    // Load the bridge up-front so a missing DLL is reported at load time, not decode time.
    if (!g_bridge.Load()) {
        ExvLog(("EntryPointFunc: " + g_bridge.load_error + " — cannot decode/encode").c_str());
        // Soft-fail: register anyway so the format appears; decode will surface the error.
    }

    // Build and register the movie AEIO_ModuleInfo for .exv.
    AEIO_ModuleInfo info;
    AEFX_CLR_STRUCT(info);
    ERR(ConstructModuleInfo(pica_basicP, &info));
    ERR(suites.RegisterSuite5()->AEGP_RegisterIO(aegp_plugin_id, 0, &info, &funcs));

    // Second module: the per-frame image-sequence exporter. Its registration failing must NOT take the movie
    // module down with it: log and carry on (the format list then simply lacks "EXV Sequence").
    if (!err) {
        AEIO_ModuleInfo seq_info;
        AEFX_CLR_STRUCT(seq_info);
        A_Err seq_err = ConstructSeqModuleInfo(pica_basicP, &seq_info);
        if (!seq_err) seq_err = suites.RegisterSuite5()->AEGP_RegisterIO(aegp_plugin_id, 0, &seq_info, &funcs);
        if (seq_err) ExvLog(("EntryPointFunc: EXV Sequence module not registered, err=" + std::to_string(seq_err)).c_str());
    }

    ERR(suites.UtilitySuite3()->AEGP_RegisterWithAEGP(NULL, "ExvAE", &S_mem_id));
    ExvLog(err ? "EntryPointFunc DONE (err)" : "EntryPointFunc DONE (ok)");
    return err;
}

// The exported symbol: a thin firewall over EntryPointImpl (no exception may cross into AE at load time).
A_Err EntryPointFunc(
    struct SPBasicSuite* pica_basicP,
    A_long major_versionL, A_long minor_versionL,
    AEGP_PluginID aegp_plugin_id,
    AEGP_GlobalRefcon* global_refconP)
{
    return Guarded("EntryPointFunc", [&] {
        return EntryPointImpl(pica_basicP, major_versionL, minor_versionL, aegp_plugin_id, global_refconP);
    });
}
