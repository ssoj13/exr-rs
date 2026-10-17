// win_thumb_provider.cpp — Windows Explorer Shell Thumbnail Provider (`ExvOsThumb.dll`) for
// `.exv` and `.exs` HDR still/video containers.
//
// Implements:
//   - `IInitializeWithFile` (receives the UTF-16 path from Explorer)
//   - `IThumbnailProvider::GetThumbnail` (decodes frame 0 via `exv_capi.dll`, tonemaps HDR -> sRGB
//     BGRA8, and returns a top-down 32-bpp `HBITMAP`)
//   - Self-registration (`regsvr32 ExvOsThumb.dll`) under `HKCU\Software\Classes` (no admin needed)

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <propsys.h>
#include <shlwapi.h>
#include <thumbcache.h>

#include "exv_os_thumb.h"

#include <atomic>
#include <mutex>
#include <new>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "advapi32.lib")

namespace {

// {8E6B3A91-4F2C-4D87-9A51-7C3E2D1B8F04}
const CLSID CLSID_ExvThumbnailProvider = {
    0x8e6b3a91, 0x4f2c, 0x4d87, {0x9a, 0x51, 0x7c, 0x3e, 0x2d, 0x1b, 0x8f, 0x04}};
const wchar_t* const kClsidStr = L"{8E6B3A91-4F2C-4D87-9A51-7C3E2D1B8F04}";
const wchar_t* const kThumbHandlerGuid = L"{e357fccd-a995-4576-b01f-234630154e96}";

std::atomic<long> g_dll_ref_count{0};
HMODULE g_hinst = nullptr;

struct CapiFns {
    decltype(&exv_abi_version) abi_version = nullptr;
    decltype(&exv_session_open) session_open = nullptr;
    decltype(&exv_session_info) session_info = nullptr;
    decltype(&exv_session_decode) session_decode = nullptr;
    decltype(&exv_session_close) session_close = nullptr;

    bool valid() const {
        return abi_version && session_open && session_info && session_decode && session_close;
    }
};

const CapiFns& load_capi() {
    static CapiFns fns;
    static std::once_flag once;
    std::call_once(once, []() {
        wchar_t mod_path[MAX_PATH] = {};
        if (GetModuleFileNameW(g_hinst, mod_path, MAX_PATH) > 0) {
            PathRemoveFileSpecW(mod_path);
            PathAppendW(mod_path, L"exv_capi.dll");
        }
        HMODULE h = LoadLibraryW(mod_path);
        if (!h) {
            h = LoadLibraryW(L"exv_capi.dll");
        }
        if (!h) return;
        fns.abi_version = reinterpret_cast<decltype(fns.abi_version)>(GetProcAddress(h, "exv_abi_version"));
        fns.session_open = reinterpret_cast<decltype(fns.session_open)>(GetProcAddress(h, "exv_session_open"));
        fns.session_info = reinterpret_cast<decltype(fns.session_info)>(GetProcAddress(h, "exv_session_info"));
        fns.session_decode = reinterpret_cast<decltype(fns.session_decode)>(GetProcAddress(h, "exv_session_decode"));
        fns.session_close = reinterpret_cast<decltype(fns.session_close)>(GetProcAddress(h, "exv_session_close"));
        if (fns.abi_version && fns.abi_version() != EXV_ABI_VERSION) {
            fns = CapiFns{};
        }
    });
    return fns;
}

std::string wide_to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), len, nullptr, nullptr);
    return out;
}

class ExvThumbProvider final : public IInitializeWithFile, public IThumbnailProvider {
    std::atomic<long> ref_{1};
    std::wstring file_path_;

public:
    ExvThumbProvider() { ++g_dll_ref_count; }
    ~ExvThumbProvider() { --g_dll_ref_count; }

    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IThumbnailProvider) {
            *ppv = static_cast<IThumbnailProvider*>(this);
        } else if (riid == IID_IInitializeWithFile) {
            *ppv = static_cast<IInitializeWithFile*>(this);
        } else {
            *ppv = nullptr;
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    IFACEMETHODIMP_(ULONG) AddRef() override { return static_cast<ULONG>(++ref_); }

    IFACEMETHODIMP_(ULONG) Release() override {
        const long r = --ref_;
        if (r == 0) delete this;
        return static_cast<ULONG>(r);
    }

    IFACEMETHODIMP Initialize(LPCWSTR pszFilePath, DWORD /*grfMode*/) override {
        if (!pszFilePath) return E_INVALIDARG;
        if (!file_path_.empty()) return HRESULT_FROM_WIN32(ERROR_ALREADY_INITIALIZED);
        file_path_ = pszFilePath;
        return S_OK;
    }

    IFACEMETHODIMP GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha) override {
        if (!phbmp || !pdwAlpha) return E_POINTER;
        *phbmp = nullptr;
        *pdwAlpha = WTSAT_UNKNOWN;
        if (file_path_.empty() || cx == 0) return E_INVALIDARG;

        const auto& api = load_capi();
        if (!api.valid()) return E_FAIL;

        const std::string utf8 = wide_to_utf8(file_path_);
        ExvSession* session = nullptr;
        if (api.session_open(utf8.c_str(), &session) != 0 || !session) {
            return E_FAIL;
        }

        ExvStreamInfo info{};
        if (api.session_info(session, &info) != 0 || info.width == 0 || info.height == 0) {
            api.session_close(session);
            return E_FAIL;
        }

        const size_t total_floats = static_cast<size_t>(info.width) * info.height * 4;
        std::vector<float> rgba(total_floats);
        size_t written = 0;
        const int32_t st = api.session_decode(
            session,
            0,
            EXV_PIXEL_FORMAT_RGBA_F32,
            reinterpret_cast<uint8_t*>(rgba.data()),
            total_floats * sizeof(float),
            0,
            &written);
        api.session_close(session);
        if (st != 0) return E_FAIL;

        uint32_t dst_w = 1;
        uint32_t dst_h = 1;
        exv_os::fit_thumbnail_dims(info.width, info.height, cx, dst_w, dst_h);
        const std::vector<uint8_t> bgra =
            exv_os::render_thumbnail_8bit(rgba.data(), info.width, info.height, dst_w, dst_h, true);

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = static_cast<LONG>(dst_w);
        bmi.bmiHeader.biHeight = -static_cast<LONG>(dst_h);  // top-down DIB
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        void* bits = nullptr;
        HBITMAP hbmp = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (!hbmp || !bits) {
            if (hbmp) DeleteObject(hbmp);
            return E_OUTOFMEMORY;
        }
        std::memcpy(bits, bgra.data(), bgra.size());
        *phbmp = hbmp;
        *pdwAlpha = WTSAT_ARGB;
        return S_OK;
    }
};

class ClassFactory final : public IClassFactory {
    std::atomic<long> ref_{1};

public:
    ClassFactory() { ++g_dll_ref_count; }
    ~ClassFactory() { --g_dll_ref_count; }

    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IClassFactory) {
            *ppv = static_cast<IClassFactory*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    IFACEMETHODIMP_(ULONG) AddRef() override { return static_cast<ULONG>(++ref_); }

    IFACEMETHODIMP_(ULONG) Release() override {
        const long r = --ref_;
        if (r == 0) delete this;
        return static_cast<ULONG>(r);
    }

    IFACEMETHODIMP CreateInstance(IUnknown* pUnkOuter, REFIID riid, void** ppv) override {
        if (pUnkOuter) return CLASS_E_NOAGGREGATION;
        auto* p = new (std::nothrow) ExvThumbProvider();
        if (!p) return E_OUTOFMEMORY;
        const HRESULT hr = p->QueryInterface(riid, ppv);
        p->Release();
        return hr;
    }

    IFACEMETHODIMP LockServer(BOOL fLock) override {
        if (fLock) ++g_dll_ref_count;
        else --g_dll_ref_count;
        return S_OK;
    }
};

HRESULT set_reg_str(HKEY root, const wchar_t* subkey, const wchar_t* name, const wchar_t* val) {
    HKEY hkey = nullptr;
    LONG res = RegCreateKeyExW(root, subkey, 0, nullptr, 0, KEY_WRITE, nullptr, &hkey, nullptr);
    if (res != ERROR_SUCCESS) return HRESULT_FROM_WIN32(res);
    res = RegSetValueExW(
        hkey,
        name,
        0,
        REG_SZ,
        reinterpret_cast<const BYTE*>(val),
        static_cast<DWORD>((wcslen(val) + 1) * sizeof(wchar_t)));
    RegCloseKey(hkey);
    return HRESULT_FROM_WIN32(res);
}

}  // namespace

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_hinst = hModule;
        DisableThreadLibraryCalls(hModule);
    }
    return TRUE;
}

extern "C" STDAPI DllCanUnloadNow() {
    return g_dll_ref_count.load() == 0 ? S_OK : S_FALSE;
}

extern "C" STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (rclsid != CLSID_ExvThumbnailProvider) return CLASS_E_CLASSNOTAVAILABLE;
    auto* f = new (std::nothrow) ClassFactory();
    if (!f) return E_OUTOFMEMORY;
    const HRESULT hr = f->QueryInterface(riid, ppv);
    f->Release();
    return hr;
}

extern "C" STDAPI DllRegisterServer() {
    wchar_t mod_path[MAX_PATH] = {};
    if (GetModuleFileNameW(g_hinst, mod_path, MAX_PATH) == 0) {
        return HRESULT_FROM_WIN32(GetLastError());
    }
    std::wstring clsid_key = std::wstring(L"Software\\Classes\\CLSID\\") + kClsidStr;
    std::wstring inproc_key = clsid_key + L"\\InProcServer32";
    HRESULT hr = set_reg_str(HKEY_CURRENT_USER, clsid_key.c_str(), nullptr, L"EXV HDR Thumbnail Provider");
    if (FAILED(hr)) return hr;
    hr = set_reg_str(HKEY_CURRENT_USER, inproc_key.c_str(), nullptr, mod_path);
    if (FAILED(hr)) return hr;
    hr = set_reg_str(HKEY_CURRENT_USER, inproc_key.c_str(), L"ThreadingModel", L"Both");
    if (FAILED(hr)) return hr;

    for (const wchar_t* ext : {L".exv", L".exs"}) {
        std::wstring shellex =
            std::wstring(L"Software\\Classes\\") + ext + L"\\ShellEx\\" + kThumbHandlerGuid;
        hr = set_reg_str(HKEY_CURRENT_USER, shellex.c_str(), nullptr, kClsidStr);
        if (FAILED(hr)) return hr;
    }
    return S_OK;
}

extern "C" STDAPI DllUnregisterServer() {
    for (const wchar_t* ext : {L".exv", L".exs"}) {
        std::wstring shellex =
            std::wstring(L"Software\\Classes\\") + ext + L"\\ShellEx\\" + kThumbHandlerGuid;
        RegDeleteTreeW(HKEY_CURRENT_USER, shellex.c_str());
    }
    std::wstring clsid_key = std::wstring(L"Software\\Classes\\CLSID\\") + kClsidStr;
    RegDeleteTreeW(HKEY_CURRENT_USER, clsid_key.c_str());
    return S_OK;
}

#endif  // defined(_WIN32)
