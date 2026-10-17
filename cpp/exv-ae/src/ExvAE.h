#pragma once
/*******************************************************************/
/*  ExvAE — EXV (.exv) AEIO plugin via exv_capi.dll.             */
/*  Single-part: named channels route to RGBA via the Channel Matrix.*/
/*******************************************************************/

#include "AEConfig.h"
#include "entry.h"

#ifdef AE_OS_WIN
    #include <windows.h>
#endif

#include "AE_IO.h"
#include "AE_Macros.h"
#include "AE_EffectCBSuites.h"
#include "AEGP_SuiteHandler.h"

// Layout revisions of the options structs below. They are byte-copied into AE project files, so a
// project saved by a build with a different layout must be DETECTED (version + size, checked on every
// handle dereference) and rebuilt from the file instead of being read as the current layout.
// Bump the matching constant on ANY field/string-table change.
#define EXV_OPT_VERSION      3   // ExvOutputOptions
#define EXV_FILE_OPT_VERSION 3   // ExvFileHeader (+ its string table)

// Per-file metadata stored in the InSpec options handle: this fixed POD part, immediately followed in the
// SAME handle by a string table of NUL-terminated UTF-8 strings (no length limit, nothing is truncated):
//   path, map_r, map_g, map_b, map_a, then `num_channels` channel names (in container order).
// map_* = the channel routed to displayed R/G/B/A: a channel name or the sentinels "<none>" (0.0) /
// "<white>" (1.0), resolved inside the FFI (exv_session_decode_mapped_rgbaf32). Single-part.
// Kept simple: only the fields needed for frame decode / AEIO spec setup. EXV carries no
// chroma_format_idc (it is linear RGBA), so we store the channel count instead.
// POD: byte-copied by Flatten/Inflate (no pointers). `size` = the WHOLE handle (fixed part + strings).
extern "C" {
typedef struct {
    A_u_long    version;       // EXV_FILE_OPT_VERSION
    A_u_long    size;          // total handle bytes: sizeof(ExvFileHeader) + string table
    A_u_long    widthLu;
    A_u_long    heightLu;
    A_Time      fpsT;          // authored fps from the file (24/1 only when the file records none)
    A_short     bitdepthS;     // 128 = ARGB 32bpc float
    A_Time      durationT;     // frame_count * fps_den / fps_num
    A_u_long    frame_countLu;
    A_long      par_numL;
    A_long      par_denL;
    A_u_long    num_channels;  // channel names in the string table
    A_u_long    alpha_type;    // authored alpha interpretation: 0 unspecified, 1 straight, 2 premultiplied
} ExvFileHeader;

// Output options (Output Module Settings -> Format Options): the export mode + sample type + quality knob.
// `mode` is an exvae::ExportMode, `dtype_code` an ExvDtype, `q` the mode's quality (0 when it has none); the
// valid combinations and their mapping to ExvEncParams live in exv_enc_opts.h (ONE table). Defaults:
// mode 0 (HTJ2K lossless), dtype 0 (F16), q 0. The loader validates the CONTENT too, so a handle that parses
// but holds an impossible combination counts as stale.
typedef struct {
    A_u_long version;         // EXV_OPT_VERSION
    A_u_long size;            // sizeof(ExvOutputOptions)
    A_u_long mode;            // exvae::ExportMode
    A_u_long dtype_code;      // ExvDtype
    A_u_long q;               // quality knob of the mode (0 when the mode has none)
} ExvOutputOptions;
} // extern "C"

extern "C" DllExport AEGP_PluginInitFuncPrototype EntryPointFunc;

A_Err ConstructFunctionBlock(AEIO_FunctionBlock4 *funcs);
