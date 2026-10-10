/*
 * C libdeflate golden-parity oracle for exr-rs.
 *
 * Proves `deflate-rs`'s zlib compressor output is BYTE-IDENTICAL to the real
 * C libdeflate (vcpkg, v1.25). For a fixed set of deterministic inputs and for
 * every compression level 1..=12, this program calls `libdeflate_zlib_compress`
 * and prints one machine-readable line per case.
 *
 * Also emits CRC-32 golden values (libdeflate_crc32) and gzip golden byte
 * streams (libdeflate_gzip_compress) for the same input set.
 * Keys:
 *   crc32.<inputname>       = <8 hex digits, the CRC-32 u32 LE>
 *   gzip<L>.<inputname>     = <lowercase hex of the exact gzip bytes>
 *
 *     level<L>.<inputname> = <lowercase hex of the exact compressed bytes>
 *
 * Finally it emits the decompressor RESULT CODES (enum libdeflate_result) for an
 * exact-size, an oversized and an undersized output buffer, which pin down the
 * LIBDEFLATE_SHORT_OUTPUT / LIBDEFLATE_INSUFFICIENT_SPACE contract:
 *   zlibresult.{exact,short,tight}.<inputname> = <result code, 2 hex digits>
 *
 * The companion Rust test `crates/deflate-rs/tests/parity_libdeflate.rs`
 * reproduces the SAME inputs (by the documented formulas below), runs
 * deflate-rs `zlib_compress` at each level, and asserts byte-identity.
 *
 * Inputs are defined ONCE here and reproduced byte-for-byte by the Rust side.
 * To make this self-verifying, each input is ALSO emitted as a line:
 *
 *     input.<inputname> = <lowercase hex of the exact input bytes>
 *
 * so the Rust test can assert its locally-reproduced input matches the oracle's
 * input before comparing compressed output (catches any input drift instantly).
 *
 * Build (driven by `cargo run -p xtask -- parity-deflate`, MSVC env applied):
 *   cl /nologo /MD /I <vcpkg>/include oracle.c /Fe:oracle.exe \
 *      /link /LIBPATH:<vcpkg>/lib deflate.lib
 *
 * libdeflate is a C library; `deflate.dll` (vcpkg bin/) must be on PATH at run.
 *
 * NEVER edit deflate-rs/src to merely make parity pass; a real divergence is a
 * port bug — diagnose it against libdeflate/lib/deflate_compress.c.
 */

#include <libdeflate.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

/*
 * ---------------------------------------------------------------------------
 * CLI mode: a live subprocess mode alongside the fixed golden
 * generator above, so the Rust side can throw ARBITRARY (large, or
 * deliberately malformed) bytes at the real C libdeflate without baking
 * megabyte-scale hex into a committed golden file.
 *
 *   oracle.exe zc <level>        stdin = raw input bytes (any size).
 *                                 stdout (binary) = libdeflate_zlib_compress
 *                                 output, or nothing + exit 1 on failure.
 *   oracle.exe zdx <out_cap>     stdin = zlib-compressed bytes (possibly
 *                                 malformed/truncated/corrupt).
 *                                 stdout (text, one line) =
 *                                 "result=<n> actual_in=<n|-1> actual_out=<n|-1>"
 *                                 via libdeflate_zlib_decompress_ex with an
 *                                 output buffer of exactly <out_cap> bytes.
 *
 * Both modes read the ENTIRE stdin into memory first (no streaming) --
 * fine for the multi-MiB inputs the parity tests use. Binary mode is forced
 * on stdin/stdout on Windows so CRLF translation never corrupts the bytes.
 * ---------------------------------------------------------------------------
 */

/* Read all of stdin into a malloc'd buffer; caller frees. Returns NULL on
 * allocation failure OR a genuine stdin read error (checked via ferror --
 * fread() returning 0 is ambiguous between "clean EOF" and "I/O error"
 * without it, and a silently-truncated input would otherwise look like a
 * legitimately short/malformed stream instead of a broken test run). */
static uint8_t *read_all_stdin(size_t *len_out)
{
    size_t cap = 1 << 20; /* 1 MiB initial */
    size_t len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        if (len == cap) {
            cap *= 2;
            uint8_t *grown = (uint8_t *)realloc(buf, cap);
            if (!grown) { free(buf); return NULL; }
            buf = grown;
        }
        size_t n = fread(buf + len, 1, cap - len, stdin);
        len += n;
        if (n == 0) {
            if (ferror(stdin)) {
                free(buf);
                return NULL;
            }
            break; /* clean EOF */
        }
    }
    *len_out = len;
    return buf;
}

static int cli_zlib_compress(int level)
{
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    size_t in_len;
    uint8_t *in = read_all_stdin(&in_len);
    if (!in) {
        fprintf(stderr, "zc: failed to read stdin (alloc failure or I/O error)\n");
        return 1;
    }
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
    if (!c) {
        fprintf(stderr, "zc: libdeflate_alloc_compressor(%d) failed\n", level);
        free(in);
        return 1;
    }
    size_t bound = libdeflate_zlib_compress_bound(c, in_len);
    uint8_t *out = (uint8_t *)malloc(bound > 0 ? bound : 1);
    if (!out) {
        fprintf(stderr, "zc: malloc(%zu) for the output buffer failed\n", bound);
        free(in);
        libdeflate_free_compressor(c);
        return 1;
    }
    size_t n = libdeflate_zlib_compress(c, in, in_len, out, bound);
    if (n == 0 && in_len != 0) {
        fprintf(stderr, "zc: libdeflate_zlib_compress returned 0\n");
        free(out);
        free(in);
        libdeflate_free_compressor(c);
        return 1;
    }
    if (n > 0)
        fwrite(out, 1, n, stdout);
    fflush(stdout);
    free(out);
    free(in);
    libdeflate_free_compressor(c);
    return 0;
}

static int cli_zlib_decompress_ex(size_t out_cap)
{
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    size_t in_len;
    uint8_t *in = read_all_stdin(&in_len);
    if (!in) {
        fprintf(stderr, "zdx: failed to read stdin (alloc failure or I/O error)\n");
        return 1;
    }
    struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();
    if (!d) {
        fprintf(stderr, "zdx: libdeflate_alloc_decompressor failed\n");
        free(in);
        return 1;
    }
    uint8_t *out = (uint8_t *)malloc(out_cap > 0 ? out_cap : 1);
    if (!out) {
        fprintf(stderr, "zdx: malloc(%zu) for the output buffer failed\n", out_cap);
        free(in);
        libdeflate_free_decompressor(d);
        return 1;
    }
    size_t actual_in = (size_t)-1;
    size_t actual_out = (size_t)-1;
    enum libdeflate_result r = libdeflate_zlib_decompress_ex(
        d, in, in_len, out, out_cap, &actual_in, &actual_out);
    /* actual_in/actual_out are only guaranteed set on SUCCESS; print -1
     * (sentinel) otherwise so the Rust side never reads uninitialised data. */
    long long ai = (r == LIBDEFLATE_SUCCESS) ? (long long)actual_in : -1;
    long long ao = (r == LIBDEFLATE_SUCCESS) ? (long long)actual_out : -1;
    printf("result=%d actual_in=%lld actual_out=%lld\n", (int)r, ai, ao);
    fflush(stdout);
    free(out);
    free(in);
    libdeflate_free_decompressor(d);
    return 0;
}

/* Parse a required positive-ish integer CLI argument (level or out_cap).
 * Returns 0 (failure) on a missing, empty, non-numeric, or trailing-garbage
 * argument -- an `atoi`-style silent-zero would let a typo'd argument
 * silently run the golden-generator's default instead of failing loudly. */
static int parse_required_int_arg(const char *mode, int argc, char **argv, long *out_value)
{
    if (argc < 3 || argv[2][0] == '\0') {
        fprintf(stderr, "%s: missing required numeric argument\n", mode);
        return 0;
    }
    char *end = NULL;
    errno = 0;
    long v = strtol(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || v < 0) {
        fprintf(stderr, "%s: invalid numeric argument %s\n", mode, argv[2]);
        return 0;
    }
    *out_value = v;
    return 1;
}

/* Same numeric-argument validation as parse_required_int_arg, but for an
 * arbitrary argv index (bench's <level> and <reps> are argv[2]/argv[3]). */
static int parse_int_arg_at(const char *mode, int argc, char **argv, int idx, long *out_value)
{
    if (argc <= idx || argv[idx][0] == '\0') {
        fprintf(stderr, "%s: missing required numeric argument at position %d\n", mode, idx);
        return 0;
    }
    char *end = NULL;
    errno = 0;
    long v = strtol(argv[idx], &end, 10);
    if (errno != 0 || end == argv[idx] || *end != '\0' || v < 0) {
        fprintf(stderr, "%s: invalid numeric argument %s\n", mode, argv[idx]);
        return 0;
    }
    *out_value = v;
    return 1;
}

/*
 * ---------------------------------------------------------------------------
 * bench mode: deflate-only A/B timing,
 * isolated from exr-core's predictor/interleave and I/O -- measures ONLY
 * libdeflate_zlib_compress / libdeflate_zlib_decompress_ex on caller-supplied
 * bytes, so the Rust side can run the identical bytes through deflate-rs and
 * get a true apples-to-apples ratio, separate from any exr-core-side cost.
 *
 *   oracle.exe bench zc <level> <reps>   stdin=raw bytes; times
 *                                         libdeflate_zlib_compress <reps>
 *                                         times (after one untimed warm-up).
 *   oracle.exe bench zd <level> <reps>   stdin=raw bytes; compresses ONCE
 *                                         (untimed) at <level>, then times
 *                                         libdeflate_zlib_decompress_ex
 *                                         <reps> times.
 * Both print one line: "median_ns=<n> min_ns=<n> reps=<n> in_len=<n> out_len=<n>".
 * High-resolution timing via QueryPerformanceCounter (Windows-only, matching
 * this whole oracle/xtask toolchain).
 * ---------------------------------------------------------------------------
 */

#if defined(_WIN32)
static double qpc_freq_hz(void)
{
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return (double)f.QuadPart;
}

static long long qpc_now(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}
#endif

static int cmp_longlong(const void *a, const void *b)
{
    long long x = *(const long long *)a;
    long long y = *(const long long *)b;
    return (x > y) - (x < y);
}

/* Prints the median+min of `n` nanosecond samples (sorts `samples` in place). */
static void print_bench_stats(long long *samples, size_t n, size_t in_len, size_t out_len)
{
    qsort(samples, n, sizeof(samples[0]), cmp_longlong);
    long long median = samples[n / 2];
    long long min_v = samples[0];
    printf("median_ns=%lld min_ns=%lld reps=%zu in_len=%zu out_len=%zu\n",
           median, min_v, n, in_len, out_len);
    fflush(stdout);
}

static int cli_bench_zc(int level, long reps)
{
#if !defined(_WIN32)
    fprintf(stderr, "bench: only implemented for _WIN32 (QueryPerformanceCounter)\n");
    return 1;
#else
    _setmode(_fileno(stdin), _O_BINARY);
    size_t in_len;
    uint8_t *in = read_all_stdin(&in_len);
    if (!in || in_len == 0) {
        fprintf(stderr, "bench zc: failed to read stdin, or empty input\n");
        return 1;
    }
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
    if (!c) {
        fprintf(stderr, "bench zc: libdeflate_alloc_compressor(%d) failed\n", level);
        return 1;
    }
    size_t bound = libdeflate_zlib_compress_bound(c, in_len);
    uint8_t *out = (uint8_t *)malloc(bound);
    long long *samples = (long long *)malloc(sizeof(long long) * (size_t)reps);
    if (!out || !samples) {
        fprintf(stderr, "bench zc: allocation failed\n");
        return 1;
    }
    double freq = qpc_freq_hz();
    size_t out_len = 0;
    /* One untimed warm-up (page faults, allocator warm-up). */
    out_len = libdeflate_zlib_compress(c, in, in_len, out, bound);
    for (long r = 0; r < reps; r++) {
        long long t0 = qpc_now();
        out_len = libdeflate_zlib_compress(c, in, in_len, out, bound);
        long long t1 = qpc_now();
        samples[r] = (long long)((double)(t1 - t0) * 1e9 / freq);
    }
    print_bench_stats(samples, (size_t)reps, in_len, out_len);
    free(samples);
    free(out);
    free(in);
    libdeflate_free_compressor(c);
    return 0;
#endif
}

static int cli_bench_zd(int level, long reps)
{
#if !defined(_WIN32)
    fprintf(stderr, "bench: only implemented for _WIN32 (QueryPerformanceCounter)\n");
    return 1;
#else
    _setmode(_fileno(stdin), _O_BINARY);
    size_t in_len;
    uint8_t *in = read_all_stdin(&in_len);
    if (!in || in_len == 0) {
        fprintf(stderr, "bench zd: failed to read stdin, or empty input\n");
        return 1;
    }
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
    struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();
    if (!c || !d) {
        fprintf(stderr, "bench zd: alloc failed\n");
        return 1;
    }
    size_t bound = libdeflate_zlib_compress_bound(c, in_len);
    uint8_t *compressed = (uint8_t *)malloc(bound);
    uint8_t *out = (uint8_t *)malloc(in_len);
    long long *samples = (long long *)malloc(sizeof(long long) * (size_t)reps);
    if (!compressed || !out || !samples) {
        fprintf(stderr, "bench zd: allocation failed\n");
        return 1;
    }
    /* Compress ONCE, untimed -- this bench measures decompress only. */
    size_t comp_len = libdeflate_zlib_compress(c, in, in_len, compressed, bound);
    if (comp_len == 0) {
        fprintf(stderr, "bench zd: initial compress failed\n");
        return 1;
    }
    double freq = qpc_freq_hz();
    size_t actual_out = 0;
    /* One untimed warm-up. */
    libdeflate_zlib_decompress(d, compressed, comp_len, out, in_len, &actual_out);
    for (long r = 0; r < reps; r++) {
        long long t0 = qpc_now();
        libdeflate_zlib_decompress(d, compressed, comp_len, out, in_len, &actual_out);
        long long t1 = qpc_now();
        samples[r] = (long long)((double)(t1 - t0) * 1e9 / freq);
    }
    print_bench_stats(samples, (size_t)reps, comp_len, actual_out);
    free(samples);
    free(out);
    free(compressed);
    free(in);
    libdeflate_free_decompressor(d);
    libdeflate_free_compressor(c);
    return 0;
#endif
}

/* ------------------------------------------------------------------------- */
/* Deterministic input generators (mirrored exactly on the Rust side).       */
/* ------------------------------------------------------------------------- */

/*
 * A 32-bit xorshift PRNG with a fixed seed. Documented formula so the Rust side
 * reproduces the identical byte stream:
 *   state starts at 0x12345678
 *   each step:  x ^= x << 13;  x ^= x >> 17;  x ^= x << 5  (all in u32 wrapping)
 *   the emitted byte is the low 8 bits of the new state.
 */
static void fill_pseudo_random(uint8_t *buf, size_t n)
{
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        buf[i] = (uint8_t)(x & 0xFFu);
    }
}

/*
 * Structured/text-like input: 1000 lines of the form
 *   "row <i>: the quick brown fox jumps over the lazy dog (<i>)\n"
 * where <i> is the decimal line index (0-based). Returns the byte length.
 * The Rust side builds the identical String with the same format.
 */
static size_t fill_structured(uint8_t *buf, size_t cap)
{
    size_t len = 0;
    for (int i = 0; i < 1000; i++) {
        char line[128];
        int m = snprintf(line, sizeof(line),
                         "row %d: the quick brown fox jumps over the lazy dog (%d)\n",
                         i, i);
        if (m < 0 || len + (size_t)m > cap)
            break;
        memcpy(buf + len, line, (size_t)m);
        len += (size_t)m;
    }
    return len;
}

/* ------------------------------------------------------------------------- */
/* Output helpers.                                                            */
/* ------------------------------------------------------------------------- */

static void print_hex_line(const char *key, const uint8_t *data, size_t n)
{
    fputs(key, stdout);
    fputs(" = ", stdout);
    for (size_t i = 0; i < n; i++)
        printf("%02x", data[i]);
    putchar('\n');
}

/* One named input buffer. */
struct input {
    const char *name;
    uint8_t    *data;
    size_t      len;
};

static int oracle_golden_main(void);

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s                          (no args: emit the golden-parity file to stdout)\n"
            "       %s zc <level>               (stdin=raw bytes -> stdout(binary)=zlib_compress output)\n"
            "       %s zdx <out_cap>            (stdin=zlib bytes -> stdout(text)=\"result=.. actual_in=.. actual_out=..\")\n"
            "       %s bench zc <level> <reps>  (stdin=raw bytes -> stdout(text)=\"median_ns=.. min_ns=..\")\n"
            "       %s bench zd <level> <reps>  (stdin=raw bytes -> stdout(text)=\"median_ns=.. min_ns=..\")\n",
            prog, prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
    /* A recognised mode word with a missing/garbage numeric argument, or an
     * UNRECOGNISED mode word, both fail loudly (usage + exit 2) instead of
     * silently falling through to the golden generator -- a typo'd CLI
     * invocation used to run (and pass!) the wrong thing entirely. */
    if (argc >= 2 && strcmp(argv[1], "zc") == 0) {
        long level;
        if (!parse_required_int_arg("zc", argc, argv, &level)) {
            print_usage(argv[0]);
            return 2;
        }
        return cli_zlib_compress((int)level);
    }
    if (argc >= 2 && strcmp(argv[1], "zdx") == 0) {
        long out_cap;
        if (!parse_required_int_arg("zdx", argc, argv, &out_cap)) {
            print_usage(argv[0]);
            return 2;
        }
        return cli_zlib_decompress_ex((size_t)out_cap);
    }
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        if (argc < 5) {
            fprintf(stderr, "bench: usage: %s bench {zc|zd} <level> <reps>\n", argv[0]);
            return 2;
        }
        long level, reps;
        if (!parse_int_arg_at("bench", argc, argv, 3, &level) ||
            !parse_int_arg_at("bench", argc, argv, 4, &reps) || reps < 1) {
            print_usage(argv[0]);
            return 2;
        }
        if (strcmp(argv[2], "zc") == 0)
            return cli_bench_zc((int)level, reps);
        if (strcmp(argv[2], "zd") == 0)
            return cli_bench_zd((int)level, reps);
        fprintf(stderr, "bench: unrecognised sub-mode %s (want zc|zd)\n", argv[2]);
        return 2;
    }
    if (argc >= 2) {
        fprintf(stderr, "unrecognised mode: %s\n", argv[1]);
        print_usage(argv[0]);
        return 2;
    }
    return oracle_golden_main();
}

static int oracle_golden_main(void)
{
    /* --- Build the fixed input set (definitions are the shared contract). --- */
    static const uint8_t short_str[] =
        "hello, deflate-rs vs libdeflate!\n";
    /* short_str length excludes the implicit NUL terminator. */
    const size_t short_len = sizeof(short_str) - 1;

    /* repetitive: 4096 bytes of the 3-byte cycle 'a','b','c' (buf[i]='a'+i%3). */
    const size_t rep_len = 4096;
    uint8_t *repetitive = (uint8_t *)malloc(rep_len);
    for (size_t i = 0; i < rep_len; i++)
        repetitive[i] = (uint8_t)('a' + (i % 3));

    /* all_same: 8192 bytes all 0x5A. */
    const size_t same_len = 8192;
    uint8_t *all_same = (uint8_t *)malloc(same_len);
    memset(all_same, 0x5A, same_len);

    /* pseudo_random: 65536 bytes from the documented xorshift32 PRNG. */
    const size_t rnd_len = 65536;
    uint8_t *pseudo_random = (uint8_t *)malloc(rnd_len);
    fill_pseudo_random(pseudo_random, rnd_len);

    /* structured: ~ up to 64 KiB of formatted text lines. */
    const size_t struct_cap = 131072;
    uint8_t *structured = (uint8_t *)malloc(struct_cap);
    size_t struct_len = fill_structured(structured, struct_cap);

    struct input inputs[] = {
        { "empty",         (uint8_t *)short_str, 0 },        /* len 0 => empty */
        { "short_str",     (uint8_t *)short_str, short_len },
        { "repetitive",    repetitive,           rep_len },
        { "all_same",      all_same,             same_len },
        { "pseudo_random", pseudo_random,        rnd_len },
        { "structured",    structured,           struct_len },
    };
    const size_t n_inputs = sizeof(inputs) / sizeof(inputs[0]);

    fputs("# exr-rs deflate-rs <-> C libdeflate golden parity (libdeflate "
          LIBDEFLATE_VERSION_STRING ")\n", stdout);

    /* --- Emit each input's exact bytes (so Rust can verify its reproduction). --- */
    for (size_t i = 0; i < n_inputs; i++) {
        char key[64];
        snprintf(key, sizeof(key), "input.%s", inputs[i].name);
        print_hex_line(key, inputs[i].data, inputs[i].len);
    }

    /* --- For every level 1..=12 and every input, compress and emit hex. --- */
    for (int level = 1; level <= 12; level++) {
        struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
        if (!c) {
            fprintf(stderr, "libdeflate_alloc_compressor(%d) failed\n", level);
            return 1;
        }
        for (size_t i = 0; i < n_inputs; i++) {
            size_t bound = libdeflate_zlib_compress_bound(c, inputs[i].len);
            uint8_t *out = (uint8_t *)malloc(bound);
            size_t n = libdeflate_zlib_compress(c, inputs[i].data, inputs[i].len,
                                                out, bound);
            if (n == 0) {
                fprintf(stderr,
                        "libdeflate_zlib_compress returned 0 (level %d, input %s)\n",
                        level, inputs[i].name);
                return 1;
            }
            char key[64];
            snprintf(key, sizeof(key), "level%d.%s", level, inputs[i].name);
            print_hex_line(key, out, n);
            free(out);
        }
        libdeflate_free_compressor(c);
    }

    /* --- CRC-32 golden values for every input. --- */
    for (size_t i = 0; i < n_inputs; i++) {
        uint32_t crc = libdeflate_crc32(0, inputs[i].data, inputs[i].len);
        char key[64];
        snprintf(key, sizeof(key), "crc32.%s", inputs[i].name);
        /* emit as 8 lowercase hex digits (the u32 value, not the LE bytes) */
        printf("%s = %08x\n", key, crc);
    }

    /* --- Gzip golden byte streams for every level 1..=12 and every input. --- */
    for (int level = 1; level <= 12; level++) {
        struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
        if (!c) {
            fprintf(stderr, "libdeflate_alloc_compressor(%d) failed (gzip)\n", level);
            return 1;
        }
        for (size_t i = 0; i < n_inputs; i++) {
            size_t bound = libdeflate_gzip_compress_bound(c, inputs[i].len);
            uint8_t *out = (uint8_t *)malloc(bound);
            size_t n = libdeflate_gzip_compress(c, inputs[i].data, inputs[i].len,
                                                out, bound);
            if (n == 0) {
                fprintf(stderr,
                        "libdeflate_gzip_compress returned 0 (level %d, input %s)\n",
                        level, inputs[i].name);
                return 1;
            }
            char key[64];
            snprintf(key, sizeof(key), "gzip%d.%s", level, inputs[i].name);
            print_hex_line(key, out, n);
            free(out);
        }
        libdeflate_free_compressor(c);
    }

    /* --- Decompression RESULT CODES: the LIBDEFLATE_SHORT_OUTPUT contract. ---
     *
     * libdeflate.h:201-203 / :222-229: with a NULL `actual_out_nbytes_ret` the
     * caller asserts it knows the exact uncompressed size, and a stream that
     * decompresses to FEWER than `out_nbytes_avail` bytes is SHORT_OUTPUT (2)
     * rather than success. deflate-rs used to have no way to express that, so
     * its facade silently returned a short Vec.
     *
     * Keys:
     *   zlibresult.exact.<input> = <result code, 2 hex digits>
     *   zlibresult.short.<input> = <result code, 2 hex digits>  (buffer 16 bytes too big)
     *   zlibresult.tight.<input> = <result code, 2 hex digits>  (buffer 1 byte too small)
     */
    {
        struct libdeflate_compressor   *c = libdeflate_alloc_compressor(6);
        struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();
        if (!c || !d) {
            fprintf(stderr, "alloc for the decompress-result cases failed\n");
            return 1;
        }
        for (size_t i = 0; i < n_inputs; i++) {
            size_t   bound = libdeflate_zlib_compress_bound(c, inputs[i].len);
            uint8_t *cmp   = (uint8_t *)malloc(bound);
            size_t   n     = libdeflate_zlib_compress(c, inputs[i].data, inputs[i].len,
                                                      cmp, bound);
            if (n == 0) {
                fprintf(stderr, "zlib_compress returned 0 for %s\n", inputs[i].name);
                return 1;
            }
            uint8_t *out = (uint8_t *)malloc(inputs[i].len + 16);
            enum libdeflate_result r;

            r = libdeflate_zlib_decompress(d, cmp, n, out, inputs[i].len, NULL);
            printf("zlibresult.exact.%s = %02x\n", inputs[i].name, (unsigned)r);

            r = libdeflate_zlib_decompress(d, cmp, n, out, inputs[i].len + 16, NULL);
            printf("zlibresult.short.%s = %02x\n", inputs[i].name, (unsigned)r);

            if (inputs[i].len > 0) {
                r = libdeflate_zlib_decompress(d, cmp, n, out, inputs[i].len - 1, NULL);
                printf("zlibresult.tight.%s = %02x\n", inputs[i].name, (unsigned)r);
            }

            free(out);
            free(cmp);
        }
        libdeflate_free_decompressor(d);
        libdeflate_free_compressor(c);
    }

    free(repetitive);
    free(all_same);
    free(pseudo_random);
    free(structured);
    return 0;
}
