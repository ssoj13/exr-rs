// Stand-alone test of the AE plugin's cache-key helper (no AE SDK needed):
//   cl /std:c++17 /EHsc /W3 /I..\src cache_key_test.cpp && cache_key_test.exe
#include "exv_cache_key.h"
#include <cstdio>
#include <cstdlib>

using namespace exvae;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main()
{
    // Numbered takes are DIFFERENT footage (the old routing key collapsed them).
    CHECK(RouteKey("C:/a/take_01.exv") != RouteKey("C:/a/take_02.exv"));
    CHECK(RouteKey("C:/a/shot.0001.exv") != RouteKey("C:/a/shot.0002.exv"));
    // Slash / ASCII case spellings of the same file are one footage item.
    CHECK(RouteKey("C:\\Clips\\Take_01.EXV") == RouteKey("c:/clips/take_01.exv"));
    // Non-ASCII bytes are kept verbatim (never folded or dropped).
    CHECK(RouteKey("C:/\xD0\x9A/a.exv") != RouteKey("C:/\xD0\x9B/a.exv"));

    const FileId a{100, 5000}, same{100, 5000}, newer{101, 5000}, bigger{100, 5001};
    const FileKey k = MakeKey("C:\\x\\a.exv", a);
    CHECK(k == MakeKey("c:/x/A.EXV", same));      // same footage + same content
    CHECK(k != MakeKey("c:/x/a.exv", newer));     // replaced file (mtime)
    CHECK(k != MakeKey("c:/x/a.exv", bigger));    // replaced file (size)
    CHECK(k != MakeKey("c:/x/b.exv", a));         // other footage
    CHECK(MakeKey("", FileId{}) == MakeKey("", FileId{}));

    if (g_fail == 0) std::puts("cache_key_test: OK");
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
