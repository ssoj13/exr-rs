"""P2/P3 smoke test: write -> read round-trips through the `OpenEXR` module.

Run after `python bootstrap.py b p` by installing the wheel (or `maturin develop`).
A subset of the reference test surface; the full reference `tests/` are the gate
for later phases (see crates/openexr-py/CLAUDE.md).
"""

import os
import tempfile
import warnings

import numpy as np

import OpenEXR as E

import data_root


def _tmp(name):
    return os.path.join(tempfile.gettempdir(), name)


def test_rgb_float32_roundtrip():
    h, w = 8, 5
    rgb = np.random.rand(h, w, 3).astype("float32")
    path = _tmp("oexr_rgb.exr")
    with E.File({"compression": E.ZIP_COMPRESSION, "type": E.scanlineimage},
                {"RGB": rgb}) as f:
        f.write(path)
    with E.File(path) as o:
        got = o.channels()["RGB"].pixels
        assert got.shape == (h, w, 3), got.shape
        assert got.dtype == np.float32
        assert np.array_equal(got, rgb)  # ZIP is lossless for float32
        assert o.header()["compression"] == E.ZIP_COMPRESSION
        assert o.header()["type"] == E.scanlineimage


def test_separate_channels():
    h, w = 8, 5
    rgb = np.random.rand(h, w, 3).astype("float32")
    path = _tmp("oexr_sep.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"RGB": rgb}) as f:
        f.write(path)
    with E.File(path, separate_channels=True) as o:
        r = o.channels()["R"].pixels
        assert r.shape == (h, w)
        assert np.array_equal(r, rgb[:, :, 0])
        assert o.channels()["R"].type() == E.HALF or o.channels()["R"].type() == E.FLOAT


def test_half_roundtrip_exact():
    h, w = 6, 4
    z = (np.arange(h * w).reshape(h, w) / 7.0).astype("float16")
    path = _tmp("oexr_half.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"Z": z}) as f:
        f.write(path)
    with E.File(path) as o:
        gz = o.channels()["Z"].pixels
        assert gz.dtype == np.float16, gz.dtype
        assert np.array_equal(gz, z)  # half is bit-exact through ZIP


def test_uint_roundtrip_exact():
    h, w = 6, 4
    u = np.arange(h * w).reshape(h, w).astype("uint32")
    path = _tmp("oexr_uint.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"id": u}) as f:
        f.write(path)
    with E.File(path) as o:
        gu = o.channels()["id"].pixels
        assert gu.dtype == np.uint32
        assert np.array_equal(gu, u)


def test_compression_extracted_not_defaulted():
    # Force a non-ZIP compression so a broken header-compression extraction (which
    # would silently default to ZIP) is caught.
    h, w = 8, 8
    rgb = np.random.rand(h, w, 3).astype("float32")
    path = _tmp("oexr_piz.exr")
    with E.File({"compression": E.PIZ_COMPRESSION}, {"RGB": rgb}) as f:
        f.write(path)
    with E.File(path) as o:
        assert o.header()["compression"] == E.PIZ_COMPRESSION
        assert np.array_equal(o.channels()["RGB"].pixels, rgb)  # PIZ lossless


def test_rgba_4channel():
    h, w = 7, 5
    rgba = np.random.rand(h, w, 4).astype("float32")
    path = _tmp("oexr_rgba.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"RGBA": rgba}) as f:
        f.write(path)
    with E.File(path) as o:
        got = o.channels()["RGBA"].pixels
        assert got.shape == (h, w, 4), got.shape
        assert np.array_equal(got, rgba)


def test_layer_3channel_reference_names():
    # A non-RGB 3-D array "X" must write X.R / X.G / X.B (reference naming).
    h, w = 5, 4
    x = np.random.rand(h, w, 3).astype("float32")
    path = _tmp("oexr_layer.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"X": x}) as f:
        f.write(path)
    with E.File(path, separate_channels=True) as o:
        names = set(o.channels().keys())
        assert {"X.R", "X.G", "X.B"} <= names, names
        assert np.array_equal(o.channels()["X.R"].pixels, x[:, :, 0])


def test_prefixed_rgba_grouping():
    # r3_surfaces.md item 1: the default (non-separate) read must group a PREFIXED
    # R/G/B/A set too (e.g. "left.R/G/B"), not only the bare top-level "R/G/B[/A]" —
    # matching the reference's channelNameToRGBA, which names the group "left" (NOT
    # "leftRGBA").
    h, w = 5, 4
    left = np.random.rand(h, w, 3).astype("float32")
    path = _tmp("oexr_prefixed_rgba.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"left": left}) as f:
        f.write(path)
    with E.File(path) as o:  # default: separate_channels=False
        names = set(o.channels().keys())
        assert "left" in names, names
        assert not ({"left.R", "left.G", "left.B"} & names), names
        got = o.channels()["left"].pixels
        assert got.shape == (h, w, 3), got.shape
        assert np.array_equal(got, left)


def test_prefixed_rgba_grouping_with_alpha():
    h, w = 5, 4
    right = np.random.rand(h, w, 4).astype("float32")
    path = _tmp("oexr_prefixed_rgba_a.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"right": right}) as f:
        f.write(path)
    with E.File(path) as o:
        got = o.channels()["right"].pixels
        assert got.shape == (h, w, 4), got.shape
        assert np.array_equal(got, right)


def test_lineorder_roundtrip():
    # r3_surfaces.md item 9: lineOrder used to be dropped on write (never copied onto
    # the Image), so it always came back INCREASING_Y regardless of what was set.
    h, w = 6, 5
    z = np.random.rand(h, w).astype("float32")
    path = _tmp("oexr_lineorder.exr")
    with E.File({"compression": E.ZIP_COMPRESSION, "lineOrder": E.DECREASING_Y}, {"Z": z}) as f:
        f.write(path)
    with E.File(path) as o:
        assert o.header()["lineOrder"] == E.DECREASING_Y
        assert np.array_equal(o.channels()["Z"].pixels, z)


def test_write_unknown_compression_raises():
    # r3_surfaces.md item 8: an out-of-range compression code used to be silently
    # reinterpreted as ZIP instead of raising.
    h, w = 3, 3
    z = np.random.rand(h, w).astype("float32")
    path = _tmp("oexr_badcomp.exr")
    with E.File({"compression": 99}, {"Z": z}) as f:
        try:
            f.write(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unknown compression code")


def test_data_window_honored():
    h, w = 4, 6
    z = np.random.rand(h, w).astype("float32")
    path = _tmp("oexr_dw.exr")
    dw = ((2, 3), (2 + w - 1, 3 + h - 1))
    with E.File({"compression": E.ZIP_COMPRESSION, "dataWindow": dw}, {"Z": z}) as f:
        f.write(path)
    with E.File(path) as o:
        got = o.header()["dataWindow"]
        assert (tuple(got[0]), tuple(got[1])) == dw, got
        assert np.array_equal(o.channels()["Z"].pixels, z)


def test_helper_classes():
    td = E.TileDescription(64, 64, E.ONE_LEVEL, E.ROUND_DOWN)
    assert td.xSize == 64 and td.ySize == 64
    assert td.mode == E.ONE_LEVEL
    r = E.Rational(30000, 1001)
    assert r.n == 30000 and r.d == 1001
    assert E.Rational(1, 2) == E.Rational(1, 2)
    kc = E.KeyCode(prefix=1, count=2)
    assert kc.prefix == 1 and kc.count == 2
    kc.prefix = 5
    assert kc.prefix == 5
    tc = E.TimeCode(hours=1, minutes=2, seconds=3, frame=4)
    assert (tc.hours, tc.minutes, tc.seconds, tc.frame) == (1, 2, 3, 4)
    tc.frame = 7
    assert tc.frame == 7
    taf = tc.timeAndFlags()
    tc2 = E.TimeCode()
    tc2.setTimeAndFlags(taf)
    assert (tc2.hours, tc2.minutes, tc2.seconds, tc2.frame) == (1, 2, 3, 7)
    p = E.PreviewRgba(10, 20, 30, 40)
    assert (p.r, p.g, p.b, p.a) == (10, 20, 30, 40)
    pi = E.PreviewImage(4, 3)
    assert pi.width() == 4 and pi.height() == 3
    # `.pixels` is a (H, W) structured array with r/g/b/a uint8 fields, matching
    # the reference's own `array_t<PreviewRgba>` shape/dtype (team-lead correction
    # 2026-09-23) -- NOT a flat (H, W, 4) array.
    assert pi.pixels.shape == (3, 4)
    assert pi.pixels.dtype.names == ("r", "g", "b", "a")


def test_header_keycode_timecode_preview():
    z = np.zeros((4, 4), "float32")
    kc = E.KeyCode(1, 2, 3, 4, 5)
    tc = E.TimeCode(1, 2, 3, 4)
    pv = E.PreviewImage(2, 2)
    # A structured element must be assigned a tuple (positional field values),
    # not a list -- numpy's structured-scalar assignment doesn't unpack a list.
    pv.pixels[0, 0] = (10, 20, 30, 40)
    pv.pixels[1, 1] = (50, 60, 70, 80)
    path = _tmp("oexr_kctcpv.exr")
    f = E.File(
        {"compression": E.ZIP_COMPRESSION, "keyCode": kc, "timeCode": tc, "preview": pv},
        {"Z": z},
    )
    f.write(path)
    with E.File(path) as o:
        h = o.header()
        assert h["keyCode"] == kc
        assert h["timeCode"] == tc
        assert np.array_equal(h["preview"].pixels, pv.pixels)


def test_header_writeback():
    z = np.zeros((4, 4), "float32")
    path = _tmp("oexr_hwb.exr")
    f = E.File(
        {
            "compression": E.ZIP_COMPRESSION,
            "comments": "test image",
            "longitude": -122.5,
            "screenWindowCenter": np.array([1.0, 2.0], "float32"),
        },
        {"Z": z},
    )
    f.write(path)
    with E.File(path) as o:
        h = o.header()
        assert h["comments"] == "test image"
        assert h["longitude"] == -122.5
        assert tuple(h["screenWindowCenter"]) == (1.0, 2.0)


def test_full_header_attributes():
    h, w = 4, 6
    z = np.zeros((h, w), "float32")
    path = _tmp("oexr_hdr.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"Z": z}) as f:
        f.write(path)
    with E.File(path) as o:
        hdr = o.header()
        assert hdr["compression"] == E.ZIP_COMPRESSION
        assert "dataWindow" in hdr and "displayWindow" in hdr
        assert "lineOrder" in hdr
        assert "pixelAspectRatio" in hdr
        assert "screenWindowWidth" in hdr
        swc = hdr["screenWindowCenter"]
        assert hasattr(swc, "shape") and tuple(swc.shape) == (2,)  # v2f -> numpy
        chans = hdr["channels"]
        assert "Z" in chans
        assert chans["Z"].type() == E.FLOAT


def test_is_openexr_file():
    z = np.zeros((4, 4), "float32")
    path = _tmp("oexr_isexr.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"Z": z}) as f:
        f.write(path)
    assert E.isOpenExrFile(path) is True
    assert E.isTiledOpenExrFile(path) is False  # we write scanline
    txt = _tmp("oexr_notexr.txt")
    with open(txt, "wb") as fp:
        fp.write(b"hello world, definitely not an exr")
    assert E.isOpenExrFile(txt) is False


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_channel_constructors():
    c = E.Channel("C", 2, 3)
    assert c.name == "C" and c.xSampling == 2 and c.ySampling == 3
    d = E.Channel()
    assert d.xSampling == 1 and d.ySampling == 1
    a = np.zeros((4, 5), "float32")
    assert E.Channel(a).type() == E.FLOAT
    cp = E.Channel("C", a)
    assert cp.name == "C" and cp.type() == E.FLOAT
    assert E.Channel(a, 2, 2).xSampling == 2
    assert E.Channel("C", a, 2, 2).ySampling == 2
    # errors
    assert _raises(lambda: E.Channel(1))
    assert _raises(lambda: E.Channel("C", 2))                     # lone sampling arg
    assert _raises(lambda: E.Channel(np.zeros(4, "uint32")))      # 1-D
    assert _raises(lambda: E.Channel(np.zeros((2, 2), "uint8")))  # bad dtype


def test_file_and_part_construction():
    f = E.File()  # empty file
    assert f.filename == ""
    assert list(f.parts) == []
    assert _raises(lambda: f.header())  # no parts -> raise
    assert _raises(lambda: f.channels())
    E.File({}, {})  # empty header/channels is OK

    p = E.Part({}, {}, name="party")
    assert p.name() == "party"
    assert p.type() == E.scanlineimage
    assert p.compression() == E.ZIP_COMPRESSION
    assert p.width() == 0 and p.height() == 0

    p2 = E.Part({"type": E.tiledimage, "compression": E.NO_COMPRESSION}, {})
    assert p2.type() == E.tiledimage
    assert p2.compression() == E.NO_COMPRESSION

    fp = E.File([E.Part({}, {}, "a"), E.Part({}, {}, "b")])
    assert len(fp.parts) == 2
    assert _raises(lambda: E.File([E.Part({}, {}), 2]))  # non-Part element


def test_part_indexed_header():
    z = np.zeros((4, 4), "float32")
    path = _tmp("oexr_pidx.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"Z": z}) as f:
        f.write(path)
    with E.File(path) as o:
        assert "compression" in o.header(0)
        assert _raises(lambda: o.header(1))   # out of range
        assert _raises(lambda: o.header(-1))
        assert _raises(lambda: o.channels(5))


def test_multipart_write():
    h, w = 6, 4
    Z0 = np.zeros((h, w), "float32")
    Z1 = np.ones((h, w), "float32")
    f = E.File([E.Part({}, {"Z": Z0}), E.Part({}, {"Z": Z1})])
    path = _tmp("oexr_2part.exr")
    f.write(path)
    with E.File(path) as o:
        assert len(o.parts) == 2
        assert o.parts[0].name() == "Part0"
        assert o.parts[1].name() == "Part1"
        assert o.parts[0].width() == w and o.parts[0].height() == h
        assert np.array_equal(o.parts[0].channels["Z"].pixels, Z0)
        assert np.array_equal(o.parts[1].channels["Z"].pixels, Z1)


@data_root.needs_data
def test_multipart_read():
    ref = str(data_root.openexr_images() / "Beachball" / "multipart.0001.exr")
    with E.File(ref) as o:
        assert len(o.parts) > 1, len(o.parts)
        saw_pixels = False
        for p in o.parts:
            assert p.width() > 0 and p.height() > 0
            for c in p.channels.values():
                if hasattr(c.pixels, "shape"):
                    assert c.pixels.shape[0] == p.height()
                    saw_pixels = True
        assert saw_pixels, "at least one part should have decoded pixels"


def test_parts_single():
    h, w = 5, 7
    z = np.zeros((h, w), "float32")
    path = _tmp("oexr_parts.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"Z": z}) as f:
        f.write(path)
    with E.File(path) as o:
        assert len(o.parts) == 1
        p = o.parts[0]
        assert p.width() == w and p.height() == h
        assert p.type() == E.scanlineimage
        assert "Z" in p.channels  # Part.channels is an attribute (like the reference)
        assert "compression" in p.header


def test_imath_companion_module():
    import Imath
    pt = Imath.PixelType(Imath.PixelType.HALF)
    assert int(pt) == 1
    assert pt == Imath.PixelType(Imath.PixelType.HALF)
    assert repr(pt) == "HALF"
    ch = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    assert int(ch.type) == 2 and ch.xSampling == 1
    b = Imath.Box2i(Imath.point(0, 0), Imath.point(9, 9))
    assert b.min.x == 0 and b.max.y == 9
    c = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    assert int(c) == 3 and repr(c) == "ZIP_COMPRESSION"


def test_legacy_chunked_write():
    import Imath
    from array import array

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    w, h, step = 8, 6, 2
    data = array("f", [float(i) for i in range(w * h)]).tobytes()

    hdr = E.Header(w, h)
    hdr["channels"] = {"R": Imath.Channel(FLOAT)}
    p0 = _tmp("chunk0.exr")
    o0 = E.OutputFile(p0, hdr)
    o0.writePixels({"R": data})
    o0.close()

    hdr2 = E.Header(w, h)
    hdr2["channels"] = {"R": Imath.Channel(FLOAT)}
    p1 = _tmp("chunk1.exr")
    o1 = E.OutputFile(p1, hdr2)
    for y in range(0, h, step):
        sub = data[y * w * 4 : (y + step) * w * 4]
        o1.writePixels({"R": sub}, step)
    o1.close()

    r0 = E.InputFile(p0).channel("R")
    r1 = E.InputFile(p1).channel("R")
    assert r0 == r1 == data


def test_deep_separate():
    h, w = 5, 4
    U = np.empty((h, w), dtype=object)
    F = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            i = y * w + x
            if i % 3 == 0:
                U[y, x] = np.array([i, i + 1], dtype="uint32")
                F[y, x] = np.array([float(i), float(i + 1)], dtype="float32")
            else:
                U[y, x] = None
                F[y, x] = None
    header = {"compression": E.ZIPS_COMPRESSION, "type": E.deepscanline}
    path = _tmp("oexr_deep.exr")
    with E.File(header, {"U": U, "F": F}) as out:
        out.write(path)
    with E.File(path, separate_channels=True) as inp:
        cu = inp.channels()["U"].pixels
        cf = inp.channels()["F"].pixels
        assert cu.shape == (h, w)
        for y in range(h):
            for x in range(w):
                if U[y, x] is None:
                    assert cu[y, x] is None
                else:
                    assert np.array_equal(cu[y, x], U[y, x])
                    assert np.array_equal(cf[y, x], F[y, x])


def test_deep_mismatched_channel_counts_raise():
    # Bug S2: every deep channel at a pixel must share the same sample count.
    # A differing count, or None-vs-samples, must be rejected before writing
    # (it would otherwise corrupt the on-disk heap layout).
    h, w = 2, 2
    header = {"compression": E.ZIPS_COMPRESSION, "type": E.deepscanline}

    # Differing sample counts between channels A and B at pixel (0, 0).
    A = np.empty((h, w), dtype=object)
    B = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            A[y, x] = np.array([1.0, 2.0], dtype="float32")
            B[y, x] = np.array([1.0, 2.0], dtype="float32")
    B[0, 0] = np.array([9.0], dtype="float32")  # 1 sample vs 2 -> mismatch
    assert _raises(lambda: E.File(header, {"A": A, "B": B}).write(_tmp("oexr_deep_mismatch.exr")))

    # None in one channel where the other has samples is also a mismatch.
    C = np.empty((h, w), dtype=object)
    D = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            C[y, x] = np.array([1.0], dtype="float32")
            D[y, x] = np.array([1.0], dtype="float32")
    D[1, 1] = None
    assert _raises(lambda: E.File(header, {"C": C, "D": D}).write(_tmp("oexr_deep_none.exr")))


def test_deep_rgb_grouping():
    h, w = 4, 3
    R = np.empty((h, w), dtype=object)
    G = np.empty((h, w), dtype=object)
    B = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            i = y * w + x + 1
            if i % 2 == 0:
                R[y, x] = np.array([j for j in range(1, i)], dtype="float32")
                G[y, x] = np.array([j * 10 for j in range(1, i)], dtype="float32")
                B[y, x] = np.array([j * 100 for j in range(1, i)], dtype="float32")
            else:
                R[y, x] = G[y, x] = B[y, x] = None
    header = {"compression": E.ZIPS_COMPRESSION, "type": E.deepscanline}
    path = _tmp("oexr_deep_rgb.exr")
    with E.File(header, {"R": R, "G": G, "B": B}) as out:
        out.write(path)
    with E.File(path) as inp:
        C = inp.channels()["RGB"].pixels
        for y in range(h):
            for x in range(w):
                p = C[y, x]
                if p is None:
                    assert R[y, x] is None
                else:
                    for i in range(p.shape[0]):
                        assert p[i, :].tolist() == [R[y, x][i], G[y, x][i], B[y, x][i]]


def test_deep_grouped_rgb_write():
    # r3_surfaces.md item 7 (second half): write_deep used to only accept an
    # already-flat, ungrouped (H,W) object array per channel; a GROUPED array
    # (each pixel an (nsamples, 3) array, exactly what a deep read's own "RGB"
    # channel hands back) failed with a confusing obj_shape/indexing error
    # instead of being split back into R/G/B on write.
    h, w = 3, 2
    RGB = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            RGB[y, x] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    header = {"compression": E.ZIPS_COMPRESSION, "type": E.deepscanline}
    path = _tmp("oexr_deep_grouped_write.exr")
    with E.File(header, {"RGB": RGB}) as f:
        f.write(path)
    with E.File(path) as inp:
        C = inp.channels()["RGB"].pixels
        for y in range(h):
            for x in range(w):
                assert np.array_equal(C[y, x], RGB[y, x])


def test_deep_nonorigin_window():
    minx, miny = 100, 100
    w, h = 4, 6
    F = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            i = y * w + x
            F[y, x] = np.array([float(i), float(i + 1)], dtype="float32") if i % 2 == 0 else None
    header = {
        "compression": E.ZIPS_COMPRESSION,
        "type": E.deepscanline,
        "dataWindow": ((minx, miny), (minx + w - 1, miny + h - 1)),
    }
    path = _tmp("oexr_deep_nonorigin.exr")
    with E.File(header, {"F": F}) as out:
        out.write(path)
    with E.File(path, separate_channels=True) as inp:
        dw = inp.header()["dataWindow"]
        assert tuple(dw[0]) == (minx, miny)
        assert tuple(dw[1]) == (minx + w - 1, miny + h - 1)
        cf = inp.channels()["F"].pixels
        for y in range(h):
            for x in range(w):
                if F[y, x] is None:
                    assert cf[y, x] is None
                else:
                    assert np.array_equal(cf[y, x], F[y, x])


def test_deep_negative_and_large_origin():
    # Bug S1: the local-raster count buffer must round-trip a negative-origin and
    # a large-positive-origin data window without a panic (negative absolute
    # address) or a huge allocation (origin-sized buffer).
    for minx, miny in [(-8, -5), (1024, -1024), (100000, 100000)]:
        w, h = 4, 3
        F = np.empty((h, w), dtype=object)
        for y in range(h):
            for x in range(w):
                i = y * w + x
                F[y, x] = (
                    np.array([float(i), float(i + 1)], dtype="float32") if i % 2 == 0 else None
                )
        header = {
            "compression": E.ZIPS_COMPRESSION,
            "type": E.deepscanline,
            "dataWindow": ((minx, miny), (minx + w - 1, miny + h - 1)),
        }
        path = _tmp(f"oexr_deep_origin_{minx}_{miny}.exr")
        with E.File(header, {"F": F}) as out:
            out.write(path)
        with E.File(path, separate_channels=True) as inp:
            dw = inp.header()["dataWindow"]
            assert tuple(dw[0]) == (minx, miny)
            cf = inp.channels()["F"].pixels
            for y in range(h):
                for x in range(w):
                    if F[y, x] is None:
                        assert cf[y, x] is None
                    else:
                        assert np.array_equal(cf[y, x], F[y, x])


def test_deep_tiled():
    h, w = 5, 4
    U = np.empty((h, w), dtype=object)
    F = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            i = y * w + x + 2
            if i % 3 == 0:
                U[y, x] = np.array([j for j in range(1, i)], dtype="uint32")
                F[y, x] = np.array([float(j * 100) for j in range(1, i)], dtype="float32")
            else:
                U[y, x] = None
                F[y, x] = None
    header = {"compression": E.ZIPS_COMPRESSION, "type": E.deeptile, "tiles": E.TileDescription()}
    path = _tmp("oexr_deeptile.exr")
    with E.File(header, {"U": U, "F": F}) as out:
        out.write(path)
    with E.File(path, separate_channels=True) as inp:
        cu = inp.channels()["U"].pixels
        cf = inp.channels()["F"].pixels
        assert cu.shape == (h, w)
        for y in range(h):
            for x in range(w):
                if U[y, x] is None:
                    assert cu[y, x] is None
                else:
                    assert np.array_equal(cu[y, x], U[y, x])
                    assert np.array_equal(cf[y, x], F[y, x])


def test_legacy_byte_api():
    import Imath
    from array import array

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    UINT = Imath.PixelType(Imath.PixelType.UINT)
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    path = _tmp("oexr_legacy.exr")

    # write two FLOAT channels as bytes, read them back exactly
    w, h = 8, 4
    size = w * h
    hdr = E.Header(w, h)
    hdr["channels"] = {"R": Imath.Channel(FLOAT), "G": Imath.Channel(FLOAT)}
    o = E.OutputFile(path, hdr)
    r = array("f", list(range(size))).tobytes()
    g = array("f", list(range(size, 2 * size))).tobytes()
    o.writePixels({"R": r, "G": g})
    o.close()

    i = E.InputFile(path)
    assert i.channel("R") == r
    assert i.channel("G") == g
    assert i.channels(["R", "G"]) == [i.channel("R"), i.channel("G")]
    assert set(i.header()["channels"].keys()) == {"R", "G"}
    assert i.header()["dataWindow"].max.x == w - 1
    assert i.isComplete()

    # write UINT, read as FLOAT (type conversion)
    original = [0, 1, 33, 79218]
    hdr2 = E.Header(len(original), 1)
    hdr2["channels"] = {"L": Imath.Channel(UINT)}
    o2 = E.OutputFile(path, hdr2)
    o2.writePixels({"L": array("I", original).tobytes()})
    o2.close()
    xin = E.InputFile(path)
    assert array("f", xin.channel("L", FLOAT)).tolist() == [float(x) for x in original]
    assert len(xin.channel("L", HALF)) == len(xin.channel("L", FLOAT)) // 2

    # scanline slice: one row of a FLOAT channel = 4 * width bytes
    hdr3 = E.Header(w, h)
    hdr3["channels"] = {"R": Imath.Channel(FLOAT)}
    o3 = E.OutputFile(path, hdr3)
    o3.writePixels({"R": array("f", list(range(size))).tobytes()})
    o3.close()
    line = E.InputFile(path).channel("R", FLOAT, scanLine1=2, scanLine2=2)
    assert len(line) == 4 * w

    # invalid paths raise
    assert _raises(lambda: E.InputFile("/bad/place/nope.exr"))
    assert _raises(lambda: E.OutputFile("/bad/place/nope.exr", E.Header(4, 4)))


def test_legacy_uint_to_half_overflow_boundary():
    # design_r5.md W4.6: legacy UINT->HALF conversion used to go through the
    # generic float_to_half rounding rule (rounds 65505..65519 down to the
    # largest finite half, 65504), but C++'s uintToHalf (ImfConvert.cpp) is
    # stricter: ANY value > HALF_MAX (65504) becomes +inf, with no rounding
    # window. Oracle-verified in tests/s01_half_boundary.rs.
    import Imath
    from array import array
    import struct

    UINT = Imath.PixelType(Imath.PixelType.UINT)
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    path = _tmp("oexr_uint_half_boundary.exr")
    values = [65504, 65505, 65519, 70000]
    hdr = E.Header(len(values), 1)
    hdr["channels"] = {"L": Imath.Channel(UINT)}
    o = E.OutputFile(path, hdr)
    o.writePixels({"L": array("I", values).tobytes()})
    o.close()

    half_bytes = E.InputFile(path).channel("L", HALF)
    half_vals = struct.unpack(f"<{len(values)}e", half_bytes)
    assert half_vals[0] == 65504.0, half_vals  # exactly HALF_MAX: still finite
    assert half_vals[1] == float("inf"), half_vals  # > HALF_MAX: must be +inf
    assert half_vals[2] == float("inf"), half_vals
    assert half_vals[3] == float("inf"), half_vals


def test_legacy_scanline_range_validation():
    # Bug S4: InputFile.channel must reject an inverted or out-of-window scanline
    # range instead of underflow-panicking inside the native extension.
    # Exception class updated to TypeError (was ValueError) to match the
    # reference exactly -- PyOpenEXR_old.cpp:407-419 raises TypeError for all
    # three of these ("scanLine1 must be <= scanLine2", "scanLine1/2 cannot be
    # outside dataWindow"), never ValueError.
    import Imath
    from array import array

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    path = _tmp("oexr_legacy_range.exr")
    w, h = 4, 4  # dataWindow y in [0, 3]
    hdr = E.Header(w, h)
    hdr["channels"] = {"R": Imath.Channel(FLOAT)}
    o = E.OutputFile(path, hdr)
    o.writePixels({"R": array("f", list(range(w * h))).tobytes()})
    o.close()

    i = E.InputFile(path)
    assert len(i.channel("R", FLOAT, 1, 2)) == 2 * w * 4  # valid: rows 1..=2
    for bad in [(2, 1), (-5, 2), (0, 99)]:  # inverted, below window, above window
        try:
            i.channel("R", FLOAT, *bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for scanline range {bad}")
    # invalid Header dimensions raise instead of building an inverted window
    assert _raises(lambda: E.Header(0, 4))


def test_legacy_exception_classes_match_reference():
    # design_r5.md: the legacy API (PyOpenEXR_old.cpp v3.4.15) uses a specific
    # exception-class split that differs from the new API's RuntimeError/
    # ValueError split -- TypeError for argument/structural errors, IOError for
    # actual I/O or file-format failures, OverflowError for allocation-size
    # overflow. No official test covers this, so we assert it ourselves.
    import Imath
    from array import array

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    UINT = Imath.PixelType(Imath.PixelType.UINT)

    # InputFile.channel: unknown channel name -> TypeError
    # (PyOpenEXR_old.cpp:427-428 "There is no channel '%s' in the image").
    path = _tmp("oexr_legacy_exc_1.exr")
    hdr = E.Header(2, 2)
    hdr["channels"] = {"R": Imath.Channel(FLOAT)}
    o = E.OutputFile(path, hdr)
    o.writePixels({"R": array("f", [0.0, 1.0, 2.0, 3.0]).tobytes()})
    o.close()
    inp = E.InputFile(path)
    try:
        inp.channel("nonexistent")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for an unknown channel name")

    # OutputFile.writePixels (via close()): wrong byte-buffer size for the
    # declared channel type -> TypeError (PyOpenEXR_old.cpp:1198-1208 "Data for
    # channel '%s' should have size %zu but got %zu").
    path2 = _tmp("oexr_legacy_exc_2.exr")
    hdr2 = E.Header(4, 4)
    hdr2["channels"] = {"R": Imath.Channel(FLOAT)}
    o2 = E.OutputFile(path2, hdr2)
    o2.writePixels({"R": array("f", [0.0, 1.0]).tobytes()})  # way too short
    try:
        o2.close()
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for a mismatched channel byte size")

    # Header dataWindow overflowing i32 (width/height computation) -> OverflowError
    # (PyOpenEXR_old.cpp:469-475 "Integer overflow computing allocation size").
    import Imath as _Imath

    hdr3 = {
        "dataWindow": _Imath.Box2i(_Imath.point(-2147483648, 0), _Imath.point(2147483647, 0)),
        "channels": {"R": Imath.Channel(FLOAT)},
    }
    path3 = _tmp("oexr_legacy_exc_3.exr")
    o3 = E.OutputFile(path3, hdr3)
    o3.writePixels({"R": b""})
    try:
        o3.close()
    except OverflowError:
        pass
    else:
        raise AssertionError("expected OverflowError for an i32-overflowing dataWindow width")

    # OutputFile open failure (bad path) -> IOError (already covered generically
    # by test_legacy_byte_api's "_raises" checks; assert the exact class here).
    try:
        E.OutputFile("/nonexistent/dir/x.exr", E.Header(4, 4))
    except IOError:
        pass
    else:
        raise AssertionError("expected IOError for an unwritable OutputFile path")

    # InputFile open failure (bad path) -> IOError.
    try:
        E.InputFile("/nonexistent/dir/x.exr")
    except IOError:
        pass
    else:
        raise AssertionError("expected IOError for a nonexistent InputFile path")

    # PixelType coercion failure (not 0/1/2) -> TypeError
    # (PyOpenEXR_old.cpp:436 "Invalid PixelType object" / :459 "Unknown type").
    path4 = _tmp("oexr_legacy_exc_4.exr")
    hdr4 = E.Header(2, 2)
    hdr4["channels"] = {"U": Imath.Channel(UINT)}
    o4 = E.OutputFile(path4, hdr4)
    o4.writePixels({"U": array("I", [0, 1, 2, 3]).tobytes()})
    o4.close()
    inp4 = E.InputFile(path4)

    class FakePixelType:
        def __int__(self):
            return 99

    try:
        inp4.channel("U", FakePixelType())
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for an invalid PixelType code")


def test_corrupted_single_part_pixel_data_is_skipped_not_raised():
    """A single-part file whose pixel data fails to decode (truncated/corrupt
    chunk) must be SKIPPED, not raised -- opening the file itself still
    succeeds with zero parts. This matches real C++ PyOpenEXR's `PyFile`
    constructor, which loops every part generically (single-part files go
    through the exact same code path as multipart ones) and wraps the pixel
    read in `try { ... } catch (const std::invalid_argument&) { throw; }
    catch (const std::exception& e) { py::print("Warning: ...", e.what()); }`
    (`PyOpenEXR.cpp:186-231`, esp. `:216-230`) -- any OTHER exception during
    a part's pixel decode is swallowed and that part is dropped, never
    raised to the caller. `read_multipart` already reproduced this
    (`test_unfinished_multipart_load` above); `read_single` did not, until
    this fix -- confirmed by this test failing (raising `RuntimeError`
    instead of returning a 0-part `File`) before the `file.rs::read_single`
    fix that mirrors `read_multipart`'s `.ok()` + `warnings.warn` pattern.
    We use `warnings.warn` in place of C++'s `printf` to stdout: visible and
    filterable, a strict improvement, not a behavioural gap in the drop-in
    contract.
    """
    h, w = 8, 6
    rgb = np.random.rand(h, w, 3).astype("float32")
    path = _tmp("oexr_corrupt_single.exr")
    with E.File({"compression": E.ZIP_COMPRESSION}, {"RGB": rgb}) as f:
        f.write(path)

    # Truncate well into the pixel data (past the header + chunk offset
    # table, which for an 8x6 image is only a few hundred bytes) so the
    # single ZIP-compressed chunk fails to inflate instead of merely
    # decoding stale/garbage bytes -- a clean, deterministic decode failure.
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.truncate(size - 16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opened = E.File(path)
        assert len(opened.parts) == 0, "the only (corrupt) part must be skipped, not raised"
        assert any(
            "unreadable" in str(w.message) or "incomplete" in str(w.message) for w in caught
        ), "expected a warnings.warn() for the skipped part (PyOpenEXR.cpp:227-229 printf equivalent)"


def test_multipart_coalesce_rejection_raises_value_error():
    """Item 6b: `read_multipart` has the same `PyOpenEXR.cpp:196-231` split as
    `read_single` -- `validateCoalescedChannels` (`:196-197`) runs BEFORE the
    per-part try/catch, so a malformed channel layout (mixed-type or
    colliding literal "R"/"G"/"B" keys under a layer prefix) must raise
    `ValueError`, not be silently skipped like a genuine pixel-decode
    failure would be. Modeled on the official `test_literal_rgb_key_
    collision_flat_rejected` (single-part), applied across two parts to
    prove the SAME rejection reaches the caller from a MULTIpart read.
    """
    h, w = 2, 2
    ok_channels = {"Z": np.zeros((h, w), "float32")}
    bad_channels = {
        "left": np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float16"),
        "left.R": np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float16"),
        "left.G": np.array([[20.0, 21.0], [22.0, 23.0]], dtype="float16"),
        "left.B": np.array([[30.0, 31.0], [32.0, 33.0]], dtype="float16"),
    }
    f = E.File([E.Part({}, ok_channels, "good"), E.Part({}, bad_channels, "bad")])
    path = _tmp("oexr_multipart_coalesce_bad.exr")
    f.write(path)

    try:
        E.File(path)
    except ValueError as e:
        assert "separate_channels" in str(e)
    else:
        raise AssertionError("expected ValueError for the colliding-key part")


def test_multipart_broken_chunk_skips_only_that_part():
    """Item 6b: a corrupt/truncated chunk in ONE part of a multipart file must
    skip only that part (with a `warnings.warn`, `PyOpenEXR.cpp:227-229`'s
    printf equivalent) while the other, intact parts still read normally --
    the multipart analogue of `test_corrupted_single_part_pixel_data_is_
    skipped_not_raised` above.
    """
    h, w = 6, 4
    z_good = np.zeros((h, w), "float32")
    z_bad = np.ones((h, w), "float32")
    f = E.File([
        E.Part({"compression": E.ZIP_COMPRESSION}, {"Z": z_good}, "good"),
        E.Part({"compression": E.ZIP_COMPRESSION}, {"Z": z_bad}, "bad"),
    ])
    path = _tmp("oexr_multipart_broken_chunk.exr")
    f.write(path)

    # Truncate the file so the LAST part's chunk data is cut off (parts are
    # written in order, so the first part's chunk stays intact), leaving one
    # broken part and one good one.
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.truncate(size - 16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opened = E.File(path)
        assert len(opened.parts) == 1, "only the broken part must be skipped"
        assert opened.parts[0].name() == "good"
        assert np.array_equal(opened.parts[0].channels["Z"].pixels, z_good)
        assert any(
            "unreadable" in str(w.message) or "incomplete" in str(w.message) for w in caught
        ), "expected a warnings.warn() for the skipped part"


def test_multipart_corrupted_chunk_data_skips_only_that_part():
    """Item 6c: `test_multipart_broken_chunk_skips_only_that_part` truncates
    the file, which makes the chunk OFFSET TABLE itself incomplete -- that
    hits `!is_complete()`'s short-circuit to `None`, BEFORE
    `Image::from_input_file`/`from_tiled_input_file` (the analogue of C++'s
    `readPixels`/`readTiles`) is ever called, so it does not actually
    exercise the `catch` this item's fix added (`file.rs::read_multipart`'s
    `Err(_) => None` arm). This test corrupts bytes strictly INSIDE the last
    part's compressed chunk PAYLOAD, leaving the file's total size (and
    therefore every chunk offset) unchanged, so the offset table stays fully
    valid and `is_complete()` still reports true -- `input_part()` opens
    cleanly and `Image::from_input_file` is the one that fails (ZIP inflate
    on garbage bytes), landing in the real catch path.
    """
    h, w = 6, 4
    z_good = np.zeros((h, w), "float32")
    z_bad = np.ones((h, w), "float32")
    f = E.File([
        E.Part({"compression": E.ZIP_COMPRESSION}, {"Z": z_good}, "good"),
        E.Part({"compression": E.ZIP_COMPRESSION}, {"Z": z_bad}, "bad"),
    ])
    path = _tmp("oexr_multipart_corrupt_chunk.exr")
    f.write(path)

    # Overwrite (not truncate) the last 16 bytes with garbage: same file
    # size, same chunk offsets, but the tail of the last part's ZIP stream
    # is now corrupt.
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.seek(size - 16)
        fh.write(b"\xff" * 16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opened = E.File(path)
        assert len(opened.parts) == 1, "only the corrupted part must be skipped"
        assert opened.parts[0].name() == "good"
        assert np.array_equal(opened.parts[0].channels["Z"].pixels, z_good)
        assert any(
            "unreadable" in str(w.message) or "incomplete" in str(w.message) for w in caught
        ), "expected a warnings.warn() for the skipped part"


def test_nonutf8_names_raise_unicode_decode_error():
    """C-08: a non-UTF-8 channel/attribute name must raise `UnicodeDecodeError`
    when the header is decoded to a Python dict, matching real C++ PyOpenEXR
    (`PyOpenEXR.cpp` converts every attribute/channel name with `py::str(name)`
    on the raw `std::string` C++ `Header`/`ChannelList` never UTF-8-checks —
    pybind11's `str(const std::string&)` calls `PyUnicode_FromStringAndSize`,
    which raises `UnicodeDecodeError` on invalid UTF-8. Verified by reading
    `PyOpenEXR.cpp` v3.4.15 directly (attribute names: `header[py::str(name)]`
    where `name = a.name()`; channel names: `py::str(py_channel_name)` where
    `py_channel_name = c.name()`), not predicted.

    The fixture is the C-08 oracle golden (`golden_nonutf8_names.exr`): a
    channel named "Z\xE9" and an attribute named "\xFFattr", both invalid
    UTF-8, written by real C++ OpenEXR 3.4.15
    (`crates/exr-core/tests/parity/oracle.cpp`'s `write-nonutf8-names`).
    """
    fixture = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "exr-core", "tests", "parity", "golden_nonutf8_names.exr",
    )
    if not os.path.exists(fixture):
        # Not committed / not regenerated in this checkout -- same convention
        # as the Rust-side parity gate (fail loudly only when it SHOULD exist;
        # here we skip, since this file has no EXR_ALLOW_SKIP-style opt-out).
        print(f"  (skip: fixture not found: {fixture})")
        return
    try:
        E.File(fixture)
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError(
            "expected UnicodeDecodeError opening a file with a non-UTF-8 channel/attribute name"
        )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: OK")
    print("ALL P2/P3 SMOKE OK")
