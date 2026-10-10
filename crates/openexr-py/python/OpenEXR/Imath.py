# SPDX-License-Identifier: BSD-3-Clause
"""Imath — the deprecated companion module of the OpenEXR Python API.

A small set of pure-Python data-holder classes (points, boxes, enums, channel
descriptors) used by the LEGACY ``OpenEXR.InputFile`` / ``OutputFile`` API. Kept
for drop-in compatibility; new code should use the numpy ``OpenEXR.File`` API and
the native enum types. No format logic lives here — these are plain value classes.
"""


class chromaticity:
    """A CIE (x, y) chromaticity coordinate. Deprecated."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return repr((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)


class point:
    """A 2-D point (x, y). Deprecated."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return repr((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)


class V2i(point):
    """Integer 2-vector. Deprecated."""


class V2f(point):
    """Float 2-vector. Deprecated."""


class Box:
    """An axis-aligned box (min, max). Deprecated."""

    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def __repr__(self):
        return repr(self.min) + " - " + repr(self.max)

    def __eq__(self, other):
        return (self.min, self.max) == (other.min, other.max)


class Box2i(Box):
    """Integer box. Deprecated."""


class Box2f(Box):
    """Float box. Deprecated."""


class Chromaticities:
    """RGB + white chromaticities. Deprecated."""

    def __init__(self, red=None, green=None, blue=None, white=None):
        self.red = red
        self.green = green
        self.blue = blue
        self.white = white

    def __repr__(self):
        return f"{self.red!r} {self.green!r} {self.blue!r} {self.white!r}"


class Enumerated:
    """Base for the deprecated named-int enums."""

    def __init__(self, v):
        self.v = getattr(self, v) if isinstance(v, str) and v in self.names else v

    def __repr__(self):
        return self.names[self.v]

    def __eq__(self, other):
        return self.v == (other.v if isinstance(other, Enumerated) else other)

    def __int__(self):
        return int(self.v)


class LineOrder(Enumerated):
    INCREASING_Y = 0
    DECREASING_Y = 1
    RANDOM_Y = 2
    names = ["INCREASING_Y", "DECREASING_Y", "RANDOM_Y"]


class Compression(Enumerated):
    NO_COMPRESSION = 0
    RLE_COMPRESSION = 1
    ZIPS_COMPRESSION = 2
    ZIP_COMPRESSION = 3
    PIZ_COMPRESSION = 4
    PXR24_COMPRESSION = 5
    B44_COMPRESSION = 6
    B44A_COMPRESSION = 7
    DWAA_COMPRESSION = 8
    DWAB_COMPRESSION = 9
    HTJ2K256_COMPRESSION = 10
    HTJ2K32_COMPRESSION = 11
    names = [
        "NO_COMPRESSION", "RLE_COMPRESSION", "ZIPS_COMPRESSION", "ZIP_COMPRESSION",
        "PIZ_COMPRESSION", "PXR24_COMPRESSION", "B44_COMPRESSION", "B44A_COMPRESSION",
        "DWAA_COMPRESSION", "DWAB_COMPRESSION", "HTJ2K256_COMPRESSION", "HTJ2K32_COMPRESSION",
    ]


class PixelType(Enumerated):
    UINT = 0
    HALF = 1
    FLOAT = 2
    names = ["UINT", "HALF", "FLOAT"]


class Channel:
    """A channel descriptor: pixel type + sampling. Deprecated."""

    def __init__(self, type=None, xSampling=1, ySampling=1):
        self.type = type if type is not None else PixelType(PixelType.HALF)
        self.xSampling = xSampling
        self.ySampling = ySampling
        if not isinstance(self.type, PixelType):
            raise TypeError("type needs to be a PixelType.")

    def __repr__(self):
        return f"{self.type!r} {(self.xSampling, self.ySampling)!r}"

    def __eq__(self, other):
        return (self.type, self.xSampling, self.ySampling) == (
            other.type, other.xSampling, other.ySampling,
        )


class Rational:
    def __init__(self, n, d):
        self.n = n
        self.d = d

    def __repr__(self):
        return repr("%s/%s (%.3f)" % (self.n, self.d, self.n / float(self.d)))

    def __eq__(self, other):
        return self.n == other.n and self.d == other.d


class TimeCode:
    def __init__(self, hours, minutes, seconds, frame, dropFrame=False, colorFrame=False,
                 fieldPhase=False, bgf0=False, bgf1=False, bgf2=False,
                 binaryGroup1=0, binaryGroup2=0, binaryGroup3=0, binaryGroup4=0,
                 binaryGroup5=0, binaryGroup6=0, binaryGroup7=0, binaryGroup8=0):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.frame = frame
        self.dropFrame = dropFrame
        self.colorFrame = colorFrame
        self.fieldPhase = fieldPhase
        self.bgf0 = bgf0
        self.bgf1 = bgf1
        self.bgf2 = bgf2
        self.binaryGroup1 = binaryGroup1
        self.binaryGroup2 = binaryGroup2
        self.binaryGroup3 = binaryGroup3
        self.binaryGroup4 = binaryGroup4
        self.binaryGroup5 = binaryGroup5
        self.binaryGroup6 = binaryGroup6
        self.binaryGroup7 = binaryGroup7
        self.binaryGroup8 = binaryGroup8

    def __repr__(self):
        return ("<Imath.TimeCode { %02d:%02d:%02d:%02d }>"
                % (self.hours, self.minutes, self.seconds, self.frame))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class KeyCode:
    def __init__(self, filmMfcCode=0, filmType=0, prefix=0, count=0, perfOffset=0,
                 perfsPerFrame=4, perfsPerCount=64):
        self.filmMfcCode = filmMfcCode
        self.filmType = filmType
        self.prefix = prefix
        self.count = count
        self.perfOffset = perfOffset
        self.perfsPerFrame = perfsPerFrame
        self.perfsPerCount = perfsPerCount

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class PreviewImage:
    """Deprecated preview placeholder."""


class LevelMode(Enumerated):
    ONE_LEVEL = 0
    MIPMAP_LEVELS = 1
    RIPMAP_LEVELS = 2
    names = ["ONE_LEVEL", "MIPMAP_LEVELS", "RIPMAP_LEVELS"]


class LevelRoundingMode(Enumerated):
    ROUND_DOWN = 0
    ROUND_UP = 1
    names = ["ROUND_DOWN", "ROUND_UP"]


class TileDescription:
    def __init__(self, xs=32, ys=32, m=None, r=None):
        self.xSize = xs
        self.ySize = ys
        self.mode = m if m is not None else LevelMode(LevelMode.ONE_LEVEL)
        self.roundingMode = r if r is not None else LevelRoundingMode(LevelRoundingMode.ROUND_DOWN)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
