"""Generate Slug curves/bands from the SAME bundled CC0 Aileron OTF.

Offline after: python -m pip install -r tools/slug-requirements.txt
Python 3.10+, fonttools 4.60.1. Output uses little-endian f32/i32 buffers.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

import fontTools
from fontTools.pens.basePen import BasePen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT_HASH = "2762f4fc2ebad8323264aea52ffa2260b86c9677493d3ce2dc4f34e5851d2aa2"
ERROR_EM = 1 / 65536
BAND_EPSILON = 1 / 1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0] if value != 0 else 0.0


class Quadratics(BasePen):
    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.curves = []
        self.start = self.current = None

    def _moveTo(self, p):
        self.start = self.current = p

    def _lineTo(self, p):
        if p != self.current:
            # Reference recommendation: duplicate the final endpoint for lines.
            self.curves.append((self.current, p, p))
        self.current = p

    def _qCurveToOne(self, control, end):
        self.curves.append((self.current, control, end))
        self.current = end

    def _curveToOne(self, *_):
        raise ValueError("Cu2QuPen left a cubic curve")

    def _closePath(self):
        self._lineTo(self.start)

    def _endPath(self):
        raise ValueError("Open contour in a filled glyph")


def bounds(curves):
    if not curves:
        return [0.0] * 4
    axes = []
    for axis in (0, 1):
        values = []
        for curve in curves:
            a, b, c = (p[axis] for p in curve)
            values.extend((a, c))
            denominator = a - 2*b + c
            if denominator:
                t = (a-b) / denominator
                if 0 < t < 1:
                    values.append((1-t)**2*a + 2*t*(1-t)*b + t*t*c)
        axes.append((min(values), max(values)))
    return [axes[0][0], axes[1][0], axes[0][1], axes[1][1]]


def main():
    if fontTools.__version__ != "4.60.1":
        raise RuntimeError("Install the pinned tools/slug-requirements.txt")
    font_path = ASSETS / "Aileron-Regular.otf"
    if sha(font_path) != FONT_HASH:
        raise RuntimeError("Unexpected source font checksum")
    atlas = json.loads((ASSETS / "aileron-mtsdf.json").read_text())
    font = TTFont(font_path)
    upem = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    curve_data, band_data, index_data, metadata = [], [], [], []
    for metric in sorted(atlas["glyphs"], key=lambda g: g["unicode"]):
        cp = metric["unicode"]
        pen = Quadratics(glyph_set)
        glyph_set[cmap[cp]].draw(Cu2QuPen(pen, ERROR_EM * upem, all_quadratic=True))
        curves = [tuple((f32(x/upem), f32(-y/upem)) for x, y in curve) for curve in pen.curves]
        # Drop a point curve; keep all other quadratics and contour orientation.
        curves = [q for q in curves if not q[0] == q[1] == q[2]]
        first_curve = len(curve_data) // 8
        for p0, p1, p2 in curves:
            curve_data.extend((*p0, *p1, *p2, 0.0, 0.0))
        box = bounds(curves)
        count = max(1, min(16, math.ceil(math.sqrt(len(curves)))))
        offsets = []
        # Horizontal bands partition Y, vertical bands partition X.
        for axis in (1, 0):
            offsets.append(len(band_data) // 2)
            low, high = box[axis], box[axis+2]
            for band in range(count):
                lo = low + (high-low)*band/count - BAND_EPSILON
                hi = low + (high-low)*(band+1)/count + BAND_EPSILON
                members = [i for i, q in enumerate(curves)
                           if not q[0][axis] == q[1][axis] == q[2][axis]
                           and min(p[axis] for p in q) <= hi
                           and max(p[axis] for p in q) >= lo]
                members.sort(key=lambda i: (-max(p[1-axis] for p in curves[i]), i))
                band_data.extend((len(index_data), len(members)))
                index_data.extend(first_curve + i for i in members)
        sx = count/(box[2]-box[0]) if curves else 0.0
        sy = count/(box[3]-box[1]) if curves else 0.0
        metadata.append({"unicode": cp, "advance": metric["advance"], "bounds": box,
                         "bandTransform": [sx, sy, -box[0]*sx, -box[1]*sy],
                         "bands": [offsets[0], count, offsets[1], count],
                         "firstCurve": first_curve, "curveCount": len(curves)})
    buffers = {"aileron-slug-curves.bin": ("f", curve_data),
               "aileron-slug-bands.bin": ("i", band_data),
               "aileron-slug-indices.bin": ("i", index_data)}
    for name, (kind, values) in buffers.items():
        (ASSETS / name).write_bytes(struct.pack(f"<{len(values)}{kind}", *values))
    data = {"font": "Aileron Regular", "version": "0.102", "license": "CC0-1.0",
            "fontSha256": FONT_HASH, "generator": "FontTools 4.60.1 Cu2QuPen",
            "cubicConversionToleranceEm": ERROR_EM, "coordinateSystem": "top-down em",
            "curveFormat": "two little-endian vec4f32 per quadratic: p0,p1 / p2,0,0",
            "bandFormat": "little-endian vec2i32: index-list offset, count",
            "indexFormat": "little-endian i32 curve index", "bandOverlapEm": BAND_EPSILON,
            "curveCount": len(curve_data)//8, "bandCount": len(band_data)//2,
            "indexCount": len(index_data), "glyphs": metadata,
            "files": {name: {"sha256": sha(ASSETS/name)} for name in buffers}}
    (ASSETS/"aileron-slug.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    subprocess.run(["node", str(ROOT / "tools/generate-font.mjs")], check=True)
    print(f"Generated {len(metadata)} glyphs, {data['curveCount']} quadratics, "
          f"{data['bandCount']} bands, {data['indexCount']} indices")


if __name__ == "__main__":
    main()
