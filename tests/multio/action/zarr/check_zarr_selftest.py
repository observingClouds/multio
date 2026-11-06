#!/usr/bin/env python3
"""Check the stores written by multio_tests_action_zarr (TestZarr.cc) with zarr-python 3 and xarray.

usage: check_zarr_selftest.py <selftest dir>     (the test's $MULTIO_ZARR_SELFTEST_DIR)

Run with a zarr 3 environment, e.g.
  env -u PYTHONPATH /ec/res4/hpcperm/dnk9255/venvs/zarr3/bin/python check_zarr_selftest.py <dir>
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import xarray as xr
import zarr

assert zarr.__version__.startswith("3"), f"zarr 3 required, got {zarr.__version__} from {zarr.__file__}"

NX, NY = 5, 4
NPTS = NX * NY
STEPS = [0, 48, 96]
TIMESTEP = 75
FRT = np.datetime64("2025-02-09T00:00:00")

# name, id, axis levels (None: 2-D), levels sent, first time index
VARS = [
    ("S_TEMPERATURE", 1, "level_S", [3, 1, 2], 0, 130),
    ("KB_ISOT_ALTIT", 2, "level_K", [2, 1], 0, 0),
    ("KT_ISOT_ALTIT", 3, "level_K", [1], 0, 0),
    ("SURFTEMPERATURE", 4, None, [0], 0, 0),
    ("CLSTEMPERATURE", 5, None, [0], 0, 0),
    ("S_HUMI.SPECIFI", 6, "level_S", [1, 3], 1, 0),
    ("S_WIND.U.PHYS", 7, "level_S", [2], 0, 0),
]
AXES = {"level_S": [1, 2, 3], "level_K": [1, 2]}
PREFIX = {"S_TEMPERATURE": ("S", "TEMPERATURE"), "KB_ISOT_ALTIT": ("KB", "ISOT_ALTIT"),
          "KT_ISOT_ALTIT": ("KT", "ISOT_ALTIT"), "SURFTEMPERATURE": ("SURF", "TEMPERATURE"),
          "CLSTEMPERATURE": ("CLS", "TEMPERATURE"), "S_HUMI.SPECIFI": ("S", "HUMI.SPECIFI"),
          "S_WIND.U.PHYS": ("S", "WIND.U.PHYS")}

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL:", msg)


def value(vid, step, level, idx):
    if vid == 4 and idx == 7:
        return np.nan
    return vid * 10000.0 + step * 100.0 + level + idx * 0.001


def expected(name):
    _, vid, axis, levels, first, _ = next(v for v in VARS if v[0] == name)
    idx = np.arange(NPTS)
    if axis is None:
        out = np.full((len(STEPS), NY, NX), np.nan, dtype=np.float32)
        for t, s in enumerate(STEPS):
            if t >= first:
                out[t] = np.array([value(vid, s, 0, i) for i in idx]).astype(np.float32).reshape(NY, NX)
        return out
    ax = AXES[axis]
    out = np.full((len(STEPS), len(ax), NY, NX), np.nan, dtype=np.float32)
    for t, s in enumerate(STEPS):
        if t < first:
            continue
        for lev in levels:
            out[t, ax.index(lev)] = np.array([value(vid, s, lev, i) for i in idx]).astype(np.float32).reshape(NY, NX)
    return out


def bit_equal(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def check_zarr(path, label, consolidated):
    g = zarr.open_group(path, mode="r", use_consolidated=consolidated)
    names = set(g.array_keys())
    want = {v[0] for v in VARS} | {"x", "y", "latitude", "longitude", "crs", "time", "level_S", "level_K"}
    check(names == want, f"{label} consolidated={consolidated}: arrays {sorted(names)} != {sorted(want)}")
    for name, vid, axis, levels, first, grib in VARS:
        a = g[name]
        e = expected(name)
        check(a.dtype == np.float32, f"{label}: {name} dtype {a.dtype}")
        check(a.shape == e.shape, f"{label} consolidated={consolidated}: {name} shape {a.shape} != {e.shape}")
        if a.shape == e.shape:
            check(bit_equal(a[...], e), f"{label} consolidated={consolidated}: {name} values differ")
        attrs = dict(a.attrs)
        pref, suff = PREFIX[name]
        check(attrs.get("fa_prefix") == pref and attrs.get("fa_suffix") == suff
              and attrs.get("long_name") == f"{pref}*{suff}", f"{label}: {name} attrs {attrs}")
        check(attrs.get("grid_mapping") == "crs" and attrs.get("coordinates") == "latitude longitude",
              f"{label}: {name} grid_mapping/coordinates {attrs}")
        check(attrs.get("grib_code", 0) == grib, f"{label}: {name} grib_code {attrs.get('grib_code')}")
        dims = ["time"] + ([axis] if axis else []) + ["y", "x"]
        check(list(a.metadata.dimension_names) == dims, f"{label}: {name} dims {a.metadata.dimension_names}")
        check(np.isnan(a.metadata.fill_value), f"{label}: {name} fill {a.metadata.fill_value}")
    check(np.array_equal(g["time"][...], np.array([s * TIMESTEP for s in STEPS], dtype=np.float64)),
          f"{label}: time {g['time'][...]}")
    check(g["time"].attrs.get("units") == "seconds since 2025-02-09T00:00:00", f"{label}: time units")
    for ax, lev in AXES.items():
        check(np.array_equal(g[ax][...], np.array(lev, dtype=np.float64)), f"{label}: {ax} {g[ax][...]}")
    check(g["level_S"].attrs.get("long_name") == "model (hybrid) level, 1 = top", f"{label}: level_S attrs")
    check(g["level_K"].attrs.get("long_name") == "level of FA prefix K", f"{label}: level_K attrs")
    # static arrays copied from coordinates-from
    check(np.array_equal(g["x"][...], 2500.0 * np.arange(NX)), f"{label}: x")
    check(g["latitude"].shape == (NY, NX), f"{label}: latitude shape")
    check(g.attrs.get("dx_m") == 2500.0 and g.attrs.get("forecast_reference_time") == "2025-02-09T00:00:00",
          f"{label}: root attrs {dict(g.attrs)}")
    return g


def check_xarray(path, label, consolidated):
    ds = xr.open_zarr(path, consolidated=consolidated, zarr_format=3)
    want_t = FRT + np.array([s * TIMESTEP for s in STEPS], dtype="timedelta64[s]")
    check(np.array_equal(ds.time.values, want_t.astype(ds.time.values.dtype)), f"{label} xr: time {ds.time.values}")
    check(ds["S_TEMPERATURE"].dims == ("time", "level_S", "y", "x"), f"{label} xr: dims {ds['S_TEMPERATURE'].dims}")
    check(ds["KT_ISOT_ALTIT"].dims == ("time", "level_K", "y", "x"), f"{label} xr: KT dims")
    check(list(ds.level_S.values) == [1.0, 2.0, 3.0], f"{label} xr: level_S")
    for name, *_ in VARS:
        check(bit_equal(ds[name].values, expected(name)), f"{label} xr consolidated={consolidated}: {name} values")
    check("latitude" in ds.coords and "longitude" in ds.coords, f"{label} xr: lat/lon coords {list(ds.coords)}")
    ds.close()


def count_files(path):
    return sum(len(f) for _, _, f in os.walk(path))


def check_shards_bytes(path):
    """Rewrite the 3-D arrays with zarr-python into a copy of the store; the shard files must be byte-identical."""
    tmp = tempfile.mkdtemp(prefix="zarr-shard-check-")
    try:
        cp = os.path.join(tmp, "copy.zarr")
        shutil.copytree(path, cp)
        for name, _, axis, *_ in VARS:
            if axis is None:
                continue
            shutil.rmtree(os.path.join(cp, name, "c"))
            a = zarr.open_array(os.path.join(cp, name), mode="r+")
            a[...] = expected(name)
            for t in range(len(STEPS)):
                rel = os.path.join(name, "c", str(t), "0", "0", "0")
                ours = os.path.join(path, rel)
                theirs = os.path.join(cp, rel)
                if not os.path.exists(theirs):
                    check(not os.path.exists(ours), f"shard {rel}: zarr-python writes none, we wrote one")
                    continue
                with open(ours, "rb") as f1, open(theirs, "rb") as f2:
                    b1, b2 = f1.read(), f2.read()
                check(b1 == b2, f"shard {rel}: {len(b1)} bytes differ from zarr-python's {len(b2)} bytes")
    finally:
        shutil.rmtree(tmp)


# --- FullPos input (TestZarr.cc runFullPos): full C+I+E payload cropped to the NX x NY window, FullPos
# article names / raw CPREF, flushes at steps 12 and 36 without fields
FP_STEPS = [0, 24]
SENTINEL = np.float32(-9999.0)
# array id, array name, level axis (None: 2-D), level of the field (as normalised), fa_prefix, fa_suffix
FP_FIELDS = [
    (1, "P_TEMPERATURE", "level_P", 85000, "P", "TEMPERATURE"),        # P / TEMPERATURE / 85000
    (1, "P_TEMPERATURE", "level_P", 50000, "P", "TEMPERATURE"),
    (1, "P_TEMPERATURE", "level_P", 100500, "P", "TEMPERATURE"),       # P100500 / TEMPERATURE / 100500
    (1, "P_TEMPERATURE", "level_P", 70000, "P", "TEMPERATURE"),        # P, no levelist, P70000TEMPERATURE
    (2, "KT_ISOT_ALTIT", "level_K", 273, "KT", "ISOT_ALTIT"),          # 'KT  ' / ISOT_ALTIT / 273
    (2, "KT_ISOT_ALTIT", "level_K", 276, "KT", "ISOT_ALTIT"),          # name only KT276ISOT_ALTIT
    (3, "H_TEMPERATURE", "level_H", 15, "H", "TEMPERATURE"),
    (3, "H_TEMPERATURE", "level_H", 500, "H", "TEMPERATURE"),          # name only H00500TEMPERATURE
    (4, "S_HUMI.SPECIFI", "level_S", 90, "S", "HUMI.SPECIFI"),         # S090 / HUMI.SPECIFI / 0
    (5, "S_TEMPERATURE", "level_S", 90, "S", "TEMPERATURE"),           # name only S090TEMPERATURE
    (6, "H_WIND.U.PHYS", "level_H", 500, "H", "WIND.U.PHYS"),          # H005 / 00WIND.U.PHYS / 0
    (7, "C_METEOSAT_09", "level_C", 6, "C", "_METEOSAT_09"),
    (7, "C_METEOSAT_09", "level_C", 2, "C", "_METEOSAT_09"),           # name only C002_METEOSAT_09
    (8, "CLSTEMPERATURE", None, 0, "CLST", "EMPERATURE"),              # CLST / EMPERATURE
    (9, "SURFTEMPERATURE", None, 0, "SURF", "TEMPERATURE"),            # name only
    (10, "SURFRAYT_SOLA_DE", None, 0, "SURF", "RAYT SOLA DE"),
]
FP_AXES = {"level_P": [50000, 70000, 85000, 100500], "level_K": [273, 276], "level_H": [15, 500],
           "level_S": [90], "level_C": [2, 6]}


def fp_value(vid, step, level, idx):
    if vid == 9 and idx == 3:
        return np.nan
    return vid * 1.0e6 + step * 1000.0 + level * 0.01 + idx * 0.001


def fp_expected(name):
    fields = [f for f in FP_FIELDS if f[1] == name]
    axis = fields[0][2]
    idx = np.arange(NPTS)
    shape = (len(FP_STEPS), NY, NX) if axis is None else (len(FP_STEPS), len(FP_AXES[axis]), NY, NX)
    out = np.full(shape, np.nan, dtype=np.float32)
    for t, s in enumerate(FP_STEPS):
        for vid, _, ax, lev, _, _ in fields:
            plane = np.array([fp_value(vid, s, lev, i) for i in idx]).astype(np.float32).reshape(NY, NX)
            if ax is None:
                out[t] = plane
            else:
                out[t, FP_AXES[ax].index(lev)] = plane
    return out


def check_fullpos(path, label, consolidated):
    g = zarr.open_group(path, mode="r", use_consolidated=consolidated)
    names = set(g.array_keys())
    want = {f[1] for f in FP_FIELDS} | set(FP_AXES) | {"x", "y", "latitude", "longitude", "crs", "time"}
    check(names == want, f"{label} consolidated={consolidated}: arrays {sorted(names)} != {sorted(want)}")
    for name in sorted({f[1] for f in FP_FIELDS}):
        if name not in names:
            continue
        f = next(f for f in FP_FIELDS if f[1] == name)
        a = g[name]
        e = fp_expected(name)
        check(a.shape == e.shape, f"{label}: {name} shape {a.shape} != {e.shape}")
        if a.shape == e.shape:
            v = a[...]
            check(bit_equal(v, e), f"{label} consolidated={consolidated}: {name} values differ")
            check(not np.any(v == SENTINEL), f"{label}: {name} contains E-zone sentinel values")
        dims = ["time"] + ([f[2]] if f[2] else []) + ["y", "x"]
        check(list(a.metadata.dimension_names) == dims, f"{label}: {name} dims {a.metadata.dimension_names}")
        attrs = dict(a.attrs)
        check(attrs.get("fa_prefix") == f[4] and attrs.get("fa_suffix") == f[5]
              and attrs.get("long_name") == f"{f[4]}*{f[5]}", f"{label}: {name} attrs {attrs}")
    for ax, lev in FP_AXES.items():
        if ax in names:
            check(np.array_equal(g[ax][...], np.array(lev, dtype=np.float64)), f"{label}: {ax} {g[ax][...]}")
    check(np.array_equal(g["time"][...], np.array([s * TIMESTEP for s in FP_STEPS], dtype=np.float64)),
          f"{label}: time {g['time'][...]} (steps without fields must not add a time step)")
    ds = xr.open_zarr(path, consolidated=consolidated, zarr_format=3)
    check(ds.sizes["time"] == len(FP_STEPS), f"{label} xr: time size {ds.sizes['time']}")
    check(ds["P_TEMPERATURE"].dims == ("time", "level_P", "y", "x"), f"{label} xr: P dims")
    check(list(ds.level_P.values) == FP_AXES["level_P"], f"{label} xr: level_P {ds.level_P.values}")
    for name in {f[1] for f in FP_FIELDS}:
        check(bit_equal(ds[name].values, fp_expected(name)), f"{label} xr consolidated={consolidated}: {name}")
    ds.close()


def check_payload_errors(root):
    path = os.path.join(root, "payload-errors", "SIZEDOM.zarr")
    g = zarr.open_group(path, mode="r")
    check(set(g.array_keys()) == {"SURFTEMPERATURE", "time"}, f"payload-errors: arrays {sorted(g.array_keys())}")
    check(np.array_equal(g["time"][...], np.array([0.0, 150.0])), f"payload-errors: time {g['time'][...]}")
    check(np.array_equal(g["SURFTEMPERATURE"][...], np.ones((2, NY, NX), dtype=np.float32)),
          "payload-errors: SURFTEMPERATURE values")
    check(not os.path.exists(os.path.join(root, "payload-errors", "OTHERDOM.zarr")), "payload-errors: OTHERDOM")


# --- static arrays sent by the client (TestZarr.cc "static arrays ..."): same formulas as staticValues()
NHYI = 4
STATIC_NAMES = ["x", "y", "latitude", "longitude", "crs", "hyai", "hybi"]
GROUP_ATTRS = {"Conventions": "CF-1.8", "title": "IAL output, domain STAT", "source": "IAL MultIO client (prototype)",
               "forecast_reference_time": "2025-02-09T00:00:00", "nx_ci": 5, "ny_ci": 4, "dx_m": 2500.0}


def static_values(name):
    if name == "x":
        return np.array([-5000.0 + 2500.0 * i for i in range(NX)])
    if name == "y":
        return np.array([-3750.0 + 2500.0 * j for j in range(NY)])
    if name == "latitude":
        return np.array([[51.0 + 0.0225 * j + 1.0e-4 * i for i in range(NX)] for j in range(NY)])
    if name == "longitude":
        return np.array([[4.9 + 0.0361 * i - 2.0e-4 * j for i in range(NX)] for j in range(NY)])
    if name == "crs":
        return np.int32(0)
    if name == "hyai":
        return np.array([1000.0 * k * (NHYI - 1 - k) for k in range(NHYI)])
    return np.array([k / (NHYI - 1) for k in range(NHYI)])


def check_statics(path, label, names, layout_vars=True):
    for consolidated in (True, False):
        g = zarr.open_group(path, mode="r", use_consolidated=consolidated)
        keys = set(g.array_keys())
        for name in names:
            check(name in keys, f"{label} consolidated={consolidated}: static array {name} missing")
            if name not in keys:
                continue
            a = g[name]
            e = static_values(name)
            if name == "crs":
                check(a.shape == () and a.dtype == np.int32 and a[...] == 0,
                      f"{label}: crs shape {a.shape} dtype {a.dtype}")
                check(a.attrs.get("grid_mapping_name") == "lambert_conformal_conic"
                      and a.attrs.get("earth_radius") == 6371229.0, f"{label}: crs attrs {dict(a.attrs)}")
                check(a.metadata.dimension_names is None, f"{label}: crs dims {a.metadata.dimension_names}")
                check(a.metadata.fill_value == 0, f"{label}: crs fill {a.metadata.fill_value}")
                continue
            check(a.dtype == np.float64 and a.shape == e.shape, f"{label}: {name} {a.dtype} {a.shape}")
            if a.shape == e.shape:
                check(np.array_equal(a[...].view(np.uint64), e.view(np.uint64)), f"{label}: {name} values")
            dims = {"x": ["x"], "y": ["y"], "latitude": ["y", "x"], "longitude": ["y", "x"]}.get(name, ["nhyi"])
            check(list(a.metadata.dimension_names) == dims, f"{label}: {name} dims {a.metadata.dimension_names}")
            check(a.chunks == a.shape, f"{label}: {name} chunks {a.chunks}")
        check(dict(g.attrs) == GROUP_ATTRS, f"{label} consolidated={consolidated}: root attrs {dict(g.attrs)}")
        if layout_vars:
            for name, *_ in VARS:
                attrs = dict(g[name].attrs)
                check(attrs.get("grid_mapping") == "crs" and attrs.get("coordinates") == "latitude longitude",
                      f"{label} consolidated={consolidated}: {name} grid_mapping/coordinates {attrs}")
                check(bit_equal(g[name][...], expected(name)), f"{label} consolidated={consolidated}: {name} values")
        ds = xr.open_zarr(path, consolidated=consolidated, zarr_format=3)
        check("latitude" in ds.coords and "longitude" in ds.coords, f"{label} xr: coords {list(ds.coords)}")
        check(ds.latitude.dims == ("y", "x"), f"{label} xr: latitude dims {ds.latitude.dims}")
        if "x" in names:
            check(np.array_equal(ds.x.values, static_values("x")), f"{label} xr: x {ds.x.values}")
        if "hyai" in names:
            check(ds.hyai.dims == ("nhyi",), f"{label} xr: hyai dims {ds.hyai.dims}")
        check(ds.sizes["time"] == len(STEPS), f"{label} xr: time {ds.sizes['time']}")
        ds.close()


def check_static_stores(root):
    check_statics(os.path.join(root, "static-before", "STATDOM.zarr"), "static-before", STATIC_NAMES)
    check_statics(os.path.join(root, "static-after", "STATLATE.zarr"), "static-after", STATIC_NAMES)
    # replacement of coordinates-from arrays: x and latitude sent, y/longitude/crs copied
    path = os.path.join(root, "static-replace", "STATCOPY.zarr")
    check_statics(path, "static-replace", ["x", "latitude"])
    g = zarr.open_group(path, mode="r")
    check(np.array_equal(g["y"][...], 2500.0 * np.arange(NY)), "static-replace: copied y")
    check(np.array_equal(g["longitude"][...][0], 4.0 + 0.1 * np.arange(NX)), "static-replace: copied longitude")
    check("hyai" not in set(g.array_keys()), "static-replace: hyai")
    # the shard-levels store with late statics equals the chunk-per-level store with early ones
    g1 = zarr.open_group(os.path.join(root, "static-before", "STATDOM.zarr"), mode="r")
    g2 = zarr.open_group(os.path.join(root, "static-after", "STATLATE.zarr"), mode="r")
    for name in STATIC_NAMES:
        check(np.array_equal(np.asarray(g1[name][...]), np.asarray(g2[name][...])), f"static before/after: {name}")
    errs = zarr.open_group(os.path.join(root, "static-errors", "STATERR.zarr"), mode="r")
    check({"x", "crs"} <= set(errs.array_keys()) and "latitude" not in set(errs.array_keys()),
          f"static-errors: arrays {sorted(errs.array_keys())}")
    ref = os.path.join(root, "static-reference", "REFDOM.zarr")
    if os.path.exists(ref):
        g = zarr.open_group(ref, mode="r")
        check(set(g.array_keys()) == set(STATIC_NAMES), f"static-reference: arrays {sorted(g.array_keys())}")
        check(g["latitude"].shape == (g["y"].shape[0], g["x"].shape[0]), "static-reference: latitude shape")
        print(f"static-reference: {sorted(g.array_keys())}, latitude {g['latitude'].shape}")


# --- zstd compression (TestZarr.cc runZstd: the synthetic run with compression zstd level 5)
ZSTD_STORES = {"zstd-chunk-per-level-t0": "chunk-per-level", "zstd-chunk-per-level-t4": "chunk-per-level",
               "zstd-chunk-per-level-t1q1": "chunk-per-level", "zstd-shard-levels-t0": "shard-levels",
               "zstd-shard-levels-t4": "shard-levels"}


def codec_names(codecs):
    return [type(c).__name__ for c in codecs]


def check_zstd_codecs(g, label, sharded):
    for name, _, axis, *_ in VARS:
        codecs = g[name].metadata.codecs
        if sharded and axis is not None:
            check(codec_names(codecs) == ["ShardingCodec"], f"{label}: {name} codecs {codec_names(codecs)}")
            inner, index = codecs[0].codecs, codecs[0].index_codecs
            check(codec_names(index) == ["BytesCodec"], f"{label}: {name} index codecs {codec_names(index)}")
        else:
            inner = codecs
        check(codec_names(inner) == ["BytesCodec", "ZstdCodec"], f"{label}: {name} codecs {codec_names(inner)}")
        z = [c for c in inner if type(c).__name__ == "ZstdCodec"]
        check(bool(z) and z[0].level == 5 and z[0].checksum is False, f"{label}: {name} zstd {z}")
    for name in ["time", "level_S", "x", "latitude"]:
        check("ZstdCodec" not in codec_names(g[name].metadata.codecs), f"{label}: {name} compressed")


def check_zstd_rewrite(path, label):
    """Rewrite the 3-D arrays with zarr-python (same codecs) into a copy; its shards decode to the same values."""
    tmp = tempfile.mkdtemp(prefix="zarr-zstd-check-")
    try:
        cp = os.path.join(tmp, "copy.zarr")
        shutil.copytree(path, cp)
        same_bytes = 0
        total = 0
        for name, _, axis, *_ in VARS:
            if axis is None:
                continue
            shutil.rmtree(os.path.join(cp, name, "c"))
            a = zarr.open_array(os.path.join(cp, name), mode="r+")
            a[...] = expected(name)
            b = zarr.open_array(os.path.join(path, name), mode="r")
            check(bit_equal(a[...], b[...]), f"{label}: {name} zarr-python rewrite differs")
            for t in range(len(STEPS)):
                rel = os.path.join(name, "c", str(t), "0", "0", "0")
                if os.path.exists(os.path.join(cp, rel)) and os.path.exists(os.path.join(path, rel)):
                    total += 1
                    with open(os.path.join(cp, rel), "rb") as f1, open(os.path.join(path, rel), "rb") as f2:
                        same_bytes += f1.read() == f2.read()
        print(f"{label}: {same_bytes} of {total} shards byte-identical to zarr-python's (informational)")
    finally:
        shutil.rmtree(tmp)


def check_zstd_stores(root, plain):
    for sub, layout in ZSTD_STORES.items():
        path = os.path.join(root, sub, "TESTDOM.zarr")
        for consolidated in (True, False):
            g = check_zarr(path, sub, consolidated)
            check_xarray(path, sub, consolidated)
        check_zstd_codecs(g, sub, layout == "shard-levels")
        gp = zarr.open_group(plain[layout], mode="r")
        for name, *_ in VARS:
            check(bit_equal(g[name][...], gp[name][...]), f"{sub}: {name} differs from the uncompressed store")
        print(f"{sub}: {count_files(path)} files, {du(path)} bytes (uncompressed {du(plain[layout])})")
    check_zstd_rewrite(os.path.join(root, "zstd-shard-levels-t4", "TESTDOM.zarr"), "zstd-shard-levels-t4")
    # ZarrStore under load (worker pool vs inline): readable, same values
    for ln in ["chunk", "shard"]:
        g0 = zarr.open_group(os.path.join(root, f"zstd-load-{ln}-t0", "LOAD.zarr"), mode="r")
        g3 = zarr.open_group(os.path.join(root, f"zstd-load-{ln}-t3", "LOAD.zarr"), mode="r")
        names = sorted(g0.array_keys())
        check(names == sorted(g3.array_keys()) and len(names) == 8, f"zstd-load-{ln}: arrays {names}")
        for n in names:
            check(np.array_equal(g0[n][...], g3[n][...], equal_nan=True), f"zstd-load-{ln}: {n} differs")
        v = g3["S_VAR0"][...]
        check(v.shape == (3, 12, 173, 211) and np.isnan(v[:, :, 0, 3]).all() and np.isfinite(v).sum() == v.size - 36,
              f"zstd-load-{ln}: S_VAR0 {v.shape}, NaN count {np.isnan(v).sum()}")


# --- quantization before zstd (TestZarr.cc QUANT_RULES, runQuant)
QUANT_STORES = {"quant-chunk-per-level-t0": "chunk-per-level", "quant-chunk-per-level-t4": "chunk-per-level",
                "quant-shard-levels-t0": "shard-levels", "quant-shard-levels-t4": "shard-levels"}
# name -> (container, attributes, error bound of a field e (float64, original float32 values))
QUANT = {
    "S_TEMPERATURE": ("quantization_bitround", {"quantization_nsb": 8}, lambda e: np.abs(e) * 2.0 ** -9),
    "KB_ISOT_ALTIT": ("quantization_absolute", {"quantization_step": 0.25, "quantization_maximum_absolute_error": 0.125},
                      lambda e: np.full(e.shape, 0.125)),
    "KT_ISOT_ALTIT": ("quantization_absolute", {"quantization_step": 0.25}, lambda e: np.full(e.shape, 0.125)),
    "S_HUMI.SPECIFI": ("quantization_range_bits", {"quantization_range_bits": 1}, None),
    "SURFTEMPERATURE": ("quantization_bitround", {"quantization_nsb": 4}, lambda e: np.abs(e) * 2.0 ** -5),
    "CLSTEMPERATURE": (None, {}, lambda e: np.zeros(e.shape)),
    "S_WIND.U.PHYS": ("quantization_bitround", {"quantization_nsb": 12}, lambda e: np.abs(e) * 2.0 ** -13),
}
ALGORITHMS = {"quantization_bitround": "bitround", "quantization_absolute": "absolute_power_of_two",
              "quantization_range_bits": "range_bits_power_of_two"}


def pow2floor(x):
    return 2.0 ** np.floor(np.log2(x))


def range_bits_bound(x, n):
    """D/2 per field (last two axes), D = largest power of two <= (max - min) / 2^n of the field's finite values"""
    b = np.zeros(x.shape)
    for idx in np.ndindex(x.shape[:-2]):
        f = x[idx]
        ok = np.isfinite(f)
        if ok.any() and np.ptp(f[ok]) > 0:
            b[idx] = pow2floor(np.ptp(f[ok].astype(np.float64)) / 2.0 ** n) / 2
    return b


def check_quant_values(label, name, q, x, bound):
    nan_x, nan_q = np.isnan(x), np.isnan(q)
    check(np.array_equal(nan_x, nan_q), f"{label}: {name} NaN mask changed")
    ok = ~nan_x
    check(np.isfinite(q[ok]).all(), f"{label}: {name} non-finite values")
    err = np.abs(q.astype(np.float64) - x.astype(np.float64))
    check((err[ok] <= bound[ok]).all(), f"{label}: {name} error {err[ok].max()} above bound")
    return int((err[ok] > 0).sum())


def check_quant_stores(root):
    for sub, layout in QUANT_STORES.items():
        path = os.path.join(root, sub, "TESTDOM.zarr")
        for consolidated in (True, False):
            g = zarr.open_group(path, mode="r", use_consolidated=consolidated)
            names = set(g.array_keys())
            want = {v[0] for v in VARS} | {"x", "y", "latitude", "longitude", "crs", "time", "level_S", "level_K"} \
                | set(ALGORITHMS)
            check(names == want, f"{sub} consolidated={consolidated}: arrays {sorted(names ^ want)}")
            changed = 0
            for name, (container, attrs, bound) in QUANT.items():
                a = g[name]
                x = expected(name)
                q = a[...]
                at = dict(a.attrs)
                check(at.get("quantization") == container, f"{sub}: {name} quantization {at.get('quantization')}")
                for k, v in attrs.items():
                    check(at.get(k) == v, f"{sub}: {name} {k} = {at.get(k)}, expected {v}")
                b = range_bits_bound(x, 1) if bound is None else bound(x.astype(np.float64))
                changed += check_quant_values(sub, name, q, x, b)
                if container is None:
                    check(bit_equal(q, x), f"{sub}: {name} (lossless rule) differs")
            check(changed > 0, f"{sub}: nothing quantized")
            for c, alg in ALGORITHMS.items():
                check(g[c].shape == () and g[c].attrs.get("algorithm") == alg, f"{sub}: container {c} {dict(g[c].attrs)}")
        check_zstd_codecs(g, sub, layout == "shard-levels")
        ds = xr.open_zarr(path, consolidated=True, zarr_format=3)
        check(ds["S_TEMPERATURE"].attrs.get("quantization_nsb") == 8, f"{sub} xr: attrs {ds['S_TEMPERATURE'].attrs}")
        ds.close()
    for layout in ["chunk-per-level", "shard-levels"]:
        g0 = zarr.open_group(os.path.join(root, f"quant-{layout}-t0", "TESTDOM.zarr"), mode="r")
        g4 = zarr.open_group(os.path.join(root, f"quant-{layout}-t4", "TESTDOM.zarr"), mode="r")
        for name in QUANT:
            check(bit_equal(g0[name][...], g4[name][...]), f"quant-{layout}: threads 0 vs 4 differ for {name}")
        gp = zarr.open_group(os.path.join(root, f"quant-none-{layout}-plain", "TESTDOM.zarr"), mode="r")
        for sub in [f"quant-none-{layout}", f"quant-none-{layout}-default"]:
            gn = zarr.open_group(os.path.join(root, sub, "TESTDOM.zarr"), mode="r")
            for name, *_ in VARS:
                check(bit_equal(gn[name][...], gp[name][...]) and "quantization" not in dict(gn[name].attrs),
                      f"{sub}: {name} differs from plain zstd")
    # ZarrStore under load: range-bits 10 (S_VAR0/1), absolute 2^-6 (S_VAR2), bitround 9 (default); the lossless
    # zstd load store has the same fields
    for ln in ["chunk", "shard"]:
        ref = zarr.open_group(os.path.join(root, f"zstd-load-{ln}-t0", "LOAD.zarr"), mode="r")
        g0 = zarr.open_group(os.path.join(root, f"quant-load-{ln}-t0", "LOAD.zarr"), mode="r")
        g3 = zarr.open_group(os.path.join(root, f"quant-load-{ln}-t3", "LOAD.zarr"), mode="r")
        changed = 0
        for n in sorted(ref.array_keys()):
            if n in ("time",) or n.startswith("level_"):
                continue
            x, q = ref[n][...], g3[n][...]
            check(bit_equal(g0[n][...], q), f"quant-load-{ln}: {n} threads 0 vs 3 differ")
            if n in ("S_VAR0", "S_VAR1"):
                b = range_bits_bound(x, 10)
                check(g3[n].attrs.get("quantization_range_bits") == 10, f"quant-load-{ln}: {n} attrs")
            elif n == "S_VAR2":
                b = np.full(x.shape, 2.0 ** -7)
            else:
                b = np.abs(x.astype(np.float64)) * 2.0 ** -10
                check(g3[n].attrs.get("quantization_nsb") == 9, f"quant-load-{ln}: {n} attrs")
            changed += check_quant_values(f"quant-load-{ln}", n, q, x, b)
        check(changed > 0, f"quant-load-{ln}: nothing quantized")
        print(f"quant-load-{ln}: {du(os.path.join(root, f'quant-load-{ln}-t3'))} bytes "
              f"(lossless zstd {du(os.path.join(root, f'zstd-load-{ln}-t3'))})")


# ---------------------------------------------------------------------------------------------------------------------
# Spatial tiles (TestZarr.cc "tiles: ..."): 37 x 29 window, tiles of 8 x 8 (5 x 4, edge tiles 5 wide / high)

TNX, TNY, TT = 37, 29, 8
T_FIELD_STEPS = [0, 48, 96]
# name, id, axis, levels sent, first time index
TVARS = [
    ("S_TEMPERATURE", 1, "level_S", [3, 1, 2], 0),
    ("KB_ISOT_ALTIT", 2, "level_K", [2, 1], 0),
    ("KT_ISOT_ALTIT", 3, "level_K", [1], 0),
    ("SURFTEMPERATURE", 4, None, [0], 0),
    ("CLSTEMPERATURE", 5, None, [0], 0),
    ("S_HUMI.SPECIFI", 6, "level_S", [1, 3], 1),
]
TAXES = {"level_S": [1, 2, 3], "level_K": [1, 2]}


def tvalue_grid(vid, step, level):
    y, x = np.mgrid[0:TNY, 0:TNX].astype(np.float64)
    v = vid * 1000.0 + step * 10.0 + level * 0.5 + x * 0.01 + y * 0.37
    if vid == 4:
        v[(x + 3 * y) % 11 == 0] = np.nan
    return v.astype(np.float32)


def texpected(name):
    _, vid, axis, levels, first = next(v for v in TVARS if v[0] == name)
    if axis is None:
        out = np.full((len(T_FIELD_STEPS), TNY, TNX), np.nan, dtype=np.float32)
        for t, s in enumerate(T_FIELD_STEPS):
            if t >= first:
                out[t] = tvalue_grid(vid, s, 0)
        return out
    ax = TAXES[axis]
    out = np.full((len(T_FIELD_STEPS), len(ax), TNY, TNX), np.nan, dtype=np.float32)
    for t, s in enumerate(T_FIELD_STEPS):
        if t >= first:
            for lev in levels:
                out[t, ax.index(lev)] = tvalue_grid(vid, s, lev)
    return out


def nan_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    both = np.isnan(a) & np.isnan(b) if a.dtype.kind == "f" else np.zeros(a.shape, bool)
    return bool(np.all((a == b) | both))


def check_tile_pair(ref, tiled, label, bitround):
    gr = zarr.open_group(ref, mode="r")
    gt = zarr.open_group(tiled, mode="r")
    names_r = set(gr.array_keys())
    names_t = set(gt.array_keys())
    check(names_r == names_t, f"{label}: arrays {sorted(names_t)} != {sorted(names_r)}")
    check(dict(gr.attrs) == dict(gt.attrs), f"{label}: root attributes")
    for n in sorted(names_r & names_t):
        a, b = gr[n], gt[n]
        check(a.shape == b.shape and a.dtype == b.dtype, f"{label}: {n} shape/dtype {b.shape} {b.dtype}")
        check(dict(a.attrs) == dict(b.attrs), f"{label}: {n} attributes")
        check(a.metadata.dimension_names == b.metadata.dimension_names, f"{label}: {n} dimension names")
        check(nan_equal(a[...], b[...]), f"{label}: {n} values differ")
        if "time" in (a.metadata.dimension_names or ()) and n != "time":
            tiles = (min(TT, TNY), min(TT, TNX))
            if a.ndim == 4:
                if a.shards is not None:
                    check(b.shards == (1, a.shape[1]) + tiles and b.chunks == (1, 1) + tiles,
                          f"{label}: {n} shards {b.shards} chunks {b.chunks}")
                else:
                    check(b.chunks == (1, 1) + tiles and b.shards is None, f"{label}: {n} chunks {b.chunks}")
            else:
                check(b.chunks == (1,) + tiles, f"{label}: {n} chunks {b.chunks}")
            if n in dict((v[0], v) for v in TVARS):
                want = texpected(n)
                got = a[...]
                if bitround:
                    # 9 kept mantissa bits: relative error <= 2^-10
                    ok = np.isnan(want) == np.isnan(got)
                    fin = ~np.isnan(want)
                    ok[fin] = np.abs(got[fin].astype(np.float64) - want[fin]) <= np.abs(want[fin]) * 2.0 ** -10
                    check(bool(ok.all()), f"{label}: {n} bitround bound")
                else:
                    check(bit_equal(got, want), f"{label}: {n} values of the reference store")
    # xarray reads the tiled store like the untiled one, consolidated or not
    for consolidated in (True, False):
        dr = xr.open_zarr(ref, consolidated=consolidated, zarr_format=3)
        dt = xr.open_zarr(tiled, consolidated=consolidated, zarr_format=3)
        for n in dr.data_vars:
            check(nan_equal(dr[n].values, dt[n].values), f"{label} xr consolidated={consolidated}: {n}")
        check(np.array_equal(dr.time.values, dt.time.values), f"{label} xr: time")


def check_tile_stores(root):
    found = 0
    for layout in ("chunk-per-level", "shard-levels"):
        for variant in ("raw", "zstd", "zstd-bitround"):
            sub = f"{layout}-{variant}"
            ref = os.path.join(root, f"tiles-ref-{sub}", "TILES.zarr")
            if not os.path.exists(os.path.join(ref, "zarr.json")):
                continue
            for kind in ("tiles", "tiles-mw3"):
                tiled = os.path.join(root, f"{kind}-{sub}", "TILES.zarr")
                if kind == "tiles-mw3" and variant == "zstd-bitround":
                    continue
                check(os.path.exists(os.path.join(tiled, "zarr.json")), f"{kind}-{sub}: no store")
                if os.path.exists(os.path.join(tiled, "zarr.json")):
                    check_tile_pair(ref, tiled, f"{kind}-{sub}", variant == "zstd-bitround")
                    found += 1
    check(found >= 4, f"tiles: only {found} tiled stores")
    print(f"tiles: {found} tiled stores read like their untiled references")
    g = zarr.open_group(os.path.join(root, "tiles-flush", "TFLUSH.zarr"), mode="r")
    check(g["S_TEMPERATURE"].shape == (3, 2, TNY, TNX), f"tiles-flush: shape {g['S_TEMPERATURE'].shape}")
    s = g["SURFTEMPERATURE"][...]
    check(np.all(s[:, 0:8, 0:8] == 1.0) and np.all(s[:, 16:24, 24:32] == 14.0) and np.all(s[:, 8:16, 8:16] == 7.0),
          "tiles-flush: tile values")
    check(np.isnan(s[:, 0:8, 8:16]).all(), "tiles-flush: tile without fields not NaN")


def du(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "zarr-selftest"
    print(f"zarr {zarr.__version__} ({zarr.__file__}), xarray {xr.__version__}, numpy {np.__version__}")
    stores = {}
    for sub in ["chunk-per-level", "shard-levels", "no-eos"]:
        path = os.path.join(root, sub, "TESTDOM.zarr")
        stores[sub] = path
        for consolidated in (True, False):
            check_zarr(path, sub, consolidated)
            check_xarray(path, sub, consolidated)
        print(f"{sub}: {count_files(path)} files")

    # shard-levels: same values as chunk-per-level, fewer files, 3-D arrays sharded
    g1 = zarr.open_group(stores["chunk-per-level"], mode="r")
    g2 = zarr.open_group(stores["shard-levels"], mode="r")
    for name, *_ in VARS:
        check(bit_equal(g1[name][...], g2[name][...]), f"shard vs chunk: {name}")
    check(g2["S_TEMPERATURE"].shards == (1, 3, NY, NX) and g2["S_TEMPERATURE"].chunks == (1, 1, NY, NX),
          f"shard-levels: shards {g2['S_TEMPERATURE'].shards} chunks {g2['S_TEMPERATURE'].chunks}")
    check(count_files(stores["shard-levels"]) < count_files(stores["chunk-per-level"]), "shard-levels: file count")
    check_shards_bytes(stores["shard-levels"])

    fp = {}
    for sub in ["fullpos-chunk-per-level", "fullpos-shard-levels"]:
        fp[sub] = os.path.join(root, sub, "PFTESTDOM.zarr")
        for consolidated in (True, False):
            check_fullpos(fp[sub], sub, consolidated)
        check(not os.path.exists(os.path.join(root, sub, "PFNOFIELDS.zarr")), f"{sub}: PFNOFIELDS store created")
        print(f"{sub}: {count_files(fp[sub])} files")
    g1 = zarr.open_group(fp["fullpos-chunk-per-level"], mode="r")
    g2 = zarr.open_group(fp["fullpos-shard-levels"], mode="r")
    for name in {f[1] for f in FP_FIELDS}:
        check(bit_equal(g1[name][...], g2[name][...]), f"fullpos shard vs chunk: {name}")
    check_payload_errors(root)
    check_static_stores(root)
    check_zstd_stores(root, {"chunk-per-level": stores["chunk-per-level"], "shard-levels": stores["shard-levels"]})
    check_quant_stores(root)
    check_tile_stores(root)

    if failures:
        print(f"{len(failures)} FAILURES")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
