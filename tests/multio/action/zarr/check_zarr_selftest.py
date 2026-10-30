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

    if failures:
        print(f"{len(failures)} FAILURES")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
