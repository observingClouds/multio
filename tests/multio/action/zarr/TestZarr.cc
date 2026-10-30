/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

/// Self-contained test of the zarr action: synthetic fields from two "clients" go through
/// select -> aggregate -> zarr and are written in both layouts. The stores are left on disk
/// (directory $MULTIO_ZARR_SELFTEST_DIR, default ./zarr-selftest) for the python check
/// check_zarr_selftest.py, which reads them with zarr-python 3 and xarray.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "eckit/testing/Test.h"

#include "multio/domain/Mappings.h"
#include "multio/message/Message.h"
#include "multio/message/Metadata.h"
#include "multio/message/Peer.h"

#include "../../MultioTestEnvironment.h"

namespace multio::test::zarr {

namespace fs = std::filesystem;

using multio::message::Message;
using multio::message::Metadata;
using multio::message::Peer;

constexpr std::int64_t NX = 5;
constexpr std::int64_t NY = 4;
constexpr std::int64_t NPTS = NX * NY;
constexpr std::int64_t TIMESTEP = 75;
const std::vector<std::int64_t> STEPS = {0, 48, 96};
const std::string FRT = "2025-02-09T00:00:00";

std::string outRoot() {
    const char* e = std::getenv("MULTIO_ZARR_SELFTEST_DIR");
    return e ? std::string(e) : std::string("zarr-selftest");
}

/// Synthetic field description; the python check uses the same table and formula
struct Var {
    int id;
    std::string levtype;  // empty: legacy FA name only (normalised by the action)
    std::string suffix;
    std::string faName;
    std::vector<std::int64_t> levels;  // empty: 2-D
    std::int64_t grib;
    bool single;            // single precision payload
    std::size_t firstTime;  // first time index the variable is sent at
};

const std::vector<Var> VARS = {
    {1, "S", "TEMPERATURE", "", {3, 1, 2}, 130, false, 0},
    {2, "KB", "ISOT_ALTIT", "", {2, 1}, 0, false, 0},
    {3, "KT", "ISOT_ALTIT", "", {1}, 0, false, 0},
    {4, "SURF", "TEMPERATURE", "", {}, 0, false, 0},
    {5, "CLS", "TEMPERATURE", "", {}, 0, true, 0},
    {6, "S", "HUMI.SPECIFI", "", {1, 3}, 0, false, 1},
    {7, "", "", "S002WIND.U.PHYS", {2}, 0, false, 0},
};

double value(int id, std::int64_t step, std::int64_t level, std::int64_t idx) {
    if (id == 4 && idx == 7) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return id * 10000.0 + step * 100.0 + level + idx * 0.001;
}

const Peer SERVER{"server", 0};
const std::vector<Peer> CLIENTS = {Peer{"client", 0}, Peer{"client", 1}};

/// client c owns the global indices i with i % 2 == c
std::vector<std::int32_t> clientIndices(std::size_t c, std::int64_t npts = NPTS) {
    std::vector<std::int32_t> idx;
    for (std::int32_t i = 0; i < npts; ++i) {
        if (static_cast<std::size_t>(i % 2) == c) {
            idx.push_back(i);
        }
    }
    return idx;
}

void registerDomain(const std::string& domain, std::int64_t npts = NPTS) {
    for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
        auto idx = clientIndices(c, npts);
        Metadata md;
        md.set("name", domain);
        md.set("domain", domain);
        md.set("category", std::string("ial-domain-map"));
        md.set("representation", std::string("unstructured"));
        md.set("misc-globalSize", npts);
        md.set("toAllServers", true);
        Message msg{Message::Header{Message::Tag::Domain, CLIENTS[c], SERVER, std::move(md)},
                    eckit::Buffer(idx.data(), idx.size() * sizeof(std::int32_t))};
        multio::domain::Mappings::instance().add(std::move(msg));
    }
}

Message fieldMessage(const std::string& domain, const Var& v, std::int64_t step, std::int64_t level, std::size_t c) {
    Metadata md;
    md.set("category", std::string("ial-field"));
    md.set("domain", domain);
    md.set("misc-globalSize", NPTS);
    md.set("misc-precision", std::string(v.single ? "single" : "double"));
    if (!v.levtype.empty()) {
        md.set("name", v.levtype + (v.levels.empty() ? "" : std::to_string(level)) + v.suffix);
        md.set("levtype", v.levtype);
        md.set("suffix", v.suffix);
        md.set("levelist", level);
        md.set("is3d", !v.levels.empty());
    }
    else {
        md.set("name", v.faName);  // legacy: no levtype / suffix / levelist / is3d
    }
    md.set("grib-code", v.grib);
    md.set("step", step);
    md.set("timestep", TIMESTEP);
    md.set("forecastReferenceTime", FRT);
    md.set("geometry-nx", NX);
    md.set("geometry-ny", NY);

    auto idx = clientIndices(c);
    if (v.single) {
        std::vector<float> vals;
        for (auto i : idx) {
            vals.push_back(static_cast<float>(value(v.id, step, level, i)));
        }
        return Message{Message::Header{Message::Tag::Field, CLIENTS[c], SERVER, std::move(md)},
                       eckit::Buffer(vals.data(), vals.size() * sizeof(float))};
    }
    std::vector<double> vals;
    for (auto i : idx) {
        vals.push_back(value(v.id, step, level, i));
    }
    return Message{Message::Header{Message::Tag::Field, CLIENTS[c], SERVER, std::move(md)},
                   eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
}

Message flushMessage(const std::string& domain, std::int64_t step, const std::string& kind, std::size_t c) {
    Metadata md;
    md.set("domain", domain);
    md.set("step", step);
    md.set("flushKind", kind);
    return Message{Message::Header{Message::Tag::Flush, CLIENTS[c], SERVER, std::move(md)}};
}

std::string planJson(const std::string& path, const std::string& layout, const std::string& coords) {
    std::ostringstream oss;
    oss << R"({"name": "ial-to-zarr", "actions": [)"
        << R"({"type": "select", "match": [{"category": ["ial-field"]}]},)"
        << R"({"type": "aggregate"},)"
        << R"({"type": "zarr", "path": ")" << path << R"(", "format": 3, "layout": ")" << layout << R"(")";
    if (!coords.empty()) {
        oss << R"(, "coordinates-from": ")" << coords << R"(")";
    }
    oss << "}]}";
    return oss.str();
}

void writeFile(const fs::path& p, const void* data, std::size_t n) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}

void writeText(const fs::path& p, const std::string& s) {
    writeFile(p, s.data(), s.size());
}

std::string readText(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

std::string arrayDoc(const std::string& shape, const std::string& chunk, const std::string& dtype,
                     const std::string& attrs, const std::string& dims, const std::string& fill = "\"NaN\"") {
    return R"({"zarr_format":3,"node_type":"array","shape":)" + shape + R"(,"data_type":")" + dtype
         + R"(","chunk_grid":{"name":"regular","configuration":{"chunk_shape":)" + chunk
         + R"(}},"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},"fill_value":)" + fill
         + R"(,"codecs":[{"name":"bytes","configuration":{"endian":"little"}}],"attributes":)" + attrs
         + (dims.empty() ? std::string() : (R"(,"dimension_names":)" + dims)) + "}";
}

/// A small coordinates store <dir>/<domain>.zarr as the IO-server prototype writes it, plus two
/// arrays that must not be copied (a time-dependent variable and a level coordinate)
void writeCoordinateStore(const fs::path& dir, const std::string& domain) {
    fs::path s = dir / (domain + ".zarr");
    fs::remove_all(s);
    std::vector<std::pair<std::string, std::string>> nodes;

    std::vector<double> x(NX), y(NY), lat(NPTS), lon(NPTS);
    for (int i = 0; i < NX; ++i) {
        x[i] = 2500.0 * i;
    }
    for (int j = 0; j < NY; ++j) {
        y[j] = 2500.0 * j;
    }
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            lat[j * NX + i] = 50.0 + 0.1 * j + 0.001 * i;
            lon[j * NX + i] = 4.0 + 0.1 * i + 0.001 * j;
        }
    }
    std::int32_t crs = 0;

    nodes.emplace_back("x", arrayDoc("[5]", "[5]", "float64", R"({"standard_name":"projection_x_coordinate","units":"m","axis":"X"})", R"(["x"])"));
    writeFile(s / "x" / "c" / "0", x.data(), x.size() * 8);
    nodes.emplace_back("y", arrayDoc("[4]", "[4]", "float64", R"({"standard_name":"projection_y_coordinate","units":"m","axis":"Y"})", R"(["y"])"));
    writeFile(s / "y" / "c" / "0", y.data(), y.size() * 8);
    nodes.emplace_back("latitude", arrayDoc("[4,5]", "[4,5]", "float64", R"({"standard_name":"latitude","units":"degrees_north"})", R"(["y","x"])"));
    writeFile(s / "latitude" / "c" / "0" / "0", lat.data(), lat.size() * 8);
    nodes.emplace_back("longitude", arrayDoc("[4,5]", "[4,5]", "float64", R"({"standard_name":"longitude","units":"degrees_east"})", R"(["y","x"])"));
    writeFile(s / "longitude" / "c" / "0" / "0", lon.data(), lon.size() * 8);
    nodes.emplace_back("crs", arrayDoc("[]", "[]", "int32", R"({"grid_mapping_name":"lambert_conformal_conic","earth_radius":6.3712290000000000E+006,"standard_parallel":5.1966999999999999E+001})", "", "0"));
    writeFile(s / "crs" / "c", &crs, 4);
    // must not be copied
    nodes.emplace_back("OLDVAR", arrayDoc("[1,4,5]", "[1,4,5]", "float32", R"({"long_name":"stale"})", R"(["time","y","x"])"));
    nodes.emplace_back("level_S", arrayDoc("[1]", "[1]", "float64", R"({"fa_prefix":"S"})", R"(["level_S"])"));

    std::string attrs = R"({"Conventions":"CF-1.8","title":"selftest coordinates","dx_m":2.5000000000000000E+003,"forecast_reference_time":")" + FRT + R"("})";
    std::string root = R"({"zarr_format":3,"node_type":"group","attributes":)" + attrs
                     + R"(,"consolidated_metadata":{"kind":"inline","must_understand":false,"metadata":{)";
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        root += (i ? "," : "") + std::string("\"") + nodes[i].first + "\":" + nodes[i].second;
        writeText(s / nodes[i].first / "zarr.json", nodes[i].second);
    }
    root += "}}}";
    writeText(s / "zarr.json", root);
}

/// Send all steps of the synthetic run; the clients interleave their messages
void runSteps(MultioTestEnvironment& env, const std::string& domain, const std::string& lastKind) {
    for (std::size_t t = 0; t < STEPS.size(); ++t) {
        const auto step = STEPS[t];
        for (const auto& v : VARS) {
            if (t < v.firstTime) {
                continue;
            }
            std::vector<std::int64_t> levels = v.levels.empty() ? std::vector<std::int64_t>{0} : v.levels;
            for (auto lev : levels) {
                for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                    env.process(fieldMessage(domain, v, step, lev, c));
                }
            }
        }
        const std::string kind = (t + 1 == STEPS.size()) ? lastKind : std::string("step");
        for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
            env.process(flushMessage(domain, step, kind, c));
        }
    }
}

std::vector<float> readFloats(const fs::path& p) {
    std::string s = readText(p);
    std::vector<float> v(s.size() / sizeof(float));
    std::memcpy(v.data(), s.data(), v.size() * sizeof(float));
    return v;
}

bool sameBits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

void checkLevelChunk(const std::vector<float>& vals, int id, std::int64_t step, std::int64_t level) {
    EXPECT_EQUAL(vals.size(), static_cast<std::size_t>(NPTS));
    for (std::int64_t i = 0; i < NPTS; ++i) {
        EXPECT(sameBits(vals[i], static_cast<float>(value(id, step, level, i))));
    }
}

std::size_t countStaging(const fs::path& store) {
    std::size_t n = 0;
    for (const auto& e : fs::recursive_directory_iterator(store)) {
        auto name = e.path().filename().string();
        if (name.rfind(".staging", 0) == 0 || name.find(".tmp") != std::string::npos) {
            ++n;
        }
    }
    return n;
}

void runLayout(const std::string& layout, const std::string& lastKind, const std::string& sub) {
    const std::string domain = "TESTDOM";
    const fs::path root = fs::absolute(outRoot());
    const fs::path out = root / sub;
    const fs::path coords = root / "coords";
    fs::remove_all(out);
    writeCoordinateStore(coords, domain);
    registerDomain(domain);

    {
        auto env = MultioTestEnvironment(planJson(out.string(), layout, coords.string()));
        runSteps(env, domain, lastKind);
    }  // destructor of the action finalises if no end-of-simulation flush arrived

    const fs::path s = out / (domain + ".zarr");
    EXPECT(fs::exists(s / "zarr.json"));
    EXPECT(fs::exists(s / ".last_step"));
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=96 time_index=2 inc=0002:00:00\n"));
    EXPECT_EQUAL(countStaging(s), 0u);

    const std::string rootDoc = readText(s / "zarr.json");
    EXPECT(rootDoc.find("\"consolidated_metadata\"") != std::string::npos);
    EXPECT(rootDoc.find("\"latitude\":") != std::string::npos);
    EXPECT(rootDoc.find("OLDVAR") == std::string::npos);
    EXPECT(rootDoc.find("\"dx_m\":2.5000000000000000E+003") != std::string::npos);  // copied verbatim
    EXPECT(!fs::exists(s / "OLDVAR"));
    EXPECT(fs::exists(s / "crs" / "c"));

    // per-array metadata with the final time length
    const std::string tdoc = readText(s / "S_TEMPERATURE" / "zarr.json");
    EXPECT(tdoc.find("\"shape\":[3,3,4,5]") != std::string::npos);
    EXPECT(tdoc.find("\"grib_code\":130") != std::string::npos);
    EXPECT(tdoc.find("\"grid_mapping\":\"crs\"") != std::string::npos);
    EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("\"shape\":[3,4,5]") != std::string::npos);
    EXPECT(readText(s / "KT_ISOT_ALTIT" / "zarr.json").find("\"level_K\"") != std::string::npos);
    EXPECT(readText(s / "S_WIND.U.PHYS" / "zarr.json").find("\"fa_prefix\":\"S\"") != std::string::npos);

    // 2-D: chunk per time step
    checkLevelChunk(readFloats(s / "CLSTEMPERATURE" / "c" / "1" / "0" / "0"), 5, 48, 0);

    if (layout == "chunk-per-level") {
        // level_S = [1,2,3]: k = 0 is level 1
        checkLevelChunk(readFloats(s / "S_TEMPERATURE" / "c" / "0" / "0" / "0" / "0"), 1, 0, 1);
        checkLevelChunk(readFloats(s / "S_TEMPERATURE" / "c" / "2" / "2" / "0" / "0"), 1, 96, 3);
        // level_K = [1,2]: KT only has level 1
        EXPECT(fs::exists(s / "KT_ISOT_ALTIT" / "c" / "0" / "0" / "0" / "0"));
        EXPECT(!fs::exists(s / "KT_ISOT_ALTIT" / "c" / "0" / "1" / "0" / "0"));
        // HUMI.SPECIFI: from time 1 on, levels 1 and 3 only
        EXPECT(!fs::exists(s / "S_HUMI.SPECIFI" / "c" / "0"));
        EXPECT(fs::exists(s / "S_HUMI.SPECIFI" / "c" / "1" / "2" / "0" / "0"));
        EXPECT(!fs::exists(s / "S_HUMI.SPECIFI" / "c" / "1" / "1" / "0" / "0"));
    }
    else {
        EXPECT(tdoc.find("\"sharding_indexed\"") != std::string::npos);
        EXPECT(tdoc.find("\"chunk_shape\":[1,3,4,5]") != std::string::npos);
        // one shard per (variable, time): 3 inner chunks + index of 3 x 16 bytes
        const fs::path shard = s / "S_TEMPERATURE" / "c" / "1" / "0" / "0" / "0";
        EXPECT(fs::exists(shard));
        EXPECT_EQUAL(fs::file_size(shard), static_cast<std::uintmax_t>(3 * NPTS * 4 + 3 * 16));
        EXPECT(!fs::exists(s / "S_TEMPERATURE" / "c" / "1" / "1"));
        // KT: 1 of 2 inner chunks present, the other one marked missing in the index
        const fs::path kt = s / "KT_ISOT_ALTIT" / "c" / "0" / "0" / "0" / "0";
        EXPECT_EQUAL(fs::file_size(kt), static_cast<std::uintmax_t>(NPTS * 4 + 2 * 16));
        std::string b = readText(kt);
        std::uint64_t idx[4];
        std::memcpy(idx, b.data() + NPTS * 4, sizeof(idx));
        EXPECT_EQUAL(idx[0], 0u);
        EXPECT_EQUAL(idx[1], static_cast<std::uint64_t>(NPTS * 4));
        EXPECT(idx[2] == std::numeric_limits<std::uint64_t>::max());
        EXPECT(idx[3] == std::numeric_limits<std::uint64_t>::max());
        std::vector<float> first(NPTS);
        std::memcpy(first.data(), b.data(), NPTS * 4);
        checkLevelChunk(first, 3, 0, 1);
    }
}

CASE("chunk-per-level layout") {
    runLayout("chunk-per-level", "end-of-simulation", "chunk-per-level");
}

CASE("shard-levels layout") {
    runLayout("shard-levels", "end-of-simulation", "shard-levels");
}

CASE("finalisation in the destructor without end-of-simulation") {
    runLayout("chunk-per-level", "step", "no-eos");
}

CASE("new level on a frozen axis is an error") {
    const fs::path out = fs::absolute(outRoot()) / "frozen-axis";
    fs::remove_all(out);
    const std::string domain = "ERRDOM";
    auto plan = std::string(R"({"name": "z", "actions": [{"type": "zarr", "path": ")") + out.string()
              + R"(", "layout": "chunk-per-level"}]})";
    auto env = MultioTestEnvironment(plan);

    auto full = [&](std::int64_t step, std::int64_t level) {
        Metadata md;
        md.set("domain", domain);
        md.set("misc-globalSize", NPTS);
        md.set("misc-precision", std::string("double"));
        md.set("levtype", std::string("P"));
        md.set("suffix", std::string("TEMPERATURE"));
        md.set("levelist", level);
        md.set("is3d", true);
        md.set("step", step);
        md.set("timestep", TIMESTEP);
        md.set("geometry-nx", NX);
        md.set("geometry-ny", NY);
        std::vector<double> vals(NPTS, 1.0);
        return Message{Message::Header{Message::Tag::Field, SERVER, SERVER, std::move(md)},
                       eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
    };
    auto flush = [&](std::int64_t step) {
        Metadata md;
        md.set("domain", domain);
        md.set("step", step);
        md.set("flushKind", std::string("step"));
        return Message{Message::Header{Message::Tag::Flush, SERVER, SERVER, std::move(md)}};
    };

    EXPECT_NO_THROW(env.process(full(0, 50000)));
    EXPECT_NO_THROW(env.process(full(0, 85000)));
    EXPECT_NO_THROW(env.process(flush(0)));
    EXPECT_NO_THROW(env.process(full(48, 85000)));
    EXPECT_THROWS(env.process(full(48, 70000)));
    EXPECT_NO_THROW(env.process(flush(48)));
    // level_P has been written once, with the frozen levels
    EXPECT(fs::exists(out / "ERRDOM.zarr" / "level_P" / "c" / "0"));
    EXPECT_EQUAL(fs::file_size(out / "ERRDOM.zarr" / "level_P" / "c" / "0"), 16u);
    // a field for a step that was flushed already is an error too
    EXPECT_THROWS(env.process(full(0, 50000)));
}

//----------------------------------------------------------------------------------------------------------------------
// Real IAL FullPos input: full C+I+E grid payload with a crop window, FullPos article names / raw CPREF,
// flushes for steps without fields, end-of-simulation without an open step

constexpr std::int64_t FNXF = 7;  // full grid (C+I+E)
constexpr std::int64_t FNYF = 6;
constexpr std::int64_t FX0 = 2;  // 1-based offset of the NX x NY (C+I) window
constexpr std::int64_t FY0 = 2;
constexpr std::int64_t FNPTS = FNXF * FNYF;
constexpr double SENTINEL = -9999.0;  // E zone (and the rest outside the window): must never be written
const std::vector<std::int64_t> FP_FIELD_STEPS = {0, 24};
// flushes: step 12 and 36 have no field (12: empty step, 36: end-of-simulation without an open step)
const std::vector<std::pair<std::int64_t, std::string>> FP_FLUSHES
    = {{0, "step"}, {12, "step"}, {24, "step"}, {36, "end-of-simulation"}};

/// One FullPos field (one level) as IAL sends it; the python check has the same table
struct FpField {
    int id;               // array id (value formula)
    std::string levtype;  // raw CPREF; empty: article name only
    std::string suffix;   // raw CSUFF
    std::int64_t levelist;
    bool sendLevelist;
    std::string name;   // FA article name
    std::int64_t level;  // expected level after normalisation (0: 2-D)
};

const std::vector<FpField> FP_FIELDS = {
    // INI2WRFP: level type letter + ILEVG (pressure in Pa, height in m)
    {1, "P", "TEMPERATURE", 85000, true, "P85000TEMPERATURE", 85000},
    {1, "P", "TEMPERATURE", 50000, true, "P50000TEMPERATURE", 50000},
    // pressure above 1000 hPa: the level is also written into CPREF(2:7)
    {1, "P100500", "TEMPERATURE", 100500, true, "P100500TEMPERATU", 100500},
    // bare level type without levelist: the level comes from the article name
    {1, "P", "TEMPERATURE", 0, false, "P70000TEMPERATURE", 70000},
    // iso-T altitude from the top, CPREF with trailing blanks as Fortran sends it
    {2, "KT  ", "ISOT_ALTIT", 273, true, "KT273ISOT_ALTIT", 273},
    {2, "", "", 0, false, "KT276ISOT_ALTIT", 276},
    {3, "H", "TEMPERATURE", 15, true, "H00015TEMPERATURE", 15},
    {3, "", "", 0, false, "H00500TEMPERATURE", 500},
    // level digits in the prefix (IOGRIDA_MOD convention), also cut inside the level
    {4, "S090", "HUMI.SPECIFI", 0, true, "S090HUMI.SPECIFI", 90},
    {5, "", "", 0, false, "S090TEMPERATURE", 90},
    {6, "H005", "00WIND.U.PHYS", 0, true, "H00500WIND.U.PHY", 500},
    {7, "C", "_METEOSAT_09", 6, true, "C006_METEOSAT_09", 6},
    {7, "", "", 0, false, "C002_METEOSAT_09", 2},
    // 2-D: CPREF = CLNAME(1:4)
    {8, "CLST", "EMPERATURE", 0, true, "CLSTEMPERATURE", 0},
    {9, "", "", 0, false, "SURFTEMPERATURE", 0},
    {10, "SURF", "RAYT SOLA DE", 0, true, "SURFRAYT SOLA DE", 0},
};

double fpValue(int id, std::int64_t step, std::int64_t level, std::int64_t idx) {
    if (id == 9 && idx == 3) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return id * 1.0e6 + step * 1000.0 + level * 0.01 + idx * 0.001;
}

/// Full grid payload: window point (i, j) (0-based) at (FY0-1+j)*FNXF + (FX0-1+i), everything else SENTINEL
std::vector<double> fpFullGrid(const FpField& f, std::int64_t step) {
    std::vector<double> g(FNPTS, SENTINEL);
    for (std::int64_t j = 0; j < NY; ++j) {
        for (std::int64_t i = 0; i < NX; ++i) {
            g[(FY0 - 1 + j) * FNXF + (FX0 - 1 + i)] = fpValue(f.id, step, f.level, j * NX + i);
        }
    }
    return g;
}

void setFpGeometry(Metadata& md) {
    md.set("geometry-nx", NX);
    md.set("geometry-ny", NY);
    md.set("geometry-nx-full", FNXF);
    md.set("geometry-ny-full", FNYF);
    md.set("geometry-x0", FX0);
    md.set("geometry-y0", FY0);
}

Message fpFieldMessage(const std::string& domain, const FpField& f, std::int64_t step, std::size_t c) {
    Metadata md;
    md.set("category", std::string("ial-field"));
    md.set("domain", domain);
    md.set("misc-globalSize", FNPTS);
    md.set("misc-precision", std::string("double"));
    md.set("name", f.name);
    if (!f.levtype.empty()) {
        md.set("levtype", f.levtype);
        md.set("suffix", f.suffix);
        if (f.sendLevelist) {
            md.set("levelist", f.levelist);
        }
    }
    md.set("grib-code", std::int64_t{0});
    md.set("step", step);
    md.set("timestep", TIMESTEP);
    md.set("forecastReferenceTime", FRT);
    setFpGeometry(md);

    const auto full = fpFullGrid(f, step);
    std::vector<double> vals;
    for (auto i : clientIndices(c, FNPTS)) {
        vals.push_back(full[i]);
    }
    return Message{Message::Header{Message::Tag::Field, CLIENTS[c], SERVER, std::move(md)},
                   eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
}

std::size_t countValue(const fs::path& dir, float v) {
    std::size_t n = 0;
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().filename() == "zarr.json") {
            continue;
        }
        for (float x : readFloats(e.path())) {
            n += sameBits(x, v) ? 1 : 0;
        }
    }
    return n;
}

void runFullPos(const std::string& layout, const std::string& sub) {
    const std::string domain = "PFTESTDOM";
    const fs::path root = fs::absolute(outRoot());
    const fs::path out = root / sub;
    const fs::path coords = root / "coords";
    fs::remove_all(out);
    writeCoordinateStore(coords, domain);  // static arrays of the C+I window (NX x NY)
    registerDomain(domain, FNPTS);         // the clients' points cover the full C+I+E grid
    registerDomain("PFNOFIELDS", FNPTS);   // a domain that never gets a field

    {
        auto env = MultioTestEnvironment(planJson(out.string(), layout, coords.string()));
        for (const auto& [step, kind] : FP_FLUSHES) {
            if (std::find(FP_FIELD_STEPS.begin(), FP_FIELD_STEPS.end(), step) != FP_FIELD_STEPS.end()) {
                for (const auto& f : FP_FIELDS) {
                    for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                        env.process(fpFieldMessage(domain, f, step, c));
                    }
                }
            }
            for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                EXPECT_NO_THROW(env.process(flushMessage(domain, step, kind, c)));
                EXPECT_NO_THROW(env.process(flushMessage("PFNOFIELDS", step, kind, c)));
            }
        }
    }

    EXPECT(!fs::exists(out / "PFNOFIELDS.zarr"));
    const fs::path s = out / (domain + ".zarr");
    EXPECT(fs::exists(s / "zarr.json"));
    // two time steps (0 and 24), the empty steps 12 and 36 add none
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=24 time_index=1 inc=0000:30:00\n"));
    EXPECT(readText(s / "time" / "zarr.json").find("\"shape\":[2]") != std::string::npos);
    EXPECT(!fs::exists(s / "time" / "c" / "2"));
    EXPECT_EQUAL(countStaging(s), 0u);

    // array names and level axes as the IO-server prototype names FullPos output
    const std::string pdoc = readText(s / "P_TEMPERATURE" / "zarr.json");
    EXPECT(pdoc.find("\"shape\":[2,4,4,5]") != std::string::npos);
    EXPECT(pdoc.find("\"level_P\"") != std::string::npos);
    EXPECT(pdoc.find("\"fa_prefix\":\"P\"") != std::string::npos);
    EXPECT(readText(s / "KT_ISOT_ALTIT" / "zarr.json").find("\"shape\":[2,2,4,5]") != std::string::npos);
    EXPECT(readText(s / "H_TEMPERATURE" / "zarr.json").find("\"shape\":[2,2,4,5]") != std::string::npos);
    EXPECT(readText(s / "H_WIND.U.PHYS" / "zarr.json").find("\"fa_suffix\":\"WIND.U.PHYS\"") != std::string::npos);
    EXPECT(readText(s / "S_HUMI.SPECIFI" / "zarr.json").find("\"shape\":[2,1,4,5]") != std::string::npos);
    EXPECT(readText(s / "C_METEOSAT_09" / "zarr.json").find("\"fa_suffix\":\"_METEOSAT_09\"") != std::string::npos);
    const std::string cdoc = readText(s / "CLSTEMPERATURE" / "zarr.json");
    EXPECT(cdoc.find("\"shape\":[2,4,5]") != std::string::npos);
    EXPECT(cdoc.find("\"fa_prefix\":\"CLST\",\"fa_suffix\":\"EMPERATURE\"") != std::string::npos);
    EXPECT(fs::exists(s / "SURFRAYT_SOLA_DE" / "zarr.json"));
    for (const char* bad : {"P100500TEMPERATURE", "KT273ISOT_ALTIT", "KT27", "S090HUMI.SPECIFI", "H005",
                            "C002_METEOSAT_09", "P_", "KT"}) {
        EXPECT(!fs::exists(s / bad));
    }
    auto levels = [&](const std::string& ax) {
        std::string b = readText(s / ax / "c" / "0");
        std::vector<double> v(b.size() / sizeof(double));
        std::memcpy(v.data(), b.data(), b.size());
        return v;
    };
    EXPECT(levels("level_P") == (std::vector<double>{50000, 70000, 85000, 100500}));
    EXPECT(levels("level_K") == (std::vector<double>{273, 276}));
    EXPECT(levels("level_H") == (std::vector<double>{15, 500}));
    EXPECT(levels("level_S") == (std::vector<double>{90}));
    EXPECT(levels("level_C") == (std::vector<double>{2, 6}));

    // only the window is written: no sentinel anywhere in the data
    for (const char* v : {"P_TEMPERATURE", "KT_ISOT_ALTIT", "H_TEMPERATURE", "S_TEMPERATURE", "C_METEOSAT_09",
                          "CLSTEMPERATURE", "SURFTEMPERATURE"}) {
        EXPECT_EQUAL(countValue(s / v / "c", static_cast<float>(SENTINEL)), 0u);
    }
    // 2-D, time index 1 (step 24): the window values in C order
    {
        auto vals = readFloats(s / "CLSTEMPERATURE" / "c" / "1" / "0" / "0");
        EXPECT_EQUAL(vals.size(), static_cast<std::size_t>(NPTS));
        for (std::int64_t i = 0; i < NPTS && i < static_cast<std::int64_t>(vals.size()); ++i) {
            EXPECT(sameBits(vals[i], static_cast<float>(fpValue(8, 24, 0, i))));
        }
    }
    if (layout == "chunk-per-level") {
        // level_P = [50000, 70000, 85000, 100500]: k = 3 is 100500
        auto vals = readFloats(s / "P_TEMPERATURE" / "c" / "0" / "3" / "0" / "0");
        EXPECT_EQUAL(vals.size(), static_cast<std::size_t>(NPTS));
        for (std::int64_t i = 0; i < NPTS && i < static_cast<std::int64_t>(vals.size()); ++i) {
            EXPECT(sameBits(vals[i], static_cast<float>(fpValue(1, 0, 100500, i))));
        }
    }
}

CASE("FullPos input: crop window, FullPos names, empty steps, chunk-per-level") {
    runFullPos("chunk-per-level", "fullpos-chunk-per-level");
}

CASE("FullPos input: crop window, FullPos names, empty steps, shard-levels") {
    runFullPos("shard-levels", "fullpos-shard-levels");
}

CASE("payload size, crop window and flush errors") {
    const fs::path out = fs::absolute(outRoot()) / "payload-errors";
    fs::remove_all(out);
    const std::string domain = "SIZEDOM";
    auto plan = std::string(R"({"name": "z", "actions": [{"type": "zarr", "path": ")") + out.string()
              + R"(", "layout": "chunk-per-level"}]})";
    auto env = MultioTestEnvironment(plan);

    // fld(step, npayload, globalSize, crop keys to set: bit 0 nx-full, 1 ny-full, 2 x0, 3 y0)
    auto fld = [&](std::int64_t step, std::int64_t npayload, std::int64_t globalSize, int cropBits,
                   std::int64_t x0 = FX0) {
        Metadata md;
        md.set("domain", domain);
        md.set("misc-globalSize", globalSize);
        md.set("misc-precision", std::string("double"));
        md.set("name", std::string("SURFTEMPERATURE"));
        md.set("step", step);
        md.set("timestep", TIMESTEP);
        md.set("geometry-nx", NX);
        md.set("geometry-ny", NY);
        if (cropBits & 1) {
            md.set("geometry-nx-full", FNXF);
        }
        if (cropBits & 2) {
            md.set("geometry-ny-full", FNYF);
        }
        if (cropBits & 4) {
            md.set("geometry-x0", x0);
        }
        if (cropBits & 8) {
            md.set("geometry-y0", FY0);
        }
        std::vector<double> vals(npayload, 1.0);
        return Message{Message::Header{Message::Tag::Field, SERVER, SERVER, std::move(md)},
                       eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
    };
    auto flush = [&](const std::string& dom, std::optional<std::int64_t> step, const std::string& kind) {
        Metadata md;
        md.set("domain", dom);
        if (step) {
            md.set("step", *step);
        }
        md.set("flushKind", kind);
        return Message{Message::Header{Message::Tag::Flush, SERVER, SERVER, std::move(md)}};
    };

    // flushes before any field: no store, no error
    EXPECT_NO_THROW(env.process(flush(domain, 0, "step")));
    EXPECT_NO_THROW(env.process(flush("OTHERDOM", 0, "end-of-simulation")));
    EXPECT(!fs::exists(out / "SIZEDOM.zarr"));
    EXPECT(!fs::exists(out / "OTHERDOM.zarr"));

    EXPECT_NO_THROW(env.process(fld(0, NPTS, NPTS, 0)));            // plain nx*ny
    EXPECT_NO_THROW(env.process(fld(0, FNPTS, FNPTS, 15)));         // full grid + crop window (same field again)
    EXPECT_THROWS(env.process(fld(0, FNPTS, FNPTS, 0)));            // full grid without the crop keys
    EXPECT_THROWS(env.process(fld(0, FNPTS, FNPTS, 7)));            // incomplete crop keys
    EXPECT_THROWS(env.process(fld(0, NPTS, NPTS, 15)));             // crop keys but an nx*ny payload
    EXPECT_THROWS(env.process(fld(0, FNPTS, FNPTS, 15, FNXF)));     // window outside the full grid
    EXPECT_THROWS(env.process(fld(0, NPTS, FNPTS, 0)));             // payload != misc-globalSize (not aggregated)
    EXPECT_THROWS(env.process(fld(0, FNPTS - 1, FNPTS - 1, 15)));   // payload != nx_full*ny_full
    EXPECT_NO_THROW(env.process(flush(domain, 0, "step")));
    EXPECT_NO_THROW(env.process(flush(domain, 0, "step")));         // repeated flush of a closed step
    EXPECT_NO_THROW(env.process(flush(domain, 1, "step")));         // step without fields
    EXPECT_THROWS(env.process(fld(1, NPTS, NPTS, 0)));              // field for that (flushed) step
    EXPECT_NO_THROW(env.process(fld(2, FNPTS, FNPTS, 15)));
    EXPECT_NO_THROW(env.process(flush(domain, 2, "step")));
    EXPECT_NO_THROW(env.process(flush(domain, std::nullopt, "step")));  // no open step: nothing to do
    // end-of-simulation for a step without fields: no time step, just finalise
    EXPECT_NO_THROW(env.process(flush(domain, 3, "end-of-simulation")));
    const fs::path s = out / "SIZEDOM.zarr";
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=2 time_index=1 inc=0000:02:30\n"));
    EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("\"shape\":[2,4,5]") != std::string::npos);
    EXPECT(readText(s / "time" / "zarr.json").find("\"shape\":[2]") != std::string::npos);
    EXPECT(readText(s / "zarr.json").find("\"shape\":[2,4,5]") != std::string::npos);
    // close-connection after end-of-simulation: nothing changes
    EXPECT_NO_THROW(env.process(flush("global", std::nullopt, "close-connection")));
    EXPECT(readText(s / "time" / "zarr.json").find("\"shape\":[2]") != std::string::npos);
}

CASE("configuration errors") {
    EXPECT_THROWS(MultioTestEnvironment(R"({"name": "z", "actions": [{"type": "zarr"}]})"));
    EXPECT_THROWS(MultioTestEnvironment(R"({"name": "z", "actions": [{"type": "zarr", "path": "x", "format": 2}]})"));
    EXPECT_THROWS(
        MultioTestEnvironment(R"({"name": "z", "actions": [{"type": "zarr", "path": "x", "layout": "foo"}]})"));
}

}  // namespace multio::test::zarr

int main(int argc, char** argv) {
    return eckit::testing::run_tests(argc, argv);
}
