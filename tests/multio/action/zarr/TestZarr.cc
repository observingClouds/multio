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
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <ftw.h>
#include <sys/stat.h>
#include <unistd.h>

#include "eckit/testing/Test.h"

#include "multio/action/zarr/ZarrCodec.h"
#include "multio/action/zarr/ZarrQuantize.h"
#include "multio/action/zarr/ZarrStore.h"
#include "multio/action/zarr/ZarrUtil.h"
#include "multio/domain/Mappings.h"
#include "multio/message/Message.h"
#include "multio/message/Metadata.h"
#include "multio/message/Peer.h"

#include "../../MultioTestEnvironment.h"

namespace multio::test::zarr {

using multio::message::Message;
using multio::message::Metadata;
using multio::message::Peer;

//----------------------------------------------------------------------------------------------------------------------
// Minimal path helpers (POSIX), no C++17 <fs> library: with icpc + gcc 8 headers it uses the experimental ABI,
// which crashes against the newer libstdc++ loaded at run time.

/// '/'-joined path string
class Path {
public:
    Path() = default;
    Path(std::string s) : s_(std::move(s)) {}
    Path(const char* s) : s_(s) {}
    const std::string& string() const { return s_; }
    Path parent() const {
        auto p = s_.find_last_of('/');
        return p == std::string::npos ? Path() : Path(p == 0 ? std::string("/") : s_.substr(0, p));
    }
    std::string filename() const {
        auto p = s_.find_last_of('/');
        return p == std::string::npos ? s_ : s_.substr(p + 1);
    }
    friend Path operator/(const Path& a, const std::string& b) {
        if (a.s_.empty()) {
            return Path(b);
        }
        return Path(a.s_.back() == '/' ? a.s_ + b : a.s_ + "/" + b);
    }

private:
    std::string s_;
};

[[noreturn]] void pathError(const std::string& what, const Path& p, int err) {
    throw std::runtime_error(what + " '" + p.string() + "': " + std::strerror(err));
}

bool exists(const Path& p) {
    struct stat st;
    return ::stat(p.string().c_str(), &st) == 0;
}

std::uintmax_t fileSize(const Path& p) {
    struct stat st;
    if (::stat(p.string().c_str(), &st) != 0) {
        pathError("cannot stat", p, errno);
    }
    return static_cast<std::uintmax_t>(st.st_size);
}

/// mkdir -p
void createDirectories(const Path& p) {
    if (p.string().empty() || ::mkdir(p.string().c_str(), 0755) == 0) {
        return;
    }
    int err = errno;
    struct stat st;
    if (err == EEXIST && ::stat(p.string().c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        return;
    }
    if (err == ENOENT && p.parent().string() != p.string()) {
        createDirectories(p.parent());
        if (::mkdir(p.string().c_str(), 0755) == 0 || errno == EEXIST) {
            return;
        }
        err = errno;
    }
    pathError("cannot create directory", p, err);
}

/// rm -rf; a missing path is not an error
void removeAll(const Path& p) {
    struct stat st;
    if (::lstat(p.string().c_str(), &st) != 0) {
        if (errno == ENOENT) {
            return;
        }
        pathError("cannot stat", p, errno);
    }
    auto rm = [](const char* path, const struct stat*, int, struct FTW*) { return ::remove(path) == 0 ? 0 : errno; };
    int rc = ::nftw(p.string().c_str(), rm, 64, FTW_DEPTH | FTW_PHYS);
    if (rc != 0) {
        pathError("cannot remove", p, rc > 0 ? rc : errno);
    }
}

Path absolute(const Path& p) {
    if (!p.string().empty() && p.string()[0] == '/') {
        return p;
    }
    std::vector<char> buf(4096);
    while (::getcwd(buf.data(), buf.size()) == nullptr) {
        if (errno != ERANGE) {
            pathError("cannot get the working directory for", p, errno);
        }
        buf.resize(buf.size() * 2);
    }
    return Path(std::string(buf.data())) / p.string();
}

/// Call f(path, isRegularFile) for every entry below dir (recursing into directories, not into symbolic links
/// to directories; isRegularFile follows symbolic links)
void walk(const Path& dir, const std::function<void(const Path&, bool)>& f) {
    DIR* d = ::opendir(dir.string().c_str());
    if (d == nullptr) {
        pathError("cannot open directory", dir, errno);
    }
    std::vector<std::string> names;
    while (struct dirent* e = ::readdir(d)) {
        if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0) {
            names.emplace_back(e->d_name);
        }
    }
    ::closedir(d);
    for (const auto& n : names) {
        Path p = dir / n;
        struct stat lst, st;
        if (::lstat(p.string().c_str(), &lst) != 0) {
            continue;
        }
        bool regular = ::stat(p.string().c_str(), &st) == 0 && S_ISREG(st.st_mode);
        f(p, regular);
        if (S_ISDIR(lst.st_mode)) {
            walk(p, f);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

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

/// `extra`: further members of the zarr action, e.g. R"(, "compression": {"codec": "zstd"})"
std::string planJson(const std::string& path, const std::string& layout, const std::string& coords,
                     const std::string& extra = "") {
    std::ostringstream oss;
    oss << R"({"name": "ial-to-zarr", "actions": [)"
        << R"({"type": "select", "match": [{"category": ["ial-field", "ial-static"]}]},)"
        << R"({"type": "aggregate"},)"
        << R"({"type": "zarr", "path": ")" << path << R"(", "format": 3, "layout": ")" << layout << R"(")";
    if (!coords.empty()) {
        oss << R"(, "coordinates-from": ")" << coords << R"(")";
    }
    oss << extra << "}]}";
    return oss.str();
}

void writeFile(const Path& p, const void* data, std::size_t n) {
    createDirectories(p.parent());
    std::ofstream out(p.string(), std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}

void writeText(const Path& p, const std::string& s) {
    writeFile(p, s.data(), s.size());
}

std::string readText(const Path& p) {
    std::ifstream in(p.string(), std::ios::binary);
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
void writeCoordinateStore(const Path& dir, const std::string& domain) {
    Path s = dir / (domain + ".zarr");
    removeAll(s);
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

std::vector<float> readFloats(const Path& p) {
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

std::size_t countStaging(const Path& store) {
    std::size_t n = 0;
    walk(store, [&](const Path& p, bool) {
        auto name = p.filename();
        if (name.rfind(".staging", 0) == 0 || name.find(".tmp") != std::string::npos) {
            ++n;
        }
    });
    return n;
}

void runLayout(const std::string& layout, const std::string& lastKind, const std::string& sub) {
    const std::string domain = "TESTDOM";
    const Path root = absolute(outRoot());
    const Path out = root / sub;
    const Path coords = root / "coords";
    removeAll(out);
    writeCoordinateStore(coords, domain);
    registerDomain(domain);

    {
        auto env = MultioTestEnvironment(planJson(out.string(), layout, coords.string()));
        runSteps(env, domain, lastKind);
    }  // destructor of the action finalises if no end-of-simulation flush arrived

    const Path s = out / (domain + ".zarr");
    EXPECT(exists(s / "zarr.json"));
    EXPECT(exists(s / ".last_step"));
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=96 time_index=2 inc=0002:00:00\n"));
    EXPECT_EQUAL(countStaging(s), 0u);

    const std::string rootDoc = readText(s / "zarr.json");
    EXPECT(rootDoc.find("\"consolidated_metadata\"") != std::string::npos);
    EXPECT(rootDoc.find("\"latitude\":") != std::string::npos);
    EXPECT(rootDoc.find("OLDVAR") == std::string::npos);
    EXPECT(rootDoc.find("\"dx_m\":2.5000000000000000E+003") != std::string::npos);  // copied verbatim
    EXPECT(!exists(s / "OLDVAR"));
    EXPECT(exists(s / "crs" / "c"));

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
        EXPECT(exists(s / "KT_ISOT_ALTIT" / "c" / "0" / "0" / "0" / "0"));
        EXPECT(!exists(s / "KT_ISOT_ALTIT" / "c" / "0" / "1" / "0" / "0"));
        // HUMI.SPECIFI: from time 1 on, levels 1 and 3 only
        EXPECT(!exists(s / "S_HUMI.SPECIFI" / "c" / "0"));
        EXPECT(exists(s / "S_HUMI.SPECIFI" / "c" / "1" / "2" / "0" / "0"));
        EXPECT(!exists(s / "S_HUMI.SPECIFI" / "c" / "1" / "1" / "0" / "0"));
    }
    else {
        EXPECT(tdoc.find("\"sharding_indexed\"") != std::string::npos);
        EXPECT(tdoc.find("\"chunk_shape\":[1,3,4,5]") != std::string::npos);
        // one shard per (variable, time): 3 inner chunks + index of 3 x 16 bytes
        const Path shard = s / "S_TEMPERATURE" / "c" / "1" / "0" / "0" / "0";
        EXPECT(exists(shard));
        EXPECT_EQUAL(fileSize(shard), static_cast<std::uintmax_t>(3 * NPTS * 4 + 3 * 16));
        EXPECT(!exists(s / "S_TEMPERATURE" / "c" / "1" / "1"));
        // KT: 1 of 2 inner chunks present, the other one marked missing in the index
        const Path kt = s / "KT_ISOT_ALTIT" / "c" / "0" / "0" / "0" / "0";
        EXPECT_EQUAL(fileSize(kt), static_cast<std::uintmax_t>(NPTS * 4 + 2 * 16));
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

// The clients' close-connection flushes (domain "global") pass the aggregate action one by one, so a
// client that is done closes while the last step of the others is still being aggregated. That must
// not finalise the store and drop the open step (with 624 clients the 3-D fields of the last but one
// output step were lost).
CASE("close-connection of one client while the last step is still open") {
    for (const std::string layout : {"shard-levels", "chunk-per-level"}) {
        const std::string domain = "TESTDOM";
        const Path out = absolute(outRoot()) / ("early-close-" + layout);
        removeAll(out);
        registerDomain(domain);

        auto closeMessage = [](std::size_t c) {
            Metadata md;
            md.set("domain", std::string("global"));
            md.set("flushKind", std::string("close-connection"));
            md.set("toAllServers", true);
            return Message{Message::Header{Message::Tag::Flush, CLIENTS[c], SERVER, std::move(md)}};
        };

        {
            auto env = MultioTestEnvironment(planJson(out.string(), layout, ""));
            // the parts of client c for the first or the second half of VARS
            auto send = [&](std::size_t c, std::size_t t, bool firstHalf) {
                for (std::size_t i = 0; i < VARS.size(); ++i) {
                    const auto& v = VARS[i];
                    if ((i < VARS.size() / 2) != firstHalf || t < v.firstTime) {
                        continue;
                    }
                    std::vector<std::int64_t> levels = v.levels.empty() ? std::vector<std::int64_t>{0} : v.levels;
                    for (auto lev : levels) {
                        env.process(fieldMessage(domain, v, STEPS[t], lev, c));
                    }
                }
            };
            for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                send(c, 0, true);
                send(c, 0, false);
            }
            for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                env.process(flushMessage(domain, STEPS[0], "step", c));
            }
            // last step: the first half of the variables is aggregated (open in the store), then
            // client 0 sends the rest and its end-of-simulation flush and closes before client 1 is done
            send(0, 1, true);
            send(1, 1, true);
            send(0, 1, false);
            env.process(flushMessage(domain, STEPS[1], "end-of-simulation", 0));
            env.process(closeMessage(0));
            send(1, 1, false);
            env.process(flushMessage(domain, STEPS[1], "end-of-simulation", 1));
            env.process(closeMessage(1));
        }

        const Path s = out / (domain + ".zarr");
        EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=48 time_index=1 inc=0001:00:00\n"));
        EXPECT_EQUAL(countStaging(s), 0u);
        EXPECT(readText(s / "S_TEMPERATURE" / "zarr.json").find("\"shape\":[2,3,4,5]") != std::string::npos);
        // S_TEMPERATURE was aggregated before the early close, S_WIND.U.PHYS (level 2, k = 1) after it
        if (layout == "shard-levels") {
            const Path shard = s / "S_TEMPERATURE" / "c" / "1" / "0" / "0" / "0";
            EXPECT(exists(shard));
            EXPECT_EQUAL(fileSize(shard), static_cast<std::uintmax_t>(3 * NPTS * 4 + 3 * 16));
            std::string b = readText(shard);
            std::vector<float> first(NPTS);
            std::memcpy(first.data(), b.data(), NPTS * 4);
            checkLevelChunk(first, 1, 48, 1);
            EXPECT(exists(s / "S_WIND.U.PHYS" / "c" / "1" / "0" / "0" / "0"));
        }
        else {
            checkLevelChunk(readFloats(s / "S_TEMPERATURE" / "c" / "1" / "2" / "0" / "0"), 1, 48, 3);
            checkLevelChunk(readFloats(s / "S_WIND.U.PHYS" / "c" / "1" / "1" / "0" / "0"), 7, 48, 2);
        }
        checkLevelChunk(readFloats(s / "CLSTEMPERATURE" / "c" / "1" / "0" / "0"), 5, 48, 0);
    }
}

CASE("new level on a frozen axis is an error") {
    const Path out = absolute(outRoot()) / "frozen-axis";
    removeAll(out);
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
    EXPECT(exists(out / "ERRDOM.zarr" / "level_P" / "c" / "0"));
    EXPECT_EQUAL(fileSize(out / "ERRDOM.zarr" / "level_P" / "c" / "0"), 16u);
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

std::size_t countValue(const Path& dir, float v) {
    std::size_t n = 0;
    walk(dir, [&](const Path& p, bool regular) {
        if (!regular || p.filename() == "zarr.json") {
            return;
        }
        for (float x : readFloats(p)) {
            n += sameBits(x, v) ? 1 : 0;
        }
    });
    return n;
}

void runFullPos(const std::string& layout, const std::string& sub) {
    const std::string domain = "PFTESTDOM";
    const Path root = absolute(outRoot());
    const Path out = root / sub;
    const Path coords = root / "coords";
    removeAll(out);
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

    EXPECT(!exists(out / "PFNOFIELDS.zarr"));
    const Path s = out / (domain + ".zarr");
    EXPECT(exists(s / "zarr.json"));
    // two time steps (0 and 24), the empty steps 12 and 36 add none
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=24 time_index=1 inc=0000:30:00\n"));
    EXPECT(readText(s / "time" / "zarr.json").find("\"shape\":[2]") != std::string::npos);
    EXPECT(!exists(s / "time" / "c" / "2"));
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
    EXPECT(exists(s / "SURFRAYT_SOLA_DE" / "zarr.json"));
    for (const char* bad : {"P100500TEMPERATURE", "KT273ISOT_ALTIT", "KT27", "S090HUMI.SPECIFI", "H005",
                            "C002_METEOSAT_09", "P_", "KT"}) {
        EXPECT(!exists(s / bad));
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
    const Path out = absolute(outRoot()) / "payload-errors";
    removeAll(out);
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
    EXPECT(!exists(out / "SIZEDOM.zarr"));
    EXPECT(!exists(out / "OTHERDOM.zarr"));

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
    const Path s = out / "SIZEDOM.zarr";
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=2 time_index=1 inc=0000:02:30\n"));
    EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("\"shape\":[2,4,5]") != std::string::npos);
    EXPECT(readText(s / "time" / "zarr.json").find("\"shape\":[2]") != std::string::npos);
    EXPECT(readText(s / "zarr.json").find("\"shape\":[2,4,5]") != std::string::npos);
    // close-connection after end-of-simulation: nothing changes
    EXPECT_NO_THROW(env.process(flush("global", std::nullopt, "close-connection")));
    EXPECT(readText(s / "time" / "zarr.json").find("\"shape\":[2]") != std::string::npos);
}

//----------------------------------------------------------------------------------------------------------------------
// Static arrays sent by the client (zarr-static, skipAggregation): x, y, latitude, longitude, crs, hyai, hybi and
// the root group attributes, sent by one client task through select -> aggregate -> zarr

constexpr std::int64_t NHYI = 4;
// sent with whitespace, written compact; the numbers keep their (Fortran) formatting
const std::string CRS_ATTRS_SENT
    = R"({ "grid_mapping_name": "lambert_conformal_conic", "earth_radius": 6.3712290000000000E+006,
          "standard_parallel": 5.1966999999999999E+001 })";
const std::string CRS_ATTRS
    = R"({"grid_mapping_name":"lambert_conformal_conic","earth_radius":6.3712290000000000E+006,"standard_parallel":5.1966999999999999E+001})";
const std::string X_ATTRS = R"({"standard_name":"projection_x_coordinate","units":"m","axis":"X"})";
const std::string Y_ATTRS = R"({"standard_name":"projection_y_coordinate","units":"m","axis":"Y"})";
const std::string LAT_ATTRS = R"({"standard_name":"latitude","units":"degrees_north"})";
const std::string LON_ATTRS = R"({"standard_name":"longitude","units":"degrees_east"})";
const std::string HYAI_ATTRS = R"({"long_name":"hybrid A coefficient at half levels","units":"Pa"})";
const std::string HYBI_ATTRS = R"({"long_name":"hybrid B coefficient at half levels","units":"1"})";
// the client's root attributes; forecast_reference_time is overwritten with the store's reference time
const std::string GROUP_ATTRS_SENT
    = R"J({"Conventions": "CF-1.8", "title": "IAL output, domain STAT", "source": "IAL MultIO client (prototype)",
         "forecast_reference_time": "1999-12-31T00:00:00", "nx_ci": 5, "ny_ci": 4, "dx_m": 2.5000000000000000E+003})J";
const std::string GROUP_ATTRS
    = R"J({"Conventions":"CF-1.8","title":"IAL output, domain STAT","source":"IAL MultIO client (prototype)","forecast_reference_time":")J"
    + FRT + R"(","nx_ci":5,"ny_ci":4,"dx_m":2.5000000000000000E+003})";

/// Static values; `variant` changes the values (replacement / repetition tests). The python check has the same formulas.
std::vector<double> staticValues(const std::string& name, int variant = 0) {
    std::vector<double> v;
    if (name == "x") {
        for (int i = 0; i < NX; ++i) {
            v.push_back(-5000.0 + 2500.0 * i + variant);
        }
    }
    else if (name == "y") {
        for (int j = 0; j < NY; ++j) {
            v.push_back(-3750.0 + 2500.0 * j + variant);
        }
    }
    else if (name == "latitude" || name == "longitude") {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                v.push_back(name == "latitude" ? 51.0 + 0.0225 * j + 1.0e-4 * i + variant
                                               : 4.9 + 0.0361 * i - 2.0e-4 * j + variant);
            }
        }
    }
    else if (name == "crs") {
        v.push_back(0.0);
    }
    else if (name == "hyai" || name == "hybi") {
        for (int k = 0; k < NHYI; ++k) {
            v.push_back(name == "hyai" ? 1000.0 * k * (NHYI - 1 - k) : static_cast<double>(k) / (NHYI - 1));
        }
    }
    return v;
}

struct StaticSpec {
    std::string name, dims, shape, dtype, attrs, attrsSent;
};

const std::vector<StaticSpec> STATICS = {
    {"x", "x", "5", "float64", X_ATTRS, X_ATTRS},
    {"y", "y", "4", "float64", Y_ATTRS, Y_ATTRS},
    {"latitude", "y,x", "4,5", "float64", LAT_ATTRS, LAT_ATTRS},
    {"longitude", "y,x", "4,5", "float64", LON_ATTRS, LON_ATTRS},
    {"crs", "", "", "int32", CRS_ATTRS, CRS_ATTRS_SENT},
    {"hyai", "nhyi", "4", "float64", HYAI_ATTRS, HYAI_ATTRS},
    {"hybi", "nhyi", "4", "float64", HYBI_ATTRS, HYBI_ATTRS},
};

const StaticSpec& staticSpec(const std::string& name) {
    return *std::find_if(STATICS.begin(), STATICS.end(), [&](const auto& s) { return s.name == name; });
}

/// A static message as the IAL client sends it (task 1 only); md overrides for the error tests
Message staticMessage(const std::string& domain, const std::string& name, const std::string& dims,
                      const std::string& shape, const std::string& dtype, const std::string& attrs,
                      const std::vector<double>& vals, const std::string& groupAttrs = "",
                      const std::function<void(Metadata&)>& tweak = {}) {
    Metadata md;
    md.set("category", std::string("ial-static"));
    md.set("domain", domain);
    md.set("name", name);
    md.set("skipAggregation", true);
    md.set("zarr-static", true);
    md.set("zarr-dimensions", dims);
    md.set("zarr-shape", shape);
    md.set("zarr-dtype", dtype);
    md.set("zarr-attributes", attrs);
    if (!groupAttrs.empty()) {
        md.set("zarr-group-attributes", groupAttrs);
    }
    md.set("geometry-nx", NX);
    md.set("geometry-ny", NY);
    md.set("forecastReferenceTime", FRT);
    md.set("misc-precision", std::string("double"));
    md.set("misc-globalSize", static_cast<std::int64_t>(vals.size()));
    if (tweak) {
        tweak(md);
    }
    return Message{Message::Header{Message::Tag::Field, CLIENTS[0], SERVER, std::move(md)},
                   eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
}

Message staticMessage(const std::string& domain, const std::string& name, int variant = 0,
                      const std::string& groupAttrs = "", const std::function<void(Metadata&)>& tweak = {}) {
    const auto& sp = staticSpec(name);
    return staticMessage(domain, sp.name, sp.dims, sp.shape, sp.dtype, sp.attrsSent, staticValues(name, variant),
                         groupAttrs, tweak);
}

void sendStatics(MultioTestEnvironment& env, const std::string& domain, const std::vector<std::string>& names,
                 int variant = 0) {
    for (const auto& n : names) {
        // x carries the root attributes; skipAggregation leniently as bool / int / string
        env.process(staticMessage(domain, n, variant, n == "x" ? GROUP_ATTRS_SENT : "", [&](Metadata& md) {
            if (n == "y") {
                md.set("skipAggregation", std::int64_t{1});
            }
            if (n == "latitude") {
                md.set("skipAggregation", std::string("true"));
            }
        }));
    }
}

/// zarr.json the IO server writes for a static array (member order of ZARR_STORE_ARRAY_META)
std::string expectedStaticDoc(const std::string& name) {
    const auto& sp = staticSpec(name);
    std::string dims;
    if (!sp.dims.empty()) {
        dims = "[\"" + sp.dims + "\"]";
        auto c = dims.find(',');
        if (c != std::string::npos) {
            dims.replace(c, 1, "\",\"");
        }
    }
    const std::string shape = "[" + sp.shape + "]";
    return arrayDoc(shape, shape, sp.dtype, sp.attrs, dims, sp.dtype == "int32" ? "0" : "\"NaN\"");
}

std::string staticChunk(const std::string& name) {
    const auto& sp = staticSpec(name);
    std::string c = "c";
    if (!sp.shape.empty()) {
        c += "/0";
        if (sp.shape.find(',') != std::string::npos) {
            c += "/0";
        }
    }
    return c;
}

/// Exact files of a static array: zarr.json byte-identical to the IO-server layout, one chunk with the values
void checkStaticFiles(const Path& s, const std::string& name, int variant = 0) {
    const auto& sp = staticSpec(name);
    EXPECT_EQUAL(readText(s / name / "zarr.json"), expectedStaticDoc(name));
    const Path chunk = s / name / staticChunk(name);
    EXPECT(exists(chunk));
    const auto vals = staticValues(name, variant);
    std::string expect;
    if (sp.dtype == "int32") {
        std::vector<std::int32_t> iv(vals.begin(), vals.end());
        expect.assign(reinterpret_cast<const char*>(iv.data()), iv.size() * 4);
    }
    else {
        expect.assign(reinterpret_cast<const char*>(vals.data()), vals.size() * 8);
    }
    EXPECT(readText(chunk) == expect);
    // the consolidated root holds the same document
    EXPECT(readText(s / "zarr.json").find("\"" + name + "\":" + expectedStaticDoc(name)) != std::string::npos);
}

std::string rootAttributes(const Path& s) {
    return multio::action::zarr::jsonMember(readText(s / "zarr.json"), "attributes");
}

std::vector<std::string> consolidatedNames(const Path& s) {
    namespace z = multio::action::zarr;
    std::vector<std::string> names;
    for (const auto& m : z::jsonObjectMembers(z::jsonMember(z::jsonMember(readText(s / "zarr.json"), "consolidated_metadata"), "metadata"))) {
        names.push_back(m.first);
    }
    return names;
}

const std::vector<std::string> ALL_STATICS = {"x", "y", "latitude", "longitude", "crs", "hyai", "hybi"};

CASE("static arrays before the fields (no coordinates-from)") {
    const std::string domain = "STATDOM";
    const Path out = absolute(outRoot()) / "static-before";
    removeAll(out);
    registerDomain(domain);
    {
        auto env = MultioTestEnvironment(planJson(out.string(), "chunk-per-level", ""));
        sendStatics(env, domain, ALL_STATICS);  // creates the store
        const Path s = out / (domain + ".zarr");
        // consolidated root rewritten right after each static array, root attributes replaced
        EXPECT(consolidatedNames(s) == ALL_STATICS);
        EXPECT_EQUAL(rootAttributes(s), GROUP_ATTRS);
        runSteps(env, domain, "end-of-simulation");
    }
    const Path s = out / (domain + ".zarr");
    for (const auto& n : ALL_STATICS) {
        checkStaticFiles(s, n);
    }
    EXPECT_EQUAL(fileSize(s / "latitude" / "c" / "0" / "0"), static_cast<std::uintmax_t>(NPTS * 8));
    EXPECT_EQUAL(fileSize(s / "crs" / "c"), 4u);
    EXPECT_EQUAL(rootAttributes(s), GROUP_ATTRS);
    auto names = consolidatedNames(s);
    EXPECT(names.size() > ALL_STATICS.size()
           && std::vector<std::string>(names.begin(), names.begin() + ALL_STATICS.size()) == ALL_STATICS);
    const std::string tdoc = readText(s / "S_TEMPERATURE" / "zarr.json");
    EXPECT(tdoc.find("\"grid_mapping\":\"crs\",\"coordinates\":\"latitude longitude\"") != std::string::npos);
    EXPECT(tdoc.find("\"shape\":[3,3,4,5]") != std::string::npos);
    checkLevelChunk(readFloats(s / "S_TEMPERATURE" / "c" / "2" / "2" / "0" / "0"), 1, 96, 3);
}

CASE("static arrays after the fields, repeated, after finalisation") {
    const std::string domain = "STATLATE";
    const Path out = absolute(outRoot()) / "static-after";
    removeAll(out);
    registerDomain(domain);
    const Path s = out / (domain + ".zarr");
    {
        auto env = MultioTestEnvironment(planJson(out.string(), "shard-levels", ""));
        for (std::size_t t = 0; t < STEPS.size(); ++t) {
            const auto step = STEPS[t];
            for (const auto& v : VARS) {
                if (t < v.firstTime) {
                    continue;
                }
                for (auto lev : v.levels.empty() ? std::vector<std::int64_t>{0} : v.levels) {
                    for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                        env.process(fieldMessage(domain, v, step, lev, c));
                    }
                }
            }
            if (t == 1) {
                // x, y, crs between the fields and the flush of step 48; latitude only: no grid_mapping yet
                sendStatics(env, domain, {"x", "y", "crs", "latitude"});
                EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("grid_mapping") == std::string::npos);
                EXPECT(readText(s / "zarr.json").find("\"grid_mapping\"") == std::string::npos);
                // longitude: published arrays get grid_mapping / coordinates right away
                sendStatics(env, domain, {"longitude"});
                EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("\"grid_mapping\":\"crs\"")
                       != std::string::npos);
                EXPECT(readText(s / "zarr.json").find("\"grid_mapping\":\"crs\"") != std::string::npos);
                // a repeated static array is overwritten (idempotent), different values replace
                sendStatics(env, domain, {"x"}, 1);
                sendStatics(env, domain, {"x"});
            }
            const std::string kind = (t + 1 == STEPS.size()) ? "end-of-simulation" : "step";
            for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                env.process(flushMessage(domain, step, kind, c));
            }
            if (t == 0) {
                EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("grid_mapping") == std::string::npos);
                EXPECT(rootAttributes(s).find("MultIO zarr output") != std::string::npos);
            }
        }
        // after the end-of-simulation flush: written, metadata rewritten (with a warning)
        sendStatics(env, domain, {"hyai", "hybi"});
        EXPECT(readText(s / "S_TEMPERATURE" / "zarr.json").find("\"shape\":[3,3,4,5]") != std::string::npos);
        // root attributes without forecast_reference_time: it is appended
        env.process(staticMessage(domain, "x", 0, R"({"Conventions":"CF-1.8","title":"late"})"));
        EXPECT_EQUAL(rootAttributes(s),
                     std::string(R"({"Conventions":"CF-1.8","title":"late","forecast_reference_time":")") + FRT + "\"}");
        env.process(staticMessage(domain, "x", 0, GROUP_ATTRS_SENT));
    }
    for (const auto& n : ALL_STATICS) {
        checkStaticFiles(s, n);
    }
    EXPECT_EQUAL(rootAttributes(s), GROUP_ATTRS);
    auto names = consolidatedNames(s);
    EXPECT_EQUAL(std::count(names.begin(), names.end(), std::string("x")), 1);
    EXPECT(std::find(names.begin(), names.end(), "hybi") != names.end());
    const std::string tdoc = readText(s / "S_TEMPERATURE" / "zarr.json");
    EXPECT(tdoc.find("\"grid_mapping\":\"crs\"") != std::string::npos);
    EXPECT(tdoc.find("\"shape\":[3,3,4,5]") != std::string::npos);
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=96 time_index=2 inc=0002:00:00\n"));
}

CASE("static arrays replace arrays copied by coordinates-from") {
    const std::string domain = "STATCOPY";
    const Path root = absolute(outRoot());
    const Path out = root / "static-replace";
    const Path coords = root / "coords-static";
    removeAll(out);
    writeCoordinateStore(coords, domain);
    registerDomain(domain);
    const Path s = out / (domain + ".zarr");
    {
        auto env = MultioTestEnvironment(planJson(out.string(), "chunk-per-level", coords.string()));
        runSteps(env, domain, "step");
        // copied: x, y, latitude, longitude, crs and the copy's root attributes
        EXPECT(rootAttributes(s).find("selftest coordinates") != std::string::npos);
        EXPECT(consolidatedNames(s)[0] == "x");
        // a stale chunk of the copied array must not survive the replacement
        writeText(s / "latitude" / "c" / "9" / "9", "stale");
        sendStatics(env, domain, {"latitude", "x"});
        EXPECT(!exists(s / "latitude" / "c" / "9" / "9"));
        env.process(flushMessage(domain, 96, "end-of-simulation", 0));
        env.process(flushMessage(domain, 96, "end-of-simulation", 1));
    }
    checkStaticFiles(s, "x");
    checkStaticFiles(s, "latitude");
    EXPECT_EQUAL(rootAttributes(s), GROUP_ATTRS);  // replaced, dx_m of the copy gone
    // replaced in place: order of the copy kept, no duplicates; y, longitude, crs still the copied ones
    auto names = consolidatedNames(s);
    EXPECT(std::vector<std::string>(names.begin(), names.begin() + 5)
           == (std::vector<std::string>{"x", "y", "latitude", "longitude", "crs"}));
    EXPECT_EQUAL(std::count(names.begin(), names.end(), std::string("latitude")), 1);
    EXPECT_EQUAL(readText(s / "y" / "c" / "0").size(), static_cast<std::size_t>(NY * 8));
    EXPECT(readText(s / "crs" / "zarr.json").find("\"earth_radius\":6.3712290000000000E+006") != std::string::npos);
    EXPECT(readText(s / "S_TEMPERATURE" / "zarr.json").find("\"grid_mapping\":\"crs\"") != std::string::npos);
}

CASE("static array errors") {
    const Path out = absolute(outRoot()) / "static-errors";
    removeAll(out);
    const std::string domain = "STATERR";
    auto plan = std::string(R"({"name": "z", "actions": [{"type": "zarr", "path": ")") + out.string()
              + R"(", "layout": "chunk-per-level"}]})";
    auto env = MultioTestEnvironment(plan);
    using Tweak = std::function<void(Metadata&)>;
    auto msg = [&](const std::string& name, const Tweak& t) { return staticMessage(domain, name, 0, "", t); };
    const auto lat = staticValues("latitude");

    EXPECT_NO_THROW(env.process(staticMessage(domain, "x")));
    EXPECT_NO_THROW(env.process(msg("crs", [](Metadata& md) { md.set("zarr-shape", std::string(" ")); })));
    // shape product != payload count; misc-globalSize != shape product
    EXPECT_THROWS(env.process(staticMessage(domain, "latitude", "y,x", "4,4", "float64", LAT_ATTRS, lat)));
    EXPECT_THROWS(env.process(msg("latitude", [](Metadata& md) { md.set("misc-globalSize", std::int64_t{19}); })));
    // number of dimension names != rank
    EXPECT_THROWS(env.process(staticMessage(domain, "latitude", "y", "4,5", "float64", LAT_ATTRS, lat)));
    EXPECT_THROWS(env.process(staticMessage(domain, "crs", "x", "", "int32", CRS_ATTRS, {0.0})));
    // x / y must match the store's nx / ny
    EXPECT_THROWS(env.process(staticMessage(domain, "latitude", "x,y", "4,5", "float64", LAT_ATTRS, lat)));
    EXPECT_THROWS(env.process(staticMessage(domain, "latitude", "y,x", "5,4", "float64", LAT_ATTRS, lat)));
    EXPECT_THROWS(env.process(staticMessage(domain, "x", "x", "4", "float64", X_ATTRS, std::vector<double>(4, 0.0))));
    // reserved names, unknown dtype, non-integral int32, malformed JSON / shape
    EXPECT_THROWS(env.process(staticMessage(domain, "time", "nt", "1", "float64", "{}", {0.0})));
    EXPECT_THROWS(env.process(staticMessage(domain, "level_S", "level_S", "1", "float64", "{}", {0.0})));
    EXPECT_THROWS(env.process(staticMessage(domain, "x", "x", "5", "float32", X_ATTRS, staticValues("x"))));
    EXPECT_THROWS(env.process(staticMessage(domain, "crs", "", "", "int32", CRS_ATTRS, {0.5})));
    EXPECT_THROWS(env.process(staticMessage(domain, "crs", "", "", "int32", "{\"a\":1", {0.0})));
    EXPECT_THROWS(env.process(staticMessage(domain, "crs", "", "", "int32", "[1]", {0.0})));
    EXPECT_THROWS(env.process(msg("x", [](Metadata& md) { md.set("zarr-group-attributes", std::string("{x}")); })));
    EXPECT_THROWS(env.process(msg("x", [](Metadata& md) { md.set("zarr-shape", std::string("5a")); })));
    // geometry of the message different from the store's
    EXPECT_THROWS(env.process(msg("x", [](Metadata& md) { md.set("geometry-ny", std::int64_t{5}); })));
    // an existing time-dependent array; and a field named like a static array
    Metadata fmd;
    fmd.set("domain", domain);
    fmd.set("misc-precision", std::string("double"));
    fmd.set("levtype", std::string("SURF"));
    fmd.set("suffix", std::string("TEMPERATURE"));
    fmd.set("step", std::int64_t{0});
    fmd.set("geometry-nx", NX);
    fmd.set("geometry-ny", NY);
    std::vector<double> f(NPTS, 1.0);
    EXPECT_NO_THROW(env.process(Message{Message::Header{Message::Tag::Field, SERVER, SERVER, Metadata(fmd)},
                                        eckit::Buffer(f.data(), f.size() * sizeof(double))}));
    EXPECT_THROWS(env.process(staticMessage(domain, "SURFTEMPERATURE", "y,x", "4,5", "float64", "{}", lat)));
    // failed messages wrote nothing: only x, crs and the field's directory
    const Path s = out / (domain + ".zarr");
    EXPECT(consolidatedNames(s) == (std::vector<std::string>{"x", "crs"}));
    EXPECT(!exists(s / "latitude"));
    EXPECT(!exists(s / "time" / "zarr.json"));
    checkStaticFiles(s, "x");
    checkStaticFiles(s, "crs");
}

CASE("static arrays of the reference store, byte-identical (MULTIO_ZARR_REFERENCE_STORE)") {
    // Optional: re-send the static arrays and root attributes of an IO-server store and compare the files byte by
    // byte, e.g. MULTIO_ZARR_REFERENCE_STORE=.../coords/zarr_out_accord/PFDEODDOMAIN.zarr
    const char* refEnv = std::getenv("MULTIO_ZARR_REFERENCE_STORE");
    if (refEnv == nullptr || !exists(Path(refEnv) / "zarr.json")) {
        std::cout << "MULTIO_ZARR_REFERENCE_STORE not set or not a store: skipped" << std::endl;
        return;
    }
    namespace z = multio::action::zarr;
    const Path ref(refEnv);
    const std::string refRoot = readText(ref / "zarr.json");
    const std::string refAttrs = z::jsonMember(refRoot, "attributes");
    const std::string frt = z::jsonUnquote(z::jsonMember(refAttrs, "forecast_reference_time"));
    const std::int64_t nx = std::stoll(z::jsonArrayElements(z::jsonMember(readText(ref / "x" / "zarr.json"), "shape"))[0]);
    const std::int64_t ny = std::stoll(z::jsonArrayElements(z::jsonMember(readText(ref / "y" / "zarr.json"), "shape"))[0]);
    const Path out = absolute(outRoot()) / "static-reference";
    removeAll(out);
    const std::string domain = "REFDOM";
    auto plan = std::string(R"({"name": "z", "actions": [{"type": "select", "match": [{"category": ["ial-static"]}]}, {"type": "aggregate"}, {"type": "zarr", "path": ")")
              + out.string() + R"(", "layout": "chunk-per-level"}]})";
    {
        auto env = MultioTestEnvironment(plan);
        for (const auto& name : ALL_STATICS) {
            const std::string doc = readText(ref / name / "zarr.json");
            std::vector<std::string> dims;
            if (const std::string dn = z::jsonMember(doc, "dimension_names"); !dn.empty()) {  // none for crs
                for (const auto& d : z::jsonArrayElements(dn)) {
                    dims.push_back(z::jsonUnquote(d));
                }
            }
            std::vector<std::string> shape = z::jsonArrayElements(z::jsonMember(doc, "shape"));
            auto join = [](const std::vector<std::string>& v) {
                std::string r;
                for (std::size_t i = 0; i < v.size(); ++i) {
                    r += (i ? "," : "") + v[i];
                }
                return r;
            };
            const std::string dtype = z::jsonUnquote(z::jsonMember(doc, "data_type"));
            const std::string chunk = "c" + std::string(shape.size() == 2 ? "/0/0" : shape.size() == 1 ? "/0" : "");
            const std::string raw = readText(ref / name / chunk);
            std::vector<double> vals;
            if (dtype == "int32") {
                for (std::size_t i = 0; i + 4 <= raw.size(); i += 4) {
                    std::int32_t v;
                    std::memcpy(&v, raw.data() + i, 4);
                    vals.push_back(v);
                }
            }
            else {
                vals.resize(raw.size() / 8);
                std::memcpy(vals.data(), raw.data(), vals.size() * 8);
            }
            env.process(staticMessage(domain, name, join(dims), join(shape), dtype, z::jsonMember(doc, "attributes"),
                                      vals, name == "x" ? refAttrs : "", [&](Metadata& md) {
                                          md.set("geometry-nx", nx);
                                          md.set("geometry-ny", ny);
                                          md.set("forecastReferenceTime", frt);
                                      }));
            EXPECT_EQUAL(readText(out / (domain + ".zarr") / name / "zarr.json"), doc);
            EXPECT(readText(out / (domain + ".zarr") / name / chunk) == raw);
            std::cout << "reference static array " << name << ": zarr.json and " << chunk << " ("
                      << raw.size() << " bytes) identical" << std::endl;
        }
    }
    const Path s = out / (domain + ".zarr");
    EXPECT_EQUAL(rootAttributes(s), refAttrs);
    EXPECT(consolidatedNames(s) == ALL_STATICS);
    // the static part of the consolidated metadata equals the reference's
    const std::string ours = z::jsonMember(z::jsonMember(readText(s / "zarr.json"), "consolidated_metadata"), "metadata");
    const std::string theirs = z::jsonMember(z::jsonMember(refRoot, "consolidated_metadata"), "metadata");
    for (const auto& name : ALL_STATICS) {
        EXPECT_EQUAL(z::jsonMember(ours, name), z::jsonMember(theirs, name));
    }
}

//----------------------------------------------------------------------------------------------------------------------
// zstd compression

/// Relative paths of the regular files below dir, sorted
std::vector<std::string> fileList(const Path& dir) {
    std::vector<std::string> out;
    const std::size_t n = dir.string().size() + 1;
    walk(dir, [&](const Path& p, bool regular) {
        if (regular) {
            out.push_back(p.string().substr(n));
        }
    });
    std::sort(out.begin(), out.end());
    return out;
}

/// The two trees have the same files with the same bytes
bool sameTree(const Path& a, const Path& b) {
    const auto fa = fileList(a);
    if (fa != fileList(b)) {
        return false;
    }
    for (const auto& f : fa) {
        if (readText(a / f) != readText(b / f)) {
            std::cout << "differs: " << f << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<float> decodeZstd(const std::string& b) {
    auto raw = multio::action::zarr::decompressZstd(b.data(), b.size());
    std::vector<float> v(raw.size() / sizeof(float));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
    return v;
}

const std::string ZSTD_CODECS
    = R"("codecs":[{"name":"bytes","configuration":{"endian":"little"}},{"name":"zstd","configuration":{"level":5,"checksum":false}}])";

/// The synthetic run of runLayout with zstd level 5 and `threads` workers into <selftest>/<sub>
Path runZstd(const std::string& layout, int threads, const std::string& sub, int queue = -1) {
    const std::string domain = "TESTDOM";
    const Path root = absolute(outRoot());
    const Path out = root / sub;
    const Path coords = root / "coords";
    removeAll(out);
    writeCoordinateStore(coords, domain);
    registerDomain(domain);
    std::string extra = R"(, "compression": {"codec": "zstd", "level": 5}, "compression-threads": )"
                      + std::to_string(threads);
    if (queue >= 0) {
        extra += R"(, "compression-queue": )" + std::to_string(queue);
    }
    {
        auto env = MultioTestEnvironment(planJson(out.string(), layout, coords.string(), extra));
        runSteps(env, domain, "end-of-simulation");
    }
    return out / (domain + ".zarr");
}

CASE("zstd compression, chunk-per-level: inline and worker threads give the same store") {
    if (!multio::action::zarr::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    const Path s0 = runZstd("chunk-per-level", 0, "zstd-chunk-per-level-t0");
    const Path s4 = runZstd("chunk-per-level", 4, "zstd-chunk-per-level-t4");
    const Path s1 = runZstd("chunk-per-level", 1, "zstd-chunk-per-level-t1q1", 1);
    EXPECT(sameTree(s0, s4));
    EXPECT(sameTree(s0, s1));
    EXPECT_EQUAL(countStaging(s4), 0u);
    EXPECT_EQUAL(readText(s4 / ".last_step"), std::string("step=96 time_index=2 inc=0002:00:00\n"));

    // codecs of the time-dependent arrays (also in the consolidated metadata); static arrays uncompressed
    for (const std::string name : {"S_TEMPERATURE", "SURFTEMPERATURE", "KT_ISOT_ALTIT"}) {
        EXPECT(readText(s4 / name / "zarr.json").find(ZSTD_CODECS) != std::string::npos);
    }
    EXPECT(readText(s4 / "time" / "zarr.json").find("zstd") == std::string::npos);
    EXPECT(readText(s4 / "level_S" / "zarr.json").find("zstd") == std::string::npos);
    EXPECT(readText(s4 / "x" / "zarr.json").find("zstd") == std::string::npos);
    const std::string rootDoc = readText(s4 / "zarr.json");
    std::size_t nzstd = 0;
    for (std::size_t p = rootDoc.find("\"zstd\""); p != std::string::npos; p = rootDoc.find("\"zstd\"", p + 1)) {
        ++nzstd;
    }
    EXPECT_EQUAL(nzstd, VARS.size());

    // the chunks are zstd frames of the raw values (NaN of SURFTEMPERATURE included)
    checkLevelChunk(decodeZstd(readText(s4 / "S_TEMPERATURE" / "c" / "0" / "0" / "0" / "0")), 1, 0, 1);
    checkLevelChunk(decodeZstd(readText(s4 / "S_TEMPERATURE" / "c" / "2" / "2" / "0" / "0")), 1, 96, 3);
    checkLevelChunk(decodeZstd(readText(s4 / "CLSTEMPERATURE" / "c" / "1" / "0" / "0")), 5, 48, 0);
    const auto surf = decodeZstd(readText(s4 / "SURFTEMPERATURE" / "c" / "2" / "0" / "0"));
    checkLevelChunk(surf, 4, 96, 0);
    EXPECT(std::isnan(surf[7]));
    EXPECT(!exists(s4 / "KT_ISOT_ALTIT" / "c" / "0" / "1" / "0" / "0"));
}

CASE("zstd compression, shard-levels: compressed inner chunks, index of compressed sizes") {
    if (!multio::action::zarr::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    const Path s0 = runZstd("shard-levels", 0, "zstd-shard-levels-t0");
    const Path s4 = runZstd("shard-levels", 4, "zstd-shard-levels-t4");
    EXPECT(sameTree(s0, s4));
    EXPECT_EQUAL(countStaging(s4), 0u);

    const std::string tdoc = readText(s4 / "S_TEMPERATURE" / "zarr.json");
    EXPECT(tdoc.find(R"("name":"sharding_indexed")") != std::string::npos);
    EXPECT(tdoc.find(R"("configuration":{"chunk_shape":[1,1,4,5],)" + ZSTD_CODECS
                     + R"(,"index_codecs":[{"name":"bytes","configuration":{"endian":"little"}}],"index_location":"end"})")
           != std::string::npos);
    EXPECT(readText(s4 / "SURFTEMPERATURE" / "zarr.json").find(ZSTD_CODECS) != std::string::npos);

    // shard [frames][index nlev x (offset, nbytes)]: frames back to back, each one decodes to its level
    auto checkShard = [&](const Path& shard, int id, std::int64_t step, const std::vector<std::int64_t>& axis,
                          const std::vector<std::int64_t>& present) {
        const std::string b = readText(shard);
        const std::size_t nlev = axis.size();
        EXPECT(b.size() > nlev * 16);
        std::vector<std::uint64_t> idx(2 * nlev);
        std::memcpy(idx.data(), b.data() + b.size() - nlev * 16, nlev * 16);
        std::uint64_t off = 0;
        for (std::size_t k = 0; k < nlev; ++k) {
            const bool has = std::find(present.begin(), present.end(), axis[k]) != present.end();
            if (!has) {
                EXPECT(idx[2 * k] == std::numeric_limits<std::uint64_t>::max());
                EXPECT(idx[2 * k + 1] == std::numeric_limits<std::uint64_t>::max());
                continue;
            }
            EXPECT_EQUAL(idx[2 * k], off);
            EXPECT(idx[2 * k + 1] > 0 && idx[2 * k + 1] != static_cast<std::uint64_t>(NPTS * 4));
            checkLevelChunk(decodeZstd(b.substr(idx[2 * k], idx[2 * k + 1])), id, step, axis[k]);
            off += idx[2 * k + 1];
        }
        EXPECT_EQUAL(off + nlev * 16, static_cast<std::uint64_t>(b.size()));
    };
    checkShard(s4 / "S_TEMPERATURE" / "c" / "1" / "0" / "0" / "0", 1, 48, {1, 2, 3}, {1, 2, 3});
    checkShard(s4 / "KT_ISOT_ALTIT" / "c" / "0" / "0" / "0" / "0", 3, 0, {1, 2}, {1});
    checkShard(s4 / "S_HUMI.SPECIFI" / "c" / "2" / "0" / "0" / "0", 6, 96, {1, 2, 3}, {1, 3});
    const auto surf = decodeZstd(readText(s4 / "SURFTEMPERATURE" / "c" / "0" / "0" / "0"));
    checkLevelChunk(surf, 4, 0, 0);
    EXPECT(std::isnan(surf[7]));
}

/// Many larger fields through ZarrStore directly: the worker pool (small queue, several threads) writes the same
/// bytes as inline compression, in both layouts
CASE("zstd compression: worker pool under load, ZarrStore directly") {
    namespace z = multio::action::zarr;
    if (!z::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    const std::int64_t nx = 211;
    const std::int64_t ny = 173;
    auto field = [&](int var, std::int64_t step, std::int64_t lev) {
        std::vector<float> v(static_cast<std::size_t>(nx * ny));
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double x = static_cast<double>(i % nx) / nx;
            const double y = static_cast<double>(i / nx) / ny;
            v[i] = static_cast<float>(250.0 + var + 10.0 * std::sin(6.0 * x + lev) * std::cos(4.0 * y + step)
                                      + 1e-3 * static_cast<double>((i * 2654435761u) % 1000));
        }
        if (var == 0) {
            v[3] = std::numeric_limits<float>::quiet_NaN();
        }
        return v;
    };
    for (const auto layout : {z::Layout::ChunkPerLevel, z::Layout::ShardLevels}) {
        const std::string ln = layout == z::Layout::ShardLevels ? "shard" : "chunk";
        std::vector<Path> stores;
        for (int threads : {0, 3}) {
            const Path dir = absolute(outRoot()) / ("zstd-load-" + ln + "-t" + std::to_string(threads));
            removeAll(dir);
            std::unique_ptr<z::WorkerPool> pool;
            if (threads > 0) {
                pool = std::make_unique<z::WorkerPool>(threads, 2);
            }
            z::StoreSetup setup;
            setup.name = "LOAD";
            setup.dir = dir.string();
            setup.layout = layout;
            setup.nx = nx;
            setup.ny = ny;
            setup.forecastReferenceTime = FRT;
            setup.compression = z::Compression{z::Codec::Zstd, 1};
            setup.pool = pool.get();
            {
                z::ZarrStore st(setup);
                EXPECT(st.encodesAsync() == (threads > 0));
                for (std::int64_t step = 0; step < 3; ++step) {
                    for (int var = 0; var < 6; ++var) {
                        z::FieldDescription fd;
                        fd.prefix = var < 4 ? "S" : "SURF";
                        fd.suffix = "VAR" + std::to_string(var);
                        fd.is3d = var < 4;
                        fd.step = step;
                        fd.leadTime = 3600.0 * step;
                        for (std::int64_t lev = 1; lev <= (fd.is3d ? 12 : 1); ++lev) {
                            fd.level = fd.is3d ? lev : 0;
                            if (var % 2) {
                                st.addField(fd, field(var, step, lev));  // moved
                            }
                            else {
                                auto v = field(var, step, lev);
                                st.addField(fd, v.data(), v.size());  // copied
                            }
                        }
                    }
                    st.flush(step, std::nullopt);
                }
                st.finalize();
            }
            stores.push_back(dir / "LOAD.zarr");
        }
        EXPECT(sameTree(stores[0], stores[1]));
        EXPECT_EQUAL(countStaging(stores[1]), 0u);
    }
}

/// A failing compression job (here: its chunk directory cannot be created) is reported by the flush of its
/// step, not lost; the other steps are not affected
CASE("shard-levels: complete shards are written before the flush") {
    namespace z = multio::action::zarr;
    const std::int64_t nx = 37;
    const std::int64_t ny = 29;
    const std::int64_t nlev = 8;
    auto field = [&](int var, std::int64_t step, std::int64_t lev) {
        std::vector<float> v(static_cast<std::size_t>(nx * ny));
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<float>(100.0 * var + step + 0.01 * lev + 1e-4 * static_cast<double>(i));
        }
        return v;
    };
    // Variable S_FULL has every level of the axis; S_PART misses the top level from step 1 on, so its shards wait
    // for the flush. A repeated field for a complete shard is ignored.
    auto shard = [](const Path& store, const std::string& var, std::int64_t t) {
        return store / var / "c" / std::to_string(t) / "0" / "0" / "0";
    };
    auto waitFor = [](const Path& p) {
        for (int i = 0; i < 1000 && !exists(p); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return exists(p);
    };
    struct Variant {
        std::string name;
        bool zstd;
        int threads;
    };
    std::vector<Variant> variants{{"raw", false, 0}, {"zstd-inline", true, 0}, {"zstd-t3", true, 3}};
    std::vector<Path> stores;
    for (const auto& var : variants) {
        if (var.zstd && !z::zstdAvailable()) {
            continue;
        }
        const Path dir = absolute(outRoot()) / ("early-shards-" + var.name);
        removeAll(dir);
        std::unique_ptr<z::WorkerPool> pool;
        if (var.threads > 0) {
            pool = std::make_unique<z::WorkerPool>(var.threads, 4);
        }
        z::StoreSetup setup;
        setup.name = "EARLY";
        setup.dir = dir.string();
        setup.layout = z::Layout::ShardLevels;
        setup.nx = nx;
        setup.ny = ny;
        setup.forecastReferenceTime = FRT;
        if (var.zstd) {
            setup.compression = z::Compression{z::Codec::Zstd, 3};
        }
        setup.pool = pool.get();
        const Path store = dir / "EARLY.zarr";
        {
            z::ZarrStore st(setup);
            for (std::int64_t step = 0; step < 3; ++step) {
                z::FieldDescription fd;
                fd.prefix = "S";
                fd.is3d = true;
                fd.step = step;
                fd.leadTime = 3600.0 * step;
                for (int v = 0; v < 2; ++v) {
                    fd.suffix = v == 0 ? "FULL" : "PART";
                    const std::int64_t top = (v == 1 && step > 0) ? nlev - 1 : nlev;
                    for (std::int64_t lev = 1; lev <= top; ++lev) {
                        fd.level = lev;
                        st.addField(fd, field(v, step, lev));
                    }
                }
                if (step == 0) {
                    // level axis not known yet: nothing before the flush
                    EXPECT(!exists(shard(store, "S_FULL", 0)));
                }
                else {
                    EXPECT(waitFor(shard(store, "S_FULL", step)));
                    EXPECT(!exists(shard(store, "S_PART", step)));
                    if (step == 2) {
                        fd.suffix = "FULL";
                        fd.level = 3;
                        st.addField(fd, field(7, step, 3));  // repeated after completion: ignored
                    }
                }
                st.flush(step, std::nullopt);
                EXPECT(exists(shard(store, "S_FULL", step)));
                EXPECT(exists(shard(store, "S_PART", step)));
            }
            st.finalize();
        }
        stores.push_back(store);
    }
    // same bytes with the encoding inline or on workers
    if (stores.size() == 3) {
        EXPECT(sameTree(stores[1], stores[2]));
    }
    // the shard of step 2 holds the first S_FULL level 3, not the repeated one: compare with step 1's layout
    // through the raw store (uncompressed inner chunks at fixed offsets)
    {
        const Path p = shard(stores[0], "S_FULL", 2);
        std::ifstream in(p.string(), std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::size_t chunk = static_cast<std::size_t>(nx * ny) * sizeof(float);
        EXPECT_EQUAL(bytes.size(), static_cast<std::size_t>(nlev) * chunk + static_cast<std::size_t>(nlev) * 16);
        float first;
        std::memcpy(&first, bytes.data() + 2 * chunk, sizeof(float));  // level 3 = index 2
        EXPECT_EQUAL(first, field(0, 2, 3)[0]);
    }
}

CASE("zstd compression: an error in a worker is rethrown by the flush") {
    if (!multio::action::zarr::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    if (::geteuid() == 0) {
        std::cout << "running as root, skipped" << std::endl;
        return;
    }
    const std::string domain = "TESTDOM";
    for (int threads : {2, 0}) {
        const Path out = absolute(outRoot()) / ("zstd-worker-error-t" + std::to_string(threads));
        removeAll(out);
        registerDomain(domain);
        const Path cdir = out / (domain + ".zarr") / "SURFTEMPERATURE" / "c";
        const Var& surf = VARS[3];
        bool threwAtField = false;
        bool threwAtFlush = false;
        {
            auto env = MultioTestEnvironment(planJson(
                out.string(), "chunk-per-level", "",
                R"(, "compression": {"codec": "zstd"}, "compression-threads": )" + std::to_string(threads)));
            for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                env.process(fieldMessage(domain, surf, 0, 0, c));
            }
            for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                env.process(flushMessage(domain, 0, "step", c));
            }
            ::chmod(cdir.string().c_str(), 0555);  // c/1 cannot be created
            try {
                for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                    env.process(fieldMessage(domain, surf, 48, 0, c));
                }
            }
            catch (const std::exception&) {
                threwAtField = true;
            }
            try {
                for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
                    env.process(flushMessage(domain, 48, "step", c));
                }
            }
            catch (const std::exception& e) {
                std::cout << "flush: " << e.what() << std::endl;
                threwAtFlush = true;
            }
            ::chmod(cdir.string().c_str(), 0755);
        }  // destructor: the failed step is still open and discarded
        const Path s = out / (domain + ".zarr");
        EXPECT_EQUAL(countStaging(s), 0u);
        if (threads == 0) {
            EXPECT(threwAtField);  // inline: the field itself fails, as without compression
            continue;
        }
        EXPECT(!threwAtField);
        EXPECT(threwAtFlush);
        // the failed step stayed open and was discarded by the destructor
        EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=0 time_index=0 inc=0000:00:00\n"));
        EXPECT(readText(s / "SURFTEMPERATURE" / "zarr.json").find("\"shape\":[1,4,5]") != std::string::npos);
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Quantization before zstd

/// Independent reference of bitround for a normal finite float: x = m 2^e with m in [0.5, 1); keeping k mantissa
/// bits after the implicit one is rounding m to a multiple of 2^-(k+1), ties to even (double arithmetic)
float bitRoundReference(float x, int k) {
    int e = 0;
    const double m = std::frexp(static_cast<double>(x), &e);
    return static_cast<float>(std::ldexp(std::nearbyint(std::ldexp(m, k + 1)), e - k - 1));
}

std::uint32_t bitsOf(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

CASE("quantize(): bitround, absolute, range-bits on edge values") {
    namespace z = multio::action::zarr;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float fmax = std::numeric_limits<float>::max();
    const float fmin = std::numeric_limits<float>::min();
    const float denorm = std::numeric_limits<float>::denorm_min();

    std::vector<float> base = {0.0f,   -0.0f,  1.0f, -1.0f, 1.5f, 0.1f,   -0.1f,      3.14159265f, 287.4567f, -273.15f,
                               1e-30f, -1e30f, fmax, -fmax, fmin, denorm, 1.9999999f, nan,         inf,       -inf};
    for (std::uint32_t i = 0; i < 2000; ++i) {  // spread over many binades, both signs
        const double x = std::ldexp(1.0 + static_cast<double>((i * 2654435761u) % 1000003u) / 1000003.0,
                                    static_cast<int>(i % 120) - 60);
        base.push_back(static_cast<float>(i % 2 ? -x : x));
    }
    std::uint32_t nanBits = bitsOf(nan);

    // bitround
    for (int k = 1; k <= 23; ++k) {
        z::Quantization q;
        q.mode = z::QuantMode::BitRound;
        q.bits = k;
        std::vector<float> v = base;
        z::quantize(q, v.data(), v.size());
        const std::uint32_t low = (std::uint32_t{1} << (23 - k)) - 1u;
        for (std::size_t i = 0; i < v.size(); ++i) {
            const float x = base[i];
            if (std::isnan(x)) {
                EXPECT_EQUAL(bitsOf(v[i]), nanBits);
                continue;
            }
            if (std::isinf(x)) {
                EXPECT(v[i] == x);
                continue;
            }
            EXPECT(std::isfinite(v[i]));             // no overflow to inf (fmax rounds down instead)
            EXPECT_EQUAL((bitsOf(v[i]) & low), 0u);  // trailing mantissa bits zero
            EXPECT(std::signbit(v[i]) == std::signbit(x));
            if (std::fabs(x) >= fmin && std::fabs(x) < fmax / 2) {
                EXPECT(sameBits(v[i], bitRoundReference(x, k)));
                EXPECT(std::fabs(static_cast<double>(v[i]) - x) <= std::ldexp(std::fabs(x), -(k + 1)));
            }
        }
        std::vector<float> w = v;  // idempotent
        z::quantize(q, w.data(), w.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            EXPECT_EQUAL(bitsOf(w[i]), bitsOf(v[i]));
        }
    }
    {  // ties to even: 1 + 2^-23 * (2^22 + odd) patterns on 1.0 .. 2.0 with k = 1: 1.25 -> 1.0, 1.75 -> 2.0
        z::Quantization q;
        q.mode = z::QuantMode::BitRound;
        q.bits = 1;
        std::vector<float> v = {1.25f, 1.75f, -1.25f, 1.24f};
        z::quantize(q, v.data(), v.size());
        EXPECT(v[0] == 1.0f && v[1] == 2.0f && v[2] == -1.0f && v[3] == 1.0f);
    }
    // keep-bits 23 is the identity
    {
        z::Quantization q;
        q.mode = z::QuantMode::BitRound;
        q.bits = 23;
        std::vector<float> v = base;
        z::quantize(q, v.data(), v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            EXPECT_EQUAL(bitsOf(v[i]), bitsOf(base[i]));
        }
    }

    // powers of two
    EXPECT(z::powerOfTwoFloor(0.01) == 0.0078125);
    EXPECT(z::powerOfTwoFloor(1.0) == 1.0);
    EXPECT(z::powerOfTwoFloor(0.75) == 0.5);
    EXPECT(z::powerOfTwoFloor(3000.0) == 2048.0);

    // absolute
    for (double step : {0.125, 1.0, 1024.0, std::ldexp(1.0, -40), std::ldexp(1.0, -126), std::ldexp(1.0, 100)}) {
        z::Quantization q;
        q.mode = z::QuantMode::Absolute;
        q.step = step;
        std::vector<float> v = base;
        z::quantize(q, v.data(), v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            const float x = base[i];
            if (std::isnan(x)) {
                EXPECT_EQUAL(bitsOf(v[i]), nanBits);
                continue;
            }
            if (std::isinf(x)) {
                EXPECT(v[i] == x);
                continue;
            }
            EXPECT(std::isfinite(v[i]));
            const double err = std::fabs(static_cast<double>(v[i]) - x);
            EXPECT(err <= step / 2);
            const double ref = std::fabs(x) < std::ldexp(step, 24) ? std::nearbyint(x / step) * step : x;
            EXPECT(static_cast<double>(v[i]) == ref);
            EXPECT(std::fmod(static_cast<double>(v[i]), step) == 0.0 || v[i] == x);
        }
    }
    {  // ties to even on the multiples: step 0.125
        z::Quantization q;
        q.mode = z::QuantMode::Absolute;
        q.step = 0.125;
        std::vector<float> v = {0.0625f, 0.1875f, -0.1875f, 0.3125f, 0.06f, -0.0625f};
        z::quantize(q, v.data(), v.size());
        EXPECT(v[0] == 0.0f && v[1] == 0.25f && v[2] == -0.25f && v[3] == 0.25f && v[4] == 0.0f && v[5] == 0.0f);
        EXPECT(std::signbit(v[5]));  // -0.0625 -> -0
    }

    // range-bits: D = largest power of two <= range / 2^n, per call (field)
    {
        z::Quantization q;
        q.mode = z::QuantMode::RangeBits;
        q.bits = 4;
        std::vector<float> v;
        for (int i = 0; i <= 76; ++i) {
            v.push_back(-3.0f + 0.1f * static_cast<float>(i));  // range ~7.6 -> D = 0.25
        }
        v.push_back(nan);
        v.push_back(-inf);  // ignored by the range
        std::vector<float> x = v;
        z::quantize(q, v.data(), v.size());
        const double range = static_cast<double>(x[76]) - x[0];
        const double d = z::powerOfTwoFloor(range / 16);
        EXPECT(d == 0.25);
        for (std::size_t i = 0; i <= 76; ++i) {
            EXPECT(std::fabs(static_cast<double>(v[i]) - x[i]) <= d / 2);
            EXPECT(std::fmod(static_cast<double>(v[i]), d) == 0.0);
        }
        EXPECT(std::isnan(v[77]) && v[78] == -inf);

        std::vector<float> c(10, 287.123f);  // constant: unchanged
        c[3] = nan;
        z::quantize(q, c.data(), c.size());
        EXPECT(c[0] == 287.123f && c[9] == 287.123f && std::isnan(c[3]));
        std::vector<float> allNan(5, nan);
        z::quantize(q, allNan.data(), allNan.size());
        EXPECT(std::isnan(allNan[0]));
    }

    // rule matching: ordered, first match wins, globs and regex, default
    z::QuantRules rules;
    z::QuantRule r1;
    r1.globs = {"S_*", "CLS?EMPERATURE"};
    r1.quantization.mode = z::QuantMode::BitRound;
    r1.quantization.bits = 7;
    z::QuantRule r2;
    r2.regexText = "^(S|P)_HUMI\\..*";
    r2.regex = std::make_shared<const std::regex>(r2.regexText);
    r2.quantization.mode = z::QuantMode::RangeBits;
    r2.quantization.bits = 16;
    rules.rules = {r1, r2};
    EXPECT(rules.lookup("D", "S_HUMI.SPECIFI").mode == z::QuantMode::BitRound);  // first match wins
    EXPECT(rules.lookup("D", "P_HUMI.SPECIFI").mode == z::QuantMode::RangeBits);
    EXPECT(rules.lookup("D", "CLSTEMPERATURE").mode == z::QuantMode::BitRound);
    EXPECT(rules.lookup("D", "XP_HUMI.SPECIFI").mode == z::QuantMode::None);  // regex matches the whole name
    EXPECT(rules.lookup("D", "SURFTEMPERATURE").mode == z::QuantMode::None);
    rules.fallback.mode = z::QuantMode::Absolute;
    rules.fallback.step = 0.5;
    EXPECT(rules.lookup("D", "SURFTEMPERATURE").mode == z::QuantMode::Absolute);

    // domains: a rule restricted to some stores is skipped for the others
    z::QuantRule r0;
    r0.domains = {"PF*", "OTHER"};
    r0.globs = {"S_*"};
    r0.quantization.mode = z::QuantMode::RangeBits;
    r0.quantization.bits = 6;
    rules.rules = {r0, r1};
    EXPECT(rules.lookup("PFDEODDOMAIN", "S_TEMPERATURE").mode == z::QuantMode::RangeBits);
    EXPECT(rules.lookup("OTHER", "S_TEMPERATURE").mode == z::QuantMode::RangeBits);
    EXPECT(rules.lookup("ICMSHDEOD", "S_TEMPERATURE").mode == z::QuantMode::BitRound);  // next rule
    EXPECT(rules.lookup("ICMSHDEOD", "SURFTEMPERATURE").mode == z::QuantMode::Absolute);  // default
    EXPECT(rules.lookup("PFDEODDOMAIN", "SURFTEMPERATURE").mode == z::QuantMode::Absolute);  // domain alone: no match
}

/// Quantize rules of the synthetic run (zstd level 5): every algorithm, a glob list, a regex, a lossless rule, a
/// default, a rule for other stores only (skipped) and one restricted to this store (store TESTDOM). Values are
/// id * 10000 + step * 100 + level + idx * 0.001 (SURFTEMPERATURE idx 7 NaN).
const std::string QUANT_RULES = R"(, "quantize": {"rules": [)"
                                R"({"domains": "OTHER*", "variables": "*", "keep-bits": 1},)"
                                R"({"domains": ["X", "TEST*"], "variables": "S_TEMPERATURE", "keep-bits": 8},)"
                                R"({"variables": ["KB_*", "KT_*"], "absolute": 0.3},)"
                                R"({"regex": "^S_HUMI\\..*", "range-bits": 1},)"
                                R"({"variables": "SURFTEMPERATURE", "keep-bits": 4},)"
                                R"({"variables": "CLS*", "lossless": true}],)"
                                R"("default": {"keep-bits": 12}})";

Path runQuant(const std::string& layout, int threads, const std::string& sub, const std::string& quant) {
    const std::string domain = "TESTDOM";
    const Path root = absolute(outRoot());
    const Path out = root / sub;
    const Path coords = root / "coords";
    removeAll(out);
    writeCoordinateStore(coords, domain);
    registerDomain(domain);
    const std::string extra = R"(, "compression": {"codec": "zstd", "level": 5}, "compression-threads": )"
                            + std::to_string(threads) + quant;
    {
        auto env = MultioTestEnvironment(planJson(out.string(), layout, coords.string(), extra));
        runSteps(env, domain, "end-of-simulation");
    }
    return out / (domain + ".zarr");
}

/// Expected error bound of QUANT_RULES for variable id at value x (0: exact)
double quantBound(int id, float x, std::int64_t step, std::int64_t level) {
    switch (id) {
        case 1:
            return std::ldexp(std::fabs(static_cast<double>(x)), -9);  // keep-bits 8
        case 2:
        case 3:
            return 0.25 / 2;  // absolute 0.3 -> D = 0.25
        case 6: {             // range-bits 1 over the field's range
            const double lo = static_cast<float>(value(6, step, level, 0));
            const double hi = static_cast<float>(value(6, step, level, NPTS - 1));
            return multio::action::zarr::powerOfTwoFloor((hi - lo) / 2) / 2;
        }
        case 4:
            return std::ldexp(std::fabs(static_cast<double>(x)), -5);  // keep-bits 4
        case 5:
            return 0.0;  // CLS*: lossless
        default:
            return std::ldexp(std::fabs(static_cast<double>(x)), -13);  // default keep-bits 12
    }
}

/// Decoded zstd chunk of (id, step, level) within the quantization bounds; returns the number of changed values
std::size_t checkQuantChunk(const std::vector<float>& vals, int id, std::int64_t step, std::int64_t level) {
    EXPECT_EQUAL(vals.size(), static_cast<std::size_t>(NPTS));
    std::size_t changed = 0;
    for (std::int64_t i = 0; i < NPTS; ++i) {
        const float x = static_cast<float>(value(id, step, level, i));
        if (std::isnan(x)) {
            EXPECT(std::isnan(vals[i]));
            continue;
        }
        EXPECT(std::isfinite(vals[i]));
        const double err = std::fabs(static_cast<double>(vals[i]) - x);
        if (err > quantBound(id, x, step, level)) {
            std::cout << "quantization error " << err << " of variable " << id << " step " << step << " level " << level
                      << " index " << i << " above " << quantBound(id, x, step, level) << std::endl;
        }
        EXPECT(err <= quantBound(id, x, step, level));
        changed += !sameBits(vals[i], x);
    }
    return changed;
}

CASE("quantize, chunk-per-level: inline and worker threads give the same bytes, error bounds hold") {
    if (!multio::action::zarr::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    const Path s0 = runQuant("chunk-per-level", 0, "quant-chunk-per-level-t0", QUANT_RULES);
    const Path s4 = runQuant("chunk-per-level", 4, "quant-chunk-per-level-t4", QUANT_RULES);
    EXPECT(sameTree(s0, s4));
    EXPECT_EQUAL(countStaging(s4), 0u);

    // CF quantization attributes and container variables (also in the consolidated metadata)
    EXPECT(readText(s4 / "S_TEMPERATURE" / "zarr.json")
               .find(R"("quantization":"quantization_bitround","quantization_nsb":8})")
           != std::string::npos);
    EXPECT(readText(s4 / "KB_ISOT_ALTIT" / "zarr.json")
               .find(R"("quantization":"quantization_absolute","quantization_step":0.25,)"
                     R"("quantization_maximum_absolute_error":0.125})")
           != std::string::npos);
    EXPECT(readText(s4 / "S_HUMI.SPECIFI" / "zarr.json")
               .find(R"("quantization":"quantization_range_bits","quantization_range_bits":1})")
           != std::string::npos);
    EXPECT(readText(s4 / "CLSTEMPERATURE" / "zarr.json").find("quantization") == std::string::npos);
    EXPECT(readText(s4 / "S_WIND.U.PHYS" / "zarr.json").find(R"("quantization_nsb":12)") != std::string::npos);
    EXPECT(readText(s4 / "quantization_bitround" / "zarr.json").find(R"("algorithm":"bitround")") != std::string::npos);
    const std::string rootDoc = readText(s4 / "zarr.json");
    for (const std::string c : {"quantization_bitround", "quantization_absolute", "quantization_range_bits"}) {
        EXPECT(rootDoc.find("\"" + c + "\":{\"zarr_format\":3") != std::string::npos);
    }

    // values within the bounds, NaN kept, quantization visible
    std::size_t changed = 0;
    for (std::size_t t = 0; t < STEPS.size(); ++t) {
        const std::int64_t step = STEPS[t];
        for (const auto& v : VARS) {
            if (t < v.firstTime) {
                continue;
            }
            const std::string name = v.id == 7                                      ? "S_WIND.U.PHYS"
                                   : v.levtype.size() == 1 || v.levtype.size() == 2 ? v.levtype + "_" + v.suffix
                                                                                    : v.levtype + v.suffix;
            if (v.levels.empty()) {
                changed += checkQuantChunk(decodeZstd(readText(s4 / name / "c" / std::to_string(t) / "0" / "0")), v.id,
                                           step, 0);
                continue;
            }
            const std::vector<std::int64_t> axis = v.levtype == "KB" || v.levtype == "KT"
                                                     ? std::vector<std::int64_t>{1, 2}
                                                     : std::vector<std::int64_t>{1, 2, 3};
            for (auto lev : v.levels) {
                const auto k = std::find(axis.begin(), axis.end(), lev) - axis.begin();
                changed += checkQuantChunk(
                    decodeZstd(readText(s4 / name / "c" / std::to_string(t) / std::to_string(k) / "0" / "0")), v.id,
                    step, lev);
            }
        }
    }
    EXPECT(changed > 0);
    // lossless rule: the CLS chunk is the plain zstd frame of the raw values
    checkLevelChunk(decodeZstd(readText(s4 / "CLSTEMPERATURE" / "c" / "1" / "0" / "0")), 5, 48, 0);
    EXPECT(std::isnan(decodeZstd(readText(s4 / "SURFTEMPERATURE" / "c" / "2" / "0" / "0"))[7]));
}

CASE("quantize, shard-levels: inline and worker threads give the same bytes, error bounds hold") {
    if (!multio::action::zarr::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    const Path s0 = runQuant("shard-levels", 0, "quant-shard-levels-t0", QUANT_RULES);
    const Path s4 = runQuant("shard-levels", 4, "quant-shard-levels-t4", QUANT_RULES);
    EXPECT(sameTree(s0, s4));
    EXPECT(readText(s4 / "S_TEMPERATURE" / "zarr.json").find(R"("quantization_nsb":8)") != std::string::npos);

    // shard of S_TEMPERATURE at step 48: three frames, each within the bitround bound
    const std::string b = readText(s4 / "S_TEMPERATURE" / "c" / "1" / "0" / "0" / "0");
    std::vector<std::uint64_t> idx(6);
    std::memcpy(idx.data(), b.data() + b.size() - 48, 48);
    std::size_t changed = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        changed += checkQuantChunk(decodeZstd(b.substr(idx[2 * k], idx[2 * k + 1])), 1, 48,
                                   static_cast<std::int64_t>(k + 1));
    }
    EXPECT(changed > 0);
    const std::string h = readText(s4 / "S_HUMI.SPECIFI" / "c" / "2" / "0" / "0" / "0");
    std::memcpy(idx.data(), h.data() + h.size() - 48, 48);
    checkQuantChunk(decodeZstd(h.substr(idx[0], idx[1])), 6, 96, 1);
    checkQuantChunk(decodeZstd(h.substr(idx[4], idx[5])), 6, 96, 3);
    EXPECT(std::isnan(decodeZstd(readText(s4 / "SURFTEMPERATURE" / "c" / "0" / "0" / "0"))[7]));
}

CASE("quantize: only lossless rules give the store of plain zstd, byte for byte") {
    if (!multio::action::zarr::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    for (const std::string layout : {"chunk-per-level", "shard-levels"}) {
        const Path plain = runZstd(layout, 2, "quant-none-" + layout + "-plain");
        const Path q = runQuant(layout, 2, "quant-none-" + layout,
                                R"(, "quantize": {"rules": [{"variables": "*", "lossless": true}]})");
        EXPECT(sameTree(plain, q));
        const Path d = runQuant(layout, 2, "quant-none-" + layout + "-default",
                                R"(, "quantize": {"default": {"lossless": true}})");
        EXPECT(sameTree(plain, d));
    }
}

/// Larger smooth fields through ZarrStore directly: quantized by the pool (small queue) and inline, same bytes
CASE("quantize: worker pool under load, ZarrStore directly") {
    namespace z = multio::action::zarr;
    if (!z::zstdAvailable()) {
        std::cout << "built without zstd, skipped" << std::endl;
        return;
    }
    const std::int64_t nx = 211;
    const std::int64_t ny = 173;
    auto field = [&](int var, std::int64_t step, std::int64_t lev) {  // as in the zstd load test
        std::vector<float> v(static_cast<std::size_t>(nx * ny));
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double x = static_cast<double>(i % nx) / nx;
            const double y = static_cast<double>(i / nx) / ny;
            v[i] = static_cast<float>(250.0 + var + 10.0 * std::sin(6.0 * x + lev) * std::cos(4.0 * y + step)
                                      + 1e-3 * static_cast<double>((i * 2654435761u) % 1000));
        }
        if (var == 0) {
            v[3] = std::numeric_limits<float>::quiet_NaN();
        }
        return v;
    };
    z::QuantRules rules;
    z::QuantRule r;
    r.globs = {"S_VAR0", "S_VAR1"};
    r.quantization.mode = z::QuantMode::RangeBits;
    r.quantization.bits = 10;
    rules.rules.push_back(r);
    r.globs = {"S_VAR2"};
    r.quantization.mode = z::QuantMode::Absolute;
    r.quantization.step = 1.0 / 64;
    rules.rules.push_back(r);
    rules.fallback.mode = z::QuantMode::BitRound;
    rules.fallback.bits = 9;
    for (const auto layout : {z::Layout::ChunkPerLevel, z::Layout::ShardLevels}) {
        const std::string ln = layout == z::Layout::ShardLevels ? "shard" : "chunk";
        std::vector<Path> stores;
        for (int threads : {0, 3}) {
            const Path dir = absolute(outRoot()) / ("quant-load-" + ln + "-t" + std::to_string(threads));
            removeAll(dir);
            std::unique_ptr<z::WorkerPool> pool;
            if (threads > 0) {
                pool = std::make_unique<z::WorkerPool>(threads, 2);
            }
            z::StoreSetup setup;
            setup.name = "LOAD";
            setup.dir = dir.string();
            setup.layout = layout;
            setup.nx = nx;
            setup.ny = ny;
            setup.forecastReferenceTime = FRT;
            setup.compression = z::Compression{z::Codec::Zstd, 1};
            setup.quantize = rules;
            setup.pool = pool.get();
            {
                z::ZarrStore st(setup);
                for (std::int64_t step = 0; step < 3; ++step) {
                    for (int var = 0; var < 6; ++var) {
                        z::FieldDescription fd;
                        fd.prefix = var < 4 ? "S" : "SURF";
                        fd.suffix = "VAR" + std::to_string(var);
                        fd.is3d = var < 4;
                        fd.step = step;
                        fd.leadTime = 3600.0 * step;
                        for (std::int64_t lev = 1; lev <= (fd.is3d ? 12 : 1); ++lev) {
                            fd.level = fd.is3d ? lev : 0;
                            auto v = field(var, step, lev);
                            if (var % 2) {
                                st.addField(fd, std::move(v));  // moved: quantized in place by the worker
                            }
                            else {
                                st.addField(fd, v.data(), v.size());  // copied: the caller's values unchanged
                                EXPECT(sameBits(v[100], field(var, step, lev)[100]));
                            }
                        }
                    }
                    st.flush(step, std::nullopt);
                }
                st.finalize();
            }
            stores.push_back(dir / "LOAD.zarr");
        }
        EXPECT(sameTree(stores[0], stores[1]));
        if (layout == z::Layout::ChunkPerLevel) {
            // range-bits 10 on S_VAR0 (NaN at index 3), absolute 2^-6 on S_VAR2, bitround 9 on the rest
            for (int var : {0, 2, 4}) {
                const std::string name = var < 4 ? "S_VAR" + std::to_string(var) : "SURFVAR" + std::to_string(var);
                const Path chunk = var < 4 ? stores[1] / name / "c" / "2" / "4" / "0" / "0"
                                           : stores[1] / name / "c" / "2" / "0" / "0";
                const auto q = decodeZstd(readText(chunk));
                const auto x = field(var, 2, var < 4 ? 5 : 1);
                double lo = 1e30;
                double hi = -1e30;
                for (float f : x) {
                    if (std::isfinite(f)) {
                        lo = std::min(lo, static_cast<double>(f));
                        hi = std::max(hi, static_cast<double>(f));
                    }
                }
                const double d = var == 0 ? z::powerOfTwoFloor((hi - lo) / 1024) : 1.0 / 64;
                for (std::size_t i = 0; i < x.size(); ++i) {
                    if (std::isnan(x[i])) {
                        EXPECT(std::isnan(q[i]));
                        continue;
                    }
                    const double err = std::fabs(static_cast<double>(q[i]) - x[i]);
                    EXPECT(err <= (var == 4 ? std::ldexp(std::fabs(static_cast<double>(x[i])), -10) : d / 2));
                }
            }
        }
    }
}

CASE("configuration errors") {
    EXPECT_THROWS(MultioTestEnvironment(R"({"name": "z", "actions": [{"type": "zarr"}]})"));
    EXPECT_THROWS(MultioTestEnvironment(R"({"name": "z", "actions": [{"type": "zarr", "path": "x", "format": 2}]})"));
    EXPECT_THROWS(
        MultioTestEnvironment(R"({"name": "z", "actions": [{"type": "zarr", "path": "x", "layout": "foo"}]})"));

    // compression
    auto zarrWith = [](const std::string& members) {
        return std::string(R"({"name": "z", "actions": [{"type": "zarr", "path": ")") + absolute(outRoot()).string()
             + R"(/config-errors")" + members + "}]}";
    };
    auto expectUserError = [&](const std::string& members, const std::string& what) {
        try {
            MultioTestEnvironment env(zarrWith(members));
            std::cout << "no error for " << members << std::endl;
            EXPECT(false);
        }
        catch (const eckit::UserError& e) {
            const std::string msg = e.what();
            if (msg.find(what) == std::string::npos) {
                std::cout << "unexpected message for " << members << ": " << msg << std::endl;
            }
            EXPECT(msg.find(what) != std::string::npos);
        }
    };
    expectUserError(R"(, "compression": {"codec": "lz4"})", "unknown compression codec 'lz4'");
    expectUserError(R"(, "compression": "gzip")", "unknown compression codec 'gzip'");
    expectUserError(R"(, "compression": {"level": 3})", "needs 'codec'");
    expectUserError(R"(, "compression": {"codec": "zstd", "lvl": 3})", "unknown key 'compression.lvl'");
    expectUserError(R"(, "compression": [1, 2])", "must be a codec name");
    expectUserError(R"(, "compression-threads": -1)", "'compression-threads' is -1");
    expectUserError(R"(, "compression-threads": "four")", "'compression-threads' must be an integer");
    expectUserError(R"(, "compression-queue": 100000)", "'compression-queue' is 100000");
    if (multio::action::zarr::zstdAvailable()) {
        expectUserError(R"(, "compression": {"codec": "zstd", "level": 23})", "zstd compression level 23 out of range");
        expectUserError(R"(, "compression": {"codec": "zstd", "level": "high"})", "must be an integer");
        // valid
        MultioTestEnvironment(zarrWith(R"(, "compression": {"codec": "zstd", "level": 19}, "compression-threads": 0)"));
        MultioTestEnvironment(zarrWith(R"(, "compression": "zstd")"));
        MultioTestEnvironment(zarrWith(R"(, "compression": {"codec": "none"}, "compression-threads": 8)"));

        // quantize
        const std::string z = R"(, "compression": "zstd", "quantize": )";
        expectUserError(R"(, "quantize": {"default": {"keep-bits": 10}})",
                        "'quantize' needs 'compression: {codec: zstd}'");
        expectUserError(R"(, "compression": "none", "quantize": {"rules": [{"variables": "*", "keep-bits": 10}]})",
                        "'quantize' needs 'compression");
        expectUserError(z + R"([1, 2])", "'quantize' must be a map");
        expectUserError(z + R"({"rule": []})", "unknown key 'quantize.rule'");
        expectUserError(z + R"({"rules": {"variables": "*"}})", "'quantize.rules' must be a list of maps");
        expectUserError(z + R"({"rules": [{"variables": "*", "keepbits": 3}]})",
                        "unknown key 'quantize.rules[0].keepbits'");
        expectUserError(z + R"({"rules": [{"variables": "*"}]})", "needs exactly one of keep-bits");
        expectUserError(z + R"({"rules": [{"variables": "*", "keep-bits": 3, "absolute": 0.1}]})",
                        "needs exactly one of keep-bits");
        expectUserError(z + R"({"rules": [{"keep-bits": 3}]})", "'quantize.rules[0]' needs 'variables' or 'regex'");
        expectUserError(z + R"({"rules": [{"variables": "*", "keep-bits": 0}]})",
                        "'quantize.rules[0].keep-bits' is 0, must be in [1, 23]");
        expectUserError(z + R"({"rules": [{"variables": "*", "keep-bits": 24}]})", "keep-bits' is 24");
        expectUserError(z + R"({"rules": [{"variables": "*", "keep-bits": 2.5}]})", "keep-bits' must be an integer");
        expectUserError(z + R"({"rules": [{"variables": "*", "range-bits": 30}]})", "range-bits' is 30");
        expectUserError(z + R"({"rules": [{"variables": "*", "absolute": 0}]})", "'quantize.rules[0].absolute' is 0");
        expectUserError(z + R"({"rules": [{"variables": "*", "absolute": -0.5}]})", "must be > 0");
        expectUserError(z + R"({"rules": [{"variables": "*", "absolute": "small"}]})", "absolute' must be a number");
        expectUserError(z + R"({"rules": [{"variables": "*", "absolute": 1e-50}]})", "must be in [2^-126, 2^101)");
        expectUserError(z + R"({"rules": [{"variables": "*", "lossless": false}]})", "lossless' must be true");
        expectUserError(z + R"({"rules": [{"variables": 3, "lossless": true}]})", "must be a glob or a list of globs");
        expectUserError(z + R"({"rules": [{"variables": "", "lossless": true}]})", "empty pattern");
        expectUserError(z + R"({"rules": [{"domains": 3, "variables": "*", "lossless": true}]})",
                        "'quantize.rules[0].domains' must be a glob or a list of globs");
        expectUserError(z + R"({"rules": [{"domains": "", "variables": "*", "lossless": true}]})",
                        "'quantize.rules[0].domains' has an empty pattern");
        expectUserError(z + R"({"rules": [{"domains": [], "variables": "*", "lossless": true}]})",
                        "'quantize.rules[0].domains' is empty");
        expectUserError(z + R"({"rules": [{"domains": "PF*", "keep-bits": 4}]})",
                        "'quantize.rules[0]' needs 'variables' or 'regex'");
        expectUserError(z + R"({"rules": [{"regex": "([", "keep-bits": 4}]})", "is not a valid regular expression");
        expectUserError(z + R"({"rules": [{"variables": "*", "keep-bits": 4}, {"variables": "*", "range-bits": 0}]})",
                        "'quantize.rules[1].range-bits' is 0");
        expectUserError(z + R"({"default": 12})", "'quantize.default' must be a map");
        expectUserError(z + R"({"default": {"variables": "*", "keep-bits": 4}})",
                        "unknown key 'quantize.default.variables'");
        // valid
        MultioTestEnvironment(
            zarrWith(z
                     + R"({"rules": [{"variables": ["H_*", "P_*"], "regex": "^S_.*$", "range-bits": 12},)"
                       R"({"variables": "CLS*", "absolute": 1}], "default": {"keep-bits": 7}})"));
        MultioTestEnvironment(zarrWith(z + R"({"rules": []})"));
        MultioTestEnvironment(zarrWith(R"(, "compression": "none", "quantize": {"rules": []})"));  // empty: lossless
    }
    else {
        expectUserError(R"(, "compression": {"codec": "zstd"})", "built without libzstd");
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Spatial tiles (zarr-tile-size > 0): the store's window cut into tiles, one MultIO domain <store>#<tile> per tile,
// every message one tile; several server ranks (action instances) write disjoint tiles of one store

constexpr std::int64_t TNX = 37;
constexpr std::int64_t TNY = 29;
constexpr std::int64_t TT = 8;  // 5 x 4 tiles of 8 x 8, the last column 5 wide, the last row 5 high
const std::vector<std::int64_t> TSTEPS = {0, 24, 48, 96};  // 24: no fields (an empty output step)
const std::vector<std::int64_t> TFIELD_STEPS = {0, 48, 96};

// The python check has the same table and formula
const std::vector<Var> TVARS = {
    {1, "S", "TEMPERATURE", "", {3, 1, 2}, 130, false, 0},
    {2, "KB", "ISOT_ALTIT", "", {2, 1}, 0, false, 0},
    {3, "KT", "ISOT_ALTIT", "", {1}, 0, false, 0},  // fewer levels than its axis: missing inner chunks
    {4, "SURF", "TEMPERATURE", "", {}, 0, false, 0},
    {5, "CLS", "TEMPERATURE", "", {}, 0, true, 0},
    {6, "S", "HUMI.SPECIFI", "", {1, 3}, 0, false, 1},  // from the second field step on
};

double tvalue(int id, std::int64_t step, std::int64_t level, std::int64_t x, std::int64_t y) {
    if (id == 4 && (x + 3 * y) % 11 == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return id * 1000.0 + step * 10.0 + level * 0.5 + x * 0.01 + y * 0.37;
}

struct TileGeom {
    std::int64_t x0, y0, tw, th;
};

/// T = 0: one "tile" covering the window
std::int64_t tileCount(std::int64_t T) {
    if (T == 0) {
        return 1;
    }
    const std::int64_t tx = std::min(T, TNX);
    const std::int64_t ty = std::min(T, TNY);
    return ((TNX + tx - 1) / tx) * ((TNY + ty - 1) / ty);
}

TileGeom tileGeom(std::int64_t T, std::int64_t id) {
    if (T == 0) {
        return {0, 0, TNX, TNY};
    }
    const std::int64_t tx = std::min(T, TNX);
    const std::int64_t ty = std::min(T, TNY);
    const std::int64_t ntx = (TNX + tx - 1) / tx;
    const std::int64_t x0 = (id % ntx) * tx;
    const std::int64_t y0 = (id / ntx) * ty;
    return {x0, y0, std::min(tx, TNX - x0), std::min(ty, TNY - y0)};
}

/// client 0 owns the columns x < 20, client 1 the others: tiles of the third column are shared
std::size_t tileClientOf(std::int64_t x) {
    return x < 20 ? 0 : 1;
}

/// The points (x, y) of client c in tile id, in map order (row by row)
std::vector<std::pair<std::int64_t, std::int64_t>> tilePoints(std::int64_t T, std::int64_t id, std::size_t c) {
    const auto g = tileGeom(T, id);
    std::vector<std::pair<std::int64_t, std::int64_t>> pts;
    for (std::int64_t y = g.y0; y < g.y0 + g.th; ++y) {
        for (std::int64_t x = g.x0; x < g.x0 + g.tw; ++x) {
            if (tileClientOf(x) == c) {
                pts.emplace_back(x, y);
            }
        }
    }
    return pts;
}

std::string tileDomain(const std::string& store, std::int64_t T, std::int64_t id) {
    return T == 0 ? store : store + "#" + std::to_string(id);
}

/// Domain maps of every tile domain of the store, for the clients that have points in it
void registerTiles(const std::string& store, std::int64_t T) {
    for (std::int64_t id = 0; id < tileCount(T); ++id) {
        const auto g = tileGeom(T, id);
        for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
            std::vector<std::int32_t> idx;
            for (const auto& p : tilePoints(T, id, c)) {
                idx.push_back(static_cast<std::int32_t>((p.second - g.y0) * g.tw + (p.first - g.x0)));
            }
            if (idx.empty()) {
                continue;
            }
            Metadata md;
            md.set("name", tileDomain(store, T, id));
            md.set("domain", tileDomain(store, T, id));
            md.set("category", std::string("ial-domain-map"));
            md.set("representation", std::string("unstructured"));
            md.set("misc-globalSize", g.tw * g.th);
            md.set("toAllServers", true);
            Message msg{Message::Header{Message::Tag::Domain, CLIENTS[c], SERVER, std::move(md)},
                        eckit::Buffer(idx.data(), idx.size() * sizeof(std::int32_t))};
            multio::domain::Mappings::instance().add(std::move(msg));
        }
    }
}

/// Store keys of a message: tiled, the tile keys (zarr-tile-size as int, string or real, depending on the tile)
void setTileKeys(Metadata& md, const std::string& store, std::int64_t T, std::int64_t id) {
    md.set("domain", tileDomain(store, T, id));
    md.set("geometry-nx", TNX);
    md.set("geometry-ny", TNY);
    if (T == 0) {
        return;
    }
    md.set("zarr-store", store);
    md.set("zarr-tile", id);
    switch (id % 3) {
        case 0:
            md.set("zarr-tile-size", T);
            break;
        case 1:
            md.set("zarr-tile-size", std::to_string(T));
            break;
        default:
            md.set("zarr-tile-size", static_cast<double>(T));
    }
}

Message tileFieldMessage(const std::string& store, std::int64_t T, std::int64_t id, const Var& v, std::int64_t step,
                         std::int64_t level, std::size_t c) {
    const auto g = tileGeom(T, id);
    Metadata md;
    md.set("category", std::string("ial-field"));
    setTileKeys(md, store, T, id);
    md.set("misc-globalSize", g.tw * g.th);
    md.set("misc-precision", std::string(v.single ? "single" : "double"));
    md.set("name", v.levtype + (v.levels.empty() ? "" : std::to_string(level)) + v.suffix);
    md.set("levtype", v.levtype);
    md.set("suffix", v.suffix);
    md.set("levelist", level);
    md.set("is3d", !v.levels.empty());
    md.set("grib-code", v.grib);
    md.set("step", step);
    md.set("timestep", TIMESTEP);
    md.set("forecastReferenceTime", FRT);
    std::vector<double> vals;
    for (const auto& p : tilePoints(T, id, c)) {
        vals.push_back(tvalue(v.id, step, level, p.first, p.second));
    }
    if (v.single) {
        std::vector<float> f(vals.begin(), vals.end());
        return Message{Message::Header{Message::Tag::Field, CLIENTS[c], SERVER, std::move(md)},
                       eckit::Buffer(f.data(), f.size() * sizeof(float))};
    }
    return Message{Message::Header{Message::Tag::Field, CLIENTS[c], SERVER, std::move(md)},
                   eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
}

Message tileFlushMessage(const std::string& store, std::int64_t T, std::int64_t id, std::int64_t step,
                         const std::string& kind, std::size_t c) {
    Metadata md;
    setTileKeys(md, store, T, id);
    md.set("step", step);
    md.set("flushKind", kind);
    return Message{Message::Header{Message::Tag::Flush, CLIENTS[c], SERVER, std::move(md)}};
}

const std::string TGROUP_ATTRS
    = R"J({"Conventions":"CF-1.8","title":"IAL output, domain TILES","source":"IAL MultIO client (prototype)","forecast_reference_time":")J"
    + FRT + R"(","nx_ci":37,"ny_ci":29})";

/// The static arrays of the tiled stores: whole arrays, sent by client 0 once per tile domain
Message tileStaticMessage(const std::string& store, std::int64_t T, std::int64_t id, const std::string& name) {
    std::vector<double> vals;
    std::string dims, shape, dtype = "float64", attrs;
    if (name == "x" || name == "y") {
        const std::int64_t n = name == "x" ? TNX : TNY;
        for (std::int64_t i = 0; i < n; ++i) {
            vals.push_back(2500.0 * i + (name == "y" ? 0.5 : 0.0));
        }
        dims = name;
        shape = std::to_string(n);
        attrs = name == "x" ? X_ATTRS : Y_ATTRS;
    }
    else if (name == "latitude" || name == "longitude") {
        for (std::int64_t j = 0; j < TNY; ++j) {
            for (std::int64_t i = 0; i < TNX; ++i) {
                vals.push_back(name == "latitude" ? 51.0 + 0.0225 * j + 1.0e-4 * i : 4.9 + 0.0361 * i - 2.0e-4 * j);
            }
        }
        dims = "y,x";
        shape = std::to_string(TNY) + "," + std::to_string(TNX);
        attrs = name == "latitude" ? LAT_ATTRS : LON_ATTRS;
    }
    else {
        vals.push_back(0.0);
        dtype = "int32";
        attrs = CRS_ATTRS;
    }
    Metadata md;
    md.set("category", std::string("ial-static"));
    setTileKeys(md, store, T, id);
    md.set("name", name);
    md.set("skipAggregation", true);
    md.set("zarr-static", true);
    md.set("zarr-dimensions", dims);
    md.set("zarr-shape", shape);
    md.set("zarr-dtype", dtype);
    md.set("zarr-attributes", attrs);
    if (name == "x") {
        md.set("zarr-group-attributes", TGROUP_ATTRS);
    }
    md.set("forecastReferenceTime", FRT);
    md.set("misc-precision", std::string("double"));
    md.set("misc-globalSize", static_cast<std::int64_t>(vals.size()));
    return Message{Message::Header{Message::Tag::Field, CLIENTS[0], SERVER, std::move(md)},
                   eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
}

using Event = std::function<Message()>;

/// The messages of one tile domain, by phase: 0 the static arrays, then one phase per output step (the tile's
/// fields, then its flushes, end-of-simulation for the last step)
std::vector<std::vector<Event>> tileEvents(const std::string& store, std::int64_t T, std::int64_t id) {
    std::vector<std::vector<Event>> phases(1 + TSTEPS.size());
    for (const std::string n : {"x", "y", "latitude", "longitude", "crs"}) {
        phases[0].push_back([=] { return tileStaticMessage(store, T, id, n); });
    }
    std::vector<std::size_t> clients;
    for (std::size_t c = 0; c < CLIENTS.size(); ++c) {
        if (!tilePoints(T, id, c).empty()) {
            clients.push_back(c);
        }
    }
    for (std::size_t s = 0; s < TSTEPS.size(); ++s) {
        const std::int64_t step = TSTEPS[s];
        auto fs = std::find(TFIELD_STEPS.begin(), TFIELD_STEPS.end(), step);
        if (fs != TFIELD_STEPS.end()) {
            const std::size_t ft = static_cast<std::size_t>(fs - TFIELD_STEPS.begin());
            for (const auto& v : TVARS) {
                if (ft < v.firstTime) {
                    continue;
                }
                for (auto lev : v.levels.empty() ? std::vector<std::int64_t>{0} : v.levels) {
                    for (auto c : clients) {
                        phases[1 + s].push_back([=] { return tileFieldMessage(store, T, id, v, step, lev, c); });
                    }
                }
            }
        }
        const std::string kind = s + 1 == TSTEPS.size() ? "end-of-simulation" : "step";
        for (auto c : clients) {
            phases[1 + s].push_back([=] { return tileFlushMessage(store, T, id, step, kind, c); });
        }
    }
    return phases;
}

/// Delivery orders of the messages of a set of tiles (each tile's own order is kept):
///   0: phase by phase, the tiles round robin, one message at a time
///   1: tile by tile, all phases of a tile (the last tile first): the other tiles are unknown when the first one
///      closes its steps and finalises the store
///   2: phase by phase, the tiles in a pseudo-random interleaving
std::vector<Event> deliveryOrder(const std::string& store, std::int64_t T, const std::vector<std::int64_t>& tiles,
                                 int order) {
    std::vector<std::vector<std::vector<Event>>> ev;
    for (auto id : tiles) {
        ev.push_back(tileEvents(store, T, id));
    }
    std::vector<Event> out;
    const std::size_t nphase = 1 + TSTEPS.size();
    if (order == 1) {
        for (std::size_t i = ev.size(); i-- > 0;) {
            for (auto& ph : ev[i]) {
                out.insert(out.end(), ph.begin(), ph.end());
            }
        }
        return out;
    }
    std::uint32_t rng = 12345;
    for (std::size_t p = 0; p < nphase; ++p) {
        std::vector<std::size_t> pos(ev.size(), 0);
        std::size_t left = 0;
        for (auto& e : ev) {
            left += e[p].size();
        }
        std::size_t next = 0;
        while (left > 0) {
            std::size_t i;
            if (order == 0) {
                i = next++ % ev.size();
            }
            else {
                rng = rng * 1103515245u + 12345u;
                i = (rng >> 8) % ev.size();
            }
            if (pos[i] < ev[i][p].size()) {
                out.push_back(ev[i][p][pos[i]++]);
                --left;
            }
        }
    }
    return out;
}

std::vector<std::int64_t> allTiles(std::int64_t T) {
    std::vector<std::int64_t> t;
    for (std::int64_t id = 0; id < tileCount(T); ++id) {
        t.push_back(id);
    }
    return t;
}

/// One synthetic run of store TILES into <selftest>/<sub>, untiled (T = 0) or tiled, by `writers` action instances
/// (tile id % writers; writer w gets delivery order w % 3), all writing into the same directory
Path runTiles(const std::string& sub, const std::string& layout, std::int64_t T, const std::string& extra,
              int writers = 1) {
    const std::string store = "TILES";
    const Path out = absolute(outRoot()) / sub;
    removeAll(out);
    registerTiles(store, T);
    std::vector<std::unique_ptr<MultioTestEnvironment>> envs;
    std::vector<std::vector<Event>> events;
    for (int w = 0; w < writers; ++w) {
        envs.push_back(std::make_unique<MultioTestEnvironment>(planJson(out.string(), layout, "", extra)));
        std::vector<std::int64_t> mine;
        for (auto id : allTiles(T)) {
            if (id % writers == w) {
                mine.push_back(id);
            }
        }
        events.push_back(deliveryOrder(store, T, mine, writers == 1 ? 0 : w % 3));
    }
    // the writers' messages interleaved, as server ranks running at the same time
    std::vector<std::size_t> pos(writers, 0);
    for (bool any = true; any;) {
        any = false;
        for (int w = 0; w < writers; ++w) {
            if (pos[w] < events[w].size()) {
                envs[w]->process(events[w][pos[w]++]());
                any = true;
            }
        }
    }
    envs.clear();
    return out / (store + ".zarr");
}

std::vector<std::int64_t> parseInts(const std::string& raw) {
    std::vector<std::int64_t> v;
    for (const auto& e : multio::action::zarr::jsonArrayElements(raw)) {
        v.push_back(std::stoll(e));
    }
    return v;
}

/// A minimal zarr v3 reader for the arrays of these stores: one (time, level) plane of an array, from chunk files
/// or shards (index at the end), raw or zstd, missing chunks as the fill value NaN, edge chunks cropped
std::vector<float> readPlane(const Path& store, const std::string& name, std::int64_t t, std::int64_t k) {
    namespace z = multio::action::zarr;
    const std::string doc = readText(store / name / "zarr.json");
    const auto shape = parseInts(z::jsonMember(doc, "shape"));
    const auto chunk
        = parseInts(z::jsonMember(z::jsonMember(z::jsonMember(doc, "chunk_grid"), "configuration"), "chunk_shape"));
    const std::string codecs = z::jsonMember(doc, "codecs");
    const bool sharded = codecs.find("\"sharding_indexed\"") != std::string::npos;
    const bool zstd = codecs.find("\"zstd\"") != std::string::npos;
    const std::size_t nd = shape.size();
    const std::int64_t ny = shape[nd - 2], nx = shape[nd - 1], cy = chunk[nd - 2], cx = chunk[nd - 1];
    std::vector<float> out(static_cast<std::size_t>(nx * ny), std::numeric_limits<float>::quiet_NaN());
    for (std::int64_t jy = 0; jy * cy < ny; ++jy) {
        for (std::int64_t jx = 0; jx * cx < nx; ++jx) {
            const std::string tile = "/" + std::to_string(jy) + "/" + std::to_string(jx);
            std::optional<std::string> bytes;
            if (!sharded) {
                const Path p = store / name / ("c/" + std::to_string(t) + (nd == 4 ? "/" + std::to_string(k) : "") + tile);
                if (exists(p)) {
                    bytes = readText(p);
                }
            }
            else {
                const Path p = store / name / ("c/" + std::to_string(t) + "/0" + tile);
                if (exists(p)) {
                    const std::string b = readText(p);
                    const std::size_t nlev = static_cast<std::size_t>(chunk[1]);
                    std::uint64_t idx[2];
                    std::memcpy(idx, b.data() + b.size() - nlev * 16 + static_cast<std::size_t>(k) * 16, 16);
                    if (idx[0] != std::numeric_limits<std::uint64_t>::max()) {
                        bytes = b.substr(idx[0], idx[1]);
                    }
                }
            }
            if (!bytes) {
                continue;
            }
            std::vector<float> vals;
            if (zstd) {
                vals = decodeZstd(*bytes);
            }
            else {
                vals.resize(bytes->size() / sizeof(float));
                std::memcpy(vals.data(), bytes->data(), vals.size() * sizeof(float));
            }
            if (static_cast<std::int64_t>(vals.size()) != cy * cx) {
                std::cout << name << " chunk c/" << t << "/" << k << tile << ": " << vals.size() << " values, chunk "
                          << cy << " x " << cx << std::endl;
                throw std::runtime_error("bad chunk size");
            }
            for (std::int64_t y = jy * cy; y < std::min(ny, (jy + 1) * cy); ++y) {
                for (std::int64_t x = jx * cx; x < std::min(nx, (jx + 1) * cx); ++x) {
                    out[y * nx + x] = vals[(y - jy * cy) * cx + (x - jx * cx)];
                }
            }
        }
    }
    return out;
}

bool sameOrBothNaN(float a, float b) {
    return sameBits(a, b) || (std::isnan(a) && std::isnan(b));
}

/// `tiled` reads like `ref` (untiled): same arrays, values, attributes, shapes, dimensions, root attributes, same
/// static / coordinate arrays byte for byte; the tiled consolidated metadata is sorted by name
bool sameContent(const Path& ref, const Path& tiled) {
    namespace z = multio::action::zarr;
    bool ok = true;
    auto fail = [&](const std::string& what) {
        std::cout << "tiles: " << tiled.string() << " vs " << ref.string() << ": " << what << std::endl;
        ok = false;
    };
    auto nr = consolidatedNames(ref);
    const auto nt = consolidatedNames(tiled);
    if (!std::is_sorted(nt.begin(), nt.end())) {
        fail("consolidated metadata not sorted");
    }
    std::sort(nr.begin(), nr.end());
    if (nr != nt) {
        fail("other arrays");
        return false;
    }
    if (rootAttributes(ref) != rootAttributes(tiled)) {
        fail("root attributes");
    }
    if (readText(ref / ".last_step") != readText(tiled / ".last_step")) {
        fail(".last_step");
    }
    const std::string cm = z::jsonMember(z::jsonMember(readText(tiled / "zarr.json"), "consolidated_metadata"), "metadata");
    for (const auto& n : nr) {
        const std::string dr = readText(ref / n / "zarr.json");
        const std::string dt = readText(tiled / n / "zarr.json");
        if (z::jsonMember(cm, n) != dt) {
            fail(n + ": consolidated metadata differs from zarr.json");
        }
        const std::string dims = z::jsonMember(dr, "dimension_names");
        if (n == "time" || dims.find("\"time\"") == std::string::npos) {
            if (dr != dt || !sameTree(ref / n, tiled / n)) {
                fail(n + ": static or coordinate array differs");
            }
            continue;
        }
        for (const char* m : {"shape", "data_type", "fill_value", "attributes", "dimension_names"}) {
            if (z::jsonMember(dr, m) != z::jsonMember(dt, m)) {
                fail(n + ": " + m + " differs");
            }
        }
        const auto shape = parseInts(z::jsonMember(dr, "shape"));
        const std::int64_t nlev = shape.size() == 4 ? shape[1] : 1;
        for (std::int64_t t = 0; t < shape[0]; ++t) {
            for (std::int64_t k = 0; k < nlev; ++k) {
                const auto a = readPlane(ref, n, t, k);
                const auto b = readPlane(tiled, n, t, k);
                std::size_t diff = 0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    diff += sameOrBothNaN(a[i], b[i]) ? 0 : 1;
                }
                if (diff != 0) {
                    fail(n + " time " + std::to_string(t) + " level index " + std::to_string(k) + ": "
                         + std::to_string(diff) + " values differ");
                }
            }
        }
    }
    return ok;
}

/// The reference values are the formula's (the untiled store is checked against tvalue)
std::size_t checkTileReference(const Path& ref) {
    std::size_t bad = 0;
    std::size_t nanTotal = 0;
    for (std::size_t ft = 0; ft < TFIELD_STEPS.size(); ++ft) {
        for (const auto& v : TVARS) {
            if (ft < v.firstTime) {
                continue;
            }
            const std::string name = v.levels.empty() ? v.levtype + v.suffix : v.levtype + "_" + v.suffix;
            for (auto lev : v.levels.empty() ? std::vector<std::int64_t>{0} : v.levels) {
                const std::int64_t k = v.levels.empty() ? 0 : lev - 1;  // level_S = [1, 2, 3], level_K = [1, 2]
                const auto p = readPlane(ref, name, static_cast<std::int64_t>(ft), k);
                for (std::int64_t y = 0; y < TNY; ++y) {
                    for (std::int64_t x = 0; x < TNX; ++x) {
                        const float want = static_cast<float>(tvalue(v.id, TFIELD_STEPS[ft], lev, x, y));
                        nanTotal += std::isnan(want) ? 1 : 0;
                        if (!sameOrBothNaN(p[y * TNX + x], want)) {
                            if (bad++ < 5) {
                                std::cout << "tiles reference: " << name << " time " << ft << " level " << lev
                                          << " (" << x << ", " << y << "): " << p[y * TNX + x] << ", expected "
                                          << want << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
    EXPECT(nanTotal > 0);
    return bad;
}

/// The variants of the tiled tests: (name, extra zarr configuration)
std::vector<std::pair<std::string, std::string>> tileVariants() {
    std::vector<std::pair<std::string, std::string>> v{{"raw", ""}};
    if (multio::action::zarr::zstdAvailable()) {
        v.emplace_back("zstd", R"(, "compression": {"codec": "zstd", "level": 3}, "compression-threads": 2)");
        // bitround works value by value: the tiled store has the same values as the untiled one
        v.emplace_back("zstd-bitround",
                       R"(, "compression": "zstd", "compression-threads": 0, "quantize": {"default": {"keep-bits": 9}})");
    }
    return v;
}

CASE("tiles: a tiled store reads like the untiled one (select -> aggregate -> zarr, one tile per domain)") {
    for (const std::string layout : {"chunk-per-level", "shard-levels"}) {
        for (const auto& [vname, extra] : tileVariants()) {
            const std::string sub = layout + "-" + vname;
            const Path ref = runTiles("tiles-ref-" + sub, layout, 0, extra);
            const Path tiled = runTiles("tiles-" + sub, layout, TT, extra);
            if (vname != "zstd-bitround") {  // quantized: the python check applies the error bound
                const std::size_t bad = checkTileReference(ref);
                EXPECT_EQUAL(bad, 0u);
            }
            EXPECT(sameContent(ref, tiled));
            EXPECT_EQUAL(countStaging(tiled), 0u);
            EXPECT_EQUAL(readText(tiled / ".last_step"), std::string("step=96 time_index=2 inc=0002:00:00\n"));
            EXPECT(readText(tiled / "time" / "zarr.json").find("\"shape\":[3]") != std::string::npos);

            // chunk grid of the tiles, chunk keys c/<t>/<k>/<jy>/<jx> (shards c/<t>/0/<jy>/<jx>)
            const std::string tdoc = readText(tiled / "S_TEMPERATURE" / "zarr.json");
            const std::string sdoc = readText(tiled / "SURFTEMPERATURE" / "zarr.json");
            EXPECT(tdoc.find("\"shape\":[3,3,29,37]") != std::string::npos);
            EXPECT(sdoc.find("\"chunk_shape\":[1,8,8]") != std::string::npos);
            const std::size_t raw = 8 * 8 * sizeof(float);
            if (layout == "chunk-per-level") {
                EXPECT(tdoc.find("\"chunk_shape\":[1,1,8,8]") != std::string::npos);
                for (std::int64_t jy = 0; jy < 4; ++jy) {
                    for (std::int64_t jx = 0; jx < 5; ++jx) {
                        EXPECT(exists(tiled / "S_TEMPERATURE" / ("c/2/2/" + std::to_string(jy) + "/" + std::to_string(jx))));
                    }
                }
                EXPECT(!exists(tiled / "S_TEMPERATURE" / "c/2/2/4"));
                EXPECT(!exists(tiled / "KT_ISOT_ALTIT" / "c/0/1/0/0"));
                if (vname == "raw") {
                    // edge tile (jy, jx) = (3, 4): 5 x 5 values padded with NaN to 8 x 8
                    const auto e = readFloats(tiled / "S_TEMPERATURE" / "c/0/0/3/4");
                    EXPECT_EQUAL(e.size() * sizeof(float), raw);
                    if (e.size() == 64) {
                        EXPECT(sameBits(e[0], static_cast<float>(tvalue(1, 0, 1, 32, 24))));
                        EXPECT(sameBits(e[4 * 8 + 4], static_cast<float>(tvalue(1, 0, 1, 36, 28))));
                        EXPECT(std::isnan(e[5]) && std::isnan(e[7 * 8 + 7]) && std::isnan(e[5 * 8]));
                    }
                }
            }
            else {
                EXPECT(tdoc.find("\"chunk_shape\":[1,3,8,8]") != std::string::npos);
                EXPECT(tdoc.find("\"sharding_indexed\",\"configuration\":{\"chunk_shape\":[1,1,8,8]")
                       != std::string::npos);
                EXPECT(exists(tiled / "S_TEMPERATURE" / "c/1/0/3/4"));
                EXPECT(!exists(tiled / "S_TEMPERATURE" / "c/1/1"));
                if (vname == "raw") {
                    EXPECT_EQUAL(fileSize(tiled / "S_TEMPERATURE" / "c/1/0/3/4"),
                                 static_cast<std::uintmax_t>(3 * raw + 3 * 16));
                    EXPECT_EQUAL(fileSize(tiled / "KT_ISOT_ALTIT" / "c/0/0/2/1"),
                                 static_cast<std::uintmax_t>(raw + 2 * 16));
                }
            }
            if (vname == "raw") {
                EXPECT_EQUAL(fileSize(tiled / "SURFTEMPERATURE" / "c/2/3/4"), static_cast<std::uintmax_t>(raw));
            }
        }
    }
}

CASE("tiles: three writers (server ranks) of one store, messages in different orders") {
    for (const std::string layout : {"chunk-per-level", "shard-levels"}) {
        for (const auto& [vname, extra] : tileVariants()) {
            if (vname == "zstd-bitround") {
                continue;
            }
            const std::string sub = layout + "-" + vname;
            const Path ref = absolute(outRoot()) / ("tiles-ref-" + sub) / "TILES.zarr";
            if (!exists(ref / "zarr.json")) {
                runTiles("tiles-ref-" + sub, layout, 0, extra);
            }
            const Path shared = runTiles("tiles-mw3-" + sub, layout, TT, extra, 3);
            EXPECT(sameContent(ref, shared));
            EXPECT_EQUAL(countStaging(shared), 0u);

            // every writer alone (its own directory) writes the same metadata files as the others
            std::vector<Path> alone;
            for (int w = 0; w < 3; ++w) {
                const Path out = absolute(outRoot()) / ("tiles-mw3-" + sub + "-w" + std::to_string(w));
                removeAll(out);
                {
                    auto env = MultioTestEnvironment(planJson(out.string(), layout, "", extra));
                    std::vector<std::int64_t> mine;
                    for (auto id : allTiles(TT)) {
                        if (id % 3 == w) {
                            mine.push_back(id);
                        }
                    }
                    for (auto& e : deliveryOrder("TILES", TT, mine, w)) {
                        env.process(e());
                    }
                }
                alone.push_back(out / "TILES.zarr");
            }
            std::vector<std::string> meta = {"zarr.json", ".last_step"};
            for (const auto& n : consolidatedNames(shared)) {
                meta.push_back(n + "/zarr.json");
            }
            for (const auto& f : meta) {
                const std::string want = readText(shared / f);
                for (int w = 0; w < 3; ++w) {
                    if (readText(alone[w] / f) != want) {
                        std::cout << "tiles: writer " << w << " writes another " << f << std::endl;
                        EXPECT(false);
                    }
                }
            }
            // the coordinates (time, levels, static arrays) are the same files on every writer
            for (const std::string n : {"time", "level_S", "level_K", "x", "latitude", "crs"}) {
                for (int w = 0; w < 3; ++w) {
                    EXPECT(sameTree(alone[w] / n, shared / n));
                }
            }
            // a writer alone has only its tiles: tile 0 is writer 0's, tile 1 writer 1's
            // (chunk c/<t>/<k>/<jy>/<jx> and shard c/<t>/0/<jy>/<jx> keys coincide here)
            EXPECT(exists(alone[0] / "S_TEMPERATURE" / "c/0/0/0/0"));
            EXPECT(!exists(alone[1] / "S_TEMPERATURE" / "c/0/0/0/0"));
            EXPECT(exists(alone[1] / "S_TEMPERATURE" / "c/0/0/0/1"));
            EXPECT(!exists(alone[0] / "S_TEMPERATURE" / "c/0/0/0/1"));
        }
    }
}

/// Collect the messages of an exception and its nested ones
template <typename F>
std::string errorText(F&& f) {
    std::string text;
    std::function<void(const std::exception&)> collect = [&](const std::exception& e) {
        text += e.what();
        text += "\n";
        try {
            std::rethrow_if_nested(e);
        }
        catch (const std::exception& n) {
            collect(n);
        }
        catch (...) {
        }
    };
    try {
        f();
    }
    catch (const std::exception& e) {
        collect(e);
        return text.empty() ? std::string("?") : text;
    }
    return std::string();
}

/// A tile message straight to the zarr action (no aggregation): `tweak` changes the metadata
Message directTile(const std::string& store, std::int64_t id, std::int64_t step, std::int64_t nvals,
                   const std::function<void(Metadata&)>& tweak = {}) {
    Metadata md;
    setTileKeys(md, store, TT, id);
    md.set("misc-globalSize", nvals);
    md.set("misc-precision", std::string("double"));
    md.set("levtype", std::string("SURF"));
    md.set("suffix", std::string("TEMPERATURE"));
    md.set("step", step);
    md.set("timestep", TIMESTEP);
    md.set("forecastReferenceTime", FRT);
    if (tweak) {
        tweak(md);
    }
    std::vector<double> vals(static_cast<std::size_t>(nvals), 1.0 + id);
    return Message{Message::Header{Message::Tag::Field, SERVER, SERVER, std::move(md)},
                   eckit::Buffer(vals.data(), vals.size() * sizeof(double))};
}

Message directTileFlush(const std::string& store, std::int64_t id, std::optional<std::int64_t> step,
                        const std::string& kind, const std::function<void(Metadata&)>& tweak = {}) {
    Metadata md;
    setTileKeys(md, store, TT, id);
    if (step) {
        md.set("step", *step);
    }
    md.set("flushKind", kind);
    if (tweak) {
        tweak(md);
    }
    return Message{Message::Header{Message::Tag::Flush, SERVER, SERVER, std::move(md)}};
}

std::string zarrOnlyPlan(const Path& out, const std::string& layout = "chunk-per-level") {
    return std::string(R"({"name": "z", "actions": [{"type": "zarr", "path": ")") + out.string()
         + R"(", "layout": ")" + layout + R"("}]})";
}

CASE("tiles: errors") {
    const Path out = absolute(outRoot()) / "tiles-errors";
    removeAll(out);
    auto env = MultioTestEnvironment(zarrOnlyPlan(out));
    const std::string st = "TERR";
    auto expectError = [&](Message&& m, const std::string& what) {
        const std::string e = errorText([&] { env.process(std::move(m)); });
        if (e.find(what) == std::string::npos) {
            std::cout << "expected an error with '" << what << "', got: " << (e.empty() ? "none" : e) << std::endl;
            EXPECT(false);
        }
    };
    using Md = Metadata;
    // tile 0: 8 x 8; tile 4: 5 x 8 (last column); tile 19: 5 x 5
    EXPECT_NO_THROW(env.process(directTile(st, 0, 0, 64)));
    EXPECT_NO_THROW(env.process(directTile(st, 4, 0, 40)));   // zarr-tile-size as a string
    EXPECT_NO_THROW(env.process(directTile(st, 19, 0, 25, [](Metadata& m) { m.set("zarr-tile-size", 8.0); })));
    expectError(directTile(st, 20, 0, 25), "store TERR tile 20: tile id out of range [0, 20)");
    expectError(directTile(st, -1, 0, 64), "tile -1: tile id out of range");
    expectError(directTile(st, 3, 0, 63), "field SURF*TEMPERATURE (level 0, step 0) of store TERR tile 3");
    expectError(directTile(st, 4, 0, 64), "expected tw * th = 5 * 8 of the tile");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("misc-globalSize", std::int64_t{63}); }),
                "misc-globalSize is 63");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("geometry-x0", std::int64_t{1}); }), "carries crop keys");
    expectError(directTile(st, 3, 0, 64,
                           [](Md& m) {
                               m.set("geometry-nx-full", TNX + 2);
                               m.set("geometry-ny-full", TNY + 2);
                               m.set("geometry-x0", std::int64_t{2});
                               m.set("geometry-y0", std::int64_t{2});
                           }),
                "a tiled field is cropped by the client");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("zarr-tile-size", std::int64_t{16}); }),
                "store TERR has tile size 8 (0: untiled), a message for it has zarr-tile-size 16");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("geometry-nx", TNX + 1); }), "but store TERR is 37 x 29");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("zarr-store", std::string("")); }), "no zarr-store");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("zarr-tile-size", std::string("eight")); }),
                "zarr-tile-size 'eight' is not an integer");
    expectError(directTile(st, 3, 0, 64, [](Md& m) { m.set("domain", std::string("TERR#2")); }),
                "domain 'TERR#2', expected 'TERR#3'");
    {
        Metadata md;
        md.set("zarr-store", st);
        md.set("zarr-tile-size", TT);
        md.set("domain", std::string("TERR#3"));
        Message m{Message::Header{Message::Tag::Field, SERVER, SERVER, std::move(md)}, eckit::Buffer(8)};
        expectError(std::move(m), "no zarr-tile (tile id)");
    }
    // an untiled message for the tiled store
    expectError(directTile(st, 3, 0, TNX * TNY,
                           [&](Md& m) {
                               m.set("zarr-tile-size", std::int64_t{0});
                               m.set("domain", st);
                           }),
                "store TERR has tile size 8 (0: untiled), a message for it has zarr-tile-size 0");
    expectError(directTileFlush(st, 25, 0, "step"), "flush for tile 25 of store TERR: tile id out of range");
    expectError(directTileFlush(st, 3, 0, "step", [](Md& m) { m.set("zarr-tile-size", std::int64_t{4}); }),
                "with zarr-tile-size 4, but the store has tile size 8");
    // the valid ones were written
    EXPECT_NO_THROW(env.process(directTileFlush(st, 0, 0, "step")));
    EXPECT_NO_THROW(env.process(directTileFlush(st, 4, 0, "step")));
    EXPECT_NO_THROW(env.process(directTileFlush(st, 19, 0, "end-of-simulation")));
    const Path s = out / "TERR.zarr";
    EXPECT_EQUAL(readText(s / ".last_step"), std::string("step=0 time_index=0 inc=0000:00:00\n"));
    EXPECT(exists(s / "SURFTEMPERATURE" / "c/0/0/0"));
    EXPECT(exists(s / "SURFTEMPERATURE" / "c/0/0/4"));
    EXPECT(exists(s / "SURFTEMPERATURE" / "c/0/3/4"));
    EXPECT(!exists(s / "SURFTEMPERATURE" / "c/0/0/3"));
    EXPECT_EQUAL(countStaging(s), 0u);
}

CASE("tiles: a step is closed once every tile held has flushed it") {
    const Path out = absolute(outRoot()) / "tiles-flush";
    removeAll(out);
    const std::string st = "TFLUSH";
    const Path s = out / (st + ".zarr");
    auto env = MultioTestEnvironment(zarrOnlyPlan(out));
    auto fields = [&](std::int64_t id, std::int64_t step) {
        const auto g = tileGeom(TT, id);
        env.process(directTile(st, id, step, g.tw * g.th));
        for (std::int64_t lev : {2, 1}) {
            env.process(directTile(st, id, step, g.tw * g.th, [&](Metadata& m) {
                m.set("levtype", std::string("S"));
                m.set("levelist", lev);
                m.set("is3d", true);
            }));
        }
    };
    auto flush = [&](std::int64_t id, std::int64_t step, const std::string& kind = "step") {
        env.process(directTileFlush(st, id, step, kind));
    };
    auto lastStep = [&]() { return exists(s / ".last_step") ? readText(s / ".last_step") : std::string("none"); };
    auto timeLength = [&]() {
        return exists(s / "time" / "zarr.json")
                 ? parseInts(multio::action::zarr::jsonMember(readText(s / "time" / "zarr.json"), "shape"))[0]
                 : std::int64_t{-1};
    };

    // flushes of tiles of a store without any field: no store
    flush(0, 0);
    flush(7, 0, "end-of-simulation");
    EXPECT(!exists(s));

    // tiles 0 and 6 held: step 0 is closed by the second flush only
    fields(0, 0);
    fields(6, 0);
    flush(0, 0);
    EXPECT_EQUAL(lastStep(), std::string("none"));
    EXPECT_EQUAL(timeLength(), -1);
    EXPECT(exists(s / "S_TEMPERATURE" / ".staging_s0_L2_T6"));
    flush(0, 0);  // repeated: nothing
    flush(5, 0);  // a tile without fields on this rank: recorded only
    EXPECT_EQUAL(lastStep(), std::string("none"));
    flush(6, 0);
    EXPECT_EQUAL(lastStep(), std::string("step=0 time_index=0 inc=0000:00:00\n"));
    EXPECT_EQUAL(timeLength(), 1);
    EXPECT(exists(s / "S_TEMPERATURE" / "c/0/1/1/1"));  // tile 6 = (jy, jx) = (1, 1), level 2 = index 1

    // step 48: tile 0 flushes first
    fields(0, 48);
    flush(0, 48);
    EXPECT_EQUAL(lastStep(), std::string("step=0 time_index=0 inc=0000:00:00\n"));
    EXPECT_THROWS(fields(0, 48));  // tile 0 flushed step 48 already
    fields(6, 48);
    flush(6, 48);
    EXPECT_EQUAL(lastStep(), std::string("step=48 time_index=1 inc=0001:00:00\n"));
    EXPECT_EQUAL(timeLength(), 2);

    // tile 13 shows up only now, with step 0: the step is reopened for its chunks; .last_step stays
    fields(13, 0);
    EXPECT(exists(s / "S_TEMPERATURE" / "c/0/0/2/3"));
    flush(13, 0);
    fields(13, 48);
    flush(13, 48);
    EXPECT_EQUAL(lastStep(), std::string("step=48 time_index=1 inc=0001:00:00\n"));
    EXPECT_EQUAL(timeLength(), 2);
    EXPECT(exists(s / "SURFTEMPERATURE" / "c/1/2/3"));
    EXPECT_EQUAL(readFloats(s / "SURFTEMPERATURE" / "c/0/2/3")[0], 14.0f);

    // an empty step: closed when every held tile flushed it; a field for it afterwards is an error
    flush(0, 72);
    flush(6, 72);
    flush(13, 72);
    EXPECT_THROWS(fields(0, 72));
    EXPECT_EQUAL(timeLength(), 2);

    // end-of-simulation: closed and finalised after the last held tile's
    fields(13, 96);
    fields(0, 96);
    fields(6, 96);
    flush(6, 96, "end-of-simulation");
    flush(0, 96, "end-of-simulation");
    EXPECT_EQUAL(lastStep(), std::string("step=48 time_index=1 inc=0001:00:00\n"));
    // close-connection (domain global) does not finalise a store with an open step
    {
        Metadata md;
        md.set("domain", std::string("global"));
        md.set("flushKind", std::string("close-connection"));
        env.process(Message{Message::Header{Message::Tag::Flush, SERVER, SERVER, std::move(md)}});
    }
    EXPECT_EQUAL(timeLength(), 2);
    flush(13, 96, "end-of-simulation");
    EXPECT_EQUAL(lastStep(), std::string("step=96 time_index=2 inc=0002:00:00\n"));
    EXPECT_EQUAL(timeLength(), 3);
    EXPECT(readText(s / "S_TEMPERATURE" / "zarr.json").find("\"shape\":[3,2,29,37]") != std::string::npos);
    EXPECT_NO_THROW(flush(0, 96, "end-of-simulation"));  // repeated
    EXPECT_EQUAL(countStaging(s), 0u);
    EXPECT(!exists(out / "TNONE.zarr"));
}

}  // namespace multio::test::zarr

int main(int argc, char** argv) {
    return eckit::testing::run_tests(argc, argv);
}
