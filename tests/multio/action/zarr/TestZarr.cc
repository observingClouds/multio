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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dirent.h>
#include <ftw.h>
#include <sys/stat.h>
#include <unistd.h>

#include "eckit/testing/Test.h"

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

std::string planJson(const std::string& path, const std::string& layout, const std::string& coords) {
    std::ostringstream oss;
    oss << R"({"name": "ial-to-zarr", "actions": [)"
        << R"({"type": "select", "match": [{"category": ["ial-field", "ial-static"]}]},)"
        << R"({"type": "aggregate"},)"
        << R"({"type": "zarr", "path": ")" << path << R"(", "format": 3, "layout": ")" << layout << R"(")";
    if (!coords.empty()) {
        oss << R"(, "coordinates-from": ")" << coords << R"(")";
    }
    oss << "}]}";
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
