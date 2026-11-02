/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "Zarr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <iterator>
#include <sstream>
#include <type_traits>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "multio/LibMultio.h"
#include "multio/config/MultioConfiguration.h"
#include "multio/util/PrecisionTag.h"

namespace multio::action::zarr {

namespace {

using message::Metadata;

//----------------------------------------------------------------------------------------------------------------------
// Lenient metadata access: the same key may arrive as int, bool, real or string depending on the client

std::optional<std::string> mdString(const Metadata& md, const std::string& key) {
    auto it = md.find(key);
    if (it == md.end()) {
        return std::nullopt;
    }
    return it->second.visit([](const auto& v) -> std::optional<std::string> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return v;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return std::string(v ? "true" : "false");
        }
        else if constexpr (std::is_same_v<T, std::int64_t>) {
            return std::to_string(v);
        }
        else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            return std::to_string(v);
        }
        else {
            return std::nullopt;
        }
    });
}

std::optional<double> mdDouble(const Metadata& md, const std::string& key) {
    auto it = md.find(key);
    if (it == md.end()) {
        return std::nullopt;
    }
    return it->second.visit([](const auto& v) -> std::optional<double> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            try {
                std::size_t pos = 0;
                double d = std::stod(v, &pos);
                return d;
            }
            catch (...) {
                return std::nullopt;
            }
        }
        else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>
                           || std::is_same_v<T, float>) {
            return static_cast<double>(v);
        }
        else {
            return std::nullopt;
        }
    });
}

std::optional<std::int64_t> mdInt(const Metadata& md, const std::string& key) {
    auto it = md.find(key);
    if (it == md.end()) {
        return std::nullopt;
    }
    return it->second.visit([](const auto& v) -> std::optional<std::int64_t> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            try {
                return static_cast<std::int64_t>(std::stoll(v));
            }
            catch (...) {
                return std::nullopt;
            }
        }
        else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::int64_t>) {
            return static_cast<std::int64_t>(v);
        }
        else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            return static_cast<std::int64_t>(std::llround(v));
        }
        else {
            return std::nullopt;
        }
    });
}

std::optional<bool> mdBool(const Metadata& md, const std::string& key) {
    auto it = md.find(key);
    if (it == md.end()) {
        return std::nullopt;
    }
    return it->second.visit([](const auto& v) -> std::optional<bool> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::string s;
            for (char c : v) {
                s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (s == "true" || s == "1" || s == "yes" || s == "t" || s == ".true.") {
                return true;
            }
            if (s == "false" || s == "0" || s == "no" || s == "f" || s == ".false.") {
                return false;
            }
            return std::nullopt;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return v;
        }
        else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double> || std::is_same_v<T, float>) {
            return v != 0;
        }
        else {
            return std::nullopt;
        }
    });
}

std::string trim(const std::string& s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

bool isKbKt(const std::string& p) {
    return p.size() == 2 && p[0] == 'K' && (p[1] == 'B' || p[1] == 'T');
}

/// IOZ_IS3D of the prototype: a one-letter level type prefix, or KB / KT
bool prefixIs3d(const std::string& p) {
    return p.size() == 1 || isKbKt(p);
}

/// IOZ_NORM_NAME of the prototype: move level digits out of the prefix (and off the head of the suffix)
void normaliseName(std::string& prefix, std::string& suffix, std::int64_t& level) {
    if (prefix.size() < 2) {
        return;
    }
    if (!std::isalpha(static_cast<unsigned char>(prefix[0]))) {
        return;
    }
    if (isKbKt(prefix)) {
        return;
    }
    for (std::size_t j = 1; j < prefix.size(); ++j) {
        if (!isDigit(prefix[j])) {
            return;
        }
    }
    std::string digits = prefix.substr(1);
    std::size_t idig = 0;
    while (idig < suffix.size() && isDigit(suffix[idig])) {
        ++idig;
    }
    if (idig >= suffix.size()) {
        return;
    }
    digits += suffix.substr(0, idig);
    std::int64_t lev = 0;
    try {
        lev = std::stoll(digits);
    }
    catch (...) {
        return;
    }
    prefix = prefix.substr(0, 1);
    suffix = suffix.substr(idig);
    if (level == 0) {
        level = lev;
    }
}

/// Split an FA article name into (prefix, suffix, level) the way the IO-server prototype receives it in
/// IOFLDDESC, before normaliseName:
///   KB / KT + level digits (FA writes them with 3 digits): KT273ISOT_ALTIT -> KT, ISOT_ALTIT, 273
///   everything else: 4 character prefix + rest, as IOGRIDA_MOD and INI2WRFP do for 2-D fields
///   (CLSTEMPERATURE -> CLST, EMPERATURE); a prefix with level digits (S090..., P85000..., H00500...,
///   C002...) is reduced to its level type letter by normaliseName afterwards.
/// A level found in the name is only used when `level` is 0 (ILEVG wins, as in IOZ_NORM_NAME).
void splitFaName(const std::string& name, std::string& prefix, std::string& suffix, std::int64_t& level) {
    if (name.size() > 2 && isKbKt(name.substr(0, 2)) && isDigit(name[2])) {
        std::size_t e = 2;
        while (e < name.size() && isDigit(name[e])) {
            ++e;
        }
        if (e < name.size()) {
            std::int64_t lev = 0;
            try {
                lev = std::stoll(name.substr(2, e - 2));
            }
            catch (...) {
                lev = 0;
            }
            prefix = name.substr(0, 2);
            suffix = trim(name.substr(e));
            if (level == 0) {
                level = lev;
            }
            return;
        }
    }
    prefix = trim(name.substr(0, std::min<std::size_t>(4, name.size())));
    suffix = name.size() > 4 ? trim(name.substr(4)) : std::string();
}

/// Crop window of a field whose payload is the full FullPos grid (C+I+E zones)
struct Crop {
    std::int64_t nxFull;
    std::int64_t nyFull;
    std::int64_t x0;  // 1-based
    std::int64_t y0;  // 1-based
};

Layout parseLayout(const std::string& s) {
    if (s == "chunk-per-level") {
        return Layout::ChunkPerLevel;
    }
    if (s == "shard-levels") {
        return Layout::ShardLevels;
    }
    throw eckit::UserError("zarr action: unknown layout '" + s + "' (chunk-per-level or shard-levels)", Here());
}

Codec parseCodec(const std::string& s) {
    if (s == "none") {
        return Codec::None;
    }
    if (s == "zstd") {
        if (!zstdAvailable()) {
            throw eckit::UserError("zarr action: compression codec zstd requested, but the zarr action was built "
                                   "without libzstd (see MULTIO_ZARR_ZSTD in its CMakeLists.txt)",
                                   Here());
        }
        return Codec::Zstd;
    }
    throw eckit::UserError("zarr action: unknown compression codec '" + s + "' (none or zstd)", Here());
}

/// `compression`: absent, a string (codec name) or {codec: <name>, level: <int>}
Compression parseCompression(const eckit::LocalConfiguration& cfg) {
    Compression c;
    if (!cfg.has("compression") || cfg.isNull("compression")) {
        return c;
    }
    if (cfg.isString("compression")) {
        c.codec = parseCodec(cfg.getString("compression"));
        return c;
    }
    if (!cfg.isSubConfiguration("compression")) {
        throw eckit::UserError("zarr action: 'compression' must be a codec name or {codec: zstd, level: <int>}",
                               Here());
    }
    const auto sub = cfg.getSubConfiguration("compression");
    for (const auto& k : sub.keys()) {
        if (k != "codec" && k != "level") {
            throw eckit::UserError("zarr action: unknown key 'compression." + k + "' (codec, level)", Here());
        }
    }
    if (!sub.has("codec") || !sub.isString("codec")) {
        throw eckit::UserError("zarr action: 'compression' needs 'codec' (none or zstd)", Here());
    }
    c.codec = parseCodec(sub.getString("codec"));
    if (sub.has("level")) {
        if (!sub.isIntegral("level")) {
            throw eckit::UserError("zarr action: 'compression.level' must be an integer", Here());
        }
        const long long level = sub.getLong("level");
        if (c.codec == Codec::Zstd) {
            const auto range = zstdLevelRange();
            if (level < range.first || level > range.second) {
                throw eckit::UserError("zarr action: zstd compression level " + std::to_string(level)
                                           + " out of range [" + std::to_string(range.first) + ", "
                                           + std::to_string(range.second) + "] (1 .. 22 usual, 3 default)",
                                       Here());
            }
        }
        c.level = static_cast<int>(level);
    }
    return c;
}

std::size_t parseCount(const eckit::LocalConfiguration& cfg, const std::string& key, std::size_t def,
                       long long maximum) {
    if (!cfg.has(key)) {
        return def;
    }
    if (!cfg.isIntegral(key)) {
        throw eckit::UserError("zarr action: '" + key + "' must be an integer", Here());
    }
    const long long v = cfg.getLong(key);
    if (v < 0 || v > maximum) {
        throw eckit::UserError("zarr action: '" + key + "' is " + std::to_string(v) + ", must be in [0, "
                                   + std::to_string(maximum) + "]",
                               Here());
    }
    return static_cast<std::size_t>(v);
}

}  // namespace

//----------------------------------------------------------------------------------------------------------------------

Zarr::Zarr(const ComponentConfiguration& compConf) :
    Action{compConf},
    path_{compConf.multioConfig().replaceCurly(compConf.parsedConfig().getString("path", ""))},
    format_{compConf.parsedConfig().getInt("format", 3)},
    layout_{parseLayout(compConf.parsedConfig().getString("layout", "chunk-per-level"))} {
    const auto& cfg = compConf.parsedConfig();
    if (path_.empty()) {
        throw eckit::UserError("zarr action: configuration key 'path' is required", Here());
    }
    if (format_ != 3) {
        throw eckit::UserError("zarr action: only zarr format 3 is supported (format: " + std::to_string(format_) + ")",
                               Here());
    }
    if (cfg.has("coordinates-from")) {
        std::string cf = compConf.multioConfig().replaceCurly(cfg.getString("coordinates-from"));
        if (!cf.empty()) {
            coordinatesFrom_ = cf;
        }
    }
    if (cfg.has("timestep")) {
        timestep_ = cfg.getDouble("timestep");
    }
    compression_ = parseCompression(cfg);
    compressionThreads_ = parseCount(cfg, "compression-threads", 4, 256);
    compressionQueue_ = parseCount(cfg, "compression-queue", 2 * compressionThreads_, 4096);
    if (compression_.enabled() && compressionThreads_ > 0) {
        pool_ = std::make_unique<WorkerPool>(compressionThreads_, std::max<std::size_t>(1, compressionQueue_));
    }
    makeDirs(path_);
}

Zarr::~Zarr() {
    finalizeAll();
    stores_.clear();  // waits for jobs of discarded steps
    pool_.reset();    // joins the workers
}

void Zarr::finalizeAll() noexcept {
    for (auto& s : stores_) {
        if (s.second->finalized()) {
            continue;
        }
        try {
            eckit::Log::info() << "zarr action: no end-of-simulation flush for " << s.first
                               << ", finalising the store now" << std::endl;
            s.second->finalize();
        }
        catch (const std::exception& e) {
            eckit::Log::error() << "zarr action: finalising store " << s.first << " failed: " << e.what()
                                << std::endl;
        }
        catch (...) {
            eckit::Log::error() << "zarr action: finalising store " << s.first << " failed" << std::endl;
        }
    }
}

void Zarr::executeImpl(Message msg) {
    util::ScopedTiming timing{statistics_.actionTiming_};
    switch (msg.tag()) {
        case Message::Tag::Field:
            field(msg);
            break;
        case Message::Tag::Flush:
            flush(msg);
            break;
        default:
            LOG_DEBUG_LIB(LibMultio) << "zarr action: ignoring message " << msg << std::endl;
            break;
    }
}

ZarrStore& Zarr::store(const std::string& domain, const Message& msg) {
    auto it = stores_.find(domain);
    if (it != stores_.end()) {
        return *it->second;
    }
    const auto& md = msg.metadata();
    StoreSetup setup;
    setup.name = domain;
    setup.dir = path_;
    setup.layout = layout_;
    setup.coordinatesFrom = coordinatesFrom_;
    setup.compression = compression_;
    setup.pool = pool_.get();
    setup.nx = mdInt(md, "geometry-nx");
    setup.ny = mdInt(md, "geometry-ny");
    if (auto frt = mdString(md, "forecastReferenceTime"); frt && !frt->empty()) {
        setup.forecastReferenceTime = *frt;
    }
    return *stores_.emplace(domain, std::make_unique<ZarrStore>(setup)).first->second;
}

void Zarr::field(const Message& msg) {
    const auto& md = msg.metadata();

    if (mdBool(md, "zarr-static").value_or(false)) {
        staticArray(msg);
        return;
    }

    auto domain = mdString(md, "domain");
    if (!domain || domain->empty()) {
        throw eckit::UserError("zarr action: field without 'domain' (the store name)", Here());
    }

    // Field identification. With `levtype` (IAL's CPREF, raw or already normalised), `suffix` (CSUFF) and
    // `levelist` (ILEVG) the descriptor is normalised exactly as IOZ_NORM_NAME does it on the IO server.
    // Without `levtype` the FA article name `name` is split as IOFLDDESC would carry it (splitFaName).
    const std::string name = trim(mdString(md, "name").value_or(""));
    const std::optional<std::int64_t> levelist = mdInt(md, "levelist") ? mdInt(md, "levelist") : mdInt(md, "level");
    FieldDescription fd;
    fd.level = levelist.value_or(0);
    if (auto lt = mdString(md, "levtype"); lt && !trim(*lt).empty()) {
        fd.prefix = trim(*lt);
        fd.suffix = trim(mdString(md, "suffix").value_or(""));
        normaliseName(fd.prefix, fd.suffix, fd.level);
        if (!levelist && fd.level == 0 && !name.empty() && prefixIs3d(fd.prefix)) {
            // a bare level type without `levelist`: take the level from the article name if it is the same field
            FieldDescription fn;
            splitFaName(name, fn.prefix, fn.suffix, fn.level);
            normaliseName(fn.prefix, fn.suffix, fn.level);
            if (fn.prefix == fd.prefix && fn.suffix == fd.suffix) {
                fd.level = fn.level;
            }
        }
    }
    else {
        if (name.empty()) {
            throw eckit::UserError("zarr action: field without 'levtype'/'suffix' and without 'name'", Here());
        }
        splitFaName(name, fd.prefix, fd.suffix, fd.level);
        normaliseName(fd.prefix, fd.suffix, fd.level);
    }
    fd.is3d = mdBool(md, "is3d").value_or(prefixIs3d(fd.prefix));
    fd.gribCode = mdInt(md, "grib-code").value_or(0);
    fd.step = mdInt(md, "step").value_or(0);
    if (auto ts = mdDouble(md, "timestep"); ts) {
        fd.leadTime = static_cast<double>(fd.step) * *ts;
    }
    else if (timestep_) {
        fd.leadTime = static_cast<double>(fd.step) * *timestep_;
    }

    // Field label for error messages
    auto label = [&]() {
        std::ostringstream oss;
        oss << "field " << (name.empty() ? fd.prefix + "*" + fd.suffix : name) << " (level " << fd.level << ", step "
            << fd.step << ") of domain " << *domain;
        return oss.str();
    };

    // Geometry: the store grid is the nx x ny window (geometry-nx/-ny). With geometry-nx-full, geometry-ny-full,
    // geometry-x0, geometry-y0 the payload is the full nx_full x ny_full grid (C order, x fastest, IAL's C+I+E
    // zones) and only the window starting at the 1-based point (x0, y0) is written.
    const auto gnx = mdInt(md, "geometry-nx");
    const auto gny = mdInt(md, "geometry-ny");
    const std::optional<std::int64_t> cropKeys[4] = {mdInt(md, "geometry-nx-full"), mdInt(md, "geometry-ny-full"),
                                                     mdInt(md, "geometry-x0"), mdInt(md, "geometry-y0")};
    const int ncrop = static_cast<int>(std::count_if(std::begin(cropKeys), std::end(cropKeys),
                                                     [](const auto& k) { return k.has_value(); }));
    std::optional<Crop> crop;
    if (ncrop != 0) {
        if (ncrop != 4) {
            throw eckit::UserError("zarr action: " + label()
                                       + " carries only some of geometry-nx-full, geometry-ny-full, geometry-x0, "
                                         "geometry-y0 (a crop window needs all four)",
                                   Here());
        }
        if (!gnx || !gny) {
            throw eckit::UserError("zarr action: " + label()
                                       + " has a crop window (geometry-nx-full ...) but no geometry-nx/geometry-ny",
                                   Here());
        }
        crop = Crop{*cropKeys[0], *cropKeys[1], *cropKeys[2], *cropKeys[3]};
        if (*gnx < 1 || *gny < 1 || crop->x0 < 1 || crop->y0 < 1 || crop->x0 - 1 + *gnx > crop->nxFull
            || crop->y0 - 1 + *gny > crop->nyFull) {
            std::ostringstream oss;
            oss << "zarr action: " << label() << ": crop window " << *gnx << " x " << *gny << " at (" << crop->x0
                << ", " << crop->y0 << ") (1-based) does not fit into the full grid " << crop->nxFull << " x "
                << crop->nyFull;
            throw eckit::UserError(oss.str(), Here());
        }
    }

    // Precision of the payload
    const auto globalSize = mdInt(md, "misc-globalSize");
    util::PrecisionTag prec = util::PrecisionTag::Double;
    if (auto p = mdString(md, "misc-precision"); p) {
        prec = util::decodePrecisionTag(*p);
    }
    else if (globalSize && *globalSize > 0
             && msg.payload().size() == static_cast<std::size_t>(*globalSize) * sizeof(float)) {
        prec = util::PrecisionTag::Float;
    }

    ZarrStore& st = store(*domain, msg);
    if ((gnx && *gnx != st.nx()) || (gny && *gny != st.ny())) {
        std::ostringstream oss;
        oss << "zarr action: " << label() << " has geometry " << gnx.value_or(st.nx()) << " x "
            << gny.value_or(st.ny()) << ", but store " << st.name() << " is " << st.nx() << " x " << st.ny();
        throw eckit::UserError(oss.str(), Here());
    }
    const std::size_t nx = static_cast<std::size_t>(st.nx());
    const std::size_t ny = static_cast<std::size_t>(st.ny());

    util::dispatchPrecisionTag(prec, [&](auto pt) {
        using Precision = typename decltype(pt)::type;
        if (msg.payload().size() % sizeof(Precision) != 0) {
            std::ostringstream oss;
            oss << "zarr action: " << label() << ": payload of " << msg.payload().size()
                << " bytes is not a multiple of " << sizeof(Precision) << " (misc-precision)";
            throw eckit::UserError(oss.str(), Here());
        }
        const std::size_t n = msg.payload().size() / sizeof(Precision);
        if (globalSize && static_cast<std::size_t>(*globalSize) != n) {
            std::ostringstream oss;
            oss << "zarr action: " << label() << ": payload has " << n << " values but misc-globalSize is "
                << *globalSize << " (is there an aggregate action before zarr?)";
            throw eckit::UserError(oss.str(), Here());
        }
        const std::size_t expected = crop ? static_cast<std::size_t>(crop->nxFull * crop->nyFull) : nx * ny;
        if (n != expected) {
            std::ostringstream oss;
            oss << "zarr action: " << label() << ": payload has " << n << " values, expected ";
            if (crop) {
                oss << "geometry-nx-full * geometry-ny-full = " << crop->nxFull << " * " << crop->nyFull;
            }
            else {
                oss << "nx * ny = " << nx << " * " << ny
                    << " (a payload larger than the output grid needs geometry-nx-full, geometry-ny-full, "
                       "geometry-x0, geometry-y0)";
            }
            throw eckit::UserError(oss.str(), Here());
        }
        const auto* in = static_cast<const Precision*>(msg.payload().data());
        // With compression workers a converted / cropped field goes to a fresh vector that the job takes over
        // (no second copy); otherwise the action's buffer is reused
        const bool async = st.encodesAsync();
        std::vector<float> fresh;
        std::vector<float>& out = async ? fresh : buffer_;
        if (crop) {
            out.resize(nx * ny);
            const std::size_t nxFull = static_cast<std::size_t>(crop->nxFull);
            const std::size_t i0 = static_cast<std::size_t>(crop->x0 - 1);
            const std::size_t j0 = static_cast<std::size_t>(crop->y0 - 1);
            for (std::size_t j = 0; j < ny; ++j) {
                const Precision* row = in + (j0 + j) * nxFull + i0;
                float* o = out.data() + j * nx;
                for (std::size_t i = 0; i < nx; ++i) {
                    o[i] = static_cast<float>(row[i]);
                }
            }
        }
        else if constexpr (std::is_same_v<Precision, float>) {
            st.addField(fd, in, n);  // copied by the store if a worker compresses it
            return;
        }
        else {
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(in[i]);
            }
        }
        if (async) {
            st.addField(fd, std::move(fresh));
        }
        else {
            st.addField(fd, out.data(), out.size());
        }
    });
}

void Zarr::staticArray(const Message& msg) {
    const auto& md = msg.metadata();

    auto domain = mdString(md, "domain");
    if (!domain || domain->empty()) {
        throw eckit::UserError("zarr action: static array without 'domain' (the store name)", Here());
    }
    StaticArray a;
    a.name = trim(mdString(md, "name").value_or(""));
    const std::string label = "static array '" + a.name + "' of domain " + *domain;
    if (a.name.empty()) {
        throw eckit::UserError("zarr action: " + label + ": no 'name'", Here());
    }

    // comma-separated lists, outermost first; an empty (or absent) string is a scalar
    auto split = [](const std::string& s) {
        std::vector<std::string> out;
        if (trim(s).empty()) {
            return out;
        }
        std::size_t b = 0;
        while (true) {
            std::size_t e = s.find(',', b);
            out.push_back(trim(s.substr(b, e == std::string::npos ? std::string::npos : e - b)));
            if (e == std::string::npos) {
                break;
            }
            b = e + 1;
        }
        return out;
    };
    a.dims = split(mdString(md, "zarr-dimensions").value_or(""));
    for (const auto& e : split(mdString(md, "zarr-shape").value_or(""))) {
        std::size_t pos = 0;
        long long v = -1;
        try {
            v = std::stoll(e, &pos);
        }
        catch (...) {
            pos = 0;
        }
        if (e.empty() || pos != e.size()) {
            throw eckit::UserError("zarr action: " + label + ": zarr-shape '" + mdString(md, "zarr-shape").value_or("")
                                       + "' is not a comma-separated list of integers",
                                   Here());
        }
        a.shape.push_back(static_cast<std::int64_t>(v));
    }
    a.dtype = trim(mdString(md, "zarr-dtype").value_or(""));
    if (a.dtype != "float64" && a.dtype != "int32") {
        throw eckit::UserError("zarr action: " + label + ": unknown zarr-dtype '" + a.dtype + "' (float64 or int32)",
                               Here());
    }
    a.attributes = jsonCompactObject(mdString(md, "zarr-attributes").value_or("{}"), label + ", zarr-attributes");
    std::optional<std::string> groupAttrs;
    if (auto g = mdString(md, "zarr-group-attributes"); g && !trim(*g).empty()) {
        groupAttrs = jsonCompactObject(*g, label + ", zarr-group-attributes");
    }

    // Payload: shape product == number of values == misc-globalSize
    std::int64_t count = 1;
    for (auto n : a.shape) {
        count *= n;
    }
    util::PrecisionTag prec = util::PrecisionTag::Double;
    if (auto p = mdString(md, "misc-precision"); p) {
        prec = util::decodePrecisionTag(*p);
    }
    std::vector<double> values;
    util::dispatchPrecisionTag(prec, [&](auto pt) {
        using Precision = typename decltype(pt)::type;
        if (msg.payload().size() % sizeof(Precision) != 0) {
            std::ostringstream oss;
            oss << "zarr action: " << label << ": payload of " << msg.payload().size() << " bytes is not a multiple of "
                << sizeof(Precision) << " (misc-precision)";
            throw eckit::UserError(oss.str(), Here());
        }
        const std::size_t n = msg.payload().size() / sizeof(Precision);
        const auto* in = static_cast<const Precision*>(msg.payload().data());
        values.assign(in, in + n);
    });
    const auto globalSize = mdInt(md, "misc-globalSize");
    if (static_cast<std::int64_t>(values.size()) != count || (globalSize && *globalSize != count)) {
        std::ostringstream oss;
        oss << "zarr action: " << label << ": shape [" << mdString(md, "zarr-shape").value_or("") << "] has " << count
            << " values, the payload has " << values.size();
        if (globalSize) {
            oss << ", misc-globalSize is " << *globalSize;
        }
        throw eckit::UserError(oss.str(), Here());
    }
    if (a.dtype == "float64") {
        a.data.resize(values.size() * sizeof(double));
        std::memcpy(a.data.data(), values.data(), a.data.size());
    }
    else {
        std::vector<std::int32_t> iv(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double v = values[i];
            if (!(v == std::trunc(v)) || v < static_cast<double>(std::numeric_limits<std::int32_t>::min())
                || v > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                std::ostringstream oss;
                oss << "zarr action: " << label << ": value " << v << " at index " << i << " is not an int32";
                throw eckit::UserError(oss.str(), Here());
            }
            iv[i] = static_cast<std::int32_t>(v);
        }
        a.data.resize(iv.size() * sizeof(std::int32_t));
        std::memcpy(a.data.data(), iv.data(), a.data.size());
    }

    // Store (created from geometry-nx/-ny like for a field); the message's grid must be the store's
    ZarrStore& st = store(*domain, msg);
    const auto gnx = mdInt(md, "geometry-nx");
    const auto gny = mdInt(md, "geometry-ny");
    if ((gnx && *gnx != st.nx()) || (gny && *gny != st.ny())) {
        std::ostringstream oss;
        oss << "zarr action: " << label << " has geometry " << gnx.value_or(st.nx()) << " x " << gny.value_or(st.ny())
            << ", but store " << st.name() << " is " << st.nx() << " x " << st.ny();
        throw eckit::UserError(oss.str(), Here());
    }
    st.addStatic(a, groupAttrs);
}

void Zarr::flush(const Message& msg) {
    const auto& md = msg.metadata();

    // flushKind as string (multio FlushKind names) or as integer (FlushKind enum value)
    bool endOfSimulation = false;
    bool closeOnly = false;
    if (auto it = md.find(std::string("flushKind")); it != md.end()) {
        if (auto s = mdString(md, "flushKind"); s && !s->empty() && !std::isdigit(static_cast<unsigned char>((*s)[0]))) {
            endOfSimulation = (*s == "end-of-simulation");
            closeOnly = (*s == "close-connection");
        }
        else if (auto i = mdInt(md, "flushKind"); i) {
            endOfSimulation = (*i == 4);
            closeOnly = (*i == 5);
        }
    }

    std::optional<std::int64_t> step = mdInt(md, "step");
    std::optional<double> lead;
    if (step) {
        if (auto ts = mdDouble(md, "timestep"); ts) {
            lead = static_cast<double>(*step) * *ts;
        }
    }

    // A flush for a step without fields in this store adds no time step (ZarrStore::flush); end-of-simulation
    // then only finalises. close-connection (sent by every client on multio_close_connections, after
    // end-of-simulation) finalises only what is not finalised yet. It is not aggregated (domain "global"):
    // the first client to close may do so while steps of the others are still open, so a store with open
    // steps is left alone; the aggregated end-of-simulation flush or the destructor finalises it.
    auto apply = [&](ZarrStore& s) {
        if (!closeOnly) {
            s.flush(step, lead);
        }
        if (endOfSimulation || (closeOnly && !s.finalized() && !s.hasOpenSteps())) {
            s.finalize();
        }
    };

    auto domain = mdString(md, "domain");
    if (domain && !domain->empty() && *domain != "global") {
        auto it = stores_.find(*domain);
        if (it == stores_.end()) {
            // no field of this domain so far: nothing to close, no store is created
            LOG_DEBUG_LIB(LibMultio) << "zarr action: flush for domain " << *domain << " without any field"
                                     << std::endl;
            return;
        }
        apply(*it->second);
        return;
    }
    for (auto& s : stores_) {
        apply(*s.second);
    }
}

void Zarr::print(std::ostream& os) const {
    os << "Zarr(path=" << path_ << ", format=" << format_
       << ", layout=" << (layout_ == Layout::ShardLevels ? "shard-levels" : "chunk-per-level");
    if (coordinatesFrom_) {
        os << ", coordinates-from=" << *coordinatesFrom_;
    }
    if (compression_.enabled()) {
        os << ", compression=" << compression_.describe() << ", compression-threads=" << compressionThreads_;
    }
    os << ", stores=" << stores_.size() << ")";
}

static ActionBuilder<Zarr> ZarrBuilder("zarr");

}  // namespace multio::action::zarr
