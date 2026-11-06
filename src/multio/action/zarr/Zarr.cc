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

/// Tile keys of a message of a tiled store (zarr-tile-size > 0)
struct TileInfo {
    std::string store;  // zarr-store
    std::int64_t tile;  // zarr-tile
    std::int64_t size;  // zarr-tile-size
};

/// The tile keys of a message, or nothing for an untiled message (zarr-tile-size absent or 0). `what` describes
/// the message in errors.
std::optional<TileInfo> tileInfo(const Metadata& md, const std::string& what) {
    if (md.find(std::string("zarr-tile-size")) == md.end()) {
        return std::nullopt;
    }
    const auto size = mdInt(md, "zarr-tile-size");
    if (!size || *size < 0) {
        throw eckit::UserError("zarr action: " + what + ": zarr-tile-size '"
                                   + mdString(md, "zarr-tile-size").value_or("?") + "' is not an integer >= 0",
                               Here());
    }
    if (*size == 0) {
        return std::nullopt;
    }
    TileInfo ti;
    ti.size = *size;
    ti.store = trim(mdString(md, "zarr-store").value_or(""));
    if (ti.store.empty()) {
        throw eckit::UserError("zarr action: " + what + ": zarr-tile-size " + std::to_string(*size)
                                   + " but no zarr-store (the store name of a tiled field)",
                               Here());
    }
    const auto tile = mdInt(md, "zarr-tile");
    if (!tile) {
        throw eckit::UserError("zarr action: " + what + " of tiled store " + ti.store + ": no zarr-tile (tile id)",
                               Here());
    }
    ti.tile = *tile;
    // the MultIO domain is <store>#<tile>; a different one would have been aggregated with another tile's map
    if (auto domain = mdString(md, "domain"); domain && !domain->empty() && *domain != "global") {
        const std::string want = ti.store + "#" + std::to_string(ti.tile);
        if (*domain != want) {
            throw eckit::UserError("zarr action: " + what + " of tiled store " + ti.store + ", tile "
                                       + std::to_string(ti.tile) + ": domain '" + *domain + "', expected '" + want
                                       + "'",
                                   Here());
        }
    }
    return ti;
}

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

/// One quantization of a rule or of `default`: exactly one of keep-bits, absolute, range-bits, lossless.
/// `what` names the entry in messages, e.g. "quantize.rules[2]".
Quantization parseQuantization(const eckit::LocalConfiguration& c, const std::string& what) {
    Quantization q;
    int n = 0;
    auto bitsIn = [&](const std::string& key, int lo, int hi) {
        if (!c.isIntegral(key)) {
            throw eckit::UserError("zarr action: '" + what + "." + key + "' must be an integer", Here());
        }
        const long long k = c.getLong(key);
        if (k < lo || k > hi) {
            throw eckit::UserError("zarr action: '" + what + "." + key + "' is " + std::to_string(k) + ", must be in ["
                                       + std::to_string(lo) + ", " + std::to_string(hi) + "]",
                                   Here());
        }
        return static_cast<int>(k);
    };
    if (c.has("keep-bits")) {
        ++n;
        q.mode = QuantMode::BitRound;
        q.bits = bitsIn("keep-bits", 1, 23);
    }
    if (c.has("range-bits")) {
        ++n;
        q.mode = QuantMode::RangeBits;
        q.bits = bitsIn("range-bits", 1, 23);
    }
    if (c.has("absolute")) {
        ++n;
        double e = 0;
        if (c.isIntegral("absolute")) {
            e = static_cast<double>(c.getLong("absolute"));
        }
        else if (c.isFloatingPoint("absolute")) {
            e = c.getDouble("absolute");
        }
        else {
            throw eckit::UserError("zarr action: '" + what + ".absolute' must be a number", Here());
        }
        if (!(e > 0) || !std::isfinite(e)) {
            throw eckit::UserError("zarr action: '" + what + ".absolute' is " + std::to_string(e) + ", must be > 0",
                                   Here());
        }
        // the step must be a normal float32 with a normal reciprocal (exact float32 rounding, see ZarrQuantize.cc)
        if (e < std::ldexp(1.0, -126) || e >= std::ldexp(1.0, 101)) {
            std::ostringstream oss;
            oss << "zarr action: '" << what << ".absolute' is " << e << ", must be in [2^-126, 2^101)";
            throw eckit::UserError(oss.str(), Here());
        }
        q.mode = QuantMode::Absolute;
        q.requested = e;
        q.step = powerOfTwoFloor(e);
    }
    if (c.has("lossless")) {
        ++n;
        if (!c.isBoolean("lossless") || !c.getBool("lossless")) {
            throw eckit::UserError("zarr action: '" + what + ".lossless' must be true", Here());
        }
        q.mode = QuantMode::None;
    }
    if (n != 1) {
        throw eckit::UserError(
            "zarr action: '" + what + "' needs exactly one of keep-bits, absolute, range-bits, lossless", Here());
    }
    return q;
}

/// `quantize`: {rules: [{domains: <glob or list of globs>, variables: <glob or list of globs>, regex: <regex>,
///                        <quantization>}, ...],
///              default: {<quantization>}}
/// `domains` (optional) restricts a rule to the stores whose domain name matches.
QuantRules parseQuantize(const eckit::LocalConfiguration& cfg) {
    QuantRules qr;
    if (!cfg.has("quantize") || cfg.isNull("quantize")) {
        return qr;
    }
    if (!cfg.isSubConfiguration("quantize")) {
        throw eckit::UserError("zarr action: 'quantize' must be a map {rules: [...], default: {...}}", Here());
    }
    const auto sub = cfg.getSubConfiguration("quantize");
    for (const auto& k : sub.keys()) {
        if (k != "rules" && k != "default") {
            throw eckit::UserError("zarr action: unknown key 'quantize." + k + "' (rules, default)", Here());
        }
    }
    if (sub.has("rules")) {
        if (!sub.isSubConfigurationList("rules")) {
            throw eckit::UserError("zarr action: 'quantize.rules' must be a list of maps", Here());
        }
        const auto rules = sub.getSubConfigurations("rules");
        for (std::size_t i = 0; i < rules.size(); ++i) {
            const auto& r = rules[i];
            const std::string what = "quantize.rules[" + std::to_string(i) + "]";
            for (const auto& k : r.keys()) {
                if (k != "domains" && k != "variables" && k != "regex" && k != "keep-bits" && k != "absolute"
                    && k != "range-bits" && k != "lossless") {
                    throw eckit::UserError(
                        "zarr action: unknown key '" + what + "." + k
                            + "' (domains, variables, regex, keep-bits, absolute, range-bits, lossless)",
                        Here());
                }
            }
            QuantRule rule;
            const auto globList = [&](const std::string& key) {
                std::vector<std::string> globs;
                if (r.isString(key)) {
                    globs.push_back(r.getString(key));
                }
                else if (r.isStringList(key)) {
                    globs = r.getStringVector(key);
                }
                else {
                    throw eckit::UserError("zarr action: '" + what + "." + key + "' must be a glob or a list of globs",
                                           Here());
                }
                for (const auto& g : globs) {
                    if (g.empty()) {
                        throw eckit::UserError("zarr action: '" + what + "." + key + "' has an empty pattern", Here());
                    }
                }
                return globs;
            };
            if (r.has("domains")) {
                rule.domains = globList("domains");
                if (rule.domains.empty()) {
                    throw eckit::UserError("zarr action: '" + what + ".domains' is empty", Here());
                }
            }
            if (r.has("variables")) {
                rule.globs = globList("variables");
            }
            if (r.has("regex")) {
                if (!r.isString("regex")) {
                    throw eckit::UserError("zarr action: '" + what + ".regex' must be a string", Here());
                }
                rule.regexText = r.getString("regex");
                try {
                    rule.regex = std::make_shared<const std::regex>(rule.regexText, std::regex::ECMAScript);
                }
                catch (const std::regex_error& e) {
                    throw eckit::UserError("zarr action: '" + what + ".regex' '" + rule.regexText
                                               + "' is not a valid regular expression: " + e.what(),
                                           Here());
                }
            }
            if (rule.globs.empty() && !rule.regex) {
                throw eckit::UserError("zarr action: '" + what + "' needs 'variables' or 'regex'", Here());
            }
            rule.quantization = parseQuantization(r, what);
            qr.rules.push_back(std::move(rule));
        }
    }
    if (sub.has("default")) {
        if (!sub.isSubConfiguration("default")) {
            throw eckit::UserError("zarr action: 'quantize.default' must be a map, e.g. {keep-bits: 12}", Here());
        }
        const auto d = sub.getSubConfiguration("default");
        for (const auto& k : d.keys()) {
            if (k != "keep-bits" && k != "absolute" && k != "range-bits" && k != "lossless") {
                throw eckit::UserError(
                    "zarr action: unknown key 'quantize.default." + k + "' (keep-bits, absolute, range-bits, lossless)",
                    Here());
            }
        }
        qr.fallback = parseQuantization(d, "quantize.default");
    }
    return qr;
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
    quantize_ = parseQuantize(cfg);
    if (!quantize_.empty() && compression_.codec != Codec::Zstd) {
        // quantized float32 without a compressor is as large as the raw data, and quantization runs in the
        // compression jobs
        throw eckit::UserError("zarr action: 'quantize' needs 'compression: {codec: zstd}'", Here());
    }
    for (std::size_t i = 0; i < quantize_.rules.size(); ++i) {
        const auto& r = quantize_.rules[i];
        std::string pat;
        for (const auto& g : r.globs) {
            pat += (pat.empty() ? "" : " ") + g;
        }
        if (r.regex) {
            pat += (pat.empty() ? "regex " : " regex ") + r.regexText;
        }
        if (!r.domains.empty()) {
            std::string dom;
            for (const auto& d : r.domains) {
                dom += (dom.empty() ? "" : " ") + d;
            }
            pat += " (domains " + dom + ")";
        }
        eckit::Log::info() << "zarr action: quantize rule " << i << ": " << pat << " -> " << r.quantization.describe()
                           << std::endl;
    }
    if (!quantize_.empty()) {
        eckit::Log::info() << "zarr action: quantize default -> " << quantize_.fallback.describe() << std::endl;
    }
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

ZarrStore& Zarr::store(const std::string& domain, const Message& msg, std::int64_t tileSize) {
    auto it = stores_.find(domain);
    if (it != stores_.end()) {
        if (it->second->tileSize() != tileSize) {
            std::ostringstream oss;
            oss << "zarr action: store " << domain << " has tile size " << it->second->tileSize()
                << " (0: untiled), a message for it has zarr-tile-size " << tileSize;
            throw eckit::UserError(oss.str(), Here());
        }
        return *it->second;
    }
    const auto& md = msg.metadata();
    StoreSetup setup;
    setup.name = domain;
    setup.tileSize = tileSize;
    setup.dir = path_;
    setup.layout = layout_;
    setup.coordinatesFrom = coordinatesFrom_;
    setup.compression = compression_;
    setup.quantize = quantize_;
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

    const auto ti = tileInfo(md, "field " + mdString(md, "name").value_or(""));
    auto domain = ti ? std::optional<std::string>(ti->store) : mdString(md, "domain");
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
            << fd.step << ")";
        if (ti) {
            oss << " of store " << ti->store << " tile " << ti->tile;
        }
        else {
            oss << " of domain " << *domain;
        }
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
    if (ti && ncrop != 0) {
        throw eckit::UserError("zarr action: " + label()
                                   + " carries crop keys (geometry-nx-full, geometry-ny-full, geometry-x0, "
                                     "geometry-y0): a tiled field is cropped by the client",
                               Here());
    }
    if (ti && (!gnx || !gny)) {
        throw eckit::UserError("zarr action: " + label() + " has no geometry-nx/geometry-ny (the store window)",
                               Here());
    }
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

    ZarrStore& st = store(*domain, msg, ti ? ti->size : 0);
    if ((gnx && *gnx != st.nx()) || (gny && *gny != st.ny())) {
        std::ostringstream oss;
        oss << "zarr action: " << label() << " has geometry " << gnx.value_or(st.nx()) << " x "
            << gny.value_or(st.ny()) << ", but store " << st.name() << " is " << st.nx() << " x " << st.ny();
        throw eckit::UserError(oss.str(), Here());
    }
    const std::size_t nx = static_cast<std::size_t>(st.nx());
    const std::size_t ny = static_cast<std::size_t>(st.ny());
    std::size_t tileValues = 0;  // tiled: tw * th of this tile
    if (ti) {
        if (ti->tile < 0 || ti->tile >= st.numTiles()) {
            std::ostringstream oss;
            oss << "zarr action: " << label() << ": tile id out of range [0, " << st.numTiles() << ") (tile size "
                << ti->size << " on " << nx << " x " << ny << ")";
            throw eckit::UserError(oss.str(), Here());
        }
        tileValues = static_cast<std::size_t>(st.tileWidth(ti->tile) * st.tileHeight(ti->tile));
    }

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
        const std::size_t expected
            = crop ? static_cast<std::size_t>(crop->nxFull * crop->nyFull) : ti ? tileValues : nx * ny;
        if (n != expected) {
            std::ostringstream oss;
            oss << "zarr action: " << label() << ": payload has " << n << " values, expected ";
            if (ti) {
                oss << "tw * th = " << st.tileWidth(ti->tile) << " * " << st.tileHeight(ti->tile)
                    << " of the tile";
            }
            else if (crop) {
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
            if (ti) {
                st.addTile(fd, ti->tile, in, n);
            }
            else {
                st.addField(fd, in, n);  // copied by the store if a worker compresses it
            }
            return;
        }
        else {
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(in[i]);
            }
        }
        if (ti) {
            if (async) {
                st.addTile(fd, ti->tile, std::move(fresh));
            }
            else {
                st.addTile(fd, ti->tile, out.data(), out.size());
            }
        }
        else if (async) {
            st.addField(fd, std::move(fresh));
        }
        else {
            st.addField(fd, out.data(), out.size());
        }
    });
}

void Zarr::staticArray(const Message& msg) {
    const auto& md = msg.metadata();

    const auto ti = tileInfo(md, "static array " + mdString(md, "name").value_or(""));
    auto domain = ti ? std::optional<std::string>(ti->store) : mdString(md, "domain");
    if (!domain || domain->empty()) {
        throw eckit::UserError("zarr action: static array without 'domain' (the store name)", Here());
    }
    StaticArray a;
    a.name = trim(mdString(md, "name").value_or(""));
    const std::string label = "static array '" + a.name + "' of "
                            + (ti ? "store " + ti->store + " tile " + std::to_string(ti->tile) : "domain " + *domain);
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
    ZarrStore& st = store(*domain, msg, ti ? ti->size : 0);
    const auto gnx = mdInt(md, "geometry-nx");
    const auto gny = mdInt(md, "geometry-ny");
    if ((gnx && *gnx != st.nx()) || (gny && *gny != st.ny())) {
        std::ostringstream oss;
        oss << "zarr action: " << label << " has geometry " << gnx.value_or(st.nx()) << " x " << gny.value_or(st.ny())
            << ", but store " << st.name() << " is " << st.nx() << " x " << st.ny();
        throw eckit::UserError(oss.str(), Here());
    }
    if (ti && (ti->tile < 0 || ti->tile >= st.numTiles())) {
        std::ostringstream oss;
        oss << "zarr action: " << label << ": tile id out of range [0, " << st.numTiles() << ")";
        throw eckit::UserError(oss.str(), Here());
    }
    st.addStatic(a, groupAttrs);
    if (ti) {
        st.holdTile(ti->tile);
    }
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

    // Tiled stores: a flush per tile domain <store>#<tile>. The step is closed once every tile this action holds
    // (has received fields or static arrays for) has flushed it, the store finalised once every held tile has sent
    // end-of-simulation (ZarrStore::flushTile, endOfSimulationTile). A flush for a store without any field or
    // static array creates no store.
    if (const auto ti = tileInfo(md, "flush"); ti) {
        auto it = stores_.find(ti->store);
        if (it == stores_.end()) {
            LOG_DEBUG_LIB(LibMultio) << "zarr action: flush for tile " << ti->tile << " of store " << ti->store
                                     << " without any field" << std::endl;
            return;
        }
        ZarrStore& s = *it->second;
        if (s.tileSize() != ti->size) {
            std::ostringstream oss;
            oss << "zarr action: flush for tile " << ti->tile << " of store " << ti->store << " with zarr-tile-size "
                << ti->size << ", but the store has tile size " << s.tileSize() << " (0: untiled)";
            throw eckit::UserError(oss.str(), Here());
        }
        if (ti->tile < 0 || ti->tile >= s.numTiles()) {
            std::ostringstream oss;
            oss << "zarr action: flush for tile " << ti->tile << " of store " << ti->store
                << ": tile id out of range [0, " << s.numTiles() << ")";
            throw eckit::UserError(oss.str(), Here());
        }
        if (!closeOnly) {
            s.flushTile(step, lead, ti->tile);
        }
        if (endOfSimulation) {
            if (s.endOfSimulationTile(ti->tile)) {
                s.finalize();
            }
            else {
                LOG_DEBUG_LIB(LibMultio) << "zarr action: end-of-simulation of tile " << ti->tile << " of store "
                                         << ti->store << ", waiting for the other tiles" << std::endl;
            }
        }
        else if (closeOnly && !s.finalized() && !s.hasOpenSteps()) {
            s.finalize();
        }
        return;
    }

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
