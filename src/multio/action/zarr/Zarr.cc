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

#include <cctype>
#include <cmath>
#include <iostream>
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

Layout parseLayout(const std::string& s) {
    if (s == "chunk-per-level") {
        return Layout::ChunkPerLevel;
    }
    if (s == "shard-levels") {
        return Layout::ShardLevels;
    }
    throw eckit::UserError("zarr action: unknown layout '" + s + "' (chunk-per-level or shard-levels)", Here());
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
    makeDirs(path_);
}

Zarr::~Zarr() {
    finalizeAll();
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
    setup.nx = mdInt(md, "geometry-nx");
    setup.ny = mdInt(md, "geometry-ny");
    if (auto frt = mdString(md, "forecastReferenceTime"); frt && !frt->empty()) {
        setup.forecastReferenceTime = *frt;
    }
    return *stores_.emplace(domain, std::make_unique<ZarrStore>(setup)).first->second;
}

void Zarr::field(const Message& msg) {
    const auto& md = msg.metadata();

    auto domain = mdString(md, "domain");
    if (!domain || domain->empty()) {
        throw eckit::UserError("zarr action: field without 'domain' (the store name)", Here());
    }

    FieldDescription fd;
    if (auto lt = mdString(md, "levtype")) {
        fd.prefix = trim(*lt);
        fd.suffix = trim(mdString(md, "suffix").value_or(""));
    }
    else {
        // FA article name: 4 character prefix + suffix (normalised below)
        std::string name = trim(mdString(md, "name").value_or(""));
        if (name.empty()) {
            throw eckit::UserError("zarr action: field without 'levtype'/'suffix' and without 'name'", Here());
        }
        fd.prefix = trim(name.substr(0, std::min<std::size_t>(4, name.size())));
        fd.suffix = name.size() > 4 ? trim(name.substr(4)) : std::string();
    }
    fd.level = mdInt(md, "levelist").value_or(mdInt(md, "level").value_or(0));
    normaliseName(fd.prefix, fd.suffix, fd.level);
    fd.is3d = mdBool(md, "is3d").value_or(prefixIs3d(fd.prefix));
    fd.gribCode = mdInt(md, "grib-code").value_or(0);
    fd.step = mdInt(md, "step").value_or(0);
    if (auto ts = mdDouble(md, "timestep"); ts) {
        fd.leadTime = static_cast<double>(fd.step) * *ts;
    }
    else if (timestep_) {
        fd.leadTime = static_cast<double>(fd.step) * *timestep_;
    }

    // Precision of the payload
    util::PrecisionTag prec = util::PrecisionTag::Double;
    if (auto p = mdString(md, "misc-precision"); p) {
        prec = util::decodePrecisionTag(*p);
    }
    else if (auto gs = mdInt(md, "misc-globalSize"); gs && *gs > 0
                                                       && msg.payload().size()
                                                              == static_cast<std::size_t>(*gs) * sizeof(float)) {
        prec = util::PrecisionTag::Float;
    }

    ZarrStore& st = store(*domain, msg);

    util::dispatchPrecisionTag(prec, [&](auto pt) {
        using Precision = typename decltype(pt)::type;
        const std::size_t n = msg.payload().size() / sizeof(Precision);
        const auto* in = static_cast<const Precision*>(msg.payload().data());
        if constexpr (std::is_same_v<Precision, float>) {
            st.addField(fd, in, n);
        }
        else {
            buffer_.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                buffer_[i] = static_cast<float>(in[i]);
            }
            st.addField(fd, buffer_.data(), n);
        }
    });
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

    auto apply = [&](ZarrStore& s) {
        if (!closeOnly) {
            s.flush(step, lead);
        }
        if (endOfSimulation || closeOnly) {
            s.finalize();
        }
    };

    auto domain = mdString(md, "domain");
    if (domain && !domain->empty() && *domain != "global") {
        auto it = stores_.find(*domain);
        if (it == stores_.end()) {
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
    os << ", stores=" << stores_.size() << ")";
}

static ActionBuilder<Zarr> ZarrBuilder("zarr");

}  // namespace multio::action::zarr
