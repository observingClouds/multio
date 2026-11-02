/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "ZarrStore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "multio/LibMultio.h"

namespace multio::action::zarr {

namespace {

static_assert(sizeof(float) == 4 && sizeof(double) == 8, "IEEE float32/float64 expected");

bool littleEndianHost() {
    const std::uint16_t one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

bool startsWith(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

/// Array name from prefix and suffix, as IOZ_VARNAME of the IO-server prototype
std::string variableName(const std::string& prefix, const std::string& suffix, bool is3d) {
    std::string name;
    if (is3d) {
        if (suffix.empty()) {
            name = prefix;
        }
        else if (suffix[0] == '_') {
            name = prefix + suffix;
        }
        else {
            name = prefix + "_" + suffix;
        }
    }
    else {
        name = prefix + suffix;
    }
    for (char& c : name) {
        bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.'
               || c == '-';
        if (!ok) {
            c = '_';
        }
    }
    if (name.empty()) {
        name = "UNNAMED";
    }
    return name;
}

/// CF-ish attributes of a level coordinate (LEVEL_ATTRS of io_serv_zarr_close.F90)
std::string levelAttributes(char letter) {
    std::string a = "{\"fa_prefix\":" + jsonString(std::string(1, letter));
    switch (letter) {
        case 'S':
            a += ",\"long_name\":\"model (hybrid) level, 1 = top\",\"units\":\"1\",\"positive\":\"down\"";
            break;
        case 'P':
            a += ",\"standard_name\":\"air_pressure\",\"long_name\":\"pressure level\",\"units\":\"Pa\",\"positive\":"
                 "\"down\"";
            break;
        case 'H':
            a += ",\"standard_name\":\"height\",\"long_name\":\"height above surface\",\"units\":\"m\",\"positive\":"
                 "\"up\"";
            break;
        case 'T':
            a += ",\"standard_name\":\"air_potential_temperature\",\"long_name\":\"potential temperature level\","
                 "\"units\":\"K\",\"positive\":\"up\"";
            break;
        case 'V':
            a += ",\"long_name\":\"potential vorticity level (FA units)\"";
            break;
        default:
            a += ",\"long_name\":" + jsonString(std::string("level of FA prefix ") + letter);
    }
    a += "}";
    return a;
}

const char* BYTES_CODEC = "{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}}";

/// zarr v3 array document, members in the order of ZARR_STORE_ARRAY_META
std::string arrayDocument(const std::vector<std::int64_t>& shape, const char* dtype,
                          const std::vector<std::int64_t>& chunk, const std::string& codecs, const std::string& attrs,
                          const std::vector<std::string>& dims, const char* fill = "\"NaN\"") {
    std::string j;
    j.reserve(512 + attrs.size());
    j += "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":";
    j += jsonIntArray(shape);
    j += ",\"data_type\":\"";
    j += dtype;
    j += "\",\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":";
    j += jsonIntArray(chunk);
    j += "}},\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}}";
    j += ",\"fill_value\":";
    j += fill;
    j += ",\"codecs\":";
    j += codecs;
    j += ",\"attributes\":";
    j += attrs;
    if (!dims.empty()) {
        j += ",\"dimension_names\":";
        j += jsonStringArray(dims);
    }
    j += "}";
    return j;
}

std::string formatInc(double lead) {
    long long s = static_cast<long long>(std::llround(lead));
    bool neg = s < 0;
    if (neg) {
        s = -s;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%04lld:%02lld:%02lld", neg ? "-" : "", s / 3600, (s % 3600) / 60, s % 60);
    return buf;
}

std::vector<std::int64_t> parseShape(const std::string& raw) {
    std::vector<std::int64_t> shape;
    if (raw.empty()) {
        return shape;
    }
    for (const auto& e : jsonArrayElements(raw)) {
        shape.push_back(std::stoll(e));
    }
    return shape;
}

}  // namespace

//----------------------------------------------------------------------------------------------------------------------

ZarrStore::ZarrStore(const StoreSetup& setup) :
    name_{setup.name}, layout_{setup.layout}, compression_{setup.compression}, pool_{setup.pool} {
    if (!littleEndianHost()) {
        throw eckit::NotImplemented("zarr action: big endian hosts are not supported", Here());
    }
    createStore(setup);
}

ZarrStore::~ZarrStore() {
    // the jobs refer to this store (directory cache, compression settings)
    for (auto& o : open_) {
        o.second.jobs->waitNoThrow();
    }
}

void ZarrStore::createStore(const StoreSetup& setup) {
    root_ = setup.dir + "/" + name_ + ".zarr";
    makeDirs(root_);
    dirs_.ensure(root_);

    if (pathExists(root_ + "/zarr.json")) {
        eckit::Log::warning() << "zarr action: store " << root_
                              << " exists already, it is overwritten array by array (stale chunks may remain)"
                              << std::endl;
    }

    std::string srcAttrs;
    if (setup.coordinatesFrom) {
        std::string src = *setup.coordinatesFrom + "/" + name_ + ".zarr";
        if (pathExists(src + "/zarr.json")) {
            std::string srcRoot = readFile(src + "/zarr.json");
            srcAttrs = jsonMember(srcRoot, "attributes");
            copyStaticArrays(src);
        }
        else {
            eckit::Log::warning() << "zarr action: coordinates-from: no store " << src
                                  << ", writing " << root_ << " without static coordinates" << std::endl;
        }
    }

    updateHasLatLon();

    // Grid size
    auto staticShape = [&](const std::string& nm) -> std::optional<std::int64_t> {
        for (const auto& n : staticNodes_) {
            if (n.first == nm) {
                auto s = parseShape(jsonMember(n.second, "shape"));
                if (s.size() == 1) {
                    return s[0];
                }
            }
        }
        return std::nullopt;
    };
    if (setup.nx && setup.ny) {
        nx_ = *setup.nx;
        ny_ = *setup.ny;
    }
    else if (auto sx = staticShape("x"), sy = staticShape("y"); sx && sy) {
        nx_ = *sx;
        ny_ = *sy;
    }
    else {
        throw eckit::UserError("zarr action: grid size of store " + name_
                                   + " unknown (metadata geometry-nx/geometry-ny missing and no x/y coordinates)",
                               Here());
    }
    if (auto sx = staticShape("x"); sx && *sx != nx_) {
        eckit::Log::warning() << "zarr action: " << name_ << ": geometry-nx=" << nx_ << " but coordinate x has "
                              << *sx << " points" << std::endl;
    }
    if (auto sy = staticShape("y"); sy && *sy != ny_) {
        eckit::Log::warning() << "zarr action: " << name_ << ": geometry-ny=" << ny_ << " but coordinate y has "
                              << *sy << " points" << std::endl;
    }

    // Reference time and root attributes
    std::string frt;
    std::string srcFrt = srcAttrs.empty() ? std::string() : jsonMember(srcAttrs, "forecast_reference_time");
    if (setup.forecastReferenceTime) {
        frt = *setup.forecastReferenceTime;
    }
    else if (!srcFrt.empty()) {
        frt = jsonUnquote(srcFrt);
    }
    else {
        frt = "1970-01-01T00:00:00";
        eckit::Log::warning() << "zarr action: " << name_
                              << ": no forecastReferenceTime, time units refer to 1970-01-01T00:00:00" << std::endl;
    }
    frt_ = frt;
    timeUnits_ = "seconds since " + frt;

    if (!srcAttrs.empty()) {
        rootAttrs_ = srcAttrs;
        if (srcFrt.empty()) {
            auto members = jsonObjectMembers(rootAttrs_);
            auto close = rootAttrs_.find_last_of('}');
            rootAttrs_ = rootAttrs_.substr(0, close) + (members.empty() ? "" : ",")
                       + "\"forecast_reference_time\":" + jsonString(frt) + "}";
        }
    }
    else {
        rootAttrs_ = "{\"Conventions\":\"CF-1.8\",\"title\":" + jsonString("MultIO zarr output, domain " + name_)
                   + ",\"source\":\"MultIO zarr action (prototype)\",\"nx_ci\":" + std::to_string(nx_)
                   + ",\"ny_ci\":" + std::to_string(ny_) + ",\"forecast_reference_time\":" + jsonString(frt) + "}";
    }

    writeConsolidated();

    eckit::Log::info() << "zarr action: created store " << root_ << " (" << nx_ << " x " << ny_ << ", layout "
                       << (layout_ == Layout::ShardLevels ? "shard-levels" : "chunk-per-level") << ", compression "
                       << compression_.describe() << ", " << staticNodes_.size() << " static arrays copied)" << std::endl;
}

void ZarrStore::copyStaticArrays(const std::string& src) {
    std::string rootRaw = readFile(src + "/zarr.json");

    std::vector<std::pair<std::string, std::string>> entries;
    std::string cm = jsonMember(rootRaw, "consolidated_metadata");
    if (!cm.empty() && cm != "null") {
        entries = jsonObjectMembers(jsonMember(cm, "metadata"));
    }
    else {
        for (const auto& name : listDirectory(src)) {
            const std::string node = src + "/" + name;
            if (isDirectory(node) && pathExists(node + "/zarr.json")) {
                entries.emplace_back(name, readFile(node + "/zarr.json"));
            }
        }
        std::sort(entries.begin(), entries.end());
    }

    for (auto& e : entries) {
        const std::string& nm = e.first;
        if (nm.find('/') != std::string::npos || nm == "time" || startsWith(nm, "level_")) {
            continue;
        }
        if (jsonUnquote(jsonMember(e.second, "node_type")) != "array") {
            continue;
        }
        std::string dims = jsonMember(e.second, "dimension_names");
        if (dims.find("\"time\"") != std::string::npos) {
            continue;
        }

        const std::string from = src + "/" + nm;
        if (!isDirectory(from)) {
            continue;
        }
        for (const auto& rel : listFilesRecursive(from)) {
            const auto slash = rel.find_last_of('/');
            const std::string fname = slash == std::string::npos ? rel : rel.substr(slash + 1);
            if (startsWith(fname, ".staging") || fname.find(".tmp") != std::string::npos) {
                continue;
            }
            std::string dst = root_ + "/" + nm + "/" + rel;
            dirs_.ensure(dirName(dst));
            std::string content = readFile(from + "/" + rel);
            writeFileAtomic(dst, content.data(), content.size());
        }
        staticNodes_.emplace_back(nm, std::move(e.second));
    }
}

void ZarrStore::updateHasLatLon() {
    // grid_mapping / coordinates attributes only when latitude and longitude exist (LLATLON of the prototype)
    hasLatLon_ = std::count_if(staticNodes_.begin(), staticNodes_.end(),
                               [](const auto& n) { return n.first == "latitude" || n.first == "longitude"; })
              == 2;
}

void ZarrStore::addStatic(const StaticArray& a, const std::optional<std::string>& groupAttributes) {
    auto fail = [&](const std::string& why) {
        throw eckit::UserError("zarr action: static array '" + a.name + "' of domain " + name_ + ": " + why, Here());
    };

    // Name
    if (a.name.empty()) {
        fail("empty array name");
    }
    if (a.name == "time" || startsWith(a.name, "level_")) {
        fail("the name is reserved for a generated coordinate (time, level_*)");
    }
    if (a.name[0] == '.' || a.name.find('/') != std::string::npos || a.name.find('\\') != std::string::npos) {
        fail("not a valid array name");
    }
    if (varByName_.find(a.name) != varByName_.end()) {
        fail("a time-dependent array of that name exists in the store");
    }

    // Dimensions, shape, data type
    if (a.dims.size() != a.shape.size()) {
        std::ostringstream oss;
        oss << a.dims.size() << " dimension names (zarr-dimensions) but rank " << a.shape.size() << " (zarr-shape)";
        fail(oss.str());
    }
    std::size_t elem = 0;
    const char* fill = nullptr;
    if (a.dtype == "float64") {
        elem = 8;
        fill = "\"NaN\"";
    }
    else if (a.dtype == "int32") {
        elem = 4;
        fill = "0";
    }
    else {
        fail("unknown data type '" + a.dtype + "' (float64 or int32)");
    }
    std::size_t count = 1;
    for (std::size_t i = 0; i < a.shape.size(); ++i) {
        const std::string& d = a.dims[i];
        if (d.empty()) {
            fail("empty dimension name");
        }
        if (d == "time") {
            fail("a static array cannot have the dimension time");
        }
        if (std::count(a.dims.begin(), a.dims.end(), d) != 1) {
            fail("dimension " + d + " appears twice");
        }
        if (a.shape[i] < 1) {
            fail("dimension " + d + " has size " + std::to_string(a.shape[i]) + " (must be >= 1)");
        }
        if (d == "x" && a.shape[i] != nx_) {
            fail("dimension x has size " + std::to_string(a.shape[i]) + ", but the store has nx = " + std::to_string(nx_));
        }
        if (d == "y" && a.shape[i] != ny_) {
            fail("dimension y has size " + std::to_string(a.shape[i]) + ", but the store has ny = " + std::to_string(ny_));
        }
        count *= static_cast<std::size_t>(a.shape[i]);
    }
    if (a.data.size() != count * elem) {
        std::ostringstream oss;
        oss << a.data.size() << " bytes of data, expected " << count << " values of " << a.dtype;
        fail(oss.str());
    }

    // Files: one chunk holding the whole array (key c, c/0, c/0/0, ...), then the array metadata
    const std::string dir = root_ + "/" + a.name;
    std::string chunkRel = "c";
    for (std::size_t i = 0; i < a.shape.size(); ++i) {
        chunkRel += "/0";
    }
    if (isDirectory(dir)) {
        // replaced array (coordinates-from copy, earlier message, earlier run): no stale chunks of another layout
        for (const auto& rel : listFilesRecursive(dir)) {
            if (rel != "zarr.json" && rel != chunkRel) {
                removeFile(dir + "/" + rel);
            }
        }
    }
    const std::string doc
        = arrayDocument(a.shape, a.dtype.c_str(), a.shape, std::string("[") + BYTES_CODEC + "]", a.attributes, a.dims,
                        fill);
    writeChunk(dir + "/" + chunkRel, a.data.data(), a.data.size());
    writeTextAtomic(dir + "/zarr.json", doc);

    auto it = std::find_if(staticNodes_.begin(), staticNodes_.end(), [&](const auto& n) { return n.first == a.name; });
    const bool replaced = it != staticNodes_.end();
    if (replaced) {
        it->second = doc;
    }
    else {
        staticNodes_.emplace_back(a.name, doc);
    }

    if (groupAttributes) {
        rootAttrs_ = jsonSetMember(*groupAttributes, "forecast_reference_time", jsonString(frt_));
    }

    // Variables published so far get grid_mapping / coordinates once latitude and longitude exist; after the
    // finalisation every array's metadata is rewritten
    const bool hadLatLon = hasLatLon_;
    updateHasLatLon();
    if (finalized_) {
        eckit::Log::warning() << "zarr action: " << name_ << ": static array " << a.name
                              << " arrived after the store was finalised; written, metadata rewritten" << std::endl;
    }
    if (finalized_ || (hasLatLon_ && !hadLatLon)) {
        for (const auto& v : vars_) {
            if (v.published) {
                writeArrayMeta(v);
            }
        }
    }
    if (finalized_ && timePublished_) {
        writeTextAtomic(root_ + "/time/zarr.json", timeJson());
    }
    writeConsolidated();

    eckit::Log::info() << "zarr action: " << name_ << ": static array " << a.name << " " << jsonIntArray(a.shape) << " "
                       << a.dtype << (replaced ? " (replaced)" : "")
                       << (groupAttributes ? ", root attributes replaced" : "") << std::endl;
}

//----------------------------------------------------------------------------------------------------------------------

int ZarrStore::axisIndex(char letter) {
    auto it = axisByLetter_.find(letter);
    if (it != axisByLetter_.end()) {
        return it->second;
    }
    Axis a;
    a.letter = letter;
    a.dim = std::string("level_") + letter;
    axes_.push_back(std::move(a));
    int idx = static_cast<int>(axes_.size()) - 1;
    axisByLetter_.emplace(letter, idx);
    return idx;
}

std::size_t ZarrStore::variableIndex(const FieldDescription& fd) {
    std::string key;
    key.reserve(fd.prefix.size() + fd.suffix.size() + 1);
    key += fd.prefix;
    key += '\x1f';
    key += fd.suffix;

    auto it = varByKey_.find(key);
    if (it != varByKey_.end()) {
        const Variable& v = vars_[it->second];
        if (v.is3d != fd.is3d) {
            throw eckit::UserError("zarr action: variable " + v.name + " of store " + name_
                                       + " received both as 2-D and as 3-D field",
                                   Here());
        }
        return it->second;
    }

    Variable v;
    v.prefix = fd.prefix;
    v.suffix = fd.suffix;
    v.is3d = fd.is3d;
    v.gribCode = fd.gribCode;
    v.name = variableName(fd.prefix, fd.suffix, fd.is3d);
    if (v.is3d) {
        if (fd.prefix.empty()) {
            throw eckit::UserError("zarr action: 3-D field without level type (levtype) in store " + name_, Here());
        }
        v.axis = axisIndex(fd.prefix[0]);
    }
    if (varByName_.find(v.name) != varByName_.end() || v.name == "time" || startsWith(v.name, "level_")
        || std::any_of(staticNodes_.begin(), staticNodes_.end(), [&](const auto& n) { return n.first == v.name; })) {
        throw eckit::UserError("zarr action: duplicate array name " + v.name + " in store " + name_ + " (prefix '"
                                   + fd.prefix + "', suffix '" + fd.suffix + "')",
                               Here());
    }
    std::size_t idx = vars_.size();
    vars_.push_back(std::move(v));
    varByKey_.emplace(std::move(key), idx);
    varByName_.emplace(vars_.back().name, idx);
    dirs_.ensure(varPath(vars_.back()));
    return idx;
}

ZarrStore::OpenStep& ZarrStore::openStep(std::int64_t step) {
    if (closed_.find(step) != closed_.end()) {
        std::ostringstream oss;
        oss << "zarr action: store " << name_ << " received a field for step " << step
            << " after that step had been flushed";
        throw eckit::UserError(oss.str(), Here());
    }
    auto it = stepTime_.find(step);
    if (it != stepTime_.end()) {
        return open_.at(it->second);
    }
    std::size_t t = stepTime_.size();
    stepTime_.emplace(step, t);
    OpenStep st;
    st.step = step;
    st.timeIndex = t;
    st.jobs = std::make_shared<JobGroup>();
    return open_.emplace(t, std::move(st)).first->second;
}

std::string ZarrStore::chunkPath(const Variable& v, std::size_t t, std::int64_t k) const {
    std::string p = varPath(v);
    p += "/c/";
    p += std::to_string(t);
    if (v.is3d) {
        p += "/";
        p += std::to_string(k);
    }
    p += "/0/0";
    return p;
}

void ZarrStore::writeChunk(const std::string& path, const void* data, std::size_t nbytes) {
    dirs_.ensure(dirName(path));
    writeFileAtomic(path, data, nbytes);
}

void ZarrStore::encodeChunk(const std::string& path, bool ensureDir, const float* values,
                            std::size_t nbytes) const {
    if (ensureDir) {
        dirs_.ensure(dirName(path));
    }
    if (!compression_.enabled()) {
        writeFileAtomic(path, values, nbytes);
        return;
    }
    thread_local std::vector<unsigned char> buf;  // one compression buffer per thread, kept at its capacity
    compressInto(compression_, values, nbytes, buf);
    writeFileAtomic(path, buf.data(), buf.size());
}

void ZarrStore::addField(const FieldDescription& fd, const float* values, std::size_t n) {
    if (encodesAsync()) {
        addFieldImpl(fd, values, n, std::make_shared<const std::vector<float>>(values, values + n));
        return;
    }
    addFieldImpl(fd, values, n, nullptr);
}

void ZarrStore::addField(const FieldDescription& fd, std::vector<float>&& values) {
    if (encodesAsync()) {
        auto owned = std::make_shared<const std::vector<float>>(std::move(values));
        addFieldImpl(fd, owned->data(), owned->size(), owned);
        return;
    }
    addFieldImpl(fd, values.data(), values.size(), nullptr);
}

/// `owned`: the values as a buffer the compression job keeps alive (set iff encodesAsync())
void ZarrStore::addFieldImpl(const FieldDescription& fd, const float* values, std::size_t n,
                             const std::shared_ptr<const std::vector<float>>& owned) {
    if (static_cast<std::int64_t>(n) != nx_ * ny_) {
        std::ostringstream oss;
        oss << "zarr action: field " << fd.prefix << "*" << fd.suffix << " level " << fd.level << " of store "
            << name_ << " has " << n << " values, expected nx*ny = " << nx_ << "*" << ny_;
        throw eckit::UserError(oss.str(), Here());
    }
    finalized_ = false;

    std::size_t vi = variableIndex(fd);
    OpenStep& st = openStep(fd.step);
    if (!st.leadTime && fd.leadTime) {
        st.leadTime = fd.leadTime;
    }
    const Variable& v = vars_[vi];
    const std::size_t nbytes = n * sizeof(float);

    // A chunk file (2-D, chunk-per-level): encoded and written here, or by a worker. The job only uses the path,
    // the owned values, the compression settings and the (thread-safe) directory cache. (A field sent twice for
    // the same step gives two jobs for one file; each write is atomic, but which one lands last is not defined.)
    auto chunkFile = [&](std::string path, bool ensureDir) {
        if (!owned) {
            encodeChunk(path, ensureDir, values, nbytes);
            return;
        }
        pool_->submit(st.jobs, [this, path = std::move(path), ensureDir, owned, nbytes] {
            encodeChunk(path, ensureDir, owned->data(), nbytes);
        });
    };

    if (!v.is3d) {
        chunkFile(chunkPath(v, st.timeIndex, 0), true);
        return;
    }

    Axis& a = axes_[v.axis];
    std::int64_t k = -1;
    if (a.frozen) {
        auto it = a.index.find(fd.level);
        if (it == a.index.end()) {
            std::ostringstream oss;
            oss << "zarr action: new level " << fd.level << " on frozen axis " << a.dim << " of store " << name_
                << " (variable " << v.name << ", step " << fd.step << ")";
            throw eckit::UserError(oss.str(), Here());
        }
        k = it->second;
    }
    else {
        a.pending.insert(fd.level);
    }

    if (layout_ == Layout::ShardLevels) {
        // inner chunk kept until the flush, compressed on arrival (a replaced slot stays alive for its job)
        auto slot = std::make_shared<std::vector<unsigned char>>();
        st.shards[vi][fd.level] = slot;
        if (!compression_.enabled()) {
            const auto* b = reinterpret_cast<const unsigned char*>(values);
            slot->assign(b, b + nbytes);
        }
        else if (!owned) {
            *slot = compressExact(compression_, values, nbytes);
        }
        else {
            pool_->submit(st.jobs, [this, slot, owned, nbytes] {
                *slot = compressExact(compression_, owned->data(), nbytes);
            });
        }
        return;
    }

    if (a.frozen) {
        chunkFile(chunkPath(v, st.timeIndex, k), true);
        return;
    }

    // Level index not known before the first flush: stage next to the final key, inside the
    // variable's own directory (no shared staging directory, no Lustre directory-lock contention)
    std::string staging
        = varPath(v) + "/.staging_s" + std::to_string(fd.step) + "_L" + std::to_string(fd.level);
    st.staged[{vi, fd.level}] = staging;
    chunkFile(std::move(staging), false);
}

//----------------------------------------------------------------------------------------------------------------------

void ZarrStore::flush(std::optional<std::int64_t> step, std::optional<double> leadTime) {
    if (step) {
        if (closed_.find(*step) != closed_.end()) {
            LOG_DEBUG_LIB(LibMultio) << "zarr action: " << name_ << ": step " << *step << " already flushed"
                                     << std::endl;
            return;
        }
        auto it = stepTime_.find(*step);
        if (it == stepTime_.end()) {
            // No field of this store arrived for this step (e.g. the FullPos domain had no output at this step,
            // or the flush comes from a domain-wide step notification): no time step is created. The step counts
            // as flushed, so a late field for it is reported as such.
            eckit::Log::info() << "zarr action: " << name_ << ": flush for step " << *step
                               << " without fields, no time step written" << std::endl;
            closed_.insert(*step);
            return;
        }
        closeStep(open_.at(it->second), leadTime);
        return;
    }
    bool single = open_.size() == 1;
    while (!open_.empty()) {
        closeStep(open_.begin()->second, single ? leadTime : std::nullopt);
    }
}

void ZarrStore::closeStep(OpenStep& st, std::optional<double> leadTime) {
    const std::size_t t = st.timeIndex;
    const std::int64_t step = st.step;

    // 0. Wait for the compression jobs of this step (only); an error of a job is rethrown here and the step
    // stays open
    st.jobs->wait();
    finalized_ = false;

    // 1. Freeze the level axes that were discovered with this step
    std::vector<std::size_t> newlyFrozen;
    for (std::size_t ai = 0; ai < axes_.size(); ++ai) {
        Axis& a = axes_[ai];
        if (a.frozen || a.pending.empty()) {
            continue;
        }
        a.levels.assign(a.pending.begin(), a.pending.end());
        a.pending.clear();
        for (std::size_t k = 0; k < a.levels.size(); ++k) {
            a.index.emplace(a.levels[k], static_cast<std::int64_t>(k));
        }
        a.frozen = true;
        newlyFrozen.push_back(ai);
    }

    // 2. Move staged chunks to their final keys / write the shards
    for (const auto& s : st.staged) {
        const Variable& v = vars_[s.first.first];
        std::int64_t k = axes_[v.axis].index.at(s.first.second);
        std::string dst = chunkPath(v, t, k);
        dirs_.ensure(dirName(dst));
        renamePath(s.second, dst);
    }
    for (const auto& sh : st.shards) {
        writeShard(vars_[sh.first], t, sh.second);
    }

    // 3. Time coordinate value
    double lead;
    if (leadTime) {
        lead = *leadTime;
    }
    else if (st.leadTime) {
        lead = *st.leadTime;
    }
    else {
        lead = static_cast<double>(step);
        eckit::Log::warning() << "zarr action: " << name_ << ": no lead time for step " << step
                              << " (no timestep), using the step number" << std::endl;
    }
    if (times_.size() <= t) {
        times_.resize(t + 1, std::numeric_limits<double>::quiet_NaN());
    }
    times_[t] = lead;

    // 4. Array metadata of the arrays that appeared for the first time
    for (std::size_t vi = 0; vi < vars_.size(); ++vi) {
        Variable& v = vars_[vi];
        if (v.published || (v.is3d && !axes_[v.axis].frozen)) {
            continue;
        }
        writeArrayMeta(v);
        v.published = true;
        nodes_.push_back({NodeKind::Variable, vi});
    }

    // 5. Level coordinates of the axes frozen now
    for (auto ai : newlyFrozen) {
        writeLevel(axes_[ai]);
        nodes_.push_back({NodeKind::Level, ai});
    }

    // 6. Time coordinate
    writeTime(t);
    if (!timePublished_) {
        nodes_.push_back({NodeKind::Time, 0});
        timePublished_ = true;
    }

    // 7. Consolidated metadata, then the step marker
    writeConsolidated();
    writeTextAtomic(root_ + "/.last_step", "step=" + std::to_string(step) + " time_index=" + std::to_string(t)
                                               + " inc=" + formatInc(lead) + "\n");

    eckit::Log::info() << "zarr action: " << name_ << ": step " << step << " -> time index " << t << ", "
                       << vars_.size() << " variables" << std::endl;

    closed_.insert(step);
    lastStep_ = step;
    open_.erase(t);  // invalidates st
}

void ZarrStore::finalize() {
    if (!open_.empty()) {
        std::size_t nstaged = 0;
        for (auto& o : open_) {
            if (o.second.jobs->waitNoThrow()) {
                eckit::Log::warning() << "zarr action: " << name_ << ": a compression job of the discarded step "
                                      << o.second.step << " failed" << std::endl;
            }
            for (const auto& s : o.second.staged) {
                removeFile(s.second);
            }
            nstaged += o.second.staged.size() + o.second.shards.size();
            stepTime_.erase(o.second.step);
        }
        eckit::Log::warning() << "zarr action: " << name_ << ": discarding " << open_.size()
                              << " step(s) that were never flushed (" << nstaged << " buffered fields)" << std::endl;
        open_.clear();
        for (auto& a : axes_) {
            a.pending.clear();
        }
    }

    for (const auto& v : vars_) {
        if (v.published) {
            writeArrayMeta(v);
        }
    }
    if (timePublished_) {
        dirs_.ensure(root_ + "/time");
        writeTextAtomic(root_ + "/time/zarr.json", timeJson());
    }
    writeConsolidated();
    finalized_ = true;

    eckit::Log::info() << "zarr action: " << name_ << ": finalised with " << times_.size() << " time steps, "
                       << vars_.size() << " variables" << std::endl;
}

//----------------------------------------------------------------------------------------------------------------------

std::string ZarrStore::arrayJson(const Variable& v) const {
    std::string attrs;
    attrs.reserve(256);
    attrs += "{\"long_name\":";
    attrs += jsonString(v.prefix + "*" + v.suffix);
    attrs += ",\"fa_prefix\":";
    attrs += jsonString(v.prefix);
    attrs += ",\"fa_suffix\":";
    attrs += jsonString(v.suffix);
    if (v.gribCode != 0) {
        attrs += ",\"grib_code\":";
        attrs += std::to_string(v.gribCode);
    }
    if (hasLatLon_) {
        attrs += ",\"grid_mapping\":\"crs\",\"coordinates\":\"latitude longitude\"";
    }
    attrs += "}";

    const std::int64_t nt = static_cast<std::int64_t>(times_.size());
    const std::string chunkCodecs = std::string("[") + BYTES_CODEC + compression_.codecsJsonTail() + "]";

    if (!v.is3d) {
        return arrayDocument({nt, ny_, nx_}, "float32", {1, ny_, nx_}, chunkCodecs, attrs, {"time", "y", "x"});
    }

    const Axis& a = axes_[v.axis];
    const std::int64_t nlev = static_cast<std::int64_t>(a.levels.size());
    if (layout_ == Layout::ShardLevels) {
        std::string codecs = "[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":"
                           + jsonIntArray({1, 1, ny_, nx_}) + ",\"codecs\":" + chunkCodecs + ",\"index_codecs\":["
                           + BYTES_CODEC + "],\"index_location\":\"end\"}}]";
        return arrayDocument({nt, nlev, ny_, nx_}, "float32", {1, nlev, ny_, nx_}, codecs, attrs,
                             {"time", a.dim, "y", "x"});
    }
    return arrayDocument({nt, nlev, ny_, nx_}, "float32", {1, 1, ny_, nx_}, chunkCodecs, attrs,
                         {"time", a.dim, "y", "x"});
}

std::string ZarrStore::levelJson(const Axis& a) const {
    const std::int64_t nlev = static_cast<std::int64_t>(a.levels.size());
    return arrayDocument({nlev}, "float64", {nlev}, std::string("[") + BYTES_CODEC + "]", levelAttributes(a.letter),
                         {a.dim});
}

std::string ZarrStore::timeJson() const {
    std::string attrs = "{\"standard_name\":\"time\",\"units\":" + jsonString(timeUnits_)
                      + ",\"calendar\":\"standard\",\"axis\":\"T\"}";
    return arrayDocument({static_cast<std::int64_t>(times_.size())}, "float64", {1},
                         std::string("[") + BYTES_CODEC + "]", attrs, {"time"});
}

std::string ZarrStore::rootJson() const {
    std::string j;
    j.reserve(4096 + 700 * (staticNodes_.size() + nodes_.size()));
    j += "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":";
    j += rootAttrs_;
    j += ",\"consolidated_metadata\":{\"kind\":\"inline\",\"must_understand\":false,\"metadata\":{";
    bool first = true;
    auto add = [&](const std::string& name, const std::string& doc) {
        if (!first) {
            j += ',';
        }
        first = false;
        j += jsonString(name);
        j += ':';
        j += doc;
    };
    for (const auto& n : staticNodes_) {
        add(n.first, n.second);
    }
    for (const auto& n : nodes_) {
        switch (n.kind) {
            case NodeKind::Variable:
                add(vars_[n.index].name, arrayJson(vars_[n.index]));
                break;
            case NodeKind::Level:
                add(axes_[n.index].dim, levelJson(axes_[n.index]));
                break;
            case NodeKind::Time:
                add("time", timeJson());
                break;
        }
    }
    j += "}}}";
    return j;
}

//----------------------------------------------------------------------------------------------------------------------

void ZarrStore::writeShard(const Variable& v, std::size_t t,
                           const std::map<std::int64_t, std::shared_ptr<std::vector<unsigned char>>>& levels) {
    const Axis& a = axes_[v.axis];
    const std::size_t nlev = a.levels.size();
    const std::size_t chunkBytes = static_cast<std::size_t>(nx_ * ny_) * sizeof(float);

    std::size_t total = 0;
    for (const auto& l : levels) {
        if (a.index.find(l.first) == a.index.end()) {
            std::ostringstream oss;
            oss << "zarr action: level " << l.first << " of " << v.name << " not on axis " << a.dim;
            throw eckit::SeriousBug(oss.str(), Here());
        }
        if (!compression_.enabled() && l.second->size() != chunkBytes) {
            throw eckit::SeriousBug("zarr action: inner chunk of " + v.name + " has the wrong size", Here());
        }
        total += l.second->size();
    }

    // [inner chunks][index: nlev x (offset uint64, nbytes uint64), little endian], index at the end.
    // With compression the inner chunks have their compressed sizes.
    std::vector<unsigned char> buf(total + nlev * 2 * sizeof(std::uint64_t));
    std::vector<std::uint64_t> index(2 * nlev, std::numeric_limits<std::uint64_t>::max());
    std::size_t off = 0;
    for (std::size_t k = 0; k < nlev; ++k) {
        auto it = levels.find(a.levels[k]);
        if (it == levels.end()) {
            continue;
        }
        const std::vector<unsigned char>& c = *it->second;
        if (!c.empty()) {
            std::memcpy(buf.data() + off, c.data(), c.size());
        }
        index[2 * k] = off;
        index[2 * k + 1] = c.size();
        off += c.size();
    }
    std::memcpy(buf.data() + off, index.data(), index.size() * sizeof(std::uint64_t));

    std::string path = varPath(v) + "/c/" + std::to_string(t) + "/0/0/0";
    writeChunk(path, buf.data(), buf.size());
}

void ZarrStore::writeArrayMeta(const Variable& v) {
    dirs_.ensure(varPath(v));
    writeTextAtomic(varPath(v) + "/zarr.json", arrayJson(v));
}

void ZarrStore::writeTime(std::size_t t) {
    const double value = times_[t];
    writeChunk(root_ + "/time/c/" + std::to_string(t), &value, sizeof(double));
    writeTextAtomic(root_ + "/time/zarr.json", timeJson());
}

void ZarrStore::writeLevel(const Axis& a) {
    std::vector<double> values(a.levels.begin(), a.levels.end());
    std::string dir = root_ + "/" + a.dim;
    writeChunk(dir + "/c/0", values.data(), values.size() * sizeof(double));
    writeTextAtomic(dir + "/zarr.json", levelJson(a));
}

void ZarrStore::writeConsolidated() {
    writeTextAtomic(root_ + "/zarr.json", rootJson());
}

}  // namespace multio::action::zarr
