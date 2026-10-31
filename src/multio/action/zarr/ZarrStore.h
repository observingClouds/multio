/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

/// One zarr v3 store (one output domain) written by the zarr action.
///
/// The layout and the metadata follow the IAL IO-server zarr prototype
/// (ifsaux/module/zarr_store_mod.F90, arpifs/io_serv/io_serv_zarr_*.F90):
///   3-D  <L>_<SUFFIX>  (time, level_<L>, y, x)
///   2-D  <PREF><SUFFIX> (time, y, x)
///   float32, fill value NaN, bytes codec little endian, consolidated metadata in the root zarr.json.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ZarrUtil.h"

namespace multio::action::zarr {

enum class Layout
{
    ChunkPerLevel,  // one file per (variable, time, level), written as the field arrives
    ShardLevels,    // 3-D: one sharding_indexed shard per (variable, time), written at the flush
};

/// Description of one incoming (aggregated) field
struct FieldDescription {
    std::string prefix;  // normalised FA prefix (level type), e.g. S, P, KB, SURF, CLS
    std::string suffix;  // FA suffix, e.g. TEMPERATURE
    std::int64_t level = 0;
    bool is3d = false;
    std::int64_t gribCode = 0;
    std::int64_t step = 0;
    std::optional<double> leadTime;  // seconds since the forecast reference time
};

/// A time-independent array sent by the client (x, y, latitude, longitude, crs, hyai, hybi): written as
/// <name>/zarr.json plus one chunk holding the whole array, as the IAL IO server writes its static coordinates
struct StaticArray {
    std::string name;
    std::vector<std::string> dims;    // outermost first; empty for a scalar
    std::vector<std::int64_t> shape;  // same order; empty for a scalar
    std::string dtype;                // float64 or int32
    std::string attributes;           // compact JSON object
    std::vector<unsigned char> data;  // little endian values, C order (last dimension fastest)
};

/// Everything a store needs to be created
struct StoreSetup {
    std::string name;                     // store name (domain), the store is <dir>/<name>.zarr
    std::string dir;                      // output directory
    Layout layout = Layout::ChunkPerLevel;
    std::optional<std::string> coordinatesFrom;  // directory holding <name>.zarr with static arrays
    std::optional<std::int64_t> nx;
    std::optional<std::int64_t> ny;
    std::optional<std::string> forecastReferenceTime;
};

class ZarrStore {
public:
    explicit ZarrStore(const StoreSetup& setup);
    ~ZarrStore() = default;

    ZarrStore(const ZarrStore&) = delete;
    ZarrStore& operator=(const ZarrStore&) = delete;

    /// Add one global field of nx*ny float32 values (C order, x fastest)
    void addField(const FieldDescription& fd, const float* values, std::size_t n);

    /// Close an output step. Without a step, all open steps are closed (in order of appearance).
    /// A step that has already been closed is not closed again; a step without any field creates no
    /// time step (it only counts as flushed).
    void flush(std::optional<std::int64_t> step, std::optional<double> leadTime);

    /// Write (or replace) a static array and, if given, replace the root group attributes (compact JSON object;
    /// forecast_reference_time is then set to the store's reference time). The consolidated metadata is rewritten
    /// right away; after finalisation also every array's zarr.json (with a warning).
    void addStatic(const StaticArray& a, const std::optional<std::string>& groupAttributes);

    /// Rewrite every array's zarr.json with the final time length and the consolidated metadata.
    /// Data of steps that have not been flushed is discarded.
    void finalize();

    bool finalized() const { return finalized_; }
    bool hasOpenSteps() const { return !open_.empty(); }

    const std::string& name() const { return name_; }
    const std::string& root() const { return root_; }
    std::int64_t nx() const { return nx_; }
    std::int64_t ny() const { return ny_; }
    std::size_t numTimes() const { return times_.size(); }
    std::size_t numVariables() const { return vars_.size(); }

private:
    struct Axis {
        char letter;
        std::string dim;  // level_<letter>
        std::vector<std::int64_t> levels;  // sorted ascending once frozen
        std::unordered_map<std::int64_t, std::int64_t> index;
        std::set<std::int64_t> pending;  // levels seen before the axis was frozen
        bool frozen = false;
    };

    struct Variable {
        std::string prefix;
        std::string suffix;
        std::string name;
        bool is3d = false;
        int axis = -1;
        std::int64_t gribCode = 0;
        bool published = false;  // array metadata written, part of the consolidated metadata
    };

    struct OpenStep {
        std::int64_t step;
        std::size_t timeIndex;
        std::optional<double> leadTime;
        // chunk-per-level, axis not yet frozen: (variable, level) -> staging file
        std::map<std::pair<std::size_t, std::int64_t>, std::string> staged;
        // shard-levels: variable -> level -> values
        std::unordered_map<std::size_t, std::map<std::int64_t, std::vector<float>>> shards;
    };

    enum class NodeKind
    {
        Variable,
        Level,
        Time
    };
    struct Node {
        NodeKind kind;
        std::size_t index;
    };

    void createStore(const StoreSetup& setup);
    void copyStaticArrays(const std::string& srcStore);
    void updateHasLatLon();

    std::size_t variableIndex(const FieldDescription& fd);
    int axisIndex(char letter);
    OpenStep& openStep(std::int64_t step);
    void closeStep(OpenStep& st, std::optional<double> leadTime);

    std::string arrayJson(const Variable& v) const;
    std::string levelJson(const Axis& a) const;
    std::string timeJson() const;
    std::string rootJson() const;

    std::string varPath(const Variable& v) const { return root_ + "/" + v.name; }
    std::string chunkPath(const Variable& v, std::size_t t, std::int64_t k) const;

    void writeChunk(const std::string& path, const void* data, std::size_t nbytes);
    void writeShard(const Variable& v, std::size_t t, const std::map<std::int64_t, std::vector<float>>& levels);
    void writeArrayMeta(const Variable& v);
    void writeTime(std::size_t t);
    void writeLevel(const Axis& a);
    void writeConsolidated();

    std::string name_;
    std::string root_;
    Layout layout_;
    std::int64_t nx_ = 0;
    std::int64_t ny_ = 0;
    std::string frt_;  // forecast reference time of the store
    std::string timeUnits_;
    std::string rootAttrs_;  // raw JSON object
    bool hasLatLon_ = false;

    // Static arrays, copied from the coordinates source or sent by the client: (name, raw zarr.json)
    std::vector<std::pair<std::string, std::string>> staticNodes_;
    std::vector<Node> nodes_;  // order of the generated nodes in the consolidated metadata
    bool timePublished_ = false;

    std::vector<Axis> axes_;
    std::unordered_map<char, int> axisByLetter_;

    std::vector<Variable> vars_;
    std::unordered_map<std::string, std::size_t> varByKey_;
    std::unordered_map<std::string, std::size_t> varByName_;

    std::vector<double> times_;                      // lead time per time index
    std::unordered_map<std::int64_t, std::size_t> stepTime_;  // step -> time index (open and closed)
    std::map<std::size_t, OpenStep> open_;           // time index -> open step
    std::set<std::int64_t> closed_;
    std::int64_t lastStep_ = 0;

    bool finalized_ = false;
    DirectoryCache dirs_;
};

}  // namespace multio::action::zarr
