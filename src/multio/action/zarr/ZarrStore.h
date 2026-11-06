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
/// With compression the time-dependent arrays get a zstd codec after the bytes codec (shard-levels: inside the
/// sharding codec, the shard index stays uncompressed); static arrays and coordinates stay uncompressed.

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ZarrCodec.h"
#include "ZarrQuantize.h"
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
    Compression compression;     // chunk compression of the time-dependent arrays
    QuantRules quantize;         // lossy quantization before the compression, per array (needs compression)
    WorkerPool* pool = nullptr;  // compression workers (owned by the action); nullptr: compress inline
};

class ZarrStore {
public:
    explicit ZarrStore(const StoreSetup& setup);
    ~ZarrStore();  // waits for the outstanding compression jobs (their data is discarded unless flushed)

    ZarrStore(const ZarrStore&) = delete;
    ZarrStore& operator=(const ZarrStore&) = delete;

    /// Add one global field of nx*ny float32 values (C order, x fastest). With compression and a worker pool
    /// the values are copied, quantized (if a quantize rule applies to the array) and compressed (and,
    /// chunk-per-level and 2-D, written) by a worker; the call only waits if the pool's queue is full.
    void addField(const FieldDescription& fd, const float* values, std::size_t n);

    /// As above, taking over the values (no copy for the worker)
    void addField(const FieldDescription& fd, std::vector<float>&& values);

    /// True if addField hands the values to worker threads (then the vector overload avoids a copy)
    bool encodesAsync() const { return compression_.enabled() && pool_ != nullptr; }

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
        Quantization quant;      // from the store's quantize rules (by name), applied by the compression jobs
        bool published = false;  // array metadata written, part of the consolidated metadata
    };

    using LevelChunks = std::map<std::int64_t, std::shared_ptr<std::vector<unsigned char>>>;

    /// shard-levels: a variable's shard of one step is written as soon as it has every level of its (frozen)
    /// level axis and its inner chunks are encoded, by the job that finishes last (or inline without workers),
    /// not at the flush. A frozen axis takes no new level, so such a shard is complete. Shards of the first step
    /// (axes not frozen yet) and of variables with fewer levels than their axis are written at the flush.
    /// `pending` counts the outstanding encode jobs plus one token held until all levels have arrived; whoever
    /// brings it to 0 writes the shard. The writer only uses what is copied here (vars_ and axes_ may grow
    /// meanwhile).
    struct ShardState {
        std::atomic<int> pending{1};
        bool scheduled = false;          // all levels arrived, token dropped (main thread only)
        std::atomic<bool> written{false};
        LevelChunks snapshot;            // the level chunks, taken when the token is dropped
        std::string varDir;
        std::string varName;
        std::vector<std::int64_t> axisLevels;
    };

    struct OpenStep {
        std::int64_t step;
        std::size_t timeIndex;
        std::optional<double> leadTime;
        // chunk-per-level, axis not yet frozen: (variable, level) -> staging file
        std::map<std::pair<std::size_t, std::int64_t>, std::string> staged;
        // shard-levels: variable -> level -> encoded inner chunk (raw or compressed bytes; filled by a worker
        // until `jobs` is done)
        std::unordered_map<std::size_t, LevelChunks> shards;
        // shard-levels, early writes: variable -> state
        std::unordered_map<std::size_t, std::shared_ptr<ShardState>> shardStates;
        // compression jobs of this step (closing the step waits for them)
        std::shared_ptr<JobGroup> jobs;
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

    void addFieldImpl(const FieldDescription& fd, const float* values, std::size_t n,
                      const std::shared_ptr<std::vector<float>>& owned);
    void writeChunk(const std::string& path, const void* data, std::size_t nbytes) const;
    void encodeChunk(const std::string& path, bool ensureDir, const float* values, std::size_t nbytes) const;
    void writeShard(const Variable& v, std::size_t t, const LevelChunks& levels);
    void writeShardFile(const std::string& varDir, const std::string& varName,
                        const std::vector<std::int64_t>& axisLevels, std::size_t t, const LevelChunks& levels) const;
    void writeEarlyShard(ShardState& ss, std::size_t t) const;  // any thread
    void writeArrayMeta(const Variable& v);
    void ensureQuantizationContainer(const Quantization& q);
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
    Compression compression_;
    QuantRules quantize_;
    WorkerPool* pool_ = nullptr;
    mutable DirectoryCache dirs_;  // shared with the compression workers (thread-safe)
};

}  // namespace multio::action::zarr
