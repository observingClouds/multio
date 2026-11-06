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
///
/// Spatial tiles (StoreSetup::tileSize T > 0): the nx x ny grid is cut into tiles of (Ty, Tx) = (min(T, ny),
/// min(T, nx)) points, ntx = ceil(nx / Tx) by nty = ceil(ny / Ty), tile id = jy * ntx + jx. A tile is one chunk
/// (chunk-per-level: chunk shape (1, 1, Ty, Tx) / (1, Ty, Tx), keys c/<t>/<k>/<jy>/<jx> / c/<t>/<jy>/<jx>) or one
/// shard of all levels (shard-levels: shard (1, nlev, Ty, Tx), inner chunks (1, 1, Ty, Tx), key c/<t>/0/<jy>/<jx>).
/// A field is given per tile (tw * th values of that tile, x fastest); edge tiles are padded with the fill value
/// (NaN) to the full chunk shape. Staging, early shard writes and the encode jobs work per (variable, tile);
/// range-bits quantization takes the range of the tile's values. Several ZarrStore objects (one per MultIO server
/// rank) may write disjoint tiles of the same store: each writes the metadata from its own view (the consolidated
/// nodes sorted by name, so that every writer produces the same bytes). A step is closed when every tile the
/// object holds (has received a field or static array for) has flushed it; a tile that appears only after a step
/// was closed (it was not known yet) reopens that step for its own chunks.
/// T = 0: one tile covering the grid, behaviour and bytes as without tiles.

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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
    std::int64_t tileSize = 0;   // spatial tiles of tileSize x tileSize points; 0: untiled
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

    /// Tiled stores: add the tw * th values (C order, x fastest) of one tile of a field. The tile counts as held.
    void addTile(const FieldDescription& fd, std::int64_t tile, const float* values, std::size_t n);
    void addTile(const FieldDescription& fd, std::int64_t tile, std::vector<float>&& values);

    /// True if addField hands the values to worker threads (then the vector overload avoids a copy)
    bool encodesAsync() const { return compression_.enabled() && pool_ != nullptr; }

    /// Close an output step. Without a step, all open steps are closed (in order of appearance).
    /// A step that has already been closed is not closed again; a step without any field creates no
    /// time step (it only counts as flushed).
    void flush(std::optional<std::int64_t> step, std::optional<double> leadTime);

    /// Tiled stores: flush of one tile. The step is closed once every held tile has flushed it (without a step:
    /// every open step, as flush). A tile that is not held is only recorded (a later field of it for a step it
    /// flushed is an error). Returns the number of steps closed.
    std::size_t flushTile(std::optional<std::int64_t> step, std::optional<double> leadTime, std::int64_t tile);

    /// Tiled stores: end-of-simulation of one tile; true once every held tile has sent it (then finalise)
    bool endOfSimulationTile(std::int64_t tile);

    /// Tiled stores: a static array of this tile's domain arrived (the tile counts as held)
    void holdTile(std::int64_t tile);

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

    bool tiled() const { return tileSize_ > 0; }
    std::int64_t tileSize() const { return tileSize_; }
    std::int64_t numTiles() const { return ntx_ * nty_; }
    std::int64_t tileWidth(std::int64_t tile) const;   // tw of a valid tile id
    std::int64_t tileHeight(std::int64_t tile) const;  // th of a valid tile id
    const std::set<std::int64_t>& heldTiles() const { return held_; }

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
        std::int64_t tile = 0;
    };

    using VarTile = std::pair<std::size_t, std::int64_t>;  // (variable index, tile id)

    struct OpenStep {
        std::int64_t step;
        std::size_t timeIndex;
        std::optional<double> leadTime;
        bool reopened = false;  // tiled: the step had been closed before a tile not known then sent fields for it
        // chunk-per-level, axis not yet frozen: (variable, level, tile) -> staging file
        std::map<std::tuple<std::size_t, std::int64_t, std::int64_t>, std::string> staged;
        // shard-levels: (variable, tile) -> level -> encoded inner chunk (raw or compressed bytes; filled by a
        // worker until `jobs` is done)
        std::map<VarTile, LevelChunks> shards;
        // shard-levels, early writes: (variable, tile) -> state
        std::map<VarTile, std::shared_ptr<ShardState>> shardStates;
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
    OpenStep& openStep(std::int64_t step, std::int64_t tile);
    void closeStep(OpenStep& st, std::optional<double> leadTime);

    std::string arrayJson(const Variable& v) const;
    std::string levelJson(const Axis& a) const;
    std::string timeJson() const;
    std::string rootJson() const;

    std::string varPath(const Variable& v) const { return root_ + "/" + v.name; }
    std::string chunkPath(const Variable& v, std::size_t t, std::int64_t k, std::int64_t tile) const;
    std::string tileKey(std::int64_t tile) const;  // "<jy>/<jx>"
    void checkTile(std::int64_t tile, const FieldDescription& fd, std::size_t n) const;
    bool allHeldFlushed(std::int64_t step) const;

    void addFieldImpl(const FieldDescription& fd, std::int64_t tile, const float* values, std::size_t n,
                      std::shared_ptr<std::vector<float>> owned);
    void writeChunk(const std::string& path, const void* data, std::size_t nbytes) const;
    void encodeChunk(const std::string& path, bool ensureDir, const float* values, std::size_t nbytes) const;
    void writeShard(const Variable& v, std::size_t t, std::int64_t tile, const LevelChunks& levels);
    void writeShardFile(const std::string& varDir, const std::string& varName,
                        const std::vector<std::int64_t>& axisLevels, std::size_t t, std::int64_t tile,
                        const LevelChunks& levels) const;
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
    std::optional<std::size_t> lastTimeIndex_;  // largest time index closed so far (.last_step never goes back)

    // Spatial tiles (untiled: one tile of nx x ny)
    std::int64_t tileSize_ = 0;
    std::int64_t tileX_ = 0;  // Tx
    std::int64_t tileY_ = 0;  // Ty
    std::int64_t ntx_ = 1;
    std::int64_t nty_ = 1;
    std::set<std::int64_t> held_;                                    // tiles with a field or static array
    std::map<std::int64_t, std::set<std::int64_t>> flushedTiles_;   // step -> tiles that flushed it
    std::set<std::int64_t> eosTiles_;                                // tiles that sent end-of-simulation
    std::unordered_map<std::string, std::size_t> staticHash_;       // tiled: hash of the last static array per name

    bool finalized_ = false;
    Compression compression_;
    QuantRules quantize_;
    WorkerPool* pool_ = nullptr;
    mutable DirectoryCache dirs_;  // shared with the compression workers (thread-safe)
};

}  // namespace multio::action::zarr
