/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

/// Action `zarr`: writes aggregated grid-point fields with FA-style metadata into zarr v3 stores,
/// one store per domain (<path>/<domain>.zarr).
///
/// Plan configuration:
///
///   - type: zarr
///     path: <output dir>              # required, {ENV} substitution as elsewhere in multio
///     format: 3                       # only zarr v3 is supported
///     layout: chunk-per-level         # (default) or shard-levels
///     coordinates-from: <dir>         # optional: copy static arrays and root attributes from <dir>/<domain>.zarr
///     timestep: <seconds>             # optional fallback if fields carry no `timestep`
///     compression: {codec: zstd, level: 3}   # optional, default none (or the string none / zstd)
///     compression-threads: 4          # compression workers (default 4); 0: compress on the receiving thread
///     compression-queue: 8            # fields waiting for a worker before addField blocks (default 2 x threads)
///
/// Compression (zstd, lossless) applies to the time-dependent arrays: codecs [bytes, zstd] per chunk, and for
/// shard-levels inside the sharding codec (each inner chunk compressed, the shard index uncompressed). The
/// workers compress and, for chunk files, write; the receiving thread only copies the field and queues it. A
/// step flush waits for the jobs of that step, then writes as without compression (see ZarrCodec.h).
///
/// It is a terminal action (a sink): it needs the metadata of the message, which plain DataSinks do not get.
///
/// Field payload: the global field (after aggregate) as nx*ny values (geometry-nx, geometry-ny), or, when the
/// field carries geometry-nx-full, geometry-ny-full, geometry-x0, geometry-y0, the full nx_full*ny_full grid
/// (C order, index (j-1)*nx_full+(i-1)) of which only the nx x ny window at the 1-based offset (x0, y0) is
/// written (IAL FullPos: the C+I zone of the C+I+E grid).
///
/// Static arrays: a Field message with `zarr-static` true carries one time-independent array (x, y, latitude,
/// longitude, crs, hyai, hybi of the IAL IO-server layout) described by `name`, `zarr-dimensions`, `zarr-shape`
/// (comma-separated, outermost first, empty for a scalar), `zarr-dtype` (float64 or int32), `zarr-attributes`
/// (JSON object) and optionally `zarr-group-attributes` (JSON object, replaces the root attributes). It is written
/// as <name>/zarr.json plus one chunk with the whole array and replaces an array of the same name.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "multio/action/Action.h"
#include "multio/action/zarr/ZarrStore.h"

namespace multio::action::zarr {

using message::Message;

class Zarr final : public Action {
public:
    explicit Zarr(const ComponentConfiguration& compConf);
    ~Zarr() override;

private:
    void executeImpl(Message msg) override;
    void print(std::ostream& os) const override;

    void field(const Message& msg);
    void staticArray(const Message& msg);
    void flush(const Message& msg);

    ZarrStore& store(const std::string& domain, const Message& msg);
    void finalizeAll() noexcept;

    std::string path_;
    int format_;
    Layout layout_;
    std::optional<std::string> coordinatesFrom_;
    std::optional<double> timestep_;
    Compression compression_;
    std::size_t compressionThreads_ = 4;
    std::size_t compressionQueue_ = 0;

    std::unique_ptr<WorkerPool> pool_;  // with compression and compression-threads > 0; outlives the stores
    std::map<std::string, std::unique_ptr<ZarrStore>> stores_;
    std::vector<float> buffer_;
};

}  // namespace multio::action::zarr
